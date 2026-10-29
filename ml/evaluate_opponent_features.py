#!/usr/bin/env python3
"""Do the opponent-performance features improve the model?

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/ml/evaluate_opponent_features.py

Protocol (fixed before looking at any result):
  * identical chronological split, preprocessing and hyperparameters for both variants
  * the baseline must reproduce the numbers in model_comparison exactly
  * Gradient-Boosted Trees and Logistic Regression, each with and without the features
  * the difference is reported with a match-level (cluster-robust) 95% interval, because
    balls from the same match are strongly correlated and a plain per-ball interval would
    be far too narrow
  * nothing here is tuned, and the production model is not touched

Reads  features/opponent/{batter,bowler,team} (from features/opponent_features.py).
Writes <processed>/opponent_feature_evaluation.
"""
import math
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

sys.path.insert(0, "/workspace/etl")
sys.path.insert(0, "/workspace/ml")
import schema_utils as S  # noqa: E402
from calibrate_win_probability import clip, metrics  # noqa: E402
from train_win_probability import (IMPUTED_FEATURES, LABEL, STATE_FEATURES,  # noqa: E402
                                   chronological_split, prepare)

OPP_DIR = f"{S.PROCESSED}/features/opponent"
OUT_PATH = f"{S.PROCESSED}/opponent_feature_evaluation"
# Spark's GBT is sensitive to the order rows reach it. On the plain training table both the
# production recipe and make_pipeline([]) reproduce model_comparison to 0.0 in Log-Loss; after
# the opponent joins reorder the rows the same GBT moves by about 1e-4 (Logistic Regression
# by about 2e-6). Both variants are compared on the SAME joined frame, and any improvement is
# read against this noise floor.
BASELINE_TOLERANCE = 5e-4
OPP_FEATURES = ["striker_vs_opp_ratio", "striker_vs_opp_balls",
                "bowler_vs_opp_ratio", "bowler_vs_opp_balls",
                "team_h2h_rate", "team_h2h_matches"]


def attach_opponent_features(df, spark):
    batter = spark.read.parquet(f"{OPP_DIR}/batter").select(
        F.col("match_id").alias("_m1"), F.col("player_id").alias("_p1"),
        F.col("vs_opp_ratio").alias("striker_vs_opp_ratio"),
        F.col("vs_opp_balls").alias("striker_vs_opp_balls"))
    bowler = spark.read.parquet(f"{OPP_DIR}/bowler").select(
        F.col("match_id").alias("_m2"), F.col("player_id").alias("_p2"),
        F.col("vs_opp_ratio").alias("bowler_vs_opp_ratio"),
        F.col("vs_opp_balls").alias("bowler_vs_opp_balls"))
    team = spark.read.parquet(f"{OPP_DIR}/team").select(
        F.col("match_id").alias("_m3"), F.col("team").alias("_t3"),
        F.col("h2h_rate").alias("team_h2h_rate"), F.col("h2h_n").alias("team_h2h_matches"))
    out = (df.join(batter, (F.col("match_id") == F.col("_m1")) & (F.col("batter_id") == F.col("_p1")), "left")
             .join(bowler, (F.col("match_id") == F.col("_m2")) & (F.col("bowler_id") == F.col("_p2")), "left")
             .join(team, (F.col("match_id") == F.col("_m3")) & (F.col("batting_team") == F.col("_t3")), "left")
             .drop("_m1", "_p1", "_m2", "_p2", "_m3", "_t3"))
    # No earlier record means zero balls / zero meetings; the ratios stay null for the
    # imputer (a player with too little history has no trustworthy overall rate).
    for c in ("striker_vs_opp_balls", "bowler_vs_opp_balls", "team_h2h_matches"):
        out = out.withColumn(c, F.coalesce(F.col(c), F.lit(0)).cast("double"))
    out = out.withColumn("team_h2h_rate", F.coalesce(F.col("team_h2h_rate"), F.lit(0.5)).cast("double"))
    for c in ("striker_vs_opp_ratio", "bowler_vs_opp_ratio"):
        out = out.withColumn(c, F.col(c).cast("double"))
    return out


def make_pipeline(extra, classifier):
    """Same stages as train_win_probability.build_pipeline, with optional extra inputs."""
    imputed = list(IMPUTED_FEATURES) + list(extra)
    return Pipeline(stages=[
        Imputer(inputCols=imputed, outputCols=[f"{c}_imp" for c in imputed], strategy="median"),
        StringIndexer(inputCol="match_type", outputCol="match_type_idx", handleInvalid="keep"),
        OneHotEncoder(inputCols=["match_type_idx"], outputCols=["match_type_vec"]),
        VectorAssembler(inputCols=list(STATE_FEATURES) + [f"{c}_imp" for c in imputed] + ["match_type_vec"],
                        outputCol="features", handleInvalid="error"),
        classifier])


def logloss_col(p_col):
    p, y = clip(F.col(p_col)), F.col(LABEL)
    return -(y * F.log(p) + (1 - y) * F.log(1 - p))


def main():
    spark = S.build_spark("evaluate_opponent_features")
    spark.sparkContext.setLogLevel("WARN")

    df = prepare(spark.read.parquet(S.TRAINING_TABLE))
    n0 = df.count()
    df = attach_opponent_features(df, spark).cache()
    n1 = df.count()
    print(f"training rows before/after joining the opponent features: {n0} / {n1} "
          f"[{'OK' if n0 == n1 else 'ROWS MULTIPLIED - STOP'}]", flush=True)
    assert n0 == n1

    train, test, cutoff = chronological_split(df)
    test = test.withColumn("_k", F.monotonically_increasing_id()).cache()
    n_test = test.count()
    print(f"test rows: {n_test} (seasons >= {cutoff})", flush=True)

    print("\n== coverage on the held-out rows ==", flush=True)
    for label, cond in [("striker has earlier balls vs this opponent", F.col("striker_vs_opp_balls") > 0),
                        ("bowler has earlier balls vs this opponent", F.col("bowler_vs_opp_balls") > 0),
                        ("team has earlier meetings with this opponent", F.col("team_h2h_matches") > 0),
                        ("striker ratio usable (not null)", F.col("striker_vs_opp_ratio").isNotNull()),
                        ("bowler ratio usable (not null)", F.col("bowler_vs_opp_ratio").isNotNull())]:
        print(f"  {label:<46}{test.filter(cond).count() / n_test:>8.1%}", flush=True)

    specs = [("Gradient-Boosted Trees",
              lambda: GBTClassifier(labelCol=LABEL, featuresCol="features", maxIter=60, maxDepth=5, seed=42)),
             ("Logistic Regression",
              lambda: LogisticRegression(labelCol=LABEL, featuresCol="features", maxIter=100,
                                         regParam=0.0, standardization=True))]
    variants = [("baseline", []), ("with opponent features", OPP_FEATURES)]

    scored, results = {}, {}
    for name, make in specs:
        for variant, extra in variants:
            t0 = time.time()
            model = make_pipeline(extra, make()).fit(train)
            sc = (model.transform(test).withColumn("p", vector_to_array(F.col("probability"))[1])
                  .select("_k", "match_id", "season", LABEL, "p", "balls_remaining",
                          "balls_completed_in_innings").cache())
            scored[(name, variant)] = sc
            results[(name, variant)] = metrics(sc, "p")
            print(f"[{name} | {variant}] trained and scored in {time.time() - t0:.0f}s", flush=True)

    # ---- the baseline must reproduce the model comparison exactly ----
    ref = {r["model"]: r for r in spark.read.parquet(f"{S.PROCESSED}/model_comparison").collect()}
    print("\n== baseline reproduces model_comparison? ==", flush=True)
    for name, _ in specs:
        m, r = results[(name, "baseline")], ref[name]
        d = max(abs(m["logloss"] - r["logloss"]), abs(m["auc"] - r["auc"]))
        print(f"  {name:<24} max abs difference {d:.1e}  "
              f"[{'OK' if d < BASELINE_TOLERANCE else 'DIFFERS'}] (tolerance {BASELINE_TOLERANCE:.0e}, "
              f"row-order noise floor)", flush=True)

    print("\n== held-out results ==", flush=True)
    print(f"{'model':<24}{'variant':<26}{'AUC':>8}{'Brier':>8}{'LogLoss':>9}{'ECE':>8}", flush=True)
    for (name, variant), m in results.items():
        print(f"{name:<24}{variant:<26}{m['auc']:>8.4f}{m['brier']:>8.4f}{m['logloss']:>9.4f}{m['ece']:>8.4f}",
              flush=True)

    # ---- paired difference with a match-level (cluster-robust) interval ----
    summary = []
    for name, _ in specs:
        b, e = scored[(name, "baseline")], scored[(name, "with opponent features")]
        j = (b.select("_k", "match_id", "season", LABEL, F.col("p").alias("p0"),
                      "balls_remaining", "balls_completed_in_innings")
              .join(e.select("_k", F.col("p").alias("p1")), "_k")
              .withColumn("d", logloss_col("p1") - logloss_col("p0")).cache())
        per_match = j.groupBy("match_id").agg(F.sum("d").alias("s"), F.count("*").alias("n")).collect()
        n_all = float(sum(r["n"] for r in per_match))
        delta = sum(r["s"] for r in per_match) / n_all
        var = sum((r["s"] - delta * r["n"]) ** 2 for r in per_match) / (n_all ** 2)
        se = math.sqrt(var)
        lo, hi = delta - 1.96 * se, delta + 1.96 * se
        verdict = ("IMPROVES" if hi < 0 else "WORSENS" if lo > 0 else "no distinguishable change")
        print(f"\n[{name}] change in Log-Loss with opponent features: {delta:+.5f}  "
              f"95% CI [{lo:+.5f}, {hi:+.5f}]  over {len(per_match)} matches -> {verdict}", flush=True)
        print(f"  (negative = better; baseline Log-Loss {results[(name, 'baseline')]['logloss']:.4f})", flush=True)

        print("  by held-out season (negative = the features helped):", flush=True)
        seasons = j.groupBy("season").agg(F.avg("d").alias("d")).orderBy("season").collect()
        better = sum(1 for r in seasons if r["d"] < 0)
        for r in seasons:
            print(f"    {r['season']:<9}{r['d']:>+10.5f}", flush=True)
        print(f"  helped in {better} of {len(seasons)} seasons", flush=True)

        print("  by how far through the chase (progress quarters):", flush=True)
        prog = F.col("balls_completed_in_innings") / (F.col("balls_completed_in_innings") + F.col("balls_remaining"))
        quarters = (j.withColumn("q", F.least(F.floor(prog * 4), F.lit(3)).cast("int"))
                    .groupBy("q").agg(F.avg("d").alias("d"), F.count("*").alias("n")).orderBy("q").collect())
        for r in quarters:
            print(f"    quarter {r['q'] + 1} ({r['q'] * 25:>2}-{r['q'] * 25 + 25:>3}% of the chase): "
                  f"{r['d']:>+10.5f}   ({r['n']} rows)", flush=True)
        summary.append((name, delta, lo, hi, verdict, results[(name, "baseline")], results[(name, "with opponent features")]))

    # ---- persist ----
    rows = []
    for name, delta, lo, hi, verdict, mb, me in summary:
        rows.append((name, "baseline", mb["auc"], mb["brier"], mb["logloss"], mb["ece"], None, None, None, None))
        rows.append((name, "with opponent features", me["auc"], me["brier"], me["logloss"], me["ece"],
                     delta, lo, hi, verdict))
    schema = StructType([StructField("model", StringType(), False), StructField("variant", StringType(), False),
                         StructField("auc", DoubleType(), False), StructField("brier", DoubleType(), False),
                         StructField("logloss", DoubleType(), False), StructField("ece", DoubleType(), False),
                         StructField("delta_logloss", DoubleType(), True), StructField("ci_low", DoubleType(), True),
                         StructField("ci_high", DoubleType(), True), StructField("verdict", StringType(), True)])
    spark.createDataFrame(rows, schema).write.mode("overwrite").parquet(OUT_PATH)
    n_back = spark.read.parquet(OUT_PATH).count()
    print(f"\nwrote {OUT_PATH}: {n_back} rows [{'OK' if n_back == len(rows) else 'MISMATCH'}]", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
