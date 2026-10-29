#!/usr/bin/env python3
"""Do the venue par features improve the model, and is it the VENUE doing it?

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/ml/evaluate_venue_features.py

Variants, all trained on ONE identical frame (Spark's GBT moves by about 1e-4 when the same
rows arrive in a different order, so variants must share the frame):

  baseline              the production feature set
  + venue               venue_par_rpo, venue_target_gap, venue_chase_win_rate, venue_prior_matches
  + venue (placebo)     the SAME four features, but every match is given the venue information
                        of a different, randomly chosen match
  + opponent            the step-5 opponent features
  + opponent + venue    both

Why the placebo: venue_target_gap = (target runs / overs) - venue par. The first half is just
the target's run rate, a rearrangement of information the model already holds, which could
help a linear model even with a meaningless par. If the placebo helps as much as the real
thing, the "venue effect" is that arithmetic and not venue knowledge. The comparison that
answers the question is "+ venue" against "+ venue (placebo)".

Differences are reported with a match-level (cluster-robust) 95% interval, because balls from
the same match are strongly correlated. Nothing is tuned; the production model is not touched.
"""
import math
import sys
import time

from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.functions import vector_to_array
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

sys.path.insert(0, "/workspace/etl")
sys.path.insert(0, "/workspace/ml")
import schema_utils as S  # noqa: E402
from calibrate_win_probability import metrics  # noqa: E402
from evaluate_opponent_features import (OPP_FEATURES, attach_opponent_features,  # noqa: E402
                                        logloss_col, make_pipeline)
from train_win_probability import LABEL, chronological_split, prepare  # noqa: E402

VENUE_DIR = f"{S.PROCESSED}/features/venue"
OUT_VARIANTS = f"{S.PROCESSED}/venue_feature_evaluation"
OUT_COMPARISONS = f"{S.PROCESSED}/venue_feature_comparisons"
VENUE_FEATURES = ["venue_par_rpo", "venue_target_gap", "venue_chase_win_rate", "venue_prior_matches"]
PLACEBO_FEATURES = ["p" + c for c in VENUE_FEATURES]


def venue_columns(df, v, prefix):
    """Attach the venue table `v` (one row per match) under a column-name prefix."""
    v = v.select(F.col("match_id").alias("_vm"),
                 F.col("venue_par_rpo").alias(prefix + "venue_par_rpo"),
                 F.col("venue_chase_win_rate").alias(prefix + "venue_chase_win_rate"),
                 F.col("venue_prior_matches").alias(prefix + "venue_prior_matches"))
    out = df.join(v, F.col("match_id") == F.col("_vm"), "left").drop("_vm")
    out = (out.withColumn(prefix + "venue_par_rpo", F.col(prefix + "venue_par_rpo").cast("double"))
              .withColumn(prefix + "venue_chase_win_rate",
                          F.coalesce(F.col(prefix + "venue_chase_win_rate"), F.lit(0.5)).cast("double"))
              .withColumn(prefix + "venue_prior_matches",
                          F.coalesce(F.col(prefix + "venue_prior_matches"), F.lit(0)).cast("double")))
    # The target's run rate minus the ground's par: how demanding is this target HERE?
    return out.withColumn(prefix + "venue_target_gap",
                          F.col("target_runs") / F.col("innings_overs_cap") - F.col(prefix + "venue_par_rpo"))


def shuffled_venue_table(v):
    """Each match receives the venue information of a different, randomly chosen match."""
    a = v.select("match_id").withColumn("_pos", F.row_number().over(Window.orderBy("match_id")))
    b = (v.select(F.col("match_id").alias("_src")).withColumn("_r", F.rand(2024))
          .withColumn("_pos", F.row_number().over(Window.orderBy("_r"))))
    pair = a.join(b, "_pos").select("match_id", "_src")
    src = v.select(F.col("match_id").alias("_src"), "venue_par_rpo", "venue_chase_win_rate",
                   "venue_prior_matches")
    return pair.join(src, "_src").drop("_src")


def compare(a, b, breakdown_by_prior=False):
    """Paired change in Log-Loss going from scored frame `a` to `b`, with a match-level interval."""
    j = (a.select("_k", "match_id", "season", LABEL, F.col("p").alias("p0"), "venue_prior_matches")
          .join(b.select("_k", F.col("p").alias("p1")), "_k")
          .withColumn("d", logloss_col("p1") - logloss_col("p0")).cache())
    per_match = j.groupBy("match_id").agg(F.sum("d").alias("s"), F.count("*").alias("n")).collect()
    n_all = float(sum(r["n"] for r in per_match))
    delta = sum(r["s"] for r in per_match) / n_all
    se = math.sqrt(sum((r["s"] - delta * r["n"]) ** 2 for r in per_match) / (n_all ** 2))
    lo, hi = delta - 1.96 * se, delta + 1.96 * se
    verdict = "IMPROVES" if hi < 0 else "WORSENS" if lo > 0 else "no distinguishable change"
    seasons = j.groupBy("season").agg(F.avg("d").alias("d")).collect()
    helped = sum(1 for r in seasons if r["d"] < 0)
    extra = ""
    if breakdown_by_prior:
        bucket = (F.when(F.col("venue_prior_matches") < 1, "0 earlier matches")
                   .when(F.col("venue_prior_matches") < 5, "1-4 earlier")
                   .when(F.col("venue_prior_matches") < 20, "5-19 earlier")
                   .otherwise("20+ earlier"))
        rows = (j.withColumn("b", bucket).groupBy("b").agg(F.avg("d").alias("d"), F.count("*").alias("n"))
                 .collect())
        order = ["0 earlier matches", "1-4 earlier", "5-19 earlier", "20+ earlier"]
        rows = sorted(rows, key=lambda r: order.index(r["b"]))
        extra = "\n".join(f"      depth of venue history {r['b']:<18}{r['d']:>+10.5f}   ({r['n']} rows)"
                          for r in rows)
    return delta, lo, hi, verdict, helped, len(seasons), len(per_match), extra


def main():
    spark = S.build_spark("evaluate_venue_features")
    spark.sparkContext.setLogLevel("WARN")

    df = prepare(spark.read.parquet(S.TRAINING_TABLE))
    n0 = df.count()
    v = spark.read.parquet(VENUE_DIR)
    df = attach_opponent_features(df, spark)
    df = venue_columns(df, v, "")
    df = venue_columns(df, shuffled_venue_table(v), "p").cache()
    n1 = df.count()
    print(f"training rows before/after all joins: {n0} / {n1} [{'OK' if n0 == n1 else 'ROWS MULTIPLIED - STOP'}]",
          flush=True)
    assert n0 == n1

    train, test, cutoff = chronological_split(df)
    test = test.withColumn("_k", F.monotonically_increasing_id()).cache()
    n_test = test.count()
    print(f"test rows: {n_test} (seasons >= {cutoff})", flush=True)

    print("\n== venue coverage on the held-out rows (earlier first innings at the same ground) ==", flush=True)
    for label, cond in [(">= 1", F.col("venue_prior_matches") >= 1), (">= 5", F.col("venue_prior_matches") >= 5),
                        (">= 20", F.col("venue_prior_matches") >= 20)]:
        print(f"  {label:<6}{test.filter(cond).count() / n_test:>8.1%}", flush=True)
    corr = test.select(F.corr("venue_par_rpo", "pvenue_par_rpo")).first()[0]
    print(f"  correlation between real and placebo par on held-out rows: {corr:+.3f} (should be near 0)",
          flush=True)

    variants = [("baseline", []), ("+ venue", VENUE_FEATURES), ("+ venue (placebo)", PLACEBO_FEATURES),
                ("+ opponent", OPP_FEATURES), ("+ opponent + venue", OPP_FEATURES + VENUE_FEATURES)]
    specs = [("Gradient-Boosted Trees",
              lambda: GBTClassifier(labelCol=LABEL, featuresCol="features", maxIter=60, maxDepth=5, seed=42)),
             ("Logistic Regression",
              lambda: LogisticRegression(labelCol=LABEL, featuresCol="features", maxIter=100,
                                         regParam=0.0, standardization=True))]

    scored, results = {}, {}
    for name, make in specs:
        for variant, extra in variants:
            t0 = time.time()
            model = make_pipeline(extra, make()).fit(train)
            sc = (model.transform(test).withColumn("p", vector_to_array(F.col("probability"))[1])
                  .select("_k", "match_id", "season", LABEL, "p", "venue_prior_matches").cache())
            scored[(name, variant)] = sc
            results[(name, variant)] = metrics(sc, "p")
            print(f"[{name} | {variant}] {time.time() - t0:.0f}s", flush=True)

    ref = {r["model"]: r for r in spark.read.parquet(f"{S.PROCESSED}/model_comparison").collect()}
    print("\n== baseline vs model_comparison (row-order noise floor ~1e-4 for GBT) ==", flush=True)
    for name, _ in specs:
        m, r = results[(name, "baseline")], ref[name]
        print(f"  {name:<24} max abs difference {max(abs(m['logloss'] - r['logloss']), abs(m['auc'] - r['auc'])):.1e}",
              flush=True)

    print("\n== held-out results ==", flush=True)
    print(f"{'model':<24}{'variant':<22}{'AUC':>8}{'Brier':>8}{'LogLoss':>9}{'ECE':>8}", flush=True)
    for (name, variant), m in results.items():
        print(f"{name:<24}{variant:<22}{m['auc']:>8.4f}{m['brier']:>8.4f}{m['logloss']:>9.4f}{m['ece']:>8.4f}",
              flush=True)

    comparisons = [("+ venue vs baseline", "baseline", "+ venue", False),
                   ("placebo vs baseline", "baseline", "+ venue (placebo)", False),
                   ("REAL venue vs PLACEBO venue", "+ venue (placebo)", "+ venue", True),
                   ("+ opponent + venue vs + opponent", "+ opponent", "+ opponent + venue", False)]
    comp_rows = []
    for name, _ in specs:
        print(f"\n===== {name}: change in Log-Loss (negative = better), 95% match-level interval =====", flush=True)
        for label, va, vb, breakdown in comparisons:
            delta, lo, hi, verdict, helped, n_seasons, n_matches, extra = compare(
                scored[(name, va)], scored[(name, vb)], breakdown)
            print(f"  {label:<36}{delta:>+9.5f}  [{lo:+.5f}, {hi:+.5f}]  {verdict:<26} "
                  f"helped {helped}/{n_seasons} seasons", flush=True)
            if extra:
                print(extra, flush=True)
            comp_rows.append((name, label, delta, lo, hi, verdict))

    var_rows = [(n, v_, m["auc"], m["brier"], m["logloss"], m["ece"]) for (n, v_), m in results.items()]
    spark.createDataFrame(var_rows, StructType([
        StructField("model", StringType(), False), StructField("variant", StringType(), False),
        StructField("auc", DoubleType(), False), StructField("brier", DoubleType(), False),
        StructField("logloss", DoubleType(), False), StructField("ece", DoubleType(), False)])
    ).write.mode("overwrite").parquet(OUT_VARIANTS)
    spark.createDataFrame(comp_rows, StructType([
        StructField("model", StringType(), False), StructField("comparison", StringType(), False),
        StructField("delta_logloss", DoubleType(), False), StructField("ci_low", DoubleType(), False),
        StructField("ci_high", DoubleType(), False), StructField("verdict", StringType(), False)])
    ).write.mode("overwrite").parquet(OUT_COMPARISONS)
    print(f"\nwrote {OUT_VARIANTS} ({spark.read.parquet(OUT_VARIANTS).count()} rows) and "
          f"{OUT_COMPARISONS} ({spark.read.parquet(OUT_COMPARISONS).count()} rows)", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
