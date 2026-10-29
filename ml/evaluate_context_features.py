#!/usr/bin/env python3
"""Is the venue gain really venue, or is it just WHO is playing (gender and level)?

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/ml/evaluate_context_features.py

Background: the production feature set has match_type but neither gender nor team_type. On the
held-out seasons the raw model over-predicts the chasing team by about 17 points in the
45-85% range for women's international matches and 14 for women's club matches, while it
under-predicts men's club matches, and women's cricket doubled its share of rows (14% of
training rows, 30% of held-out rows). A "venue par" built from a format norm split by gender
and team_type would smuggle that missing context in, so this script gives the baseline the
two obvious columns directly:

  is_female, is_international

Variants (all trained on one identical frame; Spark's GBT moves by about 1e-4 with row order):
  baseline | + context | + venue | + context + venue          (Logistic Regression)
  baseline | + context                                         (GBT, the production model)

Reports the paired change in Log-Loss with a match-level 95% interval, and the raw calibration
gap by gender and team type for every variant, so it is visible whether the segment gaps close.
"""
import sys
import time

from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
sys.path.insert(0, "/workspace/ml")
import schema_utils as S  # noqa: E402
from calibrate_win_probability import metrics  # noqa: E402
from evaluate_opponent_features import attach_opponent_features, make_pipeline  # noqa: E402
from evaluate_venue_features import (VENUE_DIR, VENUE_FEATURES, compare,  # noqa: E402
                                     venue_columns)
from train_win_probability import LABEL, chronological_split, prepare  # noqa: E402

CONTEXT_FEATURES = ["is_female", "is_international"]
SEGMENTS = [("female", "club"), ("female", "international"), ("male", "club"), ("male", "international")]


def segment_gaps(sc):
    """Mean predicted minus actual chaser win rate, per gender x team_type, for the 45-85% band."""
    mid = (F.col("p") >= 0.45) & (F.col("p") < 0.85)
    rows = (sc.groupBy("gender", "team_type")
              .agg(F.avg(F.when(mid, F.col("p"))).alias("pm"), F.avg(F.when(mid, F.col(LABEL))).alias("am"),
                   F.avg("p").alias("pa"), F.avg(LABEL).alias("aa")).collect())
    return {(r["gender"], r["team_type"]): (r["pa"] - r["aa"], r["pm"] - r["am"]) for r in rows}


def main():
    spark = S.build_spark("evaluate_context_features")
    spark.sparkContext.setLogLevel("WARN")

    df = prepare(spark.read.parquet(S.TRAINING_TABLE))
    n0 = df.count()
    df = attach_opponent_features(df, spark)               # keep the frame shape of the other evaluations
    df = venue_columns(df, spark.read.parquet(VENUE_DIR), "")
    df = (df.withColumn("is_female", (F.col("gender") == "female").cast("double"))
            .withColumn("is_international", (F.col("team_type") == "international").cast("double"))).cache()
    n1 = df.count()
    print(f"rows before/after joins: {n0} / {n1} [{'OK' if n0 == n1 else 'ROWS MULTIPLIED - STOP'}]", flush=True)
    assert n0 == n1

    train, test, cutoff = chronological_split(df)
    test = test.withColumn("_k", F.monotonically_increasing_id()).cache()
    print(f"test rows: {test.count()} (seasons >= {cutoff})", flush=True)

    plan = {"Logistic Regression": [("baseline", []), ("+ context", CONTEXT_FEATURES),
                                    ("+ venue", VENUE_FEATURES),
                                    ("+ context + venue", CONTEXT_FEATURES + VENUE_FEATURES)],
            "Gradient-Boosted Trees": [("baseline", []), ("+ context", CONTEXT_FEATURES)]}
    makers = {"Logistic Regression": lambda: LogisticRegression(labelCol=LABEL, featuresCol="features",
                                                               maxIter=100, regParam=0.0, standardization=True),
              "Gradient-Boosted Trees": lambda: GBTClassifier(labelCol=LABEL, featuresCol="features",
                                                              maxIter=60, maxDepth=5, seed=42)}
    scored, results, gaps = {}, {}, {}
    for name, variants in plan.items():
        for variant, extra in variants:
            t0 = time.time()
            model = make_pipeline(extra, makers[name]()).fit(train)
            sc = (model.transform(test).withColumn("p", vector_to_array(F.col("probability"))[1])
                  .select("_k", "match_id", "season", LABEL, "p", "venue_prior_matches", "gender", "team_type")
                  .cache())
            scored[(name, variant)] = sc
            results[(name, variant)] = metrics(sc, "p")
            gaps[(name, variant)] = segment_gaps(sc)
            print(f"[{name} | {variant}] {time.time() - t0:.0f}s", flush=True)

    print("\n== held-out results ==", flush=True)
    print(f"{'model':<24}{'variant':<20}{'AUC':>8}{'Brier':>8}{'LogLoss':>9}{'ECE':>8}", flush=True)
    for (name, variant), m in results.items():
        print(f"{name:<24}{variant:<20}{m['auc']:>8.4f}{m['brier']:>8.4f}{m['logloss']:>9.4f}{m['ece']:>8.4f}",
              flush=True)

    print("\n== raw calibration gap in the 45-85% band, by segment (predicted - actual; 0 is perfect) ==",
          flush=True)
    for name, variants in plan.items():
        print(f"\n[{name}]", flush=True)
        print(f"  {'segment':<24}" + "".join(f"{v:>20}" for v, _ in variants), flush=True)
        for seg in SEGMENTS:
            print(f"  {seg[0] + ' ' + seg[1]:<24}"
                  + "".join(f"{gaps[(name, v)][seg][1]:>+20.3f}" for v, _ in variants), flush=True)

    comparisons = {"Logistic Regression": [("+ context vs baseline", "baseline", "+ context"),
                                           ("+ venue vs baseline", "baseline", "+ venue"),
                                           ("venue ON TOP OF context", "+ context", "+ context + venue"),
                                           ("+ context + venue vs baseline", "baseline", "+ context + venue")],
                   "Gradient-Boosted Trees": [("+ context vs baseline", "baseline", "+ context")]}
    for name, comps in comparisons.items():
        print(f"\n===== {name}: change in Log-Loss (negative = better), 95% match-level interval =====", flush=True)
        for label, va, vb in comps:
            delta, lo, hi, verdict, helped, n_seasons, _, _ = compare(scored[(name, va)], scored[(name, vb)])
            print(f"  {label:<34}{delta:>+9.5f}  [{lo:+.5f}, {hi:+.5f}]  {verdict:<26} "
                  f"helped {helped}/{n_seasons} seasons", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
