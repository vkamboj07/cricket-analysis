#!/usr/bin/env python3
"""Calibrate the win-probability model (Platt scaling vs Isotonic Regression).

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/ml/calibrate_win_probability.py

Why the fit set is NOT the training seasons: the GBT is already well calibrated on
the seasons it trained on. Its over-optimism appears on newer seasons it never saw
(it over-predicted the chasing team in 9 of 10 held-out seasons), so a calibrator fit
on training-season predictions would learn "change nothing".

So the held-out period (the last 20% of seasons) is itself split in time:
    first half of the held-out seasons -> fit the calibrators
    second half                        -> judge them (never used for fitting)
Whole seasons go to one side only, so no match straddles the line.

Reads   win_probability_by_ball (raw GBT probabilities, from score_all_matches.py)
Writes  <models>/win_probability_isotonic         the fitted Isotonic model
        <processed>/win_probability_calibrated    raw + calibrated probability per ball,
                                                  with a `split` column saying which rows
                                                  are valid for judging (only `evaluation`)
"""
import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import IsotonicRegression
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
sys.path.insert(0, "/workspace/ml")
import schema_utils as S  # noqa: E402
from train_win_probability import chronological_split  # noqa: E402

LABEL = "label_batting_team_wins"
EPS = 1e-6
# Isotonic regression is a step function, so its extreme steps land on exactly 0.0 and
# 1.0. On the evaluation seasons 9 of the 2,639 rows stated as "0%" actually won, so a
# calibrated value is never allowed to claim certainty. 0.5% is a round display-safety
# limit in line with the real error rate in the extreme bins (about 0.2-0.5%); it is not
# fitted. Metrics are reported with and without it.
PROB_FLOOR = 0.005
PROB_CEIL = 0.995
CALIBRATED_PATH = f"{S.PROCESSED}/win_probability_calibrated"
ISOTONIC_PATH = S.MODEL_PATH.rsplit("/", 1)[0] + "/win_probability_isotonic"


def clip(col):
    return F.least(F.greatest(col, F.lit(EPS)), F.lit(1.0 - EPS))


def buckets(df, pcol):
    """10 equal-width probability buckets: n, mean predicted, mean actual."""
    b = F.least(F.floor(F.col(pcol) * 10), F.lit(9)).cast("int")
    return (df.withColumn("bucket", b).groupBy("bucket")
            .agg(F.count("*").alias("n"),
                 F.avg(pcol).alias("mean_pred"),
                 F.avg(LABEL).alias("actual"))
            .orderBy("bucket"))


def metrics(df, pcol):
    """Brier, Log-Loss, AUC and expected calibration error (10 buckets)."""
    p = clip(F.col(pcol))
    y = F.col(LABEL)
    row = df.select(F.avg((p - y) * (p - y)).alias("brier"),
                    F.avg(-(y * F.log(p) + (1 - y) * F.log(1 - p))).alias("logloss")).first()
    auc = BinaryClassificationEvaluator(rawPredictionCol=pcol, labelCol=LABEL,
                                        metricName="areaUnderROC").evaluate(df)
    n = df.count()
    ece = sum(r["n"] / n * abs(r["mean_pred"] - r["actual"])
              for r in buckets(df, pcol).collect())
    return {"brier": row["brier"], "logloss": row["logloss"], "auc": auc, "ece": ece}


def main():
    spark = S.build_spark("calibrate_win_probability")
    spark.sparkContext.setLogLevel("WARN")

    scored = spark.read.parquet(S.WIN_PROB_BY_BALL).withColumn(LABEL, F.col(LABEL).cast("double"))

    # Same chronological split the GBT was trained with, then halve the held-out seasons.
    _, test, cutoff = chronological_split(scored)
    test_seasons = sorted(r["season"] for r in test.select("season").distinct().collect())
    half = len(test_seasons) // 2
    cal_seasons, eval_seasons = test_seasons[:half], test_seasons[half:]
    print(f"GBT test cutoff season : {cutoff}")
    print(f"calibration seasons    : {cal_seasons}")
    print(f"evaluation seasons     : {eval_seasons}")

    split = (F.when(F.col("season").isin(*cal_seasons), "calibration")
             .when(F.col("season").isin(*eval_seasons), "evaluation")
             .otherwise("train"))
    df = scored.withColumn("split", split)
    cal = df.filter(F.col("split") == "calibration").cache()
    ev = df.filter(F.col("split") == "evaluation").cache()
    print(f"rows: calibration {cal.count()}  evaluation {ev.count()}")

    # ---- fit on the calibration split only ----
    iso = Pipeline(stages=[
        VectorAssembler(inputCols=["win_probability"], outputCol="features"),
        IsotonicRegression(featuresCol="features", labelCol=LABEL,
                           predictionCol="p_isotonic", isotonic=True),
    ]).fit(cal)
    logit = F.log(clip(F.col("win_probability")) / (1 - clip(F.col("win_probability"))))
    platt = Pipeline(stages=[
        VectorAssembler(inputCols=["logit_p"], outputCol="platt_features"),
        LogisticRegression(featuresCol="platt_features", labelCol=LABEL,
                           probabilityCol="platt_prob", predictionCol="platt_pred",
                           rawPredictionCol="platt_raw", regParam=0.0,
                           standardization=False, maxIter=200),
    ]).fit(cal.withColumn("logit_p", logit))

    iso_model = iso.stages[-1]
    knots = iso_model.predictions.toArray()
    monotone = all(b >= a - 1e-12 for a, b in zip(knots, knots[1:]))
    print(f"\nisotonic model: {len(iso_model.boundaries)} boundaries, "
          f"output range [{knots.min():.4f}, {knots.max():.4f}], "
          f"non-decreasing: {'OK' if monotone else 'VIOLATION'}")

    # ---- apply to every row, judge on the evaluation split only ----
    out = iso.transform(df)
    out = platt.transform(out.withColumn("logit_p", logit))
    out = (out.withColumn("p_platt", vector_to_array(F.col("platt_prob"))[1])
              .withColumn("win_probability_calibrated",
                          F.least(F.greatest(F.col("p_isotonic"), F.lit(PROB_FLOOR)),
                                  F.lit(PROB_CEIL))))
    ev_out = out.filter(F.col("split") == "evaluation").cache()

    results = {
        "raw (uncalibrated)": metrics(ev_out, "win_probability"),
        "Platt scaling": metrics(ev_out, "p_platt"),
        "Isotonic (unclipped)": metrics(ev_out, "p_isotonic"),
        "Isotonic + 0.5% clip": metrics(ev_out, "win_probability_calibrated"),
    }
    print("\n== judged on the evaluation seasons only (never used for fitting) ==")
    print(f"{'method':<22}{'Brier':>9}{'LogLoss':>10}{'AUC':>9}{'ECE':>9}")
    for name, m in results.items():
        print(f"{name:<22}{m['brier']:>9.4f}{m['logloss']:>10.4f}{m['auc']:>9.4f}{m['ece']:>9.4f}")

    print("\n== 10-bucket reliability, evaluation seasons: predicted vs actual chaser win rate ==")
    raw_b = {r["bucket"]: r for r in buckets(ev_out, "win_probability").collect()}
    iso_b = {r["bucket"]: r for r in buckets(ev_out, "win_probability_calibrated").collect()}
    print(f"{'bucket':<8}{'n_raw':>9}{'raw_pred':>10}{'actual':>8}   {'n_iso':>9}{'iso_pred':>10}{'actual':>8}")
    for b in range(10):
        r, i = raw_b.get(b), iso_b.get(b)
        if r is None or i is None:
            continue
        print(f"{b:<8}{r['n']:>9}{r['mean_pred']:>10.4f}{r['actual']:>8.4f}   "
              f"{i['n']:>9}{i['mean_pred']:>10.4f}{i['actual']:>8.4f}")

    print("\n== extremes on the evaluation seasons (why the clip exists) ==")
    for label, cond in [("unclipped isotonic == 1.0", F.col("p_isotonic") >= 1.0),
                        ("unclipped isotonic == 0.0", F.col("p_isotonic") <= 0.0)]:
        r = ev_out.filter(cond).agg(F.count("*").alias("n"), F.avg(LABEL).alias("a")).first()
        print(f"{label}: {r['n']} rows, actual chaser win rate "
              f"{'n/a' if r['n'] == 0 else format(r['a'], '.4f')}")

    # ---- persist the calibrator and the calibrated scores ----
    iso.write().overwrite().save(ISOTONIC_PATH)
    print(f"\nsaved calibrator -> {ISOTONIC_PATH}")
    (out.select("match_id", "season", "match_type", "innings_num", "ball_seq", "split",
                F.col("win_probability"), F.col("win_probability_calibrated"), LABEL)
        .write.mode("overwrite").partitionBy("match_type").parquet(CALIBRATED_PATH))

    back = spark.read.parquet(CALIBRATED_PATH)
    n_out = back.count()
    bad = back.filter(F.col("win_probability_calibrated").isNull()
                      | (F.col("win_probability_calibrated") < PROB_FLOOR - 1e-12)
                      | (F.col("win_probability_calibrated") > PROB_CEIL + 1e-12)).count()
    print(f"wrote {CALIBRATED_PATH}: {n_out} rows "
          f"[{'OK' if n_out == scored.count() else 'MISMATCH'}], "
          f"null or outside [{PROB_FLOOR}, {PROB_CEIL}]: {bad} [{'OK' if bad == 0 else 'CHECK'}]")
    spark.stop()


if __name__ == "__main__":
    main()
