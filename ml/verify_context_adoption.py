#!/usr/bin/env python3
"""Before/after check for adopting is_female and is_international into the production model (step 6b).

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/ml/verify_context_adoption.py

Compares the scored table and the calibrated table now in /data/processed with the copies saved in
/data/backup_pre6b before the re-run. Both were produced by the same pipeline on the same rows;
only the production model's features differ. Reports:

  1. the two scored tables cover exactly the same (match, ball) rows
  2. held-out Log-Loss / Brier / ECE, before and after, with a match-level 95% interval on the
     paired change (rows within one chase are not independent, so the interval resamples matches)
  3. the raw calibration gap in the 45-85% band by gender x level, before and after
  4. the calibrated (isotonic) results on the evaluation seasons, before and after, which shows how
     much work the calibrator still has to do now that the model knows who is playing
"""
import math
import sys

from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402

BACKUP = "/data/backup_pre6b"
EPS = 1e-15
SEGMENTS = [("female", "club"), ("female", "international"), ("male", "club"), ("male", "international")]


def logloss(p, y):
    p = F.least(F.greatest(p, F.lit(EPS)), F.lit(1 - EPS))
    return -(y * F.log(p) + (1 - y) * F.log(1 - p))


def ece10(df, p):
    """Expected calibration error over ten equal-width buckets."""
    b = (df.withColumn("b", F.least(F.floor(F.col(p) * 10), F.lit(9)))
           .groupBy("b").agg(F.count("*").alias("n"), F.avg(p).alias("pm"), F.avg("y").alias("am")).collect())
    n = sum(r["n"] for r in b)
    return sum(r["n"] / n * abs(r["pm"] - r["am"]) for r in b)


def main():
    spark = S.build_spark("verify_context_adoption")
    spark.sparkContext.setLogLevel("WARN")

    new = spark.read.parquet(S.WIN_PROB_BY_BALL)
    old = spark.read.parquet(f"{BACKUP}/win_probability_by_ball")
    cal_new = spark.read.parquet(f"{S.PROCESSED}/win_probability_calibrated")
    cal_old = spark.read.parquet(f"{BACKUP}/win_probability_calibrated")
    meta = (spark.read.parquet(S.TRAINING_TABLE).select("match_id", "gender", "team_type").distinct())
    assert meta.count() == meta.select("match_id").distinct().count(), "a match has two gender/team_type values"

    key = ["match_id", "innings_num", "ball_seq"]
    print("== 1. same rows? ==")
    n_new, n_old = new.count(), old.count()
    d_new = new.select(*key).distinct().count()
    both = (new.select(*key, F.col("win_probability").alias("p_new"))
            .join(old.select(*key, F.col("win_probability").alias("p_old")), key, "inner").count())
    print(f"rows new {n_new}, old {n_old}; distinct keys new {d_new}; keys present in both {both} "
          f"[{'OK' if n_new == n_old == d_new == both else 'MISMATCH'}]")

    # held-out = everything the calibrated table does not label 'train' (same season rule both times)
    split = cal_new.select(*key, "split", "match_type")
    split_old = cal_old.select(*key, F.col("split").alias("split_old"))
    chk = split.join(split_old, key).filter(F.col("split") != F.col("split_old")).count()
    print(f"rows whose train/calibration/evaluation label changed: {chk} [{'OK' if chk == 0 else 'MISMATCH'}]")

    f = (new.select(*key, "season", F.col("label_batting_team_wins").alias("y"), F.col("win_probability").alias("p_new"))
         .join(old.select(*key, F.col("win_probability").alias("p_old")), key)
         .join(split.select(*key, "split"), key).join(meta, "match_id")
         .filter(F.col("split") != "train")
         .withColumn("ll_new", logloss(F.col("p_new"), F.col("y")))
         .withColumn("ll_old", logloss(F.col("p_old"), F.col("y")))
         .withColumn("d", F.col("ll_new") - F.col("ll_old"))
         .withColumn("br_new", (F.col("p_new") - F.col("y")) ** 2)
         .withColumn("br_old", (F.col("p_old") - F.col("y")) ** 2)).cache()
    n = f.count()

    print(f"\n== 2. held-out seasons ({n} rows): raw GBT before vs after ==")
    a = f.agg(F.avg("ll_old").alias("llo"), F.avg("ll_new").alias("lln"),
              F.avg("br_old").alias("bro"), F.avg("br_new").alias("brn")).first()
    print(f"  Log-Loss {a['llo']:.4f} -> {a['lln']:.4f}   Brier {a['bro']:.4f} -> {a['brn']:.4f}   "
          f"ECE {ece10(f.withColumnRenamed('p_old', 'p'), 'p'):.4f} -> {ece10(f.withColumnRenamed('p_new', 'p'), 'p'):.4f}")
    # match-level cluster-robust interval for the mean paired difference (ratio estimator)
    cl = f.groupBy("match_id").agg(F.sum("d").alias("s"), F.count("*").alias("m"),
                                   F.sum((F.col("d") < 0).cast("int")).alias("better")).collect()
    N = sum(r["m"] for r in cl)
    mean = sum(r["s"] for r in cl) / N
    se = math.sqrt(sum((r["s"] - mean * r["m"]) ** 2 for r in cl)) / N
    print(f"  paired change in Log-Loss {mean:+.5f}, 95% interval [{mean - 1.96 * se:+.5f}, {mean + 1.96 * se:+.5f}] "
          f"over {len(cl)} chases -> {'REAL improvement' if mean + 1.96 * se < 0 else 'not distinguishable from zero'}")
    per_season = f.groupBy("season").agg(F.avg("d").alias("d")).orderBy("season").collect()
    print("  by season (negative = better): " + "  ".join(f"{r['season']} {r['d']:+.4f}" for r in per_season))
    print(f"  seasons improved: {sum(r['d'] < 0 for r in per_season)} of {len(per_season)}")

    print("\n== 3. raw calibration gap in the 45-85% band, predicted - actual (0 is perfect) ==")
    print(f"  {'segment':<22}{'rows':>9}{'before':>10}{'after':>10}")
    for g, t in SEGMENTS:
        s = f.filter((F.col("gender") == g) & (F.col("team_type") == t))
        r = {}
        for tag in ("old", "new"):
            mid = s.filter((F.col(f"p_{tag}") >= 0.45) & (F.col(f"p_{tag}") < 0.85))
            x = mid.agg(F.avg(f"p_{tag}").alias("pm"), F.avg("y").alias("am")).first()
            r[tag] = x["pm"] - x["am"]
        print(f"  {g + ' ' + t:<22}{s.count():>9}{r['old']:>+10.3f}{r['new']:>+10.3f}")

    print("\n== 4. isotonic calibration on the evaluation seasons only (never used for fitting) ==")
    for tag, c in (("before", cal_old), ("after", cal_new)):
        e = (c.filter(F.col("split") == "evaluation").withColumn("y", F.col("label_batting_team_wins"))
              .withColumn("ll_raw", logloss(F.col("win_probability"), F.col("y")))
              .withColumn("ll_cal", logloss(F.col("win_probability_calibrated"), F.col("y"))))
        m = e.agg(F.avg("ll_raw").alias("r"), F.avg("ll_cal").alias("c")).first()
        er = ece10(e, "win_probability")
        ec = ece10(e, "win_probability_calibrated")
        print(f"  {tag:<7} Log-Loss raw {m['r']:.4f} -> calibrated {m['c']:.4f} (gain {m['r'] - m['c']:.4f}); "
              f"ECE raw {er:.4f} -> calibrated {ec:.4f}")
    spark.stop()


if __name__ == "__main__":
    main()
