#!/usr/bin/env python3
"""Win Probability Added (WPA): how much each delivery moved the chase.

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/ml/compute_wpa.py

WPA_t = P_t - P_{t-1}, taken from the chasing team's point of view: positive
means the delivery helped the chasers, negative means it helped the bowling side.

Each scored row is the match state AFTER that ball (score and balls remaining
include it), so the first ball of a chase has no "previous" probability to
subtract. Its WPA is left null instead of being compared with an invented prior.

Reads  win_probability_by_ball (written by score_all_matches.py).
Writes <processed>/wpa_by_ball, partitioned by match_type.
"""
import sys

from pyspark.sql import Window
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402

WPA_BY_BALL = f"{S.PROCESSED}/wpa_by_ball"


def main():
    spark = S.build_spark("compute_wpa")
    spark.sparkContext.setLogLevel("WARN")

    scored = spark.read.parquet(S.WIN_PROB_BY_BALL)
    n_in = scored.count()

    # One chase = one (match_id, innings_num). ball_seq orders the deliveries in it.
    w = Window.partitionBy("match_id", "innings_num").orderBy("ball_seq")
    wpa = (scored
           .withColumn("prev_win_probability", F.lag("win_probability", 1).over(w))
           .withColumn("wpa", F.col("win_probability") - F.col("prev_win_probability"))
           .select("match_id", "season", "match_type", "innings_num", "batting_team",
                   "over", "ball_seq", "balls_completed_in_innings", "current_score",
                   "wickets_down", "balls_remaining", "target_runs", "runs_required",
                   "batter", "bowler", "win_probability", "prev_win_probability", "wpa"))

    wpa.write.mode("overwrite").partitionBy("match_type").parquet(WPA_BY_BALL)
    print(f"wrote WPA -> {WPA_BY_BALL}")

    # ---- checks, read back from what was actually written ----
    out = spark.read.parquet(WPA_BY_BALL)
    n_out = out.count()
    chases = out.select("match_id", "innings_num").distinct().count()
    null_wpa = out.filter(F.col("wpa").isNull()).count()
    dup_keys = (out.groupBy("match_id", "innings_num", "ball_seq").count()
                .filter(F.col("count") > 1).count())

    print("\n== checks ==")
    print(f"rows in  (win_probability_by_ball): {n_in}")
    print(f"rows out (wpa_by_ball)            : {n_out}   [{'OK' if n_in == n_out else 'MISMATCH'}]")
    print(f"distinct chases                   : {chases}")
    print(f"null WPA (expect one per chase)   : {null_wpa}   [{'OK' if null_wpa == chases else 'MISMATCH'}]")
    print(f"duplicate (match,innings,ball_seq): {dup_keys}   [{'OK' if dup_keys == 0 else 'AMBIGUOUS ORDER'}]")

    # Telescoping identity: sum of WPA over a chase == last probability - first probability.
    first_last = (out.groupBy("match_id", "innings_num")
                  # min_by/max_by need Spark 3.3; min/max over (ball_seq, p) structs
                  # picks the same rows on this cluster's Spark 3.1.1.
                  .agg(F.min(F.struct("ball_seq", "win_probability"))
                       .getField("win_probability").alias("p_first"),
                       F.max(F.struct("ball_seq", "win_probability"))
                       .getField("win_probability").alias("p_last"),
                       F.sum("wpa").alias("wpa_sum")))
    max_err = (first_last
               .select(F.max(F.abs(F.col("wpa_sum") - (F.col("p_last") - F.col("p_first"))))
                       .alias("e")).first()["e"])
    print(f"telescoping max abs error         : {max_err:.2e}   [{'OK' if max_err < 1e-9 else 'CHECK'}]")

    print("\n== WPA distribution (all deliveries) ==")
    out.select(F.min("wpa").alias("min"), F.max("wpa").alias("max"),
               F.mean("wpa").alias("mean"), F.stddev("wpa").alias("stddev")).show(truncate=False)

    print("== 10 biggest single-ball swings ==")
    (out.filter(F.col("wpa").isNotNull())
        .orderBy(F.abs(F.col("wpa")).desc())
        .select("match_id", "match_type", "over", "batter", "bowler", "current_score",
                "wickets_down", "prev_win_probability", "win_probability", "wpa")
        .show(10, truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()
