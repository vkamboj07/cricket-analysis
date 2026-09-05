#!/usr/bin/env python3
"""Per-ball match-state features for limited-overs matches.

    docker exec spark-master spark-submit \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/features/match_state_features.py

Output: /data/processed/match_state -- one row per delivery, carrying the
state of the innings *as of that ball*, plus the eventual match result so
stage 6 has a label to train against.
"""
import sys

from pyspark.sql import Window
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402

# Two lookback windows: 2 overs (recent momentum) and 5 overs (spell context).
LOOKBACKS = (12, 30)


def build(deliveries, matches):
    lo = deliveries.filter(F.col("format_group") == "limited_overs")

    m = matches.select(
        F.col("match_id").alias("m_match_id"),
        "season", "match_date", "gender", "team_type", "venue",
        F.col("winner").alias("match_winner"),
        F.col("result_type"),
        F.col("overs_cap").alias("match_overs_cap"),
    )
    df = lo.join(m, lo["match_id"] == m["m_match_id"], "left").drop("m_match_id")

    ball_order = (Window.partitionBy("match_id", "innings_num")
                  .orderBy("over", "delivery_idx"))
    running = ball_order.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    df = (df
          .withColumn("current_score", F.sum("runs_total").over(running))
          .withColumn("wickets_down", F.sum("num_wickets").over(running)))

    # Rain-shortened matches: prefer the innings target's overs when present,
    # otherwise the match-level cap.
    overs_cap = F.coalesce(F.col("target_overs"), F.col("overs_cap"),
                           F.col("match_overs_cap"))
    df = (df
          .withColumn("innings_overs_cap", overs_cap)
          .withColumn("balls_remaining",
                      F.greatest(overs_cap * 6 - F.col("balls_completed_in_innings"),
                                 F.lit(0)))
          .withColumn("wickets_remaining", F.lit(10) - F.col("wickets_down"))
          .withColumn("current_run_rate",
                      F.when(F.col("balls_completed_in_innings") > 0,
                             F.col("current_score") * 6.0
                             / F.col("balls_completed_in_innings"))))

    # Chase-specific state. target_runs is the score to reach, so runs still
    # required is target - current, floored at 0.
    df = (df
          .withColumn("runs_required",
                      F.when(F.col("innings_num") == 2,
                             F.greatest(F.col("target_runs") - F.col("current_score"),
                                        F.lit(0))))
          .withColumn("required_run_rate",
                      F.when((F.col("innings_num") == 2) & (F.col("balls_remaining") > 0),
                             F.col("runs_required") * 6.0 / F.col("balls_remaining"))))

    # Momentum windows exclude the current ball (-1 upper bound): including it
    # would leak the very event a model is asked to react to.
    for n in LOOKBACKS:
        w = ball_order.rowsBetween(-n, -1)
        df = (df
              .withColumn(f"runs_last_{n}", F.coalesce(F.sum("runs_total").over(w), F.lit(0)))
              .withColumn(f"wickets_last_{n}",
                          F.coalesce(F.sum("num_wickets").over(w), F.lit(0))))

    # Label: did the team batting on this ball go on to win the match?
    df = df.withColumn(
        "label_batting_team_wins",
        F.when(F.col("match_winner").isNull(), F.lit(None).cast("int"))
         .otherwise((F.col("match_winner") == F.col("batting_team")).cast("int")))

    return df.select(
        "match_id", "format_group", "match_type", "season", "match_date",
        "gender", "team_type", "venue", "innings_num", "batting_team",
        "over", "delivery_idx", "ball_seq", "balls_completed_in_innings",
        "batter", "batter_id", "bowler", "bowler_id", "non_striker", "non_striker_id",
        "runs_total", "is_wicket", "is_legal_ball",
        "innings_overs_cap", "target_runs",
        "current_score", "wickets_down", "wickets_remaining", "balls_remaining",
        "current_run_rate", "runs_required", "required_run_rate",
        *[f"runs_last_{n}" for n in LOOKBACKS],
        *[f"wickets_last_{n}" for n in LOOKBACKS],
        "match_winner", "result_type", "label_batting_team_wins",
    )


def main():
    spark = S.build_spark("match_state_features")
    spark.sparkContext.setLogLevel("WARN")

    deliveries = spark.read.parquet(S.DELIVERIES)
    matches = spark.read.parquet(S.MATCHES)

    out = build(deliveries, matches)
    (out.write.mode("overwrite").partitionBy("match_type").parquet(S.MATCH_STATE))
    print(f"wrote match_state -> {S.MATCH_STATE}")

    spark.stop()


if __name__ == "__main__":
    main()
