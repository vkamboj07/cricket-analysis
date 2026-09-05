#!/usr/bin/env python3
"""Rolling player-form features, computed without leakage.

    docker exec spark-master spark-submit \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/features/player_form_features.py

For every (player, match) pair this answers "what was this player's form
*going into* that match" -- the current match never contributes to its own
rolling figure. Output: /data/processed/player_form.
"""
import sys

from pyspark.sql import Window
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402

FORM_WINDOW = 10  # matches


def per_match_batting(deliveries):
    """Runs, balls faced and dismissals per batter per match."""
    return (deliveries
            .filter(F.col("batter_id").isNotNull())
            .groupBy("match_id", F.col("batter_id").alias("player_id"))
            .agg(F.sum("runs_batter").alias("runs_scored"),
                 # A wide is not a ball faced by the batter; a no-ball is.
                 F.sum((F.col("extra_wides") == 0).cast("int")).alias("balls_faced")))


def per_match_dismissals(deliveries):
    return (deliveries
            .filter(F.col("player_out_id").isNotNull())
            .groupBy("match_id", F.col("player_out_id").alias("player_id"))
            .agg(F.count("*").alias("dismissals")))


def per_match_bowling(deliveries):
    """Runs conceded, legal balls and wickets per bowler per match.

    Byes and legbyes are not charged to the bowler; run-outs are not credited
    to them either.
    """
    bowler_wicket_kinds = ("bowled", "caught", "lbw", "stumped",
                           "caught and bowled", "hit wicket")
    return (deliveries
            .filter(F.col("bowler_id").isNotNull())
            .groupBy("match_id", F.col("bowler_id").alias("player_id"))
            .agg(F.sum(F.col("runs_total") - F.col("extra_byes") - F.col("extra_legbyes"))
                  .alias("runs_conceded"),
                 F.sum(F.col("is_legal_ball").cast("int")).alias("balls_bowled"),
                 F.sum(F.when(F.col("wicket_kind").isin(*bowler_wicket_kinds), 1)
                        .otherwise(0)).alias("wickets_taken")))


def build(deliveries, matches):
    lo = deliveries.filter(F.col("format_group") == "limited_overs")
    match_dates = matches.select(
        F.col("match_id").alias("m_match_id"), "match_date", "match_type")

    batting = per_match_batting(lo)
    dismissals = per_match_dismissals(lo)
    bowling = per_match_bowling(lo)

    per_match = (batting
                 .join(dismissals, ["match_id", "player_id"], "full_outer")
                 .join(bowling, ["match_id", "player_id"], "full_outer")
                 .fillna(0, ["runs_scored", "balls_faced", "dismissals",
                             "runs_conceded", "balls_bowled", "wickets_taken"]))

    per_match = (per_match
                 .join(match_dates, per_match["match_id"] == match_dates["m_match_id"], "left")
                 .drop("m_match_id"))

    # Ordering by date with match_id as tiebreaker keeps same-day double
    # headers deterministic. rowsBetween(-K, -1) is the leakage guard: the
    # upper bound of -1 excludes the current match from its own average.
    form_window = (Window.partitionBy("player_id")
                   .orderBy("match_date", "match_id")
                   .rowsBetween(-FORM_WINDOW, -1))

    rolled = (per_match
              .withColumn("form_runs", F.sum("runs_scored").over(form_window))
              .withColumn("form_balls_faced", F.sum("balls_faced").over(form_window))
              .withColumn("form_dismissals", F.sum("dismissals").over(form_window))
              .withColumn("form_runs_conceded", F.sum("runs_conceded").over(form_window))
              .withColumn("form_balls_bowled", F.sum("balls_bowled").over(form_window))
              .withColumn("form_wickets", F.sum("wickets_taken").over(form_window))
              .withColumn("matches_in_window", F.count("*").over(form_window)))

    return rolled.select(
        "player_id", "match_id", "match_date", "match_type", "matches_in_window",
        # Nulls rather than zeros for debutants: a player with no prior matches
        # has unknown form, which is not the same as bad form. Stage 6 imputes.
        F.when(F.col("form_balls_faced") > 0,
               F.col("form_runs") * 100.0 / F.col("form_balls_faced"))
         .alias("rolling_strike_rate"),
        F.when(F.col("form_dismissals") > 0,
               F.col("form_runs") / F.col("form_dismissals"))
         .alias("rolling_batting_avg"),
        F.when(F.col("form_balls_bowled") > 0,
               F.col("form_runs_conceded") * 6.0 / F.col("form_balls_bowled"))
         .alias("rolling_economy"),
        F.when(F.col("form_wickets") > 0,
               F.col("form_balls_bowled") / F.col("form_wickets"))
         .alias("rolling_bowling_sr"),
        F.col("form_balls_faced").alias("form_balls_faced"),
        F.col("form_balls_bowled").alias("form_balls_bowled"),
    )


def main():
    spark = S.build_spark("player_form_features")
    spark.sparkContext.setLogLevel("WARN")

    deliveries = spark.read.parquet(S.DELIVERIES)
    matches = spark.read.parquet(S.MATCHES)

    out = build(deliveries, matches)
    out.write.mode("overwrite").parquet(S.PLAYER_FORM)
    print(f"wrote player_form -> {S.PLAYER_FORM}")

    spark.stop()


if __name__ == "__main__":
    main()
