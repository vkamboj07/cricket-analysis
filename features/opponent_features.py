#!/usr/bin/env python3
"""Opponent-performance features, computed without leakage.

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/features/opponent_features.py

Three feature tables, each answering "what did the record look like going INTO this
match": every figure is built from EARLIER matches only (ordered by match_date then
match_id; the window ends at the row before the current match), so the current match
never contributes to its own feature.

  batter   how a batter has scored against this opposing team, relative to their own
           overall strike rate in the same format family
  bowler   how a bowler has conceded against this opposing team, relative to their own
           overall economy in the same format family
  team     the team's win rate against this opponent in earlier meetings

Design choices
  * Format family (T20 + IT20, ODI + ODM): strike rates differ by about 40 points between
    T20 and ODI, so records are never pooled across families.
  * Ratios are shrunk toward the player's own overall rate with K pseudo-balls, so a thin
    record against one opponent barely moves the number and "no history" is exactly 1.0
    (no information) instead of noise. A player with fewer than MIN_PRIOR_BALLS earlier
    balls has no trustworthy overall rate, so the ratio is null (the model imputes it).
  * Team history is split by gender and team_type: "India" men and "India" women are
    different teams that share a name.
  * No-result matches count as neither a win nor a loss.
  * Super overs (innings 3+) are excluded from the history.

Each table is verified against an independent brute-force recomputation (a self-join on
strictly earlier matches, no window functions) on a random sample.
"""
import sys

from pyspark.sql import Window
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402

OPP_DIR = f"{S.PROCESSED}/features/opponent"
BATTER_OUT = f"{OPP_DIR}/batter"
BOWLER_OUT = f"{OPP_DIR}/bowler"
TEAM_OUT = f"{OPP_DIR}/team"

K = 30.0                # pseudo-balls pulling a thin record toward the player's own norm
MIN_PRIOR_BALLS = 60    # below this the player's own overall rate is not trusted
PRIOR = Window.unboundedPreceding


def family():
    """T20-type vs ODI-type. A function, not a module constant: Spark 3.1 cannot build a
    Column before a session exists, so it must not run at import time."""
    return F.when(F.col("match_type").isin("T20", "IT20"), "T20").otherwise("ODI")
VERIFY_SAMPLE = 400


def match_info(matches):
    return (matches.filter(F.col("format_group") == "limited_overs")
            .withColumn("fam", family())
            .select("match_id", "match_date", "fam", "gender", "team_type",
                    "team1", "team2", "winner"))


def _limited_overs(deliveries):
    return deliveries.filter((F.col("format_group") == "limited_overs")
                             & (F.col("innings_num") <= 2))


def per_match_batting(deliveries, info):
    """Runs and balls faced per batter per match, with the opposing team's name."""
    d = (_limited_overs(deliveries).filter(F.col("batter_id").isNotNull())
         .join(info.select("match_id", "match_date", "fam", "team1", "team2"), "match_id")
         .withColumn("opp", F.when(F.col("batting_team") == F.col("team1"), F.col("team2"))
                             .otherwise(F.col("team1"))))
    return (d.groupBy("match_id", "match_date", "fam", F.col("batter_id").alias("player_id"), "opp")
             # A wide is not a ball faced by the batter; a no-ball is.
             .agg(F.sum("runs_batter").alias("units"),
                  F.sum((F.col("extra_wides") == 0).cast("int")).alias("balls")))


def per_match_bowling(deliveries, info):
    """Runs conceded and legal balls per bowler per match. Byes and legbyes are not
    charged to the bowler (same rule as the MapReduce economy job)."""
    d = (_limited_overs(deliveries).filter(F.col("bowler_id").isNotNull())
         .join(info.select("match_id", "match_date", "fam"), "match_id")
         .withColumn("opp", F.col("batting_team")))
    return (d.groupBy("match_id", "match_date", "fam", F.col("bowler_id").alias("player_id"), "opp")
             .agg(F.sum(F.col("runs_total") - F.col("extra_byes") - F.col("extra_legbyes"))
                   .alias("units"),
                  F.sum(F.col("is_legal_ball").cast("int")).alias("balls")))


def add_prior_ratio(pm):
    """Prior totals against this opponent and overall, then the shrunk ratio."""
    order = ("match_date", "match_id")
    w_opp = Window.partitionBy("player_id", "fam", "opp").orderBy(*order).rowsBetween(PRIOR, -1)
    w_all = Window.partitionBy("player_id", "fam").orderBy(*order).rowsBetween(PRIOR, -1)
    x = (pm.withColumn("opp_units", F.coalesce(F.sum("units").over(w_opp), F.lit(0)))
           .withColumn("opp_balls", F.coalesce(F.sum("balls").over(w_opp), F.lit(0)))
           .withColumn("all_units", F.coalesce(F.sum("units").over(w_all), F.lit(0)))
           .withColumn("all_balls", F.coalesce(F.sum("balls").over(w_all), F.lit(0))))
    overall = F.when(F.col("all_balls") >= MIN_PRIOR_BALLS, F.col("all_units") / F.col("all_balls"))
    shrunk = (F.col("opp_units") + K * overall) / (F.col("opp_balls") + K)
    return x.withColumn("ratio", F.when(overall > 0, shrunk / overall))


def team_head_to_head(info):
    """Each team's record against each opponent in EARLIER meetings."""
    def side(team, opp):
        win = F.when(F.col("winner") == F.col(team), 1).when(F.col("winner") == F.col(opp), 0)
        return info.select("match_id", "match_date", "fam", "gender", "team_type",
                           F.col(team).alias("team"), F.col(opp).alias("opp"), win.alias("win"))

    both = side("team1", "team2").unionByName(side("team2", "team1"))
    w = (Window.partitionBy("team", "opp", "fam", "gender", "team_type")
         .orderBy("match_date", "match_id").rowsBetween(PRIOR, -1))
    return (both
            .withColumn("h2h_n", F.coalesce(F.count("win").over(w), F.lit(0)))      # count skips nulls
            .withColumn("h2h_wins", F.coalesce(F.sum("win").over(w), F.lit(0)))
            .withColumn("h2h_rate", (F.col("h2h_wins") + 1.0) / (F.col("h2h_n") + 2.0)))


def _earlier(a, b):
    """Row `a` is strictly earlier than row `b` in (match_date, match_id) order."""
    return ((F.col(f"{a}.match_date") < F.col(f"{b}.match_date"))
            | ((F.col(f"{a}.match_date") == F.col(f"{b}.match_date"))
               & (F.col(f"{a}.match_id") < F.col(f"{b}.match_id"))))


def verify_ratio(pm, out, label):
    """Recompute the prior totals with a plain self-join (no windows) on a sample."""
    sample = (out.orderBy(F.rand(7)).limit(VERIFY_SAMPLE)
              .select("match_id", "match_date", "player_id", "fam", "opp",
                      "opp_units", "opp_balls", "all_units", "all_balls").cache())
    s, e = sample.alias("s"), pm.alias("e")
    j = s.join(e, (F.col("e.player_id") == F.col("s.player_id"))
               & (F.col("e.fam") == F.col("s.fam")) & _earlier("e", "s"), "left")
    same_opp = F.col("e.opp") == F.col("s.opp")
    chk = (j.groupBy("s.match_id", "s.player_id")
            .agg(F.first("s.opp_units").alias("opp_units"), F.first("s.opp_balls").alias("opp_balls"),
                 F.first("s.all_units").alias("all_units"), F.first("s.all_balls").alias("all_balls"),
                 F.coalesce(F.sum(F.when(same_opp, F.col("e.units"))), F.lit(0)).alias("b_opp_units"),
                 F.coalesce(F.sum(F.when(same_opp, F.col("e.balls"))), F.lit(0)).alias("b_opp_balls"),
                 F.coalesce(F.sum("e.units"), F.lit(0)).alias("b_all_units"),
                 F.coalesce(F.sum("e.balls"), F.lit(0)).alias("b_all_balls")))
    n = chk.count()
    bad = chk.filter((F.col("opp_units") != F.col("b_opp_units"))
                     | (F.col("opp_balls") != F.col("b_opp_balls"))
                     | (F.col("all_units") != F.col("b_all_units"))
                     | (F.col("all_balls") != F.col("b_all_balls"))).count()
    with_hist = chk.filter(F.col("b_opp_balls") > 0).count()
    print(f"  verify {label}: {n} sampled rows, {bad} mismatches vs brute force "
          f"[{'OK' if bad == 0 else 'LEAK OR BUG'}]  ({with_hist} of them have earlier meetings, "
          f"so the check is not vacuous)", flush=True)
    return bad == 0


def verify_team(h2h):
    sample = (h2h.orderBy(F.rand(7)).limit(VERIFY_SAMPLE).cache())
    s, e = sample.alias("s"), h2h.alias("e")
    j = s.join(e, (F.col("e.team") == F.col("s.team")) & (F.col("e.opp") == F.col("s.opp"))
               & (F.col("e.fam") == F.col("s.fam")) & (F.col("e.gender") == F.col("s.gender"))
               & (F.col("e.team_type") == F.col("s.team_type")) & _earlier("e", "s"), "left")
    chk = (j.groupBy("s.match_id", "s.team")
            .agg(F.first("s.h2h_n").alias("h2h_n"), F.first("s.h2h_wins").alias("h2h_wins"),
                 F.count("e.win").alias("b_n"), F.coalesce(F.sum("e.win"), F.lit(0)).alias("b_wins")))
    n = chk.count()
    bad = chk.filter((F.col("h2h_n") != F.col("b_n")) | (F.col("h2h_wins") != F.col("b_wins"))).count()
    with_hist = chk.filter(F.col("b_n") > 0).count()
    print(f"  verify team: {n} sampled rows, {bad} mismatches vs brute force "
          f"[{'OK' if bad == 0 else 'LEAK OR BUG'}]  ({with_hist} have earlier meetings)", flush=True)
    return bad == 0


def main():
    spark = S.build_spark("opponent_features")
    spark.sparkContext.setLogLevel("WARN")

    deliveries = spark.read.parquet(S.DELIVERIES)
    info = match_info(spark.read.parquet(S.MATCHES)).cache()
    print(f"limited-overs matches: {info.count()}", flush=True)

    ok = True
    for label, pm_fn, path in [("batter", per_match_batting, BATTER_OUT),
                               ("bowler", per_match_bowling, BOWLER_OUT)]:
        pm = pm_fn(deliveries, info).cache()
        out = add_prior_ratio(pm).cache()
        # Cricsheet identifies players by name within a match, so two different players who
        # share a name on opposite sides collapse into one id and appear to play for both
        # teams. Such a key has no single honest opponent record, so it is left out (the
        # model imputes it) rather than silently duplicating rows when joined later.
        ambiguous = (out.groupBy("match_id", "player_id").count()
                     .filter(F.col("count") > 1).select("match_id", "player_id").cache())
        n_amb = ambiguous.count()
        out = out.join(ambiguous, ["match_id", "player_id"], "left_anti").cache()
        n = out.count()
        dup = out.groupBy("match_id", "player_id").count().filter(F.col("count") > 1).count()
        ok = ok and dup == 0
        with_hist = out.filter(F.col("opp_balls") > 0).count()
        have_ratio = out.filter(F.col("ratio").isNotNull()).count()
        print(f"\n[{label}] {n} (player, match) rows kept; ambiguous keys dropped: {n_amb}; "
              f"duplicate keys left: {dup} [{'OK' if dup == 0 else 'BAD'}]", flush=True)
        print(f"  with earlier balls against this opponent: {with_hist} ({with_hist / n:.1%}); "
              f"with a usable ratio: {have_ratio} ({have_ratio / n:.1%})", flush=True)
        ok = verify_ratio(pm, out, label) and ok
        (out.select("match_id", "player_id", "fam", "opp",
                    F.col("opp_balls").alias("vs_opp_balls"), F.col("ratio").alias("vs_opp_ratio"),
                    F.col("all_balls").alias("prior_balls"))
            .write.mode("overwrite").parquet(path))
        print(f"  wrote {path}", flush=True)

    h2h = team_head_to_head(info).cache()
    n = h2h.count()
    dup = h2h.groupBy("match_id", "team").count().filter(F.col("count") > 1).count()
    with_hist = h2h.filter(F.col("h2h_n") > 0).count()
    print(f"\n[team] {n} (team, match) rows; duplicate keys: {dup} [{'OK' if dup == 0 else 'BAD'}]; "
          f"with earlier meetings: {with_hist} ({with_hist / n:.1%})", flush=True)
    ok = verify_team(h2h) and ok
    (h2h.select("match_id", "team", "gender", "team_type", "h2h_n", "h2h_wins", "h2h_rate")
        .write.mode("overwrite").parquet(TEAM_OUT))
    print(f"  wrote {TEAM_OUT}", flush=True)

    print("\nALL LEAKAGE CHECKS PASSED" if ok else "\nA LEAKAGE CHECK FAILED - do not use these features")
    spark.stop()


if __name__ == "__main__":
    main()
