#!/usr/bin/env python3
"""Venue par baselines, computed without leakage.

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/features/venue_features.py

One row per match answering "what did this ground look like going INTO the match":

  venue_par_rpo         typical first-innings run rate at this ground, from EARLIER matches
                        only, shrunk toward the format-wide norm
  venue_chase_win_rate  share of earlier decisive matches here won by the side batting second,
                        shrunk toward 0.5
  venue_prior_matches   how many earlier first innings back the figures (reliability)

Design choices
  * A ground is identified by venue AND city. "County Ground" alone is six different grounds
    (261 matches) and other names also span cities; pooling them would invent a fake venue.
    Over-splitting a ground that is spelled two ways only costs history, which is safe.
  * History is split by format family (T20 + IT20, ODI + ODM) and by gender: first-innings run
    rates differ a great deal between them. The format-wide norm used for shrinkage is split
    the same way, plus team_type, and is also built from earlier matches only.
  * Shrinkage: the ground's record is blended with the format norm using K_INNINGS pseudo
    innings, so a ground with one earlier match barely moves from the norm and a ground with
    none sits exactly on it. Below MIN_FORMAT_BALLS of earlier data the norm itself is
    untrusted and the value is null (the model imputes it).
  * First innings shorter than MIN_BALLS legal balls (abandoned) add nothing to the history.
  * Only the FIRST innings feeds the par: a second-innings total depends on the target.
  * The current match never contributes: every window ends at the row before it, ordered by
    (match_date, match_id).

The output is verified against an independent brute-force recomputation (a self-join on
strictly earlier matches, no window functions) on a random sample; the script says so
loudly if that fails.
"""
import sys

from pyspark.sql import Window
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402

VENUE_OUT = f"{S.PROCESSED}/features/venue"
K_INNINGS = 4            # pseudo-innings pulling a thin venue record toward the format norm
BALLS_PER_INNINGS = {"T20": 120, "ODI": 300}
MIN_BALLS = 30           # a first innings shorter than this (abandoned) is ignored
MIN_FORMAT_BALLS = 1200  # earlier balls needed before the format-wide norm is trusted
CHASE_PRIOR = 2.0        # pseudo-wins and pseudo-losses for the chase win rate
VERIFY_SAMPLE = 300
PRIOR = Window.unboundedPreceding


def family():
    """T20-type vs ODI-type. A function: Spark 3.1 cannot build a Column before a session."""
    return F.when(F.col("match_type").isin("T20", "IT20"), "T20").otherwise("ODI")


def per_match(deliveries, matches):
    """One row per limited-overs match with its first-innings total and chase outcome."""
    first = (deliveries.filter((F.col("format_group") == "limited_overs") & (F.col("innings_num") == 1))
             .groupBy("match_id")
             .agg(F.sum("runs_total").alias("runs1_all"),
                  F.sum(F.col("is_legal_ball").cast("int")).alias("balls1_all"),
                  F.first("batting_team").alias("bat_first")))
    m = (matches.filter(F.col("format_group") == "limited_overs")
         .withColumn("fam", family())
         .withColumn("venue_key", F.concat_ws("|", F.col("venue"), F.coalesce(F.col("city"), F.lit("?"))))
         .select("match_id", "match_date", "fam", "gender", "team_type", "venue_key",
                 "team1", "team2", "winner"))
    pm = m.join(first, "match_id", "left")
    chaser = F.when(F.col("bat_first") == F.col("team1"), F.col("team2")).otherwise(F.col("team1"))
    usable = F.col("balls1_all") >= MIN_BALLS
    return (pm
            # A first innings that is too short (abandoned) must add nothing: null, not zero.
            .withColumn("runs1", F.when(usable, F.col("runs1_all")))
            .withColumn("balls1", F.when(usable, F.col("balls1_all")))
            .withColumn("chase_win", F.when(usable & (F.col("winner") == chaser), 1)
                                      .when(usable & (F.col("winner") == F.col("bat_first")), 0))
            .select("match_id", "match_date", "fam", "gender", "team_type", "venue_key",
                    "runs1", "balls1", "chase_win"))


def add_history(pm):
    order = ("match_date", "match_id")
    w_v = Window.partitionBy("venue_key", "fam", "gender").orderBy(*order).rowsBetween(PRIOR, -1)
    w_f = Window.partitionBy("fam", "gender", "team_type").orderBy(*order).rowsBetween(PRIOR, -1)
    x = (pm.withColumn("v_runs", F.coalesce(F.sum("runs1").over(w_v), F.lit(0)))
           .withColumn("v_balls", F.coalesce(F.sum("balls1").over(w_v), F.lit(0)))
           .withColumn("v_n", F.count("runs1").over(w_v))               # count skips nulls
           .withColumn("v_cw_n", F.count("chase_win").over(w_v))
           .withColumn("v_cw", F.coalesce(F.sum("chase_win").over(w_v), F.lit(0)))
           .withColumn("f_runs", F.coalesce(F.sum("runs1").over(w_f), F.lit(0)))
           .withColumn("f_balls", F.coalesce(F.sum("balls1").over(w_f), F.lit(0))))
    norm = F.when(F.col("f_balls") >= MIN_FORMAT_BALLS, F.col("f_runs") / F.col("f_balls"))
    k_balls = F.when(F.col("fam") == "T20", K_INNINGS * BALLS_PER_INNINGS["T20"]) \
               .otherwise(K_INNINGS * BALLS_PER_INNINGS["ODI"])
    shrunk = (F.col("v_runs") + k_balls * norm) / (F.col("v_balls") + k_balls)
    return (x.withColumn("venue_par_rpo", 6.0 * shrunk)
             .withColumn("venue_chase_win_rate",
                         (F.col("v_cw") + CHASE_PRIOR) / (F.col("v_cw_n") + 2 * CHASE_PRIOR))
             .withColumn("venue_prior_matches", F.col("v_n")))


def _earlier(a, b):
    return ((F.col(f"{a}.match_date") < F.col(f"{b}.match_date"))
            | ((F.col(f"{a}.match_date") == F.col(f"{b}.match_date"))
               & (F.col(f"{a}.match_id") < F.col(f"{b}.match_id"))))


def verify(pm, out):
    """Recompute every prior total with plain self-joins (no windows) on a random sample."""
    sample = (out.orderBy(F.rand(7)).limit(VERIFY_SAMPLE)
              .select("match_id", "match_date", "fam", "gender", "team_type", "venue_key",
                      "v_runs", "v_balls", "v_n", "v_cw_n", "v_cw", "f_runs", "f_balls").cache())
    s, e = sample.alias("s"), pm.alias("e")
    jv = s.join(e, (F.col("e.venue_key") == F.col("s.venue_key")) & (F.col("e.fam") == F.col("s.fam"))
                & (F.col("e.gender") == F.col("s.gender")) & _earlier("e", "s"), "left")
    cv = (jv.groupBy("s.match_id")
            .agg(F.first("s.v_runs").alias("v_runs"), F.first("s.v_balls").alias("v_balls"),
                 F.first("s.v_n").alias("v_n"), F.first("s.v_cw_n").alias("v_cw_n"),
                 F.first("s.v_cw").alias("v_cw"),
                 F.coalesce(F.sum("e.runs1"), F.lit(0)).alias("b_runs"),
                 F.coalesce(F.sum("e.balls1"), F.lit(0)).alias("b_balls"),
                 F.count("e.runs1").alias("b_n"), F.count("e.chase_win").alias("b_cw_n"),
                 F.coalesce(F.sum("e.chase_win"), F.lit(0)).alias("b_cw")))
    jf = s.join(e, (F.col("e.fam") == F.col("s.fam")) & (F.col("e.gender") == F.col("s.gender"))
                & (F.col("e.team_type") == F.col("s.team_type")) & _earlier("e", "s"), "left")
    cf = (jf.groupBy("s.match_id")
            .agg(F.first("s.f_runs").alias("f_runs"), F.first("s.f_balls").alias("f_balls"),
                 F.coalesce(F.sum("e.runs1"), F.lit(0)).alias("bf_runs"),
                 F.coalesce(F.sum("e.balls1"), F.lit(0)).alias("bf_balls")))
    bad_v = cv.filter((F.col("v_runs") != F.col("b_runs")) | (F.col("v_balls") != F.col("b_balls"))
                      | (F.col("v_n") != F.col("b_n")) | (F.col("v_cw_n") != F.col("b_cw_n"))
                      | (F.col("v_cw") != F.col("b_cw"))).count()
    bad_f = cf.filter((F.col("f_runs") != F.col("bf_runs")) | (F.col("f_balls") != F.col("bf_balls"))).count()
    n = cv.count()
    with_hist = cv.filter(F.col("b_n") > 0).count()
    ok = bad_v == 0 and bad_f == 0
    print(f"  verify: {n} sampled matches; venue-history mismatches {bad_v}, format-norm mismatches "
          f"{bad_f} [{'OK' if ok else 'LEAK OR BUG'}]  ({with_hist} of them have earlier matches at the "
          f"venue, so the check is not vacuous)", flush=True)
    return ok


def main():
    spark = S.build_spark("venue_features")
    spark.sparkContext.setLogLevel("WARN")

    pm = per_match(spark.read.parquet(S.DELIVERIES), spark.read.parquet(S.MATCHES)).cache()
    n_matches = pm.count()
    dup = pm.groupBy("match_id").count().filter(F.col("count") > 1).count()
    print(f"limited-overs matches: {n_matches}; duplicate match_id: {dup} [{'OK' if dup == 0 else 'BAD'}]",
          flush=True)
    keys = pm.select("venue_key").distinct().count()
    print(f"distinct grounds (venue + city): {keys}", flush=True)

    out = add_history(pm).cache()
    print(f"\ncoverage across matches (how many earlier first innings at the same ground):", flush=True)
    for label, cond in [(">= 1 earlier", F.col("venue_prior_matches") >= 1),
                        (">= 5 earlier", F.col("venue_prior_matches") >= 5),
                        (">= 20 earlier", F.col("venue_prior_matches") >= 20),
                        ("format norm trusted", F.col("venue_par_rpo").isNotNull())]:
        c = out.filter(cond).count()
        print(f"  {label:<22}{c:>7}  ({c / n_matches:.1%})", flush=True)
    stats = out.filter(F.col("venue_par_rpo").isNotNull()).select(
        F.min("venue_par_rpo"), F.max("venue_par_rpo"), F.mean("venue_par_rpo"),
        F.min("venue_chase_win_rate"), F.max("venue_chase_win_rate")).first()
    print(f"  venue_par_rpo range {stats[0]:.2f} .. {stats[1]:.2f} (mean {stats[2]:.2f}); "
          f"chase win rate range {stats[3]:.3f} .. {stats[4]:.3f}", flush=True)

    ok = verify(pm, out)
    (out.select("match_id", "venue_par_rpo", "venue_chase_win_rate", "venue_prior_matches")
        .write.mode("overwrite").parquet(VENUE_OUT))
    print(f"\nwrote {VENUE_OUT}", flush=True)
    print("LEAKAGE CHECK PASSED" if ok else "LEAKAGE CHECK FAILED - do not use these features", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
