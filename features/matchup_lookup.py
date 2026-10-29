#!/usr/bin/env python3
"""Batter vs bowler head-to-head lookup, with honest reliability information.

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/features/matchup_lookup.py

This is a LOOKUP, not a model feature. The deliveries are too sparse for the pairings to
feed the win-probability model (about half of all pairs have faced six balls or fewer), and
it describes the whole dataset "as of now", so it must never be used to predict a match
inside the period it summarises (that would leak).

One row per (format family, batter, bowler), plus fam = 'ALL' rows pooled over both families:

  balls, runs, dismissals, dots, fours, sixes, n_matches, first/last match date
  strike_rate, dot_pct, balls_per_dismissal
  tier            thin (<12 balls, under 2 overs), limited (<30), moderate (<60), solid (60+)
  expected_sr     what the two players' OVERALL quality predicts for this pairing:
                  batter rate x bowler rate / league rate, each excluding this pairing
  shrunk_sr       the observed rate pulled toward expected_sr with K pseudo-balls, so a short
                  record cannot look extreme
  z_vs_expected   how many standard errors the observed rate is from expected_sr

A batter can look "unusual" against a bowler merely because that bowler is good or bad, so the
benchmark is the two-player expectation, not the batter's own average. Format families (T20 +
IT20, ODI + ODM) are never pooled for expected/shrunk/z because scoring rates differ by about
40 strike-rate points; the 'ALL' rows carry raw counts only.

Definitions
  ball faced   any delivery that is not a wide (a no-ball counts as faced)
  dot          a ball faced with no runs at all (runs_total == 0). dot_pct is a percentage (0-100),
               like dot_ball_pct in bowler_economy_by_phase. The two differ by design: that table
               counts LEGAL balls on which the BOWLER was charged no runs (a bye ball is a dot
               there), while this one is the batter's view over balls faced, so a bye breaks the
               dot and a no-ball is a ball faced that can never be a dot.
  four / six   runs_batter == 4 / 6 (an all-run four is counted as a four; rare)
  dismissal    the batter is out, credited to the bowler: bowled, caught, lbw, stumped,
               caught and bowled, hit wicket (run-outs are not credited to the bowler)
  super overs  excluded (innings 3+)

Reconciliation checks compare the table's totals with a direct count of the source rows.
"""
import sys

from pyspark.sql import Window
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402

OUT_PATH = f"{S.PROCESSED}/matchup_lookup"
BOWLER_WICKETS = ("bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket")
K = 30.0               # pseudo-balls pulling a short record toward the expectation
MIN_NORM_BALLS = 30    # balls a player needs (excluding this pairing) for a trusted norm
VERIFY_SAMPLE = 200


def family():
    """T20-type vs ODI-type. A function: Spark 3.1 cannot build a Column before a session."""
    return F.when(F.col("match_type").isin("T20", "IT20"), "T20").otherwise("ODI")


def tier():
    return (F.when(F.col("balls") < 12, "thin").when(F.col("balls") < 30, "limited")
             .when(F.col("balls") < 60, "moderate").otherwise("solid"))


def faced_balls(deliveries, matches):
    d = (deliveries.filter((F.col("format_group") == "limited_overs") & (F.col("innings_num") <= 2)
                           & F.col("batter_id").isNotNull() & F.col("bowler_id").isNotNull()
                           & (F.col("extra_wides") == 0))
         .join(matches.select("match_id", "match_date"), "match_id")
         .withColumn("fam", family())
         .withColumn("dot", (F.col("runs_total") == 0).cast("int"))
         .withColumn("four", (F.col("runs_batter") == 4).cast("int"))
         .withColumn("six", (F.col("runs_batter") == 6).cast("int"))
         .withColumn("out", (F.col("wicket_kind").isin(*BOWLER_WICKETS)
                             & (F.col("player_out_id") == F.col("batter_id"))).cast("int")))
    return d


def top_names(d, id_col, name_col):
    """The most frequent name recorded for each player id."""
    w = Window.partitionBy(id_col).orderBy(F.col("n").desc(), F.col(name_col))
    return (d.groupBy(id_col, name_col).agg(F.count("*").alias("n"))
             .withColumn("rk", F.row_number().over(w)).filter(F.col("rk") == 1)
             .select(id_col, F.col(name_col).alias(id_col.replace("_id", "_name"))))


def build(d):
    pair = (d.groupBy("fam", "batter_id", "bowler_id")
             .agg(F.count("*").alias("balls"), F.sum("runs_batter").alias("runs"),
                  F.sum("out").alias("dismissals"), F.sum("dot").alias("dots"),
                  F.sum("four").alias("fours"), F.sum("six").alias("sixes"),
                  F.countDistinct("match_id").alias("n_matches"),
                  F.min("match_date").alias("first_match_date"), F.max("match_date").alias("last_match_date")))

    bat = d.groupBy("fam", "batter_id").agg(F.count("*").alias("bat_balls"), F.sum("runs_batter").alias("bat_runs"),
                                            F.sum(F.col("runs_batter") * F.col("runs_batter")).alias("bat_sq"))
    bowl = d.groupBy("fam", "bowler_id").agg(F.count("*").alias("bowl_balls"), F.sum("runs_batter").alias("bowl_runs"))
    league = d.groupBy("fam").agg((F.sum("runs_batter") / F.count("*")).alias("league_rate"))

    t = (pair.join(bat, ["fam", "batter_id"]).join(bowl, ["fam", "bowler_id"]).join(league, "fam"))
    bat_ex = F.when(F.col("bat_balls") - F.col("balls") >= MIN_NORM_BALLS,
                    (F.col("bat_runs") - F.col("runs")) / (F.col("bat_balls") - F.col("balls")))
    bowl_ex = F.when(F.col("bowl_balls") - F.col("balls") >= MIN_NORM_BALLS,
                     (F.col("bowl_runs") - F.col("runs")) / (F.col("bowl_balls") - F.col("balls")))
    expected = F.when(F.col("league_rate") > 0, bat_ex * bowl_ex / F.col("league_rate"))
    bat_var = F.col("bat_sq") / F.col("bat_balls") - (F.col("bat_runs") / F.col("bat_balls")) ** 2
    t = (t.withColumn("exp_rate", expected)
          .withColumn("bat_var", bat_var)
          .withColumn("expected_sr", 100.0 * F.col("exp_rate"))
          .withColumn("shrunk_sr", 100.0 * (F.col("runs") + K * F.col("exp_rate")) / (F.col("balls") + K))
          .withColumn("z_vs_expected",
                      F.when(F.col("exp_rate").isNotNull() & (F.col("bat_var") > 0),
                             (F.col("runs") / F.col("balls") - F.col("exp_rate"))
                             / F.sqrt(F.col("bat_var") / F.col("balls")))))

    cols = ["fam", "batter_id", "bowler_id", "balls", "runs", "dismissals", "dots", "fours", "sixes",
            "n_matches", "first_match_date", "last_match_date", "expected_sr", "shrunk_sr", "z_vs_expected"]
    fam_rows = t.select(*cols)

    all_rows = (pair.groupBy("batter_id", "bowler_id")
                .agg(F.sum("balls").alias("balls"), F.sum("runs").alias("runs"),
                     F.sum("dismissals").alias("dismissals"), F.sum("dots").alias("dots"),
                     F.sum("fours").alias("fours"), F.sum("sixes").alias("sixes"),
                     F.sum("n_matches").alias("n_matches"),    # a match belongs to exactly one family
                     F.min("first_match_date").alias("first_match_date"),
                     F.max("last_match_date").alias("last_match_date"))
                .withColumn("fam", F.lit("ALL"))
                .withColumn("expected_sr", F.lit(None).cast("double"))
                .withColumn("shrunk_sr", F.lit(None).cast("double"))
                .withColumn("z_vs_expected", F.lit(None).cast("double"))
                .select(*cols))
    return fam_rows.unionByName(all_rows)


def finish(rows, d):
    bn = top_names(d, "batter_id", "batter")
    wn = top_names(d, "bowler_id", "bowler")
    return (rows.join(bn, "batter_id", "left").join(wn, "bowler_id", "left")
            .withColumn("strike_rate", 100.0 * F.col("runs") / F.col("balls"))
            .withColumn("dot_pct", 100.0 * F.col("dots") / F.col("balls"))
            .withColumn("balls_per_dismissal",
                        F.when(F.col("dismissals") > 0, F.col("balls") / F.col("dismissals")))
            .withColumn("tier", tier())
            .select("fam", "batter_id", "bowler_id", "batter_name", "bowler_name", "balls", "runs",
                    "dismissals", "dots", "fours", "sixes", "n_matches", "first_match_date", "last_match_date",
                    "strike_rate", "dot_pct", "balls_per_dismissal", "expected_sr", "shrunk_sr",
                    "z_vs_expected", "tier"))


def main():
    spark = S.build_spark("matchup_lookup")
    spark.sparkContext.setLogLevel("WARN")

    d_all = faced_balls(spark.read.parquet(S.DELIVERIES), spark.read.parquet(S.MATCHES))
    n_all = d_all.count()
    # Same-name players on opposite sides share one Cricsheet id, so an id can appear to face
    # itself. Such a pairing has no honest record and is left out.
    d = d_all.filter(F.col("batter_id") != F.col("bowler_id")).cache()
    n = d.count()
    print(f"balls faced (limited overs, innings 1-2): {n_all}; dropped as self-pairings "
          f"(same-name id collision): {n_all - n}; kept: {n}", flush=True)

    out = finish(build(d), d).cache()
    n_rows = out.count()
    print(f"matchup_lookup rows: {n_rows}", flush=True)

    # ---------------- reconciliation against a direct count of the source ----------------
    src = d.agg(F.count("*").alias("balls"), F.sum("runs_batter").alias("runs"), F.sum("out").alias("out"),
                F.sum("dot").alias("dots"), F.sum("four").alias("fours"), F.sum("six").alias("sixes")).first()
    ok = True
    print("\n== reconciliation: table totals vs a direct count of the source rows ==", flush=True)
    for scope, flt in [("T20 + ODI rows", F.col("fam") != "ALL"), ("ALL rows", F.col("fam") == "ALL")]:
        t = out.filter(flt).agg(F.sum("balls").alias("balls"), F.sum("runs").alias("runs"),
                                F.sum("dismissals").alias("out"), F.sum("dots").alias("dots"),
                                F.sum("fours").alias("fours"), F.sum("sixes").alias("sixes")).first()
        same = all(int(t[k]) == int(src[k]) for k in ("balls", "runs", "out", "dots", "fours", "sixes"))
        ok = ok and same
        print(f"  {scope:<16} balls {t['balls']} runs {t['runs']} dismissals {t['out']} dots {t['dots']} "
              f"[{'OK' if same else 'MISMATCH'}]", flush=True)
    print(f"  source           balls {src['balls']} runs {src['runs']} dismissals {src['out']} dots {src['dots']}",
          flush=True)
    dup = out.groupBy("fam", "batter_id", "bowler_id").count().filter(F.col("count") > 1).count()
    ok = ok and dup == 0
    print(f"  duplicate keys: {dup} [{'OK' if dup == 0 else 'BAD'}]", flush=True)

    # independent recomputation for a random sample (semi-join on the sampled keys, then a different
    # aggregation expression)
    sample = (out.filter(F.col("fam") != "ALL").orderBy(F.rand(11)).limit(VERIFY_SAMPLE)
              .select("fam", "batter_id", "bowler_id", "balls", "runs", "dismissals").cache())
    chk = (d.join(sample.select("fam", "batter_id", "bowler_id"), ["fam", "batter_id", "bowler_id"], "left_semi")
            .groupBy("fam", "batter_id", "bowler_id")
            .agg(F.count(F.lit(1)).alias("b2"), F.sum(F.col("runs_batter")).alias("r2"),
                 F.sum(F.when(F.col("wicket_kind").isin(*BOWLER_WICKETS)
                              & (F.col("player_out_id") == F.col("batter_id")), 1).otherwise(0)).alias("o2")))
    bad = (sample.join(chk, ["fam", "batter_id", "bowler_id"])
           .filter((F.col("balls") != F.col("b2")) | (F.col("runs") != F.col("r2"))
                   | (F.col("dismissals") != F.col("o2"))).count())
    ok = ok and bad == 0
    print(f"  sampled recomputation: {sample.count()} pairs, {bad} mismatches [{'OK' if bad == 0 else 'BAD'}]",
          flush=True)

    # ---------------- how thin is the data? ----------------
    print("\n== sample size: how much of the table is trustworthy? ==", flush=True)
    for fam in ("T20", "ODI"):
        f = out.filter(F.col("fam") == fam)
        tot_pairs, tot_balls = f.count(), f.agg(F.sum("balls")).first()[0]
        print(f"\n[{fam}] {tot_pairs} pairings, {tot_balls} balls", flush=True)
        print(f"  {'tier':<10}{'pairings':>10}{'share':>8}{'share of balls':>16}", flush=True)
        by = {r["tier"]: r for r in f.groupBy("tier").agg(F.count("*").alias("p"), F.sum("balls").alias("b")).collect()}
        for tname in ("thin", "limited", "moderate", "solid"):
            r = by.get(tname)
            if r:
                print(f"  {tname:<10}{r['p']:>10}{r['p'] / tot_pairs:>8.1%}{r['b'] / tot_balls:>16.1%}", flush=True)
        solid = f.filter(F.col("tier").isin("moderate", "solid"))
        n_s = solid.count()
        n_out = solid.filter(F.col("dismissals") > 0).count()
        print(f"  pairings with 30+ balls: {n_s}; of those, with at least one dismissal: {n_out} ({n_out / n_s:.1%})",
              flush=True)

    # ---------------- is there ANY matchup-specific signal beyond the two players' quality? ----------------
    print("\n== is there matchup-specific signal beyond the two players' overall quality? ==", flush=True)
    print("  z = (observed - expected) / standard error, pairings with 30+ balls. If matchups were pure noise "
          "around the", flush=True)
    print("  two-player expectation, z would have variance 1.0, 4.6% beyond +/-2 and 0.3% beyond +/-3.", flush=True)
    for fam in ("T20", "ODI"):
        z = out.filter((F.col("fam") == fam) & (F.col("balls") >= 30) & F.col("z_vs_expected").isNotNull())
        r = z.agg(F.count("*").alias("n"), F.mean("z_vs_expected").alias("mean"),
                  F.variance("z_vs_expected").alias("var"),
                  F.avg((F.abs(F.col("z_vs_expected")) > 2).cast("double")).alias("gt2"),
                  F.avg((F.abs(F.col("z_vs_expected")) > 3).cast("double")).alias("gt3")).first()
        print(f"  [{fam}] pairings {r['n']}: mean z {r['mean']:+.3f}, variance {r['var']:.3f}, "
              f"beyond +/-2: {r['gt2']:.1%}, beyond +/-3: {r['gt3']:.1%}", flush=True)

    print("\n== most-faced T20 matchups ==", flush=True)
    (out.filter(F.col("fam") == "T20").orderBy(F.col("balls").desc()).limit(8)
        .select("batter_name", "bowler_name", "balls", "runs", "dismissals",
                F.round("strike_rate", 1).alias("sr"), F.round("expected_sr", 1).alias("expected_sr"),
                F.round("z_vs_expected", 2).alias("z"), "tier").show(truncate=False))

    out.write.mode("overwrite").parquet(OUT_PATH)
    print(f"wrote {OUT_PATH}", flush=True)
    print("ALL CHECKS PASSED" if ok else "A CHECK FAILED - do not use this table", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
