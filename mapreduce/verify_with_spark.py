#!/usr/bin/env python3
"""Independent check of the MapReduce bowler-economy job, recomputed in Spark.

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 4g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/mapreduce/verify_with_spark.py

The Hadoop Streaming job and this script share only the rules, not the code: this one reads
the deliveries Parquet (not the TSV the mapper reads), classifies phases with Spark
expressions (not phase_utils.get_phase), and aggregates with a Spark groupBy (not a
streaming reducer). Every (bowler, match_type, phase) row must agree on runs, legal balls,
wickets and dot balls, and the derived columns must agree to their printed precision.

The rules being checked (same as mapper.py):
  * rows with no bowler id or a negative over are skipped
  * runs charged to the bowler = runs_total - byes - legbyes (wides and no-balls ARE charged)
  * a legal ball is one with no wide and no no-ball
  * a dot ball is a legal ball on which the bowler is charged no runs
  * phase: T20/IT20 overs 0-5 powerplay, 6-14 middle, 15+ death; other formats 0-9, 10-39, 40+
"""
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402

OUT_SCHEMA = StructType([
    StructField("bowler_id", StringType()), StructField("match_type", StringType()),
    StructField("phase", StringType()), StructField("runs", IntegerType()),
    StructField("balls", IntegerType()), StructField("overs", DoubleType()),
    StructField("wickets", IntegerType()), StructField("economy", DoubleType()),
    StructField("strike_rate", DoubleType()), StructField("dot_balls", IntegerType()),
    StructField("dot_ball_pct", DoubleType())])


def recompute(spark):
    d = (spark.read.parquet(S.DELIVERIES)
         .filter(F.col("format_group") == "limited_overs")
         .filter(F.col("bowler_id").isNotNull() & (F.col("bowler_id") != "")
                 & F.col("over").isNotNull() & (F.col("over") >= 0)))
    z = lambda c: F.coalesce(F.col(c), F.lit(0))                       # noqa: E731
    charged = z("runs_total") - z("extra_byes") - z("extra_legbyes")
    legal = F.when((z("extra_wides") > 0) | (z("extra_noballs") > 0), 0).otherwise(1)
    t20 = F.col("match_type").isin("T20", "IT20")
    phase = (F.when(t20 & (F.col("over") <= 5), "powerplay").when(t20 & (F.col("over") <= 14), "middle")
              .when(t20, "death").when(F.col("over") <= 9, "powerplay").when(F.col("over") <= 39, "middle")
              .otherwise("death"))
    return (d.withColumn("phase", phase).withColumn("charged", charged).withColumn("legal", legal)
             .groupBy("bowler_id", "match_type", "phase")
             .agg(F.sum("charged").alias("runs"), F.sum("legal").alias("balls"),
                  F.sum(F.when(F.col("is_wicket"), 1).otherwise(0)).alias("wickets"),
                  F.sum(F.when((F.col("legal") == 1) & (F.col("charged") == 0), 1).otherwise(0))
                   .alias("dot_balls")))


def main():
    spark = S.build_spark("verify_economy_job")
    spark.sparkContext.setLogLevel("WARN")

    mr = spark.read.option("sep", "\t").schema(OUT_SCHEMA).csv(S.MAPREDUCE_OUTPUT).cache()
    sp = recompute(spark).cache()
    n_mr, n_sp = mr.count(), sp.count()
    print(f"rows: MapReduce {n_mr}, Spark recomputation {n_sp} [{'OK' if n_mr == n_sp else 'MISMATCH'}]")

    key = ["bowler_id", "match_type", "phase"]
    j = mr.alias("m").join(sp.alias("s"), key, "full_outer")
    only_mr = j.filter(F.col("s.runs").isNull()).count()
    only_sp = j.filter(F.col("m.runs").isNull()).count()
    print(f"keys only in MapReduce: {only_mr}; only in Spark: {only_sp} "
          f"[{'OK' if only_mr == only_sp == 0 else 'MISMATCH'}]")

    both = j.filter(F.col("m.runs").isNotNull() & F.col("s.runs").isNotNull())
    bad = {}
    for label, cond in [
        ("runs", F.col("m.runs") != F.col("s.runs")),
        ("legal balls", F.col("m.balls") != F.col("s.balls")),
        ("wickets", F.col("m.wickets") != F.col("s.wickets")),
        ("dot balls", F.col("m.dot_balls") != F.col("s.dot_balls")),
        ("overs (balls/6, 3dp)", F.abs(F.col("m.overs") - F.col("s.balls") / 6.0) > 0.0006),
        ("economy (4dp)", F.col("m.economy").isNotNull()
         & (F.abs(F.col("m.economy") - F.col("s.runs") * 6.0 / F.col("s.balls")) > 0.00006)),
        ("dot_ball_pct (2dp)", F.abs(F.col("m.dot_ball_pct") - 100.0 * F.col("s.dot_balls") / F.col("s.balls")) > 0.006)]:
        bad[label] = both.filter(cond).count()
    for label, n in bad.items():
        print(f"  {label:<24} mismatches: {n} [{'OK' if n == 0 else 'MISMATCH'}]")

    tot = mr.agg(F.sum("balls").alias("b"), F.sum("dot_balls").alias("d")).first()
    tot2 = sp.agg(F.sum("balls").alias("b"), F.sum("dot_balls").alias("d")).first()
    print(f"\ntotals: legal balls {tot['b']} vs {tot2['b']}; dot balls {tot['d']} vs {tot2['d']} "
          f"-> league-wide dot-ball rate {100.0 * tot['d'] / tot['b']:.2f}%")

    print("\nleague-wide dot-ball percentage by format and phase (from the MapReduce output):")
    print(f"  {'format':<8}{'phase':<11}{'legal balls':>14}{'dot %':>9}")
    rows = (mr.groupBy("match_type", "phase").agg(F.sum("balls").alias("b"), F.sum("dot_balls").alias("d"))
              .orderBy("match_type", "phase").collect())
    order = {"powerplay": 0, "middle": 1, "death": 2}
    for r in sorted(rows, key=lambda r: (r["match_type"], order[r["phase"]])):
        print(f"  {r['match_type']:<8}{r['phase']:<11}{r['b']:>14}{100.0 * r['d'] / r['b']:>9.2f}")

    ok = n_mr == n_sp and only_mr == only_sp == 0 and all(v == 0 for v in bad.values())
    print("\nVERIFIED: MapReduce output agrees with the independent Spark recomputation" if ok
          else "\nNOT VERIFIED: the MapReduce output disagrees with the Spark recomputation")
    spark.stop()


if __name__ == "__main__":
    main()
