#!/usr/bin/env python3
"""Flatten Cricsheet match JSON into Parquet fact/dimension tables.

    docker exec spark-master spark-submit \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/etl/flatten_matches.py

Outputs (HDFS):
    /data/processed/matches           one row per match, partitioned by format_group
    /data/processed/player_registry   (match_id, player_name, person_id)
    /data/processed/deliveries        one row per ball, partitioned by format_group/match_type
    /data/mapreduce_input/deliveries_tsv   limited-overs subset, TSV for Hadoop Streaming
"""
import sys

from pyspark.sql import Window
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402


def read_raw(spark, path, format_group):
    """Read a bucket of match JSON. match_id comes from the filename -- it is
    not present anywhere inside the document."""
    df = (spark.read
          .schema(S.CRICSHEET_SCHEMA)
          .option("multiLine", True)
          .json(path))
    # Keep the stem verbatim: stripping the `wi_` prefix of older West Indies
    # domestic files could collide with a numerically identical match id.
    match_id = F.regexp_extract(F.input_file_name(), r"([^/]+)\.json$", 1)
    return df.withColumn("match_id", match_id).withColumn("format_group", F.lit(format_group))


def build_matches(raw):
    i = F.col("info")
    outcome = i["outcome"]
    return raw.select(
        "match_id",
        "format_group",
        i["match_type"].alias("match_type"),
        i["match_type_number"].alias("match_type_number"),
        i["gender"].alias("gender"),
        i["team_type"].alias("team_type"),
        i["season"].alias("season"),
        i["dates"].alias("dates"),
        F.to_date(i["dates"].getItem(0)).alias("match_date"),
        i["teams"].getItem(0).alias("team1"),
        i["teams"].getItem(1).alias("team2"),
        i["venue"].alias("venue"),
        i["city"].alias("city"),
        i["event"]["name"].alias("event_name"),
        i["event"]["match_number"].alias("event_match_number"),
        i["event"]["stage"].alias("event_stage"),
        i["balls_per_over"].alias("balls_per_over"),
        i["overs"].alias("overs_cap"),
        i["toss"]["winner"].alias("toss_winner"),
        i["toss"]["decision"].alias("toss_decision"),
        outcome["winner"].alias("winner"),
        # outcome comes in four shapes: {by,winner}, {result}, {by,method,winner},
        # {eliminator,result}. Normalise to one result_type column.
        F.when(outcome["winner"].isNotNull(), F.lit("win"))
         .when(outcome["eliminator"].isNotNull(), F.lit("eliminator"))
         .otherwise(F.coalesce(outcome["result"], F.lit("unknown"))).alias("result_type"),
        outcome["eliminator"].alias("eliminator"),
        outcome["method"].alias("method"),
        F.when(outcome["by"]["runs"].isNotNull(), F.lit("runs"))
         .when(outcome["by"]["wickets"].isNotNull(), F.lit("wickets"))
         .when(outcome["by"]["innings"].isNotNull(), F.lit("innings")).alias("win_by_type"),
        F.coalesce(outcome["by"]["runs"], outcome["by"]["wickets"]).alias("win_by_value"),
        i["player_of_match"].alias("player_of_match"),
    )


def build_registry(raw):
    return (raw
            .select("match_id", F.explode(F.col("info")["registry"]["people"])
                    .alias("player_name", "person_id"))
            .dropDuplicates(["match_id", "player_name"]))


def build_deliveries(raw):
    """Explode innings -> overs -> deliveries into one row per ball."""
    innings = raw.select(
        "match_id", "format_group",
        F.col("info")["match_type"].alias("match_type"),
        F.col("info")["overs"].alias("overs_cap"),
        F.posexplode("innings").alias("innings_idx", "inn"),
    ).withColumn("innings_num", F.col("innings_idx") + 1)

    overs = innings.select(
        "match_id", "format_group", "match_type", "overs_cap", "innings_num",
        F.col("inn")["team"].alias("batting_team"),
        F.col("inn")["target"]["runs"].alias("target_runs"),
        F.col("inn")["target"]["overs"].alias("target_overs"),
        F.explode("inn.overs").alias("ov"),
    )

    # posexplode on deliveries gives a deterministic within-over ordering.
    # actual_delivery ("12.3") repeats across re-bowled balls, so it is kept
    # for audit but never used for ordering.
    balls = overs.select(
        "match_id", "format_group", "match_type", "overs_cap", "innings_num",
        "batting_team", "target_runs", "target_overs",
        F.col("ov")["over"].alias("over"),
        F.posexplode("ov.deliveries").alias("delivery_idx", "d"),
    )

    d = F.col("d")
    wicket = d["wickets"].getItem(0)
    flat = balls.select(
        "match_id", "format_group", "match_type", "overs_cap", "innings_num",
        "batting_team", "target_runs", "target_overs", "over", "delivery_idx",
        d["actual_delivery"].alias("actual_delivery"),
        d["batter"].alias("batter"),
        d["bowler"].alias("bowler"),
        d["non_striker"].alias("non_striker"),
        d["runs"]["batter"].alias("runs_batter"),
        d["runs"]["extras"].alias("runs_extras"),
        d["runs"]["total"].alias("runs_total"),
        F.coalesce(d["extras"]["wides"], F.lit(0)).alias("extra_wides"),
        F.coalesce(d["extras"]["noballs"], F.lit(0)).alias("extra_noballs"),
        F.coalesce(d["extras"]["byes"], F.lit(0)).alias("extra_byes"),
        F.coalesce(d["extras"]["legbyes"], F.lit(0)).alias("extra_legbyes"),
        F.coalesce(d["extras"]["penalty"], F.lit(0)).alias("extra_penalty"),
        F.when(F.size(F.coalesce(d["wickets"], F.array())) > 0, F.lit(True))
         .otherwise(F.lit(False)).alias("is_wicket"),
        F.size(F.coalesce(d["wickets"], F.array())).alias("num_wickets"),
        wicket["player_out"].alias("player_out"),
        wicket["kind"].alias("wicket_kind"),
        F.size(F.coalesce(wicket["fielders"], F.array())).alias("num_fielders"),
        d["review"]["decision"].alias("review_decision"),
    )

    # Wides and no-balls are not legal deliveries: they do not advance the ball
    # count, which every downstream overs-remaining calculation depends on.
    flat = flat.withColumn(
        "is_legal_ball",
        (F.col("extra_wides") == 0) & (F.col("extra_noballs") == 0),
    )

    ball_order = (Window.partitionBy("match_id", "innings_num")
                  .orderBy("over", "delivery_idx"))
    running = ball_order.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    return (flat
            .withColumn("ball_seq", F.row_number().over(ball_order))
            .withColumn("balls_completed_in_innings",
                        F.sum(F.col("is_legal_ball").cast("int")).over(running)))


def attach_player_ids(deliveries, registry):
    """Resolve names to Cricsheet person ids.

    The registry is per match, so the join key must include match_id -- ids are
    stable across matches, but the name -> id mapping only exists inside each
    match document.
    """
    for name_col, id_col in [("batter", "batter_id"),
                             ("bowler", "bowler_id"),
                             ("non_striker", "non_striker_id"),
                             ("player_out", "player_out_id")]:
        r = registry.select(
            F.col("match_id").alias("_mid"),
            F.col("player_name").alias("_name"),
            F.col("person_id").alias(id_col),
        )
        deliveries = (deliveries
                      .join(r, (deliveries["match_id"] == r["_mid"]) &
                            (deliveries[name_col] == r["_name"]), "left")
                      .drop("_mid", "_name"))
    return deliveries


def main():
    spark = S.build_spark("flatten_matches")
    spark.sparkContext.setLogLevel("WARN")

    limited = read_raw(spark, S.RAW_LIMITED_OVERS, "limited_overs")
    tests = read_raw(spark, S.RAW_TESTS, "test")
    raw = limited.unionByName(tests)

    matches = build_matches(raw)
    matches.write.mode("overwrite").partitionBy("format_group").parquet(S.MATCHES)
    print(f"wrote matches -> {S.MATCHES}")

    registry = build_registry(raw)
    registry.write.mode("overwrite").parquet(S.PLAYER_REGISTRY)
    print(f"wrote player_registry -> {S.PLAYER_REGISTRY}")

    registry = spark.read.parquet(S.PLAYER_REGISTRY)
    deliveries = attach_player_ids(build_deliveries(raw), registry)
    (deliveries.write.mode("overwrite")
     .partitionBy("format_group", "match_type").parquet(S.DELIVERIES))
    print(f"wrote deliveries -> {S.DELIVERIES}")

    # Hadoop Streaming reads text, not Parquet -- export the limited-overs
    # subset with exactly the columns the mapper needs, tab separated because
    # player names contain commas.
    (spark.read.parquet(S.DELIVERIES)
     .filter(F.col("format_group") == "limited_overs")
     .select(*[F.col(c) for c in S.MAPREDUCE_COLUMNS])
     .write.mode("overwrite").option("sep", "\t").option("header", False)
     .csv(S.MAPREDUCE_INPUT))
    print(f"wrote mapreduce input -> {S.MAPREDUCE_INPUT}")

    spark.stop()


if __name__ == "__main__":
    main()
