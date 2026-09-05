"""Shared schema, paths and helpers for the Spark stages of the pipeline."""
from pyspark.sql import SparkSession
from pyspark.sql.types import (ArrayType, BooleanType, IntegerType, MapType,
                               StringType, StructField, StructType)

# --- HDFS layout -----------------------------------------------------------
RAW_LIMITED_OVERS = "hdfs:///data/raw/limited_overs"
RAW_TESTS = "hdfs:///data/raw/tests"

PROCESSED = "hdfs:///data/processed"
MATCHES = f"{PROCESSED}/matches"
DELIVERIES = f"{PROCESSED}/deliveries"
PLAYER_REGISTRY = f"{PROCESSED}/player_registry"
MATCH_STATE = f"{PROCESSED}/match_state"
PLAYER_FORM = f"{PROCESSED}/player_form"
TRAINING_TABLE = f"{PROCESSED}/features/win_probability_training"
WIN_PROB_BY_BALL = f"{PROCESSED}/win_probability_by_ball"

MAPREDUCE_INPUT = "hdfs:///data/mapreduce_input/deliveries_tsv"
MAPREDUCE_OUTPUT = "hdfs:///data/mapreduce_output/bowler_economy_by_phase"
MODEL_PATH = "hdfs:///data/models/win_probability_gbt"

LIMITED_OVERS_TYPES = ("ODI", "ODM", "T20", "IT20")
T20_TYPES = ("T20", "IT20")

# Column order for the MapReduce TSV export. The streaming mapper parses
# positionally, so this list is the contract between the two stages.
MAPREDUCE_COLUMNS = [
    "match_id", "match_type", "overs_cap", "innings_num", "over",
    "bowler", "bowler_id", "runs_total", "extra_wides", "extra_noballs",
    "extra_byes", "extra_legbyes", "is_wicket",
]


def build_spark(app_name):
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
            .getOrCreate())


# --- Cricsheet JSON schema (format 1.2.0) ----------------------------------
# Declared explicitly rather than inferred: inference would cost a full extra
# pass over ~22k files, and an explicit schema turns absent optional keys
# (extras, wickets, review, target) into nulls instead of schema drift.

_FIELDER = StructType([
    StructField("name", StringType()),
    StructField("substitute", BooleanType()),
])

_WICKET = StructType([
    StructField("player_out", StringType()),
    StructField("kind", StringType()),
    StructField("fielders", ArrayType(_FIELDER)),
])

_REVIEW = StructType([
    StructField("batter", StringType()),
    StructField("by", StringType()),
    StructField("umpire", StringType()),
    StructField("decision", StringType()),
])

_EXTRAS = StructType([
    StructField("wides", IntegerType()),
    StructField("noballs", IntegerType()),
    StructField("byes", IntegerType()),
    StructField("legbyes", IntegerType()),
    StructField("penalty", IntegerType()),
])

_RUNS = StructType([
    StructField("batter", IntegerType()),
    StructField("extras", IntegerType()),
    StructField("total", IntegerType()),
    StructField("non_boundary", BooleanType()),
])

_DELIVERY = StructType([
    StructField("actual_delivery", StringType()),
    StructField("batter", StringType()),
    StructField("bowler", StringType()),
    StructField("non_striker", StringType()),
    StructField("runs", _RUNS),
    StructField("extras", _EXTRAS),
    StructField("wickets", ArrayType(_WICKET)),
    StructField("review", _REVIEW),
])

_OVER = StructType([
    StructField("over", IntegerType()),
    StructField("deliveries", ArrayType(_DELIVERY)),
])

_INNINGS = StructType([
    StructField("team", StringType()),
    StructField("overs", ArrayType(_OVER)),
    StructField("target", StructType([
        StructField("overs", IntegerType()),
        StructField("runs", IntegerType()),
    ])),
    StructField("declared", BooleanType()),
    StructField("super_over", BooleanType()),
    StructField("penalty_runs", StructType([
        StructField("pre", IntegerType()),
        StructField("post", IntegerType()),
    ])),
])

_OUTCOME = StructType([
    StructField("winner", StringType()),
    StructField("result", StringType()),
    StructField("method", StringType()),
    StructField("eliminator", StringType()),
    StructField("by", StructType([
        StructField("runs", IntegerType()),
        StructField("wickets", IntegerType()),
        StructField("innings", IntegerType()),
    ])),
])

_INFO = StructType([
    StructField("balls_per_over", IntegerType()),
    StructField("city", StringType()),
    StructField("dates", ArrayType(StringType())),
    StructField("event", StructType([
        StructField("name", StringType()),
        StructField("match_number", IntegerType()),
        StructField("group", StringType()),
        StructField("stage", StringType()),
    ])),
    StructField("gender", StringType()),
    StructField("match_type", StringType()),
    StructField("match_type_number", IntegerType()),
    StructField("outcome", _OUTCOME),
    StructField("overs", IntegerType()),
    StructField("player_of_match", ArrayType(StringType())),
    StructField("registry", StructType([
        StructField("people", MapType(StringType(), StringType())),
    ])),
    StructField("season", StringType()),
    StructField("team_type", StringType()),
    StructField("teams", ArrayType(StringType())),
    StructField("toss", StructType([
        StructField("winner", StringType()),
        StructField("decision", StringType()),
    ])),
    StructField("venue", StringType()),
])

CRICSHEET_SCHEMA = StructType([
    StructField("meta", StructType([
        StructField("data_version", StringType()),
        StructField("created", StringType()),
        StructField("revision", IntegerType()),
    ])),
    StructField("info", _INFO),
    StructField("innings", ArrayType(_INNINGS)),
])
