#!/usr/bin/env python3
"""Join match state with player form into the model training table.

    docker exec spark-master spark-submit \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/features/build_training_table.py

Restricted to second-innings deliveries: the win-probability question is
"will the chasing team get there", which needs a target to be defined.
Output: /data/processed/features/win_probability_training (partitioned by season).
"""
import sys

from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402


def attach_form(state, form, id_col, prefix):
    f = form.select(
        F.col("player_id").alias("_pid"),
        F.col("match_id").alias("_mid"),
        F.col("rolling_strike_rate").alias(f"{prefix}_rolling_sr"),
        F.col("rolling_batting_avg").alias(f"{prefix}_rolling_avg"),
        F.col("rolling_economy").alias(f"{prefix}_rolling_econ"),
        F.col("matches_in_window").alias(f"{prefix}_form_matches"),
    )
    return (state
            .join(f, (state[id_col] == f["_pid"]) & (state["match_id"] == f["_mid"]), "left")
            .drop("_pid", "_mid"))


def build(state, form):
    chase = state.filter(
        (F.col("innings_num") == 2)
        & (F.col("target_runs").isNotNull())
        # Abandoned / no-result matches have no label to learn from.
        & (F.col("label_batting_team_wins").isNotNull())
        # Super overs are a different game; exclude the handful that exist.
        & (F.col("innings_overs_cap") > 1)
    )

    chase = attach_form(chase, form, "batter_id", "striker")
    chase = attach_form(chase, form, "non_striker_id", "nonstriker")
    chase = attach_form(chase, form, "bowler_id", "bowler")

    # Derived ratios that carry most of the signal in a chase.
    chase = (chase
             .withColumn("run_rate_diff",
                         F.col("required_run_rate") - F.col("current_run_rate"))
             .withColumn("resources_left",
                         F.col("balls_remaining") * F.col("wickets_remaining"))
             .withColumn("runs_per_wicket_needed",
                         F.when(F.col("wickets_remaining") > 0,
                                F.col("runs_required") / F.col("wickets_remaining"))))

    return chase


def main():
    spark = S.build_spark("build_training_table")
    spark.sparkContext.setLogLevel("WARN")

    state = spark.read.parquet(S.MATCH_STATE)
    form = spark.read.parquet(S.PLAYER_FORM)

    out = build(state, form)
    (out.write.mode("overwrite").partitionBy("season").parquet(S.TRAINING_TABLE))
    print(f"wrote training table -> {S.TRAINING_TABLE}")

    spark.stop()


if __name__ == "__main__":
    main()
