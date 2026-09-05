#!/usr/bin/env python3
"""Score every second-innings ball with the trained model.

    docker exec spark-master spark-submit \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/ml/score_all_matches.py

Precomputing the whole win-probability curve means the dashboard reads a
table instead of holding a Spark session open per page view.
"""
import sys

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
sys.path.insert(0, "/workspace/ml")
import schema_utils as S  # noqa: E402
from train_win_probability import prepare  # noqa: E402


def main():
    spark = S.build_spark("score_all_matches")
    spark.sparkContext.setLogLevel("WARN")

    df = prepare(spark.read.parquet(S.TRAINING_TABLE))
    model = PipelineModel.load(S.MODEL_PATH)

    scored = (model.transform(df)
              .withColumn("win_probability", vector_to_array(F.col("probability"))[1])
              .select("match_id", "season", "match_type", "innings_num",
                      "batting_team", "over", "ball_seq",
                      "balls_completed_in_innings", "current_score",
                      "wickets_down", "balls_remaining", "target_runs",
                      "runs_required", "required_run_rate", "current_run_rate",
                      "batter", "bowler", "win_probability",
                      "label_batting_team_wins"))

    scored.write.mode("overwrite").partitionBy("match_type").parquet(S.WIN_PROB_BY_BALL)
    print(f"wrote win probability curves -> {S.WIN_PROB_BY_BALL}")
    spark.stop()


if __name__ == "__main__":
    main()
