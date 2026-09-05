#!/usr/bin/env python3
"""Score a single hypothetical match state against the trained model.

    python ml/predict_live_state.py --score 120 --wickets 4 --balls-remaining 36 \
        --target 180 --match-type T20

Used by the dashboard's what-if view. Keeps its own Spark session, so it is
slow for one call -- fine interactively, not for a hot loop.
"""
import argparse

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

MODEL_PATH_DEFAULT = "hdfs:///data/models/win_probability_gbt"


def state_row(score, wickets, balls_remaining, target, match_type,
              overs_cap=20, striker_sr=None, nonstriker_sr=None, bowler_econ=None):
    balls_bowled = overs_cap * 6 - balls_remaining
    runs_required = max(target - score, 0)
    crr = (score * 6.0 / balls_bowled) if balls_bowled > 0 else 0.0
    rrr = (runs_required * 6.0 / balls_remaining) if balls_remaining > 0 else 36.0
    wickets_remaining = 10 - wickets
    return {
        "match_type": match_type,
        "current_score": float(score),
        "wickets_down": float(wickets),
        "wickets_remaining": float(wickets_remaining),
        "balls_remaining": float(balls_remaining),
        "current_run_rate": crr,
        "runs_required": float(runs_required),
        "required_run_rate": rrr,
        "run_rate_diff": rrr - crr,
        "resources_left": float(balls_remaining * wickets_remaining),
        "runs_per_wicket_needed": (runs_required / wickets_remaining
                                   if wickets_remaining > 0 else float(runs_required)),
        # Recent-momentum inputs are unknown for a hypothetical state; the
        # pipeline's imputer fills them from training medians.
        "runs_last_12": None, "wickets_last_12": None,
        "runs_last_30": None, "wickets_last_30": None,
        "striker_rolling_sr": striker_sr, "striker_rolling_avg": None,
        "nonstriker_rolling_sr": nonstriker_sr, "bowler_rolling_econ": bowler_econ,
    }


def predict(spark, model, row):
    df = spark.createDataFrame([row])
    out = model.transform(df).select("probability").first()
    return float(out["probability"][1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score", type=int, required=True)
    p.add_argument("--wickets", type=int, required=True)
    p.add_argument("--balls-remaining", type=int, required=True)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--match-type", default="T20")
    p.add_argument("--overs-cap", type=int, default=20)
    p.add_argument("--model", default=MODEL_PATH_DEFAULT)
    args = p.parse_args()

    spark = SparkSession.builder.appName("predict_live_state").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load(args.model)
    row = state_row(args.score, args.wickets, args.balls_remaining,
                    args.target, args.match_type, args.overs_cap)
    print(f"win probability for the chasing team: {predict(spark, model, row):.3f}")
    spark.stop()


if __name__ == "__main__":
    main()
