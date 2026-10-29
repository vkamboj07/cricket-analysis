#!/usr/bin/env python3
"""Train a GBT win-probability model on second-innings match states.

    docker exec spark-master spark-submit \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/ml/train_win_probability.py

The split is chronological, not random: a random split would put balls from
the same match on both sides and report a badly inflated score.
"""
import argparse
import json
import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
import schema_utils as S  # noqa: E402

LABEL = "label_batting_team_wins"

STATE_FEATURES = [
    "current_score", "wickets_down", "wickets_remaining", "balls_remaining",
    "current_run_rate", "runs_required", "required_run_rate", "run_rate_diff",
    "resources_left", "runs_per_wicket_needed",
]
# Recent momentum. Never null in the training table (every ball has a history),
# but unknowable for a hypothetical state in the dashboard's what-if view, so
# these go through the Imputer alongside the form features rather than straight
# into the assembler -- otherwise scoring a hypothetical state fails with
# "Encountered null while assembling a row".
MOMENTUM_FEATURES = [
    "runs_last_12", "wickets_last_12", "runs_last_30", "wickets_last_30",
]
FORM_FEATURES = [
    "striker_rolling_sr", "striker_rolling_avg",
    "nonstriker_rolling_sr", "bowler_rolling_econ",
]
IMPUTED_FEATURES = MOMENTUM_FEATURES + FORM_FEATURES
# Who is playing. Both are known before the first ball and are never null (the training table has
# exactly four gender x team_type segments), so they go straight into the assembler. Added after
# step 6 showed the model over-predicted women's chases and under-predicted men's club chases
# because it could not tell them apart (see CLAUDE.md, step 6b). The evaluate_*.py scripts keep
# their own pre-6b recipe on purpose, so their "baseline" is the model as it was before this.
CONTEXT_FEATURES = ["is_female", "is_international"]

# When no balls remain the required rate is undefined; cap rather than drop,
# since those rows are legitimate end-of-chase states.
RRR_CAP = 36.0


def prepare(df):
    """Fill every state feature that can legitimately be null.

    Order matters: run_rate_diff is derived from the two rates, so it is
    recomputed after both have been filled. Leaving any of these null would
    reach the assembler as NaN, which GBT rejects outright.
    """
    df = (df
          # No balls bowled yet on the first delivery of the innings.
          .withColumn("current_run_rate", F.coalesce("current_run_rate", F.lit(0.0)))
          # Undefined once no balls remain.
          .withColumn("required_run_rate",
                      F.least(F.coalesce("required_run_rate", F.lit(RRR_CAP)),
                              F.lit(RRR_CAP)))
          .withColumn("run_rate_diff",
                      F.col("required_run_rate") - F.col("current_run_rate"))
          # Undefined when the batting side is all out.
          .withColumn("runs_per_wicket_needed",
                      F.coalesce("runs_per_wicket_needed",
                                 F.col("runs_required").cast("double")))
          .withColumn(LABEL, F.col(LABEL).cast("double"))
          .withColumn("is_female", (F.col("gender") == "female").cast("double"))
          .withColumn("is_international", (F.col("team_type") == "international").cast("double")))
    return df.filter(F.col(LABEL).isNotNull())


def chronological_split(df, test_fraction=0.2):
    """Hold out the most recent seasons. Returns (train, test, cutoff)."""
    seasons = [r["season"] for r in
               df.select("season").distinct().orderBy("season").collect()]
    if len(seasons) < 2:
        raise SystemExit("need at least two seasons to split chronologically")
    cutoff_idx = max(1, int(len(seasons) * (1 - test_fraction)))
    train_seasons = set(seasons[:cutoff_idx])
    cutoff = seasons[cutoff_idx]
    return (df.filter(F.col("season").isin(*train_seasons)),
            df.filter(~F.col("season").isin(*train_seasons)),
            cutoff)


def build_pipeline(max_iter, max_depth):
    imputer = Imputer(
        inputCols=IMPUTED_FEATURES,
        outputCols=[f"{c}_imp" for c in IMPUTED_FEATURES],
        strategy="median",
    )
    indexer = StringIndexer(inputCol="match_type", outputCol="match_type_idx",
                            handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=["match_type_idx"],
                            outputCols=["match_type_vec"])
    assembler = VectorAssembler(
        inputCols=STATE_FEATURES + [f"{c}_imp" for c in IMPUTED_FEATURES]
                  + CONTEXT_FEATURES + ["match_type_vec"],
        outputCol="features",
        # "error", not "keep": keep turns stray nulls into NaN, which GBT
        # rejects deep inside training with an opaque message. Fail loudly at
        # assembly instead -- prepare() is responsible for the fills.
        handleInvalid="error",
    )
    gbt = GBTClassifier(labelCol=LABEL, featuresCol="features",
                        maxIter=max_iter, maxDepth=max_depth, seed=42)
    return Pipeline(stages=[imputer, indexer, encoder, assembler, gbt])


def calibration_table(predictions):
    """Predicted probability vs actual win rate, by decile.

    A win-probability model that discriminates well can still be poorly
    calibrated, and calibration is what makes the number meaningful.
    """
    return (predictions
            .withColumn("p", vector_to_array(F.col("probability"))[1])
            .withColumn("bucket", F.least(F.floor(F.col("p") * 10), F.lit(9)))
            .groupBy("bucket")
            .agg(F.count("*").alias("n"),
                 F.round(F.avg("p"), 4).alias("mean_predicted"),
                 F.round(F.avg(LABEL), 4).alias("actual_win_rate"))
            .orderBy("bucket"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    spark = S.build_spark("train_win_probability")
    spark.sparkContext.setLogLevel("WARN")

    df = prepare(spark.read.parquet(S.TRAINING_TABLE))
    train, test, cutoff = chronological_split(df, args.test_fraction)
    n_train, n_test = train.count(), test.count()
    print(f"train rows: {n_train} (seasons < {cutoff})")
    print(f"test rows:  {n_test} (seasons >= {cutoff})")

    model = build_pipeline(args.max_iter, args.max_depth).fit(train)
    predictions = model.transform(test).cache()

    auc = BinaryClassificationEvaluator(
        labelCol=LABEL, metricName="areaUnderROC").evaluate(predictions)
    accuracy = MulticlassClassificationEvaluator(
        labelCol=LABEL, metricName="accuracy").evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol=LABEL, metricName="f1").evaluate(predictions)

    metrics = {"auc": auc, "accuracy": accuracy, "f1": f1,
               "train_rows": n_train, "test_rows": n_test}
    print("\n== metrics ==")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n== calibration ==")
    calibration_table(predictions).show(10, truncate=False)

    model.write().overwrite().save(S.MODEL_PATH)
    print(f"\nmodel saved -> {S.MODEL_PATH}")

    # mlflow is not installed in the Spark container image; log when it is
    # available (e.g. running locally) and carry on when it is not.
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed here - skipping tracking")
        with open("/workspace/ml/last_run_metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
    else:
        mlflow.set_tracking_uri("file:/workspace/ml/mlruns")
        mlflow.set_experiment("win_probability")
        with mlflow.start_run():
            mlflow.log_params({"max_iter": args.max_iter,
                               "max_depth": args.max_depth,
                               "test_cutoff_season": cutoff,
                               "state_features": ",".join(STATE_FEATURES),
                               "momentum_features": ",".join(MOMENTUM_FEATURES),
                               "form_features": ",".join(FORM_FEATURES),
                               "context_features": ",".join(CONTEXT_FEATURES)})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        print("logged run to /workspace/ml/mlruns")

    spark.stop()


if __name__ == "__main__":
    main()
