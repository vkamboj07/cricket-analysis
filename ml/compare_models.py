#!/usr/bin/env python3
"""Compare Gradient-Boosted Trees, Logistic Regression and Random Forest.

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/ml/compare_models.py

Fairness rules:
  * same prepare() cleaning, same chronological split, same features
  * the preprocessing stages (imputer, indexer, encoder, assembler) are taken from
    train_win_probability.build_pipeline, so only the final classifier differs
  * hyperparameters are fixed in advance and are NOT tuned on the held-out seasons
  * every model is judged on the identical held-out rows, with the same metric code
    that calibrate_win_probability.py uses (Brier, Log-Loss, AUC, ECE)

GBT is retrained here with its production settings to time it and to check it
reproduces the saved model; the production model at S.MODEL_PATH is not touched.
Logistic Regression and Random Forest are saved next to it for later use.

Precision, recall and F1 are for the class "chasing team wins" at a 0.5 threshold.
F1 (weighted) is the support-weighted average over both classes, which is what the
earlier training script reported as "f1".
"""
import json
import os
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import (GBTClassifier, LogisticRegression,
                                       RandomForestClassifier)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

sys.path.insert(0, "/workspace/etl")
sys.path.insert(0, "/workspace/ml")
import schema_utils as S  # noqa: E402
from calibrate_win_probability import buckets, clip, metrics  # noqa: E402
from train_win_probability import (LABEL, build_pipeline,  # noqa: E402
                                   chronological_split, prepare)

MODEL_DIR = S.MODEL_PATH.rsplit("/", 1)[0]
COMPARISON_PATH = f"{S.PROCESSED}/model_comparison"
SAVE_PATHS = {"Logistic Regression": f"{MODEL_DIR}/win_probability_logreg",
              "Random Forest": f"{MODEL_DIR}/win_probability_rf"}

# Fixed in advance. GBT matches production (train_win_probability.py defaults).
SPECS = [
    ("Gradient-Boosted Trees", "maxIter=60, maxDepth=5, seed=42",
     lambda: GBTClassifier(labelCol=LABEL, featuresCol="features",
                           maxIter=60, maxDepth=5, seed=42)),
    ("Logistic Regression", "maxIter=100, regParam=0, standardized",
     lambda: LogisticRegression(labelCol=LABEL, featuresCol="features",
                                maxIter=100, regParam=0.0, standardization=True)),
    ("Random Forest", "numTrees=100, maxDepth=8, seed=42",
     lambda: RandomForestClassifier(labelCol=LABEL, featuresCol="features",
                                    numTrees=100, maxDepth=8, seed=42)),
]


def classification_metrics(df):
    """Accuracy, precision, recall, F1 for 'chasing team wins' at threshold 0.5."""
    pred = (F.col("p") >= 0.5).cast("int")
    y = F.col(LABEL).cast("int")
    r = df.select(F.sum(pred * y).alias("tp"), F.sum(pred * (1 - y)).alias("fp"),
                  F.sum((1 - pred) * y).alias("fn"),
                  F.sum((1 - pred) * (1 - y)).alias("tn")).first()
    tp, fp, fn, tn = (float(r[k]) for k in ("tp", "fp", "fn", "tn"))
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_win = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    f1_lose = 2 * tn / (2 * tn + fn + fp) if tn else 0.0
    f1_weighted = ((tp + fn) * f1_win + (tn + fp) * f1_lose) / n
    return {"accuracy": (tp + tn) / n, "precision_win": precision,
            "recall_win": recall, "f1_win": f1_win, "f1_weighted": f1_weighted}


def main():
    spark = S.build_spark("compare_models")
    spark.sparkContext.setLogLevel("WARN")

    df = prepare(spark.read.parquet(S.TRAINING_TABLE))
    train, test, cutoff = chronological_split(df)
    n_train, n_test = train.count(), test.count()
    print(f"train rows: {n_train} (seasons < {cutoff}) | test rows: {n_test} (seasons >= {cutoff})")

    rows, preds = [], {}
    for name, params, make in SPECS:
        stages = build_pipeline(60, 5).getStages()[:-1] + [make()]   # shared preprocessing
        t0 = time.time()
        model = Pipeline(stages=stages).fit(train)
        secs = time.time() - t0
        print(f"\n[{name}] trained in {secs:.0f}s ({params})")

        scored = (model.transform(test)
                  .withColumn("p", vector_to_array(F.col("probability"))[1])
                  .select("season", LABEL, "p").cache())
        preds[name] = scored
        row = {"model": name, "hyperparameters": params, "train_seconds": float(secs),
               "test_rows": int(scored.count())}
        row.update(metrics(scored, "p"))
        row.update(classification_metrics(scored))
        rows.append(row)
        if name in SAVE_PATHS:
            model.write().overwrite().save(SAVE_PATHS[name])
            print(f"[{name}] saved -> {SAVE_PATHS[name]}")

    # ---------------- headline table ----------------
    print("\n== all models, identical held-out rows ==")
    head = (f"{'model':<24}{'AUC':>8}{'Brier':>8}{'LogLoss':>9}{'ECE':>8}"
            f"{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'F1(w)':>8}{'train s':>9}")
    print(head)
    for r in rows:
        print(f"{r['model']:<24}{r['auc']:>8.4f}{r['brier']:>8.4f}{r['logloss']:>9.4f}"
              f"{r['ece']:>8.4f}{r['accuracy']:>8.4f}{r['precision_win']:>8.4f}"
              f"{r['recall_win']:>8.4f}{r['f1_win']:>8.4f}{r['f1_weighted']:>8.4f}"
              f"{r['train_seconds']:>9.0f}")
    print("(AUC higher = better; Brier, LogLoss, ECE lower = better; precision/recall/F1 are for "
          "'chasing team wins' at 0.5)")

    # ---------------- reproducibility of the production GBT ----------------
    saved = "/workspace/ml/last_run_metrics.json"
    gbt_auc = next(r["auc"] for r in rows if r["model"] == "Gradient-Boosted Trees")
    if os.path.exists(saved):
        with open(saved) as fh:
            prod = json.load(fh)
        d = abs(gbt_auc - prod["auc"])
        print(f"\nGBT reproducibility: retrained AUC {gbt_auc:.10f} vs saved run {prod['auc']:.10f} "
              f"(difference {d:.2e}) [{'OK' if d < 1e-6 else 'DIFFERS'}]")

    # ---------------- does the ranking hold in every held-out season? ----------------
    p = clip(F.col("p"))
    y = F.col(LABEL)
    per_season = {}
    for name, scored in preds.items():
        for r in (scored.groupBy("season")
                  .agg(F.avg(-(y * F.log(p) + (1 - y) * F.log(1 - p))).alias("ll")).collect()):
            per_season.setdefault(r["season"], {})[name] = r["ll"]
    wins = {name: 0 for name in preds}
    print("\n== Log-Loss by held-out season (lower = better) ==")
    names = list(preds)
    print(f"{'season':<10}" + "".join(f"{n.split()[0][:12]:>14}" for n in names) + "   best")
    for season in sorted(per_season):
        best = min(per_season[season], key=per_season[season].get)
        wins[best] += 1
        print(f"{season:<10}" + "".join(f"{per_season[season][n]:>14.4f}" for n in names)
              + f"   {best.split()[0]}")
    print("seasons won (lowest Log-Loss): " + ", ".join(f"{n}: {w}" for n, w in wins.items()))

    # ---------------- calibration shape of each model ----------------
    print("\n== reliability: mean predicted minus actual chaser win rate, by the model's own "
          "probability decile ==")
    print(f"{'bucket':<8}" + "".join(f"{n.split()[0][:12]:>14}" for n in names))
    tables = {n: {r["bucket"]: r for r in buckets(preds[n], "p").collect()} for n in names}
    for b in range(10):
        cells = []
        for n in names:
            r = tables[n].get(b)
            cells.append(f"{(r['mean_pred'] - r['actual']):>+14.4f}" if r else f"{'-':>14}")
        print(f"{b:<8}" + "".join(cells))

    # ---------------- persist ----------------
    out = spark.createDataFrame(rows).select(
        "model", "hyperparameters", "train_seconds", "test_rows", "auc", "brier", "logloss",
        "ece", "accuracy", "precision_win", "recall_win", "f1_win", "f1_weighted")
    out.write.mode("overwrite").parquet(COMPARISON_PATH)
    n_back = spark.read.parquet(COMPARISON_PATH).count()
    print(f"\nwrote {COMPARISON_PATH}: {n_back} rows [{'OK' if n_back == len(rows) else 'MISMATCH'}]")
    spark.stop()


if __name__ == "__main__":
    main()
