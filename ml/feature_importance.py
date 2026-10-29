#!/usr/bin/env python3
"""Which features drive the win probability?

    docker exec spark-master /spark/bin/spark-submit \
        --driver-memory 6g \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --py-files /workspace/etl/schema_utils.py \
        /workspace/ml/feature_importance.py

Two independent views, because many features here are mathematically tied together
(wickets_remaining = 10 - wickets_down; resources_left, required_run_rate, run_rate_diff
and runs_required are all derived from balls_remaining, the target and the score).
Built-in tree importances split credit arbitrarily among such near-duplicates, so they
are cross-checked against a method that does not care:

  native       GBT / Random Forest .featureImportances; Logistic Regression coefficients
               multiplied by each feature's standard deviation (comparable across
               features, and signed so the direction of the effect is visible).
  permutation  Scramble a whole GROUP of related features jointly on the held-out
               seasons (each row gets the values of another random row) and measure how
               much Log-Loss rises and AUC falls. Group-level, so duplicated information
               cannot hide an important idea. Run for GBT and Logistic Regression.

One group is "Overs & wickets in hand", which is the information a Duckworth-Lewis-style
model uses, so the output quantifies how much the richer state adds beyond it.

Reads  the saved models (GBT, Logistic Regression, Random Forest) and the training table.
Writes <processed>/feature_importance (one row per model x method x feature or group).
"""
import sys

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.stat import Summarizer
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

sys.path.insert(0, "/workspace/etl")
sys.path.insert(0, "/workspace/ml")
import schema_utils as S  # noqa: E402
from calibrate_win_probability import clip  # noqa: E402
from train_win_probability import (CONTEXT_FEATURES, FORM_FEATURES, LABEL,  # noqa: E402
                                   MOMENTUM_FEATURES, STATE_FEATURES, chronological_split, prepare)

MODEL_DIR = S.MODEL_PATH.rsplit("/", 1)[0]
MODELS = {"Gradient-Boosted Trees": S.MODEL_PATH,
          "Logistic Regression": f"{MODEL_DIR}/win_probability_logreg",
          "Random Forest": f"{MODEL_DIR}/win_probability_rf"}
PERMUTATION_MODELS = ["Gradient-Boosted Trees", "Logistic Regression"]
OUT_PATH = f"{S.PROCESSED}/feature_importance"
N_REPEATS = 2

GROUPS = {
    "Chase pressure": ["runs_required", "required_run_rate", "run_rate_diff",
                       "runs_per_wicket_needed"],
    "Overs & wickets in hand (D/L-style)": ["balls_remaining", "wickets_down",
                                            "wickets_remaining", "resources_left"],
    "Scoreboard": ["current_score", "current_run_rate"],
    "Recent momentum": list(MOMENTUM_FEATURES),
    "Player form": list(FORM_FEATURES),
    "Match format": ["match_type"],
    "Who is playing (gender, level)": list(CONTEXT_FEATURES),
}
# Every group must cover exactly the model's raw inputs, nothing missing or doubled.
ALL_RAW = list(STATE_FEATURES) + list(MOMENTUM_FEATURES) + list(FORM_FEATURES) + ["match_type"] + list(CONTEXT_FEATURES)
assert sorted(c for g in GROUPS.values() for c in g) == sorted(ALL_RAW), "GROUPS do not cover the features"


def group_of(feature_name):
    base = feature_name.split("=")[0]
    for g, cols in GROUPS.items():
        if base in cols:
            return g
    return "other"


def feature_names(model, df):
    """Names of the assembled feature slots, read from the vector's own metadata."""
    attrs = model.transform(df.limit(1)).schema["features"].metadata["ml_attr"]["attrs"]
    slots = {a["idx"]: a["name"] for kind in attrs.values() for a in kind}
    names = []
    for i in range(len(slots)):
        n = slots[i]
        if n.endswith("_imp"):
            n = n[:-4]
        if n.startswith("match_type_vec_"):
            n = "match_type=" + n[len("match_type_vec_"):]
        names.append(n)
    return names


def quick_scores(scored):
    """Log-Loss and AUC of a frame that has columns p and the label."""
    p, y = clip(F.col("p")), F.col(LABEL)
    ll = scored.select(F.avg(-(y * F.log(p) + (1 - y) * F.log(1 - p)))).first()[0]
    auc = BinaryClassificationEvaluator(rawPredictionCol="p", labelCol=LABEL,
                                        metricName="areaUnderROC").evaluate(scored)
    return float(ll), float(auc)


def score(model, df):
    return model.transform(df).withColumn("p", vector_to_array(F.col("probability"))[1])


def shuffled_group(base, cols, pair):
    """base with `cols` replaced, jointly, by the values of another random row."""
    src = base.select(F.col("_k").alias("_k_src"), *[F.col(c).alias("__src_" + c) for c in cols])
    out = base.join(pair, "_k").join(src, "_k_src")
    for c in cols:
        out = out.withColumn(c, F.col("__src_" + c))
    return out.drop("_k_src", *["__src_" + c for c in cols])


def main():
    spark = S.build_spark("feature_importance")
    spark.sparkContext.setLogLevel("WARN")

    df = prepare(spark.read.parquet(S.TRAINING_TABLE))
    train, test, cutoff = chronological_split(df)
    models = {name: PipelineModel.load(path) for name, path in MODELS.items()}
    rows = []  # (model, method, name, importance, delta_auc, signed_effect, spread)

    # ------------------------------------------------------------------ native
    print("== native importances ==", flush=True)
    group_share = {}
    for name, model in models.items():
        names = feature_names(model, test)
        clf = model.stages[-1]
        if hasattr(clf, "featureImportances"):
            imp = [float(x) for x in clf.featureImportances.toArray()]
            signed = [None] * len(imp)
        else:  # logistic regression: standardised coefficient = coefficient * feature std
            feats = PipelineModel(model.stages[:-1]).transform(train)
            std = feats.select(Summarizer.metrics("std").summary(F.col("features")).alias("s")) \
                       .first()["s"]["std"].toArray()
            signed = [float(c * s) for c, s in zip(clf.coefficients.toArray(), std)]
            total = sum(abs(v) for v in signed)
            imp = [abs(v) / total for v in signed]
        order = sorted(range(len(names)), key=lambda i: -imp[i])
        print(f"\n[{name}]  {'feature':<26}{'share':>9}{'effect':>10}  group", flush=True)
        for i in order:
            eff = "" if signed[i] is None else f"{signed[i]:+.3f}"
            print(f"      {names[i]:<26}{imp[i]:>9.4f}{eff:>10}  {group_of(names[i])}", flush=True)
            rows.append((name, "native", names[i], imp[i], None, signed[i], None))
        gs = {}
        for n_, v in zip(names, imp):
            gs[group_of(n_)] = gs.get(group_of(n_), 0.0) + v
        group_share[name] = gs
        for g, v in gs.items():
            rows.append((name, "native_group", g, v, None, None, None))

    # ------------------------------------------------------------- permutation
    print("\n== permutation importance by feature group (held-out seasons) ==", flush=True)
    needed = ALL_RAW + [LABEL]
    base = (test.select(*needed).withColumn("_k", F.monotonically_increasing_id()).cache())
    n_base = base.count()
    print(f"held-out rows: {n_base}; repeats per group: {N_REPEATS}", flush=True)

    base_scores = {}
    for name in PERMUTATION_MODELS:
        base_scores[name] = quick_scores(score(models[name], base))
        print(f"baseline {name}: Log-Loss {base_scores[name][0]:.4f}  AUC {base_scores[name][1]:.4f}",
              flush=True)

    deltas = {(m, g): [] for m in PERMUTATION_MODELS for g in GROUPS}
    for r in range(N_REPEATS):
        # one random pairing per repeat, shared by every group and model in that repeat
        a = base.select("_k").withColumn("_pos", F.row_number().over(Window.orderBy("_k")))
        b = (base.select(F.col("_k").alias("_k_src")).withColumn("_r", F.rand(100 + r))
             .withColumn("_pos", F.row_number().over(Window.orderBy("_r"))))
        pair = a.join(b, "_pos").select("_k", "_k_src").cache()
        assert pair.count() == n_base
        for g, cols in GROUPS.items():
            shuf = shuffled_group(base, cols, pair).cache()
            for name in PERMUTATION_MODELS:
                ll, auc = quick_scores(score(models[name], shuf))
                d_ll, d_auc = ll - base_scores[name][0], base_scores[name][1] - auc
                deltas[(name, g)].append((d_ll, d_auc))
                print(f"  repeat {r + 1} | {name:<22} | {g:<38} | dLogLoss {d_ll:+.4f}  dAUC {d_auc:+.4f}",
                      flush=True)
            shuf.unpersist()
        pair.unpersist()

    # ----------------------------------------------------------------- summary
    print("\n== summary by group: how much worse does the model get without it? ==", flush=True)
    for name in PERMUTATION_MODELS:
        print(f"\n[{name}] baseline Log-Loss {base_scores[name][0]:.4f}", flush=True)
        print(f"  {'group':<38}{'dLogLoss':>10}{'spread':>9}{'dAUC':>9}{'native share':>14}", flush=True)
        ranked = sorted(GROUPS, key=lambda g: -sum(x[0] for x in deltas[(name, g)]) / N_REPEATS)
        for g in ranked:
            ds = deltas[(name, g)]
            mean_ll = sum(x[0] for x in ds) / N_REPEATS
            mean_auc = sum(x[1] for x in ds) / N_REPEATS
            spread = max(x[0] for x in ds) - min(x[0] for x in ds)
            print(f"  {g:<38}{mean_ll:>+10.4f}{spread:>9.4f}{mean_auc:>+9.4f}"
                  f"{group_share[name].get(g, 0.0):>14.3f}", flush=True)
            rows.append((name, "permutation", g, mean_ll, mean_auc, None, spread))

    # ----------------------------------------------------------------- persist
    ranked_rows, seen = [], {}
    for r in sorted(rows, key=lambda x: (x[0], x[1], -x[3])):
        k = (r[0], r[1])
        seen[k] = seen.get(k, 0) + 1
        ranked_rows.append((r[0], r[1], r[2], float(r[3]),
                            None if r[4] is None else float(r[4]),
                            None if r[5] is None else float(r[5]),
                            None if r[6] is None else float(r[6]), seen[k]))
    schema = StructType([
        StructField("model", StringType(), False), StructField("method", StringType(), False),
        StructField("name", StringType(), False), StructField("importance", DoubleType(), False),
        StructField("delta_auc", DoubleType(), True), StructField("signed_effect", DoubleType(), True),
        StructField("spread", DoubleType(), True), StructField("rank", IntegerType(), False)])
    spark.createDataFrame(ranked_rows, schema).write.mode("overwrite").parquet(OUT_PATH)
    n_back = spark.read.parquet(OUT_PATH).count()
    print(f"\nwrote {OUT_PATH}: {n_back} rows [{'OK' if n_back == len(ranked_rows) else 'MISMATCH'}]")
    spark.stop()


if __name__ == "__main__":
    main()
