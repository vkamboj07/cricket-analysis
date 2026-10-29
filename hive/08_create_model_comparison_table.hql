-- One row per model from ml/compare_models.py, all judged on the identical held-out seasons.
--
-- Not partitioned, so unlike the 06/07 tables there is no MSCK trap: it can be created
-- before or after the job. Re-run compare_models.py to refresh the rows.
--
-- auc                          higher is better
-- brier, logloss, ece          lower is better (ece = expected calibration error, 10 buckets)
-- precision_win, recall_win,
-- f1_win                       for the class "chasing team wins", threshold 0.5
-- f1_weighted                  support-weighted F1 over both classes (the older "f1" figure)

USE cricket;

DROP TABLE IF EXISTS model_comparison;

CREATE EXTERNAL TABLE model_comparison (
  model            STRING,
  hyperparameters  STRING,
  train_seconds    DOUBLE,
  test_rows        BIGINT,
  auc              DOUBLE,
  brier            DOUBLE,
  logloss          DOUBLE,
  ece              DOUBLE,
  accuracy         DOUBLE,
  precision_win    DOUBLE,
  recall_win       DOUBLE,
  f1_win           DOUBLE,
  f1_weighted      DOUBLE
)
STORED AS PARQUET
LOCATION 'hdfs:///data/processed/model_comparison';

SELECT model,
       ROUND(auc, 4) AS auc, ROUND(brier, 4) AS brier, ROUND(logloss, 4) AS logloss,
       ROUND(ece, 4) AS ece, ROUND(accuracy, 4) AS accuracy,
       ROUND(precision_win, 4) AS prec, ROUND(recall_win, 4) AS recall, ROUND(f1_win, 4) AS f1
FROM model_comparison
ORDER BY logloss;
