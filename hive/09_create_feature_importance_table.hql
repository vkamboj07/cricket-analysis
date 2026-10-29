-- Feature importance from ml/feature_importance.py: one row per model x method x name.
--
-- Not partitioned, so there is no MSCK trap: it can be created before or after the job.
-- Re-run feature_importance.py to refresh the rows.
--
-- method
--   native        built-in importance of one feature slot. importance = share of the model's
--                 total (sums to 1 per model). Logistic Regression: share of |coefficient x
--                 feature std|, with the sign kept in signed_effect (positive = pushes the
--                 chasing team's win probability up).
--   native_group  the native shares summed over a group of related features
--   permutation   how much WORSE the model gets when the whole group is scrambled on the
--                 held-out seasons. importance = rise in Log-Loss, delta_auc = fall in AUC,
--                 spread = difference between the repeats (a noise indicator).
-- Many features are mathematically tied together, so read the permutation rows by GROUP;
-- feature-level native shares split credit between near-duplicates.

USE cricket;

DROP TABLE IF EXISTS feature_importance;

CREATE EXTERNAL TABLE feature_importance (
  model          STRING,
  method         STRING,
  name           STRING,
  importance     DOUBLE,
  delta_auc      DOUBLE,
  signed_effect  DOUBLE,
  spread         DOUBLE,
  `rank`         INT
)
STORED AS PARQUET
LOCATION 'hdfs:///data/processed/feature_importance';

SELECT model, `rank`, name, ROUND(importance, 4) AS d_logloss, ROUND(delta_auc, 4) AS d_auc
FROM feature_importance
WHERE method = 'permutation'
ORDER BY model, `rank`;
