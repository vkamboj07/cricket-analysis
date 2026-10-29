-- Raw and calibrated win probability per delivery (output of ml/calibrate_win_probability.py).
--
-- Run this AFTER calibrate_win_probability.py, and NOT from run_all.sh: the table is
-- partitioned by match_type, so MSCK REPAIR only registers partitions that already
-- exist on HDFS (same trap as win_probability_by_ball and wpa_by_ball).
--
-- `split` says which rows can be used to judge the calibrator:
--   train        seasons the GBT trained on (the calibrator never saw them; not a valid test)
--   calibration  first half of the held-out seasons: the calibrator was FIT on these
--   evaluation   second half of the held-out seasons: the only rows valid for judging
-- win_probability_calibrated is isotonic-calibrated and clipped to [0.005, 0.995].

USE cricket;

DROP TABLE IF EXISTS win_probability_calibrated;

CREATE EXTERNAL TABLE win_probability_calibrated (
  match_id                    STRING,
  season                      STRING,
  innings_num                 INT,
  ball_seq                    INT,
  `split`                     STRING,
  win_probability             DOUBLE,
  win_probability_calibrated  DOUBLE,
  label_batting_team_wins     DOUBLE
)
PARTITIONED BY (match_type STRING)
STORED AS PARQUET
LOCATION 'hdfs:///data/processed/win_probability_calibrated';

MSCK REPAIR TABLE win_probability_calibrated;

-- Reliability on the evaluation seasons. Rows are grouped by their RAW probability
-- bucket; within each bucket, compare the real chaser win rate with the mean raw and
-- the mean calibrated probability. The calibrated gap should be much closer to 0.
SELECT bucket,
       COUNT(*)                                                           AS n,
       ROUND(AVG(label_batting_team_wins), 4)                             AS actual,
       ROUND(AVG(win_probability), 4)                                     AS raw_pred,
       ROUND(AVG(win_probability_calibrated), 4)                          AS calibrated_pred,
       ROUND(AVG(win_probability) - AVG(label_batting_team_wins), 4)      AS raw_gap,
       ROUND(AVG(win_probability_calibrated) - AVG(label_batting_team_wins), 4) AS calibrated_gap
FROM (SELECT *, LEAST(FLOOR(win_probability * 10), 9) AS bucket
      FROM win_probability_calibrated WHERE `split` = 'evaluation') t
GROUP BY bucket
ORDER BY bucket;
