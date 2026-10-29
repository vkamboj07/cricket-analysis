-- Per-delivery Win Probability Added (output of ml/compute_wpa.py).
--
-- Run this AFTER compute_wpa.py, and NOT from run_all.sh. The table is partitioned by
-- match_type, so MSCK REPAIR only registers partitions that already exist on HDFS;
-- run before the job it would create an empty table with no error (same trap as
-- win_probability_by_ball).
--
-- wpa = win_probability - prev_win_probability, from the chasing team's point of view.
-- Positive helped the chasers, negative helped the bowling side. The first ball of each
-- chase has no previous probability, so its wpa is NULL.

USE cricket;

DROP TABLE IF EXISTS wpa_by_ball;

CREATE EXTERNAL TABLE wpa_by_ball (
  match_id                    STRING,
  season                      STRING,
  innings_num                 INT,
  batting_team                STRING,
  `over`                      INT,
  ball_seq                    INT,
  balls_completed_in_innings  BIGINT,
  current_score               BIGINT,
  wickets_down                BIGINT,
  balls_remaining             BIGINT,
  target_runs                 INT,
  runs_required               BIGINT,
  batter                      STRING,
  bowler                      STRING,
  win_probability             DOUBLE,
  prev_win_probability        DOUBLE,
  wpa                         DOUBLE
)
PARTITIONED BY (match_type STRING)
STORED AS PARQUET
LOCATION 'hdfs:///data/processed/wpa_by_ball';

MSCK REPAIR TABLE wpa_by_ball;

-- The five biggest single-ball swings in the dataset.
SELECT match_id, match_type, batter, bowler, prev_win_probability, win_probability, wpa
FROM wpa_by_ball
WHERE wpa IS NOT NULL
ORDER BY ABS(wpa) DESC
LIMIT 5;
