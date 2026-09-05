-- Precomputed win-probability curves (output of ml/score_all_matches.py).
-- The dashboard reads this instead of holding a Spark session open.

USE cricket;

DROP TABLE IF EXISTS win_probability_by_ball;

CREATE EXTERNAL TABLE win_probability_by_ball (
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
  required_run_rate           DOUBLE,
  current_run_rate            DOUBLE,
  batter                      STRING,
  bowler                      STRING,
  win_probability             DOUBLE,
  label_batting_team_wins     DOUBLE
)
PARTITIONED BY (match_type STRING)
STORED AS PARQUET
LOCATION 'hdfs:///data/processed/win_probability_by_ball';

MSCK REPAIR TABLE win_probability_by_ball;

SELECT match_type, COUNT(*) AS scored_balls
FROM win_probability_by_ball GROUP BY match_type;
