-- Ball-by-ball fact table, one row per delivery.
-- Note: `over` is a reserved word in Hive (windowing), so it must be quoted.

USE cricket;

DROP TABLE IF EXISTS deliveries;

CREATE EXTERNAL TABLE deliveries (
  match_id                    STRING,
  overs_cap                   INT,
  innings_num                 INT,
  batting_team                STRING,
  target_runs                 INT,
  target_overs                INT,
  `over`                      INT,
  delivery_idx                INT,
  actual_delivery             STRING,
  batter                      STRING,
  bowler                      STRING,
  non_striker                 STRING,
  runs_batter                 INT,
  runs_extras                 INT,
  runs_total                  INT,
  extra_wides                 INT,
  extra_noballs               INT,
  extra_byes                  INT,
  extra_legbyes               INT,
  extra_penalty               INT,
  is_wicket                   BOOLEAN,
  num_wickets                 INT,
  player_out                  STRING,
  wicket_kind                 STRING,
  num_fielders                INT,
  review_decision             STRING,
  is_legal_ball               BOOLEAN,
  ball_seq                    INT,
  balls_completed_in_innings  BIGINT,
  batter_id                   STRING,
  bowler_id                   STRING,
  non_striker_id              STRING,
  player_out_id               STRING
)
PARTITIONED BY (format_group STRING, match_type STRING)
STORED AS PARQUET
LOCATION 'hdfs:///data/processed/deliveries';

MSCK REPAIR TABLE deliveries;

SELECT format_group, match_type, COUNT(*) AS balls
FROM deliveries GROUP BY format_group, match_type;
