-- Match dimension table: one row per match.
-- Spark wrote the partition directories directly, so the metastore needs
-- MSCK REPAIR to discover them -- without it the table reads as empty.

CREATE DATABASE IF NOT EXISTS cricket;
USE cricket;

DROP TABLE IF EXISTS matches;

CREATE EXTERNAL TABLE matches (
  match_id            STRING,
  match_type          STRING,
  match_type_number   INT,
  gender              STRING,
  team_type           STRING,
  season              STRING,
  dates               ARRAY<STRING>,
  match_date          DATE,
  team1               STRING,
  team2               STRING,
  venue               STRING,
  city                STRING,
  event_name          STRING,
  event_match_number  INT,
  event_stage         STRING,
  balls_per_over      INT,
  overs_cap           INT,
  toss_winner         STRING,
  toss_decision       STRING,
  winner              STRING,
  result_type         STRING,
  eliminator          STRING,
  method              STRING,
  win_by_type         STRING,
  win_by_value        INT,
  player_of_match     ARRAY<STRING>
)
PARTITIONED BY (format_group STRING)
STORED AS PARQUET
LOCATION 'hdfs:///data/processed/matches';

MSCK REPAIR TABLE matches;

SELECT format_group, COUNT(*) AS matches FROM matches GROUP BY format_group;
