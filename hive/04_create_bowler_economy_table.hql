-- Output of the Hadoop Streaming job (mapreduce/run_job.sh).
-- Plain tab-separated text, one row per (bowler, match_type, phase).

USE cricket;

DROP TABLE IF EXISTS bowler_economy_by_phase;

CREATE EXTERNAL TABLE bowler_economy_by_phase (
  bowler_id     STRING,
  match_type    STRING,
  phase         STRING,
  runs_conceded INT,
  legal_balls   INT,
  overs_bowled  DOUBLE,
  wickets       INT,
  economy       DOUBLE,
  balls_per_wicket DOUBLE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION 'hdfs:///data/mapreduce_output/bowler_economy_by_phase';

-- Most economical death bowlers with a meaningful workload.
SELECT bowler_id, match_type, overs_bowled, wickets, economy
FROM bowler_economy_by_phase
WHERE phase = 'death' AND match_type = 'T20' AND legal_balls >= 600
ORDER BY economy ASC
LIMIT 10;
