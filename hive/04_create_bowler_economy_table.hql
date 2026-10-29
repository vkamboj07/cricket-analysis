-- Output of the Hadoop Streaming job (mapreduce/run_job.sh).
-- Plain tab-separated text, one row per (bowler, match_type, phase).
--
-- dot_balls and dot_ball_pct are appended after the original nine columns, so every earlier
-- column keeps its position. A dot ball is a legal ball on which the bowler is charged no
-- runs (byes and legbyes are not charged to the bowler, so those balls count as dots);
-- dot_ball_pct is a true percentage, 0-100.

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
  balls_per_wicket DOUBLE,
  dot_balls     INT,
  dot_ball_pct  DOUBLE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION 'hdfs:///data/mapreduce_output/bowler_economy_by_phase';

-- Highest dot-ball percentage in the T20 death overs, with a meaningful workload.
SELECT bowler_id, legal_balls, dot_balls, dot_ball_pct, economy
FROM bowler_economy_by_phase
WHERE phase = 'death' AND match_type = 'T20' AND legal_balls >= 600
ORDER BY dot_ball_pct DESC
LIMIT 10;

-- Most economical death bowlers with a meaningful workload.
SELECT bowler_id, match_type, overs_bowled, wickets, economy
FROM bowler_economy_by_phase
WHERE phase = 'death' AND match_type = 'T20' AND legal_balls >= 600
ORDER BY economy ASC
LIMIT 10;
