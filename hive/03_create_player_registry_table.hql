-- Player name -> Cricsheet person id, as recorded in each match document.
-- The same person keeps one id across matches, but the mapping only exists
-- per match, so joins downstream key on (match_id, player_name).

USE cricket;

DROP TABLE IF EXISTS player_registry;

CREATE EXTERNAL TABLE player_registry (
  match_id     STRING,
  player_name  STRING,
  person_id    STRING
)
STORED AS PARQUET
LOCATION 'hdfs:///data/processed/player_registry';

SELECT COUNT(*) AS registry_rows, COUNT(DISTINCT person_id) AS distinct_players
FROM player_registry;
