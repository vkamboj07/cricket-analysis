-- Batter vs bowler head-to-head lookup (output of features/matchup_lookup.py).
--
-- A LOOKUP, not a model feature: it summarises the WHOLE dataset "as of now", so using it to
-- predict a match inside that period would leak. About half of all pairings have faced six
-- balls or fewer; always read `tier` and `balls` next to any rate.
--
-- fam            'T20' (T20 + IT20), 'ODI' (ODI + ODM), or 'ALL' (raw counts pooled over both;
--                expected_sr, shrunk_sr and z_vs_expected are null there because strike rates
--                differ by about 40 points between the families)
-- tier           thin (<12 balls), limited (<30), moderate (<60), solid (60+)
-- expected_sr    what the two players' overall quality predicts for this pairing
-- shrunk_sr      observed rate pulled toward expected_sr by 30 pseudo-balls
-- z_vs_expected  standard errors between observed and expected; |z| > 2 is a prompt to look,
--                not a conclusion (it also absorbs phase of the innings, venue and match-to-match form)
-- Not partitioned, so there is no MSCK trap: create it before or after the job.

USE cricket;

DROP TABLE IF EXISTS matchup_lookup;

CREATE EXTERNAL TABLE matchup_lookup (
  fam                  STRING,
  batter_id            STRING,
  bowler_id            STRING,
  batter_name          STRING,
  bowler_name          STRING,
  balls                BIGINT,
  runs                 BIGINT,
  dismissals           BIGINT,
  dots                 BIGINT,
  fours                BIGINT,
  sixes                BIGINT,
  n_matches            BIGINT,
  first_match_date     DATE,
  last_match_date      DATE,
  strike_rate          DOUBLE,
  dot_pct              DOUBLE,
  balls_per_dismissal  DOUBLE,
  expected_sr          DOUBLE,
  shrunk_sr            DOUBLE,
  z_vs_expected        DOUBLE,
  tier                 STRING
)
STORED AS PARQUET
LOCATION 'hdfs:///data/processed/matchup_lookup';

-- Example: one batter's most-faced T20 matchups, with the reliability columns alongside.
SELECT bowler_name, balls, runs, dismissals, ROUND(strike_rate, 1) AS sr,
       ROUND(expected_sr, 1) AS expected_sr, ROUND(z_vs_expected, 2) AS z, tier
FROM matchup_lookup
WHERE fam = 'T20' AND batter_name = 'V Kohli'
ORDER BY balls DESC
LIMIT 8;
