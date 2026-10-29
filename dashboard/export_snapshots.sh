#!/usr/bin/env bash
# Export the dashboard's backing tables to local TSV, for when PyHive cannot
# connect from the host.
#   bash dashboard/export_snapshots.sh
set -euo pipefail

OUT_DIR="$(dirname "$0")/data_cache"
mkdir -p "$OUT_DIR"

# Two beeline quirks drive these flags:
#
# 1. hive.resultset.use.unique.column.names=false stops `SELECT *` emitting
#    table-qualified headers ("bowler_economy_by_phase.legal_balls"), which the
#    dashboard's pandas lookups do not expect.
#
# 2. tsv2, not csv2. The csv2 writer disables quoting by setting the quote
#    character to NUL, but still "quotes" any value containing the delimiter --
#    wrapping it in NUL bytes. Venues such as "County Ground, New Road,
#    Worcester" therefore split into extra columns and corrupted ~56% of the
#    matches rows. The data contains no tabs, so tsv2 round-trips cleanly.
BEELINE="beeline -u jdbc:hive2://localhost:10000/cricket --outputformat=tsv2 --silent=true --hiveconf hive.resultset.use.unique.column.names=false"

dump() {
  local name=$1 sql=$2
  echo "exporting $name"
  if [[ "${RUN_LOCAL:-0}" == "1" ]]; then
    $BEELINE -e "$sql" > "$OUT_DIR/$name.tsv"
  else
    docker exec hive-server $BEELINE -e "$sql" > "$OUT_DIR/$name.tsv"
  fi
  wc -l < "$OUT_DIR/$name.tsv"
}

dump bowler_economy_by_phase "SELECT * FROM bowler_economy_by_phase;"
dump matches "SELECT match_id, match_type, match_date, team1, team2, venue, winner FROM matches WHERE format_group='limited_overs';"
dump win_probability_by_ball "SELECT * FROM win_probability_by_ball;"

# Added for the model-evaluation views. These are small except wpa_by_ball and
# matchup_lookup, which are cut down to what the dashboard displays.
dump model_comparison "SELECT * FROM model_comparison;"
dump feature_importance "SELECT * FROM feature_importance;"
# Reliability table: held-out ('evaluation') rows only, ten equal-width probability buckets.
dump calibration_buckets "SELECT LEAST(FLOOR(win_probability * 10), 9) AS bucket, COUNT(*) AS n, AVG(label_batting_team_wins) AS actual, AVG(win_probability) AS raw_pred, AVG(win_probability_calibrated) AS calibrated_pred FROM win_probability_calibrated WHERE \`split\` = 'evaluation' GROUP BY LEAST(FLOOR(win_probability * 10), 9) ORDER BY bucket;"
# Win-probability-added per ball (first ball of each chase has no previous state, so wpa is NULL and dropped).
dump wpa_by_ball "SELECT match_id, ball_seq, prev_win_probability, wpa FROM wpa_by_ball WHERE wpa IS NOT NULL;"
# Pairs with fewer than 12 balls ("thin" tier, 877k of 1.17M rows) are not shown, so are not exported.
dump matchup_lookup "SELECT fam, batter_id, bowler_id, batter_name, bowler_name, balls, runs, dismissals, dots, fours, sixes, strike_rate, dot_pct, balls_per_dismissal, expected_sr, shrunk_sr, z_vs_expected, tier FROM matchup_lookup WHERE balls >= 12;"

echo "snapshots written to $OUT_DIR"
