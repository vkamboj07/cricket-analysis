#!/usr/bin/env bash
# Export the dashboard's backing tables to local CSV, for when PyHive cannot
# connect from the host.
#   bash dashboard/export_snapshots.sh
set -euo pipefail

OUT_DIR="$(dirname "$0")/data_cache"
mkdir -p "$OUT_DIR"
BEELINE="beeline -u jdbc:hive2://localhost:10000/cricket --outputformat=csv2 --silent=true"

dump() {
  local name=$1 sql=$2
  echo "exporting $name"
  if [[ "${RUN_LOCAL:-0}" == "1" ]]; then
    $BEELINE -e "$sql" > "$OUT_DIR/$name.csv"
  else
    docker exec hive-server $BEELINE -e "$sql" > "$OUT_DIR/$name.csv"
  fi
  wc -l < "$OUT_DIR/$name.csv"
}

dump bowler_economy_by_phase "SELECT * FROM bowler_economy_by_phase;"
dump matches "SELECT match_id, match_type, match_date, team1, team2, venue, winner FROM matches WHERE format_group='limited_overs';"
dump win_probability_by_ball "SELECT * FROM win_probability_by_ball;"

echo "snapshots written to $OUT_DIR"
