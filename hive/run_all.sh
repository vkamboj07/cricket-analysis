#!/usr/bin/env bash
# Create all Hive external tables.
#   bash hive/run_all.sh
set -euo pipefail

BEELINE_URL="jdbc:hive2://localhost:10000"

run() {
  if [[ "${RUN_LOCAL:-0}" == "1" ]]; then
    beeline -u "$BEELINE_URL" -f "$1"
  else
    docker exec hive-server beeline -u "$BEELINE_URL" -f "$1"
  fi
}

for f in 01_create_matches_table.hql \
         02_create_deliveries_table.hql \
         03_create_player_registry_table.hql \
         04_create_bowler_economy_table.hql \n         05_create_win_probability_table.hql; do
  echo "=== $f ==="
  run "/workspace/hive/$f"
done
