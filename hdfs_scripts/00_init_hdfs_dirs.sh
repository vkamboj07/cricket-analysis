#!/usr/bin/env bash
# Create the HDFS directory skeleton for the pipeline.
# Run inside the namenode container:
#   docker exec namenode bash /workspace/hdfs_scripts/00_init_hdfs_dirs.sh
set -euo pipefail

DIRS=(
  /data/raw/limited_overs
  /data/raw/tests
  /data/processed
  /data/mapreduce_input
  /data/mapreduce_output
  /data/models
  /user/hive/warehouse
)

for d in "${DIRS[@]}"; do
  hdfs dfs -mkdir -p "$d"
  echo "created: $d"
done

hdfs dfs -chmod -R g+w /user/hive/warehouse
hdfs dfs -ls /data
