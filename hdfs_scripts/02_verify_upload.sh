#!/usr/bin/env bash
# Verify the HDFS upload against the local manifest.
#   docker exec namenode bash /workspace/hdfs_scripts/02_verify_upload.sh
set -euo pipefail

MANIFEST=/workspace/hdfs_scripts/upload_manifest.csv

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST (run 01_classify_and_upload.py first)" >&2
  exit 1
fi

echo "== manifest counts =="
tail -n +2 "$MANIFEST" | cut -d, -f4 | sort | uniq -c

echo
echo "== hdfs counts =="
for d in /data/raw/limited_overs /data/raw/tests; do
  n=$(hdfs dfs -ls "$d" | grep -c '\.json$' || true)
  echo "  $n  $d"
done

echo
echo "== sample file =="
sample=$(hdfs dfs -ls /data/raw/limited_overs | awk '/\.json$/ {print $8; exit}')
hdfs dfs -cat "$sample" | head -c 300
echo
