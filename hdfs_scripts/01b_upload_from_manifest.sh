#!/usr/bin/env bash
# Upload the classified Cricsheet files to HDFS from the manifest.
#
# The BDE2020 namenode image is Debian 9 and its newest packaged Python is
# 3.5, which cannot parse the f-strings in 01_classify_and_upload.py. The
# spark-master image has Python 3.7 but no `hdfs` CLI. So the pipeline splits
# across the two containers:
#   docker exec spark-master python3 /workspace/hdfs_scripts/01_classify_and_upload.py --dry-run
#   docker exec namenode bash    /workspace/hdfs_scripts/01b_upload_from_manifest.sh
set -euo pipefail

SRC_DIR=/workspace/data/raw/all_json
MANIFEST=/workspace/hdfs_scripts/upload_manifest.csv
# Same batching rationale as the Python version: ~50 JVM starts, not 22k.
BATCH_SIZE=400

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST (run 01_classify_and_upload.py --dry-run first)" >&2
  exit 1
fi

upload_bucket() {
  local bucket=$1 dest=$2
  local list total done_n=0
  list=$(mktemp)
  # csv.writer emits CRLF, so strip the trailing CR before comparing the bucket.
  awk -F, -v b="$bucket" -v dir="$SRC_DIR" \
    'NR>1 { sub(/\r$/, "", $4); if ($4 == b) print dir "/" $1 }' "$MANIFEST" >"$list"
  total=$(wc -l <"$list")

  if [[ "$total" -eq 0 ]]; then
    echo "nothing to upload for bucket $bucket"
    rm -f "$list"
    return
  fi

  echo "uploading $total files -> $dest"
  local batch=()
  while IFS= read -r path; do
    batch+=("$path")
    if [[ ${#batch[@]} -ge $BATCH_SIZE ]]; then
      # -f makes re-runs idempotent instead of failing on existing files.
      hdfs dfs -put -f "${batch[@]}" "$dest"
      done_n=$((done_n + ${#batch[@]}))
      echo "  $done_n/$total"
      batch=()
    fi
  done <"$list"

  if [[ ${#batch[@]} -gt 0 ]]; then
    hdfs dfs -put -f "${batch[@]}" "$dest"
    done_n=$((done_n + ${#batch[@]}))
    echo "  $done_n/$total"
  fi

  rm -f "$list"
}

upload_bucket limited_overs /data/raw/limited_overs
upload_bucket tests         /data/raw/tests
echo "upload complete"
