#!/usr/bin/env bash
# Bowler economy-by-phase, via Hadoop Streaming.
# Run from the host (needs docker), or inside the resourcemanager container
# with RUN_LOCAL=1:
#   bash mapreduce/run_job.sh
set -euo pipefail

INPUT=/data/mapreduce_input/deliveries_tsv
OUTPUT=/data/mapreduce_output/bowler_economy_by_phase
STREAMING_JAR_GLOB="/opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-*.jar"

run() {
  if [[ "${RUN_LOCAL:-0}" == "1" ]]; then
    "$@"
  else
    docker exec resourcemanager "$@"
  fi
}

# The exact jar filename carries the Hadoop version; resolve it rather than
# hardcoding so an image bump doesn't silently break this.
JAR=$(run bash -c "ls $STREAMING_JAR_GLOB | head -1")
echo "streaming jar: $JAR"

# Output dir must not exist; MapReduce refuses to overwrite.
run hdfs dfs -rm -r -f "$OUTPUT"

# Two ordering/environment constraints, both of which fail confusingly:
#
# 1. Generic options (-D, -files) must come before the streaming options
#    (-partitioner, -mapper, ...). GenericOptionsParser stops at the first
#    non-generic argument, so a -files after -partitioner is never parsed and
#    the job dies with "Unrecognized option: -files".
#
# 2. -cmdenv PYTHONPATH=. is required. Each -files entry is symlinked into the
#    task working dir from its own filecache slot, and Python resolves a
#    symlinked script to its real directory when setting sys.path[0]. Without
#    this the mapper looks for phase_utils in filecache/<n>/ and dies with
#    "ImportError: No module named 'phase_utils'".
run hadoop jar "$JAR" \
  -D mapreduce.job.name="bowler_economy_by_phase" \
  -D mapreduce.job.reduces=4 \
  -D stream.num.map.output.key.fields=3 \
  -D mapreduce.map.output.key.field.separator=$'\t' \
  -D mapreduce.partition.keypartitioner.options=-k1,3 \
  -files /workspace/mapreduce/mapper.py,/workspace/mapreduce/reducer.py,/workspace/mapreduce/phase_utils.py \
  -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
  -cmdenv PYTHONPATH=. \
  -mapper "python3 mapper.py" \
  -reducer "python3 reducer.py" \
  -input "$INPUT" \
  -output "$OUTPUT"

echo
echo "== output sample =="
run bash -c "hdfs dfs -cat $OUTPUT/part-00000 | head -10"
echo
run bash -c "hdfs dfs -cat $OUTPUT/part-* | wc -l"
