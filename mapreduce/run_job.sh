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

run hadoop jar "$JAR" \
  -D mapreduce.job.name="bowler_economy_by_phase" \
  -D mapreduce.job.reduces=4 \
  -D stream.num.map.output.key.fields=3 \
  -D mapreduce.map.output.key.field.separator=$'\t' \
  -D mapreduce.partition.keypartitioner.options=-k1,3 \
  -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
  -files /workspace/mapreduce/mapper.py,/workspace/mapreduce/reducer.py,/workspace/mapreduce/phase_utils.py \
  -mapper "python3 mapper.py" \
  -reducer "python3 reducer.py" \
  -input "$INPUT" \
  -output "$OUTPUT"

echo
echo "== output sample =="
run bash -c "hdfs dfs -cat $OUTPUT/part-00000 | head -10"
echo
run bash -c "hdfs dfs -cat $OUTPUT/part-* | wc -l"
