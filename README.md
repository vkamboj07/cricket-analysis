# Cricket Ball-by-Ball Big Data Analytics

Big data pipeline over the Cricsheet ball-by-ball dataset (22,818 matches, ~10-11M deliveries)
using Hadoop HDFS, PySpark, MapReduce (Hadoop Streaming), Hive, and Spark MLlib.

## Pipeline
1. Raw JSON -> HDFS (`hdfs_scripts/`)
2. Flatten to ball-by-ball Parquet (`etl/`)
3. Bowler economy-by-phase MapReduce job (`mapreduce/`)
4. Hive external tables (`hive/`)
5. Match-state + rolling player-form features (`features/`)
6. Win-probability GBTClassifier (`ml/`)
7. Dashboard (`dashboard/`)

Feature engineering, the MapReduce phase job, and the model cover **limited-overs
matches only** (ODI/ODM/T20/IT20). Tests and multi-day matches are still loaded into
HDFS and queryable in Hive, but they have no run chase, no overs cap, and no clean
powerplay/middle/death structure, so the phase and win-probability work excludes them.

## Setup

Bring up the Hadoop/Spark/Hive stack:

```
docker compose up -d
```

- HDFS UI: http://localhost:9870
- YARN UI: http://localhost:8088
- Spark master UI: http://localhost:8080
- HiveServer2: `jdbc:hive2://localhost:10000`

The stack uses Hadoop 3.2.1 / Hive 2.3.2 / Spark 3.1.1 (BDE2020 images). The project
directory is bind-mounted into the containers at `/workspace`, so every script below
runs against your working tree without rebuilding an image.

Get the data (see `data/raw/`): download `all_json.zip` from
https://cricsheet.org/downloads/ and extract it there.

## Running the pipeline

Each stage writes to HDFS and the next reads it, so run them in order and check the
verification step before moving on.

```bash
# 1. raw JSON -> HDFS
docker exec namenode bash   /workspace/hdfs_scripts/00_init_hdfs_dirs.sh
docker exec namenode python3 /workspace/hdfs_scripts/01_classify_and_upload.py
docker exec namenode bash   /workspace/hdfs_scripts/02_verify_upload.sh

# 2. flatten to Parquet (+ the TSV feed the MapReduce job consumes)
docker exec spark-master spark-submit --py-files /workspace/etl/schema_utils.py \
  /workspace/etl/flatten_matches.py

# 3. bowler economy by phase (Hadoop Streaming)
bash mapreduce/run_job.sh

# 4. Hive external tables
bash hive/run_all.sh

# 5. features
docker exec spark-master spark-submit --py-files /workspace/etl/schema_utils.py \
  /workspace/features/match_state_features.py
docker exec spark-master spark-submit --py-files /workspace/etl/schema_utils.py \
  /workspace/features/player_form_features.py
docker exec spark-master spark-submit --py-files /workspace/etl/schema_utils.py \
  /workspace/features/build_training_table.py

# 6. train, then score every historical chase
docker exec spark-master spark-submit --py-files /workspace/etl/schema_utils.py \
  /workspace/ml/train_win_probability.py
docker exec spark-master spark-submit --py-files /workspace/etl/schema_utils.py \
  /workspace/ml/score_all_matches.py

# 7. dashboard
pip install -r dashboard/requirements-dashboard.txt
streamlit run dashboard/app.py
```

### HDFS layout

| Path | Written by |
|---|---|
| `/data/raw/limited_overs`, `/data/raw/tests` | stage 1 |
| `/data/processed/{matches,deliveries,player_registry}` | stage 2 |
| `/data/mapreduce_input/deliveries_tsv` | stage 2 |
| `/data/mapreduce_output/bowler_economy_by_phase` | stage 3 |
| `/data/processed/{match_state,player_form}` | stage 5 |
| `/data/processed/features/win_probability_training` | stage 5 |
| `/data/models/win_probability_gbt` | stage 6 |
| `/data/processed/win_probability_by_ball` | stage 6 |

## Design notes

- **Player ids.** Cricsheet has no match id inside the document (it is the filename)
  and no global player file: the name -> id mapping is embedded per match under
  `info.registry.people`. Joins therefore key on `(match_id, player_name)`.
- **Ball ordering.** `actual_delivery` ("12.3") repeats across re-bowled balls, so
  ordering uses the position within the over instead, and wides/no-balls are excluded
  from the legal-ball counter that drives every overs-remaining calculation.
- **Bowler economy** excludes byes and legbyes but charges wides and no-balls, and
  wides/no-balls do not count as balls bowled.
- **No leakage in player form.** Rolling form uses
  `Window.partitionBy(player_id).orderBy(match_date).rowsBetween(-10, -1)` -- the `-1`
  upper bound excludes the current match from its own average. Debutants get nulls
  (unknown form is not the same as bad form) and the model's `Imputer` fills them.
- **Chronological train/test split.** A random split would put balls from the same
  match on both sides and report a badly inflated score.

## Local development without the cluster

The Spark stages are plain PySpark and can run against local paths for testing. On
this machine that needs two things beyond `requirements.txt`:

- **JDK 17.** Spark 3.5.1 does not run on Java 18+ (`getSubject is not supported`);
  Java 26 rejects the usual `-Djava.security.manager=allow` workaround outright.
- **`winutils.exe` + `hadoop.dll`** (Hadoop 3.3.x) on Windows, with `HADOOP_HOME` set
  and `-Djava.library.path` pointing at them, or Spark fails with
  `UnsatisfiedLinkError: NativeIO$Windows.access0`.

`setuptools` is in `requirements.txt` because pyspark 3.5.1 imports `distutils`, which
Python 3.12+ removed. Note that `toPandas()` and Python UDFs hit a cloudpickle
recursion bug under Python 3.14 -- the pipeline avoids Python UDFs anyway
(`vector_to_array` instead), which is also faster.
