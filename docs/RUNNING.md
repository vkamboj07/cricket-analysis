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
docker compose up -d --build
```

`--build` matters: three services are built rather than pulled, because the
stock images lack packages the pipeline needs -- Python in the nodemanager,
where YARN executes the streaming mapper and reducer, and numpy in the Spark
images, which `pyspark.ml` imports. See `docker/`.

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

`spark-submit` lives at `/spark/bin/spark-submit` (the image leaves `SPARK_HOME`
unset, so it is not on `PATH`), and it needs `fs.defaultFS` passed explicitly --
the Spark image ships no `core-site.xml`, so bare `hdfs:///` paths fail with
"Incomplete HDFS URI, no host". Both are folded into `SUBMIT` below.

```bash
SUBMIT="/spark/bin/spark-submit --driver-memory 6g \
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
  --py-files /workspace/etl/schema_utils.py"

# 1. raw JSON -> HDFS
# Classification and upload run in different containers: namenode has the hdfs
# CLI but only Python 3.5 (too old for this script's f-strings), and
# spark-master has Python 3.7 but no hdfs CLI.
docker exec namenode     bash    /workspace/hdfs_scripts/00_init_hdfs_dirs.sh
docker exec spark-master python3 /workspace/hdfs_scripts/01_classify_and_upload.py --dry-run
docker exec namenode     bash    /workspace/hdfs_scripts/01b_upload_from_manifest.sh
docker exec namenode     bash    /workspace/hdfs_scripts/02_verify_upload.sh

# 2. flatten to Parquet (+ the TSV feed the MapReduce job consumes)
docker exec spark-master $SUBMIT /workspace/etl/flatten_matches.py

# 3. bowler economy by phase (Hadoop Streaming)
bash mapreduce/run_job.sh
# 3b. Verify the MapReduce output against an independent Spark recomputation (reads the Parquet,
# not the TSV; classifies phases and aggregates with different code). Checks every row on runs,
# legal balls, wickets, dot balls and the derived columns. Output has 11 columns: the last two are
# dot_balls and dot_ball_pct (a legal ball on which the bowler is charged no runs; 0-100).
docker exec spark-master $SUBMIT /workspace/mapreduce/verify_with_spark.py

# 4. Hive external tables
bash hive/run_all.sh

# 5. features
docker exec spark-master $SUBMIT /workspace/features/match_state_features.py
docker exec spark-master $SUBMIT /workspace/features/player_form_features.py
docker exec spark-master $SUBMIT /workspace/features/build_training_table.py

# 5b. Opponent-performance features (batter vs opponent, bowler vs opponent, team head-to-head).
# Optional: the production model does NOT use them (see 6g). Built from earlier matches only
# and verified against a brute-force leakage check; the script fails loudly if that check fails.
docker exec spark-master $SUBMIT /workspace/features/opponent_features.py

# 5c. Venue par baselines: shrunk first-innings par run rate, chase win rate and depth of
# history per ground. A ground is keyed on venue AND city ("County Ground" alone is six grounds).
# Optional: not used by the production model. Leakage-checked like 5b.
docker exec spark-master $SUBMIT /workspace/features/venue_features.py

# 5d. Batter vs bowler head-to-head LOOKUP (not a model feature: too sparse, and it summarises the
# whole dataset so it would leak if used to predict a match inside it). Each row carries a sample-size
# tier, an expected strike rate from the two players' overall quality, and a z-score. Create the Hive
# table after the job. Not partitioned, so there is no MSCK trap.
docker exec spark-master $SUBMIT /workspace/features/matchup_lookup.py
docker exec hive-server beeline -u jdbc:hive2://localhost:10000 \
  -f /workspace/hive/10_create_matchup_lookup_table.hql

# 6. train, then score every historical chase
docker exec spark-master $SUBMIT /workspace/ml/train_win_probability.py
docker exec spark-master $SUBMIT /workspace/ml/score_all_matches.py

# The production model also takes is_female and is_international (derived from the training table's
# gender and team_type). If you change the model's features, re-run 6, 6c-6f in that order and
# re-export the dashboard snapshots. ml/verify_context_adoption.py compares the outputs with the
# copies saved in /data/backup_pre6b before the last such change.
#
# 6b. register the partitions written in step 6. hive/run_all.sh ran before the
# model existed, so its MSCK found nothing and win_probability_by_ball reports
# zero rows -- silently, with no error -- until this is run.
docker exec hive-server beeline -u jdbc:hive2://localhost:10000/cricket \
  -e "MSCK REPAIR TABLE win_probability_by_ball;"

# 6c. Win Probability Added: how much each delivery moved the chase (P_t - P_{t-1}).
# The Hive table must be created AFTER the job (same empty-partition trap as 6b),
# which is why 06_create_wpa_table.hql is not part of hive/run_all.sh.
docker exec spark-master $SUBMIT /workspace/ml/compute_wpa.py
docker exec hive-server beeline -u jdbc:hive2://localhost:10000 \
  -f /workspace/hive/06_create_wpa_table.hql

# 6d. Calibrate the probabilities (Platt vs Isotonic; Isotonic is saved and used).
# The fit set is the FIRST half of the held-out seasons and the judging set is the second
# half. Fitting on training seasons would learn nothing: the GBT is already calibrated
# there, and its over-optimism only shows on newer seasons. Output is clipped to
# [0.005, 0.995] so the display never claims certainty. As with 6c, create the Hive table
# after the job.
docker exec spark-master $SUBMIT /workspace/ml/calibrate_win_probability.py
docker exec hive-server beeline -u jdbc:hive2://localhost:10000 \
  -f /workspace/hive/07_create_win_probability_calibrated_table.hql

# 6e. Compare models: GBT vs Logistic Regression vs Random Forest on identical held-out
# rows, same preprocessing, hyperparameters fixed in advance and never tuned on the
# held-out seasons. Also saves the LR and RF models and checks the GBT reproduces.
docker exec spark-master $SUBMIT /workspace/ml/compare_models.py
docker exec hive-server beeline -u jdbc:hive2://localhost:10000 \
  -f /workspace/hive/08_create_model_comparison_table.hql

# 6f. Feature importance: built-in importances (GBT, Random Forest, Logistic Regression
# coefficients) cross-checked against group-wise permutation importance on the held-out
# seasons. Many features are near-duplicates, so feature-level tree importances alone mislead.
docker exec spark-master $SUBMIT /workspace/ml/feature_importance.py
docker exec hive-server beeline -u jdbc:hive2://localhost:10000 \
  -f /workspace/hive/09_create_feature_importance_table.hql

# 6g. Do the opponent features help? Retrains GBT and Logistic Regression with and without
# them on identical data, with a match-level 95% interval on the difference.
docker exec spark-master $SUBMIT /workspace/ml/evaluate_opponent_features.py

# 6h. Do the venue features help, is it the venue or just who is playing, and does a plain
# is_female / is_international context pair do more? Includes a placebo (venue data shuffled across matches).
docker exec spark-master $SUBMIT /workspace/ml/evaluate_venue_features.py
docker exec spark-master $SUBMIT /workspace/ml/evaluate_context_features.py

# 7. dashboard
pip install -r dashboard/requirements-dashboard.txt
bash dashboard/export_snapshots.sh   # when PyHive will not build (see below)
streamlit run dashboard/app.py
```

On Windows `pyhive[hive]` pulls in `sasl`, which needs the MSVC build tools; when
it fails the dashboard falls back to the TSV snapshots written by
`export_snapshots.sh` and says so in the sidebar. Run the export first so the
fallback has something to serve.

The dashboard has five views: bowler economy by phase (rank by economy or dot-ball %), match explorer
(win-probability curve, win-probability-added per ball and the five biggest swings), batter vs bowler
(the `matchup_lookup` table, with reliability tiers), model scorecard (model comparison, reliability
chart of raw vs calibrated probabilities, permutation importance) and the what-if scorer. The
snapshot export is cut down to what is displayed: pairs under 12 balls are not exported, and the
reliability table is aggregated in Hive before export.

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
| `/data/models/win_probability_isotonic` | stage 6d |
| `/data/processed/win_probability_calibrated` | stage 6d |
| `/data/models/win_probability_logreg`, `/data/models/win_probability_rf` | stage 6e |
| `/data/processed/model_comparison` | stage 6e |
| `/data/processed/feature_importance` | stage 6f |
| `/data/processed/features/opponent/{batter,bowler,team}` | stage 5b |
| `/data/processed/opponent_feature_evaluation` | stage 6g |
| `/data/processed/features/venue` | stage 5c |
| `/data/processed/matchup_lookup` | stage 5d |
| `/data/processed/venue_feature_evaluation`, `/data/processed/venue_feature_comparisons` | stage 6h |
| `/data/processed/win_probability_by_ball` | stage 6 |
| `/data/processed/wpa_by_ball` | stage 6c |

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
- **Momentum features go through the `Imputer`.** `runs_last_12/30` and
  `wickets_last_12/30` are never null in the training table, but they are
  unknowable for a hypothetical state in the dashboard's what-if view. Routing
  them through the imputer alongside the form features is a no-op for training
  (identical metrics) and is what makes single-state scoring possible at all --
  otherwise the assembler rejects the row with "Encountered null while
  assembling a row".
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
