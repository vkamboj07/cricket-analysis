# Cricket Ball-by-Ball Big Data Analytics

Big data pipeline over the Cricsheet ball-by-ball dataset (22,500+ matches, ~10-11M deliveries)
using Hadoop HDFS, PySpark, MapReduce (Hadoop Streaming), Hive, and Spark MLlib.

## Pipeline
1. Raw JSON -> HDFS (`hdfs_scripts/`)
2. Flatten to ball-by-ball Parquet (`etl/`)
3. Bowler economy-by-phase MapReduce job (`mapreduce/`)
4. Hive external tables (`hive/`)
5. Match-state + rolling player-form features (`features/`)
6. Win-probability GBTClassifier (`ml/`)
7. Dashboard (`dashboard/`)

## Setup

Bring up the Hadoop/Spark/Hive stack:

```
docker compose up -d
```

- HDFS UI: http://localhost:9870
- YARN UI: http://localhost:8088
- Spark master UI: http://localhost:8080
- HiveServer2: `jdbc:hive2://localhost:10000`

The stack uses Hadoop 3.2.1 / Hive 2.3.2 / Spark 3.1.1 (BDE2020 images) as a
known-good combination. **Note:** `requirements.txt` pins `pyspark==3.5.1` for
local development (e.g. exploring Parquet output in notebooks with
`master("local[*]")`). If you `spark-submit` jobs to the dockerized
`spark-master` cluster, use a matching `pyspark==3.1.1` in that environment —
mismatched client/cluster Spark versions will fail.

First-time HDFS setup once the stack is up:

```
docker exec -it namenode hdfs dfs -mkdir -p /user/hive/warehouse
docker exec -it namenode hdfs dfs -chmod g+w /user/hive/warehouse
docker exec -it namenode hdfs dfs -mkdir -p /data/raw
docker exec -it namenode hdfs dfs -put -f /path/inside/container/*.json /data/raw/
```
