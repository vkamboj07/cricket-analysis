# Same numpy requirement as the master: needed as soon as a job is submitted
# with --master spark://spark-master:7077 instead of running in local mode,
# because the Python worker processes import pyspark.ml too.
FROM bde2020/spark-worker:3.1.1-hadoop3.2

RUN apk add --no-cache py3-numpy \
 && python3 -c "import numpy; print('numpy', numpy.__version__)"
