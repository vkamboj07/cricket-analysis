# pyspark.ml imports numpy at module load, so every ML stage fails on the stock
# image with "ModuleNotFoundError: No module named 'numpy'".
#
# The image is Alpine (musl libc), so pip's manylinux wheels do not apply and a
# pip install would have to compile numpy from source. apk has a prebuilt one.
FROM bde2020/spark-master:3.1.1-hadoop3.2

RUN apk add --no-cache py3-numpy \
 && python3 -c "import numpy; print('numpy', numpy.__version__)"
