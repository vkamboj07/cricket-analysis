# YARN runs the Hadoop Streaming mapper and reducer as `python3` inside task
# containers on the nodemanager, but the BDE2020 image ships no Python at all.
# Without this, every map task dies with:
#   PipeMapRed.waitOutputThreads(): subprocess failed with code 1
#
# Debian 9 (stretch) is EOL: its packages have moved to archive.debian.org and
# the release files are past their Valid-Until date, hence the rewrite and the
# Check-Valid-Until override. Stretch ships Python 3.5, which is enough --
# mapreduce/{mapper,reducer,phase_utils}.py deliberately avoid f-strings.
FROM bde2020/hadoop-nodemanager:2.0.0-hadoop3.2.1-java8

RUN sed -i 's|deb.debian.org|archive.debian.org|g; s|security.debian.org|archive.debian.org|g; /stretch-updates/d' /etc/apt/sources.list \
 && apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends python3 \
 && rm -rf /var/lib/apt/lists/* \
 && python3 --version
