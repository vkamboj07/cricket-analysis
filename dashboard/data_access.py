"""Data access for the dashboard, with a Hive-first and CSV-fallback path.

PyHive pulls in a SASL dependency chain that needs a C toolchain on Windows.
When it cannot connect, the app falls back to CSV snapshots exported by
dashboard/export_snapshots.sh, and says so in the sidebar rather than
silently showing stale numbers.
"""
import os
import subprocess
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "data_cache"
HIVE_HOST = os.environ.get("HIVE_HOST", "localhost")
HIVE_PORT = int(os.environ.get("HIVE_PORT", "10000"))
HIVE_DATABASE = os.environ.get("HIVE_DATABASE", "cricket")


class HiveConnector:
    mode = "hive"
    label = f"Hive ({HIVE_HOST}:{HIVE_PORT})"

    def __init__(self):
        from pyhive import hive  # imported lazily so the fallback can work
        self._conn = hive.Connection(host=HIVE_HOST, port=HIVE_PORT,
                                     database=HIVE_DATABASE, auth="NONE")

    def query(self, sql, params=None):
        return pd.read_sql(sql, self._conn, params=params)


class CsvSnapshotConnector:
    mode = "csv"
    label = f"TSV snapshots ({CACHE_DIR.name}/)"

    def __init__(self):
        if not CACHE_DIR.is_dir():
            raise FileNotFoundError(
                f"{CACHE_DIR} not found - run dashboard/export_snapshots.sh")
        self._tables = {}  # the win-probability snapshot is 340 MB: parse each file once

    # Tab separated, not comma: beeline's csv2 writer disables quoting by
    # setting the quote char to NUL, then still wraps any value containing a
    # comma in NUL bytes -- which corrupted ~56% of the matches rows (venues
    # like "County Ground, New Road, Worcester"). The data contains no tabs.
    def table(self, name):
        if name not in self._tables:
            self._tables[name] = self._read(name)
        return self._tables[name]

    def _read(self, name):
        path = CACHE_DIR / f"{name}.tsv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - run dashboard/export_snapshots.sh")
        return pd.read_csv(path, sep="\t",
                           # match_id is a filename stem kept verbatim ("wi_201706"
                           # as well as "201706"), so it must stay a string: pandas
                           # otherwise infers int64 for the all-numeric tables and
                           # str for the mixed ones, and merges between them fail.
                           dtype={"match_id": str, "season": str},
                           low_memory=False)

    def query(self, sql, params=None):
        raise NotImplementedError(
            "CSV fallback serves whole tables; use table() instead of SQL")


def connect():
    """Return (connector, error). Falls back to CSV when Hive is unreachable."""
    try:
        return HiveConnector(), None
    except Exception as exc:  # noqa: BLE001 - any failure means fall back
        try:
            return CsvSnapshotConnector(), str(exc)
        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                f"no data source available. Hive: {exc}. CSV: {fallback_exc}")


def score_state_in_container(score, wickets, balls_remaining, target,
                             match_type, overs_cap, model_path,
                             gender="male", team_type="international",
                             container="spark-master", timeout=300):
    """Score a hypothetical state by running predict_live_state in the cluster.

    A local Spark session is the obvious approach, but Spark does not run on
    Java 18+ ("getSubject is not supported") and Windows additionally needs
    winutils.exe/hadoop.dll. The spark-master container already has a working
    Spark, the model, and HDFS, so shell out to it instead. Returns the
    probability as a float, or raises RuntimeError with the command output.
    """
    cmd = [
        "docker", "exec", container, "/spark/bin/spark-submit",
        "--driver-memory", "4g",
        "--conf", "spark.hadoop.fs.defaultFS=hdfs://namenode:9000",
        "/workspace/ml/predict_live_state.py",
        "--score", str(int(score)),
        "--wickets", str(int(wickets)),
        "--balls-remaining", str(int(balls_remaining)),
        "--target", str(int(target)),
        "--match-type", str(match_type),
        "--overs-cap", str(int(overs_cap)),
        "--gender", str(gender),
        "--team-type", str(team_type),
        "--model", str(model_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in (proc.stdout or "").splitlines():
        if "win probability for the chasing team:" in line:
            return float(line.rsplit(":", 1)[1])
    raise RuntimeError(
        (proc.stderr or proc.stdout or "no output").strip()[-800:])


def bowler_economy(conn, match_type=None, phase=None, min_balls=300):
    if conn.mode == "hive":
        where, params = ["legal_balls >= %(min_balls)s"], {"min_balls": int(min_balls)}
        if match_type:
            where.append("match_type = %(match_type)s")
            params["match_type"] = match_type
        if phase:
            where.append("phase = %(phase)s")
            params["phase"] = phase
        return conn.query(
            "SELECT bowler_id, match_type, phase, runs_conceded, legal_balls, "
            "overs_bowled, wickets, economy, balls_per_wicket, dot_balls, dot_ball_pct "
            f"FROM bowler_economy_by_phase WHERE {' AND '.join(where)} "
            "ORDER BY economy ASC LIMIT 500", params)

    df = conn.table("bowler_economy_by_phase")
    df = df[df["legal_balls"] >= min_balls]
    if match_type:
        df = df[df["match_type"] == match_type]
    if phase:
        df = df[df["phase"] == phase]
    return df.sort_values("economy").head(500)


def win_probability_curve(conn, match_id):
    if conn.mode == "hive":
        return conn.query(
            "SELECT w.ball_seq, w.balls_completed_in_innings, w.current_score, "
            "w.wickets_down, w.runs_required, w.required_run_rate, w.current_run_rate, "
            "w.batter, w.bowler, w.win_probability, w.batting_team, "
            "p.prev_win_probability, p.wpa "
            "FROM win_probability_by_ball w LEFT JOIN wpa_by_ball p "
            "ON w.match_id = p.match_id AND w.ball_seq = p.ball_seq "
            "WHERE w.match_id = %(match_id)s ORDER BY w.ball_seq",
            {"match_id": str(match_id)})
    df = conn.table("win_probability_by_ball")
    curve = df[df["match_id"].astype(str) == str(match_id)]
    wpa = conn.table("wpa_by_ball")
    wpa = wpa[wpa["match_id"].astype(str) == str(match_id)][
        ["ball_seq", "prev_win_probability", "wpa"]]
    return curve.merge(wpa, on="ball_seq", how="left").sort_values("ball_seq")


def available_matches(conn, limit=200):
    if conn.mode == "hive":
        return conn.query(
            "SELECT DISTINCT w.match_id, m.team1, m.team2, m.match_date, m.match_type "
            "FROM win_probability_by_ball w JOIN matches m ON w.match_id = m.match_id "
            f"ORDER BY m.match_date DESC LIMIT {int(limit)}")
    df = conn.table("win_probability_by_ball")
    ids = df[["match_id"]].drop_duplicates().head(limit)
    try:
        matches = conn.table("matches")
        return ids.merge(matches, on="match_id", how="left")
    except FileNotFoundError:
        return ids


def model_comparison(conn):
    if conn.mode == "hive":
        return conn.query("SELECT * FROM model_comparison")
    return conn.table("model_comparison")


def feature_importance(conn, method):
    """method: 'permutation', 'native' or 'native_group'."""
    if conn.mode == "hive":
        return conn.query("SELECT * FROM feature_importance WHERE method = %(m)s",
                          {"m": method})
    df = conn.table("feature_importance")
    return df[df["method"] == method]


def calibration_buckets(conn):
    """Reliability table on the held-out seasons: predicted vs actual, ten buckets."""
    if conn.mode == "hive":
        return conn.query(
            "SELECT LEAST(FLOOR(win_probability * 10), 9) AS bucket, COUNT(*) AS n, "
            "AVG(label_batting_team_wins) AS actual, AVG(win_probability) AS raw_pred, "
            "AVG(win_probability_calibrated) AS calibrated_pred "
            "FROM win_probability_calibrated WHERE `split` = 'evaluation' "
            "GROUP BY LEAST(FLOOR(win_probability * 10), 9) ORDER BY bucket")
    return conn.table("calibration_buckets")


def matchup_lookup(conn, fam, min_balls=12):
    """Batter-vs-bowler pairs for one format family ('ALL', 'T20' or 'ODI')."""
    if conn.mode == "hive":
        return conn.query(
            "SELECT * FROM matchup_lookup WHERE fam = %(fam)s AND balls >= %(mb)s",
            {"fam": fam, "mb": int(min_balls)})
    df = conn.table("matchup_lookup")
    return df[(df["fam"] == fam) & (df["balls"] >= min_balls)]
