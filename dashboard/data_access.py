"""Data access for the dashboard, with a Hive-first and CSV-fallback path.

PyHive pulls in a SASL dependency chain that needs a C toolchain on Windows.
When it cannot connect, the app falls back to CSV snapshots exported by
dashboard/export_snapshots.sh, and says so in the sidebar rather than
silently showing stale numbers.
"""
import os
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
    label = f"CSV snapshots ({CACHE_DIR.name}/)"

    def __init__(self):
        if not CACHE_DIR.is_dir():
            raise FileNotFoundError(
                f"{CACHE_DIR} not found - run dashboard/export_snapshots.sh")

    def table(self, name):
        path = CACHE_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - run dashboard/export_snapshots.sh")
        return pd.read_csv(path)

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
            "overs_bowled, wickets, economy, balls_per_wicket "
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
            "SELECT ball_seq, balls_completed_in_innings, current_score, "
            "wickets_down, runs_required, required_run_rate, current_run_rate, "
            "batter, bowler, win_probability, batting_team "
            "FROM win_probability_by_ball WHERE match_id = %(match_id)s "
            "ORDER BY ball_seq", {"match_id": str(match_id)})
    df = conn.table("win_probability_by_ball")
    return (df[df["match_id"].astype(str) == str(match_id)]
            .sort_values("ball_seq"))


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
