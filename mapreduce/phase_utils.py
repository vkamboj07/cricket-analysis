"""Phase bucketing for limited-overs cricket.

Pure stdlib: this module is shipped to every task node via `-files` and
imported by both the streaming mapper and reducer, which run under the
container's bare python3 (no Spark, no pandas).

Buckets are derived from the over number and match type rather than the
innings `powerplays` field, which is only populated on ~78% of matches.
"""

T20_TYPES = ("T20", "IT20")

# Column order of the TSV produced by etl/flatten_matches.py. The mapper
# parses positionally, so this is the contract between the two stages.
COLUMNS = [
    "match_id", "match_type", "overs_cap", "innings_num", "over",
    "bowler", "bowler_id", "runs_total", "extra_wides", "extra_noballs",
    "extra_byes", "extra_legbyes", "is_wicket",
]
IDX = {name: i for i, name in enumerate(COLUMNS)}


def get_phase(over, match_type):
    """Map a 0-indexed over number to powerplay / middle / death.

    T20: 0-5 powerplay, 6-14 middle, 15+ death.
    ODI: 0-9 powerplay, 10-39 middle, 40+ death.
    Rain-shortened matches keep the full-length boundaries -- an approximation
    that is documented rather than silently corrected.
    """
    if match_type in T20_TYPES:
        if over <= 5:
            return "powerplay"
        if over <= 14:
            return "middle"
        return "death"
    if over <= 9:
        return "powerplay"
    if over <= 39:
        return "middle"
    return "death"


def parse_line(line):
    """Split a TSV line into a dict, or return None if it is malformed."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) != len(COLUMNS):
        return None
    return dict(zip(COLUMNS, parts))


def to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_true(value):
    return str(value).strip().lower() == "true"
