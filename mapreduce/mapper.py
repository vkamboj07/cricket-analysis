#!/usr/bin/env python3
"""Hadoop Streaming mapper: bowler economy by phase.

Emits one record per legal-or-extra delivery:
    key   = bowler_id \t match_type \t phase
    value = runs_conceded \t legal_ball \t wicket \t dot_ball

Runs conceded are charged to the bowler for wides and no-balls but not for
byes/legbyes, which are not the bowler's fault. Wides and no-balls do not
count toward balls bowled.

A dot ball is a LEGAL ball on which the bowler is charged no runs, using the
same charging rule as runs_conceded. So wides and no-balls are never dots, a
wicket ball with no runs is a dot, and a bye or legbye ball is a dot (those runs
are not the bowler's). Some sources do not count bye balls as dots.
"""
import sys

from phase_utils import IDX, get_phase, is_true, parse_line, to_int

# Number of leading tab-separated fields that make up the key. Must match
# stream.num.map.output.key.fields in run_job.sh.
KEY_FIELDS = 3


def main():
    bad_lines = 0
    for line in sys.stdin:
        row = parse_line(line)
        if row is None:
            bad_lines += 1
            continue

        bowler_id = row["bowler_id"]
        if not bowler_id:
            bad_lines += 1
            continue

        over = to_int(row["over"], -1)
        if over < 0:
            bad_lines += 1
            continue

        phase = get_phase(over, row["match_type"])

        wides = to_int(row["extra_wides"])
        noballs = to_int(row["extra_noballs"])
        # Byes and legbyes are not charged to the bowler; wides and no-balls are.
        runs_conceded = (to_int(row["runs_total"])
                         - to_int(row["extra_byes"])
                         - to_int(row["extra_legbyes"]))

        legal_ball = 0 if (wides or noballs) else 1
        wicket = 1 if is_true(row["is_wicket"]) else 0
        dot_ball = 1 if (legal_ball and runs_conceded == 0) else 0

        sys.stdout.write(
            "%s\t%s\t%s\t%d\t%d\t%d\t%d\n"
            % (bowler_id, row["match_type"], phase, runs_conceded, legal_ball, wicket,
               dot_ball)
        )

    if bad_lines:
        # Counters go to stderr so they surface in the job report without
        # polluting the reducer's input stream.
        sys.stderr.write(
            "reporter:counter:cricket,mapper_bad_lines,%d\n" % bad_lines)


if __name__ == "__main__":
    main()
