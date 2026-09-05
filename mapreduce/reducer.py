#!/usr/bin/env python3
"""Hadoop Streaming reducer: aggregate bowler economy by phase.

Input arrives sorted by the 3-field key (bowler_id, match_type, phase).
Output: bowler_id, match_type, phase, runs, balls, overs, wickets, economy,
strike_rate -- tab separated, ready for a Hive external TEXTFILE table.
"""
import sys


def emit(key, runs, balls, wickets):
    if key is None:
        return
    overs = balls / 6.0
    economy = runs / overs if overs > 0 else ""
    strike_rate = balls / wickets if wickets > 0 else ""
    sys.stdout.write(
        "%s\t%d\t%d\t%.3f\t%d\t%s\t%s\n"
        % (key, runs, balls, overs, wickets,
           ("%.4f" % economy) if economy != "" else "",
           ("%.2f" % strike_rate) if strike_rate != "" else "")
    )


def main():
    current_key = None
    runs = balls = wickets = 0
    bad_lines = 0

    for line in sys.stdin:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 6:
            bad_lines += 1
            continue

        key = "\t".join(parts[:3])
        try:
            r, b, w = int(parts[3]), int(parts[4]), int(parts[5])
        except ValueError:
            bad_lines += 1
            continue

        if key != current_key:
            emit(current_key, runs, balls, wickets)
            current_key, runs, balls, wickets = key, 0, 0, 0

        runs += r
        balls += b
        wickets += w

    emit(current_key, runs, balls, wickets)

    if bad_lines:
        sys.stderr.write(
            "reporter:counter:cricket,reducer_bad_lines,%d\n" % bad_lines)


if __name__ == "__main__":
    main()
