#!/usr/bin/env python3
"""Hadoop Streaming reducer: aggregate bowler economy by phase.

Input arrives sorted by the 3-field key (bowler_id, match_type, phase).
Output: bowler_id, match_type, phase, runs, balls, overs, wickets, economy,
strike_rate, dot_balls, dot_ball_pct -- tab separated, ready for a Hive external
TEXTFILE table. The two dot-ball columns are appended at the END so the position
of every earlier column is unchanged. dot_ball_pct is a true percentage (0-100).
"""
import sys


def emit(key, runs, balls, wickets, dots):
    if key is None:
        return
    overs = balls / 6.0
    economy = runs / overs if overs > 0 else ""
    strike_rate = balls / wickets if wickets > 0 else ""
    dot_pct = 100.0 * dots / balls if balls > 0 else ""
    sys.stdout.write(
        "%s\t%d\t%d\t%.3f\t%d\t%s\t%s\t%d\t%s\n"
        % (key, runs, balls, overs, wickets,
           ("%.4f" % economy) if economy != "" else "",
           ("%.2f" % strike_rate) if strike_rate != "" else "",
           dots,
           ("%.2f" % dot_pct) if dot_pct != "" else "")
    )


def main():
    current_key = None
    runs = balls = wickets = dots = 0
    bad_lines = 0

    for line in sys.stdin:
        parts = line.rstrip("\n").split("\t")
        # Must match the mapper: key (3 fields) + runs, legal_ball, wicket, dot_ball.
        # A mapper/reducer mismatch would silently drop every line into bad_lines.
        if len(parts) != 7:
            bad_lines += 1
            continue

        key = "\t".join(parts[:3])
        try:
            r, b, w, d = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
        except ValueError:
            bad_lines += 1
            continue

        if key != current_key:
            emit(current_key, runs, balls, wickets, dots)
            current_key, runs, balls, wickets, dots = key, 0, 0, 0, 0

        runs += r
        balls += b
        wickets += w
        dots += d

    emit(current_key, runs, balls, wickets, dots)

    if bad_lines:
        sys.stderr.write(
            "reporter:counter:cricket,reducer_bad_lines,%d\n" % bad_lines)


if __name__ == "__main__":
    main()
