#!/usr/bin/env python3
"""Classify Cricsheet match files by format and upload them to HDFS.

Runs inside the namenode container (needs the `hdfs` CLI on PATH):
    docker exec namenode python3 /workspace/hdfs_scripts/01_classify_and_upload.py

Reads /workspace/data/raw/all_json/*.json, splits them into limited-overs vs
test/multi-day buckets on info.match_type, and batch-uploads each bucket to
HDFS. The local source tree is never modified.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path("/workspace/data/raw/all_json")
MANIFEST = Path("/workspace/hdfs_scripts/upload_manifest.csv")
UNCLASSIFIED_LOG = Path("/workspace/docs/unclassified_match_types.txt")

HDFS_LIMITED_OVERS = "/data/raw/limited_overs"
HDFS_TESTS = "/data/raw/tests"

LIMITED_OVERS_TYPES = {"ODI", "ODM", "T20", "IT20"}
TEST_TYPES = {"Test", "MDM"}

# `hdfs dfs -put` accepts many sources per call; batching keeps us to ~50 JVM
# starts instead of 22k (which would take hours).
BATCH_SIZE = 400


def classify(match_type):
    if match_type in LIMITED_OVERS_TYPES:
        return "limited_overs"
    if match_type in TEST_TYPES:
        return "tests"
    return "unclassified"


def read_match_type(path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh).get("info", {}).get("match_type")


def scan(src_dir):
    """Return (rows, unclassified_types). match_id is the filename stem.

    The stem is kept verbatim rather than stripping the `wi_` prefix of older
    West Indies domestic files, because `wi_201706` and a numeric `201706`
    could otherwise collide into the same id.
    """
    rows = []
    unclassified_types = {}
    files = sorted(src_dir.glob("*.json"))
    total = len(files)
    print(f"scanning {total} json files in {src_dir}")

    for i, path in enumerate(files, 1):
        try:
            match_type = read_match_type(path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! unreadable {path.name}: {exc}", file=sys.stderr)
            rows.append({"filename": path.name, "match_id": path.stem,
                         "match_type": "", "bucket": "unreadable"})
            continue

        bucket = classify(match_type)
        if bucket == "unclassified":
            unclassified_types.setdefault(match_type, []).append(path.name)

        rows.append({"filename": path.name, "match_id": path.stem,
                     "match_type": match_type or "", "bucket": bucket})

        if i % 2000 == 0:
            print(f"  scanned {i}/{total}")

    return rows, unclassified_types


def upload(rows, bucket, hdfs_dest):
    paths = [str(SRC_DIR / r["filename"]) for r in rows if r["bucket"] == bucket]
    if not paths:
        print(f"nothing to upload for bucket {bucket}")
        return
    print(f"uploading {len(paths)} files -> {hdfs_dest}")
    for start in range(0, len(paths), BATCH_SIZE):
        batch = paths[start:start + BATCH_SIZE]
        # -f makes re-runs idempotent instead of failing on existing files.
        subprocess.run(["hdfs", "dfs", "-put", "-f", *batch, hdfs_dest], check=True)
        print(f"  {min(start + BATCH_SIZE, len(paths))}/{len(paths)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="classify and write the manifest, skip the HDFS upload")
    args = parser.parse_args()

    if not SRC_DIR.is_dir():
        sys.exit(f"source dir not found: {SRC_DIR} (is ./ bind-mounted to /workspace?)")

    rows, unclassified_types = scan(SRC_DIR)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "match_id", "match_type", "bucket"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest written: {MANIFEST}")

    counts = {}
    for r in rows:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
    for bucket, n in sorted(counts.items()):
        print(f"  {bucket}: {n}")

    if unclassified_types:
        UNCLASSIFIED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with UNCLASSIFIED_LOG.open("w", encoding="utf-8") as fh:
            for match_type, names in sorted(unclassified_types.items(), key=lambda kv: str(kv[0])):
                fh.write(f"{match_type}\t{len(names)}\t{names[0]}\n")
        print(f"! {len(unclassified_types)} unrecognised match_type value(s) "
              f"logged to {UNCLASSIFIED_LOG} - review before trusting counts")

    if args.dry_run:
        print("dry run: skipping upload")
        return

    upload(rows, "limited_overs", HDFS_LIMITED_OVERS)
    upload(rows, "tests", HDFS_TESTS)
    print("upload complete")


if __name__ == "__main__":
    main()
