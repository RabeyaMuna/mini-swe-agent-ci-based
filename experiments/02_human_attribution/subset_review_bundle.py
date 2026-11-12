#!/usr/bin/env python3
"""Create a self-contained subset of an existing attribution review bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_csv(source: Path, destination: Path, selected: set[str]) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row["episode_id"] in selected]
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError(f"No CSV header in {source}")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", required=True, help="Comma-separated episode IDs")
    args = parser.parse_args()

    selected = {value.strip() for value in args.ids.split(",") if value.strip()}
    if not selected:
        raise ValueError("At least one episode ID is required")
    args.output.mkdir(parents=True, exist_ok=False)

    for filename in ("problems.jsonl", "hunks.jsonl"):
        rows = [
            row
            for row in read_jsonl(args.source / filename)
            if str(row["episode_id"]) in selected
        ]
        write_jsonl(args.output / filename, rows)

    filter_csv(
        args.source / "problem_hunk_pairs.csv",
        args.output / "problem_hunk_pairs.csv",
        selected,
    )

    manifest = json.loads((args.source / "sample_manifest.json").read_text())
    subset_manifest = [
        row for row in manifest if str(row["episode_id"]) in selected
    ]
    found = {str(row["episode_id"]) for row in subset_manifest}
    if found != selected:
        raise ValueError(f"Unknown episode IDs: {sorted(selected - found)}")
    (args.output / "sample_manifest.json").write_text(
        json.dumps(subset_manifest, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "source_bundle": str(args.source),
        "selection": "reviewer-friendly 30-repository audit",
        "episode_ids": sorted(selected, key=int),
        "episodes": len(selected),
        "repositories": len({row["repo"] for row in subset_manifest}),
        "single_problem": sum(row["ci_problem_records"] == 1 for row in subset_manifest),
        "multi_problem": sum(row["ci_problem_records"] > 1 for row in subset_manifest),
        "canonical_problems": sum(row["ci_problem_records"] for row in subset_manifest),
        "hunks": sum(row["hunks"] for row in subset_manifest),
        "candidate_pairs": sum(row["candidate_pairs"] for row in subset_manifest),
    }
    (args.output / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    raw_source = args.source / "raw_episodes_50.jsonl.gz"
    if raw_source.exists():
        with gzip.open(raw_source, "rt", encoding="utf-8") as handle:
            source_rows = [json.loads(line) for line in handle if line.strip()]
            raw_rows = [row for row in source_rows if str(row["id"]) in selected]
        with gzip.open(
            args.output / "raw_episodes_30.jsonl.gz", "wt", encoding="utf-8"
        ) as handle:
            for row in raw_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
