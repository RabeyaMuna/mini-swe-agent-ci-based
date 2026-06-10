#!/usr/bin/env python3
"""
Set up memory seed issues by id or sha_fail.

The input keys are resolved against a full benchmark dataset. The output is a
JSON list of full issue records suitable for memory reasoning/decomposition.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve memory seed issues by id or sha_fail")
    parser.add_argument("--ids", help="Comma-separated issue IDs")
    parser.add_argument("--sha-fails", help="Comma-separated sha_fail values")
    parser.add_argument("--repo", help="Optional repo filter")
    parser.add_argument(
        "--dataset",
        default="ci-benchmark-user/ci-repair-bench",
        help="Full benchmark dataset JSON path or HuggingFace dataset name",
    )
    parser.add_argument(
        "--output",
        default="data/trs/memory_seed_resolved.json",
        help="Output full issue records for memory building",
    )
    args = parser.parse_args()

    refs = []
    for value in (args.ids or "").split(","):
        value = value.strip()
        if value:
            refs.append({"id": value})
    for value in (args.sha_fails or "").split(","):
        value = value.strip()
        if value:
            refs.append({"sha_fail": value})

    if not refs:
        print("ERROR: provide --ids and/or --sha-fails")
        return 1

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(refs, fh, indent=2)
        ref_path = fh.name

    cmd = [
        sys.executable,
        "scripts/resolve_memory_seed_issues.py",
        "--dataset",
        args.dataset,
        "--seed-file",
        ref_path,
        "--output",
        args.output,
    ]
    if args.repo:
        cmd.extend(["--repo", args.repo])

    try:
        result = subprocess.run(cmd, check=False)
    finally:
        Path(ref_path).unlink(missing_ok=True)

    if result.returncode != 0:
        return result.returncode

    print("\nNext step:")
    print(
        "  ./scripts/build_memory_pipeline_cheap.sh "
        f"--seed-file {args.output} --dataset {args.dataset}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
