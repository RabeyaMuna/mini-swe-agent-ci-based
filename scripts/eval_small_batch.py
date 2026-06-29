#!/usr/bin/env python3
"""
Small batch evaluation for camel/flower issues.

Processes 5 issues at a time to avoid timeout issues.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_SEED = PROJECT_ROOT / "data" / "trs" / "memory_seed_issues.json"
LOG_DETAILS = PROJECT_ROOT / "data" / "trs" / "log_details.json"
MEMORY_ROOT = PROJECT_ROOT / "data" / "trs"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "eval_small_batch"


def load_json(path: Path):
    if not path.exists():
        return {} if "log_details" in path.name else []
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def main():
    # Get memory issue IDs
    memory_ids = set()

    # From seed issues
    for issue in load_json(MEMORY_SEED):
        if issue_id := str(issue.get("id", "")).strip():
            memory_ids.add(issue_id)

    # From log details
    for issue_id in load_json(LOG_DETAILS).keys():
        memory_ids.add(str(issue_id).strip())

    print(f"Memory issues: {len(memory_ids)}")

    # Load HuggingFace dataset
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install datasets: pip install datasets")
        sys.exit(1)

    print("Loading dataset...")
    ds = load_dataset("ci-benchmark-user/ci-repair-bench", split="train")

    # Filter camel/flower, exclude memory
    eval_issues = []
    for item in ds:
        repo = str(item.get("repo_name", "")).lower()
        issue_id = str(item.get("id", "")).strip()

        if (repo in ["camel", "flower"] or "camel" in repo or "flower" in repo):
            if issue_id and issue_id not in memory_ids:
                eval_issues.append(dict(item))

    print(f"Eval issues (camel/flower, excluding memory): {len(eval_issues)}")

    # Take small batch
    BATCH_SIZE = 5
    batch = eval_issues[:BATCH_SIZE]

    print(f"\nProcessing first {len(batch)} issues:")
    for i, issue in enumerate(batch, 1):
        print(f"  {i}. Issue {issue.get('id')} - {issue.get('repo_name')} - {issue.get('sha_fail', '')[:8]}")

    # Save dataset
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_path = OUTPUT_ROOT / f"batch_{timestamp}.json"
    write_json(dataset_path, batch)

    print(f"\nDataset saved: {dataset_path}")

    # Run
    cmd = [
        sys.executable, "-m", "minisweagent.run.benchmarks.cibench",
        "--dataset", str(dataset_path),
        "--split", "train",
        "--output", str(OUTPUT_ROOT / timestamp),
        "--workers", "1",
        "--memory-enabled",
        "--memory-root", str(MEMORY_ROOT),
        "--memory-ablation", "L1+L2+L3",
        "--memory-top-k", "3",
        "--no-save-memory",
    ]

    print(f"\nRunning: {' '.join(cmd[:4])} ...")
    print(f"Output: {OUTPUT_ROOT / timestamp}")
    print("="*80)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)

    print("\n" + "="*80)
    print("COMPLETE")
    print(f"Results: {OUTPUT_ROOT / timestamp}")


if __name__ == "__main__":
    main()
