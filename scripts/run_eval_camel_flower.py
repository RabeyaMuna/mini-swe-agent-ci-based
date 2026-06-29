#!/usr/bin/env python3
"""
Run evaluation on camel and flower issues NOT in memory.

This script:
1. Loads memory_seed_issues.json (issues used to build memory)
2. Loads all camel/flower issues from HuggingFace dataset
3. Excludes known memory issues
4. Runs evaluation on remaining issues
5. Has timeout per problem to prevent getting stuck
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_SEED = PROJECT_ROOT / "data" / "trs" / "memory_seed_issues.json"
LOG_DETAILS = PROJECT_ROOT / "data" / "trs" / "log_details.json"
MEMORY_ROOT = PROJECT_ROOT / "data" / "trs"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "eval_camel_flower"


def load_json(path: Path) -> Any:
    """Load JSON file."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_memory_issue_ids() -> set[str]:
    """Get issue IDs that are in memory (should be excluded from eval)."""
    memory_ids = set()

    # From memory_seed_issues.json
    seed_issues = load_json(MEMORY_SEED)
    for issue in seed_issues:
        issue_id = str(issue.get("id", "")).strip()
        if issue_id:
            memory_ids.add(issue_id)

    # From log_details.json
    log_details = load_json(LOG_DETAILS)
    for issue_id in log_details.keys():
        memory_ids.add(str(issue_id).strip())

    return memory_ids


def get_all_camel_flower_issues() -> list[dict[str, Any]]:
    """Load all camel and flower issues from HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets package not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading HuggingFace dataset ci-benchmark-user/ci-repair-bench...")
    ds = load_dataset("ci-benchmark-user/ci-repair-bench", split="train")

    all_issues = []
    for item in ds:
        repo = str(item.get("repo_name", "")).lower()
        if repo in ["camel", "flower"] or "camel" in repo or "flower" in repo:
            all_issues.append(dict(item))

    return all_issues


def main():
    parser = argparse.ArgumentParser(description="Evaluate on camel/flower issues NOT in memory")
    parser.add_argument("--max-issues", type=int, default=None, help="Max issues to evaluate")
    parser.add_argument("--timeout-per-problem", type=int, default=600,
                       help="Timeout per problem in seconds (default: 10 min)")
    parser.add_argument("--ablation", type=str, default="L1+L2+L3",
                       choices=["L1", "L1+L2", "L1+L2+L3"],
                       help="Memory ablation level")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without running")
    args = parser.parse_args()

    print("="*80)
    print("CAMEL/FLOWER EVALUATION (Excluding Memory Issues)")
    print("="*80)

    # Get memory issue IDs
    memory_ids = get_memory_issue_ids()
    print(f"Memory issue IDs: {len(memory_ids)}")
    print(f"Sample: {list(sorted(memory_ids))[:10]}")
    print()

    # Get all camel/flower issues
    all_issues = get_all_camel_flower_issues()
    print(f"Total camel/flower issues in dataset: {len(all_issues)}")

    # Filter out memory issues
    eval_issues = []
    for issue in all_issues:
        issue_id = str(issue.get("id", "")).strip()
        if issue_id and issue_id not in memory_ids:
            eval_issues.append(issue)

    print(f"Eval issues (excluding memory): {len(eval_issues)}")

    if args.max_issues:
        eval_issues = eval_issues[:args.max_issues]
        print(f"Limited to first {args.max_issues} issues")

    # Print selected issues
    print()
    print("Selected eval issues:")
    for i, issue in enumerate(eval_issues[:20], 1):
        issue_id = issue.get("id", "")
        repo = issue.get("repo_name", "")
        sha = issue.get("sha_fail", "")[:8]
        print(f"  {i}. Issue {issue_id} - {repo} - {sha}")

    if len(eval_issues) > 20:
        print(f"  ... and {len(eval_issues) - 20} more")
    print()

    # Create eval dataset file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_dataset_path = OUTPUT_ROOT / f"eval_dataset_{timestamp}.json"
    write_json(eval_dataset_path, eval_issues)
    print(f"Eval dataset saved to: {eval_dataset_path}")
    print()

    # Build command
    cmd = [
        sys.executable,
        "-m", "minisweagent.run.benchmarks.cibench",
        "--dataset", str(eval_dataset_path),
        "--split", "train",
        "--output", str(OUTPUT_ROOT / timestamp),
        "--workers", "1",
        "--memory-enabled",
        "--memory-root", str(MEMORY_ROOT),
        "--memory-ablation", args.ablation,
        "--memory-top-k", "3",
        "--no-save-memory",
        "--timeout", str(args.timeout_per_problem),  # Timeout per problem!
    ]

    print("Command:")
    print("  " + " ".join(cmd))
    print()

    if args.dry_run:
        print("DRY RUN - not executing")
        return

    # Run
    print("="*80)
    print(f"Running evaluation with {args.ablation}")
    print("="*80)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)

    print()
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_ROOT / timestamp}")


if __name__ == "__main__":
    main()
