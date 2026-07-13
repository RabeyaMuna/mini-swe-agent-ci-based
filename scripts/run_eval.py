#!/usr/bin/env python3
"""
Run evaluation on specific issue IDs with clean directory structure.

Results are saved to:
    results/{ablation}/{sha}/

Example:
    results/L1_L2_L3/5f98672f68c7f4acdf3e6f68fd79e66b56d0b529/

Usage:
    python3 scripts/run_eval.py --issue-ids 111,121,145
    python3 scripts/run_eval.py --issue-ids 111,121 --ablation L1+L2
    python3 scripts/run_eval.py --max-issues 10 --repos camel,flower
    python3 scripts/run_eval.py --exclude-memory --repos camel,flower
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_SEED = PROJECT_ROOT / "data" / "trs" / "memory_seed_issues.json"
LOG_DETAILS = PROJECT_ROOT / "data" / "trs" / "log_details.json"
MEMORY_ROOT = PROJECT_ROOT / "data" / "trs"
RESULTS_ROOT = PROJECT_ROOT / "results"


def load_json(path: Path) -> Any:
    """Load JSON file."""
    if not path.exists():
        return {} if "log_details" in path.name else []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_memory_issue_ids() -> set[str]:
    """Get issue IDs that are in memory."""
    memory_ids = set()

    # From memory seed
    for issue in load_json(MEMORY_SEED):
        if issue_id := str(issue.get("id", "")).strip():
            memory_ids.add(issue_id)

    # From log details
    for issue_id in load_json(LOG_DETAILS).keys():
        memory_ids.add(str(issue_id).strip())

    return memory_ids


def load_hf_dataset() -> list[dict[str, Any]]:
    """Load HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install datasets: pip install datasets")
        sys.exit(1)

    print("Loading HuggingFace dataset ci-benchmark-user/ci-repair-bench...")
    ds = load_dataset("ci-benchmark-user/ci-repair-bench", split="train")
    return [dict(item) for item in ds]


def normalize_repo(name: str) -> str:
    """Normalize repo name."""
    name = str(name or "").strip().lower()
    if "/" in name:
        return name.split("/")[-1]
    return name


def filter_issues(
    all_issues: list[dict[str, Any]],
    issue_ids: list[str] | None = None,
    repos: list[str] | None = None,
    exclude_memory: bool = False,
    max_issues: int | None = None,
) -> list[dict[str, Any]]:
    """Filter issues by criteria."""

    filtered = []
    memory_ids = get_memory_issue_ids() if exclude_memory else set()
    repo_set = {normalize_repo(r) for r in repos} if repos else None

    for issue in all_issues:
        issue_id = str(issue.get("id", "")).strip()
        repo = normalize_repo(issue.get("repo_name", ""))

        # Filter by issue ID
        if issue_ids and issue_id not in issue_ids:
            continue

        # Filter by repo
        if repo_set and repo not in repo_set:
            continue

        # Exclude memory issues
        if exclude_memory and issue_id in memory_ids:
            continue

        filtered.append(issue)

    # Limit count
    if max_issues:
        filtered = filtered[:max_issues]

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation with clean directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific issues
  python3 scripts/run_eval.py --issue-ids 111,121,145

  # Run with different ablation
  python3 scripts/run_eval.py --issue-ids 111 --ablation L1+L2

  # Run camel/flower, exclude memory issues
  python3 scripts/run_eval.py --repos camel,flower --exclude-memory --max-issues 10

  # Dry run
  python3 scripts/run_eval.py --issue-ids 111 --dry-run
        """
    )

    parser.add_argument(
        "--issue-ids",
        type=str,
        help="Comma-separated issue IDs (e.g., 111,121,145)"
    )
    parser.add_argument(
        "--issue-ids-file",
        type=str,
        help="Path to JSON file containing issue IDs (e.g., data/trs/eval_issue_ids.json)"
    )
    parser.add_argument(
        "--repos",
        type=str,
        help="Comma-separated repo names (e.g., camel,flower)"
    )
    parser.add_argument(
        "--exclude-memory",
        action="store_true",
        help="Exclude issues used to build memory"
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        help="Maximum number of issues to run"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="L1+L2+L3",
        choices=["BASELINE", "L1", "L1+L2", "L1+L2+L3"],
        help="Memory ablation level: BASELINE=no memory, L1=file-level, L1+L2=+sequences, L1+L2+L3=full (default: L1+L2+L3)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing"
    )

    args = parser.parse_args()

    print("="*80)
    print("MINI-SWE-AGENT EVALUATION")
    print("="*80)

    # Parse issue IDs
    issue_ids = None
    if args.issue_ids:
        issue_ids = [id.strip() for id in args.issue_ids.split(",")]
        print(f"Issue IDs: {', '.join(issue_ids)}")
    elif args.issue_ids_file:
        # Load from file
        import json
        from pathlib import Path
        ids_file = Path(args.issue_ids_file)
        if not ids_file.exists():
            print(f"ERROR: Issue IDs file not found: {ids_file}")
            return
        with open(ids_file) as f:
            issue_ids = json.load(f)
            # Convert to strings
            issue_ids = [str(id) for id in issue_ids]
        print(f"Loaded {len(issue_ids)} issue IDs from {ids_file}")
        print(f"Issue IDs: {', '.join(issue_ids[:10])}{'...' if len(issue_ids) > 10 else ''}")

    # Parse repos
    repos = None
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",")]
        print(f"Repos: {', '.join(repos)}")

    # Load all issues
    print()
    all_issues = load_hf_dataset()
    print(f"Total issues in dataset: {len(all_issues)}")

    # Filter issues
    selected = filter_issues(
        all_issues,
        issue_ids=issue_ids,
        repos=repos,
        exclude_memory=args.exclude_memory,
        max_issues=args.max_issues,
    )

    if not selected:
        print("\nERROR: No issues match the filters!")
        sys.exit(1)

    print(f"Selected issues: {len(selected)}")

    # Show memory exclusion
    if args.exclude_memory:
        memory_ids = get_memory_issue_ids()
        print(f"Excluded memory issues: {len(memory_ids)}")

    print()
    print("Issues to process:")
    for i, issue in enumerate(selected[:20], 1):
        issue_id = issue.get("id", "")
        repo = issue.get("repo_name", "")
        sha = issue.get("sha_fail", "")[:8]
        print(f"  {i}. Issue {issue_id} - {repo} - {sha}")

    if len(selected) > 20:
        print(f"  ... and {len(selected) - 20} more")

    # Create temporary dataset
    temp_dataset = PROJECT_ROOT / ".eval_temp_dataset.json"
    write_json(temp_dataset, selected)
    print(f"\nTemporary dataset: {temp_dataset}")

    # Build output directory
    ablation_dir = args.ablation.replace("+", "_")
    output_dir = RESULTS_ROOT / ablation_dir

    print(f"Output directory: {output_dir}/")
    print(f"  Example: {output_dir}/5f98672f68c7.../")

    # Build command
    cmd = [
        sys.executable,
        "-m", "minisweagent.run.benchmarks.cibench",
        "--dataset", str(temp_dataset),
        "--split", "train",
        "--output", str(output_dir),
        "--workers", str(args.workers),
    ]

    # Add memory flags ONLY if not baseline
    if args.ablation != "BASELINE":
        cmd.extend([
            "--memory-enabled",
            "--memory-root", str(MEMORY_ROOT),
            "--memory-ablation", args.ablation,
            "--memory-top-k", "3",
        ])

    cmd.append("--no-save-memory")

    print()
    print("Command:")
    print("  " + " ".join(cmd[:4]))
    print(f"  --dataset {temp_dataset.name}")
    print(f"  --output {output_dir}")
    print(f"  --memory-ablation {args.ablation}")
    print(f"  --workers {args.workers}")
    print()

    if args.dry_run:
        print("DRY RUN - Not executing")
        temp_dataset.unlink()
        return

    # Run
    print("="*80)
    print(f"RUNNING EVALUATION: {args.ablation}")
    print("="*80)
    print()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Command failed with code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    finally:
        # Cleanup temp dataset
        if temp_dataset.exists():
            temp_dataset.unlink()

    print()
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}/")
    print()
    print("Directory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── {selected[0].get('sha_fail', 'SHA')[:8]}.../")
    print(f"    │   ├── preds.json")
    print(f"    │   ├── run_instance.log")
    print(f"    │   └── ...")
    if len(selected) > 1:
        print(f"    ├── {selected[1].get('sha_fail', 'SHA')[:8]}.../")
        print(f"    └── ...")


if __name__ == "__main__":
    main()
