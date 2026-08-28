#!/usr/bin/env python3
"""
build_memory_l1_l2_l3.py - Build L1, L2, L3 Memory from Decomposed Issues
==========================================================================

Takes decomposed CI issues and builds three-level memory abstraction:
- L1: Concrete problem-level failures
- L2: Repair strategies (sequential, dependency-aware)
- L3: Universal cross-repo fix patterns

Usage:
    # Build from decomposed issues
    python scripts/build_memory_l1_l2_l3.py \
        --decomposed data/decomposed_issues.json \
        --model glm5.2 \
        --output-dir data/memory

    # Build only L1 and L2 (skip L3)
    python scripts/build_memory_l1_l2_l3.py \
        --decomposed data/decomposed_issues.json \
        --model glm5.2 \
        --skip-l3

    # Limit to specific issues
    python scripts/build_memory_l1_l2_l3.py \
        --decomposed data/decomposed_issues.json \
        --model glm5.2 \
        --issue-ids 102,410,500
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from build_memory import build_l2_memory, build_l3_memory
from utilities.llm_model import LitellmModel
from utilities.model_registry import configure_model_environment


def load_decomposed_issues(path: Path) -> List[Dict[str, Any]]:
    """
    Load decomposed issues from JSON.

    Supports:
    1. Flat file: decomposed_issues.json (list of issues)
    2. Directory structure: {repo}/{sha_fail}/decomposed_issue.json
    """
    if path.is_file():
        # Flat file format
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "issues" in data:
            return data["issues"]
        else:
            raise ValueError(f"Unexpected format in {path}")

    elif path.is_dir():
        # Directory structure: load all decomposed_issue.json files
        issues = []
        for repo_dir in path.iterdir():
            if not repo_dir.is_dir():
                continue
            for sha_dir in repo_dir.iterdir():
                if not sha_dir.is_dir():
                    continue
                issue_file = sha_dir / "decomposed_issue.json"
                if issue_file.exists():
                    with open(issue_file, "r") as f:
                        issues.append(json.load(f))

        return issues

    else:
        raise ValueError(f"Path does not exist: {path}")


def build_l1_from_decomposed(decomposed_issue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert decomposed issue to L1 memory format.

    Decomposed issue has:
    - issue_id, repo, workflow
    - changed_files
    - decomposed_problems: [{problem_id, verification_cmd, failure_type, problem, root_cause, fix_strategy, files, enabled}]
    """

    issue_id = str(decomposed_issue.get("issue_id") or decomposed_issue.get("original_issue_id", ""))
    repo = decomposed_issue.get("repo", "")
    workflow_path = decomposed_issue.get("workflow_path") or decomposed_issue.get("workflow", "")
    changed_files = decomposed_issue.get("changed_files", [])
    problems = decomposed_issue.get("decomposed_problems") or decomposed_issue.get("problems", [])

    return {
        "issue_id": issue_id,
        "repo": repo,
        "workflow_path": workflow_path,
        "changed_files": changed_files,
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build L1, L2, L3 memory from decomposed CI issues"
    )
    parser.add_argument(
        "--decomposed",
        required=True,
        help="Path to decomposed_issues.json",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model for L2/L3 generation (e.g., glm5.2, minimax2.5)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/memory",
        help="Output directory for L1/L2/L3 files",
    )
    parser.add_argument(
        "--skip-l3",
        action="store_true",
        help="Skip L3 generation (only build L1 and L2)",
    )
    parser.add_argument(
        "--issue-ids",
        help="Comma-separated issue IDs to process (optional filter)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of issues to process",
    )

    args = parser.parse_args()

    # Initialize model
    print(f"\n{'=' * 80}")
    args.model = configure_model_environment(args.model) or args.model
    print(f"Initializing LLM: {args.model}")
    print(f"{'=' * 80}\n")

    llm = LitellmModel(model_name=args.model)

    # Load decomposed issues
    decomposed_path = Path(args.decomposed)
    if not decomposed_path.exists():
        print(f"ERROR: Decomposed file not found: {decomposed_path}")
        return 1

    print(f"Loading decomposed issues from: {decomposed_path}")
    decomposed_issues = load_decomposed_issues(decomposed_path)
    print(f"Loaded {len(decomposed_issues)} decomposed issues")

    # Filter by issue IDs if specified
    if args.issue_ids:
        filter_ids = set(args.issue_ids.split(","))
        decomposed_issues = [
            issue
            for issue in decomposed_issues
            if str(issue.get("issue_id") or issue.get("original_issue_id", "")) in filter_ids
        ]
        print(f"Filtered to {len(decomposed_issues)} issues: {filter_ids}")

    # Limit if requested
    if args.limit:
        decomposed_issues = decomposed_issues[: args.limit]
        print(f"Limited to first {args.limit} issues")

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each issue
    l1_records = []
    l2_records = []
    l3_records = []

    for i, decomposed_issue in enumerate(decomposed_issues, 1):
        issue_id = str(decomposed_issue.get("issue_id") or decomposed_issue.get("original_issue_id", ""))
        repo = decomposed_issue.get("repo", "unknown")

        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(decomposed_issues)}] Processing Issue {issue_id} ({repo})")
        print(f"{'=' * 80}")

        try:
            # Build L1
            print("Building L1 (concrete problems)...")
            l1_memory = build_l1_from_decomposed(decomposed_issue)

            # CRITICAL: Skip empty decompositions (0 problems) - don't save to memory
            num_problems = len(l1_memory.get("problems", []))
            if num_problems == 0:
                print(f"SKIP Issue {issue_id} has 0 problems - not saving to memory files")
                continue

            l1_records.append(l1_memory)
            print(f"OK L1 built for issue {issue_id} ({num_problems} problems)")

            # Build L2
            print("Building L2 (repair strategies)...")
            l2_memory = build_l2_memory(l1_memory=l1_memory, llm=llm)
            l2_records.append(l2_memory)
            print(f"OK L2 built for issue {issue_id}")

            # Build L3 (if not skipped)
            if not args.skip_l3:
                print("Building L3 (universal patterns)...")
                l3_memory = build_l3_memory(
                    l1_memory=l1_memory, l2_memory=l2_memory, llm=llm
                )
                l3_records.append(l3_memory)
                print(f"OK L3 built for issue {issue_id}")

            print(f"OK Issue {issue_id} complete")

        except Exception as e:
            print(f"FAIL ERROR processing issue {issue_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save aggregated files (append to existing)
    print(f"\n{'=' * 80}")
    print("Saving aggregated memory files...")
    print(f"{'=' * 80}")

    # Load existing data and append
    l1_file = output_dir / "failure_memory.json"
    existing_l1 = []
    if l1_file.exists():
        with open(l1_file) as f:
            existing_l1 = json.load(f)
            if not isinstance(existing_l1, list):
                existing_l1 = [existing_l1]

    # Append new records
    existing_l1.extend(l1_records)
    with open(l1_file, "w") as f:
        json.dump(existing_l1, f, indent=2)
    print(f"OK L1 (failure_memory): {l1_file} ({len(l1_records)} added)")

    # L2 - repo memory
    l2_file = output_dir / "repo_memory.json"
    existing_l2 = []
    if l2_file.exists():
        with open(l2_file) as f:
            existing_l2 = json.load(f)
            if not isinstance(existing_l2, list):
                existing_l2 = [existing_l2]

    existing_l2.extend(l2_records)
    with open(l2_file, "w") as f:
        json.dump(existing_l2, f, indent=2)
    print(f"OK L2 (repo_memory): {l2_file} ({len(l2_records)} added)")

    # L3 - cross memory (universal patterns)
    if not args.skip_l3:
        l3_file = output_dir / "cross_memory.json"
        existing_l3 = []
        if l3_file.exists():
            with open(l3_file) as f:
                existing_l3 = json.load(f)
                if not isinstance(existing_l3, list):
                    existing_l3 = [existing_l3]

        # Flatten L3 patterns from all issues
        all_patterns = []
        for l3_record in l3_records:
            patterns = l3_record.get("patterns", [])
            all_patterns.extend(patterns)

        existing_l3.extend(all_patterns)
        with open(l3_file, "w") as f:
            json.dump(existing_l3, f, indent=2)
        print(f"OK L3 (cross_memory): {l3_file} ({len(all_patterns)} patterns added)")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total issues processed: {len(l1_records)}")
    print(f"L1 records: {len(l1_records)}")
    print(f"L2 records: {len(l2_records)}")
    if not args.skip_l3:
        print(f"L3 records: {len(l3_records)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
