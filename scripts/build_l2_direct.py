#!/usr/bin/env python3
"""
Direct L2 builder - Reads from decomposed_issues.json, creates clean L2 format.

NO complex transformations - uses YOUR agreed-upon clean structure!

Output format (per problem):
{
  "problem_id": 1,
  "verification_cmd": "...",
  "files": ["a.py"],
  "problem": {
    "what_wrong": "...",
    "root_cause": "..."
  },
  "fixes": [
    {
      "from": "old code",
      "to": "new code",
      "why_this_fix": "explanation"
    }
  ]
}

Usage:
    python scripts/build_l2_direct.py \
        --decomposed data/trs/decomposed_issues.json \
        --output data/trs/repo_memory.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def extract_files_from_problem(problem: Dict) -> List[str]:
    """Extract file list from problem (handles multiple formats)."""

    affected_files = problem.get("affected_files", [])

    if isinstance(affected_files, dict):
        # Format: {"total_count": N, "files": [...]}
        return affected_files.get("files", [])
    elif isinstance(affected_files, list):
        return affected_files

    return []


def extract_fixes_from_problem(problem: Dict) -> List[Dict[str, str]]:
    """
    Extract actual code changes (from/to) from problem.

    Looks in file_changes for change objects with from/to.
    """

    fixes = []

    file_changes = problem.get("file_changes", [])

    for fc in file_changes:
        if not isinstance(fc, dict):
            continue

        change = fc.get("change")
        if change and isinstance(change, dict):
            from_code = change.get("from", "")
            to_code = change.get("to", "")
            why = change.get("why", "")

            if from_code and to_code and from_code != "..." and to_code != "...":
                fix = {
                    "from": from_code,
                    "to": to_code
                }
                if why:
                    fix["why_this_fix"] = why

                fixes.append(fix)

    # If no real from/to found, create from how_fixed
    if not fixes:
        how_fixed = problem.get("how_fixed", "")
        if how_fixed:
            fixes.append({
                "from": "(see description)",
                "to": "(see description)",
                "why_this_fix": how_fixed
            })

    return fixes


def transform_problem_to_clean_l2(problem: Dict, issue_id: str, repo: str) -> Dict[str, Any]:
    """
    Transform decomposed problem to clean L2 format.

    YOUR agreed-upon structure - simple, flat, actionable.
    """

    problem_id = problem.get("problem_id", 0)
    validation_cmd = problem.get("validation_cmd", "")
    failure_type = problem.get("failure_type", "")

    # Extract files
    files = extract_files_from_problem(problem)

    # Extract problem description
    what_wrong = problem.get("problem", "")
    root_cause = problem.get("root_cause", "")

    # Extract fixes
    fixes = extract_fixes_from_problem(problem)

    # Build clean L2 problem
    clean_problem = {
        "problem_id": problem_id,
        "verification_cmd": validation_cmd,
        "issue_type": failure_type,  # For L3 compatibility
        "files": files,
        "problem": {
            "what_wrong": what_wrong,
            "root_cause": root_cause
        },
        "fixes": fixes
    }

    # Optional: Add dependencies if present
    if "depends_on" in problem:
        clean_problem["depends_on"] = problem["depends_on"]
    if "blocks" in problem:
        clean_problem["blocks"] = problem["blocks"]
    if "validation_order" in problem:
        clean_problem["validation_order"] = problem["validation_order"]

    return clean_problem


def build_clean_l2(decomposed_issues: List[Dict]) -> List[Dict]:
    """
    Build clean L2 memory from decomposed issues.

    One L2 entry per issue, containing clean atomic problems.
    """

    print("\nBuilding clean L2 memory from decomposed issues...")

    l2_memories = []

    for issue in decomposed_issues:
        if "error" in issue:
            continue

        issue_id = issue.get("original_issue_id", "unknown")
        repo = issue.get("repo", "unknown")
        sha_fail = issue.get("sha_fail", "")

        problems = issue.get("problems") or issue.get("atomic_problems", [])

        if not problems:
            continue

        # Transform each problem to clean format
        clean_problems = []
        for problem in problems:
            clean_problem = transform_problem_to_clean_l2(problem, issue_id, repo)
            clean_problems.append(clean_problem)

        # Build L2 entry
        l2_entry = {
            "issue_id": issue_id,
            "repo": repo,
            "sha_fail": sha_fail,
            "atomic_problems": clean_problems
        }

        l2_memories.append(l2_entry)

        print(f"  Processed issue {issue_id}: {len(clean_problems)} atomic problems")

    print(f"  Created {len(l2_memories)} L2 entries")

    return l2_memories


def main():
    parser = argparse.ArgumentParser(
        description="Build clean L2 memory directly from decomposed issues"
    )
    parser.add_argument(
        "--decomposed",
        default="data/trs/decomposed_issues.json",
        help="Path to decomposed issues"
    )
    parser.add_argument(
        "--output",
        default="data/trs/repo_memory_clean.json",
        help="Output path for clean L2 memory"
    )
    args = parser.parse_args()

    # Load decomposed issues
    decomposed_path = Path(args.decomposed)
    if not decomposed_path.exists():
        print(f"ERROR: Decomposed issues not found: {decomposed_path}")
        return 1

    print(f"Loading decomposed issues from: {decomposed_path}")
    with open(decomposed_path) as f:
        decomposed_issues = json.load(f)

    print(f"Loaded {len(decomposed_issues)} decomposed issues")

    # Build clean L2
    l2_memories = build_clean_l2(decomposed_issues)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(l2_memories, f, indent=2)

    print(f"\n✓ Saved {len(l2_memories)} L2 entries to: {output_path}")

    # Summary
    total_problems = sum(len(l2.get("atomic_problems", [])) for l2 in l2_memories)
    print(f"\nSummary:")
    print(f"  L2 entries: {len(l2_memories)}")
    print(f"  Total atomic problems: {total_problems}")
    print(f"  Format: Clean (YOUR agreed structure)")

    return 0


if __name__ == "__main__":
    exit(main())
