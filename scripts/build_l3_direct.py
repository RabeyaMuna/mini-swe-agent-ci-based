#!/usr/bin/env python3
"""
Direct L3 builder - Reads from decomposed_issues.json, creates universal patterns.

NO complex L2 transformation needed - just extracts patterns directly!

Usage:
    python scripts/build_l3_direct.py \
        --decomposed data/trs/decomposed_issues.json \
        --output data/trs/cross_memory.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List


def extract_atomic_problems_from_decomposed(decomposed_issues: List[Dict]) -> List[Dict]:
    """
    Extract all atomic problems from decomposed issues.

    Each problem already has:
    - failure_type (use as issue_type)
    - validation_order, validation_cmd
    - problem, root_cause, how_fixed
    - affected_files
    """

    all_problems = []

    for issue in decomposed_issues:
        if "error" in issue:
            continue

        repo = issue.get("repo", "unknown")
        issue_id = issue.get("original_issue_id", "unknown")
        sha_fail = issue.get("sha_fail", "")

        problems = issue.get("problems") or issue.get("atomic_problems", [])

        for problem in problems:
            # Enrich with repo context
            enriched = {
                **problem,
                "repo": repo,
                "issue_id": issue_id,
                "sha_fail": sha_fail,
                "issue_type": problem.get("failure_type", "unknown")  # Add issue_type for grouping
            }
            all_problems.append(enriched)

    return all_problems


def group_problems_by_type(problems: List[Dict]) -> Dict[str, List[Dict]]:
    """Group problems by failure_type/issue_type."""

    groups = defaultdict(list)

    for problem in problems:
        issue_type = problem.get("issue_type") or problem.get("failure_type", "unknown")
        if issue_type == "unknown":
            continue

        groups[issue_type].append(problem)

    return dict(groups)


def create_l3_pattern(issue_type: str, problems: List[Dict]) -> Dict[str, Any]:
    """
    Create one L3 universal pattern from a group of similar problems.

    Extracts:
    - Common error patterns
    - Common fix strategies
    - Affected repos
    - Example problems
    """

    if not problems:
        return None

    # Get representative problem
    first = problems[0]

    # Collect validation commands
    validation_cmds = list(set(p.get("validation_cmd", "") for p in problems if p.get("validation_cmd")))
    primary_cmd = validation_cmds[0] if validation_cmds else ""

    # Collect all root causes
    root_causes = [p.get("root_cause", "") for p in problems if p.get("root_cause")]

    # Collect all fix strategies
    fix_strategies = [p.get("how_fixed", "") for p in problems if p.get("how_fixed")]

    # Collect affected repos
    repos = list(set(p.get("repo", "") for p in problems if p.get("repo")))

    # Build universal problem description
    if len(root_causes) > 1:
        # Multiple examples - generalize
        problem_desc = f"This validation failure occurs when {root_causes[0].split('.')[0].lower()}. "
        problem_desc += f"Seen across {len(problems)} cases in {len(repos)} repo(s)."
    else:
        problem_desc = root_causes[0] if root_causes else first.get("problem", "")

    # Build universal fix strategy
    if len(fix_strategies) > 1:
        fix_strategy = f"Fix approach (based on {len(problems)} examples): {fix_strategies[0]}"
    else:
        fix_strategy = fix_strategies[0] if fix_strategies else ""

    # Extract detection patterns from validation commands
    detection = {
        "validation_commands": validation_cmds,
        "error_indicators": [issue_type],
        "file_patterns": _extract_file_patterns(problems)
    }

    # Build L3 pattern
    pattern = {
        "pattern_id": _make_pattern_id(issue_type),
        "pattern_name": issue_type,
        "issue_type": issue_type,
        "validation_cmd": primary_cmd,
        "problem": problem_desc,
        "detection": detection,
        "universal_fix_strategy": fix_strategy,
        "examples_count": len(problems),
        "repos_affected": repos,
        "validation_orders": list(set(p.get("validation_order") for p in problems if p.get("validation_order"))),
        "example_problems": [
            {
                "repo": p.get("repo"),
                "issue_id": p.get("issue_id"),
                "files": _extract_files_from_problem(p)[:3]  # First 3 files as example
            }
            for p in problems[:3]  # First 3 problems as examples
        ]
    }

    return pattern


def _make_pattern_id(issue_type: str) -> str:
    """Create pattern ID from issue type."""
    return issue_type.lower().replace(" ", "_").replace("-", "_")


def _extract_file_patterns(problems: List[Dict]) -> List[str]:
    """Extract common file patterns from problems."""

    all_files = []
    for p in problems:
        all_files.extend(_extract_files_from_problem(p))

    # Get file extensions
    extensions = set()
    for f in all_files:
        if "." in f:
            ext = f.split(".")[-1]
            extensions.add(f"*.{ext}")

    return list(extensions)[:5]  # Top 5 patterns


def _extract_files_from_problem(problem: Dict) -> List[str]:
    """Extract file list from problem (handles multiple formats)."""

    affected_files = problem.get("affected_files", [])

    if isinstance(affected_files, dict):
        # Format: {"total_count": N, "files": [...]}
        return affected_files.get("files", [])
    elif isinstance(affected_files, list):
        # Format: ["file1.py", "file2.py"]
        return affected_files

    return []


def build_l3_patterns(decomposed_issues: List[Dict]) -> List[Dict]:
    """
    Build L3 universal patterns from decomposed issues.

    Pipeline:
    1. Extract all atomic problems
    2. Group by issue_type/failure_type
    3. Create one L3 pattern per group
    """

    print("\nBuilding L3 universal patterns from decomposed issues...")

    # Step 1: Extract problems
    all_problems = extract_atomic_problems_from_decomposed(decomposed_issues)
    print(f"  Extracted {len(all_problems)} atomic problems from {len(decomposed_issues)} issues")

    # Step 2: Group by type
    groups = group_problems_by_type(all_problems)
    print(f"  Grouped into {len(groups)} failure types")

    for issue_type, problems in groups.items():
        print(f"    - {issue_type}: {len(problems)} examples")

    # Step 3: Create L3 patterns
    l3_patterns = []
    for issue_type, problems in groups.items():
        pattern = create_l3_pattern(issue_type, problems)
        if pattern:
            l3_patterns.append(pattern)

    print(f"  Created {len(l3_patterns)} L3 universal patterns")

    return l3_patterns


def main():
    parser = argparse.ArgumentParser(
        description="Build L3 universal patterns directly from decomposed issues"
    )
    parser.add_argument(
        "--decomposed",
        default="data/trs/decomposed_issues.json",
        help="Path to decomposed issues"
    )
    parser.add_argument(
        "--output",
        default="data/trs/cross_memory.json",
        help="Output path for L3 patterns"
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

    # Build L3 patterns
    l3_patterns = build_l3_patterns(decomposed_issues)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(l3_patterns, f, indent=2)

    print(f"\n✓ Saved {len(l3_patterns)} L3 patterns to: {output_path}")

    # Summary
    print("\nL3 Patterns Created:")
    for pattern in l3_patterns:
        print(f"  - {pattern['pattern_name']}: {pattern['examples_count']} examples across {len(pattern['repos_affected'])} repos")

    return 0


if __name__ == "__main__":
    exit(main())
