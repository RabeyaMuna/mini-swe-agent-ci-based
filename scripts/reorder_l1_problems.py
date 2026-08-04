#!/usr/bin/env python3
"""
Reorder L1 problems by CI verification sequence and dependencies.

The original decomposed_issues.json has problems ordered by problem_id,
but they should be ordered by:
1. CI verification order (validation_order)
2. Dependencies (config before code)
3. Cascading relationships
"""

import json
from pathlib import Path
from typing import Any


def is_config_file(filepath: str) -> bool:
    """Check if file is a configuration file."""
    config_patterns = [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "package.json",
        "tsconfig.json",
        ".pre-commit-config.yaml",
        "Dockerfile",
        ".github/workflows",
    ]
    return any(pattern in filepath for pattern in config_patterns)


def problem_sort_key(problem: dict[str, Any]) -> tuple:
    """
    Generate sort key for problem ordering.

    Order:
    1. validation_order (CI pipeline sequence)
    2. Config files first (within same validation_order)
    3. Dependencies first (dependency_type != "")
    4. Non-cascading before cascading
    5. problem_id (original order as tiebreaker)
    """
    validation_order = problem.get("validation_order", 999)

    # Config file bonus (comes first)
    files = problem.get("affected_files", [])
    is_config = any(is_config_file(f) for f in files)
    config_rank = 0 if is_config else 1

    # Dependency rank (dependencies come first)
    has_dependency = bool(problem.get("dependency_type", ""))
    dependency_rank = 0 if has_dependency else 1

    # Cascading rank (non-cascading first)
    is_cascading = problem.get("is_cascading", False)
    cascading_rank = 1 if is_cascading else 0

    # Original problem_id as tiebreaker
    problem_id = problem.get("problem_id", 0)

    return (validation_order, config_rank, dependency_rank, cascading_rank, problem_id)


def reorder_issue_problems(issue: dict[str, Any]) -> dict[str, Any]:
    """Reorder problems in a single issue."""
    problems = issue.get("problems", [])

    # Sort by the composite key
    sorted_problems = sorted(problems, key=problem_sort_key)

    # Reassign problem_id to match new order
    for idx, problem in enumerate(sorted_problems, 1):
        problem["problem_id"] = idx

    issue["problems"] = sorted_problems
    return issue


def reorder_all_issues(input_file: Path, output_file: Path):
    """Reorder problems in all issues."""
    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        issues = json.load(f)

    print(f"Reordering {len(issues)} issues...")
    reordered_issues = []

    for issue in issues:
        issue_id = issue.get("original_issue_id", "unknown")
        original_count = len(issue.get("problems", []))

        reordered_issue = reorder_issue_problems(issue)
        reordered_count = len(reordered_issue.get("problems", []))

        # Verify no problems were lost
        assert original_count == reordered_count, f"Issue {issue_id}: problem count mismatch!"

        reordered_issues.append(reordered_issue)

        # Show example for first issue
        if len(reordered_issues) == 1:
            print(f"\nExample (Issue {issue_id}):")
            print("  Original order → Reordered order:")
            original_problems = issue.get("problems", [])
            for orig, new in zip(original_problems[:5], reordered_issue.get("problems", [])[:5]):
                orig_order = orig.get("validation_order", "?")
                new_order = new.get("validation_order", "?")
                print(f"    Problem {orig.get('problem_id')} (order={orig_order}) → "
                      f"Problem {new.get('problem_id')} (order={new_order})")

    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reordered_issues, f, indent=2, ensure_ascii=False)

    print(f"✓ Done! Reordered {len(reordered_issues)} issues.")


def main():
    """Main entry point."""
    data_dir = Path(__file__).parent.parent / "data" / "back_trs"

    input_file = data_dir / "decomposed_issues.json"
    output_file = data_dir / "decomposed_issues_reordered.json"

    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return

    reorder_all_issues(input_file, output_file)

    print(f"\nTo use the reordered data:")
    print(f"  mv {output_file} {input_file}")
    print(f"  # Or update your memory_root path to use decomposed_issues_reordered.json")


if __name__ == "__main__":
    main()
