#!/usr/bin/env python3
"""
Clean decomposed_issues.json by removing L1/L2/L3 pipeline artifacts.

Removes fields that should NOT be in decomposition cache:
- is_merged
- merged_from
- repair_sequence_index
- l1, l2, l3 sections
- Any other L1/L2/L3-specific fields

Usage:
    python scripts/clean_decomposed_cache.py data/trs/decomposed_issues.json

    # With backup
    python scripts/clean_decomposed_cache.py data/trs/decomposed_issues.json --backup
"""

import argparse
import json
import shutil
from pathlib import Path


def clean_problem(problem: dict) -> dict:
    """Remove L1/L2/L3 artifacts from a problem dict."""
    # Fields that should be in decomposition

    # Fields that should NOT be in decomposition (L1/L2/L3 artifacts)
    forbidden_fields = {
        "is_merged",
        "merged_from",
        "repair_sequence_index",
        "cluster_id",
        "merge_reason",
        "l1_description",
        "l2_step",
        "l3_analysis",
    }

    cleaned = {}
    removed_fields = []

    for key, value in problem.items():
        if key in forbidden_fields:
            removed_fields.append(key)
            continue
        cleaned[key] = value

    if removed_fields:
        print(f"      Removed fields: {', '.join(removed_fields)}")

    return cleaned


def clean_decomposed_issue(issue: dict) -> dict:
    """Remove L1/L2/L3 artifacts from entire issue."""
    cleaned = {}

    # Copy top-level allowed fields

    # Fields to remove (L1/L2/L3 sections)
    forbidden_top_fields = {
        "l1",
        "l2",
        "l3",
        "optimized_problems",
        "dependency_graph",
        "repair_trajectory",
    }

    removed_top = []
    for key, value in issue.items():
        if key in forbidden_top_fields:
            removed_top.append(key)
            continue

        # Clean problems array
        if key == "problems" and isinstance(value, list):
            cleaned[key] = [clean_problem(p) for p in value]
        else:
            cleaned[key] = value

    if removed_top:
        issue_id = issue.get("original_issue_id", "?")
        print(
            f"    Issue {issue_id}: Removed top-level fields: {', '.join(removed_top)}"
        )

    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Clean decomposed_issues.json by removing L1/L2/L3 artifacts"
    )
    parser.add_argument("input_file", type=Path, help="Path to decomposed_issues.json")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before cleaning (.bak file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without modifying file",
    )

    args = parser.parse_args()

    input_file = args.input_file

    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        return 1

    print(f"Loading: {input_file}")
    with open(input_file) as f:
        issues = json.load(f)

    print(f"Found {len(issues)} issues")

    # Clean each issue
    print("\nCleaning...")
    cleaned_issues = []
    total_problems_before = 0
    total_problems_after = 0

    for issue in issues:
        total_problems_before += len(issue.get("problems", []))
        cleaned = clean_decomposed_issue(issue)
        total_problems_after += len(cleaned.get("problems", []))
        cleaned_issues.append(cleaned)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total issues: {len(issues)}")
    print(f"Total problems before: {total_problems_before}")
    print(f"Total problems after: {total_problems_after}")
    print(f"Problems removed: {total_problems_before - total_problems_after}")

    # Calculate file size difference
    original_size = input_file.stat().st_size
    cleaned_json = json.dumps(cleaned_issues, indent=2)
    new_size = len(cleaned_json.encode("utf-8"))
    size_diff = original_size - new_size
    size_diff_pct = (size_diff / original_size) * 100 if original_size > 0 else 0

    print("\nFile size:")
    print(f"  Before: {original_size:,} bytes ({original_size / 1024 / 1024:.2f} MB)")
    print(f"  After:  {new_size:,} bytes ({new_size / 1024 / 1024:.2f} MB)")
    print(
        f"  Saved:  {size_diff:,} bytes ({size_diff / 1024 / 1024:.2f} MB, {size_diff_pct:.1f}%)"
    )

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN - No files modified")
        print(f"{'=' * 60}")
        return 0

    # Create backup if requested
    if args.backup:
        backup_file = input_file.with_suffix(input_file.suffix + ".bak")
        print(f"\nCreating backup: {backup_file}")
        shutil.copy2(input_file, backup_file)

    # Save cleaned version
    print(f"\nSaving cleaned version to: {input_file}")
    with open(input_file, "w") as f:
        f.write(cleaned_json)

    print(f"\n{'=' * 60}")
    print("✅ DONE - Cache cleaned successfully")
    print(f"{'=' * 60}")
    print("\nCleaned fields:")
    print("  ✓ Removed is_merged, merged_from, repair_sequence_index")
    print("  ✓ Removed l1, l2, l3 sections")
    print("  ✓ Kept only decomposition data")
    print("\nThe cache is now clean and ready to use!")

    return 0


if __name__ == "__main__":
    exit(main())
