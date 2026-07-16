#!/usr/bin/env python3
"""
Add specific issues to the memory training set.

Usage:
    python scripts/add_issue_to_memory.py 128
    python scripts/add_issue_to_memory.py 128 130 135
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def add_issues_to_memory(issue_ids):
    """Add issue IDs to memory_issue_ids.json."""

    memory_ids_file = PROJECT_ROOT / "data" / "trs" / "memory_issue_ids.json"

    # Load current memory IDs
    if memory_ids_file.exists():
        with open(memory_ids_file) as f:
            memory_ids = json.load(f)
    else:
        print(f"Creating new {memory_ids_file}")
        memory_ids = []

    print(f"Current memory size: {len(memory_ids)} issues")

    # Add new issues
    added = []
    already_exists = []

    for issue_id in issue_ids:
        issue_id = str(issue_id).strip()
        if issue_id in memory_ids:
            already_exists.append(issue_id)
        else:
            memory_ids.append(issue_id)
            added.append(issue_id)

    if added:
        # Save updated memory IDs
        memory_ids_file.parent.mkdir(parents=True, exist_ok=True)
        with open(memory_ids_file, 'w') as f:
            json.dump(memory_ids, f, indent=2)

        print(f"\n[OK] Added {len(added)} issue(s) to memory:")
        for issue_id in added:
            print(f"   - Issue {issue_id}")

        print(f"\nNew memory size: {len(memory_ids)} issues")

    if already_exists:
        print(f"\n[WARN] {len(already_exists)} issue(s) already in memory:")
        for issue_id in already_exists:
            print(f"   - Issue {issue_id}")

    if added:
        print("\nNext steps:")
        print("1. Check if these issues are in filtered_issues.jsonl")
        print("2. Check if they're already decomposed (decomposed_issues.json)")
        print("3. If not decomposed, run:")
        print(f"   python scripts/decompose_ci_failure.py --issue-id {' '.join(added)}")
        print("4. Regenerate L1/L2/L3 memory:")
        print("   python scripts/generate_memory.py")

    return added, already_exists


def verify_issues_exist(issue_ids):
    """Check if issues exist in filtered_issues.jsonl."""

    issues_file = PROJECT_ROOT / "data" / "trs" / "filtered_issues.jsonl"

    if not issues_file.exists():
        print(f"\n[WARN] {issues_file} not found, skipping verification")
        return {}

    found = {}
    with open(issues_file) as f:
        for line in f:
            try:
                issue = json.loads(line)
                issue_id = str(issue.get('id', ''))
                if issue_id in issue_ids:
                    found[issue_id] = issue
            except json.JSONDecodeError:
                continue

    print(f"\nVerification against filtered_issues.jsonl:")
    for issue_id in issue_ids:
        if issue_id in found:
            issue = found[issue_id]
            diff_len = len(issue.get('diff', ''))
            print(f"   [OK] Issue {issue_id}: {issue.get('repo', 'unknown')} (diff: {diff_len} chars)")
        else:
            print(f"   [FAIL] Issue {issue_id}: NOT FOUND in filtered_issues.jsonl")

    return found


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/add_issue_to_memory.py <issue_id> [issue_id2 ...]")
        print("\nExample:")
        print("  python scripts/add_issue_to_memory.py 128")
        print("  python scripts/add_issue_to_memory.py 128 130 135")
        sys.exit(1)

    issue_ids = sys.argv[1:]

    print(f"Adding {len(issue_ids)} issue(s) to memory training set...")
    print("=" * 80)

    # Verify issues exist
    found_issues = verify_issues_exist(issue_ids)

    # Add to memory
    added, already_exists = add_issues_to_memory(issue_ids)

    print("\n" + "=" * 80)
    print("[OK] Done!")
