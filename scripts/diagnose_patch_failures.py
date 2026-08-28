#!/usr/bin/env python3
"""
Diagnose why patches fail to apply by analyzing git errors.

This script takes the patch failure JSON and provides detailed diagnostics
for each failure, explaining what went wrong and how to fix it.

Usage:
    python scripts/diagnose_patch_failures.py <failure_json_file>
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import from minisweagent
sys.path.insert(0, str(Path(__file__).parent.parent))

from minisweagent.run.benchmarks.utils.patch_merger import diagnose_patch_failure


def analyze_failures(failure_data: Dict[str, Any]) -> None:
    """Analyze and report on patch failures."""

    print("\n" + "=" * 80)
    print("PATCH FAILURE DIAGNOSTIC REPORT")
    print("=" * 80)

    unable_to_apply = failure_data.get("unable_to_apply", [])

    print(f"\nTotal failures: {len(unable_to_apply)}")
    print(f"Corrupted patches: {failure_data.get('summary', {}).get('corrupted_patch_count', 0)}")

    # Group failures by type
    failure_types = {}

    for failure in unable_to_apply:
        instance_id = failure.get("id", "unknown")
        repo = f"{failure.get('repo_owner', 'unknown')}/{failure.get('repo_name', 'unknown')}"
        sha_fail = failure.get("sha_fail", "unknown")[:12]
        git_error = failure.get("git_error", "")

        print("\n" + "-" * 80)
        print(f"Instance ID: {instance_id}")
        print(f"Repository: {repo}")
        print(f"Commit: {sha_fail}")
        print(f"Diff lines: {failure.get('diff_lines', 0)}")
        print(f"Return code: {failure.get('returncode', -1)}")

        # Run diagnosis
        diagnosis = diagnose_patch_failure(diff_text="", git_error=git_error)

        failure_type = diagnosis["failure_type"]
        if failure_type not in failure_types:
            failure_types[failure_type] = []
        failure_types[failure_type].append(instance_id)

        print(f"\n  Failure Type: {failure_type.upper()}")
        print(f"  Affected Files: {', '.join(diagnosis['affected_files']) if diagnosis['affected_files'] else 'N/A'}")
        print(f"\n  Diagnosis:")
        for line in diagnosis["diagnosis"].split('\n'):
            print(f"    {line}")
        print(f"\n  Suggested Fix:")
        for line in diagnosis["suggested_fix"].split('\n'):
            print(f"    {line}")

        print(f"\n  Git Error (first 300 chars):")
        print(f"    {git_error[:300]}")
        if len(git_error) > 300:
            print(f"    ... ({len(git_error) - 300} more characters)")

    # Summary by failure type
    print("\n" + "=" * 80)
    print("FAILURE TYPE SUMMARY")
    print("=" * 80)

    for failure_type, instance_ids in sorted(failure_types.items(), key=lambda x: -len(x[1])):
        print(f"\n{failure_type.upper()}: {len(instance_ids)} instances")
        print(f"  IDs: {', '.join(instance_ids)}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDED FIXES")
    print("=" * 80)

    recommendations = []

    if "missing_file" in failure_types:
        recommendations.append({
            "priority": 1,
            "issue": "New files not properly marked in patches",
            "fix": "Implemented in _collect_final_workspace_diff: Use 'git add -N .' before diff",
            "affected": len(failure_types["missing_file"])
        })

    if "context_mismatch" in failure_types:
        recommendations.append({
            "priority": 2,
            "issue": "Patch context doesn't match file state",
            "fix": "Implemented validation in _collect_final_workspace_diff: Check patch applies before saving",
            "affected": len(failure_types["context_mismatch"])
        })

    if "corrupt_patch" in failure_types:
        recommendations.append({
            "priority": 1,
            "issue": "Malformed patch structure",
            "fix": "Implemented in patch_merger.py: Validate structure with _is_valid_patch_structure",
            "affected": len(failure_types["corrupt_patch"])
        })

    if "missing_blob" in failure_types:
        recommendations.append({
            "priority": 3,
            "issue": "Missing git objects for 3-way merge",
            "fix": "Consider removing --shared from git clone or fetch full objects",
            "affected": len(failure_types["missing_blob"])
        })

    for i, rec in enumerate(sorted(recommendations, key=lambda x: x["priority"]), 1):
        print(f"\n{i}. [{rec['priority']}] {rec['issue']}")
        print(f"   Affects: {rec['affected']} instance(s)")
        print(f"   Fix: {rec['fix']}")

    print("\n" + "=" * 80)
    print("END REPORT")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Use default path if no argument provided
        default_path = Path(__file__).parent.parent / "patch_failures.json"
        if default_path.exists():
            json_path = default_path
            print(f"Using default path: {json_path}")
        else:
            print(f"Usage: {sys.argv[0]} <failure_json_file>")
            print(f"\nExpected default file not found: {default_path}")
            sys.exit(1)
    else:
        json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    try:
        with open(json_path, 'r') as f:
            failure_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_path}: {e}")
        sys.exit(1)

    analyze_failures(failure_data)


if __name__ == "__main__":
    main()
