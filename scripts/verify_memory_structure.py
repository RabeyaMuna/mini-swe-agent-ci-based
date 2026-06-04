#!/usr/bin/env python3
"""
Memory Structure Verification Script

Run this to check if your memory has the correct structure.
Shows EXACTLY what's wrong and what needs to be fixed.

Usage:
    python scripts/verify_memory_structure.py --memory-root data/trs
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def verify_l1(l1_records: List[Dict]) -> Dict[str, Any]:
    """Verify L1 (failure_memory.json) structure."""
    issues = []
    warnings = []

    print(f"\n{'='*60}")
    print(f"L1 (Per-File Memory) Verification")
    print(f"{'='*60}")

    if not l1_records:
        print("❌ CRITICAL: L1 is EMPTY! (0 records)")
        print("   Expected: 50-100 records (one per changed file)")
        print("   Impact: Cannot match at file level, loses 45% of matching signal")
        return {"status": "FAIL", "issues": ["L1 completely missing"]}

    print(f"✅ L1 exists: {len(l1_records)} records")

    # Check required fields
    required_fields = [
        "repo", "file", "error_type", "failure_pattern",
        "failure_reason", "fix_strategy", "dependent_files"
    ]

    sample = l1_records[0]
    for field in required_fields:
        if field not in sample:
            issues.append(f"Missing required field: {field}")

    # Check field population
    empty_failure_pattern = sum(1 for r in l1_records if not r.get('failure_pattern'))
    empty_failure_reason = sum(1 for r in l1_records if not r.get('failure_reason'))
    empty_fix_strategy = sum(1 for r in l1_records if not r.get('fix_strategy'))

    if empty_failure_pattern > len(l1_records) * 0.5:
        issues.append(f"{empty_failure_pattern}/{len(l1_records)} records have empty failure_pattern")

    if empty_failure_reason > len(l1_records) * 0.5:
        issues.append(f"{empty_failure_reason}/{len(l1_records)} records have empty failure_reason")

    if empty_fix_strategy > len(l1_records) * 0.5:
        issues.append(f"{empty_fix_strategy}/{len(l1_records)} records have empty fix_strategy")

    # Check dependent_files structure
    for i, record in enumerate(l1_records[:10]):  # Check first 10
        deps = record.get('dependent_files', [])
        if deps and isinstance(deps[0], str):
            issues.append(f"Record {i}: dependent_files is list of strings, should be [{{file, reason, must_change}}]")
            break

    print(f"\nField Population:")
    print(f"  failure_pattern: {len(l1_records) - empty_failure_pattern}/{len(l1_records)} populated")
    print(f"  failure_reason:  {len(l1_records) - empty_failure_reason}/{len(l1_records)} populated")
    print(f"  fix_strategy:    {len(l1_records) - empty_fix_strategy}/{len(l1_records)} populated")

    if issues:
        print(f"\n❌ Issues found: {len(issues)}")
        for issue in issues:
            print(f"   - {issue}")
        return {"status": "FAIL", "issues": issues}
    else:
        print(f"\n✅ L1 structure is CORRECT")
        return {"status": "PASS", "issues": []}


def verify_l2(l2_records: List[Dict]) -> Dict[str, Any]:
    """Verify L2 (repo_memory.json) structure."""
    issues = []
    warnings = []

    print(f"\n{'='*60}")
    print(f"L2 (Repo-Level Memory) Verification")
    print(f"{'='*60}")

    if not l2_records:
        print("❌ CRITICAL: L2 is EMPTY!")
        return {"status": "FAIL", "issues": ["L2 completely missing"]}

    print(f"✅ L2 exists: {len(l2_records)} records")

    # Check for stringified types
    stringified_types = 0
    for i, record in enumerate(l2_records):
        error_type = str(record.get('error_type', ''))
        if '{' in error_type and 'category' in error_type:
            stringified_types += 1
            if stringified_types == 1:
                issues.append(f"Record {i}: error_type is stringified dict: {error_type[:80]}")

    if stringified_types > 0:
        issues.append(f"{stringified_types}/{len(l2_records)} records have stringified error_type")

    # Check overall_failure_reason
    stringified_reasons = 0
    empty_reasons = 0
    for i, record in enumerate(l2_records):
        reason = record.get('overall_failure_reason', '')
        if isinstance(reason, list):
            stringified_reasons += 1
        elif isinstance(reason, str) and reason.startswith('['):
            stringified_reasons += 1
            if stringified_reasons == 1:
                issues.append(f"Record {i}: overall_failure_reason is stringified list: {reason[:80]}")
        elif not reason or reason == 'N/A':
            empty_reasons += 1

    if stringified_reasons > 0:
        issues.append(f"{stringified_reasons}/{len(l2_records)} records have stringified overall_failure_reason")

    if empty_reasons > len(l2_records) * 0.3:
        issues.append(f"{empty_reasons}/{len(l2_records)} records have empty overall_failure_reason")

    # Check changed_files structure (CRITICAL!)
    broken_changed_files = 0
    missing_fields = {'reason': 0, 'failure_reason': 0, 'fix_strategy': 0}

    for i, record in enumerate(l2_records):
        cf = record.get('changed_files', [])

        if not cf:
            continue

        # Check if it's just strings
        if isinstance(cf[0], str):
            broken_changed_files += 1
            if broken_changed_files == 1:
                issues.append(f"Record {i}: changed_files is list of strings {cf}, should be [{{file, reason, failure_reason, fix_strategy}}]")
            continue

        # Check if it has required fields
        if 'reason' not in cf[0]:
            missing_fields['reason'] += 1
        if 'failure_reason' not in cf[0]:
            missing_fields['failure_reason'] += 1
        if 'fix_strategy' not in cf[0]:
            missing_fields['fix_strategy'] += 1

    if broken_changed_files > 0:
        issues.append(f"❌ CRITICAL: {broken_changed_files}/{len(l2_records)} records have broken changed_files structure!")
        print(f"\n❌ CRITICAL: changed_files is list of strings, not objects!")
        print(f"   Current:  [\"file1.py\", \"file2.py\"]")
        print(f"   Needed:   [{{\"file\": \"file1.py\", \"reason\": \"...\", \"failure_reason\": \"...\", \"fix_strategy\": \"...\"}}]")
        print(f"   Impact:   Agent doesn't know WHY files changed or HOW they were fixed")

    for field, count in missing_fields.items():
        if count > len(l2_records) * 0.3:
            issues.append(f"{count}/{len(l2_records)} changed_files missing '{field}' field")

    # Check fix_strategy population
    empty_fix_strategy = sum(1 for r in l2_records if not r.get('fix_strategy'))
    if empty_fix_strategy > len(l2_records) * 0.3:
        issues.append(f"{empty_fix_strategy}/{len(l2_records)} records have empty fix_strategy")

    # Check failure_pattern
    empty_pattern = sum(1 for r in l2_records if not r.get('failure_pattern'))
    if empty_pattern > len(l2_records) * 0.3:
        issues.append(f"{empty_pattern}/{len(l2_records)} records have empty failure_pattern")

    print(f"\nField Population:")
    print(f"  failure_pattern:         {len(l2_records) - empty_pattern}/{len(l2_records)} populated")
    print(f"  overall_failure_reason:  {len(l2_records) - empty_reasons}/{len(l2_records)} populated")
    print(f"  fix_strategy:            {len(l2_records) - empty_fix_strategy}/{len(l2_records)} populated")
    print(f"  changed_files structure: {len(l2_records) - broken_changed_files}/{len(l2_records)} correct")

    if issues:
        print(f"\n❌ Issues found: {len(issues)}")
        for issue in issues:
            print(f"   - {issue}")
        return {"status": "FAIL", "issues": issues}
    else:
        print(f"\n✅ L2 structure is CORRECT")
        return {"status": "PASS", "issues": []}


def verify_l3(l3_records: List[Dict]) -> Dict[str, Any]:
    """Verify L3 (cross_memory.json) structure."""
    issues = []

    print(f"\n{'='*60}")
    print(f"L3 (Universal Principles) Verification")
    print(f"{'='*60}")

    if not l3_records:
        print("⚠️  L3 is EMPTY (not critical)")
        return {"status": "WARN", "issues": ["L3 empty"]}

    print(f"✅ L3 exists: {len(l3_records)} records")

    # Check for repo-specific details (should NOT exist!)
    repo_specific = 0
    file_paths = 0

    for i, record in enumerate(l3_records[:20]):  # Check first 20
        # Check for repo names
        if 'repo' in record:
            repo_specific += 1
            if repo_specific == 1:
                issues.append(f"Record {i}: Contains 'repo' field (L3 should be repo-agnostic)")

        # Check for file paths in text
        for key, value in record.items():
            if isinstance(value, str):
                if '.py' in value or '.toml' in value or '.yaml' in value:
                    file_paths += 1
                    if file_paths == 1:
                        issues.append(f"Record {i}: Contains file paths in {key} (L3 should be generic)")
                    break

    if repo_specific > 0:
        issues.append(f"{repo_specific} records contain repo-specific fields")

    if file_paths > len(l3_records) * 0.3:
        issues.append(f"{file_paths} records contain file paths")

    # Check critical fields
    has_principle = sum(1 for r in l3_records if r.get('principle'))
    has_fix_strategy = sum(1 for r in l3_records if r.get('fix_strategy'))

    print(f"\nField Population:")
    print(f"  principle:     {has_principle}/{len(l3_records)} populated")
    print(f"  fix_strategy:  {has_fix_strategy}/{len(l3_records)} populated")

    if issues:
        print(f"\n⚠️  Issues found: {len(issues)}")
        for issue in issues:
            print(f"   - {issue}")
        return {"status": "WARN", "issues": issues}
    else:
        print(f"\n✅ L3 structure is CORRECT")
        return {"status": "PASS", "issues": []}


def main():
    parser = argparse.ArgumentParser(description="Verify memory structure")
    parser.add_argument("--memory-root", required=True, help="Path to memory directory")
    args = parser.parse_args()

    memory_root = Path(args.memory_root)

    # Load memory files
    l1_path = memory_root / "failure_memory.json"
    l2_path = memory_root / "repo_memory.json"
    l3_path = memory_root / "cross_memory.json"

    l1 = json.loads(l1_path.read_text()) if l1_path.exists() else []
    l2 = json.loads(l2_path.read_text()) if l2_path.exists() else []
    l3 = json.loads(l3_path.read_text()) if l3_path.exists() else []

    print(f"{'='*60}")
    print(f"MEMORY STRUCTURE VERIFICATION")
    print(f"{'='*60}")
    print(f"Memory Root: {memory_root}")

    # Verify each level
    results = {
        "L1": verify_l1(l1),
        "L2": verify_l2(l2),
        "L3": verify_l3(l3)
    }

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    failed = [level for level, result in results.items() if result["status"] == "FAIL"]
    warned = [level for level, result in results.items() if result["status"] == "WARN"]
    passed = [level for level, result in results.items() if result["status"] == "PASS"]

    print(f"✅ Passed: {', '.join(passed) if passed else 'None'}")
    if warned:
        print(f"⚠️  Warnings: {', '.join(warned)}")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")

    total_issues = sum(len(r["issues"]) for r in results.values())
    print(f"\nTotal issues: {total_issues}")

    if failed:
        print(f"\n❌ MEMORY STRUCTURE IS BROKEN!")
        print(f"   Run: cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines")
        print(f"        python scripts/build_memory_bank.py --help")
        return 1
    elif warned:
        print(f"\n⚠️  Memory structure has warnings but may work")
        return 0
    else:
        print(f"\n✅ MEMORY STRUCTURE IS CORRECT!")
        return 0


if __name__ == "__main__":
    exit(main())
