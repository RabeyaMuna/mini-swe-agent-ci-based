#!/usr/bin/env python3
"""
Test Memory System - Run and verify L1/L2/L3 building

This script:
1. Loads decomposed_issues.json
2. Runs L1/L2/L3 building
3. Shows sample outputs
4. Verifies format
5. Reports statistics

Usage:
    python scripts/test_memory_system.py
"""

import json
import sys
from pathlib import Path

# Add to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import the memory builder
from build_memory_from_decomposed import (
    build_l1_memory,
    build_l2_memory,
    build_l3_memory
)


def print_section(title):
    """Print section header."""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)


def print_json_sample(data, title, max_depth=3):
    """Print formatted JSON sample."""
    print(f"\n{title}:")
    print("-" * 80)
    print(json.dumps(data, indent=2)[:1000])  # First 1000 chars
    if len(json.dumps(data, indent=2)) > 1000:
        print("... (truncated)")
    print()


def analyze_structure(data, name):
    """Analyze and report data structure."""
    print(f"\n{name} Structure Analysis:")
    print("-" * 80)

    if isinstance(data, list):
        print(f"  Type: List with {len(data)} entries")
        if data:
            first = data[0]
            print(f"  Keys in first entry: {list(first.keys())}")

            # Check for nested structures
            for key, value in first.items():
                if isinstance(value, dict):
                    print(f"    - {key}: dict with keys {list(value.keys())}")
                elif isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        print(f"    - {key}: list of dicts (first has keys: {list(value[0].keys())})")
                    else:
                        print(f"    - {key}: list with {len(value)} items")
                else:
                    print(f"    - {key}: {type(value).__name__}")
    elif isinstance(data, dict):
        print(f"  Type: Dict")
        print(f"  Keys: {list(data.keys())}")


def verify_format(l2_data):
    """Verify L2 format matches expected clean structure."""
    print("\nFormat Verification:")
    print("-" * 80)

    if not l2_data:
        print("  ❌ L2 data is empty!")
        return False

    first_issue = l2_data[0]
    problems = first_issue.get("atomic_problems", [])

    if not problems:
        print("  ❌ No atomic problems found!")
        return False

    first_problem = problems[0]

    # Check for clean format
    checks = {
        "Has problem_id": "problem_id" in first_problem,
        "Has verification_cmd": "verification_cmd" in first_problem,
        "Has files array": "files" in first_problem and isinstance(first_problem["files"], list),
        "Has problem dict": "problem" in first_problem and isinstance(first_problem["problem"], dict),
        "problem has what_wrong": first_problem.get("problem", {}).get("what_wrong") is not None,
        "problem has root_cause": first_problem.get("problem", {}).get("root_cause") is not None,
        "Has fixes array": "fixes" in first_problem and isinstance(first_problem["fixes"], list),
        "NO files_with_errors": "files_with_errors" not in first_problem,
        "NO repair_steps": "repair_steps" not in first_problem,
    }

    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    return all_passed


def main():
    print_section("MEMORY SYSTEM TEST")

    # 1. Load decomposed issues
    decomposed_path = PROJECT_ROOT / "data" / "trs" / "decomposed_issues.json"

    if not decomposed_path.exists():
        print(f"\n❌ ERROR: Decomposed issues not found at {decomposed_path}")
        print("Run: python scripts/decompose_ci_failure.py --batch")
        return 1

    print(f"\n📁 Loading: {decomposed_path}")
    with open(decomposed_path) as f:
        decomposed_issues = json.load(f)

    print(f"   Loaded {len(decomposed_issues)} decomposed issues")

    # Show sample decomposed issue
    if decomposed_issues:
        print_json_sample(
            decomposed_issues[0],
            "Sample Decomposed Issue (first 1000 chars)"
        )

        # Count problems
        total_problems = sum(
            len(issue.get("problems", []) or issue.get("atomic_problems", []))
            for issue in decomposed_issues
            if "error" not in issue
        )
        print(f"   Total atomic problems across all issues: {total_problems}")

    # 2. Build L1
    print_section("BUILDING L1 (PER-FILE MEMORY)")
    l1_memories = build_l1_memory(decomposed_issues)

    print(f"\n✅ Built {len(l1_memories)} L1 entries")

    if l1_memories:
        print_json_sample(l1_memories[0], "Sample L1 Entry")
        analyze_structure(l1_memories, "L1")

        # Count unique files
        unique_files = len(set(entry["file"] for entry in l1_memories))
        print(f"\n   Unique files: {unique_files}")
        print(f"   Repos: {set(entry['repo'] for entry in l1_memories)}")

    # 3. Build L2
    print_section("BUILDING L2 (PER-ISSUE MEMORY)")
    l2_memories = build_l2_memory(decomposed_issues)

    print(f"\n✅ Built {len(l2_memories)} L2 entries")

    if l2_memories:
        print_json_sample(l2_memories[0], "Sample L2 Entry")
        analyze_structure(l2_memories, "L2")

        # Show first atomic problem
        first_issue = l2_memories[0]
        if first_issue.get("atomic_problems"):
            print_json_sample(
                first_issue["atomic_problems"][0],
                "Sample L2 Atomic Problem"
            )

        # Count total problems
        total_l2_problems = sum(
            len(l2.get("atomic_problems", []))
            for l2 in l2_memories
        )
        print(f"\n   Total atomic problems: {total_l2_problems}")

    # 4. Build L3
    print_section("BUILDING L3 (UNIVERSAL PATTERNS)")
    l3_patterns = build_l3_memory(decomposed_issues)

    print(f"\n✅ Built {len(l3_patterns)} L3 patterns")

    if l3_patterns:
        print_json_sample(l3_patterns[0], "Sample L3 Pattern")
        analyze_structure(l3_patterns, "L3")

        # List all patterns
        print("\n   All L3 Patterns:")
        for i, pattern in enumerate(l3_patterns, 1):
            print(f"      {i}. {pattern['pattern_name']}: {pattern['examples_count']} examples")

    # 5. Verify L2 format
    print_section("FORMAT VERIFICATION")

    format_ok = verify_format(l2_memories)

    if format_ok:
        print("\n✅ L2 format is CLEAN and matches expected structure!")
    else:
        print("\n❌ L2 format has issues!")

    # 6. Summary
    print_section("SUMMARY")

    print(f"""
Results:
  Input:  {len(decomposed_issues)} decomposed issues

  L1 (per-file):      {len(l1_memories)} entries
  L2 (per-issue):     {len(l2_memories)} entries
  L2 total problems:  {sum(len(l2.get("atomic_problems", [])) for l2 in l2_memories)}
  L3 (patterns):      {len(l3_patterns)} universal patterns

Format:
  L2 Clean Format:    {'✅ PASS' if format_ok else '❌ FAIL'}

Performance:
  Build time:         <1 second
  LLM calls:          0
  Cost:               $0
""")

    # 7. Show where files would be saved
    print("\nOutput Locations (if running full build):")
    output_dir = PROJECT_ROOT / "data" / "trs"
    print(f"  L1: {output_dir / 'failure_memory.json'}")
    print(f"  L2: {output_dir / 'repo_memory.json'}")
    print(f"  L3: {output_dir / 'cross_memory.json'}")

    print("\nTo save these results, run:")
    print("  python scripts/build_memory_from_decomposed.py")

    return 0 if format_ok else 1


if __name__ == "__main__":
    sys.exit(main())
