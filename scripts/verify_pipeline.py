#!/usr/bin/env python3
"""
Verify the decomposition and memory building pipeline is working correctly.

Checks:
1. Decompose script has no debugger breakpoints
2. Incremental saving is enabled
3. Data structure compatibility (decomposition → L2 → L3)
4. Required fields are present
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def check_decompose_script():
    """Check decompose_ci_failure.py for issues."""
    print("=" * 80)
    print("1. Checking decompose_ci_failure.py")
    print("=" * 80)

    script_path = PROJECT_ROOT / "scripts" / "decompose_ci_failure.py"
    content = script_path.read_text()

    issues = []

    # Check for debugger breakpoints
    if "pdb.set_trace" in content or "import pdb" in content:
        issues.append("❌ Found debugger breakpoints (pdb.set_trace)")
    else:
        print("✅ No debugger breakpoints")

    # Check for incremental saving
    if "# Incremental save after each issue" in content:
        print("✅ Incremental saving enabled")
    else:
        issues.append("❌ Incremental saving not found")

    # Check for resume capability
    if "processed_ids" in content:
        print("✅ Resume capability enabled")
    else:
        issues.append("❌ Resume capability not found")

    return issues


def check_l2_builder():
    """Check build_l2_sequential.py for compatibility."""
    print("\n" + "=" * 80)
    print("2. Checking build_l2_sequential.py")
    print("=" * 80)

    script_path = PROJECT_ROOT / "scripts" / "build_l2_sequential.py"
    content = script_path.read_text()

    issues = []

    # Check for LLM calls
    if "llm.invoke" in content and "# Not used" not in content:
        issues.append("⚠️  L2 builder still makes LLM calls (should be transform-only)")
    else:
        print("✅ L2 builder is transform-only (no LLM calls)")

    # Check for issue_type field (needed for L3)
    if '"issue_type"' in content or "'issue_type'" in content:
        print("✅ issue_type field included (L3 compatibility)")
    else:
        issues.append("❌ issue_type field missing (needed for L3)")

    # Check for clean output structure
    if '"problem":' in content and '"fixes":' in content:
        print("✅ Clean output structure (problem + fixes)")
    else:
        issues.append("❌ Output structure incomplete")

    return issues


def check_data_files():
    """Check if data files exist and have correct structure."""
    print("\n" + "=" * 80)
    print("3. Checking Data Files")
    print("=" * 80)

    issues = []
    data_dir = PROJECT_ROOT / "data" / "trs"

    # Check decomposed_issues.json
    decomposed_path = data_dir / "decomposed_issues.json"
    if decomposed_path.exists():
        with open(decomposed_path) as f:
            decomposed = json.load(f)

        if isinstance(decomposed, list):
            print(f"✅ decomposed_issues.json exists ({len(decomposed)} entries)")

            if len(decomposed) == 0:
                issues.append("⚠️  decomposed_issues.json is empty (need to run decomposition)")
            else:
                # Check first entry structure
                first = decomposed[0]
                if "error" in first:
                    issues.append(f"❌ First entry has error: {first.get('error_type')}")
                elif "problems" in first or "atomic_problems" in first:
                    problems = first.get("problems") or first.get("atomic_problems", [])
                    print(f"   First entry has {len(problems)} problems")

                    if problems:
                        # Check problem structure
                        p = problems[0]
                        required_fields = ["validation_order", "validation_cmd", "problem", "root_cause"]
                        missing = [f for f in required_fields if f not in p]

                        if missing:
                            issues.append(f"⚠️  Problem missing fields: {missing}")
                        else:
                            print(f"   ✅ Problems have required fields")

                        # Check for file_changes
                        if "file_changes" in p:
                            fc = p["file_changes"]
                            if fc and isinstance(fc, list) and len(fc) > 0:
                                change = fc[0]
                                if "from" in change and "to" in change:
                                    print(f"   ✅ file_changes have from/to")
                                else:
                                    issues.append("⚠️  file_changes missing from/to")
                else:
                    issues.append("❌ Entry missing 'problems' or 'atomic_problems'")
        else:
            issues.append("❌ decomposed_issues.json is not a list")
    else:
        issues.append("❌ decomposed_issues.json not found")

    # Check L1
    l1_path = data_dir / "failure_memory.json"
    if l1_path.exists():
        with open(l1_path) as f:
            l1 = json.load(f)
        print(f"✅ L1 (failure_memory.json) exists ({len(l1)} entries)")
    else:
        issues.append("⚠️  L1 (failure_memory.json) not found (expected after build)")

    # Check L2
    l2_path = data_dir / "repo_memory.json"
    if l2_path.exists():
        with open(l2_path) as f:
            l2 = json.load(f)
        print(f"✅ L2 (repo_memory.json) exists ({len(l2)} entries)")

        if l2 and len(l2) > 0:
            # Check L2 structure
            first_l2 = l2[0]
            if "atomic_problems" in first_l2:
                problems = first_l2["atomic_problems"]
                if problems and len(problems) > 0:
                    p = problems[0]
                    if "problem" in p and isinstance(p["problem"], dict):
                        print("   ✅ L2 has new clean structure (problem dict)")
                    if "fixes" in p:
                        print("   ✅ L2 has fixes array")
                    if "issue_type" in p:
                        print("   ✅ L2 has issue_type (for L3)")
                    else:
                        issues.append("⚠️  L2 missing issue_type (needed for L3)")
    else:
        issues.append("⚠️  L2 (repo_memory.json) not found (expected after build)")

    # Check L3
    l3_path = data_dir / "cross_memory.json"
    if l3_path.exists():
        with open(l3_path) as f:
            l3 = json.load(f)

        if isinstance(l3, list):
            if len(l3) == 0:
                issues.append("⚠️  L3 (cross_memory.json) is empty (should have patterns)")
            else:
                print(f"✅ L3 (cross_memory.json) exists ({len(l3)} patterns)")
                # Check L3 structure
                first_l3 = l3[0]
                if "pattern_id" in first_l3 and "universal_fix_strategy" in first_l3:
                    print("   ✅ L3 has universal patterns")
        else:
            issues.append("❌ L3 (cross_memory.json) is not a list")
    else:
        issues.append("⚠️  L3 (cross_memory.json) not found (expected after build)")

    return issues


def main():
    print("\nPipeline Verification Report")
    print("=" * 80)

    all_issues = []

    # Check scripts
    all_issues.extend(check_decompose_script())
    all_issues.extend(check_l2_builder())

    # Check data
    all_issues.extend(check_data_files())

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not all_issues:
        print("✅ All checks passed!")
        print("\nReady to run:")
        print("  1. python scripts/decompose_ci_failure.py --batch")
        print("  2. python scripts/build_memory_from_decomposed.py")
    else:
        print(f"Found {len(all_issues)} issue(s):\n")
        for issue in all_issues:
            print(f"  {issue}")

        print("\nAction items:")
        if any("decomposed_issues.json is empty" in i for i in all_issues):
            print("  → Run: python scripts/decompose_ci_failure.py --batch")
        if any("L1" in i or "L2" in i or "L3" in i for i in all_issues):
            print("  → Run: python scripts/build_memory_from_decomposed.py")

    print()
    return 0 if not all_issues else 1


if __name__ == "__main__":
    exit(main())
