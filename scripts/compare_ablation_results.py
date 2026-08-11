#!/usr/bin/env python3
"""
Compare memory ablation test results across L1, L1+L2, L1+L2+L3.

Usage:
    python scripts/compare_ablation_results.py data/ablation_test
"""

import json
import sys
from pathlib import Path
from typing import Any


def load_decomposed_result(output_dir: Path) -> dict[str, Any] | None:
    """Load decomposed_issues.json from output directory."""
    decomposed_file = output_dir / "decomposed_issues.json"
    if not decomposed_file.exists():
        return None

    try:
        with open(decomposed_file) as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data[0]  # First issue
            return data
    except Exception as e:
        print(f"Error loading {decomposed_file}: {e}")
        return None


def analyze_problems(result: dict[str, Any]) -> dict[str, Any]:
    """Analyze problems from decomposition result."""
    if not result:
        return {
            "total_problems": 0,
            "ci_problems": 0,
            "dependency_problems": 0,
            "consecutive_problems": 0,
            "common_problems": 0,
            "problem_types": {},
        }

    all_problems = result.get("all_problems", [])

    # Count by problem_type
    problem_types = {}
    ci_count = 0
    dep_count = 0
    consec_count = 0
    common_count = 0

    for prob in all_problems:
        ptype = prob.get("problem_type", "unknown")
        problem_types[ptype] = problem_types.get(ptype, 0) + 1

        if ptype == "ci_failure":
            ci_count += 1
        elif ptype == "dependent":
            dep_count += 1
        elif ptype == "consecutive":
            consec_count += 1
        elif ptype == "common":
            common_count += 1

    return {
        "total_problems": len(all_problems),
        "ci_problems": ci_count,
        "dependency_problems": dep_count,
        "consecutive_problems": consec_count,
        "common_problems": common_count,
        "problem_types": problem_types,
    }


def compare_results(base_dir: Path):
    """Compare ablation results across L1, L1+L2, L1+L2+L3."""
    levels = {
        "L1 Only": base_dir / "l1_only",
        "L1+L2": base_dir / "l1_l2",
        "L1+L2+L3": base_dir / "l1_l2_l3",
    }

    print("=" * 80)
    print("Memory Ablation Comparison")
    print("=" * 80)
    print()

    results = {}
    for level_name, output_dir in levels.items():
        print(f"Loading {level_name}: {output_dir}")
        result = load_decomposed_result(output_dir)
        if result:
            results[level_name] = analyze_problems(result)
            print(f"  ✓ Loaded")
        else:
            results[level_name] = None
            print(f"  ✗ Not found")
    print()

    # Print comparison table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()
    print(f"{'Metric':<30} {'L1 Only':>12} {'L1+L2':>12} {'L1+L2+L3':>12}")
    print("-" * 80)

    metrics = [
        ("total_problems", "Total Problems"),
        ("ci_problems", "CI Failure Problems"),
        ("dependency_problems", "Dependency Problems"),
        ("consecutive_problems", "Consecutive Problems"),
        ("common_problems", "Common Problems"),
    ]

    for key, label in metrics:
        l1_val = results.get("L1 Only", {}).get(key, "N/A") if results.get("L1 Only") else "N/A"
        l12_val = results.get("L1+L2", {}).get(key, "N/A") if results.get("L1+L2") else "N/A"
        l123_val = results.get("L1+L2+L3", {}).get(key, "N/A") if results.get("L1+L2+L3") else "N/A"

        print(f"{label:<30} {str(l1_val):>12} {str(l12_val):>12} {str(l123_val):>12}")

    print("-" * 80)
    print()

    # Expected behavior
    print("=" * 80)
    print("EXPECTED BEHAVIOR")
    print("=" * 80)
    print()
    print("L1 Only (Failure Sequences):")
    print("  - Should retrieve L1 matches (similar failure sequences)")
    print("  - CI problems only (no dependency/consecutive/common)")
    print()
    print("L1+L2 (+ Repair Strategies):")
    print("  - Should retrieve L1 + L2 matches")
    print("  - CI problems + repair strategies from L2")
    print("  - May have dependency/consecutive problems")
    print()
    print("L1+L2+L3 (Full STAIR):")
    print("  - Should retrieve L1 + L2 + L3 matches")
    print("  - CI problems + dependencies + consecutive + COMMON patterns")
    print("  - Highest total problem count (most comprehensive)")
    print()

    # Validation
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print()

    l1_total = results.get("L1 Only", {}).get("total_problems", 0) if results.get("L1 Only") else 0
    l12_total = results.get("L1+L2", {}).get("total_problems", 0) if results.get("L1+L2") else 0
    l123_total = results.get("L1+L2+L3", {}).get("total_problems", 0) if results.get("L1+L2+L3") else 0

    # Check expected progression
    checks = []

    # L1+L2 should have >= L1 problems
    if l12_total >= l1_total:
        checks.append(("✓", f"L1+L2 ({l12_total}) >= L1 ({l1_total})"))
    else:
        checks.append(("✗", f"L1+L2 ({l12_total}) < L1 ({l1_total}) - UNEXPECTED!"))

    # L1+L2+L3 should have >= L1+L2 problems
    if l123_total >= l12_total:
        checks.append(("✓", f"L1+L2+L3 ({l123_total}) >= L1+L2 ({l12_total})"))
    else:
        checks.append(("✗", f"L1+L2+L3 ({l123_total}) < L1+L2 ({l12_total}) - UNEXPECTED!"))

    # L3 should have common problems
    l123_common = results.get("L1+L2+L3", {}).get("common_problems", 0) if results.get("L1+L2+L3") else 0
    if l123_common > 0:
        checks.append(("✓", f"L1+L2+L3 has {l123_common} common problems"))
    else:
        checks.append(("⚠", f"L1+L2+L3 has 0 common problems (may be OK if none found)"))

    # L1 should NOT have common problems
    l1_common = results.get("L1 Only", {}).get("common_problems", 0) if results.get("L1 Only") else 0
    if l1_common == 0:
        checks.append(("✓", f"L1 has 0 common problems (correct - L3 not loaded)"))
    else:
        checks.append(("✗", f"L1 has {l1_common} common problems - UNEXPECTED!"))

    for status, message in checks:
        print(f"{status} {message}")

    print()
    print("=" * 80)

    # Save comparison to JSON
    comparison_file = base_dir / "comparison.json"
    with open(comparison_file, "w") as f:
        json.dump({
            "results": results,
            "validation": {
                "l1_total": l1_total,
                "l12_total": l12_total,
                "l123_total": l123_total,
                "l1_common": l1_common,
                "l123_common": l123_common,
                "checks": [{"status": s, "message": m} for s, m in checks],
            }
        }, f, indent=2)

    print(f"Comparison saved to: {comparison_file}")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/compare_ablation_results.py <ablation_test_dir>")
        print("Example: python scripts/compare_ablation_results.py data/ablation_test")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    compare_results(base_dir)
