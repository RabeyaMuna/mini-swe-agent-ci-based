#!/usr/bin/env python3
"""
CLEAN L1/L2 Memory Builder - Preserves ALL information from decomposed issues

NO DATA LOSS - Directly uses decomposed_issues.json with full details:
- problem, root_cause, how_fixed, why_fix_works
- Serial order by validation_order
- Dependency chains preserved
- All problems included (not just first!)

Usage:
    python scripts/build_clean_memory.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_clean_l1(issues: List[Dict]) -> List[Dict]:
    """
    Build L1 per-file memory with COMPLETE information.

    For each problem, create entries for each affected file.
    Preserve ALL fields from decomposition.
    """
    print("\n" + "="*80)
    print("BUILDING CLEAN L1 (Per-File Memory)")
    print("="*80)

    l1_entries = []
    entry_id = 1

    for issue in issues:
        if "error" in issue:
            continue

        problems = issue.get("problems", [])
        issue_id = issue.get("original_issue_id", "")
        repo = issue.get("repo", "")

        # Build dependency map
        dep_graph = issue.get("dependency_graph", {})
        edges = dep_graph.get("edges", [])

        enables_map = defaultdict(list)
        enabled_by_map = defaultdict(list)

        for edge in edges:
            from_id = edge.get("from")
            to_id = edge.get("to")
            if from_id and to_id:
                enables_map[from_id].append(to_id)
                enabled_by_map[to_id].append(from_id)

        # Create L1 entry for each problem
        for problem in problems:
            problem_id = problem.get("problem_id")
            files = problem.get("affected_files", [])

            if not files:
                files = ["<no specific file>"]

            # Get what this enables/enabled_by
            enables = [f"problem_{pid}" for pid in enables_map.get(problem_id, [])]
            enabled_by = [f"problem_{pid}" for pid in enabled_by_map.get(problem_id, [])]

            for file_path in files:
                l1_entries.append({
                    "id": f"L1-{entry_id:04d}",
                    "file": file_path,
                    "repo": repo,
                    "issue_id": issue_id,
                    "problem_id": problem_id,
                    "validation_order": problem.get("validation_order"),
                    "validation_cmd": problem.get("validation_cmd", ""),
                    "failure_type": problem.get("failure_type", ""),
                    "issue_type": problem.get("issue_type", ""),
                    "problem": problem.get("problem", ""),
                    "root_cause": problem.get("root_cause", ""),
                    "how_fixed": problem.get("how_fixed", ""),
                    "why_fix_works": problem.get("why_fix_works", ""),
                    "enables": enables,
                    "enabled_by": enabled_by
                })
                entry_id += 1

    print(f"  ✓ Created {len(l1_entries)} L1 entries from {len([i for i in issues if 'error' not in i])} issues")
    return l1_entries


def build_clean_l2(issues: List[Dict]) -> List[Dict]:
    """
    Build L2 repair trajectory with COMPLETE information in SERIAL order.

    Groups problems by validation_order to show repair sequence.
    Preserves ALL problem details.
    Shows primary vs hidden failures.
    """
    print("\n" + "="*80)
    print("BUILDING CLEAN L2 (Serial Repair Trajectory)")
    print("="*80)

    l2_entries = []

    for issue in issues:
        if "error" in issue:
            continue

        problems = issue.get("problems", [])
        if not problems:
            continue

        issue_id = issue.get("original_issue_id", "")
        repo = issue.get("repo", "")

        # Sort by validation_order for serial sequence
        sorted_problems = sorted(problems, key=lambda p: p.get("validation_order", 999))

        # Group by validation_order and validation_cmd
        validation_groups = defaultdict(list)
        for p in sorted_problems:
            key = (p.get("validation_order", 999), p.get("validation_cmd", ""))
            validation_groups[key].append(p)

        # Build repair trajectory
        trajectory = []
        prev_validation = None

        for step_num, (key, group_problems) in enumerate(sorted(validation_groups.items()), 1):
            val_order, val_cmd = key

            # Collect all files from all problems in this step
            all_files = []
            for p in group_problems:
                all_files.extend(p.get("affected_files", []))
            all_files = list(dict.fromkeys(all_files))

            # Build problems list with FULL details
            problems_detail = []
            for p in group_problems:
                problems_detail.append({
                    "problem_id": p.get("problem_id"),
                    "issue_type": p.get("issue_type", ""),
                    "files": p.get("affected_files", []),
                    "problem": p.get("problem", ""),
                    "root_cause": p.get("root_cause", ""),
                    "how_fixed": p.get("how_fixed", ""),
                    "why_fix_works": p.get("why_fix_works", "")
                })

            # Create step
            step = {
                "step": step_num,
                "type": "primary" if step_num == 1 else "hidden",
                "validation_order": val_order,
                "validation_cmd": val_cmd,
                "failure_type": group_problems[0].get("failure_type", ""),
                "total_files": len(all_files),
                "problems_count": len(group_problems),
                "problems": problems_detail,
                "revealed_by": prev_validation if step_num > 1 else None,
                "depends_on": prev_validation if step_num > 1 else None
            }

            trajectory.append(step)
            prev_validation = val_cmd

        l2_entries.append({
            "issue_id": issue_id,
            "repo": repo,
            "total_problems": len(problems),
            "repair_steps": len(trajectory),
            "repair_trajectory": trajectory
        })

        print(f"  ✓ Issue {issue_id}: {len(trajectory)} steps, {len(problems)} problems")

    return l2_entries


def main():
    print("="*80)
    print("CLEAN L1/L2 MEMORY BUILDER")
    print("Preserves ALL information from decomposed_issues.json")
    print("="*80)

    # Input/Output paths
    decomposed_path = PROJECT_ROOT / "data/trs/decomposed_issues.json"
    l1_path = PROJECT_ROOT / "data/trs/failure_memory.json"
    l2_path = PROJECT_ROOT / "data/trs/repo_memory.json"

    # Load decomposed issues
    print(f"\nLoading: {decomposed_path}")

    if not decomposed_path.exists():
        print(f"ERROR: {decomposed_path} not found!")
        return 1

    with open(decomposed_path) as f:
        issues = json.load(f)

    print(f"Loaded {len(issues)} issues")

    # Build L1
    l1_memories = build_clean_l1(issues)

    # Build L2
    l2_memories = build_clean_l2(issues)

    # Save
    print("\n" + "="*80)
    print("SAVING")
    print("="*80)

    with open(l1_path, 'w') as f:
        json.dump(l1_memories, f, indent=2)
    print(f"✓ L1: {l1_path}")
    print(f"  {len(l1_memories)} entries")

    with open(l2_path, 'w') as f:
        json.dump(l2_memories, f, indent=2)
    print(f"✓ L2: {l2_path}")
    print(f"  {len(l2_memories)} trajectories")

    # Summary
    print("\n" + "="*80)
    print("COMPLETE - NO DATA LOSS!")
    print("="*80)
    print("\nL1 includes:")
    print("  ✓ problem, root_cause, how_fixed, why_fix_works")
    print("  ✓ validation_order, validation_cmd")
    print("  ✓ enables, enabled_by")
    print("\nL2 includes:")
    print("  ✓ Serial repair trajectory (validation_order)")
    print("  ✓ Primary vs hidden failures")
    print("  ✓ ALL problems per step (not just first!)")
    print("  ✓ Full details: problem, root_cause, how_fixed, why_fix_works")
    print("  ✓ revealed_by, depends_on chains")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
