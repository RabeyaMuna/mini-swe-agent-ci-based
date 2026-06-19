#!/usr/bin/env python3
"""
Integration module: Enhance existing decompose_ci_failure.py with CI-specific enhancements.

This wraps the existing decomposition to add:
1. Validation-aware diff splitting
2. Sequential decomposition
3. Backward temporal timeline
4. Structured trajectories per validation

Usage:
    from integrate_ci_enhancements import decompose_issue_enhanced

    result = decompose_issue_enhanced(issue, llm)
"""

import sys
from pathlib import Path

# Import existing decomposition
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from decompose_ci_failure import (
    decompose_issue as decompose_issue_original,
    build_benchmark_ci_context,
    validate_required_ci_inputs
)
from decompose_ci_enhanced import (
    split_diff_by_validation_dynamic,
    decompose_by_validation_sequence_dynamic,
    build_temporal_timeline_dynamic,
    build_sequential_trajectory_dynamic
)


def decompose_issue_enhanced(issue: dict, llm) -> dict:
    """
    Enhanced decomposition with CI validation-aware processing.

    Flow:
    1. Use existing code to get benchmark CI context
    2. Enhancement: Split diff by validation (not by size)
    3. Enhancement: Decompose validation-by-validation
    4. Enhancement: Build temporal timeline
    5. Enhancement: Build structured trajectories
    6. Merge with existing decomposition
    """

    issue_id = issue.get('id', '?')
    print(f"\n{'='*80}")
    print(f"Enhanced Reverse Engineering: Issue {issue_id}")
    print(f"  Repo: {issue.get('repo_name', issue.get('repo', '?'))}")
    print(f"{'='*80}")

    # Step 1: Build CI context (use existing code)
    print(f"  [1/5] Building CI context...")
    benchmark_context = build_benchmark_ci_context(issue, llm=llm)

    if not validate_required_ci_inputs(benchmark_context):
        print("  ERROR: Invalid CI context")
        return {}

    validation_sequence = benchmark_context.get("validation_sequence", [])
    diff = issue.get("diff", issue.get("patch", ""))

    if not diff:
        print("  WARNING: No diff available, falling back to original decomposition")
        return decompose_issue_original(issue, llm)

    if not validation_sequence:
        print("  WARNING: No validation sequence, falling back to original decomposition")
        return decompose_issue_original(issue, llm)

    print(f"  Found {len(validation_sequence)} sequential validations")

    # Step 2: Split diff by validation (ENHANCEMENT)
    print(f"  [2/5] Splitting diff by validation order...")
    try:
        validation_chunks = split_diff_by_validation_dynamic(
            diff=diff,
            validation_sequence=validation_sequence,
            llm=llm
        )
        print(f"  Created {len(validation_chunks)} validation-specific diff chunks")
    except Exception as e:
        print(f"  WARNING: Diff splitting failed: {e}")
        print("  Falling back to original decomposition")
        return decompose_issue_original(issue, llm)

    # Step 3: Decompose by validation sequence (ENHANCEMENT)
    print(f"  [3/5] Decomposing validation-by-validation...")
    try:
        atomic_problems = decompose_by_validation_sequence_dynamic(
            issue=issue,
            validation_chunks=validation_chunks,
            validation_sequence=validation_sequence,
            llm=llm
        )
        print(f"  Identified {len(atomic_problems)} atomic problems")
    except Exception as e:
        print(f"  ERROR: Sequential decomposition failed: {e}")
        print("  Falling back to original decomposition")
        return decompose_issue_original(issue, llm)

    # Step 4: Build temporal timeline (ENHANCEMENT)
    print(f"  [4/5] Reconstructing temporal timeline...")
    try:
        temporal_timeline = build_temporal_timeline_dynamic(
            atomic_problems=atomic_problems,
            verified_patch=diff,
            llm=llm
        )
        print(f"  Built timeline with {temporal_timeline.get('total_state_transitions', 0)} state transitions")
    except Exception as e:
        print(f"  WARNING: Timeline reconstruction failed: {e}")
        temporal_timeline = {}

    # Step 5: Build structured trajectories per validation (ENHANCEMENT)
    print(f"  [5/5] Building structured trajectories...")
    trajectories = []
    for problem in atomic_problems:
        try:
            trajectory = build_sequential_trajectory_dynamic(
                atomic_problem=problem,
                validation_order=problem.get('validation_order', 1),
                total_validations=len(atomic_problems),
                verified_patch=diff,
                llm=llm
            )
            problem['repair_trajectory'] = trajectory
            trajectories.append(trajectory)
        except Exception as e:
            print(f"  WARNING: Trajectory building failed for problem {problem.get('problem_id')}: {e}")
            problem['repair_trajectory'] = {
                "investigation_steps": [],
                "fix_decisions": [],
                "verification_steps": []
            }

    print(f"  Built {len(trajectories)} structured trajectories")

    # Build final result
    result = {
        "original_issue_id": issue_id,
        "repo": issue.get('repo_name', issue.get('repo')),
        "sha_fail": issue.get('sha_fail', ''),

        # Atomic problems with enhancements
        "atomic_problems": atomic_problems,

        # Temporal timeline (NEW)
        "temporal_timeline": temporal_timeline,

        # Sequential workflow metadata (NEW)
        "sequential_workflow_metadata": {
            "total_validations": len(validation_sequence),
            "validation_dependency_chain": build_dependency_chain(validation_sequence),
            "repair_order_sequence": temporal_timeline.get('repair_sequence_order', list(range(1, len(validation_sequence) + 1)))
        },

        # Benchmark context (from original)
        "benchmark_ci_context": benchmark_context,

        # Diff analysis metadata (NEW)
        "diff_analysis_metadata": {
            "total_files_changed": len(issue.get('changed_files', [])),
            "files_per_validation": {
                val_id: len(extract_files_from_chunk(chunk))
                for val_id, chunk in validation_chunks.items()
            },
            "splitting_method": "validation_aware"
        }
    }

    print(f"  ✓ Enhanced decomposition complete")
    return result


def build_dependency_chain(validation_sequence):
    """Build dependency chain from validation sequence."""
    chain = {}
    for val in validation_sequence:
        val_id = str(val.get('order'))
        chain[val_id] = {
            "name": val.get('name', f"validation_{val_id}"),
            "blocks": val.get('blocks', []),
            "depends_on": val.get('depends_on', []),
            "must_fix_first": val.get('order') == 1
        }

        if val.get('order', 1) > 1:
            chain[val_id]["revealed_after"] = f"validation_{val.get('order') - 1}_passed"

    return chain


def extract_files_from_chunk(chunk):
    """Extract file list from diff chunk."""
    import re
    files = []
    for line in chunk.split('\n'):
        if line.startswith('diff --git'):
            match = re.search(r'b/(.+)$', line)
            if match:
                files.append(match.group(1))
    return files


# Backward compatibility: also export as decompose_issue
decompose_issue = decompose_issue_enhanced
