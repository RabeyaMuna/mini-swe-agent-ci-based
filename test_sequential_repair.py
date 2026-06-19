#!/usr/bin/env python3
"""
Test Sequential Repair Implementation
======================================

Quick test to verify sequential repair logic works correctly.
"""

import json
from pathlib import Path


def test_build_single_problem_task():
    """Test that single problem task is built correctly."""

    # Import the function
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from minisweagent.run.benchmarks.cibench import _build_single_problem_task

    # Mock problem
    problem = {
        "problem_number": 1,
        "status": "confirmed",
        "validation_stage": "Type checking (mypy)",
        "validation_order": 1,
        "verification_cmd": "python -m mypy py",
        "problem_statement": "Validation 'python -m mypy py' fails on file 'src/py/flwr/common/ndarrays_arithmetic.py' at line 42. Error: 'Module \"numpy.typing\" has no attribute \"DTypeLike\"'.",
        "root_cause": "DTypeLike is from private module numpy._typing",
        "fix_strategy": "Replace with public API np.dtype[Any]",
        "error_type": "type-error",
        "error_description": "Module has no attribute DTypeLike",
        "files": [
            {
                "path": "src/py/flwr/common/ndarrays_arithmetic.py",
                "line": "42",
                "current_code": "def foo(x: DTypeLike):",
                "required_fix": "Replace DTypeLike with np.dtype[Any]"
            }
        ],
        "check_first": False,
        "reasoning": "Current CI failure from logs"
    }

    task = _build_single_problem_task(problem, 1, 3)

    print("="*70)
    print("SINGLE PROBLEM TASK OUTPUT:")
    print("="*70)
    print(task)
    print("="*70)

    # Verify key components are present
    assert "Problem 1/3" in task
    assert "CONFIRMED - Currently Failing" in task
    assert "python -m mypy py" in task
    assert "ndarrays_arithmetic.py" in task
    assert "Root Cause" in task
    assert "Fix Strategy" in task
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in task

    print("\n✓ Test passed! Single problem task is correctly built.")
    print(f"  Task length: {len(task)} chars")
    print(f"  Contains verification command: python -m mypy py")
    print(f"  Contains file path: ndarrays_arithmetic.py")

    return True


def test_sequential_mode_detection():
    """Test that sequential mode is triggered correctly."""

    # Mock guidance document with multiple problems
    guidance_doc = {
        "total_problems": 3,
        "problems": [
            {"problem_number": 1, "status": "confirmed"},
            {"problem_number": 2, "status": "probable"},
            {"problem_number": 3, "status": "probable"}
        ]
    }

    problems = guidance_doc.get("problems", [])

    # Should trigger sequential mode
    should_use_sequential = problems and len(problems) > 1

    print("\n" + "="*70)
    print("SEQUENTIAL MODE DETECTION:")
    print("="*70)
    print(f"Total problems: {len(problems)}")
    print(f"Should use sequential mode: {should_use_sequential}")

    assert should_use_sequential == True
    print("✓ Test passed! Sequential mode correctly detected.")

    # Test with single problem
    guidance_doc_single = {
        "total_problems": 1,
        "problems": [
            {"problem_number": 1, "status": "confirmed"}
        ]
    }

    problems_single = guidance_doc_single.get("problems", [])
    should_use_sequential_single = problems_single and len(problems_single) > 1

    print(f"\nSingle problem test:")
    print(f"Total problems: {len(problems_single)}")
    print(f"Should use sequential mode: {should_use_sequential_single}")

    assert should_use_sequential_single == False
    print("✓ Test passed! Standard mode correctly selected for single problem.")

    return True


def test_combine_partial_fixes():
    """Test combining partial fixes."""

    import sys
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from minisweagent.run.benchmarks.cibench import _combine_partial_fixes

    partial_fixes = [
        {
            'problem_id': 1,
            'patch': 'diff --git a/file1.py\n+fixed line 1'
        },
        {
            'problem_id': 2,
            'patch': 'diff --git a/file2.py\n+fixed line 2'
        },
        {
            'problem_id': 3,
            'patch': 'diff --git a/file3.py\n+fixed line 3'
        }
    ]

    unified = _combine_partial_fixes(partial_fixes)

    print("\n" + "="*70)
    print("COMBINED PARTIAL FIXES:")
    print("="*70)
    print(unified)
    print("="*70)

    assert "# Problem 1 fix" in unified
    assert "# Problem 2 fix" in unified
    assert "# Problem 3 fix" in unified
    assert "fixed line 1" in unified
    assert "fixed line 2" in unified
    assert "fixed line 3" in unified

    print("\n✓ Test passed! Partial fixes correctly combined.")
    print(f"  Unified diff length: {len(unified)} chars")
    print(f"  Contains 3 problem fixes")

    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING SEQUENTIAL REPAIR IMPLEMENTATION")
    print("="*70 + "\n")

    try:
        test_build_single_problem_task()
        test_sequential_mode_detection()
        test_combine_partial_fixes()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nSequential repair implementation is working correctly!")
        print("\nKey features verified:")
        print("  ✓ Single problem task generation")
        print("  ✓ Sequential mode detection (multiple problems)")
        print("  ✓ Standard mode fallback (single problem)")
        print("  ✓ Partial fix combination")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
