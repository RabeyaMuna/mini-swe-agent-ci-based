#!/usr/bin/env python3
"""
Test script to verify file frequency fix works correctly.

This simulates the exit_code_test.py scenario where one file appears
in multiple problems but wasn't being selected.
"""

import sys
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Import the functions we want to test
sys.path.insert(0, '/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/src')
from minisweagent.run.benchmarks.utils.ci_memory_l2_analysis import (
    _analyze_file_frequency,
    _format_file_frequency_for_prompt,
    _validate_file_frequency_coverage
)

def test_file_frequency_analysis():
    """Test that file frequency analysis correctly identifies high-frequency files."""

    print("="*80)
    print("TEST 1: File Frequency Analysis")
    print("="*80)

    # Simulate candidates similar to the exit_code_test.py case
    candidates = [
        {
            "candidate_id": "C001",
            "files": ["framework/py/flwr/common/exit/exit_code_test.py"],
            "issue_type": "pylint-false-positive",
            "failure_type": "Code Linting",
            "problem": "The original code used chained method calls...",
            "frequency": 1,
            "appears_in_issues": ["flower_126"]
        },
        {
            "candidate_id": "C002",
            "files": ["framework/py/flwr/common/exit/exit_code_test.py"],
            "issue_type": "Docstring signature check",
            "failure_type": "Docstring",
            "problem": "The docsig validator checks that function/method signatures...",
            "frequency": 1,
            "appears_in_issues": ["flower_120"]
        },
        {
            "candidate_id": "C003",
            "files": ["framework/py/flwr/common/exit/exit_code_test.py"],
            "issue_type": "merge conflict markers",
            "failure_type": "Code Quality / Linting",
            "problem": "The file contained git merge conflict markers...",
            "frequency": 1,
            "appears_in_issues": ["flower_117"]
        },
        {
            "candidate_id": "C004",
            "files": ["framework/py/flwr/cli/build.py"],
            "issue_type": "Unused import",
            "failure_type": "Type Checking",
            "problem": "The import was present but unused...",
            "frequency": 1,
            "appears_in_issues": ["flower_120"]
        },
        {
            "candidate_id": "C005",
            "files": ["framework/py/flwr/server/strategy/aggregate.py"],
            "issue_type": "invalid type annotation",
            "failure_type": "Type Checking",
            "problem": "numpy type annotation issue...",
            "frequency": 2,
            "appears_in_issues": ["flower_120", "flower_117"]
        }
    ]

    # Run analysis
    result = _analyze_file_frequency(candidates)

    print(f"\nTotal candidates analyzed: {len(candidates)}")
    print(f"High-frequency files found: {result['high_frequency_count']}")
    print(f"\nHigh-frequency files:")
    for file, info in result['high_frequency_files'].items():
        print(f"  - {file}: {info['frequency']} problems")
        print(f"    Issue types: {info['issue_types']}")
        print(f"    Problem IDs: {info['problem_ids']}")

    # Check that exit_code_test.py is identified
    assert "framework/py/flwr/common/exit/exit_code_test.py" in result['high_frequency_files']
    assert result['high_frequency_files']["framework/py/flwr/common/exit/exit_code_test.py"]["frequency"] == 3

    print("\n✓ TEST 1 PASSED: exit_code_test.py correctly identified as high-frequency (3 occurrences)")
    return result

def test_format_for_prompt(file_analysis):
    """Test that formatting for prompt is correct."""

    print("\n" + "="*80)
    print("TEST 2: Format for LLM Prompt")
    print("="*80)

    formatted = _format_file_frequency_for_prompt(file_analysis)

    print("\nFormatted prompt section:")
    print(formatted)

    # Check that the formatted string contains key information
    assert "exit_code_test.py" in formatted
    assert "HIGH PRIORITY" in formatted
    assert "3 problems" in formatted

    print("\n✓ TEST 2 PASSED: Prompt formatting includes high-frequency file info")

def test_validation_safeguard():
    """Test that validation safeguard adds missing high-frequency files."""

    print("\n" + "="*80)
    print("TEST 3: Validation Safeguard")
    print("="*80)

    # Simulate LLM selection that MISSED exit_code_test.py
    selected = [
        {
            "files": ["framework/py/flwr/cli/build.py"],
            "problem": "Unused import",
            "validation_cmd": "python -m mypy py",
            "failure_type": "Type Checking"
        }
    ]

    candidates = [
        {
            "candidate_id": "C001",
            "files": ["framework/py/flwr/common/exit/exit_code_test.py"],
            "problem": "Pylint issue",
            "frequency": 1
        },
        {
            "candidate_id": "C002",
            "files": ["framework/py/flwr/common/exit/exit_code_test.py"],
            "problem": "Docstring issue",
            "frequency": 1
        },
        {
            "candidate_id": "C003",
            "files": ["framework/py/flwr/common/exit/exit_code_test.py"],
            "problem": "Merge conflict",
            "frequency": 2  # Highest frequency for this file
        },
        {
            "candidate_id": "C004",
            "files": ["framework/py/flwr/cli/build.py"],
            "problem": "Unused import",
            "frequency": 1
        }
    ]

    print(f"\nBefore validation:")
    print(f"  Selected problems: {len(selected)}")
    print(f"  Files in selection: {[f for p in selected for f in p.get('files', [])]}")

    # Run validation (should add exit_code_test.py)
    validated = _validate_file_frequency_coverage(
        selected,
        candidates,
        min_frequency_threshold=3
    )

    print(f"\nAfter validation:")
    print(f"  Selected problems: {len(validated)}")
    all_files = [f for p in validated for f in p.get('files', [])]
    print(f"  Files in selection: {all_files}")

    # Check that exit_code_test.py was added
    assert "framework/py/flwr/common/exit/exit_code_test.py" in all_files
    assert len(validated) == len(selected) + 1  # One problem added

    print("\n✓ TEST 3 PASSED: High-frequency file (3+ occurrences) was force-added by safeguard")

def test_no_false_positives():
    """Test that validation doesn't add files when threshold not met."""

    print("\n" + "="*80)
    print("TEST 4: No False Positives")
    print("="*80)

    # File appears only 2 times (below threshold of 3)
    selected = [
        {
            "files": ["framework/py/flwr/cli/build.py"],
            "problem": "Unused import"
        }
    ]

    candidates = [
        {
            "candidate_id": "C001",
            "files": ["framework/py/flwr/common/exit/exit_code_test.py"],
            "problem": "Pylint issue",
            "frequency": 1
        },
        {
            "candidate_id": "C002",
            "files": ["framework/py/flwr/common/exit/exit_code_test.py"],
            "problem": "Docstring issue",
            "frequency": 1
        },
        {
            "candidate_id": "C003",
            "files": ["framework/py/flwr/cli/build.py"],
            "problem": "Unused import",
            "frequency": 1
        }
    ]

    # Run validation (should NOT add anything - frequency is 2, threshold is 3)
    validated = _validate_file_frequency_coverage(
        selected,
        candidates,
        min_frequency_threshold=3
    )

    print(f"\nFile frequency: 2 (below threshold of 3)")
    print(f"Problems before: {len(selected)}")
    print(f"Problems after: {len(validated)}")

    # Should be unchanged
    assert len(validated) == len(selected)

    print("\n✓ TEST 4 PASSED: No false positives (file with freq=2 not added when threshold=3)")

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("FILE FREQUENCY FIX - INTEGRATION TEST")
    print("="*80)
    print("\nTesting the fix for the exit_code_test.py priority bug...")
    print()

    try:
        # Run tests
        file_analysis = test_file_frequency_analysis()
        test_format_for_prompt(file_analysis)
        test_validation_safeguard()
        test_no_false_positives()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nThe file frequency fix is working correctly!")
        print("\nKey features verified:")
        print("  ✓ High-frequency files (3+ occurrences) are identified")
        print("  ✓ File frequency info is formatted for LLM prompt")
        print("  ✓ Validation safeguard adds missing high-frequency files")
        print("  ✓ No false positives (respects frequency threshold)")
        print()

        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
