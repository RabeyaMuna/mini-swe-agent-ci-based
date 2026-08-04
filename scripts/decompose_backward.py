#!/usr/bin/env python3
"""
decompose_backward.py - CI-Based Backward Traces
=================================================

Backward trace approach: CI Failure -> Problem
Uses backward_decomposition/ module to analyze CI failures and build L1/L2/L3 memory.

Pipeline:
1. backward_decomposition/ - Extract problems from CI failures
2. build_memory/build_l1.py - Build L1 (failure sequences)
3. build_memory/build_l2.py - Build L2 (repair strategies)
4. build_memory/build_l3.py - Build L3 (universal patterns)

Output:
- data/back_trs/decomposed_issues.json
- data/back_trs/failure_memory.json (L1)
- data/back_trs/repo_memory.json (L2)
- data/back_trs/cross_memory.json (L3)

Usage:
    python scripts/decompose_backward.py --batch --use-huggingface --model minimax2.5 --limit 10
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backward_decomposition"))

# Import the main function from backward_decomposition module
from backward_decomposition.decompose_ci_failure import main as decompose_main


def main():
    """
    Main entry point for CI-based backward decomposition.

    This wrapper:
    1. Uses backward_decomposition/ module for CI failure analysis
    2. Builds L1/L2/L3 memory
    3. Saves to data/back_trs/ (backward traces)
    """
    print("=" * 80)
    print("CI-BASED DECOMPOSITION (Backward Traces: CI Failure -> Problem)")
    print("=" * 80)
    print("Using: backward_decomposition/ module")
    print("Output: data/back_trs/")
    print("=" * 80)
    print()

    # Call the main decomposition function
    # It already handles all args and saves to data/back_trs/
    decompose_main()

    print()
    print("=" * 80)
    print("OK CI-based decomposition complete!")
    print("=" * 80)
    print("Output saved to:")
    print("  - data/back_trs/decomposed_issues.json")
    print("  - data/back_trs/failure_memory.json (L1)")
    print("  - data/back_trs/repo_memory.json (L2)")
    print("  - data/back_trs/cross_memory.json (L3)")
    print("=" * 80)


if __name__ == "__main__":
    main()
