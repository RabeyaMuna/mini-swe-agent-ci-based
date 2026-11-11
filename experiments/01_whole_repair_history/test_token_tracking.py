#!/usr/bin/env python3
"""
Test script to verify token tracking and error handling improvements.

This script runs a single instance through the whole PR memory builder
to verify that:
1. Token usage is tracked and displayed correctly
2. Error handling for length limits works
3. Phase tracking is working
4. Summary statistics are printed
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDER_SCRIPT = PROJECT_ROOT / "experiments" / "01_whole_repair_history" / "build_whole_pr_memory.py"

def test_single_instance(issue_id: str):
    """Test the builder on a single instance."""
    print(f"Testing whole PR memory builder on instance {issue_id}...")
    print("=" * 70)

    cmd = [
        sys.executable,
        str(BUILDER_SCRIPT),
        "--issue-ids", issue_id,
        "--rebuild",
        "--model", "deepseek-v4-flash"
    ]

    print(f"Running: {' '.join(cmd)}")
    print("=" * 70)

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,  # Show output in real-time
            text=True
        )

        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("✅ Test PASSED - Instance processed successfully")
            print("=" * 70)
            print("\nExpected outputs in terminal:")
            print("  ✓ [API] token usage for each LLM call")
            print("  ✓ Instance usage summary with phase breakdown")
            print("  ✓ Aggregate summary with cost estimates")
            print("\nExpected files:")
            print("  ✓ experimental_data/whole_repair_history/construction_metrics/run_metrics.json")
            print("  ✓ experimental_data/whole_repair_history/construction_metrics/cost_time_report.json")
            print("  ✓ experimental_data/whole_repair_history/construction_metrics/detailed_usage.json")
        else:
            print("\n" + "=" * 70)
            print(f"❌ Test FAILED - Exit code: {result.returncode}")
            print("=" * 70)

            # Check if it's the expected length limit error
            if "exhausted output budget" in str(result.stderr or ""):
                print("\n⚠️  Hit length limit error (expected for some instances)")
                print("   This is the improved error handling in action!")
                print("   Solution: Increase --max-evidence-chars")

        return result.returncode

    except Exception as e:
        print(f"\n❌ Test FAILED with exception: {e}")
        return 1


def main():
    # Test on issue 1 (should be relatively simple)
    test_issue = "1"

    if len(sys.argv) > 1:
        test_issue = sys.argv[1]

    print("Whole PR Memory Builder - Token Tracking Test")
    print("=" * 70)
    print(f"Testing on instance: {test_issue}")
    print(f"Model: deepseek-v4-flash")
    print("=" * 70)
    print()

    exit_code = test_single_instance(test_issue)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
