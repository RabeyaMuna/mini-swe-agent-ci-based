#!/usr/bin/env python3
"""
Run Complete Cold-Start Analysis

Executes all 4 steps in sequence:
1. Analyze instances and repos
2. Partition instances
3. Calculate results for all models
4. Generate Table R7

Usage:
  python run_analysis.py              # Run all steps
  python run_analysis.py --from-step 3   # Start from step 3
"""

import subprocess
import sys
from pathlib import Path
import argparse

SCRIPT_DIR = Path(__file__).parent


def run_step(step_num, script_name, description):
    """Run one analysis step."""
    print("\n" + "╔" + "="*68 + "╗")
    print(f"║  STEP {step_num}: {description:<58} ║")
    print("╚" + "="*68 + "╝\n")

    script_path = SCRIPT_DIR / script_name
    result = subprocess.run([sys.executable, str(script_path)])

    if result.returncode != 0:
        print(f"\n❌ Step {step_num} failed!")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run cold-start analysis")
    parser.add_argument("--from-step", type=int, choices=[1,2,3,4], default=1,
                       help="Start from this step (default: 1)")
    args = parser.parse_args()

    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "COLD-START ANALYSIS - COMPLETE" + " "*23 + "║")
    print("╚" + "="*68 + "╝")

    steps = [
        (1, "01_analyze_instances_and_repos.py", "Analyze Instances & Repos"),
        (2, "02_partition_instances.py", "Partition Instances"),
        (3, "03_calculate_results.py", "Calculate Results"),
        (4, "04_generate_table_r7.py", "Generate Table R7"),
    ]

    # Run steps starting from requested step
    for step_num, script_name, description in steps:
        if step_num >= args.from_step:
            success = run_step(step_num, script_name, description)
            if not success:
                sys.exit(1)

    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "ANALYSIS COMPLETE!" + " "*29 + "║")
    print("╚" + "="*68 + "╝\n")

    print("Generated files:")
    print("  1. results/cold_start/01_instance_repos.json")
    print("  2. results/cold_start/02_partition.json")
    print("  3. results/cold_start/03_results_*.json (per model)")
    print("  4. results/cold_start/table_r7.json")
    print("  5. results/cold_start/table_r7.md")
    print("  6. results/cold_start/table_r7.tex\n")

    print("View results:")
    print("  cat results/cold_start/table_r7.md\n")


if __name__ == "__main__":
    main()
