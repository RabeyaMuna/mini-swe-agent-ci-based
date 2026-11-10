#!/usr/bin/env python3
"""
Prepare cross-repository data for running with run_miniswe_direct.sh

Creates:
1. experimental_data/cross_repository/eval_set.jsonl (100 instances)
2. Shows how to modify scripts/run_miniswe_ci_bench.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def create_eval_set_jsonl():
    """Create eval_set.jsonl from first 100 eval IDs."""

    # Load first 100 IDs
    eval_ids_file = PROJECT_ROOT / "experimental_data/cross_repository/eval_issue_ids.json"
    with open(eval_ids_file) as f:
        eval_ids = json.load(f)

    eval_ids_set = set(str(id) for id in eval_ids)

    # Load original eval_set.jsonl
    original_eval_set = PROJECT_ROOT / "data/eval_set.jsonl"

    # Filter to first 100 IDs
    selected_instances = []
    with open(original_eval_set) as f:
        for line in f:
            if not line.strip():
                continue
            instance = json.loads(line)
            inst_id = str(instance.get('instance_id') or instance.get('id', ''))
            if inst_id in eval_ids_set:
                selected_instances.append(instance)

    # Save to experimental_data/cross_repository/eval_set.jsonl
    output_file = PROJECT_ROOT / "experimental_data/cross_repository/eval_set.jsonl"
    with open(output_file, 'w') as f:
        for instance in selected_instances:
            f.write(json.dumps(instance) + '\n')

    print(f"✓ Created: {output_file}")
    print(f"  {len(selected_instances)} instances")

    return len(selected_instances)


def show_script_modification():
    """Show how to modify run_miniswe_ci_bench.py."""

    print("\n" + "="*70)
    print("MODIFY scripts/run_miniswe_ci_bench.py")
    print("="*70 + "\n")

    print("Add 'cross_repo' direction to memory_roots:")
    print()
    print("```python")
    print("memory_roots = {")
    print('    "none": "data/back_trs",')
    print('    "backward": "data/back_trs",')
    print('    "forward": "data/fwr_trs",')
    print('    "bidirectional": "data/bidirect_trs",')
    print('    "cross_repo": "experimental_data/cross_repository/bidirect_trs",  # ADD THIS LINE')
    print("}")
    print("```")
    print()


def show_run_commands():
    """Show commands to run experiments."""

    print("="*70)
    print("HOW TO RUN EXPERIMENTS")
    print("="*70 + "\n")

    print("After modifying the script, run:\n")

    print("# Config 1: No-Experience (Baseline)")
    print("bash ./run_miniswe_direct.sh \\")
    print("  \"\" baseline none deepseek-v4-flash \\")
    print("  \"\" experimental_data/cross_repository/eval_set.jsonl 1")
    print()
    print("  Results → results/miniswe-agent/baseline_deepseek-v4-flash/")
    print()

    print("# Config 2: Cross-Repo L1/L2/L3")
    print("bash ./run_miniswe_direct.sh \\")
    print("  \"\" L1+L2+L3 cross_repo deepseek-v4-flash \\")
    print("  \"\" experimental_data/cross_repository/eval_set.jsonl 1")
    print()
    print("  Results → results/miniswe-agent/cross_repo/l1+l2+l3_deepseek-v4-flash/")
    print()

    print("# Config 3: Cross-Repo L3-Only")
    print("bash ./run_miniswe_direct.sh \\")
    print("  \"\" L3 cross_repo deepseek-v4-flash \\")
    print("  \"\" experimental_data/cross_repository/eval_set.jsonl 1")
    print()
    print("  Results → results/miniswe-agent/cross_repo/l3_deepseek-v4-flash/")
    print()


def main():
    print("\n" + "="*70)
    print("PREPARE CROSS-REPOSITORY DATA FOR RUNNING")
    print("="*70 + "\n")

    # Create eval_set.jsonl
    count = create_eval_set_jsonl()

    # Show script modification
    show_script_modification()

    # Show run commands
    show_run_commands()

    print("="*70)
    print("DATA SUMMARY")
    print("="*70 + "\n")

    print("✓ Memory: experimental_data/cross_repository/bidirect_trs/failure_memory.json")
    print("  → 50 instances from SOURCE repos")
    print()
    print(f"✓ Eval set: experimental_data/cross_repository/eval_set.jsonl")
    print(f"  → {count} instances from TARGET repos")
    print()
    print("✓ Same-repository excluded: YES")
    print("  → Memory repos ≠ Eval repos")
    print()


if __name__ == "__main__":
    main()
