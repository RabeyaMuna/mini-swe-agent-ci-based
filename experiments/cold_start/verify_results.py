#!/usr/bin/env python3
"""
Verify Pass@1 calculations with detailed breakdown.

Show which specific instances passed in each group.
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_results_detailed(jsonl_path):
    """Load results with full details."""
    results = {}
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            inst_id = entry['id']
            results[inst_id] = {
                'conclusion': entry['conclusion'],
                'passed': entry['conclusion'] == 'success',
                'repo': entry.get('repo_name', 'unknown')
            }
    return results


def verify_group(group_name, instance_ids, baseline_results, recap_results):
    """Verify Pass@1 for a group with detailed output."""

    print(f"\n{'='*70}")
    print(f"{group_name.upper()} - DETAILED VERIFICATION")
    print(f"{'='*70}\n")

    print(f"Total instances: {len(instance_ids)}\n")

    # Check baseline
    baseline_passed = []
    baseline_failed = []
    baseline_missing = []

    for inst_id in instance_ids:
        if inst_id not in baseline_results:
            baseline_missing.append(inst_id)
        elif baseline_results[inst_id]['passed']:
            baseline_passed.append(inst_id)
        else:
            baseline_failed.append(inst_id)

    # Check ReCAP
    recap_passed = []
    recap_failed = []
    recap_missing = []

    for inst_id in instance_ids:
        if inst_id not in recap_results:
            recap_missing.append(inst_id)
        elif recap_results[inst_id]['passed']:
            recap_passed.append(inst_id)
        else:
            recap_failed.append(inst_id)

    # Report baseline
    print("BASELINE (No-Experience):")
    print(f"  Passed: {len(baseline_passed)}/{len(instance_ids)} = {len(baseline_passed)/len(instance_ids)*100:.1f}%")
    print(f"  Failed: {len(baseline_failed)}")
    print(f"  Missing: {len(baseline_missing)}")

    if baseline_missing:
        print(f"  Missing IDs: {baseline_missing}")

    print(f"\n  Passed instances: {sorted(baseline_passed)[:10]}{'...' if len(baseline_passed) > 10 else ''}")
    print()

    # Report ReCAP
    print("ReCAP (Bidirectional L1+L2+L3):")
    print(f"  Passed: {len(recap_passed)}/{len(instance_ids)} = {len(recap_passed)/len(instance_ids)*100:.1f}%")
    print(f"  Failed: {len(recap_failed)}")
    print(f"  Missing: {len(recap_missing)}")

    if recap_missing:
        print(f"  Missing IDs: {recap_missing}")

    print(f"\n  Passed instances: {sorted(recap_passed)[:10]}{'...' if len(recap_passed) > 10 else ''}")
    print()

    # Calculate gain
    baseline_rate = len(baseline_passed) / len(instance_ids) * 100
    recap_rate = len(recap_passed) / len(instance_ids) * 100
    gain = recap_rate - baseline_rate

    print(f"GAIN: +{gain:.1f} percentage points")

    # Show improvements
    improved = [inst_id for inst_id in instance_ids
                if inst_id in baseline_results and inst_id in recap_results
                and not baseline_results[inst_id]['passed']
                and recap_results[inst_id]['passed']]

    regressed = [inst_id for inst_id in instance_ids
                 if inst_id in baseline_results and inst_id in recap_results
                 and baseline_results[inst_id]['passed']
                 and not recap_results[inst_id]['passed']]

    print(f"\nReCAP improved: {len(improved)} instances")
    if improved:
        print(f"  Example improved: {improved[:5]}")

    print(f"ReCAP regressed: {len(regressed)} instances")
    if regressed:
        print(f"  Regressed: {regressed}")

    return {
        'baseline_passed': len(baseline_passed),
        'baseline_rate': baseline_rate,
        'recap_passed': len(recap_passed),
        'recap_rate': recap_rate,
        'gain': gain,
        'improved': len(improved),
        'regressed': len(regressed)
    }


def main():
    """Verify Pass@1 calculations."""

    print("\n" + "="*70)
    print("VERIFYING PASS@1 CALCULATIONS")
    print("="*70)

    # Load partition
    with open(PROJECT_ROOT / "results/cold_start/partition.json") as f:
        partition = json.load(f)

    history_available_ids = partition["history_available_instances"]["instance_ids"]
    cold_start_ids = partition["cold_start_instances"]["instance_ids"]

    # Load results
    baseline_path = PROJECT_ROOT / "results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl"
    recap_path = PROJECT_ROOT / "results/miniswe-agent/bidirectional/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/jobs_results_diff.jsonl"

    print(f"\nLoading results from:")
    print(f"  Baseline: {baseline_path.name}")
    print(f"  ReCAP: {recap_path.name}")

    baseline_results = load_results_detailed(baseline_path)
    recap_results = load_results_detailed(recap_path)

    print(f"\nLoaded:")
    print(f"  Baseline: {len(baseline_results)} results")
    print(f"  ReCAP: {len(recap_results)} results")

    # Verify History-Available
    ha_stats = verify_group(
        "History-Available (375 instances)",
        history_available_ids,
        baseline_results,
        recap_results
    )

    # Verify Cold-Start
    cs_stats = verify_group(
        "Cold-Start (33 instances)",
        cold_start_ids,
        baseline_results,
        recap_results
    )

    # Overall summary
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print()
    print("| Group | Instances | Baseline Pass | Baseline % | ReCAP Pass | ReCAP % | Gain |")
    print("|-------|-----------|---------------|------------|------------|---------|------|")
    print(f"| History-Available | 375 | {ha_stats['baseline_passed']} | {ha_stats['baseline_rate']:.1f}% | {ha_stats['recap_passed']} | {ha_stats['recap_rate']:.1f}% | +{ha_stats['gain']:.1f} pp |")
    print(f"| Cold-Start | 33 | {cs_stats['baseline_passed']} | {cs_stats['baseline_rate']:.1f}% | {cs_stats['recap_passed']} | {cs_stats['recap_rate']:.1f}% | +{cs_stats['gain']:.1f} pp |")
    print()

    # Verification summary
    print("="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print()
    print("✓ History-Available:")
    print(f"  - {ha_stats['baseline_passed']} baseline passes → {ha_stats['recap_passed']} ReCAP passes")
    print(f"  - {ha_stats['improved']} improvements, {ha_stats['regressed']} regressions")
    print()
    print("✓ Cold-Start:")
    print(f"  - {cs_stats['baseline_passed']} baseline passes → {cs_stats['recap_passed']} ReCAP passes")
    print(f"  - {cs_stats['improved']} improvements, {cs_stats['regressed']} regressions")


if __name__ == "__main__":
    main()
