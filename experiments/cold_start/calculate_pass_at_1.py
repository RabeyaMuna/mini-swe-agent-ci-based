#!/usr/bin/env python3
"""
Calculate Pass@1 for History-Available vs Cold-Start groups.

Uses existing evaluation results (no new runs needed).
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_results(jsonl_path):
    """Load results from JSONL file."""
    results = {}
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            inst_id = entry['id']
            conclusion = entry['conclusion']  # 'success' or 'failure'
            results[inst_id] = (conclusion == 'success')
    return results


def calculate_pass_at_1(instance_ids, results):
    """Calculate Pass@1 for a group of instances."""
    total = len(instance_ids)
    passed = sum(1 for inst_id in instance_ids if results.get(inst_id, False))

    if total == 0:
        return 0.0, 0, 0

    pass_rate = (passed / total) * 100
    return pass_rate, passed, total


def main():
    """Calculate Pass@1 for both groups using existing results."""

    print()
    print("="*70)
    print("CALCULATING PASS@1 FOR HISTORY-AVAILABLE VS COLD-START")
    print("="*70)
    print()

    # Load partition
    with open(PROJECT_ROOT / "results/cold_start/partition.json") as f:
        partition = json.load(f)

    history_available_ids = partition["history_available_instances"]["instance_ids"]
    cold_start_ids = partition["cold_start_instances"]["instance_ids"]

    print(f"History-Available: {len(history_available_ids)} instances")
    print(f"Cold-Start: {len(cold_start_ids)} instances")
    print()

    # Paths to results
    baseline_path = PROJECT_ROOT / "results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl"
    recap_path = PROJECT_ROOT / "results/miniswe-agent/bidirectional/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/jobs_results_diff.jsonl"

    # Load results
    print("Loading results...")
    baseline_results = load_results(baseline_path)
    recap_results = load_results(recap_path)

    print(f"  Baseline: {len(baseline_results)} instances")
    print(f"  ReCAP: {len(recap_results)} instances")
    print()

    # Calculate Pass@1 for each group
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    # History-Available
    print("History-Available Group (375 instances):")
    ha_baseline_rate, ha_baseline_passed, ha_baseline_total = calculate_pass_at_1(
        history_available_ids, baseline_results
    )
    ha_recap_rate, ha_recap_passed, ha_recap_total = calculate_pass_at_1(
        history_available_ids, recap_results
    )

    print(f"  No-Experience:  {ha_baseline_passed}/{ha_baseline_total} = {ha_baseline_rate:.1f}%")
    print(f"  ReCAP:          {ha_recap_passed}/{ha_recap_total} = {ha_recap_rate:.1f}%")
    print(f"  Gain:           +{ha_recap_rate - ha_baseline_rate:.1f} pp")
    print()

    # Cold-Start
    print("Cold-Start Group (33 instances):")
    cs_baseline_rate, cs_baseline_passed, cs_baseline_total = calculate_pass_at_1(
        cold_start_ids, baseline_results
    )
    cs_recap_rate, cs_recap_passed, cs_recap_total = calculate_pass_at_1(
        cold_start_ids, recap_results
    )

    print(f"  No-Experience:  {cs_baseline_passed}/{cs_baseline_total} = {cs_baseline_rate:.1f}%")
    print(f"  ReCAP:          {cs_recap_passed}/{cs_recap_total} = {cs_recap_rate:.1f}%")
    print(f"  Gain:           +{cs_recap_rate - cs_baseline_rate:.1f} pp")
    print()

    # Overall (verify against known values)
    print("Overall (408 instances):")
    all_ids = history_available_ids + cold_start_ids
    overall_baseline_rate, overall_baseline_passed, overall_baseline_total = calculate_pass_at_1(
        all_ids, baseline_results
    )
    overall_recap_rate, overall_recap_passed, overall_recap_total = calculate_pass_at_1(
        all_ids, recap_results
    )

    print(f"  No-Experience:  {overall_baseline_passed}/{overall_baseline_total} = {overall_baseline_rate:.1f}%")
    print(f"  ReCAP:          {overall_recap_passed}/{overall_recap_total} = {overall_recap_rate:.1f}%")
    print(f"  Gain:           +{overall_recap_rate - overall_baseline_rate:.1f} pp")
    print()

    # Generate Table R7
    table_r7 = {
        "model": "DeepSeek-V4-Flash",
        "agent": "mini-SWE-agent",
        "history_available": {
            "instances": ha_baseline_total,
            "no_experience": {
                "passed": ha_baseline_passed,
                "total": ha_baseline_total,
                "pass_rate": round(ha_baseline_rate, 1)
            },
            "recap": {
                "passed": ha_recap_passed,
                "total": ha_recap_total,
                "pass_rate": round(ha_recap_rate, 1)
            },
            "gain_pp": round(ha_recap_rate - ha_baseline_rate, 1)
        },
        "cold_start": {
            "instances": cs_baseline_total,
            "no_experience": {
                "passed": cs_baseline_passed,
                "total": cs_baseline_total,
                "pass_rate": round(cs_baseline_rate, 1)
            },
            "recap": {
                "passed": cs_recap_passed,
                "total": cs_recap_total,
                "pass_rate": round(cs_recap_rate, 1)
            },
            "gain_pp": round(cs_recap_rate - cs_baseline_rate, 1)
        },
        "overall": {
            "instances": overall_baseline_total,
            "no_experience": {
                "passed": overall_baseline_passed,
                "total": overall_baseline_total,
                "pass_rate": round(overall_baseline_rate, 1)
            },
            "recap": {
                "passed": overall_recap_passed,
                "total": overall_recap_total,
                "pass_rate": round(overall_recap_rate, 1)
            },
            "gain_pp": round(overall_recap_rate - overall_baseline_rate, 1)
        }
    }

    # Save results
    output_file = PROJECT_ROOT / "results/cold_start/table_r7.json"
    with open(output_file, 'w') as f:
        json.dump(table_r7, f, indent=2)

    print("="*70)
    print("TABLE R7 DATA")
    print("="*70)
    print()
    print("| Repository Group | Instances | No-Experience Pass@1 (%) | ReCAP Pass@1 (%) | Gain (pp) |")
    print("|------------------|-----------|--------------------------|------------------|-----------|")
    print(f"| History-Available | {ha_baseline_total} | {ha_baseline_rate:.1f} | {ha_recap_rate:.1f} | +{ha_recap_rate - ha_baseline_rate:.1f} |")
    print(f"| Cold-Start | {cs_baseline_total} | {cs_baseline_rate:.1f} | {cs_recap_rate:.1f} | +{cs_recap_rate - cs_baseline_rate:.1f} |")
    print(f"| Overall | {overall_baseline_total} | {overall_baseline_rate:.1f} | {overall_recap_rate:.1f} | +{overall_recap_rate - overall_baseline_rate:.1f} |")
    print()
    print(f"✓ Saved to {output_file}")


if __name__ == "__main__":
    main()
