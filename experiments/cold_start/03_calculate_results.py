#!/usr/bin/env python3
"""
Step 3: Calculate Pass@1 for Each Model

For each model configuration:
- Loads baseline (no-experience) and ReCAP results
- Calculates Pass@1 for History-Available and Cold-Start groups

Output: results/cold_start/03_results_{model}.json
"""

import json
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent


# Model configurations
MODELS = {
    "deepseek_v4_flash": {
        "name": "DeepSeek-V4-Flash",
        "baseline_path": "results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl",
        "recap_path": "results/miniswe-agent/bidirectional/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/jobs_results_diff.jsonl",
    },
    "minimax_m2_5": {
        "name": "MiniMax-M2.5",
        "baseline_path": "results/miniswe-agent/baseline_minimax-m2.5/jobs_results_diff.jsonl",
        "recap_path": "results/miniswe-agent/bidirectional/l1_l2_l3_minimax-m2.5/L1_L2_L3/jobs_results_diff.jsonl",
    },
}


def load_results(jsonl_path, verbose=False):
    """Load results from JSONL file."""
    results = {}
    path = PROJECT_ROOT / jsonl_path

    if not path.exists():
        print(f"  ⚠️  File not found: {path}")
        return results

    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            inst_id = entry['id']
            passed = (entry['conclusion'] == 'success')
            results[inst_id] = passed

    if verbose:
        print(f"  Loaded {len(results)} results from {path.name}")

    return results


def calculate_pass_at_1(instance_ids, results, verbose=False):
    """Calculate Pass@1 for a group."""
    passed = sum(1 for inst_id in instance_ids if results.get(inst_id, False))
    total = len(instance_ids)

    if total == 0:
        return 0.0, 0, total

    pass_rate = (passed / total) * 100

    if verbose:
        print(f"  {passed}/{total} = {pass_rate:.1f}%")

    return pass_rate, passed, total


def calculate_for_model(model_key, partition, verbose=False):
    """Calculate results for one model."""

    model = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"MODEL: {model['name']}")
    print(f"{'='*70}\n")

    # Load results
    print("Loading results...")
    baseline_results = load_results(model['baseline_path'], verbose)
    recap_results = load_results(model['recap_path'], verbose)

    if not baseline_results or not recap_results:
        print(f"❌ Missing results for {model['name']}")
        return None

    print()

    # Get instance IDs from partition
    ha_ids = partition["history_available"]["instance_ids"]
    cs_ids = partition["cold_start"]["instance_ids"]
    all_ids = ha_ids + cs_ids

    # Calculate for History-Available
    print("History-Available:")
    ha_baseline_rate, ha_baseline_passed, ha_baseline_total = calculate_pass_at_1(ha_ids, baseline_results, verbose)
    ha_recap_rate, ha_recap_passed, ha_recap_total = calculate_pass_at_1(ha_ids, recap_results, verbose)

    print(f"  Baseline: {ha_baseline_passed}/{ha_baseline_total} = {ha_baseline_rate:.1f}%")
    print(f"  ReCAP:    {ha_recap_passed}/{ha_recap_total} = {ha_recap_rate:.1f}%")
    print(f"  Gain:     +{ha_recap_rate - ha_baseline_rate:.1f} pp\n")

    # Calculate for Cold-Start
    print("Cold-Start:")
    cs_baseline_rate, cs_baseline_passed, cs_baseline_total = calculate_pass_at_1(cs_ids, baseline_results, verbose)
    cs_recap_rate, cs_recap_passed, cs_recap_total = calculate_pass_at_1(cs_ids, recap_results, verbose)

    print(f"  Baseline: {cs_baseline_passed}/{cs_baseline_total} = {cs_baseline_rate:.1f}%")
    print(f"  ReCAP:    {cs_recap_passed}/{cs_recap_total} = {cs_recap_rate:.1f}%")
    print(f"  Gain:     +{cs_recap_rate - cs_baseline_rate:.1f} pp\n")

    # Calculate Overall
    print("Overall:")
    overall_baseline_rate, overall_baseline_passed, overall_baseline_total = calculate_pass_at_1(all_ids, baseline_results, verbose)
    overall_recap_rate, overall_recap_passed, overall_recap_total = calculate_pass_at_1(all_ids, recap_results, verbose)

    print(f"  Baseline: {overall_baseline_passed}/{overall_baseline_total} = {overall_baseline_rate:.1f}%")
    print(f"  ReCAP:    {overall_recap_passed}/{overall_recap_total} = {overall_recap_rate:.1f}%")
    print(f"  Gain:     +{overall_recap_rate - overall_baseline_rate:.1f} pp")

    # Build results dict
    results = {
        "model": model['name'],
        "model_key": model_key,
        "history_available": {
            "instances": ha_baseline_total,
            "baseline": {
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
            "baseline": {
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
            "baseline": {
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

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                       help="Model to calculate results for")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("STEP 3: CALCULATE RESULTS FOR EACH MODEL")
    print("="*70)

    # Load partition from Step 2
    partition_file = PROJECT_ROOT / "results/cold_start/02_partition.json"

    if not partition_file.exists():
        print("\n❌ Error: Step 2 output not found!")
        print("   Run: python 02_partition_instances.py first\n")
        return

    with open(partition_file) as f:
        partition = json.load(f)

    print(f"\nLoaded partition:")
    print(f"  History-Available: {partition['history_available']['count']} instances")
    print(f"  Cold-Start: {partition['cold_start']['count']} instances")

    # Calculate for requested models
    models_to_run = [args.model] if args.model != "all" else list(MODELS.keys())

    for model_key in models_to_run:
        results = calculate_for_model(model_key, partition, args.verbose)

        if results:
            # Save results
            output_file = PROJECT_ROOT / f"results/cold_start/03_results_{model_key}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Saved to: {output_file}")

    print("\n" + "="*70)
    print("STEP 3 COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
