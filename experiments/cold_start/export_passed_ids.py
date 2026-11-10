#!/usr/bin/env python3
"""
Export all passed instance IDs for each configuration.

Outputs:
- results/cold_start/passed_ids_history_available.json
- results/cold_start/passed_ids_cold_start.json
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_results(jsonl_path):
    """Load results from JSONL file."""
    results = {}
    path = PROJECT_ROOT / jsonl_path

    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            inst_id = str(entry['id'])
            passed = (entry['conclusion'] == 'success')
            results[inst_id] = passed

    return results


def main():
    print("\n" + "="*70)
    print("EXPORT PASSED INSTANCE IDs")
    print("="*70 + "\n")

    # Load partition
    partition_file = PROJECT_ROOT / "results/cold_start/02_partition.json"
    with open(partition_file) as f:
        partition = json.load(f)

    ha_ids = [str(i) for i in partition["history_available"]["instance_ids"]]
    cs_ids = [str(i) for i in partition["cold_start"]["instance_ids"]]

    # Load all results
    print("Loading results...")
    ds_baseline = load_results("results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl")
    ds_recap = load_results("results/miniswe-agent/bidirectional/l1_l2_l3_deepseek-v4-flash/L1_L2_L3/jobs_results_diff.jsonl")
    mm_baseline = load_results("results/miniswe-agent/baseline_minimax-m2.5/jobs_results_diff.jsonl")
    mm_recap = load_results("results/miniswe-agent/bidirectional/l1_l2_l3_minimax-m2.5/L1_L2_L3/jobs_results_diff.jsonl")
    print("✓ All results loaded\n")

    # ========== History-Available Passed IDs ==========
    print("="*70)
    print("HISTORY-AVAILABLE PASSED IDs")
    print("="*70 + "\n")

    ha_deepseek_baseline_passed = sorted([i for i in ha_ids if ds_baseline.get(i, False)])
    ha_deepseek_recap_passed = sorted([i for i in ha_ids if ds_recap.get(i, False)])
    ha_minimax_baseline_passed = sorted([i for i in ha_ids if mm_baseline.get(i, False)])
    ha_minimax_recap_passed = sorted([i for i in ha_ids if mm_recap.get(i, False)])

    print(f"DeepSeek Baseline ({len(ha_deepseek_baseline_passed)}/375):")
    print(f"  {ha_deepseek_baseline_passed}\n")

    print(f"DeepSeek ReCAP ({len(ha_deepseek_recap_passed)}/375):")
    print(f"  {ha_deepseek_recap_passed}\n")

    print(f"MiniMax Baseline ({len(ha_minimax_baseline_passed)}/375):")
    print(f"  {ha_minimax_baseline_passed}\n")

    print(f"MiniMax ReCAP ({len(ha_minimax_recap_passed)}/375):")
    print(f"  {ha_minimax_recap_passed}\n")

    # ========== Cold-Start Passed IDs ==========
    print("="*70)
    print("COLD-START PASSED IDs")
    print("="*70 + "\n")

    cs_deepseek_baseline_passed = sorted([i for i in cs_ids if ds_baseline.get(i, False)])
    cs_deepseek_recap_passed = sorted([i for i in cs_ids if ds_recap.get(i, False)])
    cs_minimax_baseline_passed = sorted([i for i in cs_ids if mm_baseline.get(i, False)])
    cs_minimax_recap_passed = sorted([i for i in cs_ids if mm_recap.get(i, False)])

    print(f"DeepSeek Baseline ({len(cs_deepseek_baseline_passed)}/33):")
    print(f"  {cs_deepseek_baseline_passed}\n")

    print(f"DeepSeek ReCAP ({len(cs_deepseek_recap_passed)}/33):")
    print(f"  {cs_deepseek_recap_passed}\n")

    print(f"MiniMax Baseline ({len(cs_minimax_baseline_passed)}/33):")
    print(f"  {cs_minimax_baseline_passed}\n")

    print(f"MiniMax ReCAP ({len(cs_minimax_recap_passed)}/33):")
    print(f"  {cs_minimax_recap_passed}\n")

    # ========== Save to JSON Files ==========
    output_ha = {
        "group": "History-Available",
        "total_instances": len(ha_ids),
        "deepseek_v4_flash": {
            "baseline": {
                "passed_count": len(ha_deepseek_baseline_passed),
                "passed_ids": ha_deepseek_baseline_passed
            },
            "recap": {
                "passed_count": len(ha_deepseek_recap_passed),
                "passed_ids": ha_deepseek_recap_passed
            }
        },
        "minimax_m2_5": {
            "baseline": {
                "passed_count": len(ha_minimax_baseline_passed),
                "passed_ids": ha_minimax_baseline_passed
            },
            "recap": {
                "passed_count": len(ha_minimax_recap_passed),
                "passed_ids": ha_minimax_recap_passed
            }
        }
    }

    output_cs = {
        "group": "Cold-Start",
        "total_instances": len(cs_ids),
        "deepseek_v4_flash": {
            "baseline": {
                "passed_count": len(cs_deepseek_baseline_passed),
                "passed_ids": cs_deepseek_baseline_passed
            },
            "recap": {
                "passed_count": len(cs_deepseek_recap_passed),
                "passed_ids": cs_deepseek_recap_passed
            }
        },
        "minimax_m2_5": {
            "baseline": {
                "passed_count": len(cs_minimax_baseline_passed),
                "passed_ids": cs_minimax_baseline_passed
            },
            "recap": {
                "passed_count": len(cs_minimax_recap_passed),
                "passed_ids": cs_minimax_recap_passed
            }
        }
    }

    results_dir = PROJECT_ROOT / "results/cold_start"

    output_ha_file = results_dir / "passed_ids_history_available.json"
    with open(output_ha_file, 'w') as f:
        json.dump(output_ha, f, indent=2)
    print(f"✓ Saved: {output_ha_file}")

    output_cs_file = results_dir / "passed_ids_cold_start.json"
    with open(output_cs_file, 'w') as f:
        json.dump(output_cs, f, indent=2)
    print(f"✓ Saved: {output_cs_file}")

    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
