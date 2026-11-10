#!/usr/bin/env python3
"""
Step 2: Partition Instances → History-Available vs Cold-Start

Takes repos from Step 1 and partitions 408 eval instances.

Output: results/cold_start/02_partition.json
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    print("\n" + "="*70)
    print("STEP 2: PARTITION INSTANCES")
    print("="*70 + "\n")

    # Load results from Step 1
    input_file = PROJECT_ROOT / "results/cold_start/01_instance_repos.json"

    if not input_file.exists():
        print("❌ Error: Step 1 output not found!")
        print(f"   Run: python 01_analyze_instances_and_repos.py first\n")
        return

    with open(input_file) as f:
        data = json.load(f)

    eval_instance_to_repo = data["eval_instances"]["instance_to_repo"]
    history_available_repos = set(data["repo_overlap"]["history_available_repos"])
    cold_start_repos = set(data["repo_overlap"]["cold_start_repos"])

    print(f"Loaded from Step 1:")
    print(f"  {len(eval_instance_to_repo)} eval instances")
    print(f"  {len(history_available_repos)} History-Available repos")
    print(f"  {len(cold_start_repos)} Cold-Start repos\n")

    # ========== Partition Instances ==========
    print("Partitioning instances...")

    history_available_instances = []
    cold_start_instances = []

    for inst_id, repo in eval_instance_to_repo.items():
        if repo in history_available_repos:
            history_available_instances.append(inst_id)
        elif repo in cold_start_repos:
            cold_start_instances.append(inst_id)

    print(f"✓ History-Available: {len(history_available_instances)} instances")
    print(f"✓ Cold-Start: {len(cold_start_instances)} instances")
    print(f"✓ Total: {len(history_available_instances) + len(cold_start_instances)}\n")

    # ========== Save Partition ==========
    partition = {
        "history_available": {
            "instance_ids": sorted(history_available_instances),
            "count": len(history_available_instances),
            "repos": sorted(list(history_available_repos)),
            "repo_count": len(history_available_repos),
        },
        "cold_start": {
            "instance_ids": sorted(cold_start_instances),
            "count": len(cold_start_instances),
            "repos": sorted(list(cold_start_repos)),
            "repo_count": len(cold_start_repos),
        },
        "total": len(history_available_instances) + len(cold_start_instances),
    }

    output_file = PROJECT_ROOT / "results/cold_start/02_partition.json"
    with open(output_file, 'w') as f:
        json.dump(partition, f, indent=2)

    print("="*70)
    print("STEP 2 COMPLETE")
    print("="*70)
    print(f"\n✓ Saved to: {output_file}")
    print(f"\nPartition:")
    print(f"  History-Available: {len(history_available_instances)} instances ({len(history_available_instances)/partition['total']*100:.1f}%)")
    print(f"  Cold-Start: {len(cold_start_instances)} instances ({len(cold_start_instances)/partition['total']*100:.1f}%)\n")


if __name__ == "__main__":
    main()
