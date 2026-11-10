#!/usr/bin/env python3
"""
Show detailed instance partition for History-Available vs Cold-Start.

For each repo, show which specific instances belong to that repo.
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    """Show detailed instance breakdown by repo."""

    print()
    print("="*70)
    print("DETAILED INSTANCE PARTITION")
    print("="*70)
    print()

    # Load partition
    with open(PROJECT_ROOT / "results/cold_start/partition.json") as f:
        partition = json.load(f)

    # Load instance -> repo mapping
    with open(PROJECT_ROOT / "results/cold_start/instance_to_repo.json") as f:
        instance_to_repo = json.load(f)

    # Group instances by repo for each partition
    history_available_ids = set(partition["history_available_instances"]["instance_ids"])
    cold_start_ids = set(partition["cold_start_instances"]["instance_ids"])

    # Build repo -> instances mapping
    ha_repo_to_instances = defaultdict(list)
    cs_repo_to_instances = defaultdict(list)

    for inst_id, repo in instance_to_repo.items():
        if inst_id in history_available_ids:
            ha_repo_to_instances[repo].append(inst_id)
        elif inst_id in cold_start_ids:
            cs_repo_to_instances[repo].append(inst_id)

    # Load memory instances for reference
    with open(PROJECT_ROOT / "data/bidirect_trs/failure_memory.json") as f:
        l1_data = json.load(f)

    memory_repo_to_instances = defaultdict(list)
    for entry in l1_data:
        repo = entry["repo"]
        if '/' in repo:
            repo = repo.split('/')[-1]
        memory_repo_to_instances[repo].append(entry["issue_id"])

    # ========== History-Available Instances ==========
    print("HISTORY-AVAILABLE INSTANCES (375 total)")
    print("="*70)
    print("Repos with historical repair experience")
    print()

    total_ha = 0
    for repo in sorted(ha_repo_to_instances.keys()):
        instances = sorted(ha_repo_to_instances[repo])
        memory_instances = sorted(memory_repo_to_instances.get(repo, []))

        print(f"{repo}:")
        print(f"  Eval instances ({len(instances)}): {', '.join(instances)}")
        print(f"  Memory instances ({len(memory_instances)}): {', '.join(memory_instances)}")
        print()

        total_ha += len(instances)

    print(f"Total History-Available: {total_ha} instances")
    print()

    # ========== Cold-Start Instances ==========
    print()
    print("COLD-START INSTANCES (33 total)")
    print("="*70)
    print("Repos WITHOUT historical repair experience")
    print()

    total_cs = 0
    for repo in sorted(cs_repo_to_instances.keys()):
        instances = sorted(cs_repo_to_instances[repo])

        print(f"{repo}:")
        print(f"  Eval instances ({len(instances)}): {', '.join(instances)}")
        print(f"  Memory instances: NONE (cold-start)")
        print()

        total_cs += len(instances)

    print(f"Total Cold-Start: {total_cs} instances")
    print()

    # ========== Save Detailed Partition ==========
    detailed_partition = {
        "history_available": {
            "total_instances": total_ha,
            "total_repos": len(ha_repo_to_instances),
            "repos": {
                repo: {
                    "eval_instances": sorted(instances),
                    "memory_instances": sorted(memory_repo_to_instances.get(repo, [])),
                }
                for repo, instances in ha_repo_to_instances.items()
            }
        },
        "cold_start": {
            "total_instances": total_cs,
            "total_repos": len(cs_repo_to_instances),
            "repos": {
                repo: {
                    "eval_instances": sorted(instances),
                    "memory_instances": [],
                }
                for repo, instances in cs_repo_to_instances.items()
            }
        }
    }

    output_file = PROJECT_ROOT / "results/cold_start/detailed_partition.json"
    with open(output_file, 'w') as f:
        json.dump(detailed_partition, f, indent=2)

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"History-Available:")
    print(f"  {total_ha} instances from {len(ha_repo_to_instances)} repos")
    print()
    print(f"Cold-Start:")
    print(f"  {total_cs} instances from {len(cs_repo_to_instances)} repos")
    print()
    print(f"✓ Saved to {output_file}")


if __name__ == "__main__":
    main()
