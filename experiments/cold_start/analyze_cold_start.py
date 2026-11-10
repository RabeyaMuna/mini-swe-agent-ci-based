#!/usr/bin/env python3
"""
Complete Cold-Start Analysis

Now that we have repo info for both memory and eval instances:
1. Define History-Available vs Cold-Start repos
2. Partition 408 eval instances
3. Report breakdown
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    """Run complete cold-start analysis."""

    print()
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "COLD-START ANALYSIS" + " "*29 + "║")
    print("╚" + "="*68 + "╝")
    print()

    # ========== STEP 1: Load Memory Repos ==========
    print("STEP 1: Load Memory Repositories (from L1 memory)")
    print("-" * 70)

    with open(PROJECT_ROOT / "data/bidirect_trs/failure_memory.json") as f:
        l1_data = json.load(f)

    memory_repos = set()
    memory_repo_counts = defaultdict(int)

    for entry in l1_data:
        # Normalize repo name (just the repo part, not owner/repo)
        repo = entry["repo"]
        if '/' in repo:
            repo_name = repo.split('/')[-1]
        else:
            repo_name = repo

        memory_repos.add(repo_name)
        memory_repo_counts[repo_name] += 1

    print(f"  Memory instances: {len(l1_data)}")
    print(f"  Unique repos in memory: {len(memory_repos)}")
    print()

    # ========== STEP 2: Load Eval Instance -> Repo Mapping ==========
    print("STEP 2: Load Evaluation Instance Repositories")
    print("-" * 70)

    with open(PROJECT_ROOT / "results/cold_start/instance_to_repo.json") as f:
        eval_instance_to_repo = json.load(f)

    eval_repos = set(eval_instance_to_repo.values())

    print(f"  Eval instances: {len(eval_instance_to_repo)}")
    print(f"  Unique repos in eval: {len(eval_repos)}")
    print()

    # ========== STEP 3: Define Repo Partition ==========
    print("STEP 3: Partition Repositories")
    print("-" * 70)

    history_available_repos = memory_repos & eval_repos  # Repos in BOTH
    cold_start_repos = eval_repos - memory_repos         # Repos ONLY in eval

    print(f"  History-Available repos: {len(history_available_repos)}")
    print(f"    (repos with historical repair experience)")
    print()
    print(f"  Cold-Start repos: {len(cold_start_repos)}")
    print(f"    (repos WITHOUT historical repair experience)")
    print()

    # ========== STEP 4: Partition Eval Instances ==========
    print("STEP 4: Partition Evaluation Instances")
    print("-" * 70)

    history_available_instances = []
    cold_start_instances = []

    for inst_id, repo in eval_instance_to_repo.items():
        if repo in history_available_repos:
            history_available_instances.append(inst_id)
        else:
            cold_start_instances.append(inst_id)

    print(f"  History-Available instances: {len(history_available_instances)}")
    print(f"  Cold-Start instances: {len(cold_start_instances)}")
    print(f"  Total: {len(history_available_instances) + len(cold_start_instances)}")
    print()

    # ========== STEP 5: Show Distribution ==========
    print("STEP 5: Repository Distribution")
    print("-" * 70)

    print("\nHistory-Available repos (top 10):")
    ha_repo_counts = defaultdict(int)
    for inst_id in history_available_instances:
        ha_repo_counts[eval_instance_to_repo[inst_id]] += 1

    for repo, count in sorted(ha_repo_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        mem_count = memory_repo_counts.get(repo, 0)
        print(f"  {repo}: {count} eval instances ({mem_count} in memory)")

    print("\nCold-Start repos (top 10):")
    cs_repo_counts = defaultdict(int)
    for inst_id in cold_start_instances:
        cs_repo_counts[eval_instance_to_repo[inst_id]] += 1

    for repo, count in sorted(cs_repo_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {repo}: {count} eval instances (0 in memory)")

    # ========== Save Results ==========
    output = {
        "memory_repos": {
            "total": len(memory_repos),
            "repos": sorted(list(memory_repos)),
            "repo_counts": dict(memory_repo_counts),
        },
        "eval_repos": {
            "total": len(eval_repos),
            "repos": sorted(list(eval_repos)),
        },
        "history_available_repos": {
            "total": len(history_available_repos),
            "repos": sorted(list(history_available_repos)),
        },
        "cold_start_repos": {
            "total": len(cold_start_repos),
            "repos": sorted(list(cold_start_repos)),
        },
        "history_available_instances": {
            "total": len(history_available_instances),
            "instance_ids": history_available_instances,
        },
        "cold_start_instances": {
            "total": len(cold_start_instances),
            "instance_ids": cold_start_instances,
        },
    }

    output_file = PROJECT_ROOT / "results/cold_start/partition.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"✓ History-Available: {len(history_available_instances)} instances from {len(history_available_repos)} repos")
    print(f"✓ Cold-Start: {len(cold_start_instances)} instances from {len(cold_start_repos)} repos")
    print(f"✓ Total: {len(eval_instance_to_repo)} evaluation instances")
    print()
    print(f"✓ Saved partition to {output_file}")
    print()
    print("NEXT STEP:")
    print("  Calculate Pass@1 for each group from existing results")
    print("  → No new runs needed!")


if __name__ == "__main__":
    main()
