#!/usr/bin/env python3
"""
Simple Cold-Start Analysis (Reviewer u2Pr Q2 - Simplified)

What the professor actually wants:
1. Take EXISTING 157 historical instances
2. Take EXISTING 408 evaluation instances
3. Partition 408 into:
   - History-Available: repos with historical instances
   - Cold-Start: repos WITHOUT historical instances
4. Report Pass@1 for each group using EXISTING results

NO NEW RUNS NEEDED - just data analysis!
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent


def load_instances_with_repos(jsonl_path):
    """Load instances and extract repo info."""
    instances = {}
    with open(jsonl_path) as f:
        for line in f:
            inst = json.loads(line)
            instances[inst["instance_id"]] = {
                "id": inst["instance_id"],
                "repo": inst["repo"],
                **inst
            }
    return instances


def analyze_cold_start():
    """Analyze which eval instances have repo-level historical experience."""

    print("="*70)
    print("SIMPLIFIED COLD-START ANALYSIS")
    print("="*70)
    print()

    # Load IDs
    with open(PROJECT_ROOT / "data/memory_issue_ids.json") as f:
        historical_ids = set(json.load(f))

    with open(PROJECT_ROOT / "data/eval_issue_ids.json") as f:
        eval_ids = set(json.load(f))

    print(f"Historical instances: {len(historical_ids)}")
    print(f"Evaluation instances: {len(eval_ids)}")
    print()

    # Load full dataset with repo info
    print("Loading dataset...")
    all_instances = load_instances_with_repos(PROJECT_ROOT / "data/eval_set.jsonl")

    # Get repos from historical instances
    historical_repos = set()
    for iid in historical_ids:
        if iid in all_instances:
            historical_repos.add(all_instances[iid]["repo"])

    print(f"Unique repos in historical set: {len(historical_repos)}")
    print()

    # Partition evaluation instances
    history_available = []  # Eval instances from repos WITH historical experience
    cold_start = []         # Eval instances from repos WITHOUT historical experience

    eval_repos = set()
    for iid in eval_ids:
        if iid in all_instances:
            repo = all_instances[iid]["repo"]
            eval_repos.add(repo)

            if repo in historical_repos:
                history_available.append(iid)
            else:
                cold_start.append(iid)

    print(f"Unique repos in evaluation set: {len(eval_repos)}")
    print()

    print("="*70)
    print("PARTITION RESULTS")
    print("="*70)
    print()
    print(f"History-Available: {len(history_available)} instances")
    print(f"  (repos with historical repair experience)")
    print()
    print(f"Cold-Start: {len(cold_start)} instances")
    print(f"  (repos WITHOUT historical repair experience)")
    print()
    print(f"Total: {len(history_available) + len(cold_start)} instances")
    print()

    # Analyze repo overlap
    repos_with_history = historical_repos & eval_repos
    repos_without_history = eval_repos - historical_repos

    print("Repository Overlap:")
    print(f"  Repos in BOTH historical and eval: {len(repos_with_history)}")
    print(f"  Repos ONLY in eval (cold-start): {len(repos_without_history)}")
    print()

    # Save partition
    partition = {
        "historical_repos": sorted(list(historical_repos)),
        "history_available": {
            "instance_ids": history_available,
            "count": len(history_available),
            "description": "Eval instances from repos with historical repair experience"
        },
        "cold_start": {
            "instance_ids": cold_start,
            "count": len(cold_start),
            "description": "Eval instances from repos WITHOUT historical repair experience"
        },
        "repos_with_history": sorted(list(repos_with_history)),
        "repos_without_history": sorted(list(repos_without_history)),
    }

    output_file = PROJECT_ROOT / "results/cold_start_partition.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(partition, f, indent=2)

    print(f"✓ Saved partition to {output_file}")
    print()

    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print()
    print("1. Use EXISTING evaluation results (no new runs needed!)")
    print("2. Calculate Pass@1 for:")
    print(f"   - History-Available group ({len(history_available)} instances)")
    print(f"   - Cold-Start group ({len(cold_start)} instances)")
    print("3. Compare:")
    print("   - Does ReCAP help both groups?")
    print("   - Does it help more when repo history is available?")
    print()
    print("This is just data analysis - reuse existing results!")


if __name__ == "__main__":
    analyze_cold_start()
