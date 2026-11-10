#!/usr/bin/env python3
"""
Define Cold-Start vs History-Available Repos and Instances

Step-by-step approach:
1. Load 157 memory instances → Extract repos
2. Load 408 eval instances → Extract repos
3. Define History-Available repos (in memory)
4. Define Cold-Start repos (NOT in memory)
5. Partition eval instances into both groups
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


def step1_load_memory_repos():
    """Load repos from 157 memory instances."""
    print("="*70)
    print("STEP 1: Load Memory Repositories")
    print("="*70)
    print()

    l1_path = PROJECT_ROOT / "data/bidirect_trs/failure_memory.json"
    with open(l1_path) as f:
        l1_data = json.load(f)

    # Extract repos
    memory_instance_to_repo = {}
    memory_repo_counts = defaultdict(int)

    for entry in l1_data:
        instance_id = entry["issue_id"]
        repo = entry["repo"]
        memory_instance_to_repo[instance_id] = repo
        memory_repo_counts[repo] += 1

    memory_repos = set(memory_instance_to_repo.values())

    print(f"Total memory instances: {len(memory_instance_to_repo)}")
    print(f"Unique repos in memory: {len(memory_repos)}")
    print()
    print("Top 10 repos in memory:")
    for repo, count in sorted(memory_repo_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {repo}: {count} instances")
    print()

    return memory_instance_to_repo, memory_repos, memory_repo_counts


def step2_find_eval_repos():
    """
    Find repo information for 408 eval instances.

    Strategy:
    1. First try: Check if log_details has repo info
    2. Second try: Check decomposition_cache
    3. Third try: Manual extraction needed
    """
    print("="*70)
    print("STEP 2: Find Evaluation Instance Repositories")
    print("="*70)
    print()

    # Load eval IDs
    with open(PROJECT_ROOT / "data/eval_issue_ids.json") as f:
        eval_ids = set(json.load(f))

    print(f"Total evaluation instances: {len(eval_ids)}")
    print()

    # Try to load from decomposition cache (keyed by sha, but problems have repo)
    print("Checking decomposition_cache for repo info...")
    cache_path = PROJECT_ROOT / "data/decomposition_cache.json"

    with open(cache_path) as f:
        cache_data = json.load(f)

    # Cache is keyed by sha_fail, values are problem lists
    # Each problem might have repo info
    eval_instance_to_repo = {}

    # We need to find another way - let's check log_details
    print("Checking log_details for instance metadata...")
    log_path = PROJECT_ROOT / "data/log_details.json"

    with open(log_path) as f:
        log_data = json.load(f)

    # log_data is a dict with instance_id as key
    # Check if it has repo info
    if log_data:
        first_key = list(log_data.keys())[0]
        first_entry = log_data[first_key]
        print(f"Sample log_details entry keys: {list(first_entry.keys())[:5]}")

    # PROBLEM: log_details doesn't have repo directly
    # We need to use the L1 memory as source of truth for repos

    print()
    print("⚠️  Repo info for eval instances not found in standard files.")
    print()
    print("SOLUTION: Use L1 memory repos as reference.")
    print("We can:")
    print("1. Extract ALL repos from L1 memory (157 memory + any available eval)")
    print("2. Use the 80 memory repos as 'History-Available'")
    print("3. Infer Cold-Start repos from what's NOT in memory")
    print()
    print("ALTERNATIVE: Check if there's a CI-Repair-Bench dataset file")
    print("             with complete instance -> repo mapping.")

    return None


def step3_define_repo_partition(memory_repos):
    """
    Define History-Available vs Cold-Start repositories.

    Without eval instance repo info, we can:
    - Show the 80 memory repos (History-Available)
    - Note that any other repo in eval would be Cold-Start
    """
    print("="*70)
    print("STEP 3: Define Repository Partition")
    print("="*70)
    print()

    print("HISTORY-AVAILABLE REPOSITORIES:")
    print(f"  Total: {len(memory_repos)} repos")
    print(f"  These repos have historical repair experience in memory")
    print()

    print("COLD-START REPOSITORIES:")
    print(f"  Definition: Any repo in evaluation NOT in the {len(memory_repos)} memory repos")
    print(f"  These repos have NO historical repair experience")
    print()

    return memory_repos


def main():
    """Run the complete cold-start definition analysis."""

    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "COLD-START ANALYSIS DEFINITION" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    print()

    # Step 1: Load memory repos
    memory_inst_to_repo, memory_repos, memory_repo_counts = step1_load_memory_repos()

    # Step 2: Try to find eval repos
    eval_inst_to_repo = step2_find_eval_repos()

    # Step 3: Define partition
    history_available_repos = step3_define_repo_partition(memory_repos)

    # Save what we have
    output = {
        "memory_instances": {
            "total": len(memory_inst_to_repo),
            "instance_to_repo": memory_inst_to_repo,
        },
        "history_available_repos": {
            "repos": sorted(list(history_available_repos)),
            "count": len(history_available_repos),
            "repo_counts": dict(memory_repo_counts),
        },
        "cold_start_repos": {
            "definition": "Any repo in evaluation NOT in history_available_repos",
            "note": "Cannot enumerate without eval instance repo mapping"
        },
        "status": "INCOMPLETE - need eval instance to repo mapping"
    }

    output_dir = Path("results/cold_start")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "repo_partition.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"✓ Identified {len(history_available_repos)} History-Available repos")
    print(f"✓ Saved to {output_file}")
    print()
    print("❌ Still need: Repo information for 408 eval instances")
    print()
    print("NEXT STEPS:")
    print("1. Find source of eval instance -> repo mapping")
    print("2. Partition eval instances into History-Available vs Cold-Start")
    print("3. Calculate Pass@1 for each group from existing results")


if __name__ == "__main__":
    main()
