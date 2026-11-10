#!/usr/bin/env python3
"""
Create clean cross-repository evaluation data directly in experimental_data/

Structure:
experimental_data/cross_repository/
├── bidirect_trs/
│   ├── failure_memory.json  (50 instances from SOURCE repos)
│   └── empty_memory.json    (empty array for baseline)
└── eval_issue_ids.json      (100 first IDs)
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_first_100_eval_ids():
    """Get first 100 eval IDs."""
    eval_ids_file = PROJECT_ROOT / "data/eval_issue_ids.json"
    with open(eval_ids_file) as f:
        all_ids = json.load(f)
    return [str(id) for id in all_ids[:100]]


def get_target_repos(eval_ids):
    """Get repos for eval instances."""
    result_file = PROJECT_ROOT / "results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl"

    target_repos = set()
    with open(result_file) as f:
        for line in f:
            entry = json.loads(line)
            if str(entry['id']) in eval_ids:
                target_repos.add(entry['repo_name'])

    return target_repos


def load_and_filter_memory(target_repos):
    """Load memory and filter to SOURCE repos only."""
    memory_file = PROJECT_ROOT / "data/bidirect_trs/failure_memory.json"

    with open(memory_file) as f:
        all_memory = json.load(f)

    # Filter to SOURCE repos (not in target)
    source_instances = [
        inst for inst in all_memory
        if inst.get("repo_name") not in target_repos
    ]

    return source_instances


def select_50_instances(source_instances):
    """Select 50 instances with diversity."""
    # Group by repo
    by_repo = defaultdict(list)
    for inst in source_instances:
        repo = inst.get("repo_name")
        by_repo[repo].append(inst)

    # Sort repos by instance count
    sorted_repos = sorted(by_repo.items(), key=lambda x: -len(x[1]))

    selected = []
    repo_count = {}

    # Phase 1: Take 1 from each repo (up to 30 repos)
    for repo, instances in sorted_repos[:30]:
        selected.append(instances[0])
        repo_count[repo] = 1

    # Phase 2: Add more from top repos
    for repo, instances in sorted_repos[:10]:
        if len(selected) >= 50:
            break
        already_taken = repo_count.get(repo, 0)
        can_take = min(2, len(instances) - already_taken, 50 - len(selected))
        for i in range(can_take):
            selected.append(instances[already_taken + i])
        repo_count[repo] = already_taken + can_take

    return selected[:50]


def main():
    print("\n" + "="*70)
    print("CREATE CLEAN CROSS-REPOSITORY DATA")
    print("="*70 + "\n")

    # Create directory
    exp_dir = PROJECT_ROOT / "experimental_data/cross_repository"
    bidirect_dir = exp_dir / "bidirect_trs"
    bidirect_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get first 100 eval IDs
    print("1. Getting first 100 eval IDs...")
    first_100 = load_first_100_eval_ids()
    print(f"   ✓ {len(first_100)} IDs: {first_100[:10]}...\n")

    # 2. Identify TARGET repos
    print("2. Identifying TARGET repos...")
    target_repos = get_target_repos(first_100)
    print(f"   ✓ {len(target_repos)} TARGET repos\n")

    # 3. Filter memory to SOURCE repos
    print("3. Filtering memory to SOURCE repos...")
    source_instances = load_and_filter_memory(target_repos)
    print(f"   ✓ {len(source_instances)} instances from SOURCE repos\n")

    # 4. Select 50 instances
    print("4. Selecting 50 instances with diversity...")
    selected_50 = select_50_instances(source_instances)
    print(f"   ✓ Selected {len(selected_50)} instances\n")

    # Count repos in selection
    selected_repos = set(inst.get("repo_name") for inst in selected_50)
    print(f"   From {len(selected_repos)} different SOURCE repos\n")

    # 5. Save files
    print("="*70)
    print("SAVING FILES")
    print("="*70 + "\n")

    # Save failure_memory.json (50 instances)
    memory_file = bidirect_dir / "failure_memory.json"
    with open(memory_file, 'w') as f:
        json.dump(selected_50, f, indent=2)
    print(f"✓ Saved: {memory_file}")
    print(f"  {len(selected_50)} instances from {len(selected_repos)} SOURCE repos\n")

    # Save empty_memory.json (for baseline)
    empty_file = bidirect_dir / "empty_memory.json"
    with open(empty_file, 'w') as f:
        json.dump([], f, indent=2)
    print(f"✓ Saved: {empty_file}")
    print(f"  Empty array for baseline\n")

    # Save eval_issue_ids.json (100 IDs)
    eval_file = exp_dir / "eval_issue_ids.json"
    with open(eval_file, 'w') as f:
        json.dump(first_100, f, indent=2)
    print(f"✓ Saved: {eval_file}")
    print(f"  {len(first_100)} eval instance IDs\n")

    # Print structure
    print("="*70)
    print("FINAL STRUCTURE")
    print("="*70 + "\n")
    print("experimental_data/cross_repository/")
    print("├── bidirect_trs/")
    print(f"│   ├── failure_memory.json    ({len(selected_50)} instances)")
    print(f"│   └── empty_memory.json      (0 instances)")
    print(f"└── eval_issue_ids.json        ({len(first_100)} IDs)")
    print()

    print("="*70)
    print("USAGE")
    print("="*70 + "\n")
    print("Point your system to: experimental_data/cross_repository/")
    print()
    print("Config 1 (No-Experience):")
    print("  Memory: bidirect_trs/empty_memory.json")
    print("  Eval: eval_issue_ids.json")
    print()
    print("Config 2 (Cross-Repo L1/L2/L3):")
    print("  Memory: bidirect_trs/failure_memory.json")
    print("  Eval: eval_issue_ids.json")
    print()
    print("Config 3 (Cross-Repo L3-Only):")
    print("  Memory: bidirect_trs/failure_memory.json")
    print("  Eval: eval_issue_ids.json")
    print()


if __name__ == "__main__":
    main()
