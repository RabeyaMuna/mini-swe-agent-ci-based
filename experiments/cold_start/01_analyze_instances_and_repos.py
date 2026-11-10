#!/usr/bin/env python3
"""
Step 1: Analyze Memory & Eval Instances → Extract Repos

Loads:
- 157 memory instances from L1 memory
- 408 eval instances from results
- Extracts repository information for each

Output: results/cold_start/01_instance_repos.json
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    print("\n" + "="*70)
    print("STEP 1: ANALYZE INSTANCES AND REPOS")
    print("="*70 + "\n")

    # ========== Load Memory Instances ==========
    print("Loading memory instances (157)...")
    l1_path = PROJECT_ROOT / "data/bidirect_trs/failure_memory.json"

    with open(l1_path) as f:
        l1_data = json.load(f)

    memory_instance_to_repo = {}
    for entry in l1_data:
        inst_id = entry["issue_id"]
        repo = entry["repo"]
        # Normalize: just repo name, not owner/repo
        if '/' in repo:
            repo = repo.split('/')[-1]
        memory_instance_to_repo[inst_id] = repo

    memory_repos = set(memory_instance_to_repo.values())

    print(f"✓ Loaded {len(memory_instance_to_repo)} memory instances")
    print(f"✓ Found {len(memory_repos)} unique repos in memory\n")

    # ========== Load Eval Instances ==========
    print("Loading eval instances (408)...")

    # Load from result files (they have repo info)
    eval_instance_to_repo = {}
    result_files = list(PROJECT_ROOT.glob("results/**/jobs_results_diff.jsonl"))

    for result_file in result_files:
        with open(result_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    inst_id = entry.get('id')
                    repo = entry.get('repo_name')

                    if inst_id and repo and inst_id not in eval_instance_to_repo:
                        # Normalize
                        if '/' in repo:
                            repo = repo.split('/')[-1]
                        eval_instance_to_repo[inst_id] = repo
                except:
                    continue

    eval_repos = set(eval_instance_to_repo.values())

    print(f"✓ Loaded {len(eval_instance_to_repo)} eval instances")
    print(f"✓ Found {len(eval_repos)} unique repos in eval\n")

    # ========== Analyze Repo Overlap ==========
    print("Analyzing repo overlap...")

    repos_in_both = memory_repos & eval_repos  # History-Available
    repos_only_eval = eval_repos - memory_repos  # Cold-Start

    print(f"✓ {len(repos_in_both)} repos in BOTH memory and eval (History-Available)")
    print(f"✓ {len(repos_only_eval)} repos ONLY in eval (Cold-Start)\n")

    # ========== Save Results ==========
    output = {
        "memory_instances": {
            "count": len(memory_instance_to_repo),
            "instance_to_repo": memory_instance_to_repo,
            "unique_repos": sorted(list(memory_repos)),
        },
        "eval_instances": {
            "count": len(eval_instance_to_repo),
            "instance_to_repo": eval_instance_to_repo,
            "unique_repos": sorted(list(eval_repos)),
        },
        "repo_overlap": {
            "history_available_repos": sorted(list(repos_in_both)),
            "cold_start_repos": sorted(list(repos_only_eval)),
            "history_available_count": len(repos_in_both),
            "cold_start_count": len(repos_only_eval),
        }
    }

    output_dir = PROJECT_ROOT / "results/cold_start"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "01_instance_repos.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("="*70)
    print("STEP 1 COMPLETE")
    print("="*70)
    print(f"\n✓ Saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Memory: {len(memory_instance_to_repo)} instances, {len(memory_repos)} repos")
    print(f"  Eval: {len(eval_instance_to_repo)} instances, {len(eval_repos)} repos")
    print(f"  History-Available repos: {len(repos_in_both)}")
    print(f"  Cold-Start repos: {len(repos_only_eval)}\n")


if __name__ == "__main__":
    main()
