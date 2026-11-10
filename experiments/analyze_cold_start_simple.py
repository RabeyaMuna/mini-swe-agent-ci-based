#!/usr/bin/env python3
"""
Simplified Cold-Start Analysis - Just use L1 memory!

The professor's suggestion:
1. Load 157 historical instances from L1 memory (has repo info)
2. Load 408 evaluation IDs
3. For eval instances, we need repo info - THIS IS THE ISSUE

SIMPLER APPROACH:
Just analyze which repos appear in historical vs which would appear in eval.
If we have existing results with repo info, we can partition those.
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    print("="*70)
    print("COLD-START ANALYSIS (Simplified)")
    print("="*70)
    print()

    # Load historical instances from L1 memory (has repo info!)
    print("Loading historical instances from L1 memory...")
    with open("data/bidirect_trs/failure_memory.json") as f:
        l1_data = json.load(f)

    # Build repo mapping
    historical_repos = set()
    hist_repo_counts = defaultdict(int)

    for entry in l1_data:
        repo = entry["repo"]
        historical_repos.add(repo)
        hist_repo_counts[repo] += 1

    print(f"Total historical instances: {len(l1_data)}")
    print(f"Unique repos in historical: {len(historical_repos)}")
    print()

    # Show top repos
    print("Top 10 repos in historical set:")
    for repo, count in sorted(hist_repo_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {repo}: {count} instances")
    print()

    # Load eval IDs
    with open("data/eval_issue_ids.json") as f:
        eval_ids = json.load(f)

    print(f"Total evaluation instances: {len(eval_ids)}")
    print()

    print("="*70)
    print("WHAT WE NEED")
    print("="*70)
    print()
    print("To complete the analysis, we need repo information for the 408 eval instances.")
    print()
    print("Options:")
    print("1. Extract from existing result files (if they contain repo info)")
    print("2. Extract from CI-Repair-Bench dataset (if available)")
    print("3. Parse from decomposition_cache or log_details")
    print()
    print("Once we have eval instance -> repo mapping:")
    print("  - History-Available = eval instances where repo ∈ historical_repos")
    print("  - Cold-Start = eval instances where repo ∉ historical_repos")
    print()
    print("Then use EXISTING Pass@1 results to compare both groups.")
    print()

    # Save historical repos for reference
    output = {
        "historical_repos": sorted(list(historical_repos)),
        "historical_repo_counts": dict(hist_repo_counts),
        "total_historical_instances": len(l1_data),
        "unique_historical_repos": len(historical_repos),
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/historical_repos.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Saved historical repo info to results/historical_repos.json")


if __name__ == "__main__":
    main()
