#!/usr/bin/env python3
"""
Step 1: Analyze data for SOURCE/TARGET repository split

Analyzes:
1. Which repos appear in memory (157 instances)
2. Which repos appear in eval (408 instances)
3. Repo overlap and distribution
4. Temporal distribution
5. Suggests SOURCE/TARGET split strategies

Output: results/cross_repository/01_repo_analysis.json
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_memory_instances():
    """Load 157 memory instances and extract repos."""
    memory_file = PROJECT_ROOT / "data/bidirect_trs/failure_memory.json"

    with open(memory_file) as f:
        memory_data = json.load(f)

    instances = memory_data.get("memory", [])

    # Extract repos from each instance
    repo_to_instances = defaultdict(list)
    for inst in instances:
        repo = inst.get("repo_name", "unknown")
        inst_id = inst.get("instance_id", "unknown")
        repo_to_instances[repo].append({
            "instance_id": inst_id,
            "timestamp": inst.get("timestamp", ""),
            "error_type": inst.get("error_type", ""),
        })

    return dict(repo_to_instances), instances


def load_eval_instances():
    """Load eval instance repos from result files."""
    # Use existing result file to get repo info
    result_file = PROJECT_ROOT / "results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl"

    repo_to_instances = defaultdict(list)

    with open(result_file) as f:
        for line in f:
            entry = json.loads(line)
            inst_id = str(entry['id'])
            repo = entry['repo_name']

            repo_to_instances[repo].append({
                "instance_id": inst_id,
            })

    return dict(repo_to_instances)


def suggest_split_strategies(memory_repos, eval_repos, overlap_repos):
    """Suggest SOURCE/TARGET split strategies."""

    strategies = {}

    # Strategy 1: Random from overlap
    # Use repos that appear in BOTH memory and eval
    # SOURCE: subset of overlap (for memory)
    # TARGET: remaining overlap (for eval)
    overlap_list = sorted(list(overlap_repos))
    n_overlap = len(overlap_list)

    # Try to get ~15 SOURCE, ~15 TARGET from overlap
    if n_overlap >= 30:
        strategies["random_overlap"] = {
            "description": "Random split from repos appearing in both memory and eval",
            "source_repos": overlap_list[:15],
            "target_repos": overlap_list[15:30],
            "note": "Both sets from overlap - ensures semantic similarity"
        }

    # Strategy 2: Memory-only SOURCE, Eval-only TARGET
    # SOURCE: repos only in memory (no eval instances)
    # TARGET: repos only in eval (no memory instances)
    memory_only = sorted(list(set(memory_repos.keys()) - overlap_repos))
    eval_only = sorted(list(set(eval_repos.keys()) - overlap_repos))

    if len(memory_only) >= 10 and len(eval_only) >= 10:
        strategies["disjoint_pure"] = {
            "description": "Completely disjoint: memory-only SOURCE, eval-only TARGET",
            "source_repos": memory_only[:15] if len(memory_only) >= 15 else memory_only,
            "target_repos": eval_only[:15] if len(eval_only) >= 15 else eval_only,
            "note": "Maximum separation - tests pure cross-repo transfer"
        }

    # Strategy 3: Mixed - some overlap, some disjoint
    # SOURCE: mix of overlap and memory-only
    # TARGET: mix of overlap and eval-only
    source_mixed = overlap_list[:5] + memory_only[:10]
    target_mixed = overlap_list[5:10] + eval_only[:10]

    strategies["mixed"] = {
        "description": "Mixed: some overlap, some disjoint repos",
        "source_repos": source_mixed[:15],
        "target_repos": target_mixed[:15],
        "note": "Balanced - some related, some novel repos"
    }

    return strategies


def main():
    print("\n" + "="*70)
    print("CROSS-REPOSITORY EVALUATION: DATA ANALYSIS")
    print("="*70 + "\n")

    # Load memory instances
    print("Loading memory instances...")
    memory_repos, memory_instances = load_memory_instances()
    print(f"✓ Loaded {len(memory_instances)} memory instances from {len(memory_repos)} repos\n")

    # Load eval instances
    print("Loading eval instances...")
    eval_repos = load_eval_instances()
    total_eval = sum(len(insts) for insts in eval_repos.values())
    print(f"✓ Loaded {total_eval} eval instances from {len(eval_repos)} repos\n")

    # Analyze overlap
    memory_repo_set = set(memory_repos.keys())
    eval_repo_set = set(eval_repos.keys())
    overlap_repos = memory_repo_set & eval_repo_set
    memory_only_repos = memory_repo_set - eval_repo_set
    eval_only_repos = eval_repo_set - memory_repo_set

    print("="*70)
    print("REPOSITORY DISTRIBUTION")
    print("="*70 + "\n")

    print(f"Repos in MEMORY only: {len(memory_only_repos)}")
    print(f"Repos in EVAL only: {len(eval_only_repos)}")
    print(f"Repos in BOTH (overlap): {len(overlap_repos)}")
    print(f"Total unique repos: {len(memory_repo_set | eval_repo_set)}\n")

    # Show top repos by instance count
    print("Top Memory Repos (by instance count):")
    memory_sorted = sorted(memory_repos.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for repo, instances in memory_sorted:
        print(f"  {repo}: {len(instances)} instances")

    print("\nTop Eval Repos (by instance count):")
    eval_sorted = sorted(eval_repos.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for repo, instances in eval_sorted:
        print(f"  {repo}: {len(instances)} instances")

    # Suggest split strategies
    print("\n" + "="*70)
    print("SUGGESTED SPLIT STRATEGIES")
    print("="*70 + "\n")

    strategies = suggest_split_strategies(memory_repos, eval_repos, overlap_repos)

    for strategy_name, strategy in strategies.items():
        print(f"Strategy: {strategy_name}")
        print(f"  Description: {strategy['description']}")
        print(f"  SOURCE repos: {len(strategy['source_repos'])}")
        print(f"  TARGET repos: {len(strategy['target_repos'])}")

        # Count instances
        source_memory_count = sum(len(memory_repos.get(r, [])) for r in strategy['source_repos'])
        target_eval_count = sum(len(eval_repos.get(r, [])) for r in strategy['target_repos'])

        print(f"  SOURCE memory instances: {source_memory_count}")
        print(f"  TARGET eval instances: {target_eval_count}")
        print(f"  Note: {strategy['note']}\n")

    # Save analysis
    output = {
        "summary": {
            "total_memory_instances": len(memory_instances),
            "total_eval_instances": total_eval,
            "memory_repos": len(memory_repos),
            "eval_repos": len(eval_repos),
            "overlap_repos": len(overlap_repos),
            "memory_only_repos": len(memory_only_repos),
            "eval_only_repos": len(eval_only_repos),
        },
        "repos": {
            "memory_repos": {repo: len(insts) for repo, insts in memory_repos.items()},
            "eval_repos": {repo: len(insts) for repo, insts in eval_repos.items()},
            "overlap_repos": sorted(list(overlap_repos)),
            "memory_only_repos": sorted(list(memory_only_repos)),
            "eval_only_repos": sorted(list(eval_only_repos)),
        },
        "suggested_strategies": strategies,
        "recommendation": "random_overlap" if "random_overlap" in strategies else "mixed"
    }

    output_dir = PROJECT_ROOT / "results/cross_repository"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "01_repo_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("="*70)
    print("RECOMMENDATION")
    print("="*70 + "\n")

    if "random_overlap" in strategies:
        rec = strategies["random_overlap"]
        print("✓ Use 'random_overlap' strategy:")
        print(f"  - Both SOURCE and TARGET from overlap repos")
        print(f"  - Ensures semantic similarity")
        print(f"  - {len(rec['source_repos'])} SOURCE repos → ~{sum(len(memory_repos.get(r, [])) for r in rec['source_repos'])} memory instances")
        print(f"  - {len(rec['target_repos'])} TARGET repos → ~{sum(len(eval_repos.get(r, [])) for r in rec['target_repos'])} eval instances")
    else:
        print("✓ Use 'mixed' strategy (fallback)")

    print(f"\n✓ Saved to: {output_file}")
    print("\nNext step: Select specific instances from SOURCE/TARGET repos\n")


if __name__ == "__main__":
    main()
