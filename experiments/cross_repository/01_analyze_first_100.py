#!/usr/bin/env python3
"""
Step 1: Analyze First 100 Evaluation Instances

Takes first 100 instances from eval_issue_ids.json and:
1. Identifies which repos they belong to (TARGET repos)
2. Identifies SOURCE repos (all others with memory)
3. Ensures "same-repository experience excluded"

Output: results/cross_repository/01_first_100_analysis.json
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_first_100_eval_ids():
    """Load first 100 instance IDs from eval_issue_ids.json."""
    eval_ids_file = PROJECT_ROOT / "data/eval_issue_ids.json"

    with open(eval_ids_file) as f:
        all_ids = json.load(f)

    # Take first 100
    first_100 = [str(id) for id in all_ids[:100]]

    return first_100


def get_repos_for_instances(instance_ids):
    """Get repo for each instance from result files."""
    result_file = PROJECT_ROOT / "results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl"

    instance_to_repo = {}

    with open(result_file) as f:
        for line in f:
            entry = json.loads(line)
            inst_id = str(entry['id'])

            if inst_id in instance_ids:
                instance_to_repo[inst_id] = entry['repo_name']

    return instance_to_repo


def load_memory_instances():
    """Load all memory instances and group by repo."""
    memory_file = PROJECT_ROOT / "data/bidirect_trs/failure_memory.json"

    with open(memory_file) as f:
        memory_data = json.load(f)

    # Handle both list and dict formats
    if isinstance(memory_data, dict):
        instances = memory_data.get("memory", [])
    else:
        instances = memory_data

    # Group by repo
    by_repo = defaultdict(list)
    for inst in instances:
        repo = inst.get("repo_name", "unknown")
        by_repo[repo].append({
            "issue_id": inst.get("issue_id"),
            "workflow": inst.get("workflow", ""),
            "workflow_name": inst.get("workflow_name", ""),
            "problems": inst.get("problems", []),
        })

    return dict(by_repo), instances


def main():
    print("\n" + "="*70)
    print("STEP 1: ANALYZE FIRST 100 EVALUATION INSTANCES")
    print("="*70 + "\n")

    # Load first 100 eval IDs
    print("Loading first 100 eval instance IDs...")
    first_100_ids = load_first_100_eval_ids()
    print(f"✓ Loaded first 100 IDs: {first_100_ids[:10]}...\n")

    # Get repos for these instances
    print("Identifying repos for first 100 instances...")
    instance_to_repo = get_repos_for_instances(first_100_ids)
    print(f"✓ Found repo info for {len(instance_to_repo)}/100 instances\n")

    # Identify TARGET repos
    target_repos = set(instance_to_repo.values())
    print(f"TARGET repos (contain eval instances): {len(target_repos)} repos")

    # Count instances per TARGET repo
    target_repo_counts = defaultdict(int)
    for inst_id, repo in instance_to_repo.items():
        target_repo_counts[repo] += 1

    print("\nTARGET repo distribution:")
    for repo, count in sorted(target_repo_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {repo}: {count} instances")
    if len(target_repo_counts) > 15:
        print(f"  ... and {len(target_repo_counts) - 15} more repos")
    print()

    # Load memory instances
    print("Loading memory instances...")
    memory_by_repo, all_memory = load_memory_instances()
    print(f"✓ Loaded {len(all_memory)} memory instances from {len(memory_by_repo)} repos\n")

    # Identify SOURCE repos (repos with memory but NOT in TARGET)
    all_memory_repos = set(memory_by_repo.keys())
    source_repos = all_memory_repos - target_repos
    overlap_repos = all_memory_repos & target_repos

    print("="*70)
    print("REPOSITORY CLASSIFICATION")
    print("="*70 + "\n")

    print(f"TARGET repos (eval instances): {len(target_repos)} repos")
    print(f"SOURCE repos (memory only): {len(source_repos)} repos")
    print(f"OVERLAP repos (both memory and eval): {len(overlap_repos)} repos")
    print(f"Total memory repos: {len(all_memory_repos)} repos\n")

    # Count memory instances available from SOURCE repos
    source_memory_instances = []
    for repo in source_repos:
        for inst in memory_by_repo[repo]:
            source_memory_instances.append({
                "repo": repo,
                **inst
            })

    print(f"Available memory instances from SOURCE repos: {len(source_memory_instances)}")
    print(f"  Recommendation: Select ~50 instances from these {len(source_repos)} SOURCE repos\n")

    # Show SOURCE repo distribution
    print("SOURCE repo distribution (top 15 by memory instances):")
    source_sorted = sorted(
        [(r, len(memory_by_repo[r])) for r in source_repos],
        key=lambda x: -x[1]
    )[:15]
    for repo, count in source_sorted:
        print(f"  {repo}: {count} memory instances")
    if len(source_repos) > 15:
        print(f"  ... and {len(source_repos) - 15} more SOURCE repos")
    print()

    # Verify "same-repository experience excluded"
    print("="*70)
    print("VERIFICATION: Same-Repository Experience Excluded")
    print("="*70 + "\n")

    violations = []
    for inst_id in first_100_ids:
        eval_repo = instance_to_repo.get(inst_id)
        if eval_repo and eval_repo in source_repos:
            violations.append(f"Instance {inst_id} from repo {eval_repo}")

    if violations:
        print("⚠️  WARNING: Found violations!")
        for v in violations[:5]:
            print(f"  {v}")
        if len(violations) > 5:
            print(f"  ... and {len(violations) - 5} more")
    else:
        print("✓ No violations found!")
        print("  All eval instances are from TARGET repos")
        print("  All memory will come from SOURCE repos (disjoint)")
        print("  → Same-repository experience is EXCLUDED\n")

    # Save analysis
    output = {
        "evaluation_set": {
            "total_instances": len(first_100_ids),
            "instance_ids": first_100_ids,
            "instance_to_repo": instance_to_repo,
            "target_repos": sorted(list(target_repos)),
            "target_repo_count": len(target_repos),
        },
        "memory_split": {
            "source_repos": sorted(list(source_repos)),
            "source_repo_count": len(source_repos),
            "overlap_repos": sorted(list(overlap_repos)),
            "overlap_repo_count": len(overlap_repos),
            "total_source_memory_instances": len(source_memory_instances),
        },
        "verification": {
            "same_repo_excluded": len(violations) == 0,
            "violations": violations if violations else []
        }
    }

    output_dir = PROJECT_ROOT / "results/cross_repository"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "01_first_100_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("="*70)
    print("STEP 1 COMPLETE")
    print("="*70)
    print(f"\n✓ Saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Evaluation: {len(first_100_ids)} instances from {len(target_repos)} TARGET repos")
    print(f"  - Memory pool: {len(source_memory_instances)} instances from {len(source_repos)} SOURCE repos")
    print(f"  - Same-repository excluded: {'YES ✓' if len(violations) == 0 else 'NO ✗'}")
    print(f"\nNext step: Select ~50 memory instances from SOURCE repos\n")


if __name__ == "__main__":
    main()
