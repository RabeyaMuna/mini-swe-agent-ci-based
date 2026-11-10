#!/usr/bin/env python3
"""
Step 2: Select ~50 Cross-Repository Memory Instances

Selects memory instances from SOURCE repos (not in TARGET) to maximize:
1. Repository diversity (multiple SOURCE repos)
2. Failure type diversity (different problems)
3. Workflow diversity (different CI workflows)
4. Coverage of L1/L2/L3 patterns

Output: experimental_data/cross_repository/cross_repo_memory.json
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent

TARGET_MEMORY_SIZE = 50


def load_analysis():
    """Load results from Step 1."""
    analysis_file = PROJECT_ROOT / "results/cross_repository/01_first_100_analysis.json"

    with open(analysis_file) as f:
        return json.load(f)


def load_all_memory():
    """Load all memory instances."""
    memory_file = PROJECT_ROOT / "data/bidirect_trs/failure_memory.json"

    with open(memory_file) as f:
        memory_data = json.load(f)

    if isinstance(memory_data, dict):
        return memory_data.get("memory", [])
    else:
        return memory_data


def select_instances_with_diversity(source_instances, target_count=50):
    """
    Select instances to maximize diversity across:
    - Repositories (distribute across SOURCE repos)
    - Workflows (different CI workflows)
    - Problems (different failure types)
    """

    # Group by repo
    by_repo = defaultdict(list)
    for inst in source_instances:
        repo = inst.get("repo_name")
        by_repo[repo].append(inst)

    # Sort repos by instance count (descending)
    sorted_repos = sorted(by_repo.items(), key=lambda x: -len(x[1]))

    print(f"Available SOURCE repos: {len(sorted_repos)}")
    print(f"Total SOURCE instances: {len(source_instances)}\n")

    # Selection strategy: Distribute across repos
    # Take instances from multiple repos to ensure diversity
    selected = []
    repo_selection_count = {}

    # Phase 1: Take at least 1 instance from top repos
    print("Phase 1: Selecting from top SOURCE repos...")
    for repo, instances in sorted_repos[:min(30, len(sorted_repos))]:
        if len(selected) >= target_count:
            break
        # Take 1 instance from each repo initially
        selected.append(instances[0])
        repo_selection_count[repo] = 1
        print(f"  {repo}: selected 1/{len(instances)}")

    print(f"\nAfter Phase 1: {len(selected)} instances selected\n")

    # Phase 2: Fill remaining slots from top repos with most instances
    print("Phase 2: Adding more instances from top repos...")
    remaining_needed = target_count - len(selected)

    if remaining_needed > 0:
        for repo, instances in sorted_repos:
            if len(selected) >= target_count:
                break

            # How many already selected from this repo
            already_selected = repo_selection_count.get(repo, 0)
            # How many more can we take
            available = len(instances) - already_selected

            if available > 0:
                # Take up to 2 more from repos with many instances
                take_count = min(2, available, target_count - len(selected))
                for i in range(take_count):
                    selected.append(instances[already_selected + i])
                repo_selection_count[repo] = already_selected + take_count
                print(f"  {repo}: added {take_count} more (total: {repo_selection_count[repo]}/{len(instances)})")

    print(f"\nAfter Phase 2: {len(selected)} instances selected\n")

    # Print final distribution
    print("="*70)
    print("FINAL SELECTION DISTRIBUTION")
    print("="*70 + "\n")

    for repo in sorted(repo_selection_count.keys(), key=lambda r: -repo_selection_count[r]):
        count = repo_selection_count[repo]
        total = len(by_repo[repo])
        print(f"  {repo}: {count}/{total} instances")

    return selected


def main():
    print("\n" + "="*70)
    print("STEP 2: SELECT CROSS-REPOSITORY MEMORY INSTANCES")
    print("="*70 + "\n")

    # Load analysis
    print("Loading analysis from Step 1...")
    analysis = load_analysis()
    source_repos = set(analysis['memory_split']['source_repos'])
    print(f"✓ SOURCE repos: {len(source_repos)} repos\n")

    # Load all memory instances
    print("Loading all memory instances...")
    all_memory = load_all_memory()
    print(f"✓ Total memory instances: {len(all_memory)}\n")

    # Filter to SOURCE repos only
    print("Filtering to SOURCE repos only...")
    source_instances = [
        inst for inst in all_memory
        if inst.get("repo_name") in source_repos
    ]
    print(f"✓ SOURCE instances: {len(source_instances)}\n")

    # Select ~50 instances with diversity
    print("="*70)
    print(f"SELECTING {TARGET_MEMORY_SIZE} INSTANCES WITH DIVERSITY")
    print("="*70 + "\n")

    selected_instances = select_instances_with_diversity(
        source_instances,
        target_count=TARGET_MEMORY_SIZE
    )

    # Prepare output
    cross_repo_memory = {
        "memory": selected_instances,
        "metadata": {
            "total_instances": len(selected_instances),
            "source_repos": sorted(list(set(inst.get("repo_name") for inst in selected_instances))),
            "source_repo_count": len(set(inst.get("repo_name") for inst in selected_instances)),
            "excluded_target_repos": analysis['evaluation_set']['target_repos'],
            "excluded_target_repo_count": len(analysis['evaluation_set']['target_repos']),
            "selection_strategy": "Repository diversity prioritization",
            "same_repository_excluded": True,
        }
    }

    # Save to experimental_data
    output_dir = PROJECT_ROOT / "experimental_data/cross_repository"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "cross_repo_memory.json"
    with open(output_file, 'w') as f:
        json.dump(cross_repo_memory, f, indent=2)

    # Also save instance IDs list for reference
    instance_ids = [inst.get("issue_id") for inst in selected_instances]
    ids_file = output_dir / "selected_memory_issue_ids.json"
    with open(ids_file, 'w') as f:
        json.dump(instance_ids, f, indent=2)

    print("\n" + "="*70)
    print("STEP 2 COMPLETE")
    print("="*70)
    print(f"\n✓ Selected {len(selected_instances)} instances from {cross_repo_memory['metadata']['source_repo_count']} SOURCE repos")
    print(f"✓ Excluded {cross_repo_memory['metadata']['excluded_target_repo_count']} TARGET repos")
    print(f"\n✓ Saved to: {output_file}")
    print(f"✓ Instance IDs: {ids_file}")
    print(f"\nSelected issue IDs: {instance_ids[:10]}...")
    print(f"\nThese {TARGET_MEMORY_SIZE} instances will be used for:")
    print(f"  - Config 2: ReCAP Cross-Repository (L1/L2/L3)")
    print(f"  - Config 3: ReCAP L3-Only\n")


if __name__ == "__main__":
    main()
