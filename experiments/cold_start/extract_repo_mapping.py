#!/usr/bin/env python3
"""
Extract instance -> repo mapping from existing result files.

Found: results files have 'repo_name' and 'id' fields!
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


def extract_from_result_files():
    """Extract instance -> repo mapping from result JSONL files."""

    print("Extracting repo mappings from result files...")

    instance_to_repo = {}

    # Find all jobs_results_diff.jsonl files
    result_files = list(PROJECT_ROOT.glob("results/**/jobs_results_diff.jsonl"))

    print(f"Found {len(result_files)} result files")

    for result_file in result_files:
        with open(result_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    inst_id = entry.get('id')
                    repo = entry.get('repo_name')

                    if inst_id and repo:
                        # Normalize repo name (remove owner prefix if present)
                        if '/' in repo:
                            repo_name = repo.split('/')[-1]
                        else:
                            repo_name = repo

                        instance_to_repo[inst_id] = repo_name
                except:
                    continue

    return instance_to_repo


def main():
    """Extract and save complete instance -> repo mapping."""

    print("="*70)
    print("EXTRACTING INSTANCE -> REPO MAPPING")
    print("="*70)
    print()

    # Extract from result files
    instance_to_repo = extract_from_result_files()

    print(f"\nTotal instances with repo info: {len(instance_to_repo)}")

    # Load memory and eval IDs
    with open(PROJECT_ROOT / 'data/memory_issue_ids.json') as f:
        memory_ids = set(json.load(f))

    with open(PROJECT_ROOT / 'data/eval_issue_ids.json') as f:
        eval_ids = set(json.load(f))

    # Check coverage
    memory_coverage = len([i for i in memory_ids if i in instance_to_repo])
    eval_coverage = len([i for i in eval_ids if i in instance_to_repo])

    print(f"\nCoverage:")
    print(f"  Memory: {memory_coverage}/{len(memory_ids)}")
    print(f"  Eval: {eval_coverage}/{len(eval_ids)}")

    # Save mapping
    output_dir = PROJECT_ROOT / "results/cold_start"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "instance_to_repo.json"
    with open(output_file, 'w') as f:
        json.dump(instance_to_repo, f, indent=2, sort_keys=True)

    print(f"\n✓ Saved mapping to {output_file}")

    # Show repo distribution
    repo_counts = defaultdict(int)
    for repo in instance_to_repo.values():
        repo_counts[repo] += 1

    print(f"\nUnique repos: {len(repo_counts)}")
    print("\nTop 10 repos:")
    for repo, count in sorted(repo_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {repo}: {count} instances")


if __name__ == "__main__":
    main()
