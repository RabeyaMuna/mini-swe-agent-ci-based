#!/usr/bin/env python3
"""
Create eval_set.jsonl for cross-repository evaluation.

Takes the first 100 eval IDs and creates a JSONL file with instance data.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    # Load first 100 eval IDs
    eval_ids_file = PROJECT_ROOT / "experimental_data/cross_repository/eval_issue_ids.json"
    with open(eval_ids_file) as f:
        eval_ids = [str(id) for id in json.load(f)]

    eval_ids_set = set(eval_ids)
    print(f"Target: {len(eval_ids)} eval IDs\n")

    # Load instance data from result file
    result_file = PROJECT_ROOT / "results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl"

    instances = []
    with open(result_file) as f:
        for line in f:
            entry = json.loads(line)
            inst_id = str(entry['id'])

            if inst_id in eval_ids_set:
                # Create instance in expected format
                instance = {
                    "instance_id": inst_id,
                    "id": inst_id,
                    "repo_name": entry['repo_name'],
                    "repo_owner": "Muna4029",  # Default
                    "repo": f"Muna4029/{entry['repo_name']}",
                }
                instances.append(instance)

    print(f"Found: {len(instances)} instances\n")

    # Save to eval_set.jsonl
    output_file = PROJECT_ROOT / "experimental_data/cross_repository/eval_set.jsonl"
    with open(output_file, 'w') as f:
        for instance in instances:
            f.write(json.dumps(instance) + '\n')

    print(f"✓ Created: {output_file}")
    print(f"  {len(instances)} instances")

    # Show first few
    print("\nFirst 5 instances:")
    for inst in instances[:5]:
        print(f"  ID {inst['id']}: {inst['repo_name']}")


if __name__ == "__main__":
    main()
