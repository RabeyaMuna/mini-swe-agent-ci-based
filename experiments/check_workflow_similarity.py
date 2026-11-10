#!/usr/bin/env python3
"""
Check if specific instance IDs share the same CI workflow.
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent

# Instance IDs to check
TARGET_IDS = ['466', '488', '494', '479', '485', '447', '490', '473', '495',
              '459', '480', '493', '491', '475', '457']


def load_memory_instances():
    """Load memory instances."""
    memory_file = PROJECT_ROOT / "data/bidirect_trs/failure_memory.json"

    with open(memory_file) as f:
        memory_data = json.load(f)

    return memory_data.get("memory", [])


def check_result_files():
    """Check result files for instance info."""
    result_file = PROJECT_ROOT / "results/miniswe-agent/baseline_deepseek-v4-flash/jobs_results_diff.jsonl"

    instances = {}
    with open(result_file) as f:
        for line in f:
            entry = json.loads(line)
            inst_id = str(entry['id'])
            if inst_id in TARGET_IDS:
                instances[inst_id] = {
                    'id': inst_id,
                    'repo': entry.get('repo_name', 'unknown'),
                    'conclusion': entry.get('conclusion', 'unknown')
                }

    return instances


def main():
    print("\n" + "="*70)
    print("CHECKING CI WORKFLOW SIMILARITY")
    print("="*70 + "\n")

    print(f"Target IDs: {', '.join(TARGET_IDS)}\n")

    # Load memory instances
    print("Loading memory instances...")
    memory_instances = load_memory_instances()

    # Find matching instances in memory
    memory_matches = {}
    for inst in memory_instances:
        inst_id = str(inst.get('instance_id', ''))
        if inst_id in TARGET_IDS:
            memory_matches[inst_id] = {
                'id': inst_id,
                'repo': inst.get('repo_name', 'unknown'),
                'error_type': inst.get('error_type', 'unknown'),
                'workflow_file': inst.get('workflow_file', 'unknown'),
                'workflow_name': inst.get('workflow_name', 'unknown'),
                'job_name': inst.get('job_name', 'unknown'),
            }

    print(f"Found {len(memory_matches)} instances in memory\n")

    # Check result files for repo info
    print("Checking result files for additional info...")
    result_matches = check_result_files()
    print(f"Found {len(result_matches)} instances in result files\n")

    # Group by repo
    print("="*70)
    print("INSTANCES BY REPOSITORY")
    print("="*70 + "\n")

    by_repo = defaultdict(list)
    for inst_id in TARGET_IDS:
        if inst_id in memory_matches:
            repo = memory_matches[inst_id]['repo']
        elif inst_id in result_matches:
            repo = result_matches[inst_id]['repo']
        else:
            repo = 'unknown'
        by_repo[repo].append(inst_id)

    for repo, ids in sorted(by_repo.items()):
        print(f"{repo}: {len(ids)} instances")
        print(f"  IDs: {', '.join(sorted(ids))}\n")

    # Check workflow similarity for memory instances
    if memory_matches:
        print("="*70)
        print("WORKFLOW INFORMATION (from memory)")
        print("="*70 + "\n")

        by_workflow = defaultdict(list)
        for inst_id, info in sorted(memory_matches.items()):
            workflow = info.get('workflow_file', 'unknown')
            by_workflow[workflow].append(inst_id)

            print(f"ID {inst_id}:")
            print(f"  Repo: {info['repo']}")
            print(f"  Workflow file: {info.get('workflow_file', 'N/A')}")
            print(f"  Workflow name: {info.get('workflow_name', 'N/A')}")
            print(f"  Job name: {info.get('job_name', 'N/A')}")
            print(f"  Error type: {info.get('error_type', 'N/A')}\n")

        print("="*70)
        print("GROUPED BY WORKFLOW FILE")
        print("="*70 + "\n")

        for workflow, ids in sorted(by_workflow.items()):
            print(f"Workflow: {workflow}")
            print(f"  Instances: {', '.join(sorted(ids))}")
            print(f"  Count: {len(ids)}\n")

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70 + "\n")

    axolotl_ids = [id for id in TARGET_IDS if id in memory_matches and memory_matches[id]['repo'] == 'axolotl']

    if axolotl_ids:
        print(f"Axolotl instances found: {len(axolotl_ids)}")
        print(f"  IDs: {', '.join(sorted(axolotl_ids))}\n")

        # Check if they share the same workflow
        workflows = set()
        for inst_id in axolotl_ids:
            if inst_id in memory_matches:
                workflow = memory_matches[inst_id].get('workflow_file', 'unknown')
                workflows.add(workflow)

        if len(workflows) == 1:
            print(f"✓ All axolotl instances use the SAME workflow: {list(workflows)[0]}")
        else:
            print(f"✗ Axolotl instances use DIFFERENT workflows:")
            for w in sorted(workflows):
                ids_with_w = [id for id in axolotl_ids if memory_matches.get(id, {}).get('workflow_file') == w]
                print(f"  {w}: {', '.join(sorted(ids_with_w))}")
    else:
        print("⚠️  No axolotl instances found in memory")
        print("    These IDs might be in eval set only, not in memory")

        axolotl_in_results = [id for id in TARGET_IDS if id in result_matches and result_matches[id]['repo'] == 'axolotl']
        if axolotl_in_results:
            print(f"\n    Found {len(axolotl_in_results)} axolotl instances in result files:")
            print(f"    IDs: {', '.join(sorted(axolotl_in_results))}")

    print()


if __name__ == "__main__":
    main()
