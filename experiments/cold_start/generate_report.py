#!/usr/bin/env python3
"""
Generate complete partition report from saved data.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    """Generate markdown report from partition data."""

    # Load detailed partition
    with open(PROJECT_ROOT / "results/cold_start/detailed_partition.json") as f:
        partition = json.load(f)

    ha_data = partition["history_available"]
    cs_data = partition["cold_start"]

    report = []
    report.append("# Complete Cold-Start Partition Report\n")
    report.append("## Overview\n")
    report.append(f"- **Total Evaluation Instances**: {ha_data['total_instances'] + cs_data['total_instances']}")
    report.append(f"- **History-Available Instances**: {ha_data['total_instances']} ({ha_data['total_instances']/(ha_data['total_instances']+cs_data['total_instances'])*100:.1f}%)")
    report.append(f"- **Cold-Start Instances**: {cs_data['total_instances']} ({cs_data['total_instances']/(ha_data['total_instances']+cs_data['total_instances'])*100:.1f}%)\n")
    report.append("---\n")

    # History-Available
    report.append("## Part 1: History-Available Repos and Instances\n")
    report.append("**Definition**: Repos that appear in the 157 memory instances (have historical repair experience)\n")
    report.append(f"**Total**: {ha_data['total_repos']} repos, {ha_data['total_instances']} evaluation instances\n")
    report.append("### Detailed Breakdown by Repository\n")

    # Sort by number of eval instances (descending)
    ha_repos_sorted = sorted(
        ha_data['repos'].items(),
        key=lambda x: len(x[1]['eval_instances']),
        reverse=True
    )

    for i, (repo, data) in enumerate(ha_repos_sorted, 1):
        eval_ids = ', '.join(data['eval_instances'])
        mem_ids = ', '.join(data['memory_instances'])

        report.append(f"#### {i}. {repo}")
        report.append(f"- **Evaluation instances ({len(data['eval_instances'])})**:")
        report.append(f"  {eval_ids}")
        report.append(f"- **Memory instances ({len(data['memory_instances'])})**:")
        report.append(f"  {mem_ids}\n")

    # Cold-Start
    report.append("---\n")
    report.append("## Part 2: Cold-Start Repos and Instances\n")
    report.append("**Definition**: Repos that do NOT appear in the 157 memory instances (NO historical repair experience)\n")
    report.append(f"**Total**: {cs_data['total_repos']} repos, {cs_data['total_instances']} evaluation instances\n")
    report.append("### Complete List\n")

    # Sort alphabetically
    cs_repos_sorted = sorted(cs_data['repos'].items())

    for i, (repo, data) in enumerate(cs_repos_sorted, 1):
        eval_ids = ', '.join(data['eval_instances'])
        report.append(f"{i}. **{repo}** - Eval: {eval_ids}")

    report.append("\n---\n")
    report.append("## Summary Statistics\n")
    report.append("### History-Available")
    report.append(f"- **Repos**: {ha_data['total_repos']}")
    report.append(f"- **Eval Instances**: {ha_data['total_instances']}")
    report.append(f"- **Avg eval per repo**: {ha_data['total_instances']/ha_data['total_repos']:.1f}\n")

    report.append("### Cold-Start")
    report.append(f"- **Repos**: {cs_data['total_repos']}")
    report.append(f"- **Eval Instances**: {cs_data['total_instances']}")
    report.append(f"- **Avg eval per repo**: {cs_data['total_instances']/cs_data['total_repos']:.1f}")
    report.append(f"- **Pattern**: Each repo has exactly 1 eval instance (truly cold-start)\n")

    # Save report
    output_file = PROJECT_ROOT / "experiments/cold_start/REPOS_AND_INSTANCES.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Generated report: {output_file}")
    print(f"  History-Available: {ha_data['total_instances']} instances from {ha_data['total_repos']} repos")
    print(f"  Cold-Start: {cs_data['total_instances']} instances from {cs_data['total_repos']} repos")


if __name__ == "__main__":
    main()
