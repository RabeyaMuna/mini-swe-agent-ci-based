#!/usr/bin/env python3
"""
analyze_three_views_cost.py - Analyze Cost Breakdown by Three Views + Memory Levels
==================================================================================

Analyzes batch results and breaks down costs by:
1. Endpoint View (CI logs + workflow)
2. Development View + Reconciliation
3. L1 (Concrete Experiences)
4. L2 (Contextual Strategies)
5. L3 (Transferable Patterns)

Finds maximum token usage per stage for conservative 157-instance estimate.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analyze_batch_results(batch_dir: Path) -> Dict[str, Any]:
    """
    Analyze all instances in a batch directory.

    Returns breakdown by stage with max/avg/total statistics.
    """
    per_instance_dir = batch_dir / "per_instance"

    if not per_instance_dir.exists():
        print(f"ERROR: {per_instance_dir} not found")
        return {}

    instances = []

    # Collect all instance results
    for instance_path in sorted(per_instance_dir.iterdir()):
        if not instance_path.is_dir():
            continue

        instance_id = instance_path.name
        detailed_usage_path = instance_path / "detailed_usage.json"

        if not detailed_usage_path.exists():
            print(f"  SKIP {instance_id}: no detailed_usage.json")
            continue

        with open(detailed_usage_path) as f:
            usage_data = json.load(f)

        if not usage_data:
            continue

        instance_data = usage_data[0]
        phase_breakdown = instance_data.get('phase_breakdown', {})

        # Extract metrics for each stage
        # Support both old and new phase names
        stages = {
            'endpoint_view': (
                phase_breakdown.get('endpoint_view_backward', {}) or
                phase_breakdown.get('endpoint_view_reconstruction', {})
            ),
            'development_view': (
                phase_breakdown.get('development_view_commit_based', {}) or
                phase_breakdown.get('development_view_and_reconciliation', {})  # Old combined
            ),
            'reconciliation': phase_breakdown.get('cross_view_reconciliation_bidirectional', {}),
            'problem_dedup': phase_breakdown.get('problem_deduplication', {}),
            'l1_build': phase_breakdown.get('build_l1', {}),
            'l2_build': phase_breakdown.get('build_l2', {}),
            'l3_build': phase_breakdown.get('build_l3', {}),
        }

        # Calculate totals for this instance
        instance_summary = {
            'instance_id': instance_id,
            'total_tokens': instance_data.get('summary', {}).get('total_tokens', 0),
            'total_cost': instance_data.get('summary', {}).get('total_cost_usd', 0),
            'stages': {}
        }

        for stage_name, stage_data in stages.items():
            if stage_data:
                instance_summary['stages'][stage_name] = {
                    'tokens': stage_data.get('total_tokens', 0),
                    'input_tokens': stage_data.get('input_tokens', 0),
                    'output_tokens': stage_data.get('output_tokens', 0),
                    'cost_usd': stage_data.get('total_cost_usd', 0),
                    'call_count': stage_data.get('call_count', 0),
                }
            else:
                instance_summary['stages'][stage_name] = {
                    'tokens': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'cost_usd': 0,
                    'call_count': 0,
                }

        instances.append(instance_summary)

    if not instances:
        print("ERROR: No valid instances found")
        return {}

    # Calculate statistics per stage
    stage_stats = {}
    stage_names = [
        'endpoint_view',
        'development_view',
        'reconciliation',
        'problem_dedup',
        'l1_build',
        'l2_build',
        'l3_build',
    ]

    for stage in stage_names:
        stage_tokens = [inst['stages'][stage]['tokens'] for inst in instances]
        stage_costs = [inst['stages'][stage]['cost_usd'] for inst in instances]

        stage_stats[stage] = {
            'min_tokens': min(stage_tokens),
            'max_tokens': max(stage_tokens),
            'avg_tokens': sum(stage_tokens) / len(stage_tokens),
            'total_tokens': sum(stage_tokens),
            'min_cost': min(stage_costs),
            'max_cost': max(stage_costs),
            'avg_cost': sum(stage_costs) / len(stage_costs),
            'total_cost': sum(stage_costs),
            'instances_with_calls': sum(1 for inst in instances if inst['stages'][stage]['call_count'] > 0),
        }

        # Find instance with max tokens for this stage
        max_instance = max(instances, key=lambda x: x['stages'][stage]['tokens'])
        stage_stats[stage]['max_instance_id'] = max_instance['instance_id']

    # Overall statistics
    total_tokens = [inst['total_tokens'] for inst in instances]
    total_costs = [inst['total_cost'] for inst in instances]

    overall_stats = {
        'num_instances': len(instances),
        'total_tokens': sum(total_tokens),
        'avg_tokens_per_instance': sum(total_tokens) / len(total_tokens),
        'max_tokens_per_instance': max(total_tokens),
        'min_tokens_per_instance': min(total_tokens),
        'total_cost': sum(total_costs),
        'avg_cost_per_instance': sum(total_costs) / len(total_costs),
        'max_cost_per_instance': max(total_costs),
        'min_cost_per_instance': min(total_costs),
    }

    max_instance = max(instances, key=lambda x: x['total_tokens'])
    overall_stats['max_instance_id'] = max_instance['instance_id']

    return {
        'overall': overall_stats,
        'stages': stage_stats,
        'instances': instances,
    }


def print_analysis(analysis: Dict[str, Any], output_dir: Path):
    """Print and save detailed analysis."""

    print("=" * 80)
    print("THREE-VIEW COST BREAKDOWN ANALYSIS")
    print("=" * 80)
    print()

    overall = analysis['overall']
    stages = analysis['stages']

    print(f"Total Instances: {overall['num_instances']}")
    print(f"Total Tokens: {overall['total_tokens']:,}")
    print(f"Total Cost: ${overall['total_cost']:.6f}")
    print()

    print("=" * 80)
    print("PER-STAGE BREAKDOWN (using MAXIMUM token usage per stage)")
    print("=" * 80)
    print()

    stage_labels = {
        'endpoint_view': 'VIEW 1: Endpoint (Backward from CI)',
        'development_view': 'VIEW 2: Development (Commit-based)',
        'reconciliation': 'VIEW 3: Reconciliation (Bidirectional)',
        'problem_dedup': 'Problem Deduplication',
        'l1_build': 'L1: Concrete Experiences',
        'l2_build': 'L2: Contextual Strategies',
        'l3_build': 'L3: Transferable Patterns',
    }

    # Calculate percentages based on max
    total_max_tokens = sum(stages[s]['max_tokens'] for s in stage_labels.keys())

    print(f"{'Stage':<45} {'Max Tokens':>12} {'Max Cost':>12} {'% of Total':>10}")
    print("-" * 80)

    for stage_key, stage_label in stage_labels.items():
        stage_data = stages[stage_key]
        max_tokens = stage_data['max_tokens']
        max_cost = stage_data['max_cost']
        max_instance = stage_data['max_instance_id']
        pct = (max_tokens / total_max_tokens * 100) if total_max_tokens > 0 else 0

        print(f"{stage_label:<45} {max_tokens:>12,} ${max_cost:>11.6f} {pct:>9.1f}%")
        print(f"  (Instance {max_instance})")

    print("-" * 80)
    print(f"{'TOTAL (using max per stage)':<45} {total_max_tokens:>12,} ${sum(stages[s]['max_cost'] for s in stage_labels.keys()):>11.6f} {100:>9.1f}%")
    print()

    print("=" * 80)
    print("PROJECTION FOR 157 INSTANCES (using maximum per stage)")
    print("=" * 80)
    print()

    # Conservative estimate: use max tokens/cost per stage
    total_projected_tokens = sum(stages[s]['max_tokens'] for s in stage_labels.keys()) * 157
    total_projected_cost = sum(stages[s]['max_cost'] for s in stage_labels.keys()) * 157

    print(f"{'Stage':<45} {'Tokens/inst':>12} {'Cost/inst':>12} {'Total (157)':>15}")
    print("-" * 80)

    for stage_key, stage_label in stage_labels.items():
        stage_data = stages[stage_key]
        max_tokens = stage_data['max_tokens']
        max_cost = stage_data['max_cost']
        projected_tokens = max_tokens * 157
        projected_cost = max_cost * 157

        print(f"{stage_label:<45} {max_tokens:>12,} ${max_cost:>11.6f} {projected_tokens/1e6:>9.1f}M / ${projected_cost:>6.2f}")

    print("-" * 80)
    total_cost_per_inst = sum(stages[s]['max_cost'] for s in stage_labels.keys())
    print(f"{'TOTAL':<45} {total_max_tokens:>12,} ${total_cost_per_inst:>11.6f} {total_projected_tokens/1e6:>9.1f}M / ${total_projected_cost:>6.2f}")
    print()

    # Save detailed breakdown
    output_file = output_dir / "three_view_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Detailed analysis saved to: {output_file}")
    print()

    # Create CSV for easy import
    import csv
    csv_file = output_dir / "stage_breakdown.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stage', 'Max_Tokens', 'Max_Cost_USD', 'Projected_157_Tokens', 'Projected_157_Cost_USD'])

        for stage_key, stage_label in stage_labels.items():
            stage_data = stages[stage_key]
            writer.writerow([
                stage_label,
                stage_data['max_tokens'],
                stage_data['max_cost'],
                stage_data['max_tokens'] * 157,
                stage_data['max_cost'] * 157,
            ])

        writer.writerow([
            'TOTAL',
            total_max_tokens,
            total_cost_per_inst,
            total_projected_tokens,
            total_projected_cost,
        ])

    print(f"CSV saved to: {csv_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_three_views_cost.py <batch_results_dir>")
        print()
        print("Example:")
        print("  python scripts/analyze_three_views_cost.py results/metrics/batch_25_three_views")
        return 1

    batch_dir = Path(sys.argv[1])

    if not batch_dir.exists():
        print(f"ERROR: {batch_dir} does not exist")
        return 1

    print(f"Analyzing batch results from: {batch_dir}")
    print()

    analysis = analyze_batch_results(batch_dir)

    if not analysis:
        return 1

    print_analysis(analysis, batch_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
