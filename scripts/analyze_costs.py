#!/usr/bin/env python3
"""
Analyze API costs and time from benchmark trajectories.

Usage:
    python scripts/analyze_costs.py results/minimax/L1+L2+L3/
    python scripts/analyze_costs.py results/ --group-by model
    python scripts/analyze_costs.py results/ --group-by ablation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_trajectory(traj_path: Path) -> Dict[str, Any]:
    """Load trajectory JSON file."""
    with open(traj_path, 'r') as f:
        return json.load(f)


def extract_cost_info(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract cost tracking info from trajectory."""
    info = traj.get("info", {})
    cost_tracking = info.get("cost_tracking", {})

    return {
        "instance_id": traj.get("instance_id", "unknown"),
        "model": cost_tracking.get("model", "unknown"),
        "ablation": cost_tracking.get("ablation", "unknown"),
        "duration_seconds": cost_tracking.get("total_duration_seconds", 0),
        "total_cost_usd": cost_tracking.get("total_cost_usd", 0),
        "agent_cost_usd": cost_tracking.get("agent_cost_usd", 0),
        "input_tokens": cost_tracking.get("total_input_tokens", 0),
        "output_tokens": cost_tracking.get("total_output_tokens", 0),
    }


def collect_costs(results_dir: Path) -> List[Dict[str, Any]]:
    """Collect cost data from all trajectories."""
    costs = []

    # Find all .traj.json files
    for traj_file in results_dir.rglob("*.traj.json"):
        try:
            traj = load_trajectory(traj_file)
            cost_info = extract_cost_info(traj)
            costs.append(cost_info)
        except Exception as e:
            print(f"Warning: Failed to load {traj_file}: {e}")

    return costs


def calculate_stats(costs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics."""
    if not costs:
        return {
            "count": 0,
            "total_cost": 0,
            "avg_cost": 0,
            "min_cost": 0,
            "max_cost": 0,
            "total_duration": 0,
            "avg_duration": 0,
        }

    total_cost = sum(c["total_cost_usd"] for c in costs)
    total_duration = sum(c["duration_seconds"] for c in costs)

    return {
        "count": len(costs),
        "total_cost_usd": round(total_cost, 4),
        "avg_cost_usd": round(total_cost / len(costs), 4),
        "min_cost_usd": round(min(c["total_cost_usd"] for c in costs), 4),
        "max_cost_usd": round(max(c["total_cost_usd"] for c in costs), 4),
        "total_duration_seconds": round(total_duration, 2),
        "avg_duration_seconds": round(total_duration / len(costs), 2),
        "total_duration_hours": round(total_duration / 3600, 2),
    }


def group_by_field(costs: List[Dict[str, Any]], field: str) -> Dict[str, List[Dict[str, Any]]]:
    """Group costs by a specific field."""
    groups = defaultdict(list)
    for cost in costs:
        groups[cost[field]].append(cost)
    return dict(groups)


def print_summary(stats: Dict[str, Any], title: str = "Summary"):
    """Print summary statistics in a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print('=' * 80)
    print(f"{'Total Issues:':<30} {stats['count']:>15,}")
    print(f"{'Total Cost (USD):':<30} ${stats['total_cost_usd']:>14,.4f}")
    print(f"{'Average Cost per Issue (USD):':<30} ${stats['avg_cost_usd']:>14,.4f}")
    print(f"{'Min Cost (USD):':<30} ${stats['min_cost_usd']:>14,.4f}")
    print(f"{'Max Cost (USD):':<30} ${stats['max_cost_usd']:>14,.4f}")
    print(f"{'Total Duration:':<30} {stats['total_duration_hours']:>12,.2f} hours")
    print(f"{'Average Duration per Issue:':<30} {stats['avg_duration_seconds']:>12,.2f} seconds")
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark costs and timing")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--group-by", choices=["model", "ablation", "none"], default="none",
                        help="Group results by model or ablation")
    parser.add_argument("--output", help="Save detailed results to JSON file")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return 1

    print(f"Collecting cost data from: {results_dir}")
    costs = collect_costs(results_dir)

    if not costs:
        print("No trajectory files found!")
        return 1

    print(f"Found {len(costs)} trajectories\n")

    # Overall summary
    overall_stats = calculate_stats(costs)
    print_summary(overall_stats, "Overall Summary")

    # Group by model or ablation if requested
    if args.group_by != "none":
        print(f"\n\nGrouping by: {args.group_by}")
        groups = group_by_field(costs, args.group_by)

        for group_name, group_costs in sorted(groups.items()):
            group_stats = calculate_stats(group_costs)
            print_summary(group_stats, f"{args.group_by.capitalize()}: {group_name}")

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "summary": overall_stats,
            "details": costs,
        }

        if args.group_by != "none":
            groups = group_by_field(costs, args.group_by)
            output_data["grouped"] = {
                name: calculate_stats(group_costs)
                for name, group_costs in groups.items()
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Saved detailed results to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
