#!/usr/bin/env python3
"""
Unified cost viewer for both Codex and Minisweagent results.

Supports both formats:
- cost_time_report.json (Codex format - detailed with tokens, timing)
- costs.json (Minisweagent format - simple instance costs)

Usage:
    python scripts/view_all_costs.py results/
    python scripts/view_all_costs.py results/codex/backward/
    python scripts/view_all_costs.py results/miniswe-agent/baseline_deepseek-v4-flash/
    python scripts/view_all_costs.py results/ --format table
    python scripts/view_all_costs.py results/ --format json --output total_costs.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def read_codex_cost_report(report_path: Path) -> List[Dict[str, Any]]:
    """Read Codex cost_time_report.json format."""
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        run_info = data.get("run", {})

        costs = []
        for instance in data.get("instances", []):
            costs.append({
                "instance_id": instance.get("instance_id"),
                "agent": run_info.get("agent", "codex"),
                "model": run_info.get("model", "unknown"),
                "ablation": run_info.get("ablation", "unknown"),
                "direction": run_info.get("direction", "unknown"),
                "cost_usd": instance.get("total_cost_usd", 0.0),
                "input_tokens": instance.get("total_input_tokens", 0),
                "output_tokens": instance.get("total_output_tokens", 0),
                "wall_time_seconds": instance.get("total_wall_time_seconds", 0),
                "api_time_seconds": instance.get("total_api_time_seconds", 0),
                "attempts": instance.get("attempts", 1),
                "status": instance.get("last_status", "unknown"),
                "format": "codex"
            })
        return costs
    except Exception as e:
        print(f"Warning: Failed to read {report_path}: {e}", file=sys.stderr)
        return []


def read_miniswe_costs(costs_path: Path) -> List[Dict[str, Any]]:
    """Read Minisweagent costs.json format."""
    try:
        data = json.loads(costs_path.read_text(encoding="utf-8"))
        costs = []

        for config_key, config_data in data.items():
            model = config_data.get("model", "unknown")
            ablation = config_data.get("ablation", "unknown")
            direction = config_data.get("direction", "unknown")

            for instance_id, cost_usd in config_data.get("instances", {}).items():
                costs.append({
                    "instance_id": instance_id,
                    "agent": "miniswe-agent",
                    "model": model,
                    "ablation": ablation,
                    "direction": direction,
                    "cost_usd": cost_usd,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "wall_time_seconds": 0,
                    "api_time_seconds": 0,
                    "attempts": 1,
                    "status": "completed",
                    "format": "miniswe"
                })
        return costs
    except Exception as e:
        print(f"Warning: Failed to read {costs_path}: {e}", file=sys.stderr)
        return []


def collect_all_costs(root_dir: Path) -> List[Dict[str, Any]]:
    """Collect costs from both Codex and Minisweagent formats."""
    all_costs = []

    # Find all cost_time_report.json (Codex)
    for report_path in root_dir.rglob("cost_time_report.json"):
        costs = read_codex_cost_report(report_path)
        all_costs.extend(costs)
        if costs:
            print(f"✓ Loaded {len(costs)} instances from {report_path.relative_to(root_dir)}", file=sys.stderr)

    # Find all costs.json (Minisweagent)
    for costs_path in root_dir.rglob("costs.json"):
        costs = read_miniswe_costs(costs_path)
        all_costs.extend(costs)
        if costs:
            print(f"✓ Loaded {len(costs)} instances from {costs_path.relative_to(root_dir)}", file=sys.stderr)

    return all_costs


def group_costs(costs: List[Dict[str, Any]], group_by: str) -> Dict[str, List[Dict[str, Any]]]:
    """Group costs by a field."""
    groups = defaultdict(list)
    for cost in costs:
        key = cost.get(group_by, "unknown")
        groups[key].append(cost)
    return dict(groups)


def calculate_stats(costs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics for a group of costs."""
    if not costs:
        return {
            "count": 0,
            "total_cost_usd": 0.0,
            "avg_cost_usd": 0.0,
            "min_cost_usd": 0.0,
            "max_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_wall_time_hours": 0.0,
            "total_api_time_seconds": 0.0,
        }

    total_cost = sum(c["cost_usd"] for c in costs)
    total_input_tokens = sum(c["input_tokens"] for c in costs)
    total_output_tokens = sum(c["output_tokens"] for c in costs)
    total_wall_time = sum(c["wall_time_seconds"] for c in costs)
    total_api_time = sum(c["api_time_seconds"] for c in costs)

    return {
        "count": len(costs),
        "total_cost_usd": round(total_cost, 4),
        "avg_cost_usd": round(total_cost / len(costs), 4),
        "min_cost_usd": round(min(c["cost_usd"] for c in costs), 4),
        "max_cost_usd": round(max(c["cost_usd"] for c in costs), 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_wall_time_hours": round(total_wall_time / 3600, 2),
        "total_api_time_seconds": round(total_api_time, 2),
    }


def format_table(stats: Dict[str, Any], title: str = "Summary") -> str:
    """Format statistics as a table."""
    lines = [
        "=" * 100,
        f"  {title}",
        "=" * 100,
        f"{'Instances:':<40} {stats['count']:>15,}",
        f"{'Total Cost (USD):':<40} ${stats['total_cost_usd']:>14,.4f}",
        f"{'Average Cost per Instance (USD):':<40} ${stats['avg_cost_usd']:>14,.4f}",
        f"{'Min Cost (USD):':<40} ${stats['min_cost_usd']:>14,.4f}",
        f"{'Max Cost (USD):':<40} ${stats['max_cost_usd']:>14,.4f}",
    ]

    if stats['total_tokens'] > 0:
        lines.extend([
            f"{'Total Input Tokens:':<40} {stats['total_input_tokens']:>15,}",
            f"{'Total Output Tokens:':<40} {stats['total_output_tokens']:>15,}",
            f"{'Total Tokens:':<40} {stats['total_tokens']:>15,}",
        ])

    if stats['total_wall_time_hours'] > 0:
        lines.append(f"{'Total Wall Time (hours):':<40} {stats['total_wall_time_hours']:>15,.2f}")

    if stats['total_api_time_seconds'] > 0:
        lines.append(f"{'Total API Time (seconds):':<40} {stats['total_api_time_seconds']:>15,.2f}")

    lines.append("=" * 100)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Unified cost viewer for Codex and Minisweagent results"
    )
    parser.add_argument(
        "results_dir",
        help="Path to results directory (supports both formats)"
    )
    parser.add_argument(
        "--group-by",
        choices=["agent", "model", "ablation", "direction", "none"],
        default="agent",
        help="Group results by field (default: agent)"
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)"
    )
    parser.add_argument(
        "--output",
        help="Save results to file (JSON format only)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}", file=sys.stderr)
        return 1

    print(f"\n🔍 Scanning {results_dir} for cost reports...\n", file=sys.stderr)
    all_costs = collect_all_costs(results_dir)

    if not all_costs:
        print("\n❌ No cost reports found!", file=sys.stderr)
        print("   Looked for: cost_time_report.json (Codex) and costs.json (Minisweagent)", file=sys.stderr)
        return 1

    print(f"\n✅ Found {len(all_costs)} total instances\n", file=sys.stderr)

    # Calculate overall stats
    overall_stats = calculate_stats(all_costs)

    if args.format == "table":
        print(format_table(overall_stats, "Overall Summary"))

        if args.group_by != "none":
            grouped = group_costs(all_costs, args.group_by)
            for group_name in sorted(grouped.keys()):
                group_instances = grouped[group_name]
                group_stats = calculate_stats(group_instances)
                print(f"\n{format_table(group_stats, f'{args.group_by.capitalize()}: {group_name}')}")

    # JSON output
    output_data = {
        "summary": overall_stats,
        "instances": all_costs,
    }

    if args.group_by != "none":
        grouped = group_costs(all_costs, args.group_by)
        output_data["grouped"] = {
            name: calculate_stats(instances)
            for name, instances in grouped.items()
        }

    if args.format == "json" or args.output:
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
            print(f"\n✅ Saved to: {output_path}", file=sys.stderr)
        else:
            print(json.dumps(output_data, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
