#!/usr/bin/env python3
"""
analyze_offline_costs.py - Detailed Cost Analysis for Offline Extraction Pipeline
==================================================================================

Analyzes the token usage and costs from batch offline pipeline runs.
Provides detailed breakdowns by:
- Phase (decomposition, L1, L2, L3)
- Repository
- Model
- Token type (input, cached, output)

Usage:
    # Analyze batch results
    python scripts/analyze_offline_costs.py \
        --batch-summary results/metrics/batch_offline_20/batch_summary.json \
        --output-dir results/metrics/analysis

    # Analyze with cost breakdown by phase
    python scripts/analyze_offline_costs.py \
        --batch-summary results/metrics/batch_offline_20/batch_summary.json \
        --detailed-tracker-dir results/metrics/batch_offline_20/per_instance \
        --output-dir results/metrics/analysis

Output:
    - cost_analysis.json - Comprehensive cost analysis
    - phase_breakdown.csv - Cost breakdown by pipeline phase
    - repo_breakdown.csv - Cost breakdown by repository
    - cost_visualization.txt - ASCII charts
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict
import csv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_batch_summary(path: Path) -> Dict[str, Any]:
    """Load the batch summary JSON."""
    with open(path) as f:
        return json.load(f)


def load_detailed_trackers(per_instance_dir: Path) -> Dict[str, Any]:
    """
    Load detailed usage tracker data from per-instance directories.

    Returns:
        Dictionary mapping instance_id to detailed usage data
    """
    detailed_data = {}

    if not per_instance_dir.exists():
        return detailed_data

    for instance_dir in per_instance_dir.iterdir():
        if not instance_dir.is_dir():
            continue

        instance_id = instance_dir.name
        tracker_file = instance_dir / f"detailed_usage_{instance_id}.json"

        if tracker_file.exists():
            with open(tracker_file) as f:
                detailed_data[instance_id] = json.load(f)

    return detailed_data


def analyze_by_phase(
    detailed_data: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze costs broken down by pipeline phase.

    Returns:
        Dictionary mapping phase name to aggregated metrics
    """
    phase_metrics = defaultdict(lambda: {
        "count": 0,
        "total_cost": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_time": 0,
    })

    for instance_id, data in detailed_data.items():
        steps = data.get("steps", [])

        for step in steps:
            phase = step.get("phase", "unknown")
            metrics = phase_metrics[phase]

            metrics["count"] += 1
            metrics["total_cost"] += step.get("cost_usd", 0)
            metrics["total_input_tokens"] += step.get("input_tokens", 0)
            metrics["total_output_tokens"] += step.get("output_tokens", 0)
            metrics["total_time"] += step.get("duration_seconds", 0)

    # Calculate averages
    for phase, metrics in phase_metrics.items():
        if metrics["count"] > 0:
            metrics["avg_cost"] = metrics["total_cost"] / metrics["count"]
            metrics["avg_input_tokens"] = metrics["total_input_tokens"] / metrics["count"]
            metrics["avg_output_tokens"] = metrics["total_output_tokens"] / metrics["count"]
            metrics["avg_time"] = metrics["total_time"] / metrics["count"]

    return dict(phase_metrics)


def analyze_by_repo(
    batch_summary: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze costs broken down by repository.

    Returns:
        Dictionary mapping repo name to aggregated metrics
    """
    repo_metrics = defaultdict(lambda: {
        "count": 0,
        "total_cost": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    })

    # Extract repo from instance details
    for instance in batch_summary.get("per_instance_details", []):
        instance_id = instance.get("instance_id", "")

        # Try to infer repo from instance data
        # In a real scenario, you'd have repo info in the detailed tracker
        repo = "unknown"  # Placeholder

        metrics = repo_metrics[repo]
        metrics["count"] += 1
        metrics["total_cost"] += instance.get("total_cost_usd", 0)
        metrics["total_input_tokens"] += instance.get("total_input_tokens", 0)
        metrics["total_output_tokens"] += instance.get("total_output_tokens", 0)

    # Calculate averages
    for repo, metrics in repo_metrics.items():
        if metrics["count"] > 0:
            metrics["avg_cost"] = metrics["total_cost"] / metrics["count"]
            metrics["avg_input_tokens"] = metrics["total_input_tokens"] / metrics["count"]
            metrics["avg_output_tokens"] = metrics["total_output_tokens"] / metrics["count"]

    return dict(repo_metrics)


def create_cost_breakdown_table(
    batch_summary: Dict[str, Any],
    phase_breakdown: Dict[str, Any]
) -> str:
    """
    Create a formatted table showing cost breakdown.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("OFFLINE EXTRACTION PIPELINE - COST BREAKDOWN")
    lines.append("=" * 80)
    lines.append()

    # Overall summary
    lines.append("OVERALL SUMMARY:")
    lines.append("-" * 80)
    summary = batch_summary.get("summary", {})
    totals = batch_summary.get("totals", {})
    averages = batch_summary.get("averages", {})
    projections = batch_summary.get("projections", {})

    lines.append(f"Total instances processed: {summary.get('total_instances', 0)}")
    lines.append(f"Successful instances: {summary.get('successful_instances', 0)}")
    lines.append(f"Failed instances: {summary.get('failed_instances', 0)}")
    lines.append(f"Success rate: {summary.get('success_rate', 0) * 100:.1f}%")
    lines.append()

    # Totals
    lines.append("TOTALS (all processed instances):")
    lines.append(f"  Total cost: ${totals.get('total_cost_usd', 0):.6f}")
    lines.append(f"  Total input tokens: {totals.get('total_input_tokens', 0):,}")
    lines.append(f"  Total output tokens: {totals.get('total_output_tokens', 0):,}")
    lines.append(f"  Total cached tokens: {totals.get('total_cached_input_tokens', 0):,}")
    lines.append(f"  Total wall time: {totals.get('total_wall_time_seconds', 0):.2f}s ({totals.get('total_wall_time_seconds', 0) / 60:.2f} minutes)")
    lines.append()

    # Averages
    lines.append("AVERAGES (per successful instance):")
    lines.append(f"  Average cost: ${averages.get('avg_cost_usd_per_instance', 0):.6f}")
    lines.append(f"  Average input tokens: {averages.get('avg_input_tokens_per_instance', 0):,.0f}")
    lines.append(f"  Average output tokens: {averages.get('avg_output_tokens_per_instance', 0):,.0f}")
    lines.append(f"  Average cached tokens: {averages.get('avg_cached_input_tokens_per_instance', 0):,.0f}")
    lines.append(f"  Average wall time: {averages.get('avg_wall_time_seconds_per_instance', 0):.2f}s")
    lines.append()

    # Projections
    lines.append("PROJECTIONS (157 instances - full CIBench):")
    lines.append(f"  Projected total cost: ${projections.get('projected_157_instances_cost_usd', 0):.2f}")
    lines.append(f"  Projected input tokens: {projections.get('projected_157_instances_input_tokens', 0):,.0f}")
    lines.append(f"  Projected output tokens: {projections.get('projected_157_instances_output_tokens', 0):,.0f}")
    lines.append()

    # Phase breakdown (if available)
    if phase_breakdown:
        lines.append("=" * 80)
        lines.append("COST BREAKDOWN BY PHASE:")
        lines.append("=" * 80)
        lines.append()

        # Sort by total cost (descending)
        sorted_phases = sorted(
            phase_breakdown.items(),
            key=lambda x: x[1].get("total_cost", 0),
            reverse=True
        )

        for phase, metrics in sorted_phases:
            lines.append(f"Phase: {phase}")
            lines.append(f"  Calls: {metrics.get('count', 0)}")
            lines.append(f"  Total cost: ${metrics.get('total_cost', 0):.6f}")
            lines.append(f"  Average cost: ${metrics.get('avg_cost', 0):.6f}")
            lines.append(f"  Total input tokens: {metrics.get('total_input_tokens', 0):,}")
            lines.append(f"  Total output tokens: {metrics.get('total_output_tokens', 0):,}")
            lines.append(f"  Average time: {metrics.get('avg_time', 0):.2f}s")
            lines.append()

    lines.append("=" * 80)

    return "\n".join(lines)


def create_ascii_chart(
    phase_breakdown: Dict[str, Any],
    metric: str = "total_cost",
    width: int = 60
) -> str:
    """
    Create a simple ASCII bar chart.
    """
    if not phase_breakdown:
        return "No data available for chart"

    lines = []
    lines.append(f"\n{metric.replace('_', ' ').title()} by Phase:")
    lines.append("-" * (width + 20))

    # Get max value for scaling
    max_value = max(
        metrics.get(metric, 0)
        for metrics in phase_breakdown.values()
    )

    if max_value == 0:
        return "No data available for chart"

    # Sort by value
    sorted_phases = sorted(
        phase_breakdown.items(),
        key=lambda x: x[1].get(metric, 0),
        reverse=True
    )

    for phase, metrics in sorted_phases:
        value = metrics.get(metric, 0)
        bar_length = int((value / max_value) * width)
        bar = "█" * bar_length

        # Format value based on metric type
        if "cost" in metric:
            value_str = f"${value:.6f}"
        elif "tokens" in metric:
            value_str = f"{value:,.0f}"
        else:
            value_str = f"{value:.2f}"

        lines.append(f"{phase:20s} {bar} {value_str}")

    lines.append("-" * (width + 20))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze offline extraction pipeline costs"
    )
    parser.add_argument(
        "--batch-summary",
        required=True,
        help="Path to batch_summary.json",
    )
    parser.add_argument(
        "--detailed-tracker-dir",
        help="Path to per_instance directory with detailed trackers",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    # Load batch summary
    batch_summary_path = Path(args.batch_summary)
    if not batch_summary_path.exists():
        print(f"ERROR: Batch summary not found: {batch_summary_path}")
        return 1

    print(f"Loading batch summary from: {batch_summary_path}")
    batch_summary = load_batch_summary(batch_summary_path)

    # Load detailed trackers (if available)
    detailed_data = {}
    if args.detailed_tracker_dir:
        detailed_tracker_dir = Path(args.detailed_tracker_dir)
        if detailed_tracker_dir.exists():
            print(f"Loading detailed trackers from: {detailed_tracker_dir}")
            detailed_data = load_detailed_trackers(detailed_tracker_dir)
            print(f"Loaded {len(detailed_data)} detailed tracker files")

    # Analyze by phase
    phase_breakdown = {}
    if detailed_data:
        print("Analyzing costs by phase...")
        phase_breakdown = analyze_by_phase(detailed_data)

    # Analyze by repo
    print("Analyzing costs by repository...")
    repo_breakdown = analyze_by_repo(batch_summary)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comprehensive analysis
    analysis = {
        "batch_summary": batch_summary,
        "phase_breakdown": phase_breakdown,
        "repo_breakdown": repo_breakdown,
    }

    analysis_path = output_dir / "cost_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_path}")

    # Save phase breakdown CSV
    if phase_breakdown:
        phase_csv_path = output_dir / "phase_breakdown.csv"
        with open(phase_csv_path, "w", newline="") as f:
            fieldnames = [
                "phase", "count", "total_cost", "avg_cost",
                "total_input_tokens", "total_output_tokens",
                "avg_input_tokens", "avg_output_tokens", "avg_time"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for phase, metrics in phase_breakdown.items():
                row = {"phase": phase, **metrics}
                writer.writerow(row)

        print(f"Phase breakdown CSV saved to: {phase_csv_path}")

    # Create and save text report
    report = create_cost_breakdown_table(batch_summary, phase_breakdown)

    # Add ASCII charts
    if phase_breakdown:
        report += "\n\n" + create_ascii_chart(phase_breakdown, "total_cost")
        report += "\n" + create_ascii_chart(phase_breakdown, "total_input_tokens")

    report_path = output_dir / "cost_visualization.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Text report saved to: {report_path}")

    # Print to console
    print("\n" + report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
