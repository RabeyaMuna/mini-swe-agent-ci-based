#!/usr/bin/env python3
"""
batch_offline_metrics.py - Batch Process Instances for Offline Pipeline Metrics
===============================================================================

Processes multiple instances through the offline extraction pipeline and
collects comprehensive token/cost metrics for analysis.

This script:
1. Loads a sample of instances from different repositories
2. Runs the offline pipeline on each instance
3. Collects all metrics into aggregated reports
4. Calculates averages and totals

Usage:
    # Process 20 instances from different repos
    python scripts/batch_offline_metrics.py \
        --dataset data/cibench_test.json \
        --num-instances 20 \
        --model deepseek-v4-flash \
        --output-dir results/metrics/batch_offline_20

    # Process specific instances
    python scripts/batch_offline_metrics.py \
        --dataset data/cibench_test.json \
        --instance-ids 102,410,500,234,567 \
        --model deepseek-v4-flash \
        --output-dir results/metrics/batch_specific

Output:
    - batch_summary.json - Overall statistics across all instances
    - batch_costs.csv - Per-instance cost breakdown
    - per_instance/{instance_id}/ - Individual instance metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load the dataset file (supports both JSON and JSONL formats)."""
    # Check if it's a JSONL file (one JSON object per line)
    if dataset_path.suffix == '.jsonl':
        with open(dataset_path) as f:
            instances = [json.loads(line) for line in f if line.strip()]
        return instances

    # Otherwise treat as regular JSON
    with open(dataset_path) as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "instances" in data:
        return data["instances"]
    else:
        raise ValueError(f"Unknown dataset format in {dataset_path}")


def select_diverse_instances(
    instances: List[Dict[str, Any]],
    num_instances: int,
) -> List[Dict[str, Any]]:
    """
    Select diverse instances from different repositories.

    Ensures representation from multiple repos (e.g., flower, agno, etc.)
    """
    # Group by repository
    by_repo = {}
    for inst in instances:
        repo = inst.get("repo", "unknown")
        if repo not in by_repo:
            by_repo[repo] = []
        by_repo[repo].append(inst)

    # Select instances round-robin from different repos
    selected = []
    repo_names = list(by_repo.keys())
    idx = 0

    while len(selected) < num_instances and any(by_repo.values()):
        repo = repo_names[idx % len(repo_names)]
        if by_repo[repo]:
            selected.append(by_repo[repo].pop(0))
        idx += 1

    return selected[:num_instances]


def run_single_instance(
    instance: Dict[str, Any],
    model: str,
    output_dir: Path,
    dataset_path: Path,
    skip_l3: bool = False,
) -> Dict[str, Any]:
    """
    Run the offline pipeline on a single instance.

    Returns:
        Metrics summary for this instance
    """
    instance_id = str(instance.get("instance_id", instance.get("id", "unknown")))
    repo = instance.get("repo", "unknown")

    instance_output = output_dir / "per_instance" / instance_id
    instance_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Processing Instance {instance_id} ({repo})")
    print(f"{'=' * 80}")

    # Build command - use the new three-view pipeline script
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "offline_pipeline_with_three_views.py"),
        "--instance-id", instance_id,
        "--dataset", str(dataset_path),
        "--model", model,
        "--output-dir", str(instance_output),
    ]

    if skip_l3:
        cmd.append("--skip-l3")

    # Run the pipeline
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)

        # Load the metrics report
        report_path = instance_output / "cost_time_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            return report["instances"][0] if report["instances"] else {}
        else:
            return {"instance_id": instance_id, "status": "no_report"}

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Pipeline failed for instance {instance_id}")
        print(e.stderr)
        return {
            "instance_id": instance_id,
            "status": "failed",
            "error": str(e),
        }


def aggregate_metrics(
    instance_metrics: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate metrics across all instances.

    Returns:
        Summary statistics including totals and averages
    """
    total_instances = len(instance_metrics)
    successful = [m for m in instance_metrics if m.get("last_status") == "completed"]
    failed = [m for m in instance_metrics if m.get("last_status") != "completed"]

    # Calculate totals
    total_cost = sum(m.get("total_cost_usd", 0) for m in successful)
    total_input_tokens = sum(m.get("total_input_tokens", 0) for m in successful)
    total_output_tokens = sum(m.get("total_output_tokens", 0) for m in successful)
    total_cached_tokens = sum(m.get("total_cached_input_tokens", 0) for m in successful)
    total_wall_time = sum(m.get("total_wall_time_seconds", 0) for m in successful)
    total_api_time = sum(m.get("total_api_time_seconds", 0) for m in successful)

    # Calculate averages (only for successful instances)
    num_successful = len(successful)
    avg_cost = total_cost / num_successful if num_successful > 0 else 0
    avg_input_tokens = total_input_tokens / num_successful if num_successful > 0 else 0
    avg_output_tokens = total_output_tokens / num_successful if num_successful > 0 else 0
    avg_cached_tokens = total_cached_tokens / num_successful if num_successful > 0 else 0
    avg_wall_time = total_wall_time / num_successful if num_successful > 0 else 0
    avg_api_time = total_api_time / num_successful if num_successful > 0 else 0

    # Project to 157 instances (CIBench size)
    projected_157_cost = avg_cost * 157
    projected_157_input_tokens = avg_input_tokens * 157
    projected_157_output_tokens = avg_output_tokens * 157

    return {
        "summary": {
            "total_instances": total_instances,
            "successful_instances": num_successful,
            "failed_instances": len(failed),
            "success_rate": num_successful / total_instances if total_instances > 0 else 0,
        },
        "totals": {
            "total_cost_usd": round(total_cost, 8),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_input_tokens": total_cached_tokens,
            "total_wall_time_seconds": round(total_wall_time, 2),
            "total_api_time_seconds": round(total_api_time, 2),
        },
        "averages": {
            "avg_cost_usd_per_instance": round(avg_cost, 8),
            "avg_input_tokens_per_instance": round(avg_input_tokens, 2),
            "avg_output_tokens_per_instance": round(avg_output_tokens, 2),
            "avg_cached_input_tokens_per_instance": round(avg_cached_tokens, 2),
            "avg_wall_time_seconds_per_instance": round(avg_wall_time, 2),
            "avg_api_time_seconds_per_instance": round(avg_api_time, 2),
        },
        "projections": {
            "description": "Projected costs for 157 instances (full CIBench)",
            "projected_157_instances_cost_usd": round(projected_157_cost, 2),
            "projected_157_instances_input_tokens": round(projected_157_input_tokens, 0),
            "projected_157_instances_output_tokens": round(projected_157_output_tokens, 0),
        },
        "per_instance_details": instance_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch process instances for offline pipeline metrics collection"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset file (JSON)",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=20,
        help="Number of instances to process (default: 20)",
    )
    parser.add_argument(
        "--instance-ids",
        help="Comma-separated instance IDs to process (overrides num-instances)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model to use",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for batch results",
    )
    parser.add_argument(
        "--skip-l3",
        action="store_true",
        help="Skip L3 generation",
    )

    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        return 1

    print(f"Loading dataset from: {dataset_path}")
    instances = load_dataset(dataset_path)
    print(f"Loaded {len(instances)} total instances")

    # Select instances
    if args.instance_ids:
        # Use specific instance IDs
        filter_ids = set(args.instance_ids.split(","))
        selected_instances = [
            inst for inst in instances
            if str(inst.get("instance_id", inst.get("id", ""))) in filter_ids
        ]
    else:
        # Select diverse instances
        selected_instances = select_diverse_instances(instances, args.num_instances)

    print(f"Selected {len(selected_instances)} instances for processing")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each instance
    instance_metrics = []
    for i, instance in enumerate(selected_instances, 1):
        instance_id = str(instance.get("instance_id", instance.get("id", "unknown")))
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(selected_instances)}] Processing Instance {instance_id}")
        print(f"{'=' * 80}")

        metrics = run_single_instance(
            instance=instance,
            model=args.model,
            output_dir=output_dir,
            dataset_path=dataset_path,
            skip_l3=args.skip_l3,
        )
        instance_metrics.append(metrics)

    # Aggregate results
    print(f"\n{'=' * 80}")
    print("Aggregating Results")
    print(f"{'=' * 80}")

    aggregated = aggregate_metrics(instance_metrics)

    # Save batch summary
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Batch summary saved to: {summary_path}")

    # Save CSV for easy analysis
    import csv
    csv_path = output_dir / "batch_costs.csv"
    with open(csv_path, "w", newline="") as f:
        if instance_metrics:
            writer = csv.DictWriter(f, fieldnames=instance_metrics[0].keys())
            writer.writeheader()
            writer.writerows(instance_metrics)
    print(f"Batch costs CSV saved to: {csv_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total instances: {aggregated['summary']['total_instances']}")
    print(f"Successful: {aggregated['summary']['successful_instances']}")
    print(f"Failed: {aggregated['summary']['failed_instances']}")
    print()
    print("AVERAGES PER INSTANCE:")
    print(f"  Cost: ${aggregated['averages']['avg_cost_usd_per_instance']:.6f}")
    print(f"  Input tokens: {aggregated['averages']['avg_input_tokens_per_instance']:.0f}")
    print(f"  Output tokens: {aggregated['averages']['avg_output_tokens_per_instance']:.0f}")
    print(f"  Wall time: {aggregated['averages']['avg_wall_time_seconds_per_instance']:.2f}s")
    print()
    print("TOTALS:")
    print(f"  Total cost: ${aggregated['totals']['total_cost_usd']:.6f}")
    print(f"  Total input tokens: {aggregated['totals']['total_input_tokens']:,}")
    print(f"  Total output tokens: {aggregated['totals']['total_output_tokens']:,}")
    print()
    print("PROJECTIONS (157 instances):")
    print(f"  Projected cost: ${aggregated['projections']['projected_157_instances_cost_usd']:.2f}")
    print(f"  Projected input tokens: {aggregated['projections']['projected_157_instances_input_tokens']:,.0f}")
    print(f"  Projected output tokens: {aggregated['projections']['projected_157_instances_output_tokens']:,.0f}")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
