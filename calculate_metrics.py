#!/usr/bin/env python3
"""
Calculate evaluation metrics for predictions against ground truth.

Metrics:
- Exact Match: prediction files exactly match ground truth files (order doesn't matter)
- Top-1: ground truth files are subset of top-1 prediction
- Top-3: ground truth files are subset of top-3 predictions
- Precision: |intersection| / |prediction|
- Success rate: percentage of instances with exact match
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def normalize_path(path: str) -> str:
    """Normalize file path by removing 'framework/' prefix if present."""
    path = path.strip()
    if path.startswith("framework/"):
        return path[len("framework/"):]
    return path


def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Load ground truth from eval_issues.json."""
    with open(gt_path, 'r') as f:
        data = json.load(f)

    ground_truth = {}
    for issue in data:
        issue_id = str(issue["id"])  # Convert to string to match predictions

        # Get changed_files as ground truth
        files = set()
        for file in issue.get("changed_files", []):
            files.add(normalize_path(file))

        ground_truth[issue_id] = files

    return ground_truth


def load_predictions(pred_path: str) -> Dict[str, Dict[str, any]]:
    """Load predictions from preds.json."""
    with open(pred_path, 'r') as f:
        return json.load(f)


def calculate_metrics(ground_truth: Dict[str, Set[str]],
                     predictions: Dict[str, Dict[str, any]]) -> Dict[str, any]:
    """Calculate all metrics."""

    results = {
        "per_instance": [],
        "aggregated": {
            "exact_match": 0,
            "top_1": 0,
            "top_3": 0,
            "avg_precision": 0.0,
            "success_rate": 0.0,
            "total_instances": 0
        }
    }

    exact_match_count = 0
    top_1_count = 0
    top_3_count = 0
    total_precision = 0.0
    total_instances = 0

    # Map prediction keys to ground truth keys
    # predictions use instance_id (which matches issue_id in ground truth)
    for instance_id, pred_data in predictions.items():
        sha_fail = pred_data.get("sha_fail", "")

        if instance_id not in ground_truth:
            print(f"Warning: instance {instance_id} not found in ground truth")
            continue

        gt_files = ground_truth[instance_id]
        pred_diff = pred_data.get("diff", "")

        # Extract files from diff
        pred_files = extract_files_from_diff(pred_diff)

        # Calculate metrics for this instance
        gt_set = gt_files
        pred_set = set(normalize_path(f) for f in pred_files)

        # Exact match
        is_exact_match = gt_set == pred_set

        # Top-1 (all predictions are top-1 in this case)
        is_top_1 = gt_set.issubset(pred_set) if pred_set else False

        # Top-3 (same as top-1 since we only have one prediction per instance)
        is_top_3 = is_top_1

        # Precision
        if len(pred_set) > 0:
            precision = len(gt_set.intersection(pred_set)) / len(pred_set)
        else:
            precision = 0.0

        # Store per-instance results
        results["per_instance"].append({
            "instance_id": instance_id,
            "sha_fail": sha_fail,
            "exact_match": is_exact_match,
            "top_1": is_top_1,
            "top_3": is_top_3,
            "precision": precision,
            "ground_truth_files": sorted(list(gt_set)),
            "predicted_files": sorted(list(pred_set)),
            "intersection": sorted(list(gt_set.intersection(pred_set))),
            "missing": sorted(list(gt_set - pred_set)),
            "extra": sorted(list(pred_set - gt_set))
        })

        # Update counts
        if is_exact_match:
            exact_match_count += 1
        if is_top_1:
            top_1_count += 1
        if is_top_3:
            top_3_count += 1
        total_precision += precision
        total_instances += 1

    # Calculate aggregated metrics
    if total_instances > 0:
        results["aggregated"]["exact_match"] = exact_match_count / total_instances
        results["aggregated"]["top_1"] = top_1_count / total_instances
        results["aggregated"]["top_3"] = top_3_count / total_instances
        results["aggregated"]["avg_precision"] = total_precision / total_instances
        results["aggregated"]["success_rate"] = exact_match_count / total_instances
        results["aggregated"]["total_instances"] = total_instances
        results["aggregated"]["exact_match_count"] = exact_match_count
        results["aggregated"]["top_1_count"] = top_1_count
        results["aggregated"]["top_3_count"] = top_3_count

    return results


def extract_files_from_diff(diff: str) -> List[str]:
    """Extract file paths from a git diff."""
    files = []
    for line in diff.split('\n'):
        if line.startswith('diff --git'):
            # Format: diff --git a/path b/path
            parts = line.split(' ')
            if len(parts) >= 3:
                # Get the b/ path (after modification)
                file_path = parts[-1]
                if file_path.startswith('b/'):
                    file_path = file_path[2:]  # Remove b/ prefix
                files.append(normalize_path(file_path))
    return files


def print_summary(results: Dict[str, any]):
    """Print summary table."""
    agg = results["aggregated"]

    print("\n" + "="*80)
    print("EVALUATION METRICS SUMMARY")
    print("="*80)
    print(f"\nTotal Instances: {agg['total_instances']}")
    print(f"\n{'Metric':<20} {'Count':<10} {'Rate':<10}")
    print("-"*80)
    print(f"{'Exact Match':<20} {agg['exact_match_count']:<10} {agg['exact_match']:.2%}")
    print(f"{'Top-1':<20} {agg['top_1_count']:<10} {agg['top_1']:.2%}")
    print(f"{'Top-3':<20} {agg['top_3_count']:<10} {agg['top_3']:.2%}")
    print(f"{'Avg Precision':<20} {'-':<10} {agg['avg_precision']:.2%}")
    print(f"{'Success Rate':<20} {agg['exact_match_count']:<10} {agg['success_rate']:.2%}")
    print("="*80)


def print_detailed_results(results: Dict[str, any], show_all: bool = False):
    """Print detailed per-instance results."""
    print("\n" + "="*80)
    print("DETAILED PER-INSTANCE RESULTS")
    print("="*80)

    for i, instance in enumerate(results["per_instance"], 1):
        # Only show failed instances by default
        if not show_all and instance["exact_match"]:
            continue

        print(f"\n[{i}] Instance ID: {instance['instance_id']}")
        print(f"    SHA: {instance['sha_fail']}")
        print(f"    Exact Match: {instance['exact_match']}")
        print(f"    Top-1: {instance['top_1']}")
        print(f"    Precision: {instance['precision']:.2%}")

        print(f"\n    Ground Truth Files ({len(instance['ground_truth_files'])}):")
        for f in instance['ground_truth_files']:
            print(f"      - {f}")

        print(f"\n    Predicted Files ({len(instance['predicted_files'])}):")
        for f in instance['predicted_files']:
            print(f"      - {f}")

        if instance['intersection']:
            print(f"\n    OK Correct ({len(instance['intersection'])}):")
            for f in instance['intersection']:
                print(f"      - {f}")

        if instance['missing']:
            print(f"\n    FAIL Missing ({len(instance['missing'])}):")
            for f in instance['missing']:
                print(f"      - {f}")

        if instance['extra']:
            print(f"\n    + Extra ({len(instance['extra'])}):")
            for f in instance['extra']:
                print(f"      - {f}")

        print("-"*80)


def main():
    # Paths
    pred_path = "/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/results/preds.json"
    gt_path = "/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/data/trs/eval_issues.json"
    output_path = "/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/results/evaluation_metrics.json"

    print("Loading data...")
    ground_truth = load_ground_truth(gt_path)
    predictions = load_predictions(pred_path)

    print(f"Ground Truth: {len(ground_truth)} instances")
    print(f"Predictions: {len(predictions)} instances")

    print("\nCalculating metrics...")
    results = calculate_metrics(ground_truth, predictions)

    # Print summary
    print_summary(results)

    # Print detailed results (only failures)
    print_detailed_results(results, show_all=False)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nOK Detailed results saved to: {output_path}")
    print("\nTo see all instances (including successes), set show_all=True in the script.")


if __name__ == "__main__":
    main()
