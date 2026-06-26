#!/usr/bin/env python3
"""
Evaluate predictions against ground truth from ci_dataset.jsonl

Metrics:
- Exact Match: prediction files exactly match ground truth files
- Top-1: all ground truth files are contained in prediction
- Top-3: same as Top-1 (only one prediction per instance)
- Precision: |intersection| / |prediction|
- Recall: |intersection| / |ground_truth|
- F1-Score: harmonic mean of precision and recall
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


def load_ground_truth_from_dataset(dataset_path: str) -> Dict[str, Set[str]]:
    """Load ground truth from ci_dataset.jsonl."""
    ground_truth = {}

    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            instance_id = str(data['id'])

            # Get changed_files as ground truth
            files = set()
            for file in data.get('changed_files', []):
                files.add(normalize_path(file))

            ground_truth[instance_id] = files

    return ground_truth


def load_predictions(pred_path: str) -> Dict[str, Dict]:
    """Load predictions from preds.json."""
    with open(pred_path, 'r') as f:
        return json.load(f)


def extract_files_from_diff(diff: str) -> Set[str]:
    """Extract file paths from a git diff."""
    files = set()
    for line in diff.split('\n'):
        if line.startswith('diff --git'):
            # Format: diff --git a/path b/path
            parts = line.split(' ')
            if len(parts) >= 3:
                # Get the b/ path (after modification)
                file_path = parts[-1]
                if file_path.startswith('b/'):
                    file_path = file_path[2:]  # Remove b/ prefix
                files.add(normalize_path(file_path))
    return files


def calculate_metrics_for_instance(gt_files: Set[str], pred_files: Set[str]) -> Dict:
    """Calculate metrics for a single instance."""
    intersection = gt_files.intersection(pred_files)

    # Exact match
    exact_match = gt_files == pred_files

    # Top-1: all ground truth files are in prediction
    top_1 = gt_files.issubset(pred_files) if pred_files else False

    # Precision: |intersection| / |prediction|
    precision = len(intersection) / len(pred_files) if len(pred_files) > 0 else 0.0

    # Recall: |intersection| / |ground_truth|
    recall = len(intersection) / len(gt_files) if len(gt_files) > 0 else 0.0

    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'exact_match': exact_match,
        'top_1': top_1,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'intersection_count': len(intersection),
        'gt_count': len(gt_files),
        'pred_count': len(pred_files)
    }


def evaluate(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Dict]) -> Dict:
    """Evaluate all predictions against ground truth."""

    results = {
        "per_instance": [],
        "aggregated": {
            "total_instances": 0,
            "exact_match_count": 0,
            "top_1_count": 0,
            "top_3_count": 0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "success_rate": 0.0
        }
    }

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    exact_match_count = 0
    top_1_count = 0

    for instance_id, pred_data in predictions.items():
        if instance_id not in ground_truth:
            print(f"Warning: Instance {instance_id} not found in ground truth")
            continue

        # Get ground truth files
        gt_files = ground_truth[instance_id]

        # Extract predicted files from diff
        pred_diff = pred_data.get('diff', '')
        pred_files = extract_files_from_diff(pred_diff)

        # Calculate metrics
        metrics = calculate_metrics_for_instance(gt_files, pred_files)

        # Store per-instance results
        instance_result = {
            "instance_id": instance_id,
            "sha_fail": pred_data.get("sha_fail", ""),
            "exact_match": metrics['exact_match'],
            "top_1": metrics['top_1'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "ground_truth_files": sorted(list(gt_files)),
            "predicted_files": sorted(list(pred_files)),
            "intersection": sorted(list(gt_files.intersection(pred_files))),
            "missing": sorted(list(gt_files - pred_files)),
            "extra": sorted(list(pred_files - gt_files)),
            "counts": {
                "ground_truth": len(gt_files),
                "predicted": len(pred_files),
                "correct": metrics['intersection_count']
            }
        }

        results["per_instance"].append(instance_result)

        # Update aggregated counts
        if metrics['exact_match']:
            exact_match_count += 1
        if metrics['top_1']:
            top_1_count += 1

        total_precision += metrics['precision']
        total_recall += metrics['recall']
        total_f1 += metrics['f1']

    # Calculate aggregated metrics
    total_instances = len(results["per_instance"])
    if total_instances > 0:
        results["aggregated"]["total_instances"] = total_instances
        results["aggregated"]["exact_match_count"] = exact_match_count
        results["aggregated"]["top_1_count"] = top_1_count
        results["aggregated"]["top_3_count"] = top_1_count  # Same as top-1 for this dataset
        results["aggregated"]["exact_match_rate"] = exact_match_count / total_instances
        results["aggregated"]["top_1_rate"] = top_1_count / total_instances
        results["aggregated"]["top_3_rate"] = top_1_count / total_instances
        results["aggregated"]["avg_precision"] = total_precision / total_instances
        results["aggregated"]["avg_recall"] = total_recall / total_instances
        results["aggregated"]["avg_f1"] = total_f1 / total_instances
        results["aggregated"]["success_rate"] = exact_match_count / total_instances

    return results


def print_results_table(results: Dict):
    """Print results in a clean table format."""
    agg = results["aggregated"]

    print("\n" + "="*100)
    print("EVALUATION METRICS - PREDICTIONS vs GROUND TRUTH")
    print("="*100)
    print(f"\nDataset: ci_dataset.jsonl")
    print(f"Total Test Instances: {agg['total_instances']}")
    print()

    # Main metrics table
    print("┌─────────────────────────┬─────────────┬──────────────┬─────────────┐")
    print("│ Condition               │ Count       │ Rate         │ Percentage  │")
    print("├─────────────────────────┼─────────────┼──────────────┼─────────────┤")
    print(f"│ Exact Match             │ {agg['exact_match_count']:>11} │ {agg['exact_match_count']}/{agg['total_instances']:<10} │ {agg['exact_match_rate']:>10.2%}  │")
    print(f"│ Top-1 (Recall=1)        │ {agg['top_1_count']:>11} │ {agg['top_1_count']}/{agg['total_instances']:<10} │ {agg['top_1_rate']:>10.2%}  │")
    print(f"│ Top-3                   │ {agg['top_3_count']:>11} │ {agg['top_3_count']}/{agg['total_instances']:<10} │ {agg['top_3_rate']:>10.2%}  │")
    print(f"│ Precision (avg)         │ {'-':>11} │ {'-':<12} │ {agg['avg_precision']:>10.2%}  │")
    print(f"│ Recall (avg)            │ {'-':>11} │ {'-':<12} │ {agg['avg_recall']:>10.2%}  │")
    print(f"│ F1-Score (avg)          │ {'-':>11} │ {'-':<12} │ {agg['avg_f1']:>10.2%}  │")
    print(f"│ Success Rate            │ {agg['exact_match_count']:>11} │ {agg['exact_match_count']}/{agg['total_instances']:<10} │ {agg['success_rate']:>10.2%}  │")
    print("└─────────────────────────┴─────────────┴──────────────┴─────────────┘")
    print()

    # Summary statistics
    print("Summary:")
    print(f"  OK Perfect matches (Exact Match): {agg['exact_match_count']}/{agg['total_instances']}")
    print(f"  OK Contains all GT files (Top-1): {agg['top_1_count']}/{agg['total_instances']}")
    print(f"  FAIL Failures: {agg['total_instances'] - agg['exact_match_count']}/{agg['total_instances']}")
    print("="*100)


def print_detailed_failures(results: Dict):
    """Print detailed analysis of failed instances."""
    failures = [inst for inst in results["per_instance"] if not inst["exact_match"]]

    if not failures:
        print("\nOK All instances have exact matches!")
        return

    print("\n" + "="*100)
    print(f"DETAILED FAILURE ANALYSIS ({len(failures)} instances)")
    print("="*100)

    for i, inst in enumerate(failures, 1):
        print(f"\n[{i}] Instance ID: {inst['instance_id']} | SHA: {inst['sha_fail'][:12]}...")
        print(f"    Metrics: Precision={inst['precision']:.1%}, Recall={inst['recall']:.1%}, F1={inst['f1']:.1%}, Top-1={'OK' if inst['top_1'] else 'FAIL'}")
        print(f"    Files: GT={inst['counts']['ground_truth']}, Pred={inst['counts']['predicted']}, Correct={inst['counts']['correct']}")

        print(f"\n    Ground Truth ({inst['counts']['ground_truth']} files):")
        for f in inst['ground_truth_files']:
            print(f"      - {f}")

        print(f"\n    Predicted ({inst['counts']['predicted']} files):")
        for f in inst['predicted_files']:
            status = "OK" if f in inst['intersection'] else "FAIL"
            print(f"      {status} {f}")

        if inst['missing']:
            print(f"\n    WARNING Missing from prediction ({len(inst['missing'])} files):")
            for f in inst['missing']:
                print(f"      - {f}")

        if inst['extra']:
            print(f"\n    WARNING Extra in prediction ({len(inst['extra'])} files):")
            for f in inst['extra']:
                print(f"      + {f}")

        print("-"*100)


def main():
    print("Loading data...")

    # Paths
    dataset_path = "/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/data/ci_dataset.jsonl"
    pred_path = "/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/results/preds.json"
    output_path = "/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/results/evaluation_results.json"

    # Load data
    ground_truth = load_ground_truth_from_dataset(dataset_path)
    predictions = load_predictions(pred_path)

    print(f"Ground Truth: {len(ground_truth)} instances")
    print(f"Predictions: {len(predictions)} instances")

    # Evaluate
    print("\nEvaluating predictions...")
    results = evaluate(ground_truth, predictions)

    # Print results
    print_results_table(results)
    print_detailed_failures(results)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nOK Detailed results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
