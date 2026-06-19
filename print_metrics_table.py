#!/usr/bin/env python3
"""Print metrics in a clean table format."""

import json

# Load results
with open('/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/results/evaluation_metrics.json') as f:
    results = json.load(f)

agg = results["aggregated"]

print("\n" + "="*80)
print("EVALUATION METRICS RESULTS")
print("="*80)
print()
print(f"Total Test Instances: {agg['total_instances']}")
print()
print("┌─────────────────────────┬─────────────┬──────────────┬─────────────┐")
print("│ Condition               │ Count       │ Rate         │ Percentage  │")
print("├─────────────────────────┼─────────────┼──────────────┼─────────────┤")
print(f"│ Exact Match             │ {agg['exact_match_count']:>11} │ {agg['exact_match_count']}/{agg['total_instances']:>11} │ {agg['exact_match']:>10.2%}  │")
print(f"│ Top-1                   │ {agg['top_1_count']:>11} │ {agg['top_1_count']}/{agg['total_instances']:>11} │ {agg['top_1']:>10.2%}  │")
print(f"│ Top-3                   │ {agg['top_3_count']:>11} │ {agg['top_3_count']}/{agg['total_instances']:>11} │ {agg['top_3']:>10.2%}  │")
print(f"│ Precision (avg)         │ {'-':>11} │ {'-':>12} │ {agg['avg_precision']:>10.2%}  │")
print(f"│ Success Rate            │ {agg['exact_match_count']:>11} │ {agg['exact_match_count']}/{agg['total_instances']:>11} │ {agg['success_rate']:>10.2%}  │")
print("└─────────────────────────┴─────────────┴──────────────┴─────────────┘")
print()

# Calculate some additional stats
print("Additional Statistics:")
print(f"  - Perfect matches (Exact Match): {agg['exact_match_count']}/{agg['total_instances']}")
print(f"  - Contains all ground truth files (Top-1): {agg['top_1_count']}/{agg['total_instances']}")
print(f"  - Failures: {agg['total_instances'] - agg['exact_match_count']}/{agg['total_instances']}")
print()

# Show failure details
failures = [inst for inst in results["per_instance"] if not inst["exact_match"]]
print(f"Failed Instances ({len(failures)}):")
for inst in failures:
    precision = inst['precision']
    gt_count = len(inst['ground_truth_files'])
    pred_count = len(inst['predicted_files'])
    correct_count = len(inst['intersection'])
    
    print(f"\n  Instance {inst['instance_id']}:")
    print(f"    - Ground Truth: {gt_count} file(s)")
    print(f"    - Predicted: {pred_count} file(s)")
    print(f"    - Correct: {correct_count} file(s)")
    print(f"    - Precision: {precision:.1%}")
    print(f"    - Top-1: {'✓' if inst['top_1'] else '✗'}")

print()
print("="*80)
