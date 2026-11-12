"""
Analyze False Positives from Full ReCAP
Classify FPs as: incorrect, hidden problems, or unrelated edits
"""

import json
from pathlib import Path
from collections import defaultdict

# Load results
results_dir = Path("/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based/results/02_human_attribution_final")

with open(results_dir / "detailed_results.json", 'r') as f:
    results = json.load(f)

# Get Full ReCAP results
recap_results = results['bidirectional']['per_instance_results']

# Analyze FPs
print("FALSE POSITIVE ANALYSIS - Full ReCAP")
print("="*80)

total_fp = 0
for instance in recap_results:
    instance_id = instance['instance_id']
    fp_count = instance['false_positives']

    if fp_count > 0:
        total_fp += fp_count
        print(f"\nInstance {instance_id}: {fp_count} FPs")
        print(f"  GT associations: {instance['gt_associations']}")
        print(f"  Predicted: {instance['pred_associations']}")
        print(f"  TP: {instance['true_positives']}")

print(f"\n{'='*80}")
print(f"Total False Positives: {total_fp}")
print(f"\nNext step: Manually examine these to classify as:")
print("  1. Incorrect associations")
print("  2. Valid hidden problems (missing from ground truth)")
print("  3. Unrelated edits (README, formatting, etc.)")
