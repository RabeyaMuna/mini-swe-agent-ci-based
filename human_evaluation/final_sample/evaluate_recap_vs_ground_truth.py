#!/usr/bin/env python3
"""
BALANCED Pair-Based Evaluation: ReCAP vs Ground Truth
------------------------------------------------------
Balanced approach with proper type normalization:
  ✓ Normalize type variations (type_check = Type Checking)
  ✓ Keep reasonable compatibility (Config ↔ Dependency for causal chains)
  ✓ Moderate file overlap (≥0.65)
  ✓ Binary matching (no partial credit)
  ✓ Recognize that same file can fix multiple related problems

Based on user insights:
  - "type_check" vs "Type Checking" → SAME type, different spelling
  - Config ↔ Dependency → Related (package change → format change)
  - Same file in multiple problems → OK if problems are related
"""

import pandas as pd
import json
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based")
GT_FILE = BASE_DIR / "human_evaluation/final_sample/ground_truth_with_reasoning.csv"
RECAP_FILE = BASE_DIR / "human_evaluation/final_sample/recap_problem_repair_pairs.csv"
OUTPUT_DIR = BASE_DIR / "human_evaluation/final_sample"

print("="*80)
print("BALANCED PAIR-BASED EVALUATION")
print("="*80)
print("\nMethodology (BALANCED):")
print("  ✓ Proper type normalization (type_check = Type Checking)")
print("  ✓ Reasonable compatibility (Config ↔ Dependency ↔ Package)")
print("  ✓ Moderate file overlap: ≥65%")
print("  ✓ Binary matching (no partial credit)")

# Load data
df_gt = pd.read_csv(GT_FILE)
df_recap = pd.read_csv(RECAP_FILE)

print(f"\nData:")
print(f"  Ground Truth: {len(df_gt)} pairs")
print(f"  ReCAP Output: {len(df_recap)} pairs")

def normalize_type(type_str):
    """
    Normalize type variations to canonical form.

    Examples:
      - type_check, Type Checking, Type Error → Type Checking
      - lint, Linting, Code linting → Linting
      - format, Code Formatting, Formatting → Code Formatting
    """
    if pd.isna(type_str):
        return 'Unknown'

    t = str(type_str).strip().lower()

    # Type checking variations
    if 'type' in t and ('check' in t or 'error' in t):
        return 'Type Checking'

    # Linting variations
    if 'lint' in t:
        return 'Linting'

    # Formatting variations
    if 'format' in t:
        if 'doc' in t or 'rst' in t:
            return 'Documentation Formatting'
        return 'Code Formatting'

    # Test variations
    if 'test' in t and ('fail' in t or 'error' in t):
        return 'Test Failure'
    if t == 'test' or t == 'testing':
        return 'Test Failure'

    # Assertion
    if 'assert' in t:
        return 'Assertion Error'

    # Config variations
    if 'config' in t or 'ci_config' in t:
        return 'Configuration Error'

    # Dependency/Install/Package variations
    if any(x in t for x in ['depend', 'package', 'install']):
        return 'Dependency Issues'

    # Documentation
    if 'doc' in t or 'docs' in t:
        return 'Doc/Docstring'

    # Runtime
    if 'runtime' in t:
        return 'Runtime Error'

    # Import
    if 'import' in t:
        return 'Import Error'

    # Build
    if 'build' in t:
        return 'Build Error'

    return type_str

# Compatibility groups - these types can match each other
COMPATIBLE_GROUPS = [
    # Config ↔ Dependency ↔ Package (user's insight: causal chains!)
    {'Configuration Error', 'Dependency Issues'},

    # Format variations
    {'Code Formatting', 'Documentation Formatting'},

    # Test variations
    {'Test Failure', 'Assertion Error'},
]

def types_compatible(type1, type2):
    """Check if two types are compatible (same or in same group)."""
    if not type1 or not type2:
        return False

    t1 = normalize_type(type1)
    t2 = normalize_type(type2)

    # Exact match after normalization
    if t1 == t2:
        return True

    # Check compatibility groups
    for group in COMPATIBLE_GROUPS:
        if t1 in group and t2 in group:
            return True

    return False

def calculate_overlap(recap_files, gt_files):
    """Calculate file overlap - return best of Jaccard/Containment."""
    if pd.isna(recap_files) or pd.isna(gt_files):
        return 0.0

    recap_set = set(str(recap_files).split(','))
    gt_set = set(str(gt_files).split(','))

    recap_set = {f.strip() for f in recap_set if f.strip() and f.strip() != 'nan'}
    gt_set = {f.strip() for f in gt_set if f.strip() and f.strip() != 'nan'}

    if not recap_set or not gt_set:
        return 0.0

    intersection = len(recap_set & gt_set)

    # Multiple metrics, take best
    jaccard = intersection / len(recap_set | gt_set) if recap_set | gt_set else 0.0
    containment_r = intersection / len(recap_set) if recap_set else 0.0
    containment_g = intersection / len(gt_set) if gt_set else 0.0

    return max(jaccard, containment_r, containment_g)

def pair_matches(recap_row, gt_row):
    """
    BALANCED matching: both problem AND repair must be correct.
    Returns: (matches: bool, overlap: float, match_reason: str)
    """
    # 1. Type check with normalization
    if not types_compatible(recap_row['recap_failure_type'], gt_row['problem_type']):
        return False, 0.0, 'type_mismatch'

    # 2. Moderate file overlap threshold (65%)
    overlap = calculate_overlap(recap_row['recap_files'], gt_row['related_files'])

    # BALANCED: ≥65% overlap required
    if overlap >= 0.65:
        return True, overlap, 'match'
    else:
        return False, overlap, 'low_overlap'

# Evaluation
print("\n" + "="*80)
print("Matching Pairs (BALANCED)")
print("="*80)

true_positives = []
false_positives = []
matched_gt_ids = set()

for _, recap_row in df_recap.iterrows():
    inst_id = recap_row['instance_id']
    inst_id_int = int(inst_id) if isinstance(inst_id, str) else inst_id
    gt_inst = df_gt[df_gt['instance_id'] == inst_id_int]

    best_match = None
    best_score = 0.0
    best_reason = None

    for _, gt_row in gt_inst.iterrows():
        matches, overlap, reason = pair_matches(recap_row, gt_row)

        if matches and overlap > best_score:
            best_match = gt_row
            best_score = overlap
            best_reason = reason

    if best_match is not None:
        # TRUE POSITIVE
        true_positives.append({
            'instance_id': inst_id,
            'recap_id': recap_row['recap_problem_id'],
            'gt_id': best_match['problem_id'],
            'overlap': best_score,
            'recap_type': normalize_type(recap_row['recap_failure_type']),
            'gt_type': normalize_type(best_match['problem_type'])
        })
        matched_gt_ids.add((inst_id_int, best_match['problem_id']))
    else:
        # FALSE POSITIVE
        false_positives.append({
            'instance_id': inst_id,
            'recap_id': recap_row['recap_problem_id'],
            'type': normalize_type(recap_row['recap_failure_type'])
        })

# FALSE NEGATIVES
false_negatives = []
for _, gt_row in df_gt.iterrows():
    inst_id = gt_row['instance_id']
    gt_id = gt_row['problem_id']

    if (inst_id, gt_id) not in matched_gt_ids:
        false_negatives.append({
            'instance_id': inst_id,
            'gt_id': gt_id,
            'type': normalize_type(gt_row['problem_type'])
        })

# Calculate metrics (BINARY - no partial credit)
tp = len(true_positives)
fp = len(false_positives)
fn = len(false_negatives)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / len(df_gt) if len(df_gt) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

# Results
print(f"\n{'='*80}")
print("FINAL RESULTS (BALANCED)")
print(f"{'='*80}")

print(f"\nConfusion Matrix:")
print(f"  True Positives:   {tp:>3}")
print(f"  False Positives:  {fp:>3}")
print(f"  False Negatives:  {fn:>3}")

print(f"\nMetrics:")
print(f"  Precision: {precision:.1%}  ({tp} / {tp + fp})")
print(f"  Recall:    {recall:.1%}  ({tp} / {len(df_gt)})")
print(f"  F1 Score:  {f1:.3f}")

print("\n" + "="*80)
print("COMPARISON: All Three Approaches")
print("="*80)

print("""
Approach        Precision  Recall   F1     Method
───────────────────────────────────────────────────────────────────
Lenient           74.6%    71.9%  0.732   Broad compat, 50% overlap, partial
BALANCED          {:.1%}   {:.1%}  {:.3f}   Good normalization, 65% overlap, binary
STRICT            52.2%    58.1%  0.550   Exact types only, 70% overlap, binary
""".format(precision, recall, f1))

print("\nBalanced Evaluation Details:")
print(f"  ✓ Type normalization: type_check → Type Checking")
print(f"  ✓ Compatibility: Config ↔ Dependency (causal chains)")
print(f"  ✓ File overlap: ≥65% required")
print(f"  ✓ No partial credit (binary matching)")
print(f"  ✓ Reasonable and defensible")

# Type matching examples
print("\n" + "="*80)
print("Type Normalization Examples")
print("="*80)

example_types = [
    ('type_check', 'Type Checking', 'Type Error'),
    ('lint', 'Linting', 'Code linting'),
    ('format', 'Code Formatting', 'Formatting'),
    ('test', 'Test Failure', 'Testing'),
    ('install', 'Dependency Issues', 'Package Install Error'),
]

for types in example_types:
    normalized = [normalize_type(t) for t in types]
    print(f"  {types[0]:20s} → {normalized[0]}")
    for t, n in zip(types[1:], normalized[1:]):
        print(f"  {t:20s} → {n} {'✓ SAME' if n == normalized[0] else '✗ DIFFERENT'}")
    print()

# Save
results = {
    'methodology': 'BALANCED pair-based (normalized types, reasonable compat, 65% overlap, binary)',
    'thresholds': {
        'file_overlap_minimum': 0.65,
        'type_matching': 'normalized with compatibility groups',
        'partial_credit': False
    },
    'type_normalization': {
        'type_checking': ['type_check', 'Type Checking', 'Type Error'],
        'linting': ['lint', 'Linting', 'Code linting'],
        'formatting': ['format', 'Code Formatting', 'Formatting'],
        'testing': ['test', 'Test Failure', 'Testing'],
        'dependency': ['install', 'Dependency Issues', 'Package Install Error', 'dependency']
    },
    'compatibility_groups': [
        ['Configuration Error', 'Dependency Issues'],
        ['Code Formatting', 'Documentation Formatting'],
        ['Test Failure', 'Assertion Error']
    ],
    'confusion_matrix': {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    },
    'metrics': {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    },
    'dataset': {
        'gt_pairs': len(df_gt),
        'recap_pairs': len(df_recap)
    }
}

output_file = OUTPUT_DIR / "BALANCED_pair_evaluation.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved: {output_file}")

print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}")
print("""
Use BALANCED evaluation for reviewer:
  ✓ Proper type normalization (addresses spelling variations)
  ✓ Recognizes causal relationships (Config ↔ Dependency)
  ✓ Moderate thresholds (not too strict, not too lenient)
  ✓ Defensible and reasonable
  ✓ Aligns with real-world problem relationships
""")
