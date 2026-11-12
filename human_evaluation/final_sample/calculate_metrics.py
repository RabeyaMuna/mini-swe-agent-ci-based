#!/usr/bin/env python3
"""
Calculate precision, recall, and F1 for ReCAP attribution evaluation
Run AFTER manual annotation is complete
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict, Counter

# File paths
CSV_FILE = Path(__file__).parent / "human_eval_recap_attribution_FIXED.csv"
OUTPUT_DIR = Path(__file__).parent

print("="*80)
print("ReCAP ATTRIBUTION EVALUATION METRICS")
print("="*80)
print(f"Reading: {CSV_FILE}")

# Load annotations
df = pd.read_csv(CSV_FILE)

print(f"\nTotal rows: {len(df)}")
print(f"Total instances: {df['instance_id'].nunique()}")

# Filter annotated rows
annotated = df[df['overall_attribution_correct'].notna() & (df['overall_attribution_correct'] != '')]

print(f"Annotated rows: {len(annotated)}")

if len(annotated) == 0:
    print("\n⚠️  No annotations found!")
    print("Please fill in the 'overall_attribution_correct' column first.")
    print(f"\nExpected values: YES, NO, or PARTIAL")
    exit(1)

if len(annotated) < len(df):
    print(f"\n⚠️  Only {len(annotated)}/{len(df)} rows annotated ({len(annotated)/len(df)*100:.1f}%)")
    print("Calculating metrics on annotated rows only...\n")

# ============================================================================
# 1. ATTRIBUTION PRECISION
# ============================================================================
print("\n" + "="*80)
print("1. ATTRIBUTION PRECISION")
print("="*80)

# Count correct attributions
correct = len(annotated[annotated['overall_attribution_correct'].str.upper() == 'YES'])
partial = len(annotated[annotated['overall_attribution_correct'].str.upper() == 'PARTIAL'])
incorrect = len(annotated[annotated['overall_attribution_correct'].str.upper() == 'NO'])
total_attrs = len(annotated)

precision = correct / total_attrs if total_attrs > 0 else 0
precision_with_partial = (correct + 0.5 * partial) / total_attrs if total_attrs > 0 else 0

print(f"\nCorrect (YES): {correct}")
print(f"Partial: {partial}")
print(f"Incorrect (NO): {incorrect}")
print(f"Total annotated: {total_attrs}")
print(f"\n📊 Attribution Precision (strict): {precision:.2%}")
print(f"📊 Attribution Precision (with partial): {precision_with_partial:.2%}")

# Component breakdown
print("\n" + "-"*80)
print("Component-level accuracy (of annotated rows):")
print("-"*80)

for component, col in [
    ('Problem Description', 'correct_problem_description'),
    ('Root Cause', 'correct_root_cause'),
    ('Repair Attribution', 'correct_repair_attribution'),
    ('Files', 'correct_files'),
]:
    component_annotated = annotated[annotated[col].notna() & (annotated[col] != '')]
    if len(component_annotated) > 0:
        comp_correct = len(component_annotated[component_annotated[col].str.upper() == 'YES'])
        comp_partial = len(component_annotated[component_annotated[col].str.upper() == 'PARTIAL'])
        comp_acc = comp_correct / len(component_annotated)
        comp_acc_partial = (comp_correct + 0.5 * comp_partial) / len(component_annotated)
        print(f"  {component:25s}: {comp_acc:6.1%} (with partial: {comp_acc_partial:6.1%})")

# ============================================================================
# 2. BREAKDOWN BY REPO
# ============================================================================
print("\n" + "="*80)
print("2. PRECISION BY REPOSITORY")
print("="*80)

for repo_name in ['agno', 'flower', 'camel']:
    repo_rows = annotated[annotated['repo'].str.lower().str.contains(repo_name, na=False)]
    if len(repo_rows) > 0:
        repo_correct = len(repo_rows[repo_rows['overall_attribution_correct'].str.upper() == 'YES'])
        repo_partial = len(repo_rows[repo_rows['overall_attribution_correct'].str.upper() == 'PARTIAL'])
        repo_prec = repo_correct / len(repo_rows)
        repo_prec_partial = (repo_correct + 0.5 * repo_partial) / len(repo_rows)
        print(f"\n{repo_name.upper()}:")
        print(f"  Annotated rows: {len(repo_rows)}")
        print(f"  Correct: {repo_correct}, Partial: {repo_partial}")
        print(f"  Precision (strict): {repo_prec:.2%}")
        print(f"  Precision (with partial): {repo_prec_partial:.2%}")

# ============================================================================
# 3. FALSE POSITIVES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. FALSE POSITIVES (ReCAP Wrong)")
print("="*80)

false_positives = annotated[annotated['overall_attribution_correct'].str.upper() == 'NO']
print(f"\nTotal False Positives: {len(false_positives)} ({len(false_positives)/len(annotated)*100:.1f}%)")

if len(false_positives) > 0:
    print("\nMost common issues:")

    # Check human_problem_exists to see if problem is hallucinated
    fp_with_exists = false_positives[false_positives['human_problem_exists'].notna()]
    if len(fp_with_exists) > 0:
        hallucinated = len(fp_with_exists[fp_with_exists['human_problem_exists'].str.upper() == 'NO'])
        exists = len(fp_with_exists[fp_with_exists['human_problem_exists'].str.upper() == 'YES'])
        print(f"  - Hallucinated problems (problem doesn't exist): {hallucinated}")
        print(f"  - Problem exists but attribution wrong: {exists}")

# ============================================================================
# 4. FALSE NEGATIVES (From Notes)
# ============================================================================
print("\n" + "="*80)
print("4. FALSE NEGATIVES (ReCAP Missed)")
print("="*80)

# Check notes for mentions of missed problems
notes_with_content = annotated[annotated['notes'].notna() & (annotated['notes'] != '')]
missed_keywords = ['missed', 'missing', 'didn\'t find', 'not found', 'overlooked', 'failed to identify']

potential_fn = []
for idx, row in notes_with_content.iterrows():
    note = str(row['notes']).lower()
    if any(keyword in note for keyword in missed_keywords):
        potential_fn.append(row)

print(f"\nPotential False Negatives (from notes): {len(potential_fn)}")
if len(potential_fn) > 0:
    print("⚠️  Review 'notes' column for mentions of problems ReCAP missed")

# ============================================================================
# 5. RECALL ESTIMATION
# ============================================================================
print("\n" + "="*80)
print("5. RECALL ESTIMATION")
print("="*80)

# True recall requires knowing all ground truth problems
# We can estimate based on annotations where human_problem_exists = YES
problems_exist = annotated[annotated['human_problem_exists'].str.upper() == 'YES']
recall_denominator = len(problems_exist) + len(potential_fn)

if recall_denominator > 0:
    recall = len(problems_exist) / recall_denominator
    print(f"\nReCAP found problems: {len(problems_exist)}")
    print(f"Potentially missed: {len(potential_fn)}")
    print(f"📊 Estimated Recall: {recall:.2%}")
    print("\n⚠️  Note: Recall calculation is approximate.")
    print("   For precise recall, count ALL problems in each PR (even ones ReCAP missed)")
else:
    print("\n⚠️  Cannot estimate recall - need 'human_problem_exists' annotations")

# ============================================================================
# 6. F1 SCORE
# ============================================================================
if recall_denominator > 0:
    print("\n" + "="*80)
    print("6. F1 SCORE")
    print("="*80)

    if precision > 0 and recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"\n📊 F1 Score: {f1:.2%}")
        print(f"   (Based on Precision={precision:.2%}, Recall={recall:.2%})")
    else:
        print("\n⚠️  Cannot calculate F1 - precision or recall is 0")

# ============================================================================
# 7. SUMMARY FOR PAPER
# ============================================================================
print("\n" + "="*80)
print("📄 SUMMARY FOR PAPER")
print("="*80)

summary = f"""
Human Evaluation Results:

Sample: {df['instance_id'].nunique()} multi-problem instances (205 problem attributions)
- Agno: {len(df[df['repo'].str.contains('agno', case=False, na=False)]['instance_id'].unique())} instances
- Flower: {len(df[df['repo'].str.contains('flower', case=False, na=False)]['instance_id'].unique())} instances
- CAMEL: {len(df[df['repo'].str.contains('camel', case=False, na=False)]['instance_id'].unique())} instances

Annotation Progress: {len(annotated)}/{len(df)} rows ({len(annotated)/len(df)*100:.1f}%)

Key Metrics:
- Attribution Precision: {precision:.1%} (strict) / {precision_with_partial:.1%} (with partial credit)
- Estimated Recall: {recall:.1%} (based on {recall_denominator} problems)
- F1 Score: {f1:.1%} (if precision and recall available)

Attribution Breakdown:
- Correct: {correct} ({correct/total_attrs*100:.1f}%)
- Partial: {partial} ({partial/total_attrs*100:.1f}%)
- Incorrect: {incorrect} ({incorrect/total_attrs*100:.1f}%)

Component Accuracy (of annotated):
"""

for component, col in [
    ('Problem Description', 'correct_problem_description'),
    ('Root Cause', 'correct_root_cause'),
    ('Repair Attribution', 'correct_repair_attribution'),
    ('Files', 'correct_files'),
]:
    component_annotated = annotated[annotated[col].notna() & (annotated[col] != '')]
    if len(component_annotated) > 0:
        comp_correct = len(component_annotated[component_annotated[col].str.upper() == 'YES'])
        comp_acc = comp_correct / len(component_annotated)
        summary += f"- {component}: {comp_acc:.1%}\n"

print(summary)

# Save summary
summary_file = OUTPUT_DIR / "metrics_summary.txt"
with open(summary_file, 'w') as f:
    f.write(summary)
print(f"\n✅ Summary saved to: {summary_file}")

# Save detailed results as JSON
results = {
    "total_instances": int(df['instance_id'].nunique()),
    "total_rows": len(df),
    "annotated_rows": len(annotated),
    "annotation_progress": f"{len(annotated)/len(df)*100:.1f}%",
    "precision": {
        "strict": f"{precision:.2%}",
        "with_partial": f"{precision_with_partial:.2%}",
        "correct": int(correct),
        "partial": int(partial),
        "incorrect": int(incorrect),
    },
    "recall": {
        "estimated": f"{recall:.2%}" if recall_denominator > 0 else "N/A",
        "problems_found": int(len(problems_exist)) if recall_denominator > 0 else 0,
        "potentially_missed": len(potential_fn),
    },
    "f1_score": f"{f1:.2%}" if (recall_denominator > 0 and precision > 0 and recall > 0) else "N/A",
    "by_repo": {}
}

for repo_name in ['agno', 'flower', 'camel']:
    repo_rows = annotated[annotated['repo'].str.lower().str.contains(repo_name, na=False)]
    if len(repo_rows) > 0:
        repo_correct = len(repo_rows[repo_rows['overall_attribution_correct'].str.upper() == 'YES'])
        repo_prec = repo_correct / len(repo_rows)
        results["by_repo"][repo_name] = {
            "instances": int(df[df['repo'].str.contains(repo_name, case=False, na=False)]['instance_id'].nunique()),
            "annotated_rows": len(repo_rows),
            "precision": f"{repo_prec:.2%}",
        }

results_file = OUTPUT_DIR / "metrics_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Detailed results saved to: {results_file}")

print("\n" + "="*80)
print("✅ METRICS CALCULATION COMPLETE")
print("="*80)
print(f"\nFiles created:")
print(f"  1. {summary_file}")
print(f"  2. {results_file}")
print("\n🎯 Use these results in your paper!")
