# Attribution Evaluation - Quick Reference

## Core Concepts

### Problem-Hunk Pair
```
(CI Problem, Code Hunk)
  ↓
"test_login fails" × "auth/session.py:45-47"
```

### Association Labels
```
DIRECT     → Hunk directly fixes this problem
SHARED     → Hunk contributes (multi-problem)
UNRELATED  → Hunk doesn't address this
UNCERTAIN  → Can't determine (excluded from metrics)
```

## Metrics Formulas

### Definitions
```
G_pos = Ground truth positive pairs (DIRECT ∪ SHARED, excluding UNCERTAIN)
R_pos = ReCAP predicted positive pairs

TP = |R_pos ∩ G_pos|  # Correct predictions
FP = |R_pos - G_pos|  # Incorrect predictions  
FN = |G_pos - R_pos|  # Missed associations
```

### Calculations
```
Precision = TP / (TP + FP)
  "Of what ReCAP predicted, what % was correct?"

Recall = TP / (TP + FN)  
  "Of what should be predicted, what % did ReCAP find?"

F1 = 2 × Precision × Recall / (Precision + Recall)
  "Harmonic mean of precision and recall"
```

### Micro vs Macro
```
Micro-averaged:
  - Pool all pairs across all episodes
  - Calculate metrics once
  - Weights episodes by size
  - P_micro = Σ(TP) / (Σ(TP) + Σ(FP))

Macro-averaged:
  - Calculate metrics per episode
  - Average across episodes
  - All episodes weighted equally
  - P_macro = mean([P_ep1, P_ep2, ...])
```

## Algorithm (Condensed)

```
For each of N=50 stratified episodes:
  
  1. Extract P problems, U hunks
     → C = P × U possible pairs
  
  2. LLM annotates C
     (NO access to ReCAP predictions)
  
  3. Human verifies ALL pairs in C
     → Labels: direct/shared/unrelated/uncertain
     → G_pos = direct ∪ shared pairs
  
  4. Map ReCAP predictions to canonical
     → R_pos
  
  5. Calculate:
     TP += |R_pos ∩ G_pos|
     FP += |R_pos - G_pos|
     FN += |G_pos - R_pos|

Final:
  P = Σ(TP) / (Σ(TP) + Σ(FP))
  R = Σ(TP) / (Σ(TP) + Σ(FN))
  F1 = 2PR / (P + R)
```

## Example Walkthrough

### Input
```
Episode: repo/commit_abc

Problems:
  P1: test_timeout fails
  P2: test_memory fails

Hunks:
  H1: timeout.py:10-15 (increase timeout)
  H2: cache.py:20-25 (fix memory leak)
  H3: test.py:30-35 (add test)

All pairs: P1×H1, P1×H2, P1×H3, P2×H1, P2×H2, P2×H3
```

### Ground Truth (Human)
```
P1-H1: DIRECT     ✓ Timeout fix for timeout test
P1-H3: DIRECT     ✓ Test for timeout
P2-H2: DIRECT     ✓ Memory fix for memory test
P1-H2: SHARED     ✓ Memory fix also helps timeout
P2-H1: UNRELATED  ✗
P2-H3: UNRELATED  ✗

G_pos = {P1-H1, P1-H3, P2-H2, P1-H2}  # 4 positive
```

### ReCAP Predictions
```
R_pos = {P1-H1, P1-H3, P2-H2}  # 3 predicted
```

### Calculation
```
TP = |{P1-H1, P1-H3, P2-H2} ∩ {P1-H1, P1-H3, P2-H2, P1-H2}|
   = |{P1-H1, P1-H3, P2-H2}|
   = 3

FP = |{P1-H1, P1-H3, P2-H2} - {P1-H1, P1-H3, P2-H2, P1-H2}|
   = |{}|
   = 0

FN = |{P1-H1, P1-H3, P2-H2, P1-H2} - {P1-H1, P1-H3, P2-H2}|
   = |{P1-H2}|
   = 1

Precision = 3 / (3 + 0) = 100%
Recall    = 3 / (3 + 1) = 75%
F1        = 2 × 1.0 × 0.75 / 1.75 = 85.7%
```

## Stratification Target

```
N = 50 episodes

Problem Count:
  - Single: 20 (40%)
  - Multi:  30 (60%)

Patch Size:
  - Small (<50 lines):    15 (30%)
  - Medium (50-200):      25 (50%)
  - Large (200+):         10 (20%)

Failure Types:
  - Test failures: 25 (50%)
  - Build errors:  15 (30%)
  - Lint/other:    10 (20%)

Repositories:
  - Diverse set (>10 repos)
```

## Annotation Decision Tree

```
For pair (Problem P, Hunk H):

1. Does H change code that directly causes P?
   YES → DIRECT
   NO  → Continue

2. Does H fix a root cause shared by P and other problems?
   YES → SHARED
   NO  → Continue

3. Is H completely unrelated to P?
   YES → UNRELATED
   NO  → Continue

4. Can't determine from available evidence?
   YES → UNCERTAIN
```

## LLM Prompt Template

```
Problem: {problem.description}
  Type: {problem.failure_type}
  Test: {problem.test_name}
  Error: {problem.error_message}

Hunk: {hunk.file_path}:{hunk.lines}
  Diff:
  {hunk.diff}

Context:
  - CI Logs: {excerpt}
  - Commits: {history}

Does this hunk fix this problem?
  - DIRECT: Yes, directly
  - SHARED: Yes, among others
  - UNRELATED: No
  - UNCERTAIN: Can't tell

Respond JSON:
{
  "association_type": "...",
  "confidence": 0.0-1.0,
  "explanation": "..."
}
```

## Common Pitfalls

### ❌ Don't
- Show ReCAP predictions to LLM
- Modify predictions based on ground truth
- Include uncertain pairs in metrics
- Forget to normalize hunk IDs
- Assume one hunk = one problem

### ✓ Do
- Keep LLM annotation independent
- Map predictions to canonical hunks only
- Exclude uncertain from primary metrics
- Handle multi-problem scenarios
- Review ALL possible pairs

## Output Format

### Results Table
```
Configuration        | N  | Precision | Recall | F1
---------------------|----|-----------| -------|------
Overall              | 50 | 78.5%     | 82.3%  | 80.4%
Single-problem       | 20 | 85.2%     | 88.1%  | 86.6%
Multi-problem        | 30 | 73.8%     | 78.5%  | 76.1%
```

### Error Breakdown
```
False Positives (FP=342):
  - Over-attributed shared:  130 (38%)
  - Included refactoring:     86 (25%)
  - Added defensive code:     62 (18%)

False Negatives (FN=268):
  - Missed test additions:   112 (42%)
  - Missed config changes:    75 (28%)
  - Subtle logic fixes:       40 (15%)
```

## Paper Text Template

```
We evaluated attribution quality on 50 stratified historical 
repair episodes through LLM-assisted, human-verified annotation 
at the problem-hunk level. Each episode's CI problems and code 
hunks were extracted independently, and an LLM (without access 
to ReCAP predictions) proposed candidate associations. Human 
annotators reviewed all possible problem-hunk pairs, correcting 
LLM errors and labeling each as direct, shared, unrelated, or 
uncertain. Uncertain pairs (N={count}) were excluded from primary 
evaluation.

Full ReCAP achieved {P}% precision, {R}% recall, and {F1}% F1 
(micro-averaged). For single-problem episodes (N={n1}), precision 
was {P1}% and recall {R1}%. For multi-problem episodes (N={n2}), 
precision was {P2}% and recall {R2}%, reflecting the challenge 
of attributing shared changes.

Common false positives included {top FP category} ({%}%), while 
false negatives were dominated by {top FN category} ({%}%).
```

## File Checklist

```
✓ recap_predictions.json      # Your system output
✓ stratified_sample.json       # 50 episodes selected
✓ llm_annotations/*.json       # LLM proposals per episode
✓ human_annotations/*.json     # Verified ground truth
✓ results.json                 # Final metrics
```

## Commands

```bash
# Run complete pipeline
python annotation_pipeline.py \
  --recap-output data.json \
  --n-episodes 50

# Calculate metrics only (if annotations exist)
python -c "
from annotation_pipeline import AnnotationPipeline
p = AnnotationPipeline()
p.step5_evaluate(episodes)
"

# Generate report
python -c "
from problem_hunk_evaluation import save_results
save_results(result, 'report.json')
"
```

## Inter-Annotator Agreement

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(
    annotator1_labels,
    annotator2_labels
)

# Interpretation:
# > 0.8: Excellent
# 0.6-0.8: Good
# 0.4-0.6: Moderate
# < 0.4: Poor (refine guidelines)
```
