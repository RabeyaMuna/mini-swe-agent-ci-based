# Attribution Precision & Recall Evaluation Workflow

## Overview

This document describes the complete process for evaluating your PR generation system's attribution quality using **LLM-assisted + Human verification**.

## Methodology

### 1. Definitions

**Attribution**: Which specific code locations (files, lines) the system decides to modify

**Ground Truth**: The "correct" set of changes that should be made for a given issue

**Precision**: Of all changes the system made, what percentage were correct?
```
Precision = TP / (TP + FP)
         = Correct changes / Total changes made by system
```

**Recall**: Of all changes that should be made, what percentage did the system find?
```
Recall = TP / (TP + FN)
       = Correct changes / Total changes needed
```

Where:
- **TP (True Positive)**: System made the change, and it WAS needed ✅
- **FP (False Positive)**: System made the change, but it was NOT needed ❌
- **FN (False Negative)**: System DIDN'T make the change, but it WAS needed 🔴

### 2. Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Prepare Test Set                                    │
│ - Select 20-50 representative PRs                           │
│ - Include diverse issue types                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Run Your System                                     │
│ - Generate PRs for each issue                               │
│ - Save system outputs (diffs)                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: LLM Creates Initial Annotations                     │
│ - LLM analyzes each PR                                      │
│ - Identifies TP, FP, FN candidates                          │
│ - Provides reasoning                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Human Verification                                  │
│ - Humans review LLM annotations                             │
│ - Correct any mistakes                                      │
│ - Add missed changes                                        │
│ - Mark confidence                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Calculate Metrics                                   │
│ - Run evaluation script                                     │
│ - Generate precision/recall/F1                              │
│ - Create detailed report                                    │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Steps

### Step 1: Prepare Test Set

Sample at least **20-50 PRs** with diversity:

**Issue Type Distribution:**
- Bug fixes: 40%
- New features: 30%
- Refactoring: 20%
- Documentation/Tests: 10%

**Complexity Distribution:**
- Simple (1-2 files): 30%
- Medium (3-5 files): 50%
- Complex (6+ files): 20%

**Store test data:**
```json
{
  "pr_id": "PR-123",
  "issue_description": "Fix authentication timeout bug",
  "repository": "owner/repo",
  "issue_url": "https://github.com/...",
  "metadata": {
    "issue_type": "bug",
    "complexity": "medium",
    "files_modified": 3
  }
}
```

### Step 2: Generate System Outputs

Run your system on each issue and save:
```json
{
  "pr_id": "PR-123",
  "system_diff": "unified diff output...",
  "system_changes": [
    {
      "file_path": "auth/session.py",
      "start_line": 45,
      "end_line": 47,
      "change_type": "modify",
      "content": "Increase timeout from 30s to 60s"
    }
  ]
}
```

### Step 3: LLM Initial Annotation

Use the LLM to create first-pass annotations:

**Prompt Template** (see `create_llm_annotation_prompt()` in the code):

```
You are evaluating a code change attribution system.

Issue: [description]
System Changes: [diff]

Identify:
1. True Positives: Correct and necessary changes
2. False Positives: Incorrect or unnecessary changes  
3. False Negatives: Needed changes the system missed

For each change, specify: file, lines, type, reasoning
```

**LLM Output:**
```json
{
  "true_positives": [
    {
      "file_path": "auth/session.py",
      "start_line": 45,
      "end_line": 47,
      "change_type": "modify",
      "content": "Timeout increase",
      "reasoning": "Directly addresses the timeout issue"
    }
  ],
  "false_positives": [
    {
      "file_path": "utils/logger.py",
      "start_line": 10,
      "end_line": 12,
      "change_type": "modify",
      "content": "Added debug logging",
      "reasoning": "Not required for the fix, adds unnecessary overhead"
    }
  ],
  "false_negatives": [
    {
      "file_path": "tests/test_auth.py",
      "start_line": 100,
      "end_line": 110,
      "change_type": "add",
      "content": "Test case for timeout scenario",
      "reasoning": "System should have added test coverage"
    }
  ],
  "confidence": 0.85,
  "notes": "Uncertain if the logging is truly unnecessary"
}
```

### Step 4: Human Verification

Create a review interface (web app or CLI) where humans can:

**Review Tasks:**
1. **Verify TPs**: Are these changes really correct?
2. **Verify FPs**: Are these really unnecessary?
3. **Verify FNs**: Were these changes actually missing?
4. **Add missed FNs**: What else should have been changed?

**Annotation Interface Example:**
```
PR-123: Fix authentication timeout bug
─────────────────────────────────────────────────────────────

SYSTEM CHANGE #1: auth/session.py:45-47 (modify)
  Code: timeout = 60  # Changed from 30
  
  LLM Says: TRUE POSITIVE
  Reasoning: "Directly addresses the timeout issue"
  
  → Your Verdict: [TP] [FP] [FN]
  → Notes: ___________________________________________
  
─────────────────────────────────────────────────────────────

SYSTEM CHANGE #2: utils/logger.py:10-12 (modify)
  Code: logger.debug("Session timeout check")
  
  LLM Says: FALSE POSITIVE
  Reasoning: "Not required, adds overhead"
  
  → Your Verdict: [TP] [FP] [FN]
  → Notes: ___________________________________________

─────────────────────────────────────────────────────────────

MISSING CHANGES (False Negatives):
  LLM identified: tests/test_auth.py:100-110 (add test)
  
  Are there other changes that should have been made?
  → Add: [File] [Lines] [Description]
```

**Best Practices:**
- Have 2-3 annotators per PR for reliability
- Calculate inter-annotator agreement (Cohen's Kappa)
- Resolve disagreements through discussion
- Focus verification on high-uncertainty cases (LLM confidence < 0.7)

### Step 5: Calculate Metrics

Run the evaluation script:

```bash
python attribution_evaluation.py
```

**Output:**
```
=== Aggregate Results (50 PRs) ===

Micro-averaged (all changes equally weighted):
  Precision: 78.5%
  Recall:    82.3%
  F1 Score:  80.4%

Macro-averaged (all PRs equally weighted):
  Precision: 76.2%
  Recall:    81.7%
  F1 Score:  78.9%

Total Changes:
  True Positives:  1,245
  False Positives:   342
  False Negatives:   268

Per-PR Breakdown:
  PR-001: P=85%, R=90%, F1=87.4%
  PR-002: P=72%, R=78%, F1=74.9%
  ...
```

## Efficiency Tips

### LLM-First Strategy

1. **High-confidence LLM annotations** (confidence > 0.9): Accept without human review (~30% of cases)
2. **Medium-confidence** (0.7-0.9): Quick human spot-check (~50% of cases)
3. **Low-confidence** (<0.7): Full human review (~20% of cases)

This can reduce human effort by ~60% while maintaining quality.

### Batching

Process in batches:
1. Batch 1: 10 PRs → Full LLM + Human pipeline → Validate process
2. Batch 2: 20 PRs → Refined process
3. Batch 3: 20+ PRs → Full scale

### Inter-Annotator Agreement

For reliability, have 2 annotators review 20% of PRs:

```python
from sklearn.metrics import cohen_kappa_score

# If agreement is high (kappa > 0.8), one annotator is sufficient
# If low (kappa < 0.6), need better guidelines or 3 annotators
```

## Reporting Results

### In Your Paper

**Table: Attribution Quality**
```
Metric              | Micro-avg | Macro-avg
--------------------|-----------|----------
Precision           |   78.5%   |   76.2%
Recall              |   82.3%   |   81.7%
F1 Score            |   80.4%   |   78.9%
```

**Text:**
> "We evaluated attribution quality on 50 diverse PRs through LLM-assisted 
> annotation with human verification. Our system achieved 78.5% precision 
> and 82.3% recall (micro-averaged), indicating it correctly identifies 
> most necessary changes while maintaining low false positive rate. 
> Common false positives included overly defensive null checks (18%) and 
> unnecessary logging statements (12%). False negatives primarily consisted 
> of missing test cases (42%) and documentation updates (28%)."

### Qualitative Analysis

Break down errors by category:

**False Positive Analysis:**
```
Category                    | Count | %
----------------------------|-------|-----
Unnecessary null checks     |   62  | 18%
Excessive logging           |   41  | 12%
Over-refactoring           |   38  | 11%
...
```

**False Negative Analysis:**
```
Category                    | Count | %
----------------------------|-------|-----
Missing test cases          |  112  | 42%
Missing documentation       |   75  | 28%
Edge case handling          |   48  | 18%
...
```

## Tools Provided

1. **attribution_evaluation.py**: Core evaluation framework
2. **annotation_pipeline.py**: End-to-end automation script
3. **annotation_ui.py**: Web interface for human review
4. **llm_annotator.py**: LLM annotation helper

See individual files for usage details.
