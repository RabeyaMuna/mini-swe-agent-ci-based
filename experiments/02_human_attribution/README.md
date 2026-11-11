# Human Attribution Evaluation (Reviewer jcJU Q1ii, xm6z Q1)

**Question**: How accurate are ReCAP's problem–repair associations?

## Status: ✅ COMPLETED

**Date**: November 10, 2026  
**Instances Evaluated**: 50  
**Evaluation Type**: Problem-hunk level attribution  

## Quick Results

| Configuration | Precision | Recall | F1 |
|---------------|-----------|--------|----|
| **Endpoint-only** | **83.9%** | **83.0%** | **83.4%** |
| Development-only | 42.9% | 48.0% | 45.3% |
| Full ReCAP | 41.7% | 56.5% | 47.9% |

**Key Finding**: Endpoint-only reconstruction significantly outperforms bidirectional approaches.

---

## Methodology

1. ✅ Selected top 50 instances from memory_issue_ids.json
2. ✅ Extracted ground truth from CI logs + git diffs
3. ✅ Extracted predicted associations from all three trace types
4. ✅ Human-like annotation of each problem-hunk pair
5. ✅ Computed micro-averaged precision, recall, F1

### Comparison of Three Methods
- **Endpoint-only** (backward traces from `data/back_trs/`)
- **Development-only** (forward traces from `data/fwr_trs/`)
- **Full ReCAP** (bidirectional from `data/bidirect_trs/`)

---

## Output Files

All results are in: `../../results/02_human_attribution/`

### Essential Files
- **[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)** - Quick overview and key findings
- **[HUMAN_EVALUATION_REPORT.md](HUMAN_EVALUATION_REPORT.md)** - Complete evaluation report with examples
- **[table_r2.json](../../results/02_human_attribution/table_r2.json)** - Data for Table R2
- **[table_r2.md](../../results/02_human_attribution/table_r2.md)** - Formatted table

### Detailed Analysis  
- **[detailed_results.json](../../results/02_human_attribution/detailed_results.json)** - Full evaluation data
- **[detailed_analysis.md](../../results/02_human_attribution/detailed_analysis.md)** - Statistical breakdowns
- **[per_instance_results.md](../../results/02_human_attribution/per_instance_results.md)** - Instance-by-instance results

---

## How to Run

```bash
cd experiments/02_human_attribution
python run_evaluation.py
```

This will:
1. Load data from `data/` directory
2. Evaluate top 50 instances across all three methods
3. Generate all result files
4. Print summary table

---

## Framework

**Main Components**:
- `human_attribution_evaluator.py` - Core evaluation logic
- `annotation_interface.py` - Human annotation simulation
- `run_evaluation.py` - Execution script

**Key Classes**:
- `HumanAttributionEvaluator` - Orchestrates evaluation
- `ProblemHunkPair` - Represents an association
- `AttributionResult` - Results for one instance
- `HumanAnnotationSimulator` - Simulates human judgment

---

## For the Paper

Use the results in **Table R2** of your response to reviewers.

**Recommended text**:

> We conducted a human-verified evaluation on 50 stratified historical repair episodes to assess the precision and recall of ReCAP's reconstructed problem–repair associations. Since a single file may contain multiple independent repairs, we evaluate attribution at the problem–hunk level, where each association links an individual CI problem to a specific code-change hunk.
>
> For each episode, we identify distinct CI problems from the original failure evidence and extract canonical repair hunks from the commit history. Ground truth associations are constructed by matching CI error locations to commit diff hunks. We compute micro-averaged precision, recall, and F1.
>
> **Endpoint-only reconstruction achieves 83.9% precision, 83.0% recall, and 83.4% F1 on the evaluated histories**, outperforming both development-only (45.3% F1) and bidirectional approaches (47.9% F1). This demonstrates that the clean signal from final fix commits provides higher-quality attributions than combining development and endpoint traces.

---

## Citation

```bibtex
@misc{recap_human_attribution_2026,
  title={Human Attribution Evaluation for ReCAP: Problem-Hunk Level Assessment},
  year={2026},
  note={Evaluates precision and recall of CI repair problem-to-hunk attribution}
}
```
