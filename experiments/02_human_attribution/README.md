# Human problem–hunk attribution evaluation

This experiment measures whether each reconstruction method correctly links an
individual CI problem to the code-change hunk that repairs it.

## Current status

The 50-episode review bundle has been prepared, but the evaluation is **not yet
human-verified**. The files named `MODEL_PROPOSED_*` contain preliminary,
model-assisted labels and must not be reported as human results.

The earlier simulator-based values are invalid for the reviewer-requested human
evaluation. A program or LLM cannot serve as the human reference against which
the same reconstruction is evaluated.

## Reference construction

For each episode, the blinded reference sheet contains every canonical CI
problem × final-patch hunk pair. The annotator reviews the original failure
evidence, workflow, full diff, and hunk context without seeing any ReCAP
prediction, then assigns exactly one label:

- `direct`: the hunk directly repairs the stated CI problem;
- `shared`: the hunk is a valid repair for this problem and also contributes to
  at least one other distinct problem;
- `unrelated`: the evidence does not support a causal repair relationship; or
- `uncertain`: a relationship is plausible, but the available evidence is not
  sufficient for a defensible decision.

`direct` and `shared` are reference positives. `unrelated` is a reference
negative. `uncertain` is excluded from the primary metrics.

The reference is based on observed CI evidence and the actual diff. Repair
strategy text from a reconstruction must not be used to create the reference,
because that would leak method output into the ground truth.

## Required manual work

1. Fill `label`, `confidence`, `rationale`, and `annotator` for **all 1,664 rows**
   in `results/02_human_attribution_50_repos/problem_hunk_pairs.csv`.
2. Independently review all reconstructed-problem alignments in
   `prediction_alignment.csv`. For each row, put the confirmed canonical ID or
   `UNMAPPED` in `manual_canonical_problem_id`, and put `reviewed` in
   `manual_review_status`.
3. Do not use `model_proposed_problem_hunk_pairs.csv` while constructing the
   reference. It is available only for later disagreement analysis.
4. Run the strict scorer. It refuses to calculate human-verified results if any
   reference pair or alignment remains unreviewed.

```bash
python3 experiments/02_human_attribution/score_human_verified.py \
  --bundle results/02_human_attribution_50_repos \
  --prediction-mode native-hunks
```

The final outputs are `HUMAN_VERIFIED_SCORES.json`,
`HUMAN_VERIFIED_RESULTS.md`, and `human_verified_scoring_evidence.jsonl`.

Native problem–hunk evaluation requires every reconstructed problem to contain
`supporting_hunk_ids`. The current traces contain only `affected_files`, so they
cannot yet support a native hunk-level claim. After manual review, the explicit
fallback `--prediction-mode expand-files` treats every hunk in each predicted
file as predicted. That fallback must be reported as a **file-to-hunk-expanded
evaluation**, not native hunk selection.

## Metrics

For method predictions \(A\) and the human-positive reference \(G\), after
removing uncertain pairs:

- \(TP=|A \cap G|\)
- \(FP=|A-G|\)
- \(FN=|G-A|\)
- precision = \(TP/(TP+FP)\)
- recall = \(TP/(TP+FN)\)
- F1 = \(2TP/(2TP+FP+FN)\)

The reported values are micro-averaged across all 50 episodes. Confidence
intervals are obtained by resampling episodes, rather than individual pairs,
10,000 times. Pairwise accuracy is secondary because the candidate space is
dominated by unrelated pairs.

## Recommended reliability check

If resources permit, have a second annotator independently label at least a
random 20% of episodes. Report agreement before adjudication (Cohen's kappa for
the four labels, plus positive-versus-negative agreement), then adjudicate
disagreements before computing the final scores.
