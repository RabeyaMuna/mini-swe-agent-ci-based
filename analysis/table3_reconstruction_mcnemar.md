# Table 3: reconstruction strategy with paired significance tests

The outcome denominator is the fixed 408-instance evaluation set for mini-SWE
and the fixed 200-instance subset for Codex. Success means
`conclusion == "success"` in `jobs_results_diff.jsonl`. Duplicate records are
resolved by keeping the last record for an instance, and an absent result is
counted as unresolved so that the reported denominators remain 408 and 200.

The table reports two-sided exact McNemar tests comparing each one-way
reconstruction strategy with ReCAP (full). `p_Holm` applies Holm correction to
the six planned full-versus-one-way comparisons in this table.

| Agent | Model | Configuration | Resolved | Pass@1 (%) | Input (K/inst.) | Cache Hit (%) | Cost ($/inst.) | Exact p vs. full | p_Holm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mini-SWE | MiniMax-M2.5 | ReCAP Endpoint | 116/408 | 28.4 | 1,262.2 | 86.2 | 0.108 | 0.0201 | **0.0401** |
|  |  | ReCAP Development | 111/408 | 27.2 | 1,564.7 | 87.1 | 0.127 | 0.00338 | **0.0101** |
|  |  | ReCAP (full) | 130/408 | 31.9 | 1,903.2 | 89.1 | 0.139 | -- | -- |
| mini-SWE | DeepSeek-V4-Flash | ReCAP Endpoint | 116/408 | 28.4 | 1,194.0 | 86.0 | 0.040 | 0.00143 | **0.00572** |
|  |  | ReCAP Development | 113/408 | 27.7 | 1,480.1 | 87.0 | 0.048 | 0.000508 | **0.00305** |
|  |  | ReCAP (full) | 134/408 | 32.8 | 1,800.3 | 88.9 | 0.052 | -- | -- |
| Codex | MiniMax-M2.5 | ReCAP Endpoint | 51/200 | 25.5 | 1,670.5 | 94.3 | 0.095 | 0.3438 | 0.3438 |
|  |  | ReCAP Development | **44/200** | **22.0** | 2,632.3 | 95.7 | 0.136 | 0.000977 | **0.00488** |
|  |  | ReCAP (full) | 55/200 | 27.5 | 2,210.0 | 95.0 | 0.118 | -- | -- |

Suggested table note:

> Resolved denotes the number of successfully repaired instances; Input is the
> average input-token usage in thousands per instance; Cache Hit is the
> percentage of input tokens served from the prompt cache; and Cost is the
> average inference cost per instance. McNemar p-values are from two-sided
> exact paired tests against ReCAP (full); p_Holm applies Holm correction across
> the six planned comparisons. Bold adjusted p-values are below 0.05.

## Paired counts

In each row below, `A only` is the number resolved only by the first named
configuration and `B only` is the number resolved only by the second. McNemar's
test uses these discordant pairs.

| Agent/model | Pair (A vs. B) | Both resolved | A only | B only | Neither | Exact p |
|---|---|---:|---:|---:|---:|---:|
| mini-SWE / MiniMax-M2.5 | Endpoint vs. Development | 93 | 23 | 18 | 274 | 0.5327 |
|  | Endpoint vs. full | 107 | 9 | 23 | 269 | 0.0201 |
|  | Development vs. full | 101 | 10 | 29 | 268 | 0.00338 |
| mini-SWE / DeepSeek-V4-Flash | Endpoint vs. Development | 102 | 14 | 11 | 281 | 0.6900 |
|  | Endpoint vs. full | 110 | 6 | 24 | 268 | 0.00143 |
|  | Development vs. full | 106 | 7 | 28 | 267 | 0.000508 |
| Codex / MiniMax-M2.5 | Endpoint vs. Development | 41 | 10 | 3 | 146 | 0.0923 |
|  | Endpoint vs. full | 48 | 3 | 7 | 142 | 0.3438 |
|  | Development vs. full | 44 | 0 | 11 | 145 | 0.000977 |

## Overall three-strategy tests

McNemar's test is pairwise, so it does not produce one valid overall p-value
for Endpoint, Development, and full together. The corresponding omnibus test
for three paired binary configurations is Cochran's Q (two degrees of
freedom):

| Agent/model | Cochran's Q | df | Omnibus p |
|---|---:|---:|---:|
| mini-SWE / MiniMax-M2.5 | 10.393 | 2 | 0.00554 |
| mini-SWE / DeepSeek-V4-Flash | 17.200 | 2 | 0.000184 |
| Codex / MiniMax-M2.5 | 10.941 | 2 | 0.00421 |

All three omnibus tests reject equal repair rates among the reconstruction
strategies. The post-hoc exact McNemar tests show that full significantly
outperforms both one-way strategies in the two mini-SWE blocks. For Codex,
full significantly outperforms Development but not Endpoint after Holm
correction.

There is no single cross-agent/model p-value in the report. Pooling the three
blocks would count many of the same benchmark instances repeatedly and violate
the independence assumption. The three block-level omnibus tests and the six
Holm-adjusted planned contrasts are therefore the appropriate overall report.

## Data-integrity findings

- The Codex Development outcome file contains 199 of the expected 200 IDs and
  reports 44 successes. Instance `459` is absent. Therefore the current files
  support 44/200 (22.0%), not 47/200 (23.5%). The three additional successes
  must be restored as per-instance outcomes before a valid paired test can be
  computed for the 47/200 result.
- The mini-SWE/MiniMax full file is missing instance `423` and repeats failed
  instance `355`. Instance `423` failed under both one-way strategies, so
  treating its missing full result as unresolved does not alter any discordant
  pair or McNemar p-value.
- The Codex full file repeats instances `62`, `63`, and `92`; the repeated
  conclusions agree, so deduplication does not change its 55/200 total.
