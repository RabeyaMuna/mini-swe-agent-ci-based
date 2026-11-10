# McNemar Test Verification for Table

## CRITICAL ISSUE FOUND ⚠️

**The table caption claims**: "Across both evaluation sets, \tool significantly outperforms the corresponding baselines (exact McNemar tests, $p<0.05$)."

**Reality**: This is **NOT TRUE** for the 200-instance matched subset!

---

## Full Evaluation (408 instances)

### mini-SWE + MiniMax-M2.5

#### No-Experience vs Tool (Bidirectional)
- **Baseline**: 80/408 (19.6%)
- **Tool**: 130/408 (31.9%)
- **Improvement**: +50 instances (+12.3 pp) ✓ matches table
- **Contingency Table**:
  - Both pass: 67
  - Baseline only (b): 13
  - Tool only (c): 63
  - Both fail: 265
- **McNemar χ²**: (|13-63|-1)²/(13+63) = 31.5921
- **p-value**: < 0.000001
- **Result**: ✓✓✓ **HIGHLY SIGNIFICANT** (p < 0.001)

#### Historical-plan vs Tool
**⚠️ PROBLEM**: Table shows Historical-plan = 68/408 (16.7%)
- NO file in results matches this!
- Backward L1_L2_L3 = 110/408 (27.0%)
- Forward L1_L2_L3 = 111/408 (27.2%)

**Using Backward as Historical-plan**:
- **Baseline**: 110/408 (27.0%)
- **Tool**: 130/408 (31.9%)
- **Improvement**: +20 instances (+4.9 pp)
- **Contingency Table**:
  - Both pass: 100
  - Baseline only (b): 10
  - Tool only (c): 30
  - Both fail: 268
- **McNemar χ²**: 9.0250
- **p-value**: 0.002663
- **Result**: ✓✓ **VERY SIGNIFICANT** (p < 0.01)

---

### mini-SWE + DeepSeek-V4-Flash

#### No-Experience vs Tool (Bidirectional)
- **Baseline**: 95/408 (23.3%) ✓ matches table
- **Tool**: 134/408 (32.8%) ✓ matches table
- **Improvement**: +39 instances (+9.6 pp) ✓ matches table (9.5 pp)
- **Contingency Table**:
  - Both pass: 90
  - Baseline only (b): 5
  - Tool only (c): 44
  - Both fail: 269
- **McNemar χ²**: 29.4694
- **p-value**: < 0.000001
- **Result**: ✓✓✓ **HIGHLY SIGNIFICANT** (p < 0.001)

#### Historical-plan vs Tool
**⚠️ PROBLEM**: Table shows Historical-plan = 73/408 (17.9%)
- NO file in results matches this!
- Backward L1_L2_L3 = 116/408 (28.4%)

**Using Backward as Historical-plan**:
- **Baseline**: 116/408 (28.4%)
- **Tool**: 134/408 (32.8%)
- **Improvement**: +18 instances (+4.4 pp)
- **Contingency Table**:
  - Both pass: 110
  - Baseline only (b): 6
  - Tool only (c): 24
  - Both fail: 268
- **McNemar χ²**: 9.6333
- **p-value**: 0.001911
- **Result**: ✓✓ **VERY SIGNIFICANT** (p < 0.01)

---

## Matched Subset (200 instances)

### mini-SWE + MiniMax-M2.5

#### No-Experience vs Tool (Bidirectional)
- **Baseline**: 46/200 (23.0%) ✓ matches table
- **Tool**: 55/200 (27.5%) ✓ matches table
- **Improvement**: +9 instances (+4.5 pp) ✓ matches table
- **Contingency Table**:
  - Both pass: 38
  - Baseline only (b): 8
  - Tool only (c): 17
  - Both fail: 137
- **McNemar χ²**: (|8-17|-1)²/(8+17) = 2.5600
- **p-value**: 0.109599
- **Result**: ✗ **NOT SIGNIFICANT** (p ≥ 0.05) ❌❌❌

---

### Codex + MiniMax-M2.5

**⚠️ CANNOT VERIFY**: No `jobs_results_diff.jsonl` file found in `results/codex/baseline_minimax_minimax-m2_5/`
- Only `predictions.json` exists
- Cannot calculate McNemar test without baseline results

---

## Summary Table

| Comparison | Dataset | Baseline | Tool | Δ (pp) | χ² | p-value | Significant? |
|------------|---------|----------|------|--------|-----|---------|--------------|
| **mini-SWE + MiniMax** | 408 | 80/408 | 130/408 | +12.3 | 31.59 | <0.001 | ✓✓✓ YES |
| **mini-SWE + MiniMax** | 408 | 110/408* | 130/408 | +4.9 | 9.03 | 0.003 | ✓✓ YES |
| **mini-SWE + DeepSeek** | 408 | 95/408 | 134/408 | +9.6 | 29.47 | <0.001 | ✓✓✓ YES |
| **mini-SWE + DeepSeek** | 408 | 116/408* | 134/408 | +4.4 | 9.63 | 0.002 | ✓✓ YES |
| **mini-SWE + MiniMax** | 200 | 46/200 | 55/200 | +4.5 | 2.56 | 0.110 | ✗ **NO** ❌ |
| **Codex + MiniMax** | 200 | 31/200 | 55/200 | +12.0 | ? | ? | **CANNOT VERIFY** |

*Using Backward L1_L2_L3 as "Historical-plan" (actual file found)

---

## Recommendations

### 1. Fix the table caption
**Current**: "Across both evaluation sets, \tool significantly outperforms the corresponding baselines (exact McNemar tests, $p<0.05$)."

**Should be**: 
- Option A: "On the full evaluation (408 instances), \tool significantly outperforms the corresponding baselines (exact McNemar tests, $p<0.01$)."
- Option B: Remove the significance claim and add footnotes for specific comparisons
- Option C: Add the caveat: "*except for mini-SWE + MiniMax-M2.5 on the 200-instance subset (p=0.11)"

### 2. Clarify "Historical-plan"
- The table shows Historical-plan = 68/408 and 73/408
- No result files match these numbers
- Backward L1_L2_L3 shows 110/408 and 116/408
- **Action needed**: Identify what "Historical-plan" actually refers to

### 3. Generate Codex baseline results
- Need `results/codex/baseline_minimax_minimax-m2_5/jobs_results_diff.jsonl`
- Currently only `predictions.json` exists
- Cannot verify the Codex comparison without this

### 4. Consider reporting effect sizes
Along with p-values, consider reporting:
- Odds ratio: OR = (c/b) = how many times more likely the tool succeeds vs baseline
- For mini-SWE+MiniMax (200): OR = 17/8 = 2.125 (tool 2.1x more likely to succeed on different instances)

---

## Statistical Interpretation

### Why mini-SWE + MiniMax (200) is not significant:
- Small improvement: only +9 instances
- Discordant pairs: 8 vs 17 (not extreme enough)
- χ² = 2.56 is below the critical value of 3.84 for α=0.05

### What IS significant:
- **Full 408-instance evaluations**: ALL comparisons p<0.01
- Larger sample sizes provide more statistical power
- Larger effect sizes (more discordant pairs) are easier to detect

---

## Conclusion

**The table caption is INCORRECT** as stated. The claim of significance "across both evaluation sets" is false because the 200-instance mini-SWE + MiniMax comparison has p=0.110 (not significant at α=0.05).

**Action required**: Revise the caption or run additional experiments to achieve significance on the 200-instance subset.
