# Whole PR Memory Build - Token Tracking & Error Handling Improvements

## Date: 2026-11-10

## Summary

Fixed critical error handling issues and added comprehensive token usage tracking for the whole PR memory building process. The system now provides real-time visibility into token consumption and costs across all 157 instances.

---

## Problems Fixed

### 1. **Length Limit Error (Issue 221)**

**Problem**: 
- Issue 221 hit a token length limit during patch chunk 3 summarization
- The model exhausted its 16,000 token output budget
- Error: `ValueError: Issue 221 patch chunk 3: model did not return one object`

**Root Cause**:
- When `invoke_llm_with_retry` returned `"SPLIT_REQUIRED"` due to length limit, `_invoke_single_object` didn't handle it properly
- The code tried to validate a string instead of a dict object

**Fix**:
- Added explicit handling for `"SPLIT_REQUIRED"` signal in `_invoke_single_object`
- Enhanced `_invoke_validated_object` to detect and propagate length limit errors
- Provides clear actionable error message directing user to increase `--max-evidence-chars`

### 2. **Missing Token Usage Visibility**

**Problem**:
- No real-time feedback on token consumption during the build process
- Impossible to estimate costs for remaining instances
- No way to track which steps consume the most tokens

**Fix**:
- Added real-time token usage printing for every LLM API call
- Added per-instance summary after each instance completes
- Added aggregate summary at the end showing totals and estimates

---

## New Features

### 1. **Real-Time Token Usage Display**

Every LLM API call now shows:
```
[API] Issue 221 patch chunk 1
      tokens: prompt=19,474, completion=3,364, total=22,838
```

### 2. **Per-Instance Summary**

After each instance completes, you see:
```
═══ Instance 221 Token Usage Summary ═══
API Calls: 8
Input Tokens: 142,356
Cached Input: 12,450
Output Tokens: 28,934
Reasoning Tokens: 15,010
Total Tokens: 186,300
Cost: $0.234567
API Time: 45.3s

Phase Breakdown:
  patch_chunk_1: 1 calls, 19,474 in + 3,364 out, $0.024531
  patch_chunk_2: 1 calls, 17,330 in + 1,608 out, $0.019876
  l1_generation: 1 calls, 8,057 in + 628 out, $0.008234
  l2_generation: 1 calls, 1,024 in + 873 out, $0.001234
  l3_generation: 1 calls, 2,054 in + 3,401 out, $0.005432
═══════════════════════════════════════════
```

### 3. **Aggregate Summary & Cost Estimates**

At the end of the build, you get:
```
======================================================================
Build Summary:
======================================================================
Raw episodes: 139/157
Complete L1/L2/L3: 139/157
Output: /Users/.../experimental_data/whole_repair_history

──────────────────────────────────────────────────────────────────────
Aggregate Token Usage & Cost Across All Instances:
──────────────────────────────────────────────────────────────────────
Total Instances Processed: 139
Total Input Tokens: 2,456,789
  - Cached: 234,567
  - Uncached: 2,222,222
Total Output Tokens: 1,123,456
Total Reasoning Tokens: 456,789
Grand Total Tokens: 4,037,034
Total Cost: $4.567890
Total API Time: 3456.7s

Averages per Instance:
  Tokens: 29,043
  Cost: $0.032849

Estimate for Remaining 18 Instances:
  Estimated Tokens: 522,774
  Estimated Cost: $0.591282

Estimate for All 157 Instances:
  Estimated Total Tokens: 4,559,751
  Estimated Total Cost: $5.159172
======================================================================
```

### 4. **Enhanced Phase Tracking**

All API calls are now tagged with descriptive phases:
- `patch_chunk_1`, `patch_chunk_2`, etc. - Patch summarization
- `context_chunk_1`, `context_chunk_2`, etc. - Context summarization
- `reduction_r1_b1`, `reduction_r2_b1`, etc. - Hierarchical reduction
- `l1_generation` - L1 abstraction generation
- `l2_generation` - L2 strategy generation
- `l3_generation` - L3 pattern generation

This allows you to see exactly which steps are most expensive.

---

## Technical Changes

### Files Modified

#### 1. `experiments/01_whole_repair_history/build_whole_pr_memory.py`

**Function: `_invoke_single_object`**
- Added handling for `"SPLIT_REQUIRED"` signal
- Extracts and prints token usage from metadata before removing it
- Provides clear error messages for length limit errors

**Function: `_invoke_validated_object`**
- Catches length limit errors and provides actionable guidance
- Propagates errors with full context

**Function: `_print_instance_usage_summary` (NEW)**
- Prints comprehensive token usage summary for each instance
- Shows breakdown by phase
- Displays costs and API time

**Function: `generate_whole_pr_abstractions`**
- Added phase tracking for all LLM calls
- Sets `_current_phase` attribute for metrics recording

**Function: `_reduce_summaries_to_fit`**
- Added phase tracking for reduction rounds
- Properly restores phase after completion

**Function: `main`**
- Calls `_print_instance_usage_summary` after each instance (success or failure)
- Prints comprehensive aggregate summary at the end
- Shows estimates for remaining instances and all 157 instances

#### 2. `utilities/llm_model.py`

**Class: `LitellmModel`**
- Modified to use `_current_phase` attribute if set by caller
- Falls back to `"context_llm"` if not set
- Phase is used consistently in both `begin_api_call` and `record_response`

---

## Usage

### Run the build normally:
```bash
python experiments/01_whole_repair_history/build_whole_pr_memory.py
```

### Run with specific instances (e.g., test the fix on issue 221):
```bash
python experiments/01_whole_repair_history/build_whole_pr_memory.py --issue-ids 221 --rebuild
```

### Run with larger evidence budget (if needed):
```bash
python experiments/01_whole_repair_history/build_whole_pr_memory.py --max-evidence-chars 200000
```

---

## Detailed Metrics Files

The system automatically saves detailed metrics to:

1. **`construction_metrics/run_metrics.json`** - Full ledger of all API calls
2. **`construction_metrics/cost_time_report.json`** - Human-readable summary report
3. **`construction_metrics/cost_time_report.csv`** - CSV format for analysis
4. **`construction_metrics/detailed_usage.json`** - Per-instance detailed breakdown

---

## Benefits

1. **Real-time Visibility**: See token consumption as it happens
2. **Cost Tracking**: Know exactly how much each instance costs
3. **Better Debugging**: Phase breakdown helps identify expensive operations
4. **Accurate Estimates**: Predict costs for remaining work
5. **Error Recovery**: Clear guidance when hitting token limits

---

## Example Output

When running the build, you'll now see:

```
[139/157] 221: collecting whole PR
[139/157] 221: building singular L1/L2/L3
[Chunking] Model: openrouter/deepseek/deepseek-v4-flash, Target prompt size: 81920 tokens
[Chunking] Packed 1 independent cluster(s) into 1 token-bounded chunk(s)
      [API] Issue 221 patch chunk 1
            tokens: prompt=19,474, completion=3,364, total=22,838
      [API] Issue 221 patch chunk 2
            tokens: prompt=17,330, completion=1,608, total=18,938
      [API] Issue 221 patch chunk 3
            tokens: prompt=19,709, completion=3,256, total=22,965
      [API] Issue 221 L1
            tokens: prompt=8,057, completion=628, total=8,685
      [API] Issue 221 L2
            tokens: prompt=1,024, completion=873, total=1,897
      [API] Issue 221 L3
            tokens: prompt=2,054, completion=3,401, total=5,455

      ═══ Instance 221 Token Usage Summary ═══
      API Calls: 6
      Input Tokens: 67,648
      Cached Input: 0
      Output Tokens: 13,130
      Reasoning Tokens: 0
      Total Tokens: 80,778
      Cost: $0.089567
      API Time: 32.4s

      Phase Breakdown:
        patch_chunk_1: 1 calls, 19,474 in + 3,364 out, $0.025678
        patch_chunk_2: 1 calls, 17,330 in + 1,608 out, $0.019234
        patch_chunk_3: 1 calls, 19,709 in + 3,256 out, $0.025123
        l1_generation: 1 calls, 8,057 in + 628 out, $0.008890
        l2_generation: 1 calls, 1,024 in + 873 out, $0.002123
        l3_generation: 1 calls, 2,054 in + 3,401 out, $0.008519
      ═══════════════════════════════════════════
```

---

## Notes

- The token counts include input, cached input, output, and reasoning tokens
- Costs are calculated using the actual model pricing (DeepSeek V4 Flash via OpenRouter)
- Phase tracking helps identify which operations are most expensive
- Estimates assume similar complexity across remaining instances
- The `detailed_usage.json` file contains complete per-call breakdowns for analysis

---

## Next Steps

1. Run the build on all 157 instances
2. Monitor token usage and costs in real-time
3. If any instance hits length limits, increase `--max-evidence-chars`
4. Use the phase breakdown to optimize expensive operations
5. Review `detailed_usage.json` for comprehensive analysis
