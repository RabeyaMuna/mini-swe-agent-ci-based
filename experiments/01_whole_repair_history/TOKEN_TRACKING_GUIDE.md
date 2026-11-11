# Token Tracking & Cost Estimation Guide

## Overview

The whole PR memory builder now provides comprehensive real-time token usage tracking and cost estimation. This guide shows you how to use these features to monitor and optimize your build process.

---

## Quick Start

### Run the build with full tracking:

```bash
python experiments/01_whole_repair_history/build_whole_pr_memory.py \
  --model deepseek-v4-flash
```

You'll see:
1. Real-time token usage for each API call
2. Per-instance summaries after each instance completes
3. Aggregate statistics at the end
4. Cost estimates for remaining work

---

## What You'll See

### 1. Real-Time Token Usage

For every LLM API call:
```
[API] Issue 221 patch chunk 1
      tokens: prompt=19,474, completion=3,364, total=22,838
```

### 2. Per-Instance Summary

After each instance completes:
```
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

### 3. Aggregate Summary

At the end of the build:
```
======================================================================
Build Summary:
======================================================================
Raw episodes: 139/157
Complete L1/L2/L3: 139/157

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

---

## Understanding the Phases

The system tracks token usage across different phases:

| Phase | Description |
|-------|-------------|
| `patch_chunk_1`, `patch_chunk_2`, ... | Summarizing dependency-aware patch chunks |
| `context_chunk_1`, `context_chunk_2`, ... | Summarizing CI/workflow context chunks |
| `reduction_r1_b1`, `reduction_r2_b1`, ... | Hierarchical reduction (round, batch) |
| `l1_generation` | Generating L1 concrete episode abstraction |
| `l2_generation` | Generating L2 repository strategy |
| `l3_generation` | Generating L3 transferable pattern |

---

## Detailed Metrics Files

All metrics are automatically saved to:

### 1. `construction_metrics/run_metrics.json`
Full ledger of every API call with:
- Timestamp
- Phase
- Model
- Token counts (input, cached, output, reasoning)
- Cost and pricing source
- Duration

### 2. `construction_metrics/cost_time_report.json`
Human-readable summary with:
- Per-instance totals
- Aggregate statistics
- Success/failure status

### 3. `construction_metrics/cost_time_report.csv`
CSV format for spreadsheet analysis

### 4. `construction_metrics/detailed_usage.json`
Per-instance detailed breakdown with:
- Complete call-by-call history
- Phase-wise aggregation
- Timestamps and execution time

---

## Example: Analyzing Costs

### Find most expensive phases:
```bash
# View detailed usage
cat experimental_data/whole_repair_history/construction_metrics/detailed_usage.json | jq '.[] | {instance_id, phase_breakdown}'

# Find instances with high token usage
cat experimental_data/whole_repair_history/construction_metrics/cost_time_report.json | jq '.instances[] | select(.total_tokens > 100000) | {instance_id, total_tokens, total_cost_usd}'
```

### Analyze by phase:
```python
import json

# Load detailed usage
with open('experimental_data/whole_repair_history/construction_metrics/detailed_usage.json') as f:
    data = json.load(f)

# Aggregate by phase across all instances
phase_totals = {}
for instance in data:
    for phase, stats in instance['phase_breakdown'].items():
        if phase not in phase_totals:
            phase_totals[phase] = {'calls': 0, 'tokens': 0, 'cost': 0.0}
        phase_totals[phase]['calls'] += stats['call_count']
        phase_totals[phase]['tokens'] += stats['total_tokens']
        phase_totals[phase]['cost'] += stats['total_cost_usd']

# Print sorted by cost
for phase, totals in sorted(phase_totals.items(), key=lambda x: x[1]['cost'], reverse=True):
    print(f"{phase}: {totals['calls']} calls, {totals['tokens']:,} tokens, ${totals['cost']:.6f}")
```

---

## Troubleshooting

### Error: "exhausted output budget"

**Cause**: The model ran out of its 16,000 token output budget while generating a response.

**Solution**:
```bash
# Increase the evidence budget
python experiments/01_whole_repair_history/build_whole_pr_memory.py \
  --max-evidence-chars 200000 \
  --model deepseek-v4-flash
```

### High token usage on specific instances

**Investigate**:
```bash
# Check which phases are expensive
cat construction_metrics/detailed_usage.json | jq '.[] | select(.instance_id == "221") | .phase_breakdown'

# See if chunking was applied
cat experimental_data/whole_repair_history/abstraction_metadata.json | jq '.[] | select(.issue_id == "221")'
```

### Cost is higher than expected

**Check**:
1. Verify the model pricing in `utilities/model_token_config.py`
2. Look for instances with many reduction rounds (sign of oversized evidence)
3. Check if caching is working (cached_input_tokens > 0)

---

## Best Practices

### 1. Monitor costs in real-time
- Watch the per-instance summaries
- Stop the build if costs are too high
- Adjust `--max-evidence-chars` as needed

### 2. Use estimates to plan
- Look at the "Estimate for All 157 Instances" projection
- Budget accordingly before running the full build

### 3. Analyze patterns
- Use the phase breakdown to identify expensive operations
- Consider optimizing prompts for high-cost phases

### 4. Resume safely
- The build automatically resumes from completed instances
- All costs are tracked incrementally
- No duplicate API calls are made

---

## Testing

Test the tracking on a single instance:

```bash
# Test on instance 1
python experiments/01_whole_repair_history/test_token_tracking.py 1

# Test on a specific instance (e.g., 221)
python experiments/01_whole_repair_history/test_token_tracking.py 221
```

---

## Cost Projection Examples

### Example 1: Early Estimate (after 10 instances)
```
Averages per Instance:
  Tokens: 35,000
  Cost: $0.042

Estimate for All 157 Instances:
  Estimated Total Tokens: 5,495,000
  Estimated Total Cost: $6.594
```

### Example 2: Mid-Build Estimate (after 50 instances)
```
Averages per Instance:
  Tokens: 28,500
  Cost: $0.034

Estimate for All 157 Instances:
  Estimated Total Tokens: 4,474,500
  Estimated Total Cost: $5.338
```

### Example 3: Nearly Complete (after 139 instances)
```
Averages per Instance:
  Tokens: 29,043
  Cost: $0.032849

Estimate for All 157 Instances:
  Estimated Total Tokens: 4,559,751
  Estimated Total Cost: $5.159172
```

As you process more instances, the estimates become more accurate.

---

## Advanced: Custom Analysis

### Export to CSV for Excel/Sheets:
```bash
cp construction_metrics/cost_time_report.csv ~/Desktop/
```

### Generate summary charts:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('construction_metrics/cost_time_report.csv')

# Plot token usage distribution
plt.figure(figsize=(12, 6))
plt.hist(df['total_input_tokens'] + df['total_output_tokens'], bins=30)
plt.xlabel('Total Tokens')
plt.ylabel('Frequency')
plt.title('Token Usage Distribution Across Instances')
plt.savefig('token_distribution.png')

# Plot cost vs tokens
plt.figure(figsize=(12, 6))
plt.scatter(df['total_input_tokens'] + df['total_output_tokens'], df['total_cost_usd'])
plt.xlabel('Total Tokens')
plt.ylabel('Cost (USD)')
plt.title('Cost vs Token Usage')
plt.savefig('cost_vs_tokens.png')
```

---

## Summary

The enhanced token tracking provides:
- ✅ Real-time visibility into API usage
- ✅ Accurate cost tracking and estimation
- ✅ Phase-level breakdowns for optimization
- ✅ Automatic saving of all metrics
- ✅ Easy analysis and reporting

Use these tools to monitor your build, optimize costs, and plan for the full 157-instance run.
