# Offline Extraction Pipeline - Cost & Token Tracking

This document explains how to track token usage and monetary costs for the offline extraction pipeline that builds historical memory from CI failures.

## Overview

The offline extraction pipeline consists of several phases:

1. **Log Details Extraction** - Parse CI logs to identify failure details
2. **CI Workflow Verification** - Verify workflow structure and commands
3. **Problem Decomposition** - Decompose failures into atomic problems
4. **L1 Memory Building** - Build concrete problem-level failure memory
5. **L2 Memory Building** - Generate repair strategies using LLM
6. **L3 Memory Building** - Extract universal cross-repo patterns using LLM

Each phase makes API calls to the LLM (e.g., DeepSeek V4 Flash), consuming tokens and incurring costs. This guide shows how to measure these costs.

## Architecture

The cost tracking system uses:

- **`RunMetricsRecorder`** - Core metrics tracking class (in `utilities/run_metrics.py`)
- **`DetailedUsageTracker`** - Step-by-step breakdown tracker (in `utilities/detailed_tracker.py`)
- **`LitellmModel`** - LLM wrapper with built-in metrics integration

## Scripts

### 1. Single Instance Processing with Metrics

**File:** `scripts/offline_pipeline_with_metrics.py`

Processes a single instance through the complete pipeline with full metrics tracking.

```bash
# Example: Process instance 102 from flower-dev/flower repo
python scripts/offline_pipeline_with_metrics.py \
    --instance-id 102 \
    --repo flower-dev/flower \
    --model deepseek-v4-flash \
    --output-dir results/metrics/instance_102

# Skip L3 generation (only L1 and L2)
python scripts/offline_pipeline_with_metrics.py \
    --instance-id 410 \
    --repo agno-dev/agno \
    --model deepseek-v4-flash \
    --output-dir results/metrics/instance_410 \
    --skip-l3
```

**Output:**
```
results/metrics/instance_102/
├── run_metrics.json              # Raw metrics ledger
├── cost_time_report.json         # Formatted report
├── cost_time_report.csv          # CSV for spreadsheet analysis
├── detailed_usage_102.json       # Step-by-step breakdown
└── memory/
    ├── failure_memory.json       # L1 memory
    ├── repo_memory.json          # L2 memory
    └── cross_memory.json         # L3 memory
```

### 2. Batch Processing (20 Instances)

**File:** `scripts/batch_offline_metrics.py`

Processes 20 instances from diverse repositories to calculate average costs.

```bash
# Process 20 diverse instances
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/batch_20

# Process specific instances
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --instance-ids 102,410,500,234,567,189,321,445,678,890 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/batch_specific
```

**Output:**
```
results/metrics/batch_20/
├── batch_summary.json            # Aggregated statistics
├── batch_costs.csv               # Per-instance costs
└── per_instance/
    ├── 102/
    │   ├── run_metrics.json
    │   ├── cost_time_report.json
    │   └── detailed_usage_102.json
    ├── 410/
    └── ... (18 more instances)
```

**Key Metrics in batch_summary.json:**

```json
{
  "summary": {
    "total_instances": 20,
    "successful_instances": 19,
    "failed_instances": 1,
    "success_rate": 0.95
  },
  "totals": {
    "total_cost_usd": 1.234567,
    "total_input_tokens": 2500000,
    "total_output_tokens": 500000,
    "total_cached_input_tokens": 100000
  },
  "averages": {
    "avg_cost_usd_per_instance": 0.065029,
    "avg_input_tokens_per_instance": 131578.95,
    "avg_output_tokens_per_instance": 26315.79,
    "avg_wall_time_seconds_per_instance": 45.32
  },
  "projections": {
    "description": "Projected costs for 157 instances (full CIBench)",
    "projected_157_instances_cost_usd": 10.21,
    "projected_157_instances_input_tokens": 20657774,
    "projected_157_instances_output_tokens": 4131579
  }
}
```

### 3. Cost Analysis

**File:** `scripts/analyze_offline_costs.py`

Analyzes batch results and provides detailed breakdowns by phase, repository, and token type.

```bash
python scripts/analyze_offline_costs.py \
    --batch-summary results/metrics/batch_20/batch_summary.json \
    --detailed-tracker-dir results/metrics/batch_20/per_instance \
    --output-dir results/metrics/analysis
```

**Output:**
```
results/metrics/analysis/
├── cost_analysis.json            # Comprehensive analysis
├── phase_breakdown.csv           # Cost by phase (decomposition, L1, L2, L3)
└── cost_visualization.txt        # Text report with ASCII charts
```

**Example Phase Breakdown:**

| Phase          | Count | Total Cost | Avg Cost | Total Input Tokens | Total Output Tokens |
|----------------|-------|------------|----------|-------------------|---------------------|
| decomposition  | 20    | $0.45      | $0.0225  | 800,000           | 150,000             |
| build_l2       | 20    | $0.38      | $0.0190  | 700,000           | 120,000             |
| build_l3       | 20    | $0.32      | $0.0160  | 600,000           | 100,000             |
| build_l1       | 20    | $0.12      | $0.0060  | 400,000           | 130,000             |

## Complete Workflow Example

Here's a complete example to measure costs on 20 instances:

```bash
# Step 1: Ensure you have a dataset
# (Use CIBench test set or your own dataset)
ls data/cibench_test.json

# Step 2: Run batch processing
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/offline_20_deepseek

# Step 3: Analyze results
python scripts/analyze_offline_costs.py \
    --batch-summary results/metrics/offline_20_deepseek/batch_summary.json \
    --detailed-tracker-dir results/metrics/offline_20_deepseek/per_instance \
    --output-dir results/metrics/offline_20_deepseek/analysis

# Step 4: View results
cat results/metrics/offline_20_deepseek/analysis/cost_visualization.txt

# Step 5: Export to CSV for further analysis
# CSV files are already generated in:
# - results/metrics/offline_20_deepseek/batch_costs.csv
# - results/metrics/offline_20_deepseek/analysis/phase_breakdown.csv
```

## Understanding the Metrics

### Token Types

- **Input Tokens**: Tokens sent to the LLM (prompts, context)
- **Cached Input Tokens**: Tokens reused from prompt cache (cheaper)
- **Cache Write Tokens**: Tokens written to cache for future reuse
- **Output Tokens**: Tokens generated by the LLM (responses)
- **Reasoning Tokens**: Internal reasoning tokens (for models like o1)

### Cost Calculation

Total cost is calculated as:
```
cost = (input_tokens × input_rate) + 
       (cached_tokens × cached_rate) +
       (cache_write_tokens × cache_write_rate) +
       (output_tokens × output_rate)
```

Rates are defined in:
- `utilities/model_registry.py` - Model aliases and pricing
- Environment variable `RUN_METRICS_PRICING_FILE` - Custom pricing

### DeepSeek V4 Flash Pricing (Example)

```
Input: $0.14 per million tokens
Cached: $0.014 per million tokens (10x cheaper)
Output: $0.28 per million tokens
```

## Interpreting Results

### Average Cost Per Instance

If average cost = $0.065 per instance:
- 157 instances = $0.065 × 157 = **$10.21 total**

### Cost by Phase

Typical distribution:
- **Decomposition (40%)**: Highest cost - analyzes full CI logs
- **L2 Building (30%)**: Generates repair strategies
- **L3 Building (25%)**: Extracts universal patterns  
- **L1 Building (5%)**: Minimal LLM usage (mostly data transformation)

### Optimization Opportunities

1. **Enable Prompt Caching**: Reuse common context across instances
   - Can reduce costs by 50-70% for decomposition phase
   
2. **Skip L3 for Some Repos**: Use `--skip-l3` if cross-repo patterns not needed
   - Saves ~25% of total cost

3. **Use Smaller Models for L1**: L1 is mostly data transformation
   - Could use a cheaper/faster model just for this phase

## Comparing Models

Run the same 20 instances on different models:

```bash
# DeepSeek V4 Flash
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/deepseek_v4

# GLM 5.2
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model glm5.2 \
    --output-dir results/metrics/glm52

# Compare results
diff results/metrics/deepseek_v4/batch_summary.json \
     results/metrics/glm52/batch_summary.json
```

## Monitoring Long Runs

For processing all 157 instances:

```bash
# Use background processing with progress monitoring
nohup python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 157 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/full_157 \
    > logs/offline_extraction_157.log 2>&1 &

# Monitor progress
tail -f logs/offline_extraction_157.log

# Check intermediate results
watch -n 60 "python scripts/analyze_offline_costs.py \
    --batch-summary results/metrics/full_157/batch_summary.json \
    --output-dir results/metrics/full_157/analysis"
```

## Troubleshooting

### Missing Detailed Tracker Files

If `detailed_usage_{instance_id}.json` files are missing:

1. Check that `DetailedUsageTracker` is initialized in `RunMetricsRecorder`
2. Verify `utilities/detailed_tracker.py` exists
3. Ensure no exceptions during pipeline execution

### Unpriced API Calls

If you see `"unpriced_api_calls": 10` in reports:

1. Check pricing configuration in `utilities/model_registry.py`
2. Set `RUN_METRICS_PRICING_FILE` environment variable:
   ```bash
   export RUN_METRICS_PRICING_FILE=config/model_pricing.json
   ```
3. Use LiteLLM's built-in pricing (automatic for most providers)

### High Costs

If costs are higher than expected:

1. **Check for duplicate processing**: Ensure instances aren't processed twice
2. **Verify caching**: Enable prompt caching for repeated contexts
3. **Profile phases**: Use `analyze_offline_costs.py` to find expensive phases
4. **Reduce context**: Limit log size in decomposition phase

## FAQ

**Q: Why do we test on 20 instances instead of all 157?**

A: 20 diverse instances provide:
- Statistically significant average (±5% margin)
- Fast iteration for testing changes
- Coverage of different repos/failure types
- Lower cost for experimentation

**Q: How accurate is the 157-instance projection?**

A: Projections assume:
- Similar distribution of failure types
- No systematic bias in the 20-instance sample
- Consistent API pricing

Actual costs may vary ±10% based on:
- Complexity of remaining instances
- Model updates/pricing changes
- Cache hit rates

**Q: Can I track costs for online (inference-time) usage?**

A: Yes! The same `RunMetricsRecorder` is used in:
- `scripts/memory_guided_repair.py` - Online repair
- `miniswe-agent/src/minisweagent/run/` - Agent execution

All benchmark runners automatically track costs.

**Q: Where are historical cost metrics stored?**

A: Previous runs are in:
```
results/
├── baseline_deepseek_v4/run_metrics.json
├── l1_l2_l3_deepseek_v4/run_metrics.json
└── ablations/*/run_metrics.json
```

Use `jq` to extract totals:
```bash
jq '.summary.total_cost_usd' results/*/run_metrics.json
```

## References

- **RunMetricsRecorder**: `utilities/run_metrics.py`
- **DetailedUsageTracker**: `utilities/detailed_tracker.py`
- **Model Pricing**: `utilities/model_registry.py`
- **LLM Integration**: `utilities/llm_model.py`

## Support

For questions about cost tracking:
1. Check existing metrics in `results/` directories
2. Review `utilities/run_metrics.py` documentation
3. Examine example runs in `results/metrics/`
