# Quick Start: Offline Pipeline Cost Tracking

This guide helps you quickly measure the token usage and monetary costs of the offline extraction pipeline.

## TL;DR

```bash
# 1. Quick demo (3 instances)
bash scripts/demo_cost_tracking.sh

# 2. Full measurement (20 instances for accurate average)
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/batch_20

# 3. View results
cat results/metrics/batch_20/analysis/cost_visualization.txt
```

## What Gets Measured?

The offline pipeline processes historical CI failures to build memory. Each instance goes through:

1. **Decomposition** (uses LLM)
   - Parse CI logs
   - Identify failure points
   - Extract verification commands
   
2. **L1 Memory** (minimal LLM)
   - Structure problems
   - Link to files
   
3. **L2 Memory** (uses LLM)
   - Generate repair strategies
   - Identify patterns
   
4. **L3 Memory** (uses LLM)
   - Extract universal patterns
   - Cross-repo generalization

**Total per instance**: ~100K-150K tokens, ~$0.05-$0.10 (DeepSeek V4 Flash)

## Step 1: Test on 3 Instances (1 minute)

```bash
# Run demo
bash scripts/demo_cost_tracking.sh
```

**Expected output:**
```
Processing 3 instances...
  Instance 102 (flower-dev/flower): $0.067
  Instance 410 (agno-dev/agno): $0.052
  Instance 500 (ray-project/ray): $0.071

TOTALS:
  Total cost: $0.190
  Average cost: $0.063 per instance
  
PROJECTIONS (157 instances):
  Estimated cost: $9.89
```

## Step 2: Measure on 20 Instances (10-15 minutes)

For accurate statistics, process 20 diverse instances:

```bash
# Set up pricing (optional - uses LiteLLM defaults otherwise)
export RUN_METRICS_PRICING_FILE=config/model_pricing.json

# Run batch processing
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/batch_20_deepseek
```

This will:
- Select 20 instances from different repositories
- Process each through the complete pipeline
- Track tokens and costs for each phase
- Calculate averages and project to 157 instances

**Monitoring progress:**
```bash
# In another terminal
tail -f results/metrics/batch_20_deepseek/per_instance/*/cost_time_report.json
```

## Step 3: Analyze Results

```bash
python scripts/analyze_offline_costs.py \
    --batch-summary results/metrics/batch_20_deepseek/batch_summary.json \
    --detailed-tracker-dir results/metrics/batch_20_deepseek/per_instance \
    --output-dir results/metrics/batch_20_deepseek/analysis
```

**View results:**
```bash
# Text report with charts
cat results/metrics/batch_20_deepseek/analysis/cost_visualization.txt

# JSON for programmatic access
cat results/metrics/batch_20_deepseek/batch_summary.json | jq '.averages'

# CSV for spreadsheet
open results/metrics/batch_20_deepseek/batch_costs.csv
```

## Understanding the Output

### batch_summary.json

Key fields:
- `averages.avg_cost_usd_per_instance` - Average cost per instance
- `projections.projected_157_instances_cost_usd` - Total cost for all 157
- `totals.total_input_tokens` - Total tokens sent to LLM
- `totals.total_output_tokens` - Total tokens generated

### phase_breakdown.csv

Shows cost by pipeline phase:
- Which phases are most expensive?
- Where can we optimize?
- Token distribution across phases

### Example Output

```
AVERAGES (per successful instance):
  Average cost: $0.065029
  Average input tokens: 131,579
  Average output tokens: 26,316
  Average wall time: 45.32s

PROJECTIONS (157 instances):
  Projected total cost: $10.21
  Projected input tokens: 20,657,774
  Projected output tokens: 4,131,579
  
COST BREAKDOWN BY PHASE:
  decomposition:  $0.0225 (35%)  ████████████████
  build_l2:       $0.0190 (29%)  █████████████
  build_l3:       $0.0160 (25%)  ███████████
  build_l1:       $0.0060 (11%)  ████
```

## Comparing Models

Test different models to find the best cost/performance:

```bash
# DeepSeek V4 Flash (fast, cheap)
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/deepseek

# GLM 5.2 (balanced)
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model glm5.2 \
    --output-dir results/metrics/glm

# Compare
echo "DeepSeek cost:"
jq '.averages.avg_cost_usd_per_instance' results/metrics/deepseek/batch_summary.json

echo "GLM cost:"
jq '.averages.avg_cost_usd_per_instance results/metrics/glm/batch_summary.json
```

## Cost Optimization Tips

### 1. Enable Prompt Caching

For repeated contexts (same repo, similar failures):

```python
# Already enabled in LitellmModel
# Reduces costs by 50-70% for cached tokens
```

Check cache hit rate:
```bash
jq '.totals.total_cached_input_tokens / .totals.total_input_tokens' \
    results/metrics/batch_20/batch_summary.json
```

### 2. Skip L3 for Single-Repo

If you're only building memory for one repository:

```bash
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --skip-l3 \  # Saves ~25% cost
    --output-dir results/metrics/batch_no_l3
```

### 3. Batch Processing

Process multiple instances together to benefit from caching:

```bash
# Better: Process 20 together (cache hits)
python scripts/batch_offline_metrics.py --num-instances 20 ...

# Worse: Process 20 separately (no cache hits)
for i in {1..20}; do
    python scripts/offline_pipeline_with_metrics.py ...
done
```

## Troubleshooting

### "Unpriced API calls" Warning

If you see unpriced calls in the report:

```bash
# Set custom pricing file
export RUN_METRICS_PRICING_FILE=config/model_pricing.json

# Or update model registry
# Edit: utilities/model_registry.py
```

### High Costs

If costs are higher than expected:

1. **Check token counts**: Are logs unusually large?
   ```bash
   jq '.per_instance_details[] | 
       {id: .instance_id, tokens: .total_input_tokens}' \
       results/metrics/batch_20/batch_summary.json
   ```

2. **Profile phases**: Which phase is expensive?
   ```bash
   cat results/metrics/batch_20/analysis/phase_breakdown.csv
   ```

3. **Verify caching**: Should be >50% for repeated contexts
   ```bash
   jq '.totals.total_cached_input_tokens, .totals.total_input_tokens' \
       results/metrics/batch_20/batch_summary.json
   ```

### Failed Instances

If some instances fail:

```bash
# Check failure reasons
jq '.per_instance_details[] | 
    select(.last_status != "completed") | 
    {id: .instance_id, status: .last_status}' \
    results/metrics/batch_20/batch_summary.json

# View logs for specific instance
cat results/metrics/batch_20/per_instance/102/run_metrics.json
```

## Next Steps

After measuring on 20 instances:

### Production Run (All 157 Instances)

```bash
# Full CIBench processing
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 157 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/full_cibench_157

# Monitor progress (in another terminal)
watch -n 60 'jq ".summary" results/metrics/full_cibench_157/batch_summary.json'
```

### Cost Reporting for Paper

```bash
# Generate final numbers for paper
python scripts/analyze_offline_costs.py \
    --batch-summary results/metrics/full_cibench_157/batch_summary.json \
    --detailed-tracker-dir results/metrics/full_cibench_157/per_instance \
    --output-dir results/paper_ready

# Extract key numbers
jq '{
    total_cost: .totals.total_cost_usd,
    avg_cost: .averages.avg_cost_usd_per_instance,
    total_tokens: (.totals.total_input_tokens + .totals.total_output_tokens),
    cache_rate: (.totals.total_cached_input_tokens / .totals.total_input_tokens)
}' results/paper_ready/cost_analysis.json
```

## FAQ

**Q: How long does it take to process 20 instances?**

A: ~10-15 minutes on a standard machine with DeepSeek V4 Flash.

**Q: Can I interrupt and resume?**

A: Yes! The batch processor saves results per-instance. Re-run with the same output directory:
```bash
python scripts/batch_offline_metrics.py \
    --output-dir results/metrics/batch_20  # Same directory
    # ... other args
```

**Q: How do I export for a spreadsheet?**

A: CSV files are automatically generated:
```bash
open results/metrics/batch_20/batch_costs.csv
open results/metrics/batch_20/analysis/phase_breakdown.csv
```

**Q: What if my dataset has a different format?**

A: Update `load_dataset()` in `batch_offline_metrics.py` to handle your format.

## Support

See [OFFLINE_COST_TRACKING.md](./OFFLINE_COST_TRACKING.md) for comprehensive documentation.
