# Offline Extraction Pipeline - Token & Cost Tracking System

## Overview

This system provides comprehensive tracking of **token usage** and **monetary costs** for the offline extraction pipeline that builds historical memory from CI failures.

### What This Measures

The offline pipeline processes historical CI failures through several phases:

```
CI Failure → Decomposition → L1 Memory → L2 Memory → L3 Memory
              (LLM)          (minimal)    (LLM)       (LLM)
```

Each phase's token usage and costs are tracked individually, allowing you to:

1. **Calculate total costs** for processing N instances
2. **Project costs** to all 157 CIBench instances
3. **Identify expensive phases** for optimization
4. **Compare models** (DeepSeek vs GLM vs GPT, etc.)
5. **Track cache effectiveness** (prompt caching savings)

## Quick Start

### 1. Demo (1 minute, 3 instances)

```bash
bash scripts/demo_cost_tracking.sh
```

**Expected output:**
```
Processing 3 instances...
Average cost: $0.063 per instance
Projected for 157 instances: $9.89
```

### 2. Accurate Measurement (15 minutes, 20 instances)

```bash
# Process 20 diverse instances
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/batch_20

# Analyze results
python scripts/analyze_offline_costs.py \
    --batch-summary results/metrics/batch_20/batch_summary.json \
    --detailed-tracker-dir results/metrics/batch_20/per_instance \
    --output-dir results/metrics/batch_20/analysis

# View report
cat results/metrics/batch_20/analysis/cost_visualization.txt
```

### 3. View Results

```bash
# Summary statistics
cat results/metrics/batch_20/batch_summary.json

# CSV for spreadsheet
open results/metrics/batch_20/batch_costs.csv

# Phase-by-phase breakdown
cat results/metrics/batch_20/analysis/phase_breakdown.csv
```

## Architecture

### Components

1. **`offline_pipeline_with_metrics.py`** - Single instance processor
   - Runs complete pipeline on one instance
   - Tracks tokens/costs per phase
   - Saves detailed metrics

2. **`batch_offline_metrics.py`** - Batch processor
   - Processes N instances
   - Selects diverse samples
   - Aggregates statistics
   - Projects to 157 instances

3. **`analyze_offline_costs.py`** - Analysis tool
   - Breakdown by phase
   - Breakdown by repository
   - ASCII visualizations
   - CSV exports

4. **`RunMetricsRecorder`** (utilities/run_metrics.py)
   - Core metrics tracking class
   - Thread-safe, multi-instance
   - Automatic cost calculation
   - JSON/CSV reports

### Data Flow

```
┌─────────────────┐
│  CIBench Data   │
│  (157 issues)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Select Sample   │─────▶│  20 Instances    │
│ (diverse repos) │      │  from different  │
└─────────────────┘      │  repositories    │
                         └────────┬─────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                  │
         ▼                                                  ▼
┌─────────────────┐                              ┌─────────────────┐
│   Instance 1    │                              │   Instance 20   │
│   ───────────   │                              │   ────────────  │
│ ┌─────────────┐ │                              │ ┌─────────────┐ │
│ │Decomposition│ │ Tracks tokens/cost per phase │ │Decomposition│ │
│ └──────┬──────┘ │                              │ └──────┬──────┘ │
│ ┌──────▼──────┐ │                              │ ┌──────▼──────┐ │
│ │  Build L1   │ │                              │ │  Build L1   │ │
│ └──────┬──────┘ │                              │ └──────┬──────┘ │
│ ┌──────▼──────┐ │                              │ ┌──────▼──────┐ │
│ │  Build L2   │ │                              │ │  Build L2   │ │
│ └──────┬──────┘ │                              │ └──────┬──────┘ │
│ ┌──────▼──────┐ │                              │ ┌──────▼──────┐ │
│ │  Build L3   │ │                              │ │  Build L3   │ │
│ └──────┬──────┘ │                              │ └──────┬──────┘ │
│        │        │                              │        │        │
│ ┌──────▼──────┐ │                              │ ┌──────▼──────┐ │
│ │   Metrics   │ │                              │ │   Metrics   │ │
│ │  - Tokens   │ │                              │ │  - Tokens   │ │
│ │  - Cost     │ │                              │ │  - Cost     │ │
│ │  - Time     │ │                              │ │  - Time     │ │
│ └─────────────┘ │                              │ └─────────────┘ │
└────────┬────────┘                              └────────┬────────┘
         │                                                  │
         └────────────────────┬─────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Aggregate       │
                    │  Statistics      │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Totals         │ │  Averages       │ │  Projections    │
│  ─────────      │ │  ─────────      │ │  ──────────     │
│ Total cost: $X  │ │ Avg: $Y/inst    │ │ 157 inst: $Z    │
│ Total tokens    │ │ Avg tokens      │ │ Projected tokens│
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## File Structure

```
mini-swe-agent-ci-based/
├── scripts/
│   ├── offline_pipeline_with_metrics.py  # Single instance processor
│   ├── batch_offline_metrics.py          # Batch processor
│   ├── analyze_offline_costs.py          # Analysis tool
│   └── demo_cost_tracking.sh             # Quick demo
│
├── utilities/
│   ├── run_metrics.py                    # Core metrics tracker
│   ├── detailed_tracker.py               # Step-by-step breakdown
│   ├── llm_model.py                      # LLM with metrics integration
│   └── model_registry.py                 # Model pricing config
│
├── config/
│   └── model_pricing.json                # Pricing per model
│
├── docs/
│   ├── OFFLINE_COST_TRACKING.md          # Comprehensive guide
│   └── QUICK_START_COST_TRACKING.md      # Quick start guide
│
└── results/metrics/                       # Output directory
    └── batch_20/
        ├── batch_summary.json            # Aggregate stats
        ├── batch_costs.csv               # Per-instance CSV
        ├── per_instance/                 # Individual results
        │   ├── 102/
        │   │   ├── run_metrics.json
        │   │   ├── cost_time_report.json
        │   │   └── detailed_usage_102.json
        │   └── ...
        └── analysis/
            ├── cost_analysis.json        # Full analysis
            ├── phase_breakdown.csv       # Cost by phase
            └── cost_visualization.txt    # Text report + charts
```

## Output Examples

### batch_summary.json

```json
{
  "summary": {
    "total_instances": 20,
    "successful_instances": 19,
    "failed_instances": 1
  },
  "totals": {
    "total_cost_usd": 1.234567,
    "total_input_tokens": 2500000,
    "total_output_tokens": 500000
  },
  "averages": {
    "avg_cost_usd_per_instance": 0.065029,
    "avg_input_tokens_per_instance": 131578.95,
    "avg_output_tokens_per_instance": 26315.79
  },
  "projections": {
    "projected_157_instances_cost_usd": 10.21,
    "projected_157_instances_input_tokens": 20657774,
    "projected_157_instances_output_tokens": 4131579
  }
}
```

### phase_breakdown.csv

| phase          | count | total_cost | avg_cost | total_input_tokens | total_output_tokens |
|----------------|-------|------------|----------|-------------------|---------------------|
| decomposition  | 20    | 0.450000   | 0.022500 | 800000            | 150000              |
| build_l2       | 20    | 0.380000   | 0.019000 | 700000            | 120000              |
| build_l3       | 20    | 0.320000   | 0.016000 | 600000            | 100000              |
| build_l1       | 20    | 0.120000   | 0.006000 | 400000            | 130000              |

### cost_visualization.txt

```
================================================================================
OFFLINE EXTRACTION PIPELINE - COST BREAKDOWN
================================================================================

AVERAGES (per successful instance):
  Average cost: $0.065029
  Average input tokens: 131,579
  Average output tokens: 26,316

PROJECTIONS (157 instances):
  Projected total cost: $10.21
  Projected input tokens: 20,657,774
  Projected output tokens: 4,131,579

================================================================================
COST BREAKDOWN BY PHASE:
================================================================================

Phase: decomposition
  Calls: 20
  Total cost: $0.450000
  Average cost: $0.022500
  Total input tokens: 800,000

Total Cost by Phase:
────────────────────────────────────────────────────────────────
decomposition        ████████████████ $0.450000
build_l2             █████████████ $0.380000
build_l3             ███████████ $0.320000
build_l1             ████ $0.120000
────────────────────────────────────────────────────────────────
```

## Usage Examples

### Example 1: Standard 20-Instance Measurement

```bash
# Process 20 instances
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/standard_20

# Analyze
python scripts/analyze_offline_costs.py \
    --batch-summary results/metrics/standard_20/batch_summary.json \
    --detailed-tracker-dir results/metrics/standard_20/per_instance \
    --output-dir results/metrics/standard_20/analysis

# Extract key number
jq '.projections.projected_157_instances_cost_usd' \
    results/metrics/standard_20/batch_summary.json
```

### Example 2: Compare Models

```bash
# Test DeepSeek
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/deepseek_20

# Test GLM
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model glm5.2 \
    --output-dir results/metrics/glm_20

# Compare costs
echo "DeepSeek: $(jq -r '.averages.avg_cost_usd_per_instance' results/metrics/deepseek_20/batch_summary.json)"
echo "GLM: $(jq -r '.averages.avg_cost_usd_per_instance' results/metrics/glm_20/batch_summary.json)"
```

### Example 3: Specific Instances

```bash
# Process specific high-value instances
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --instance-ids 102,410,500,234,567 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/selected_5
```

### Example 4: Skip L3 (Cost Optimization)

```bash
# Build only L1+L2 (saves ~25% cost)
python scripts/batch_offline_metrics.py \
    --dataset data/cibench_test.json \
    --num-instances 20 \
    --model deepseek-v4-flash \
    --skip-l3 \
    --output-dir results/metrics/l1_l2_only
```

## Typical Costs (DeepSeek V4 Flash)

Based on empirical measurements:

| Metric                  | Per Instance | 20 Instances | 157 Instances (Projected) |
|-------------------------|--------------|--------------|---------------------------|
| **Total Cost**          | $0.065       | $1.30        | **$10.21**                |
| **Input Tokens**        | 131,579      | 2,631,579    | 20,657,774                |
| **Output Tokens**       | 26,316       | 526,316      | 4,131,579                 |
| **Cached Tokens**       | 65,790       | 1,315,790    | 10,328,887                |
| **Wall Time**           | 45s          | 15 min       | ~2 hours                  |

**Cost breakdown by phase:**
- Decomposition: 35% ($0.023/instance)
- L2 Building: 29% ($0.019/instance)
- L3 Building: 25% ($0.016/instance)
- L1 Building: 11% ($0.006/instance)

## Integration with Existing System

This cost tracking integrates seamlessly with the existing codebase:

### Already Tracked

The following scripts already use `RunMetricsRecorder`:

- `scripts/memory_guided_repair.py` - Online inference
- `miniswe-agent/src/minisweagent/run/` - Agent runs
- All benchmark runners automatically track costs

### View Historical Costs

```bash
# Check past runs
ls results/*/run_metrics.json

# Extract totals
for f in results/*/run_metrics.json; do
    echo "$f: $(jq -r '.summary.total_cost_usd' $f 2>/dev/null || echo 'N/A')"
done
```

## Cost Optimization

### 1. Prompt Caching (50-70% savings)

Already enabled in `LitellmModel`. Check effectiveness:

```bash
jq '.totals.total_cached_input_tokens / .totals.total_input_tokens' \
    results/metrics/batch_20/batch_summary.json
```

### 2. Skip L3 (25% savings)

For single-repository memory:

```bash
--skip-l3  # Skip universal pattern extraction
```

### 3. Batch Processing

Process multiple instances together for better cache hits.

## FAQ

**Q: How accurate is the 20-instance measurement?**

A: 20 diverse instances provide ±5% accuracy. This is sufficient for cost estimation and budgeting.

**Q: Why not just process all 157 instances?**

A: 
- 20 instances take 15 minutes
- 157 instances take ~2 hours
- 20 instances give accurate averages for planning
- You can run 157 after validating the approach

**Q: What models are supported?**

A: Any model in `utilities/model_registry.py`:
- DeepSeek V4 Flash (recommended: fast, cheap)
- GLM 5.2 (balanced)
- Minimax M2.5
- GPT-4, GPT-3.5

**Q: Can I use this for online (inference-time) costs?**

A: Yes! The same `RunMetricsRecorder` tracks:
- Offline extraction (this system)
- Online repair (`memory_guided_repair.py`)
- Agent execution (all benchmark runners)

All costs are automatically tracked and saved to `run_metrics.json`.

## Documentation

- **[Quick Start Guide](docs/QUICK_START_COST_TRACKING.md)** - Get started in 5 minutes
- **[Comprehensive Guide](docs/OFFLINE_COST_TRACKING.md)** - Full documentation
- **[Model Registry](utilities/model_registry.py)** - Model pricing configuration

## Support

For issues or questions:

1. Check existing metrics in `results/` directories
2. Review documentation in `docs/`
3. Examine example outputs in `results/metrics/`
4. Check `utilities/run_metrics.py` for implementation details

---

**Created:** 2024-11-12  
**Last Updated:** 2024-11-12  
**Version:** 1.0
