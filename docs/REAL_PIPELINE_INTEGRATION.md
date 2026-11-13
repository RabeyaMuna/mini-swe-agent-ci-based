# Real Pipeline Integration - Cost Tracking

## Overview

The cost tracking scripts now use your **ACTUAL EXISTING MODULES** for:

1. ✅ **Log details extraction** (via `build_benchmark_ci_context` in `decompose_issue`)
2. ✅ **CI workflow verification** (via `analyze_workflow_from_benchmark` in `decompose_issue`)
3. ✅ **Problem decomposition** (via `decompose_issue` from `backward_decomposition`)
4. ✅ **L1 memory building** (via `generate_l1_from_decomposed_problems` from `build_memory`)
5. ✅ **L2 memory building** (via `build_l2_memory` from `build_memory`)
6. ✅ **L3 memory building** (via `build_l3_memory` from `build_memory`)

## Real Pipeline Flow

### Phase 1: Decomposition (with CI Analysis)

**Function:** `decompose_issue(issue, llm, output_dir)`  
**Location:** `backward_decomposition/decompose_ci_failure.py`

This single function call performs THREE steps:

1. **`build_benchmark_ci_context`** - Extracts log details and verifies CI workflow
   - Calls `_run_log_analysis()` - Parses CI logs
   - Calls `analyze_workflow_from_benchmark()` - Verifies workflow structure
   - Returns: Log analysis, workflow info, validation sequence

2. **`analyze_diff_chunks`** - Decomposes the problem
   - Takes: Issue + benchmark context + LLM
   - Returns: Atomic problems with dependencies

3. **Clustering & Merging** - Optimizes problems
   - `_cluster_and_merge_problems()` - Deduplicates similar problems
   - `_reorder_by_repair_trajectory()` - Orders by dependency

**Output:**
```python
{
    "original_issue_id": "102",
    "repo": "flower-dev/flower",
    "problems": [
        {
            "problem_id": 1,
            "problem": "Description...",
            "root_cause": "Analysis...",
            "fix_strategy": "How to fix...",
            "files": ["file1.py", "file2.py"],
            "validation_cmd": "pytest ...",
            "enabled": True
        },
        # ... more problems
    ],
    "benchmark_ci_context": {
        "workflow_path": ".github/workflows/test.yml",
        "validation_sequence": [...],
        "error_types": [...]
    }
}
```

### Phase 2: L1 Memory Building

**Function:** `generate_l1_from_decomposed_problems(decomposed_issue)`  
**Location:** `build_memory/build_l1.py`

**What it does:**
- Takes decomposed problems
- Structures them into L1 format (concrete, repo-specific)
- Links problems to files and dependencies

**Minimal LLM usage** - mostly data transformation

### Phase 3: L2 Memory Building

**Function:** `build_l2_memory(l1_memory, llm)`  
**Location:** `build_memory/build_l2.py`

**What it does:**
- Analyzes L1 problems
- **Uses LLM** to generate repair strategies
- Identifies automation tools (ruff, black, etc.)
- Creates reusable patterns

**Significant LLM usage** - strategy generation

### Phase 4: L3 Memory Building

**Function:** `build_l3_memory(l1_memory, l2_memory, llm)`  
**Location:** `build_memory/build_l3.py`

**What it does:**
- Analyzes L1 + L2 memory
- **Uses LLM** to extract universal patterns
- Creates cross-repo generalizations

**Significant LLM usage** - pattern extraction

## How Metrics Are Tracked

### LLM Integration

The `LitellmModel` class (in `utilities/llm_model.py`) already has built-in metrics support:

```python
llm = LitellmModel(
    model_name="deepseek-v4-flash",
    metrics_recorder=metrics_recorder  # RunMetricsRecorder instance
)
```

Every LLM call automatically:
1. Records tokens (input, output, cached)
2. Calculates costs
3. Tracks duration
4. Logs to `run_metrics.json`

### Phase Tracking

By setting `llm._current_phase`, we can track which phase each API call belongs to:

```python
llm._current_phase = "decomposition_with_ci_analysis"
# ... LLM calls here are tagged as "decomposition_with_ci_analysis"

llm._current_phase = "build_l2"
# ... LLM calls here are tagged as "build_l2"
```

This allows us to see cost breakdowns like:

| Phase | Calls | Cost | Tokens |
|-------|-------|------|--------|
| decomposition_with_ci_analysis | 1 | $0.023 | 120K |
| build_l2 | 1 | $0.019 | 95K |
| build_l3 | 1 | $0.016 | 80K |

## Code Changes Made

### 1. `scripts/offline_pipeline_with_metrics.py`

**Before (placeholder):**
```python
def phase_decomposition(self):
    # Placeholder - just creates stub data
    decomposed_issue = {
        "repo": self.repo,
        "problems": []
    }
    return decomposed_issue
```

**After (real integration):**
```python
def phase_decomposition(self):
    # Uses REAL decompose_issue function
    decomposed_result = decompose_issue(
        issue=self.issue,  # Full issue from dataset
        llm=self.llm,      # LLM with metrics tracking
        output_dir=str(self.output_dir),
    )
    return decomposed_result
```

**Key changes:**
- ✅ Takes full `issue` dict from dataset (not just ID and repo)
- ✅ Uses real `decompose_issue()` function
- ✅ Passes `llm` with metrics recorder
- ✅ Returns actual decomposed problems

### 2. `scripts/batch_offline_metrics.py`

**Key change:**
- Added `--dataset` parameter to `offline_pipeline_with_metrics.py` call
- Each instance is now processed with full issue data from dataset

```python
cmd = [
    sys.executable,
    "scripts/offline_pipeline_with_metrics.py",
    "--instance-id", instance_id,
    "--dataset", str(dataset_path),  # ← NEW: Pass dataset
    "--model", model,
    "--output-dir", str(instance_output),
]
```

## Usage Example

### Processing a Single Instance

```bash
# Using local dataset
python scripts/offline_pipeline_with_metrics.py \
    --dataset data/cibench_test.json \
    --instance-id 102 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/instance_102

# Using HuggingFace dataset
python scripts/offline_pipeline_with_metrics.py \
    --use-huggingface \
    --instance-id 102 \
    --model deepseek-v4-flash \
    --output-dir results/metrics/instance_102
```

### What Actually Happens

1. **Loads full issue data** from dataset (not just ID)
   - All CI logs
   - Ground truth diff
   - Workflow information
   - Changed files

2. **Phase 1: decompose_issue()**
   - Analyzes CI logs (real log parsing)
   - Verifies workflow (real workflow analysis)
   - Decomposes into problems (real LLM decomposition)
   - **Metrics tracked:** tokens, cost, duration

3. **Phase 2: generate_l1_from_decomposed_problems()**
   - Structures problems into L1 format
   - Minimal LLM usage
   - **Metrics tracked:** tokens, cost, duration

4. **Phase 3: build_l2_memory()**
   - Generates repair strategies
   - Uses real LLM prompt from `prompt_template/memory_build.py`
   - **Metrics tracked:** tokens, cost, duration

5. **Phase 4: build_l3_memory()**
   - Extracts universal patterns
   - Uses real LLM prompt from `prompt_template/memory_build.py`
   - **Metrics tracked:** tokens, cost, duration

6. **Saves all metrics** to:
   - `run_metrics.json` - Raw ledger
   - `cost_time_report.json` - Formatted report
   - `detailed_usage_{instance_id}.json` - Step-by-step breakdown

## Verification

### Check Real Integration

```bash
# Process one instance
python scripts/offline_pipeline_with_metrics.py \
    --dataset data/cibench_test.json \
    --instance-id 102 \
    --model deepseek-v4-flash \
    --output-dir results/test_real_integration

# Verify decomposed output has real data
jq '.problems | length' results/test_real_integration/decomposed_102.json
# Should show: 3-10 problems (real decomposition)

# Verify L1 memory has real problems
jq '.[0].problems | length' results/test_real_integration/memory/failure_memory.json
# Should match decomposed problems count

# Verify metrics tracked phases
jq '.instances."102".attempts[0].api_calls[] | .phase' \
    results/test_real_integration/run_metrics.json
# Should show: "decomposition_with_ci_analysis", "build_l2", "build_l3"
```

### Expected Output

**decomposed_102.json:**
```json
{
  "original_issue_id": "102",
  "repo": "flower-dev/flower",
  "problems": [
    {
      "problem_id": 1,
      "problem": "ModuleNotFoundError when importing ray[...REAL PROBLEM...]",
      "root_cause": "Missing ray dependency[...REAL ANALYSIS...]",
      "fix_strategy": "Add ray to requirements[...REAL STRATEGY...]",
      "files": ["src/py/flwr/simulation/ray_transport/ray_client_proxy.py"],
      "validation_cmd": "python -c 'import ray'",
      "enabled": true
    },
    // ... more REAL problems
  ],
  "benchmark_ci_context": {
    "workflow_path": ".github/workflows/framework.yml",
    "validation_sequence": [...REAL VALIDATION SEQUENCE...],
    "error_types": [...REAL ERROR TYPES...]
  }
}
```

**run_metrics.json:**
```json
{
  "instances": {
    "102": {
      "attempts": [
        {
          "api_calls": [
            {
              "phase": "decomposition_with_ci_analysis",
              "model": "deepseek-v4-flash",
              "input_tokens": 15234,
              "output_tokens": 3421,
              "cost_usd": 0.003142,
              "duration_seconds": 4.23
            },
            {
              "phase": "build_l2",
              "model": "deepseek-v4-flash",
              "input_tokens": 12893,
              "output_tokens": 2134,
              "cost_usd": 0.002401,
              "duration_seconds": 3.12
            },
            {
              "phase": "build_l3",
              "model": "deepseek-v4-flash",
              "input_tokens": 10234,
              "output_tokens": 1892,
              "cost_usd": 0.001987,
              "duration_seconds": 2.84
            }
          ]
        }
      ]
    }
  }
}
```

## Comparison: Placeholder vs Real

### Placeholder (Old)

- ❌ No real log analysis
- ❌ No real workflow verification
- ❌ No real decomposition
- ❌ Stub data only
- ❌ Not representative of actual costs

### Real Integration (New)

- ✅ Real CI log parsing (`_run_log_analysis`)
- ✅ Real workflow verification (`analyze_workflow_from_benchmark`)
- ✅ Real problem decomposition (`decompose_issue`)
- ✅ Real L1/L2/L3 building (`build_memory` module)
- ✅ **Accurate cost measurements**

## Next Steps

1. **Run the demo** to verify it works:
   ```bash
   bash scripts/demo_cost_tracking.sh
   ```

2. **Process 20 instances** for accurate averages:
   ```bash
   python scripts/batch_offline_metrics.py \
       --dataset data/cibench_test.json \
       --num-instances 20 \
       --model deepseek-v4-flash \
       --output-dir results/metrics/real_pipeline_20
   ```

3. **Analyze the results**:
   ```bash
   python scripts/analyze_offline_costs.py \
       --batch-summary results/metrics/real_pipeline_20/batch_summary.json \
       --detailed-tracker-dir results/metrics/real_pipeline_20/per_instance \
       --output-dir results/metrics/real_pipeline_20/analysis
   ```

4. **Check phase breakdown**:
   ```bash
   cat results/metrics/real_pipeline_20/analysis/phase_breakdown.csv
   ```

This will show you the **REAL token usage and costs** for:
- Log details extraction
- CI workflow verification
- Problem decomposition
- L1, L2, L3 memory building

All using your actual production modules!
