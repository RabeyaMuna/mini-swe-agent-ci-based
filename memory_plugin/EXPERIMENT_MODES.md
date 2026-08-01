# STAIR Memory Retrieval - Experiment Modes

## Overview

The implementation supports **baseline comparison** and **ablation studies** to measure the impact of hierarchical memory abstraction.

## Experiment Configurations

| Mode | Config | L1 Loaded | L2 Loaded | L3 Loaded | Use Case |
|------|--------|-----------|-----------|-----------|----------|
| **Baseline** | `baseline_mode=True` | ❌ | ❌ | ❌ | No memory (comparison baseline) |
| **L1 Only** | `memory_levels="l1"` | ✅ | ❌ | ❌ | Measure L1 contribution |
| **L1+L2** | `memory_levels="l1+l2"` | ✅ | ✅ | ❌ | Measure L2 contribution |
| **Full STAIR** | `memory_levels="l1+l2+l3"` | ✅ | ✅ | ✅ | Full hierarchical abstraction |

## Memory Level Details

### L1 (Repo + Workflow Specific)
- **File**: `failure_memory.json`
- **Scope**: Exact repo AND workflow
- **Content**: Concrete problems with files, fixes, validation commands
- **Retrieval Filter**: `repo=X AND workflow=Y`
- **Example**: Issue #113 in `CI-Repair/flower` + `.github/workflows/framework.yml`

### L2 (Repo Patterns)
- **File**: `repo_memory.json`
- **Scope**: Repo-wide patterns
- **Content**: Repair strategies, causal chains, frequency info
- **Retrieval Filter**: `repo=X` (no workflow)
- **Example**: "dependency_version (poetry) - 4 problems" across all workflows

### L3 (Universal Patterns)
- **File**: `cross_memory.json`
- **Scope**: Cross-repo, abstract patterns
- **Content**: Universal fix approaches, dependent changes, when-to-apply rules
- **Retrieval Filter**: None (universal)
- **Example**: "Plugin removal breaks dependent types" (applies to any repo)

## Code Examples

### 1. Baseline (No Memory)

```python
baseline = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    baseline_mode=True  # Skip all memory
)

result = baseline.retrieve(ci_failure)

# Returns:
# {
#   'problems': [],
#   'metadata': {
#     'mode': 'baseline',
#     'enabled_levels': [],
#     'retrieved': {'l1': 0, 'l2': 0, 'l3': 0},
#     ...
#   }
# }
```

**Use case**: Compare with no memory to measure overall memory impact.

---

### 2. L1 Only

```python
l1_only = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    memory_levels="l1"  # Only load L1
)

result = l1_only.retrieve(ci_failure)

# Stage 1: Retrieves from L1 only (repo+workflow filtered)
# Stages 2-6: LLM works with L1 data only
# Returns problems synthesized from L1
```

**Use case**: Measure contribution of repo+workflow specific fixes.

**Expected**: Concrete, specific problems matching exact workflow.

---

### 3. L1+L2

```python
l1_l2 = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    memory_levels="l1+l2"  # L1 + L2
)

result = l1_l2.retrieve(ci_failure)

# Stage 1: Retrieves from L1 (repo+workflow) + L2 (repo patterns)
# Stages 2-6: LLM synthesizes from L1+L2
#   - Root cause: L1 concrete + L2 pattern
#   - Rationale: L2 HOW + L1 WHAT
#   - Consecutive chains: From L2 causal_chain
```

**Use case**: Measure contribution of repo-level patterns.

**Expected**: More problems, better causal understanding, consecutive chains.

---

### 4. Full STAIR (L1+L2+L3)

```python
full = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    memory_levels="l1+l2+l3"  # Full STAIR (default)
)

result = full.retrieve(ci_failure)

# Stage 1: Retrieves from all 3 levels
# Stages 2-6: LLM synthesizes from L1+L2+L3
#   - Root cause: L1 concrete + L2 pattern + L3 universal
#   - Rationale: L3 WHY + L2 HOW + L1 WHAT
#   - Dependent problems: From L3 dependent_changes
#   - Universal applicability: From L3 when_to_apply
```

**Use case**: Full hierarchical abstraction, maximum coverage.

**Expected**: Most problems, complete WHY/HOW/WHAT rationale, dependent problems.

---

## Ablation Study Script

Run comprehensive comparison:

```bash
python memory_plugin/ablation_experiment.py
```

This runs:
1. Baseline (no memory)
2. L1 only
3. L1+L2
4. Full STAIR (L1+L2+L3)

And outputs:
- Problem counts for each
- Impact of each level (+X problems)
- Detailed problem comparison
- Quality analysis
- Results saved to `ablation_results.json`

## Expected Improvements

Based on STAIR paper and implementation:

| Metric | Baseline | L1 | L1+L2 | L1+L2+L3 |
|--------|----------|----|----|------|
| **Problem Coverage** | 0% | ~40% | ~70% | ~90%+ |
| **Root Cause Depth** | None | Concrete | + Pattern | + Universal |
| **Rationale Completeness** | None | WHAT | + HOW | + WHY |
| **Dependency Detection** | None | None | Consecutive | + Dependent |
| **Cross-Repo Transfer** | None | Low | Medium | High |

## Metadata Analysis

Each result includes tracking metadata:

```json
{
  "metadata": {
    "mode": "baseline" | "memory",
    "enabled_levels": ["l1", "l2", "l3"],
    "retrieved": {
      "l1": 5,  // Top-5 from L1
      "l2": 5,  // Top-5 from L2
      "l3": 5   // Top-5 from L3
    },
    "common_detected": 3,    // Stage 2
    "filtered": 8,           // Stage 3
    "clusters": 5,           // Stage 4
    "final": 4               // Stage 5
  }
}
```

**Analysis points**:
- `retrieved`: How many items passed similarity threshold
- `common_detected`: How many cross-issue patterns found
- `filtered`: How many problems LLM selected as relevant
- `final`: Final problem count after clustering/merging

## Research Questions

Use ablation to answer:

1. **Q**: Does L1 (repo+workflow specific) improve repair success?
   - **Test**: Compare Baseline vs L1 only

2. **Q**: Does L2 (repo patterns) add value beyond L1?
   - **Test**: Compare L1 only vs L1+L2

3. **Q**: Does L3 (universal patterns) enable cross-repo transfer?
   - **Test**: Compare L1+L2 vs Full STAIR
   - **Test on**: Different repo than training data

4. **Q**: Which level contributes most to success?
   - **Test**: Measure delta between each level

5. **Q**: Does hierarchical abstraction improve problem quality?
   - **Test**: Compare root_cause, rationale completeness across levels

## Integration with Repair Agent

```python
# In your agent loop
if config.use_memory:
    if config.ablation_mode == 'baseline':
        retrieval = STAIRRetrieval(baseline_mode=True)
    elif config.ablation_mode == 'l1':
        retrieval = STAIRRetrieval(memory_levels="l1")
    elif config.ablation_mode == 'l1+l2':
        retrieval = STAIRRetrieval(memory_levels="l1+l2")
    else:  # full
        retrieval = STAIRRetrieval(memory_levels="l1+l2+l3")

    result = retrieval.retrieve(ci_failure)
    problems = result['problems']
else:
    problems = []  # No memory fallback
```

## Summary

✅ **4 experiment modes** supported
✅ **Clean separation** - just change one parameter
✅ **Metadata tracking** for analysis
✅ **Ablation script** included
✅ **Production ready** for research experiments
