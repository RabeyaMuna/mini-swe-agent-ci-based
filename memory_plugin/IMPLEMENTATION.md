# STAIR Memory Retrieval - Implementation Summary

## Overview

Complete implementation of STAIR-inspired hierarchical memory retrieval with **LLM-driven decision making** for all stages except cosine similarity.

**Supports**:
- ✅ **Baseline mode**: No memory retrieval (for comparison)
- ✅ **Ablation study**: Control memory levels (l1, l1+l2, l1+l2+l3)
- ✅ **Full STAIR**: All 3 levels with cross-level synthesis

## Architecture

### File Organization

```
memory_plugin/
├── stair_retrieval.py (389 lines)  - Main 6-stage pipeline
├── __init__.py                     - Module exports
├── README.md                       - Usage guide
├── example_usage.py                - Examples
└── IMPLEMENTATION.md               - This file

prompt_template/stair_retrieval/
├── stair_prompts.py (384 lines)    - Stages 2-5 prompts
├── repair_plan_prompt.py (188 lines) - Stage 6 repair plan prompts
└── __init__.py                     - Prompt exports

utilities/ (existing, reused)
├── llm_invoker.py                  - Robust LLM calls with retry
├── llm_chunking.py                 - Smart chunking
└── ...                             - Other utilities
```

**Total new code**: ~961 lines (clean, production-ready)

### Integration with Existing Project

✅ **Reuses existing utilities**:
- `utilities.llm_invoker.invoke_llm_with_retry()` - All LLM calls
- `utilities.llm_invoker._load_json_flexible()` - JSON parsing
- `utilities.llm_invoker.STRICT_JSON_RULES` - JSON format rules

✅ **Follows project conventions**:
- Prompts in `prompt_template/` directory
- Separate concerns (prompts vs logic)
- Standard error handling patterns

## 6-Stage Pipeline

### Stage 1: Cosine Similarity Retrieval
**File**: `memory_plugin/stair_retrieval.py`  
**Method**: `_stage_1_retrieval()`  
**Logic**: **Non-LLM** (embedding-based)

Retrieves top-K from each level:
- **L1**: Repo + Workflow filtered (exact match)
- **L2**: Repo filtered only
- **L3**: Universal (no filters)

Uses `sentence-transformers` for embeddings, cosine similarity for ranking.

---

### Stage 2: Common Problem Detection
**File**: `prompt_template/stair_retrieval/stair_prompts.py`  
**Function**: `build_common_detection_prompt()`  
**Logic**: **LLM decides**

**LLM analyzes**:
- L1 problem frequency (40%+ issues OR 3+ occurrences)
- L2 frequency markers ("dependency_version - 4 problems")
- Config-related patterns (.toml, .yml files)

**Output**: JSON with common_problems, config_problems, repair_action for each

---

### Stage 3: Filtering + Dependency Analysis
**File**: `prompt_template/stair_retrieval/stair_prompts.py`  
**Function**: `build_filtering_prompt()`  
**Logic**: **LLM decides**

**LLM tasks**:
1. Filter relevant problems (signal matching)
2. Combine L1/L2/L3 information (root_cause, rationale, signals, repair_actions)
3. Extract dependencies:
   - **Consecutive**: From L2 `causal_chain` (pipeline cascade)
   - **Dependent**: From L3 `dependent_changes` (hidden problems)
4. Provide evidence for selections

**Output**: JSON with problems (WHY/HOW/WHAT rationale), consecutive_sequences, dependencies

---

### Stage 4: Clustering
**File**: `prompt_template/stair_retrieval/stair_prompts.py`  
**Function**: `build_clustering_prompt()`  
**Logic**: **LLM decides**

**LLM groups**:
- Same failure_type + same files → should_merge=true
- Different contexts → should_merge=false

**Output**: JSON with clusters, merge decisions, reasoning

---

### Stage 5: Final Problem Generation
**File**: `prompt_template/stair_retrieval/stair_prompts.py`  
**Function**: `build_final_generation_prompt()`  
**Logic**: **LLM decides**

**LLM generates**:
- Merged problems (if should_merge=true)
- Individual problems (if should_merge=false)
- Complete structure: problem, root_cause, rationale, signals, repair_actions
- Priority ordering: ci_failure (1) > common (2) > dependent (3) > consecutive (4)

**Output**: JSON with final_problems ready for agent execution

---

### Stage 6: Repair Plan Generation (Optional)
**File**: `prompt_template/stair_retrieval/repair_plan_prompt.py`  
**Functions**:
- `build_repair_plan_prompt()` - Single problem plan
- `build_batch_repair_plans_prompt()` - Multi-problem with ordering

**Logic**: **LLM decides**

**LLM generates**:
- Pre-checks (verify before starting)
- Detailed steps (CODE/CONFIG/DEPENDENCY/VALIDATION)
- Validation command
- Rollback plan
- Risk assessment
- Execution order (for batch)

**Output**: JSON with repair_plan, steps, validation, rollback

---

## LLM Decision Points

| Stage | What LLM Decides | Prompt |
|-------|-----------------|--------|
| 2 | Common/frequent problems, repair actions | `build_common_detection_prompt()` |
| 3 | Relevance, dependencies, signal matches | `build_filtering_prompt()` |
| 4 | Clustering, merge decisions | `build_clustering_prompt()` |
| 5 | Final structure, priorities | `build_final_generation_prompt()` |
| 6 | Repair steps, execution order | `build_repair_plan_prompt()` |

**Only Stage 1 is non-LLM** (cosine similarity).

---

## Key Features

### 1. Cross-Level Information Synthesis

Problems combine information from **all 3 levels**:

```json
{
  "root_cause": "Synthesized from L1 (concrete) + L2 (pattern) + L3 (universal)",
  "rationale": {
    "why": "Universal principle from L3",
    "how": "Pattern/strategy from L2",
    "what": "Concrete instance from L1"
  },
  "signals": {
    "error_signals": ["from L1/L2/L3"],
    "config_signals": ["from L1/L2/L3"],
    "match_evidence": "Why these match current failure"
  },
  "repair_actions": {
    "strategy": "High-level approach from L2/L3",
    "steps": ["Concrete steps from L1"],
    "files": ["from L1"],
    "validation_cmd": "from L1/L2"
  }
}
```

### 2. Dependency Detection

**Consecutive** (Pipeline cascade):
```
Fix dependency → enables type check → reveals formatting
```
From L2 `causal_chain`

**Dependent** (Hidden problems):
```
Fix plugin removal → surfaces missing return types
```
From L3 `dependent_changes`

### 3. Evidence-Based Selection

LLM must cite:
- Which signals match
- Which levels provided info
- Why problem is relevant

### 4. Robust Error Handling

All LLM calls use `invoke_llm_with_retry()`:
- Automatic retry on rate limits (exponential backoff)
- Timeout handling (60-90s per stage)
- JSON parsing with multiple fallbacks
- Empty response handling

---

## Usage Examples

### Full STAIR (Default)

```python
from memory_plugin import STAIRRetrieval
import openai

client = openai.OpenAI(api_key="...")

# Full STAIR: L1+L2+L3
retrieval = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    memory_levels="l1+l2+l3"  # or omit (default)
)

ci_failure = {
    'repo': 'CI-Repair/flower',
    'workflow': '.github/workflows/framework.yml',
    'problem_statement': 'Type checking failed with DTypeLike error',
    'error_signals': ['Cannot resolve type annotation "DTypeLike"'],
    'config_signals': ['numpy.typing.mypy_plugin removed'],
    'failure_type': 'type_checking'
}

# Get problems + repair plans
result = retrieval.retrieve(ci_failure, top_k=5, generate_plans=True)

# Problems ready for agent
for problem in result['problems']:
    print(problem['problem'])
    print(problem['repair_actions']['steps'])

# Repair plans with detailed steps
for plan in result['repair_plans']:
    for step in plan['repair_plan']['steps']:
        print(f"{step['step_number']}. {step['action_type']}: {step['description']}")
```

### Baseline Mode (No Memory)

```python
# Baseline: skip memory retrieval
baseline = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    baseline_mode=True
)

result = baseline.retrieve(ci_failure)
# Returns: {'problems': [], 'metadata': {'mode': 'baseline', ...}}
```

### Ablation Study

```python
# L1 only (repo+workflow specific)
l1_only = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    memory_levels="l1"
)

# L1+L2 (add repo patterns)
l1_l2 = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    memory_levels="l1+l2"
)

# Compare results
baseline_result = baseline.retrieve(ci_failure)
l1_result = l1_only.retrieve(ci_failure)
l1_l2_result = l1_l2.retrieve(ci_failure)
full_result = retrieval.retrieve(ci_failure)

print(f"Baseline: {len(baseline_result['problems'])} problems")
print(f"L1 only: {len(l1_result['problems'])} problems")
print(f"L1+L2: {len(l1_l2_result['problems'])} problems")
print(f"Full: {len(full_result['problems'])} problems")
```

---

## Design Principles

1. ✅ **Separation of concerns**: Prompts separate from logic
2. ✅ **Reuse utilities**: No duplicate LLM/JSON handling
3. ✅ **LLM-driven**: All decisions via prompts, not rules
4. ✅ **Simple pipeline**: Linear flow, no nested calls
5. ✅ **Production ready**: Error handling, retry logic, timeouts
6. ✅ **Evidence-based**: LLM cites sources and matches

---

## Summary

**What's implemented**:
- ✅ 6-stage pipeline (retrieval → common → filter → cluster → generate → plan)
- ✅ All prompts in `prompt_template/stair_retrieval/`
- ✅ Reuses existing `utilities/llm_invoker.py`
- ✅ LLM makes ALL decisions (Stages 2-6)
- ✅ Cross-level synthesis (L1+L2+L3)
- ✅ Dependency detection (consecutive + dependent)
- ✅ Repair plan generation
- ✅ **Baseline mode** (no memory, for comparison)
- ✅ **Ablation study support** (control levels: l1, l1+l2, l1+l2+l3)
- ✅ Clean, simple, production-ready code

**Experiment Support**:
- **Baseline**: `baseline_mode=True` → No memory, empty results
- **L1 only**: `memory_levels="l1"` → Repo+workflow specific
- **L1+L2**: `memory_levels="l1+l2"` → Add repo patterns
- **L1+L2+L3**: `memory_levels="l1+l2+l3"` → Full STAIR (default)

**Lines of code**:
- `stair_retrieval.py`: 389 lines
- `stair_prompts.py`: 384 lines
- `repair_plan_prompt.py`: 188 lines
- **Total**: ~961 lines

No nested LLM calls, no rule-based decisions (except Stage 1), all prompts visible and editable.
