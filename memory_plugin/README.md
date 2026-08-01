# STAIR Memory Retrieval

STAIR-inspired hierarchical memory retrieval for CI repair with LLM-driven decision making.

## Architecture

- **Stage 1**: Cosine similarity retrieval (L1/L2/L3)
- **Stage 2**: LLM detects common/frequent problems
- **Stage 3**: LLM filters relevant problems + dependency analysis
- **Stage 4**: LLM clusters similar problems
- **Stage 5**: LLM generates final structured problem list
- **Stage 6**: LLM generates repair plans (optional)

## File Structure

```
memory_plugin/
├── stair_retrieval.py          - Main pipeline (uses utilities)
├── __init__.py                 - Module exports
├── README.md                   - This file
└── example_usage.py            - Usage examples

prompt_template/stair_retrieval/
├── stair_prompts.py            - Stages 2-5 prompts
├── repair_plan_prompt.py       - Stage 6 repair plan prompts
└── __init__.py                 - Prompt exports

utilities/
├── llm_invoker.py              - Robust LLM calls with retry
├── llm_chunking.py             - Smart chunking
└── ...                         - Other shared utilities
```

## Usage

### Basic Usage

```python
from memory_plugin import STAIRRetrieval
import openai

# Initialize with LLM client (full STAIR: L1+L2+L3)
client = openai.OpenAI(api_key="...")
retrieval = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    memory_levels="l1+l2+l3"  # Default: all levels
)

# Query
ci_failure = {
    'repo': 'owner/repo',
    'workflow': '.github/workflows/ci.yml',
    'problem_statement': 'Type checking failed',
    'error_signals': ['Cannot resolve type annotation DTypeLike'],
    'config_signals': ['mypy removed from pyproject.toml'],
    'failure_type': 'type_checking'
}

# Get problems only
result = retrieval.retrieve(ci_failure, top_k=5)

# Or get problems + repair plans
result = retrieval.retrieve(ci_failure, top_k=5, generate_plans=True)

# Use results
for problem in result['problems']:
    print(f"Problem: {problem['problem']}")
    print(f"Root Cause: {problem['root_cause']}")
    print(f"Steps: {problem['repair_actions']['steps']}")

# If repair plans generated
if 'repair_plans' in result:
    for plan in result['repair_plans']:
        print(f"Repair Plan: {plan['repair_plan']['summary']}")
        for step in plan['repair_plan']['steps']:
            print(f"  {step['step_number']}. {step['description']}")
```

### Baseline Mode (No Memory)

For comparison experiments:

```python
# Baseline: no memory retrieval
baseline = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    baseline_mode=True  # Skip all memory
)

result = baseline.retrieve(ci_failure)
# Returns: {'problems': [], 'metadata': {'mode': 'baseline', ...}}
```

### Ablation Study (Control Memory Levels)

For measuring impact of each level:

```python
# Ablation: Only L1 (repo+workflow specific)
l1_only = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    memory_levels="l1"
)

# Ablation: L1 + L2 (repo patterns)
l1_l2 = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    memory_levels="l1+l2"
)

# Full STAIR: L1 + L2 + L3 (universal patterns)
full = STAIRRetrieval(
    memory_dir='data/fwr_trs',
    llm_client=client,
    memory_levels="l1+l2+l3"  # or just omit (default)
)

# Compare results
baseline_result = baseline.retrieve(ci_failure)
l1_result = l1_only.retrieve(ci_failure)
l1_l2_result = l1_l2.retrieve(ci_failure)
full_result = full.retrieve(ci_failure)

print(f"Baseline: {len(baseline_result['problems'])} problems")
print(f"L1 only: {len(l1_result['problems'])} problems")
print(f"L1+L2: {len(l1_l2_result['problems'])} problems")
print(f"L1+L2+L3: {len(full_result['problems'])} problems")
```

## LLM Client Support

Works with any client that `utilities.llm_invoker` supports:
- OpenAI (`openai.OpenAI()`)
- Anthropic (`anthropic.Anthropic()`)
- Custom clients with compatible interface

All LLM calls use `invoke_llm_with_retry()` with:
- Automatic retry on rate limits
- Timeout handling
- JSON parsing with multiple fallbacks
- Robust error handling

## Prompts

All prompts are in `prompt_template/stair_retrieval/`:

- **Stage 2**: `build_common_detection_prompt()` - Cross-issue common problem detection
- **Stage 3**: `build_filtering_prompt()` - Relevance filtering + dependency analysis
- **Stage 4**: `build_clustering_prompt()` - Similar problem clustering
- **Stage 5**: `build_final_generation_prompt()` - Final structured output
- **Stage 6**: `build_repair_plan_prompt()` - Detailed repair plans

## Decision Making

**LLM decides** (Stages 2-6):
- Which problems are common/frequent
- Which problems are relevant to current failure
- Dependencies (consecutive, dependent problems)
- How to cluster problems
- Final problem structure
- Repair plan steps

**Non-LLM** (Stage 1 only):
- Cosine similarity retrieval
- Top-k selection
