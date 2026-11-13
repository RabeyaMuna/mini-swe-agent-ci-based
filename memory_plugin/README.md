# Memory Plugin - Agent-Agnostic CI Repair Memory

## Overview

The memory plugin provides **hierarchical memory retrieval** for CI repair agents. It retrieves similar past fixes from L1/L2/L3 memory and detects common repository/workflow patterns using **flattening + clustering + LLM validation**.

## Key Features

OK **Agent-agnostic**: Works with any CI repair agent (Codex, OpenDevin, etc.)  
OK **Hierarchical memory**: L1 (repo+workflow), L2 (repo), L3 (universal)  
OK **Common pattern detection**: Finds recurring problems across issues  
OK **Frequency-based**: Counts distinct CI failures, not problem instances  
OK **LLM-driven**: Dynamic decisions for filtering, validation, and synthesis  
OK **Ablation support**: Control which levels to use (baseline, L1, L1+L2, L1+L2+L3)  
OK **Target-evidence gate**: Withholds recurring/dependency hypotheses that are not supported by the current target failure evidence
OK **Failure-state source gate**: Confirms a remembered defect in the checked-out target before repair

### Target-evidence policy

Problems decomposed from the current CI log remain actionable. Problems recalled
only from historical dependency chains or recurring patterns are treated as
hypotheses: they must match the target's observed file, failed tool/failure type,
and error evidence or subsequently pass the failure-state source gate before
they are sent to the repair agent. Unsupported candidates are returned in
`rejected_memory_problems` for audit and source review.

The gate is enabled by default. Set `RECAP_ENABLE_EVIDENCE_GATE=0` only for a
controlled ablation.

After the failing commit is checked out, a second read-only gate resolves every
remembered candidate's claimed files. It first runs a bounded, non-mutating
verification command when the candidate provides a safe `ruff`, `mypy`,
`pyright`, or targeted `pytest` invocation. A passing command rejects the
remembered failure; a non-zero result confirms it only when the recorded failure
signal appears in the output. Missing tools, setup errors, unrelated failures,
unsafe commands, and unavailable commands fall back to failure-state source
review. Only `confirmed` memory candidates reach the repair agent. Missing locations,
contradicted/already-fixed code, ambiguous evidence, and runtime-only claims are
saved in `source_rejected_memory_problems` but withheld from repair. Problems
directly observed in the target CI log do not need this historical-memory check.
This second gate also reviews hidden/dependency candidates absent from the log,
because an earlier CI failure may have prevented them from executing.
Set `RECAP_ENABLE_SOURCE_VALIDATION=0` only for a controlled ablation.

The production handoff exposes this operation explicitly as
`filter_selected_memory_problems(...)`. It runs after checkout and before agent
construction. Its output `problems` contains only `observed` or `confirmed`
items; rejected and unverifiable recurring/dependency candidates are moved to
`source_rejected_memory_problems`. ReCAP's final problem ordering is then applied
to this filtered set, never to the unverified candidate set.

### Compatibility rollback

Set `RECAP_LEGACY_MEMORY_FLOW=1` to restore the pre-gate behavior without
reverting files. In this mode the plugin uses the original repo/workflow-only
retrieval query and bypasses both the target-evidence gate and the failure-state
source/command gate. This switch is intended for rollback and controlled A/B
comparison while the new filter is validated end to end.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Plugin API                        │
│  retrieve(ci_failure, verification, metadata)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  STAIR Retrieval Pipeline                   │
│                                                             │
│  1. Cosine Search (L1/L2/L3 top-k)                         │
│  2. LLM Relevance Filter                                   │
│  3. Extract Consecutive Problems                           │
│  4. Common Pattern Detection (NEW):                        │
│     a. Flatten ALL L1/L2 problems                          │
│     b. Cluster by similarity                               │
│     c. Count distinct issue_ids (frequency)                │
│     d. LLM validates each group                            │
│  5. Final Clustering (deduplication)                       │
│  6. LLM Synthesis (merge/prioritize)                       │
│  7. Deterministic Ordering                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     Memory Storage                          │
│  - L1: failure_memory.json (repo+workflow specific)        │
│  - L2: repo_memory.json (repo-wide patterns)               │
│  - L3: cross_memory.json (universal patterns)              │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```python
from memory_plugin import MemoryPlugin

# Initialize plugin
plugin = MemoryPlugin(
    memory_root=Path("data/back_trs"),
    result_dir="results/retrieval",
    ablation="L1+L2+L3",  # or "baseline", "L1", "L1+L2"
    top_k=5,
    llm=llm_client,
    enabled=True
)

# Retrieve similar past fixes
retrieval = plugin.retrieve(
    ci_failure={
        "error_context": [...],
        "failure_signals": [...],
        "relevant_files": [...],
        "error_types": [...],
        "workflow_name": "CI",
    },
    verification={
        "validation_sequence": [...]
    },
    issue_metadata={
        "task_id": "123",
        "sha_fail": "abc123",
        "repo": "org/repo"
    }
)

# Use results
problems = retrieval['problems']
metadata = retrieval['metadata']
common_problems = retrieval['common_problems']
```

### Format for Agent Prompt

```python
# Generic markdown format
prompt_text = plugin.format_for_prompt(retrieval)

# Custom format
for problem in retrieval['problems']:
    print(f"Problem: {problem['problem']}")
    print(f"Root Cause: {problem['root_cause']}")
    print(f"Fix Strategy: {problem.get('repair_strategy', {}).get('summary')}")
    print(f"Confidence: {problem['confidence']}")
```

## Common Pattern Detection Algorithm

### The Flattening Approach

Instead of pre-aggregating problems, we:

1. **Flatten ALL problems** from L1/L2 memory (not just top-k)
2. **Cluster by similarity** (deterministic)
3. **Count distinct issue_ids** per cluster (frequency)
4. **LLM validates** each group

### Why Flattening Works

```python
# Example: 3 issues with similar import errors
flattened = [
    {"issue_id": "123", "problem": "import error in test.py"},
    {"issue_id": "123", "problem": "import error in app.py"},   # same issue
    {"issue_id": "456", "problem": "import error in utils.py"},
    {"issue_id": "789", "problem": "import error in config.py"},
]

# After clustering
cluster = {
    "frequency": 3,  # ← 3 distinct issues (not 4 problems!)
    "issue_ids": ["123", "456", "789"],
    "representative_problem": "import error",
    "examples": [...]
}

# LLM sees:
# "This problem appeared in 3 distinct CI failures"
# -> Decides if it's a real recurring pattern
```

### Key Insight: issue_id, not problem_id

We **only track issue_id** because:

- **Frequency = distinct CI failures**, not problem count
- If one issue has same problem 3 times -> counts as 1 issue
- Clustering is problem-level, not problem_id-level
- Simpler data structure

## Retrieval Result Structure

```python
{
    "problems": [
        {
            "problem": "Clear problem description",
            "root_cause": "Why it happened",
            "failure_type": "type_checking|linting|test_failure|...",
            "files": ["file.py"],
            "failure_signals": ["signal"],
            "repair_strategy": {
                "summary": "High-level approach",
                "actions": ["specific action"],
                "pitfalls": ["avoid this"],
                "validation_cmd": "pytest tests/"
            },
            "confidence": "HIGH|MEDIUM|LOW",
            "priority": 1,
            "source": {
                "l1": ["issue_id"],
                "l2": ["issue_id"],
                "l3": ["pattern_id"],
                "common_pattern": true,
                "frequency": 5,
                "coverage": 0.35
            }
        }
    ],
    "metadata": {
        "mode": "memory",
        "enabled_levels": ["L1", "L2", "L3"],
        "retrieved": {"l1": 5, "l2": 5, "l3": 5},
        "common_detected": 3,
        "filtered": 8,
        "consecutive": 2,
        "clusters": 6,
        "final": 4
    },
    "common_problems": [
        {
            "cluster_id": "import_error_pattern",
            "frequency": 5,
            "coverage": 0.35,
            "relevance": "HIGH",
            "representative_problem": "...",
            "examples": [{"issue_id": "123", "problems": ["..."]}]
        }
    ]
}
```

## Configuration

### Ablation Modes

```python
# Baseline (no memory)
ablation="baseline"

# L1 only (same repo + workflow)
ablation="L1"

# L1 + L2 (repo-wide patterns)
ablation="L1+L2"

# L1 + L2 + L3 (universal patterns)
ablation="L1+L2+L3"
```

### Frequency Thresholds

```python
# Small repo (< 5 issues)
min_issues = 2
min_coverage = 0.40  # 40%

# Larger repo (>= 5 issues)
min_issues = 3
min_coverage = 0.30  # 30%
```

### Clustering Threshold

```python
# Similarity threshold for clustering
threshold = 0.68  # 68% similar -> same cluster
```

## Files

- **`memory_plugin.py`**: Main plugin interface
- **`stair_retrieval.py`**: STAIR retrieval implementation
- **`FLATTENING_ALGORITHM.md`**: Detailed algorithm documentation
- **`README.md`**: This file

## Example Integration

```python
# In your CI repair agent
from memory_plugin import MemoryPlugin

class MyRepairAgent:
    def __init__(self, memory_plugin: MemoryPlugin):
        self.memory = memory_plugin

    def repair(self, ci_failure, verification):
        # Retrieve similar past fixes
        retrieval = self.memory.retrieve(
            ci_failure=ci_failure,
            verification=verification
        )

        # Build prompt with memory context
        prompt = f"""
        Current CI Failure:
        {ci_failure}

        {self.memory.format_for_prompt(retrieval)}

        Task: Fix the CI failure.
        """

        # Run repair with memory context
        fix = self.llm(prompt)
        return fix
```

## Testing

```bash
# Test flattening logic
python test_flattening_logic.py

# Test memory plugin
python -m pytest memory_plugin/tests/
```

## Benefits

### 1. Evidence-Based Decisions
- LLM sees actual examples with issue_ids
- Not pre-aggregated summaries

### 2. Frequency Visibility
- LLM knows "5 distinct issues had this problem"
- Clear signal for recurring patterns

### 3. Contextual Merging
- LLM merges based on seeing all variations
- Better than pure text similarity

### 4. Relevance Filtering
- Common patterns marked HIGH/MEDIUM/LOW relevance
- Current failure gets highest priority

### 5. Preserves Granularity
- Each problem keeps its issue_id
- Full traceability back to source

## See Also

- **[FLATTENING_ALGORITHM.md](FLATTENING_ALGORITHM.md)**: Detailed algorithm explanation
- **[../prompt_template/stair_retrieval.py](../prompt_template/stair_retrieval.py)**: LLM prompts
- **[../utilities/llm_invoker.py](../utilities/llm_invoker.py)**: LLM call utilities
