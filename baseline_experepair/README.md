# ExpeRepair Baseline (No Memory/Iteration)

**Uses ExpeRepair's actual repair code** with memory/iteration components disabled.

## Clean Structure

```
baseline_experepair/
├── experepair_core/              # ExpeRepair core (essential files only)
│   ├── agents/
│   │   ├── agent_write_patch.py  # write_multiple_patch_wo_memory()
│   │   ├── agent_common.py       # Shared agent utilities
│   │   └── patch_utils.py        # Patch validation
│   ├── config.py                 # Configuration
│   ├── data_structures.py        # BugLocation, etc.
│   ├── log.py                    # Logging
│   ├── post_process.py           # Response parsing
│   └── utils.py                  # General utilities
│
├── wrapper.py                    # Simple wrapper for CI issues
└── README.md                     # This file
```

## What's Removed

❌ **Removed files** (not needed for baseline):
- `agent_reproducer.py` - test reproduction (we use CI validation)
- `agent_reviewer.py` - iterative review (no iteration)
- `agent_search.py` - bug localization (use changed_files)
- `agentless_utils.py` - not needed
- `compress_file.py` - not needed
- `raw_tasks.py` - not needed
- `task.py` - not needed
- `task_counter.py` - not needed

## Usage

```python
from baseline_experepair.wrapper import generate_patch_experepair_baseline

# For CI issue
result = generate_patch_experepair_baseline(
    issue_description=ci_failure_logs,
    changed_files=['file1.py', 'file2.py'],
    repo_path="/path/to/repo",
    model="minimax/minimax-m2.5",
    diff=git_diff,
    workflow=workflow_yaml,
    validation_commands="pytest tests/"
)

# Returns: {"patch": "...", "cost": 0.002, "model": "...", "method": "experepair_baseline"}
```

## Key Method

From `experepair_core/agents/agent_write_patch.py`:

```python
def write_multiple_patch_wo_memory(
    self, test_content, orig_repro_result, retries: int = 3, patch_nums=4
):
    # ExpeRepair's built-in no-memory method
```

We call with:
- `retries=1` → no iteration
- `patch_nums=1` → single patch (not 4 candidates)

## Attribution

Core repair logic from: https://github.com/ExpeRepair/ExpeRepair

Modified for: CI failure repair (no memory/iteration)
