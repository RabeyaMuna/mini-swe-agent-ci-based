# ExpeRepair Simple Baseline for CI Issues

## Goal
Extract a **simple 1-2 prompt baseline** from ExpeRepair for CI issue repair, WITHOUT the expensive memory/retrieval system.

## What to Extract from ExpeRepair

### ✅ Use (Simple Prompts)
- **System Prompt** (without semantic rules)
- **Patch Generation Prompt** (without retrieved demonstrations)

### ❌ Don't Use (Memory Components)
- Episodic memory retrieval
- Semantic memory insights
- Retrieved demonstrations from past issues
- Iterative refinement loops
- Seed issue collection

---

## Implementation Plan

### Step 1: Adapt CI Context Structure

ExpeRepair expects:
- Issue description
- Code context (localized files)
- Test script (optional)

For CI issues, you have:
- CI failure logs
- Workflow YAML
- Changed files (from commit)
- Validation sequence (from your backward decomposition)

**Mapping:**
```
ExpeRepair → CI Baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Issue description → CI failure summary
Code context → Changed files + workflow
Test script → Validation commands
```

### Step 2: Create Simple Prompt Template

Based on ExpeRepair's `WRITE_PATCH_PROMPT`:

```python
EXPEREPAIR_BASELINE_PROMPT = """
You are a software developer maintaining the repository {repo_name}.
A CI workflow has failed. Your task is to write a patch that resolves this failure.

### Phase 1: FIX ANALYSIS
1. Review the CI failure description and state clearly what the problem is.
2. Analyze the workflow configuration and identify which step failed.
3. Review the changed files and locate where the problem occurs in the code.
4. State clearly the best practices to take into account in the fix.
5. State clearly how to fix the problem.

### Phase 2: FIX IMPLEMENTATION
1. Focus on making minimal, precise, and relevant changes to resolve the CI failure.
2. Include any necessary imports or configuration changes.
3. Write the patch using the strict format specified below:

- Each modification must be enclosed in:
  - `<file>...</file>`: actual file path
  - `<original>...</original>`: original code snippet
  - `<patched>...</patched>`: fixed version

- The `<original>` block must contain an exact, continuous block of code
- Pay attention to indentation (Python code)
- DO NOT include line numbers
- You can write up to three modifications if needed

EXAMPLE PATCH FORMAT:
# modification 1
```
<file>...</file>
<original>...</original>
<patched>...</patched>
```

===== INPUT =====

<ci_failure>
{ci_failure_info}
</ci_failure>

<workflow>
{workflow_yaml}
</workflow>

<changed_files>
{changed_files_context}
</changed_files>

<validation_sequence>
{validation_commands}
</validation_sequence>

===== END INPUT =====

Now provide your fix analysis and patch.
"""
```

### Step 3: Create Baseline Script

**File:** `scripts/run_experepair_baseline.py`

```python
def run_experepair_baseline(issue_data: dict, model: str = "deepseek-v4-flash"):
    """
    Simple ExpeRepair-style baseline for CI issue repair.
    
    No memory, no retrieval, no iteration - just:
    1. Format CI issue into ExpeRepair prompt structure
    2. Call LLM once
    3. Parse patch
    """
    
    # 1. Extract CI context
    ci_failure_info = extract_ci_failure_summary(issue_data)
    workflow_yaml = issue_data.get('workflow', '')
    changed_files = get_changed_files_content(issue_data)
    validation_commands = extract_validation_sequence(issue_data)
    
    # 2. Format prompt (NO memory retrieval)
    prompt = EXPEREPAIR_BASELINE_PROMPT.format(
        repo_name=issue_data['repo_name'],
        ci_failure_info=ci_failure_info,
        workflow_yaml=workflow_yaml,
        changed_files_context=changed_files,
        validation_commands=validation_commands
    )
    
    # 3. Single LLM call (no iteration)
    llm = get_llm(model)
    response = llm.invoke(prompt)
    
    # 4. Parse patch
    patch = extract_patch_from_response(response)
    
    return {
        "patch": patch,
        "analysis": extract_analysis_from_response(response),
        "model": model,
        "method": "experepair_baseline"
    }
```

### Step 4: Integration Points

**Add to your existing pipeline:**

```python
# In your main evaluation script
from scripts.run_experepair_baseline import run_experepair_baseline

# After running forward/backward/bidirectional
experepair_result = run_experepair_baseline(
    issue_data=issue,
    model=args.model
)

# Compare results
results = {
    "forward": forward_patch,
    "backward": backward_patch,
    "bidirectional": bidirectional_patch,
    "experepair_baseline": experepair_result["patch"]
}
```

---

## Expected Output Structure

```json
{
  "issue_id": "28",
  "baselines": {
    "experepair_simple": {
      "analysis": "The isort check fails because...",
      "patch": "diff --git a/...",
      "resolved": true,
      "cost": 0.002
    }
  },
  "your_methods": {
    "forward": {...},
    "backward": {...},
    "bidirectional": {...}
  }
}
```

---

## What Makes This a "Simple Baseline"

✅ **Keeps:**
- Clear two-phase structure (Analysis → Implementation)
- Structured patch format
- Focus on minimal changes

❌ **Removes:**
- Episodic memory (retrieved demonstrations)
- Semantic memory (summarized insights)
- Iterative refinement
- Multiple candidate sampling
- Seed issue collection

This is a **single-shot, stateless baseline** - exactly what your professor asked for!

---

## Next Steps

1. ✅ Create `prompt_template/experepair_baseline/` directory
2. ✅ Write the prompt template (above)
3. ✅ Write helper functions:
   - `extract_ci_failure_summary()` - from logs
   - `get_changed_files_content()` - from git diff
   - `extract_validation_sequence()` - from your backward decomp
4. ✅ Create `scripts/run_experepair_baseline.py`
5. ✅ Test on a few CI issues
6. ✅ Integrate into your evaluation pipeline

---

## Comparison Study

You'll compare:

| Method | Description | Memory? | Iterative? |
|--------|-------------|---------|------------|
| ExpeRepair Baseline | Simple LLM prompting | ❌ | ❌ |
| Your Forward | Commit-based | ❌ | ❌ |
| Your Backward | CI-based | ❌ | ❌ |
| Your Bidirectional | Unified | ❌ | ❌ |

This shows whether your **decomposition approach** adds value over **simple LLM prompting**.
