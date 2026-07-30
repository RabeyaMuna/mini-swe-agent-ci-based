# Codex Prompt Structure - Complete Analysis

## Overview

The Codex prompt is a carefully structured document that combines CI failure analysis, memory context (L1/L2/L3), and workflow verification to guide the AI in generating accurate fixes.

---

## Prompt Flow: How It Works

### 1. Extract Problems from CI Failure

**Function:** `extract_problem_list(ci_failure)`

```python
def extract_problem_list(ci_failure: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract individual problems from CI failure analysis.
    
    Each problem = one file that needs fixing
    """
    files = ci_failure.get("relevant_files") or []
    problems = []
    
    for idx, item in enumerate(files, start=1):
        file_name = item.get("file")
        reason = item.get("reason")
        failed_cmd = item.get("failed_cmd")
        
        problems.append({
            "number": idx,
            "title": file_name,
            "file": file_name,
            "reason": reason,
            "failed_cmd": failed_cmd,
        })
    
    return problems
```

**Example:** Issue 43 has 2 relevant files → 2 problems

```json
[
  {
    "number": 1,
    "title": "wandb/integration/ultralytics/bbox_utils.py",
    "file": "wandb/integration/ultralytics/bbox_utils.py",
    "reason": "Modified by black-jupyter hook to add blank line after imports",
    "failed_cmd": "./core/scripts/code-checks.sh update"
  },
  {
    "number": 2,
    "title": "wandb/integration/ultralytics/callback.py",
    "file": "wandb/integration/ultralytics/callback.py",
    "reason": "Reformatted by black-jupyter hook; added blank line after imports",
    "failed_cmd": "./core/scripts/code-checks.sh update"
  }
]
```

---

### 2. Load Memory Context (L1/L2/L3)

**Function:** `load_memory_context(..., ablation)`

**Ablations:**
- `baseline` → No memory (returns empty string)
- `L1` → Failure memory only
- `L1+L2` → Failure + Repo memory
- `L1+L2+L3` → Failure + Repo + Cross-repo memory

**How it works:**

```python
def load_memory_context(issue, ci_failure, verification, result_dir, memory_root, ablation, top_k):
    # Baseline = no memory
    if ablation.lower() == "baseline":
        return ""
    
    # Build query from CI failure
    query = {
        "task_id": issue["id"],
        "sha_fail": issue["sha_fail"],
        "repo": issue["repo"],
        "failure_reason": ci_failure["error_context"],
        "relevant_files": ci_failure["relevant_files"],
        "failed_cmd": verification["validation_sequence"],
        ...
    }
    
    # Retrieve relevant memories
    plugin = MemoryPlugin(config, ablation, memory_root, top_k)
    retrieval = plugin.retrieve(query)
    
    # Format for prompt
    formatted = plugin.format_for_prompt(retrieval)
    
    # Include metadata
    details = {
        "level_scores": retrieval["level_scores"],
        "l1_matches": retrieval["l1_matches"][:top_k],  # Top-K failure memories
        "l2_matches": retrieval["l2_matches"][:top_k],  # Top-K repo memories
        "l3_matches": retrieval["l3_matches"][:top_k],  # Top-K cross-repo memories
    }
    
    return f"{formatted}\n\n```json\n{details}\n```"
```

**Memory Levels:**

| Level | What It Contains | Example |
|-------|------------------|---------|
| **L1** | Similar CI failures in ANY repo | "Pre-commit black formatting failures in optuna/optuna" |
| **L2** | Past issues in SAME repo | "Previous formatting fixes in wandb/wandb" |
| **L3** | Cross-repo patterns | "Black-jupyter hook patterns across multiple repos" |

---

### 3. Compose Issue Document (The Prompt)

**Function:** `compose_issue_document(...)`

**Structure:**

```markdown
# CI Repair Task

You are fixing a CI failure in this repository.

## Problem To Fix
Problem {number} of {total}: {file}

Repository: {repo}
Failing commit: {sha_fail}
Workflow: {workflow_path}
File: {file}
Reason: {reason}
Failed command: {failed_cmd}

Full CI context:
{CI_FAILURE_ANALYSIS}

## Previous Experience / Memory
{MEMORY_CONTEXT}

Use this section only as guidance.
Do not copy a previous fix blindly.

## CI Verification
{WORKFLOW_VALIDATION_SEQUENCE}

## Required Workflow
1. Inspect the repository
2. Understand the problem
3. Use memory to guide strategy
4. Identify minimal fix
5. Make the change
6. Verify locally
7. Leave final fix in working tree

## Scope Rules
- Fix this problem only
- Preserve earlier patches
- Minimal correct change
- Prefer source fixes over test suppression

## Final Response
Report:
- root cause
- files changed
- verification command
- verification result
- remaining risk
```

---

## Complete Example: Issue 43, Problem 1

### Input Data

**CI Failure:**
```json
{
  "error_context": [
    "Pre-commit hooks made automatic code style fixes",
    "black-jupyter reformatted files, causing exit code 1"
  ],
  "relevant_files": [
    {
      "file": "wandb/integration/ultralytics/bbox_utils.py",
      "reason": "Modified by black-jupyter hook to add blank line after imports",
      "failed_cmd": "./core/scripts/code-checks.sh update",
      "failed_tool": "black-jupyter"
    }
  ],
  "error_types": [
    {
      "category": "Code Formatting",
      "subcategory": "Pre-commit hook black-jupyter reformatted file"
    }
  ]
}
```

**Workflow Verification:**
```json
{
  "validation_sequence": [
    {
      "order": 1,
      "validates": "Environment setup",
      "installation_cmd": "./core/scripts/code-checks.sh update"
    },
    {
      "order": 2,
      "validates": "Pre-commit hooks",
      "validation_cmd": "pre-commit run --hook-stage pre-push --all-files"
    }
  ]
}
```

**Memory Context (Baseline):**
```
No memory context is enabled for this run.
```

### Generated Prompt

```markdown
# CI Repair Task

You are fixing a CI failure in this repository.

## Problem To Fix
Problem 1 of 2: wandb/integration/ultralytics/bbox_utils.py

Repository: wandb/wandb
Failing commit: 077f6aaac3ebb96626ac747fb126a0b4d752489c
Workflow: .github/workflows/pre-commit.yml
File: wandb/integration/ultralytics/bbox_utils.py
Reason: Modified by black-jupyter hook to add a blank line after imports
Failed command: ./core/scripts/code-checks.sh update

Full CI context:
## Failure Context
- The CI pre-commit job failed because black-jupyter hook reformatted files

## Failure Signals  
- pre-commit action@v3.0.0: Exit code 1 due to file modifications
- black-jupyter hook: reformatted files adding blank line after imports

## Relevant Files
- wandb/integration/ultralytics/bbox_utils.py (Code Formatting)
- wandb/integration/ultralytics/callback.py (Code Formatting)

## Previous Experience / Memory
No memory context is enabled for this run.

## CI Verification
### Step 1: Environment setup
- Install: ./core/scripts/code-checks.sh update

### Step 2: Pre-commit hooks
- Validate: pre-commit run --hook-stage pre-push --all-files

## Required Workflow
1. Inspect the repository
2. Understand the failing problem
3. Identify smallest set of files
4. Make minimal correct change
5. Run verification command
6. Leave final fix in working tree

## Scope Rules
- Fix this problem only
- Preserve earlier patches
- Do not modify unrelated files

## Final Response
When finished, report:
- root cause
- files changed
- verification command run
- verification result
```

---

## With Memory: L1+L2+L3 Example

**Same issue, but with L1+L2+L3 ablation:**

```markdown
## Previous Experience / Memory

### L1: Similar Failures Across Repos

**Match 1: Pre-commit black formatting in optuna/optuna**
- Issue: #28
- Failure: Black hook reformatted imports
- Fix: Added blank line after imports in optuna/study/_multi_objective.py
- Success: Yes
- Similarity: 0.92

**Match 2: Pre-commit formatting in online-ml/river**
- Issue: #37  
- Failure: Black hook type annotation fix
- Fix: Added type hint to stats.base.Univariate
- Success: Yes
- Similarity: 0.88

### L2: Past Issues in wandb/wandb

**Match 1: Previous import formatting fix**
- PR: #1234
- Date: 2026-06-15
- Files: wandb/integration/keras/callback.py
- Fix: Added blank lines after import blocks
- Pattern: Same black-jupyter hook requirement

### L3: Cross-Repo Patterns

**Pattern: Black-jupyter blank line requirement**
- Repos: 15 matches
- Fix pattern: Add blank line after last import, before first code
- Common in: Jupyter-integrated codebases
- Success rate: 98%

### Candidate Files (from memory)
- wandb/integration/ultralytics/bbox_utils.py (HIGH CONFIDENCE)
- wandb/integration/ultralytics/callback.py (HIGH CONFIDENCE)

### High-level Hints
- This is a standard black-jupyter formatting requirement
- Add blank line after import blocks
- Both files likely need same fix
- Verify with: pre-commit run --hook-stage pre-push --all-files

```json
{
  "level_scores": {
    "l1_score": 0.92,
    "l2_score": 0.85,
    "l3_score": 0.78
  },
  "weighted_similarity": 0.87,
  "l1_matches": [...],
  "l2_matches": [...],
  "l3_matches": [...]
}
```

Use this section only as guidance.
Do not copy a previous fix blindly.
```

---

## How Codex Uses This

### 1. Understanding Phase
Codex reads:
- **Problem To Fix** → What file needs fixing and why
- **CI Failure Analysis** → Detailed error context
- **Memory** → Similar past solutions (if available)

### 2. Planning Phase
Codex determines:
- Root cause from failure signals
- Which files to examine
- Strategy based on memory patterns
- Verification command to run

### 3. Execution Phase
Codex performs:
- Reads the problematic file
- Makes minimal fix (e.g., adds blank line)
- Runs verification command
- Checks if fix works

### 4. Iteration Phase
If verification fails:
- Inspect new error output
- Refine fix
- Re-run verification
- Continue until pass

### 5. Response Phase
Codex reports:
```
Root cause: Black-jupyter hook requires blank line after imports
Files changed: wandb/integration/ultralytics/bbox_utils.py
Verification: pre-commit run --hook-stage pre-push --all-files
Result: PASSED
Remaining risk: None
```

---

## Multiple Problems Handling

**Issue 43 has 2 problems:**

1. **Problem 1:** bbox_utils.py
   - Codex receives prompt for problem 1
   - Makes fix
   - Leaves patch in working tree
   
2. **Problem 2:** callback.py
   - Codex receives NEW prompt for problem 2
   - Working tree ALREADY has fix for problem 1
   - Scope rule: "Preserve earlier patches"
   - Codex adds fix for problem 2
   - Final patch has BOTH fixes

**Result:** Single `patch.diff` with both files fixed

---

## Prompt Structure Benefits

✅ **Clear problem definition** - Knows exactly what to fix  
✅ **Rich context** - CI failure + verification + memory  
✅ **Guidance not prescription** - Memory guides, doesn't dictate  
✅ **Scope control** - Fix only current problem  
✅ **Verification built-in** - Knows how to test the fix  
✅ **Iterative repair** - Can retry if first attempt fails  
✅ **Multiple problems** - Preserves earlier fixes  

---

## Key Design Decisions

### 1. Why One Problem at a Time?

**Reason:** Focused attention, clearer context

**Alternative considered:** All problems in one prompt
- ❌ Confusing when problems interact
- ❌ Harder to preserve incremental progress
- ❌ Verification becomes ambiguous

**Current approach:**
- ✅ Clear focus per problem
- ✅ Incremental verification
- ✅ Earlier fixes preserved

### 2. Why Memory is "Guidance Only"?

**Reason:** Past fixes might not apply to current state

**Example:**
- Memory: "Fix by adding type hint"
- Current: Type hint already exists, different issue
- Without warning: Codex might blindly copy fix
- With warning: Codex verifies before applying

### 3. Why Include Workflow Verification?

**Reason:** Codex needs to know HOW to verify the fix

**Without verification:**
- Codex makes change
- No way to confirm it works
- Might introduce new bugs

**With verification:**
- Codex makes change
- Runs: `pre-commit run --all-files`
- Sees result: PASSED
- Confident the fix works

---

## Ablation Comparison

| Ablation | Memory | Prompt Size | Fix Quality |
|----------|--------|-------------|-------------|
| **Baseline** | None | ~2K tokens | Good (no bias) |
| **L1** | Failure patterns | ~4K tokens | Better (learns from failures) |
| **L1+L2** | + Repo history | ~6K tokens | Better (repo-specific) |
| **L1+L2+L3** | + Cross-repo | ~8K tokens | Best (pattern recognition) |

**Trade-off:**
- More memory → Better guidance
- More memory → Larger prompt → Slower
- More memory → Risk of bias from past

---

## Files Involved

1. **run_codex_ci_repair.py**
   - `extract_problem_list()` → Extract problems from CI failure
   - `load_memory_context()` → Load L1/L2/L3 memories
   - `compose_issue_document()` → Build the prompt
   - `write_issue_document()` → Save to file

2. **memory_plugin/**
   - `retrieve()` → Find similar memories
   - `format_for_prompt()` → Format memories for prompt

3. **utilities/ci_log_analyzer.py**
   - Generates CI failure analysis

4. **utilities/ci_workflow_aware_retrieval.py**
   - Extracts workflow validation sequence

---

## Output Files

For each problem, generates:

```
results/codex/baseline_minimax2_5/43/
├── issue_document_problem_1.md       ← The prompt for problem 1
├── issue_document_problem_2.md       ← The prompt for problem 2
├── codex_transcript_problem_1.txt    ← Codex output for problem 1
├── codex_transcript_problem_2.txt    ← Codex output for problem 2
├── memory_context.md                 ← Memory section (if L1/L2/L3)
├── memory_retrieval.json             ← Raw memory data
├── patch.diff                        ← Final combined fix
└── result.json                       ← Metadata
```

---

## Summary

The Codex prompt is a **structured document** that combines:

1. **Problem definition** - What to fix
2. **CI failure analysis** - Why it failed
3. **Memory context** - How others fixed similar issues (L1/L2/L3)
4. **Workflow verification** - How to test the fix
5. **Scope rules** - What NOT to change
6. **Response format** - What to report

This structure enables:
- ✅ Focused, targeted fixes
- ✅ Learning from past solutions
- ✅ Iterative refinement
- ✅ Built-in verification
- ✅ Multiple problem handling
- ✅ Minimal, correct changes

---

## Date
2026-07-30
