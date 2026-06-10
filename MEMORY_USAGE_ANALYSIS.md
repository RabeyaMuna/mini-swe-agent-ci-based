# How Previous Experiences Are Used in CI Repair

## Complete Flow Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CI REPAIR FLOW                               │
└─────────────────────────────────────────────────────────────────────┘

1. NEW CI FAILURE (Input)
   ├─ CI log
   ├─ Failed commit (sha_fail)
   ├─ Workflow path
   └─ Changed files

2. PHASE A: CI Log Analysis (CILogAnalyzer)
   ├─ Extract: overall_failure_reasons
   ├─ Extract: failed_jobs/commands
   ├─ Extract: affected_files
   └─ Extract: error_types

3. PHASE B: Workflow Analysis
   ├─ Extract validation sequence
   ├─ Extract validation commands
   └─ Extract installation commands

4. PHASE C: Memory Retrieval & Synthesis ⭐ THIS IS WHERE MEMORY IS USED
   │
   ├─ Step 1: Build Query from CI Context
   │   ├─ task_id, sha_fail, repo, workflow_path
   │   ├─ overall_failure_reasons
   │   ├─ error_types
   │   ├─ relevant_files
   │   └─ changed_files
   │
   ├─ Step 2: Embed Query
   │   └─ sentence-transformers / fastembed
   │
   ├─ Step 3: Cosine Similarity Search
   │   ├─ Search L1 (file-level memories)
   │   ├─ Search L2 (issue-level memories)
   │   ├─ Search L3 (pattern-level memories)
   │   └─ Rank by similarity score
   │
   ├─ Step 4: LLM Synthesis (or Deterministic Fallback)
   │   ├─ Input: Current CI context + Retrieved memories
   │   ├─ LLM reasons over memories
   │   └─ Output: guidance_document
   │
   └─ Step 5: Format for Agent
       └─ Convert guidance_document → Markdown

5. PHASE D: Build Problem Statement
   ├─ CI Failure Report (from Phase A)
   ├─ Validation Hints (from Phase B)
   └─ Memory Context (from Phase C) ⭐ INJECTED HERE

6. AGENT RECEIVES PROBLEM STATEMENT
   └─ Makes repair with memory guidance
```

---

## Phase C: Memory Retrieval (Detailed)

### Step 1: Build Query

**File:** `memory_plugin.py::build_query()`

**What it does:**
- Takes CI log analysis output
- Builds a structured query containing:
  ```python
  {
    "task_id": "issue_id",
    "sha_fail": "commit_hash",
    "repo": "owner/repo",
    "workflow_path": ".github/workflows/test.yml",
    "error_type": ["type_error", "import_error"],
    "failure_pattern": "test failure",
    "overall_failure_reason": "mypy type check failed",
    "relevant_files": ["src/client.py", "src/server.py"],
    "changed_files": ["src/client.py"],
    "failed_cmd": ["mypy src/"],
    "failed_tool": ["mypy"]
  }
  ```

### Step 2: Embed Query

**File:** `memory_plugin.py::retrieve()`

**What it does:**
- Converts query to embedding vector using:
  - `sentence-transformers` (default)
  - or `fastembed` (fallback)
- Embedding captures semantic meaning of the failure

### Step 3: Cosine Similarity Search

**File:** `memory_plugin.py::retrieve()`

**What it does:**
- Compares query embedding with ALL memory embeddings
- Calculates cosine similarity score (0.0 to 1.0)
- Retrieves top-K matches per level (default K=3)

**Retrieval per level:**
```python
L1 (file-level):
  - Filters: same repo OR same workflow
  - Returns: top 3 most similar file-level memories
  - Fields: file, problem, fix_strategy, diff_evidence

L2 (issue-level):
  - Filters: same repo (broader scope)
  - Returns: top 3 most similar issue-level memories
  - Fields: atomic_problems, repair_trajectory

L3 (pattern-level):
  - Filters: NONE (cross-repo)
  - Returns: top 3 most similar universal patterns
  - Fields: principle, pattern_name, diagnostic_guide, fix_guide
```

**Example retrieval result:**
```python
{
  "matches": [
    {
      "memory_level": "L1",
      "similarity_score": 0.82,
      "file": "src/client.py",
      "problem": "Type error at line 45...",
      "fix_strategy": "Update annotation...",
      "repo": "flower",
      "workflow_path": ".github/workflows/test.yml"
    },
    {
      "memory_level": "L2",
      "similarity_score": 0.75,
      "repo": "flower",
      "atomic_problems": [
        {
          "problem_id": 1,
          "problem": "Type errors in 22 files...",
          "file_changes": [...]
        }
      ],
      "repair_trajectory": "Fix order: install → lint → types..."
    },
    {
      "memory_level": "L3",
      "similarity_score": 0.68,
      "principle": "Type system covariance after dependency upgrade",
      "pattern_name": "list_to_sequence_type_migration",
      "diagnostic_guide": {...},
      "fix_guide": {...}
    }
  ],
  "level_scores": {
    "L1": 0.82,
    "L2": 0.75,
    "L3": 0.68
  },
  "weighted_similarity": 0.78
}
```

### Step 4: LLM Synthesis

**File:** `ci_memory_system.py::_run_single_llm_synthesis()`

**What it does:**
1. Builds a synthesis prompt with:
   - Current CI context
   - Retrieved memory candidates
   - Deterministic fallback summary

2. Asks LLM to reason:
   - What is the real root cause?
   - What files need changing (including hidden ones)?
   - What might break after fixing the visible failure?
   - What's the fix approach?
   - How confident are we?

3. LLM output structure:
```json
{
  "diagnosis": "Root cause: API upgrade changed type signatures...",
  
  "primary_files": [
    {
      "file": "src/client.py",
      "reason": "Mentioned in CI log and has prior type error repair",
      "fix": "Update List → Sequence at line 45"
    }
  ],
  
  "full_scope": {
    "files_in_log": ["src/client.py"],
    "primary_files": [...],
    "additional_files": [
      {
        "file": "src/server.py",
        "reason": "Not in log but prior repair shows this file also needed type updates",
        "fix": "Update Dict → Mapping at line 67"
      }
    ],
    "l1_followup_queries": [
      {"file": "src/server.py", "reason": "Fetch L1 details for this file"}
    ]
  },
  
  "linked_issues": [
    {
      "root_cause": "Same API upgrade affects multiple files",
      "affected_files": ["src/client.py", "src/server.py", "src/utils.py"],
      "fix_pattern": "Update covariant types: List→Sequence, Dict→Mapping",
      "missing_from_log": "server.py and utils.py will fail mypy after client.py is fixed",
      "workflow_stage": "type_check"
    }
  ],
  
  "fix_approach": [
    "1. Fix client.py first (visible failure)",
    "2. Update server.py and utils.py (hidden failures from memory)",
    "3. Run mypy src/ to verify all type errors resolved"
  ],
  
  "post_fix_patterns": [
    {
      "pattern": "Test failures may appear after type fixes due to mock signatures",
      "likelihood": "medium",
      "how_to_fix": "Update test mocks to match new type signatures"
    }
  ],
  
  "verification": {
    "command": "mypy src/",
    "expected_output": "Success: no issues found",
    "files_to_check": ["src/client.py", "src/server.py", "src/utils.py"]
  },
  
  "confidence": "high",
  "confidence_reason": "Best memory similarity 0.82, pattern matches exactly",
  
  "summary": "Prior repairs show this API upgrade requires type updates in 3 files, not just the one in the CI log. Fix pattern: List→Sequence, Dict→Mapping.",
  
  "relevant_candidates": [
    {"index": 0, "memory_level": "L1", "similarity_score": 0.82, "relevance": "high"},
    {"index": 1, "memory_level": "L2", "similarity_score": 0.75, "relevance": "high"}
  ]
}
```

**Fallback:** If LLM fails, uses deterministic guidance (no LLM reasoning)

### Step 5: Format for Agent

**File:** `ci_memory_system.py::format_memory_context()`

**What it does:**
- Converts guidance_document to Markdown
- Structures it for agent consumption

**Output format:**
```markdown
## Memory Context — Repair Guidance from Past Experience

**Confidence:** 🟢 HIGH — Best memory similarity 0.82, pattern matches exactly

### What is really happening
Root cause: API upgrade changed type signatures requiring covariant type updates...

### Primary files to inspect first
  - `src/client.py` — Mentioned in CI log and has prior type error repair
    → **prior fix pattern:** Update List → Sequence at line 45

### Files to fix — including those NOT in the log
  - `src/server.py` — Not in log but prior repair shows needed
    → **fix:** Update Dict → Mapping at line 67
  - `src/utils.py` — Prior repair evidence
    → **fix:** Update Set → AbstractSet

### Linked Issues
**Issue 1:** API upgrade affects multiple files
  - **Root cause:** Same dependency change
  - **Affected files:** client.py, server.py, utils.py
  - **Fix pattern:** Update covariant types: List→Sequence, Dict→Mapping
  - **Hidden from log:** server.py and utils.py will fail mypy after client.py fixed

### Fix Approach
  1. Fix client.py first (visible failure)
  2. Update server.py and utils.py (hidden failures from memory)
  3. Run mypy src/ to verify all type errors resolved

### What May Break After This Fix
  - **Pattern:** Test failures due to mock signatures
    **Likelihood:** 🟡 MEDIUM
    **How to fix:** Update test mocks to match new type signatures

### How to Verify the Fix
**Command:** `mypy src/`
**Expected output:** Success: no issues found
**Files to check:** src/client.py, src/server.py, src/utils.py

### L1 Follow-up Queries
If more detail needed, fetch file-level L1 memory for:
  - `src/server.py` — Fetch specific L1 repair details
```

---

## Phase D: Problem Statement Integration

**File:** `ci_context.py::build_problem_statement()`

**Final problem statement structure:**
```markdown
# CI Failure Report

**Repository**: `flower`
**Failing Commit (sha_fail)**: `abc123def`
**Workflow**: `.github/workflows/test.yml`

**Error categories**: type_error

## Overall CI Summary
  - CI failed during mypy type checking step

## Why the CI Failed
  - mypy type check failed in src/client.py

## Failed Jobs / Commands
  - **Job:** test / **Step:** type-check
    **Command:** `mypy src/`
    **Status:** failed

## Affected Files
  - `src/client.py` — Type error at line 45

## Validation Hints
  - mypy src/
  - pytest tests/

## Installation Hints
  - poetry install

## Memory Context — Repair Guidance from Past Experience
[... memory guidance from above ...]

## Suggested Repair Plan
[... if repair planner is enabled ...]
```

---

## How Memory Helps the Agent

### Without Memory:
```markdown
## Why the CI Failed
  - mypy type check failed in src/client.py

## Affected Files
  - src/client.py
```

**Agent sees:** One file mentioned, fixes it, CI still fails on server.py

### With Memory:
```markdown
## Why the CI Failed
  - mypy type check failed in src/client.py

## Affected Files
  - src/client.py

## Memory Context — Repair Guidance from Past Experience

### Files to fix — including those NOT in the log
  - src/server.py — Prior repair shows needed (hidden failure)
  - src/utils.py — Prior repair shows needed (hidden failure)

### Fix Approach
  1. Fix all 3 files together
  2. Pattern: List→Sequence, Dict→Mapping
```

**Agent sees:** 3 files need fixing, knows the pattern, fixes all at once

---

## Key Insights

### 1. Memory Shows Hidden Failures
- CI log only shows **first failure**
- Memory reveals **downstream failures** that would appear after first fix
- Example: "Fix client.py, but also server.py and utils.py"

### 2. Memory Provides Concrete Fixes
- Not just "fix types"
- Specific: "Update List → Sequence at line 45"
- Pattern: "Use covariant types per PEP-484"

### 3. Memory Gives Repair Order
- "Fix dependency first, then types, then tests"
- Based on CI validation sequence from prior repairs
- Prevents fixing in wrong order

### 4. Memory Includes Verification
- Exact command to run: `mypy src/`
- What success looks like: "no issues found"
- Which files to check

### 5. Confidence Scoring
- 🟢 HIGH (0.65+): Trust the guidance
- 🟡 MEDIUM (0.35-0.65): Consider it
- 🔴 LOW (<0.35): Use with caution

---

## Current Limitations

### 1. **Memory Content Quality**
**Problem:** Current memory has vague content
```json
{
  "problem": "Type error",
  "fix_strategy": "Fixed types"
}
```

**Impact:** LLM synthesis produces generic guidance

**Solution:** ✅ IMPLEMENTED — Enhanced prompts generate concrete content

### 2. **Pattern Recognition**
**Problem:** 110 files with same fix → 110 separate L1 entries

**Impact:** Memory retrieval may miss the pattern

**Solution:** Pattern-based L1 entries (grouped)

### 3. **Workflow-Aware Retrieval**
**Problem:** Memory doesn't consider CI validation order

**Impact:** May not identify which failures are downstream

**Solution:** Could enhance query with validation_step metadata

---

## What Makes Memory Useful

### ✅ Good Memory Entry:
```json
{
  "problem": "Type error at line 45 in client.py. Symptom: mypy 'incompatible type List[int]; expected Sequence[int]'. Root cause: API upgrade requires covariant types.",
  
  "fix_strategy": "Update annotation at line 45. Step 1: Change List to Sequence. Step 2: Import Sequence. Before: 'List[int]'. After: 'Sequence[int]'. Verification: mypy src/"
}
```

**Why it's useful:**
- Agent knows **exactly** what's wrong (line 45, specific error)
- Agent knows **exactly** what to do (step-by-step)
- Agent knows **how to verify** (command to run)
- Agent understands **why** (API upgrade, covariance)

### ❌ Bad Memory Entry:
```json
{
  "problem": "Type error",
  "fix_strategy": "Fixed types"
}
```

**Why it's not useful:**
- Too vague to apply
- No line numbers
- No concrete steps
- No verification

---

## Summary

**Memory flow:**
1. ✅ Query built from CI context
2. ✅ Similarity search retrieves relevant memories
3. ✅ LLM synthesizes guidance document
4. ✅ Formatted as Markdown section
5. ✅ Injected into problem statement
6. ✅ Agent receives context + memory

**What's working:**
- Retrieval mechanism
- Synthesis with LLM
- Integration into problem statement

**What needs improvement:**
- ✅ **DONE:** Memory content quality (enhanced prompts)
- ⏳ **TODO:** Pattern-based grouping for large changes
- ⏳ **TODO:** Workflow-aware retrieval

**Impact of enhanced content:**
- Better similarity matching (concrete symptoms)
- Better LLM synthesis (detailed input)
- Better agent repairs (actionable guidance)
