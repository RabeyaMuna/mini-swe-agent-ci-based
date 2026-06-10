# Professor Meeting Summary

## 1. What Professor Instructed vs. What You Implemented

### ✅ What You DID Correctly

| Professor's Guidance | Your Implementation | Status |
|---------------------|---------------------|--------|
| **Don't store only failure + file + fix** | ✅ You store: root cause, why fix works, sub-problems, repair strategy, adaptation guidance | **DONE** |
| **Use abstraction and adaptation** | ✅ You have L1/L2/L3 hierarchical abstraction:<br>- L1: File-level details<br>- L2: Issue-level with atomic problems<br>- L3: Cross-repo universal patterns | **DONE** |
| **One CI failure = multiple problems** | ✅ You decompose each CI failure into atomic problems (visible + hidden) | **DONE** |
| **Retrieve at sub-problem level** | ✅ Your memory retrieval searches for similar atomic problems, not just whole issues | **DONE** |
| **Use reasoning/explanation generation** | ✅ You use LLM to generate:<br>- Root cause<br>- Why fix works<br>- Sub-problems<br>- Repair trajectory | **DONE** |
| **Start with one bug first** | ✅ You have decompose_ci_failure.py that processes single issues | **DONE** |

### ❌ What's MISSING or INCOMPLETE

| Professor's Guidance | Current Gap | Impact |
|---------------------|-------------|--------|
| **Memory should help guide the fix, not copy old patch** | ⚠️ Your memory context is being SHOWN to agent, but agent is NOT using it effectively to fix ALL problems | **Critical - This is why results are poor** |
| **Separate/prune irrelevant memory** | ⚠️ You retrieve relevant memories, but the AGENT still only fixes the first visible problem | **Critical - Separation at agent execution level missing** |
| **Compare Mini-SWE baseline vs enhanced** | ❌ Not done - no clear comparison showing baseline vs. your improved system | **Missing evaluation** |

---

## 2. High-Level Architecture of Your System

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE MEMORY BUILDING                             │
└────────────────────────────────────────────────────────────────────────────┘

Historical CI Failure (from dataset)
         │
         ├─→ [1. decompose_ci_failure.py]
         │    • Input: CI log + ground-truth diff + workflow YAML
         │    • Process: LLM reverse-engineers ALL atomic problems
         │               (visible problem + hidden problems)
         │    • Output: decomposed_issues.json
         │              └─ problems: [{
         │                   problem_id, visibility, symptom, root_cause,
         │                   how_fixed, why_fix_works, affected_files,
         │                   workflow_validation_order, ci_command
         │                 }]
         │              └─ repair_trajectory: ordered repair sequence
         │
         ├─→ [2. build_memory_from_decomposed.py]
         │    • Input: decomposed_issues.json
         │    • Process: LLM abstracts into 3 memory levels
         │    • Output: 
         │       • L1 (failure_memory.json) - per-file memory
         │       • L2 (repo_memory.json) - per-issue with atomic problems
         │       • L3 (cross_memory.json) - universal repair patterns
         │
         └─→ Memory Bank Ready (data/trs/)
              • L1: File-level failure records (repo-scoped)
              • L2: Issue-level with atomic_problems + repair_trajectory_summary
              • L3: Cross-repo universal principles


┌────────────────────────────────────────────────────────────────────────────┐
│                         ONLINE REPAIR EXECUTION                             │
└────────────────────────────────────────────────────────────────────────────┘

New CI Failure Instance
         │
         ├─→ [Phase A: CILogAnalyzer]
         │    • Parse CI logs
         │    • Extract: error_context, relevant_files, error_types, failed_jobs
         │
         ├─→ [Phase B: Workflow Analysis]
         │    • Parse workflow YAML
         │    • Extract: validation_sequence (ordered CI steps)
         │
         ├─→ [Phase C: Memory Retrieval]
         │    • Build query from Phase A + B
         │    • Embed query (sentence-transformers)
         │    • Cosine similarity search across L1/L2/L3
         │    • LLM synthesis: combine retrieved memories into guidance
         │    • Output: 
         │        - diagnosis
         │        - primary_files
         │        - additional_files (hidden problems)
         │        - linked_issues
         │        - fix_approach
         │        - verification commands
         │
         ├─→ [Phase D: Problem Statement Assembly]
         │    • Build problem_statement with:
         │       - CI failure info
         │       - Validation commands
         │       - Memory context (guidance from past repairs)
         │
         └─→ [Mini-SWE-Agent Execution]
              • Agent receives problem_statement
              • Agent runs in environment
              • ❌ PROBLEM: Agent only fixes FIRST visible problem
              • Agent ignores hidden problems from memory guidance


┌────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY STRUCTURE EXAMPLE                            │
└────────────────────────────────────────────────────────────────────────────┘

L2 Memory Entry (repo_memory.json):
{
  "issue_id": "410",
  "repo": "fish-speech/fish-speech",
  "workflow_path": ".github/workflows/lint.yaml",
  
  "atomic_problems": [
    {
      "problem_id": 1,
      "visibility": "visible_in_log",  ← Agent sees this
      "issue_type": "dependency_constraint_error",
      "failed_cmd": "uv pip install -e .",
      "problem": "Package fish-audio-sdk>=2024.12.5 not available",
      "root_cause": "SDK version constraint too new",
      "how_fixed": "Relaxed version constraint",
      "why_fix_works": "Allows installation with available SDK version",
      "file_changes": [{"file": "pyproject.toml", "fix": "Changed fish-audio-sdk>=2024.12.5 to >=2024.11.0"}]
    },
    {
      "problem_id": 2,
      "visibility": "hidden",  ← Agent SHOULD fix this but DOESN'T
      "issue_type": "type_annotation_error",
      "failed_cmd": "mypy fish_speech/",
      "problem": "Type hints incompatible with new SDK",
      "root_cause": "SDK API changed return types",
      "how_fixed": "Updated type annotations in 11 files",
      "why_fix_works": "Type hints match new SDK API",
      "file_changes": [
        {"file": "fish_speech/audio/processor.py", "fix": "Changed AudioData -> tuple[np.ndarray, int]"},
        {"file": "fish_speech/audio/encoder.py", "fix": "Changed AudioData -> tuple[np.ndarray, int]"},
        ...  // 9 more files
      ]
    },
    {
      "problem_id": 3,
      "visibility": "hidden",  ← Agent SHOULD fix this but DOESN'T
      "issue_type": "test_expectation_mismatch",
      "failed_cmd": "pytest tests/",
      "problem": "Test expectations outdated after SDK upgrade",
      "file_changes": [{"file": "tests/test_audio.py", "fix": "Updated mock expectations"}]
    }
  ],
  
  "repair_trajectory_summary": "Fix dependency constraint first (Problem 1), then mypy will run and reveal type errors (Problem 2), then tests will run and reveal test failures (Problem 3). Each problem blocks the next validation stage."
}
```

---

## 3. What Is The Problem? Why Results Are Still Poor?

### The Core Issue

**Your system BUILDS the right memory and RETRIEVES the right guidance, but the Mini-SWE-Agent IGNORES it.**

### Why Agent Only Fixes First Problem

```
┌────────────────────────────────────────────────────────────────┐
│  What Memory System Provides                                   │
└────────────────────────────────────────────────────────────────┘

Problem Statement includes:
  "## Memory Context — Repair Guidance from Past Experience
   
   ### Primary files to inspect first
     - `pyproject.toml` — Dependency constraint error
   
   ### Files to fix — including those NOT in the log
     - `fish_speech/audio/processor.py` — Type annotation error (HIDDEN)
     - `fish_speech/audio/encoder.py` — Type annotation error (HIDDEN)
     - ... (9 more files with HIDDEN type errors)
   
   ### Fix approach
     1. Fix pyproject.toml dependency constraint
     2. Update type annotations in 11 audio/* files  ← AGENT IGNORES THIS
     3. Update test expectations in tests/           ← AGENT IGNORES THIS
   
   ### Post-fix patterns
     - [high] After fixing dependency, mypy will reveal type errors
     - [medium] After fixing types, pytest will reveal test failures"


┌────────────────────────────────────────────────────────────────┐
│  What Mini-SWE-Agent Actually Does                             │
└────────────────────────────────────────────────────────────────┘

Agent's behavior:
  1. Reads problem statement
  2. Sees "pip install failed on pyproject.toml"
  3. Fixes pyproject.toml only  ← Fixes Problem 1
  4. Runs verification (install passes ✓)
  5. STOPS and submits patch  ← NEVER ATTEMPTS Problems 2-3
  
Result:
  • CI failure 1 fixed ✓
  • CI failures 2-3 still broken ✗
  • Patch solves only 33% of the total problem


┌────────────────────────────────────────────────────────────────┐
│  Why This Happens                                               │
└────────────────────────────────────────────────────────────────┘

Mini-SWE-Agent default behavior:
  • Trained to fix "the bug" (singular)
  • Stops after first fix + verification passes
  • Does NOT understand multi-problem CI failures
  • Memory guidance is PASSIVE (shown in context)
  • Agent needs ACTIVE multi-problem repair loop
```

### The Missing Pieces

| What's Built | What's Missing | Impact |
|--------------|----------------|--------|
| ✅ Memory identifies 3 problems | ❌ Agent repair loop only fixes 1 problem | Solves 33% instead of 100% |
| ✅ Memory provides hidden files list | ❌ Agent doesn't proactively check hidden files | Ignores Problems 2-3 |
| ✅ Memory provides repair trajectory | ❌ Agent doesn't follow multi-step trajectory | Stops too early |
| ✅ Memory provides validation sequence | ❌ Agent runs only first validation | Misses downstream failures |

---

## 4. What Professor Wants To See Next

### Immediate Actions (for next meeting)

1. **Show baseline comparison**
   ```
   Experiment Structure:
   ├─ Baseline: Mini-SWE-Agent without memory
   │   └─ How many problems does it fix? (Expected: only visible problem)
   │
   ├─ Current System: Mini-SWE-Agent with memory context
   │   └─ How many problems does it fix? (Your current result)
   │
   └─ Gap Analysis:
       └─ Why doesn't agent use memory guidance to fix all problems?
   ```

2. **Fix the agent execution gap**
   ```python
   # Option 1: Modify agent prompt to require multi-problem fixing
   prompt = f"""
   This CI failure contains {len(atomic_problems)} distinct problems.
   You MUST fix ALL problems in sequence:
   
   Problem 1 (VISIBLE): {problem_1}
   Problem 2 (HIDDEN): {problem_2}  
   Problem 3 (HIDDEN): {problem_3}
   
   After fixing each problem, run its validation command.
   Do not stop until ALL problems are fixed.
   """
   
   # Option 2: Multi-iteration repair loop
   for problem in atomic_problems:
       agent.fix_problem(problem)
       agent.verify(problem.validation_cmd)
       if not passed:
           continue_fixing()
   ```

3. **Prepare 1-2 detailed case studies**
   ```
   Case Study: fish-speech Issue #410
   
   What memory provided:
     - 3 atomic problems (1 visible, 2 hidden)
     - File list for each problem
     - Repair trajectory
     - Validation commands
   
   What agent did:
     - Fixed Problem 1 only
     - Ignored Problems 2-3
   
   Why:
     - Agent stops after first validation passes
     - No multi-problem repair loop
   
   What would fix it:
     - Modify agent to iterate through all atomic problems
     - Run full validation sequence, not just first command
   ```

---

## 5. Key Metrics to Track

| Metric | Definition | Your Target |
|--------|------------|-------------|
| **Problem Decomposition Accuracy** | % of cases where your system correctly identifies ALL atomic problems (visible + hidden) | >90% |
| **Memory Retrieval Precision** | % of retrieved memories that are actually relevant | >70% |
| **Agent Multi-Problem Fix Rate** | % of cases where agent fixes ALL identified problems (not just first) | Currently ~33%, should be >80% |
| **Full CI Pass Rate** | % of cases where entire CI workflow passes after patch | This is your ultimate metric |

---

## 6. Simple Summary for Professor

**What You Built (Good):**
- ✅ System that decomposes CI failures into atomic problems (visible + hidden)
- ✅ L1/L2/L3 hierarchical memory with abstraction
- ✅ Retrieval that finds similar atomic problems from history
- ✅ Memory synthesis that explains root causes and repair strategies

**What's Broken (The Gap):**
- ❌ Mini-SWE-Agent only fixes the FIRST visible problem
- ❌ Agent ignores hidden problems even though memory provides them
- ❌ No multi-problem repair loop
- ❌ Agent stops after first validation passes, never attempts full workflow

**Why It's Broken:**
- Mini-SWE-Agent was designed for "one bug, one fix" scenarios
- Your CI failures are "one CI run, multiple bugs" scenarios
- The agent's default behavior (fix → verify → stop) doesn't match multi-problem repairs

**Next Steps:**
1. Add multi-problem repair loop to agent
2. Make agent iterate through ALL atomic problems from memory
3. Run full validation sequence (not just first command)
4. Compare baseline vs. your enhanced system properly

---

## For Your PPT: Use These Simple Diagrams

### Slide 1: Professor's Guidance vs. Implementation
- Checklist format showing ✅ Done and ❌ Missing items

### Slide 2: System Architecture
- Use the 3-box diagram above (Offline → Online → Example)

### Slide 3: The Problem
- Show the flow: Memory provides 3 problems → Agent fixes 1 → Gap identified

### Slide 4: Example Case
- Issue #410: fish-speech
- Show: 3 problems identified, only 1 fixed, why

### Slide 5: Next Steps
- Multi-problem repair loop
- Baseline comparison
- Target metrics
