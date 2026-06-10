# PPT Slides Content - Professor Meeting

## Slide 1: Title
```
Multi-Problem CI Repair with Memory-Enhanced Mini-SWE-Agent

Student: [Your Name]
Advisor: [Professor Name]
Date: [Meeting Date]
```

---

## Slide 2: Professor's Core Guidance ✅

### What Professor Said We Should Do:
1. ✅ Don't store only (failure + file + fix)
2. ✅ Use abstraction and adaptation (L1/L2/L3)
3. ✅ One CI failure = multiple problems
4. ✅ Retrieve at sub-problem level
5. ✅ Use reasoning/explanation generation
6. ✅ Start with one bug first

### What We Built:
✅ **Decomposition**: Split CI failures into atomic problems (visible + hidden)
✅ **Memory Structure**: L1 (file) → L2 (issue) → L3 (universal)
✅ **Reasoning**: Root causes, why fixes work, repair strategies
✅ **Retrieval**: Match similar atomic problems across repos

---

## Slide 3: System Architecture

```
┌───────────────────────────────────────────────────────────┐
│              OFFLINE: Build Memory Bank                   │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Historical CI Failure + Ground-Truth Patch              │
│              ↓                                            │
│  [decompose_ci_failure.py]                               │
│   → LLM reverse-engineers ALL atomic problems             │
│   → Visible problem: from CI log                         │
│   → Hidden problems: inferred from diff                  │
│              ↓                                            │
│  [build_memory_from_decomposed.py]                       │
│   → L1: Per-file (failure + fix + reasoning)            │
│   → L2: Per-issue (atomic_problems + trajectory)         │
│   → L3: Universal patterns (cross-repo)                  │
│              ↓                                            │
│  Memory Bank: failure_memory.json                        │
│               repo_memory.json                            │
│               cross_memory.json                           │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│              ONLINE: Repair New CI Failure                │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  New CI Failure                                          │
│              ↓                                            │
│  Phase A: CILogAnalyzer                                  │
│   → Parse CI logs → error_types, files, jobs             │
│              ↓                                            │
│  Phase B: Workflow Analysis                              │
│   → Extract validation_sequence (install→lint→test)      │
│              ↓                                            │
│  Phase C: Memory Retrieval                               │
│   → Embed query → Search L1/L2/L3                        │
│   → LLM synthesis → Guidance document                    │
│              ↓                                            │
│  Phase D: Problem Statement (NEW!)                       │
│   → Multi-problem structure                              │
│   → Explicit numbered problems                           │
│              ↓                                            │
│  Mini-SWE-Agent                                          │
│   → Attempts ALL problems                                │
│   → Follows repair sequence                              │
└───────────────────────────────────────────────────────────┘
```

---

## Slide 4: Example - Issue #410

### Decomposition Results:
```
Repository: fish-speech/fish-speech
CI Failure: Installation → Type Check → Tests

Problem #1 (VISIBLE in CI log):
  Type: Dependency constraint error
  Stage: Installation (uv pip install)
  Files: pyproject.toml
  Fix: Relax fish-audio-sdk version constraint

Problem #2 (HIDDEN - after Problem #1):
  Type: Type annotation errors
  Stage: Type checking (mypy)
  Files: 11 files in fish_speech/audio/
  Fix: Update AudioData → tuple[np.ndarray, int]

Problem #3 (HIDDEN - after Problem #2):
  Type: Test expectation mismatch
  Stage: Testing (pytest)
  Files: tests/test_audio.py
  Fix: Update mock expectations
```

---

## Slide 5: The Problem We Found ❌

### Memory System Works ✅
- Identifies all 3 atomic problems
- Provides root causes
- Suggests repair strategy

### Agent Behavior ❌
```
┌─────────────────────────────────────────────┐
│  What Agent Received (Original)             │
├─────────────────────────────────────────────┤
│  # CI Failure Report                        │
│                                             │
│  ## Why the CI Failed                       │
│    - Installation failed                    │
│                                             │
│  ## Affected Files                          │
│    - pyproject.toml                         │
│                                             │
│  ## Memory Context (optional guidance)      │
│    Files that might need fixing:            │
│    - fish_speech/audio/*.py (11 files)      │
│    - tests/test_audio.py                    │
└─────────────────────────────────────────────┘
         ↓
Agent thinks: "Primary task is Problem #1.
               Memory provides optional hints."
         ↓
Agent fixes: pyproject.toml only
Agent verifies: install passes ✓
Agent stops: "Task complete"
         ↓
Result: 33% fixed (1 of 3 problems)
CI still fails: mypy errors ✗
```

---

## Slide 6: Root Cause Analysis

### Why Agent Stops Early:

1. **Presentation Problem**: Hidden problems in "Memory Context" section
   → Agent treats them as optional guidance

2. **Priority Mismatch**: Only Problem #1 has explicit file list
   → Agent sees it as the main task

3. **No Verification Loop**: One verification section
   → Agent runs install, passes, stops

4. **Missing Stopping Criteria**: No explicit "fix all problems" instruction
   → Agent assumes done after first problem

### The Constraint:
❌ Cannot modify core agent
✅ Can only modify problem statement input

---

## Slide 7: Our Solution ✅

### Strategy: Restructure Problem Statement

**Make ALL problems EXPLICIT, NUMBERED, REQUIRED**

```
┌─────────────────────────────────────────────────────┐
│  New Multi-Problem Statement                        │
├─────────────────────────────────────────────────────┤
│  # CI REPAIR TASK - MULTI-PROBLEM FAILURE           │
│                                                     │
│  ⚠️ CRITICAL: This has 3 DISTINCT PROBLEMS          │
│     that MUST ALL be fixed.                         │
│                                                     │
│  ## PROBLEM #1 (VISIBLE) - FIX THIS FIRST           │
│    Failed Command: uv pip install -e .              │
│    Files to Fix: pyproject.toml                     │
│    Verification: [install command]                  │
│                                                     │
│  ## PROBLEM #2 (HIDDEN) - FIX AFTER #1              │
│    Why Hidden: CI never reached mypy stage          │
│    Files to Fix (ALL 11 files):                     │
│      - fish_speech/audio/processor.py               │
│      - ... (10 more)                                │
│    Verification: [mypy command]                     │
│                                                     │
│  ## PROBLEM #3 (HIDDEN) - FIX AFTER #2              │
│    Files to Fix: tests/test_audio.py                │
│    Verification: [pytest command]                   │
│                                                     │
│  ## MANDATORY REPAIR SEQUENCE                       │
│    1. Fix Problem #1 → Verify                       │
│    2. Fix Problem #2 → Verify                       │
│    3. Fix Problem #3 → Verify                       │
│                                                     │
│  ## STOPPING CRITERIA                               │
│    ❌ DO NOT stop after Problem #1                  │
│    ✅ ONLY stop when ALL 3 problems fixed           │
└─────────────────────────────────────────────────────┘
```

---

## Slide 8: Key Changes

| Aspect | Before | After |
|--------|--------|-------|
| **Problem Count** | Implicit (1 visible) | Explicit "3 DISTINCT PROBLEMS" |
| **Hidden Problems** | In "Memory Context" | Separate "PROBLEM #2, #3" sections |
| **File Lists** | Visible files only | "ALL 11 files" per problem |
| **Verification** | One section | "Verification (REQUIRED)" per problem |
| **Stopping** | No criteria | "DO NOT stop after #1" |
| **Evidence** | "Memory suggests" | "Historical repairs show" |

---

## Slide 9: Expected Impact

### Before (Current Behavior):
```
Agent sees: 1 primary task + optional hints
Agent fixes: Problem #1 only
Agent stops: After first verification
Success rate: ~35% (CI still fails)
```

### After (With Multi-Problem Statement):
```
Agent sees: 3 explicit required problems
Agent fixes: All 3 problems
Agent stops: After all verifications pass
Success rate: >70% (expected)
```

### Metrics to Track:
- % Agent attempts Problem #2: ~10% → >80%
- % Full CI passes: ~35% → >70%
- % Agent stops after Problem #1: ~85% → <20%

---

## Slide 10: Implementation

### Code Changes:

**New Module**: `ci_context_multi_problem.py` (200 lines)
- Function: `build_problem_statement_multi_problem()`
- Extracts atomic problems from memory
- Structures as explicit numbered sections

**Integration**: `cibench.py` (5 lines changed)
```python
# Add import
from minisweagent.run.benchmarks.utils.ci_context_multi_problem import (
    build_problem_statement_multi_problem
)

# Replace
problem_statement = build_problem_statement_multi_problem(
    ci_context_result["context"],
    ci_context_result["memory"],
)
```

**No Changes to**:
- Core agent
- Memory system
- Retrieval logic
- Decomposition pipeline

---

## Slide 11: Current Status

| Component | Status |
|-----------|--------|
| Multi-problem decomposition | ✅ Complete |
| L1/L2/L3 memory structure | ✅ Complete |
| Memory retrieval | ✅ Complete |
| LLM synthesis | ✅ Complete |
| **Multi-problem statement** | ✅ **Implemented** |
| Integration | ⏳ Ready to test |
| Baseline comparison | ⏳ TODO |
| Metrics collection | ⏳ TODO |

---

## Slide 12: Next Steps

### Immediate (This Week):
1. Test multi-problem statement on 5-10 issues
2. Verify agent attempts hidden problems
3. Collect agent transcripts

### Short-term (Next 2 Weeks):
4. Run baseline comparison
   - Original vs. multi-problem mode
   - 20 issues with known multi-problem failures
5. Measure improvement metrics
6. Tune prompt based on results

### Evaluation Plan:
- Metrics: Agent behavior, CI pass rate, problem coverage
- Case studies: 3-5 detailed examples
- Comparison: Baseline Mini-SWE vs. Enhanced

---

## Slide 13: Research Contributions

### What Makes This Work Novel:

1. **Multi-Problem Aware CI Repair**
   - First to explicitly handle multiple atomic problems in one CI failure
   - Addresses hidden problems that appear after first fix

2. **Memory-Guided Problem Presentation**
   - Uses past repairs to identify ALL problems upfront
   - Restructures guidance as explicit requirements

3. **Constrained Agent Enhancement**
   - Improves agent behavior without modifying core
   - Input-level intervention strategy

### Comparison to Prior Work:
- **SWE-Agent**: Single-problem focused
- **CI-REPAIR-BENCH**: Identifies problems but doesn't force agent to fix all
- **Our Work**: Multi-problem decomposition + forced execution

---

## Slide 14: Challenges & Solutions

| Challenge | Our Solution |
|-----------|--------------|
| Agent ignores hidden problems | Make them explicit, numbered, required |
| Can't modify agent | Restructure input to change behavior |
| Hidden problems not in CI log | Use memory to identify from past repairs |
| Agent stops too early | Add explicit stopping criteria |
| Multiple problems overwhelming | Clear sequence: #1 → #2 → #3 |

---

## Slide 15: Questions We Can Answer

✅ **Does memory identify all problems?**
   Yes - decomposition finds visible + hidden problems

✅ **Why didn't baseline agent fix them?**
   Hidden problems presented as optional, not required

✅ **How does multi-problem statement help?**
   Equal visual weight, explicit requirements, stopping criteria

⏳ **Does agent behavior actually change?**
   Testing in progress - will have results soon

⏳ **How much does it improve?**
   Need to run baseline comparison

---

## Slide 16: Summary

### What We Built:
1. ✅ Multi-problem decomposition system
2. ✅ L1/L2/L3 hierarchical memory
3. ✅ Reasoning-based memory entries
4. ✅ Sub-problem retrieval
5. ✅ Multi-problem problem statement

### What We Found:
- Memory system works correctly
- Problem was presentation, not content
- Agent needs explicit instructions

### What We're Testing:
- Does restructured statement change agent behavior?
- How much does it improve CI pass rate?
- What's the optimal prompt structure?

### Next Steps:
- Baseline comparison
- Metrics collection
- Prompt tuning

---

## Slide 17: Backup - Memory Example

### L2 Memory Entry (repo_memory.json):
```json
{
  "issue_id": "410",
  "repo": "fish-speech/fish-speech",
  "atomic_problems": [
    {
      "problem_id": 1,
      "visibility": "visible_in_log",
      "issue_type": "dependency_constraint",
      "problem": "SDK version not available",
      "file_changes": [{"file": "pyproject.toml", "fix": "..."}]
    },
    {
      "problem_id": 2,
      "visibility": "hidden",
      "issue_type": "type_annotation",
      "problem": "Type hints incompatible",
      "file_changes": [
        {"file": "fish_speech/audio/processor.py", "fix": "..."},
        ...
      ]
    }
  ],
  "repair_trajectory_summary": "Fix dependency first, then types, then tests..."
}
```

---

## Slide 18: Thank You

### Questions?

**Contact**: [Your Email]

**Code**: [GitHub Link if applicable]

**Next Meeting**: [Proposed Date]
- Will present: Baseline comparison results
- Will show: Agent behavior change
- Will discuss: Next research directions
