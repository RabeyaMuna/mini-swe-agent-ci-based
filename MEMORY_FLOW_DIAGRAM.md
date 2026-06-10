# Memory Flow Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MEMORY-ENHANCED CI REPAIR                         │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  CI FAILURE  │
│   (Input)    │
│              │
│ • CI log     │
│ • sha_fail   │
│ • workflow   │
│ • changed    │
│   files      │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE A: CI Log Analysis                                    │
│  (CILogAnalyzer)                                            │
├──────────────────────────────────────────────────────────────┤
│  Output:                                                      │
│  • overall_failure_reasons                                   │
│  • failed_jobs                                               │
│  • affected_files                                            │
│  • error_types                                               │
└──────┬───────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE B: Workflow Analysis                                  │
│  (CI Workflow Analyzer)                                      │
├──────────────────────────────────────────────────────────────┤
│  Output:                                                      │
│  • validation_sequence                                       │
│  • validation_commands                                       │
│  • installation_commands                                     │
└──────┬───────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE C: MEMORY RETRIEVAL & SYNTHESIS ⭐                    │
│  (CIMemorySystem)                                           │
└──────────────────────────────────────────────────────────────┘
       │
       ├─────► Step 1: Build Query
       │       └─ Extract failure context for embedding
       │
       ├─────► Step 2: Embed Query
       │       └─ sentence-transformers / fastembed
       │
       ├─────► Step 3: Search Memory
       │       ├─ L1: File-level (same repo/workflow)
       │       ├─ L2: Issue-level (same repo)
       │       └─ L3: Pattern-level (cross-repo)
       │
       ├─────► Step 4: Rank by Similarity
       │       └─ Cosine similarity scores
       │
       ├─────► Step 5: LLM Synthesis
       │       ├─ Input: CI context + memories
       │       ├─ Reasoning: What's really happening?
       │       └─ Output: guidance_document
       │
       └─────► Step 6: Format as Markdown
               └─ Convert to agent-readable format
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE D: Build Problem Statement                           │
│  (combine all phases)                                        │
├──────────────────────────────────────────────────────────────┤
│  Sections:                                                    │
│  1. CI Failure Report (Phase A)                             │
│  2. Validation Hints (Phase B)                              │
│  3. Memory Context (Phase C) ⭐                             │
└──────┬───────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  REPAIR AGENT                                                │
│  (receives problem statement with memory)                    │
├──────────────────────────────────────────────────────────────┤
│  Makes repair using:                                         │
│  • Current CI context                                        │
│  • Memory guidance (hidden failures, patterns, fixes)       │
└──────────────────────────────────────────────────────────────┘
```

---

## Detailed Memory Retrieval Flow

```
STEP 1: BUILD QUERY
═══════════════════════════════════════════════════════════════
Current CI Failure
│
├─ task_id: "issue_102"
├─ sha_fail: "abc123def"
├─ repo: "flower"
├─ workflow_path: ".github/workflows/test.yml"
│
├─ Failure Context:
│  ├─ error_type: ["type_error"]
│  ├─ failure_pattern: "mypy check failed"
│  ├─ overall_failure_reason: "Incompatible type List[int]"
│  ├─ relevant_files: ["src/client.py"]
│  ├─ failed_cmd: ["mypy src/"]
│  └─ failed_tool: ["mypy"]
│
└─ Query Text: "Type error in src/client.py: incompatible type 
              List[int] expected Sequence[int] in mypy check"


STEP 2: EMBED QUERY
═══════════════════════════════════════════════════════════════
Query Text
│
└─► sentence-transformers
    │
    └─► Embedding Vector [768 dimensions]
        [0.23, -0.45, 0.67, 0.12, ..., -0.89]


STEP 3: SEARCH MEMORY BANKS
═══════════════════════════════════════════════════════════════

┌────────────────────┐
│   L1: File-Level   │  Filter: same repo OR same workflow
│   Memory Bank      │  ───────────────────────────────────
│                    │  Search 500 L1 entries
│  Each entry:       │  │
│  • file path       │  ├─► Compute cosine similarity
│  • problem         │  │    with query embedding
│  • fix_strategy    │  │
│  • repo/workflow   │  └─► Rank by score
└────────┬───────────┘       │
         │                   ▼
         │              Top 3 matches:
         │              1. score=0.82 (src/client.py type error)
         │              2. score=0.76 (src/server.py type error)
         │              3. score=0.71 (src/utils.py type error)

┌────────────────────┐
│   L2: Issue-Level  │  Filter: same repo
│   Memory Bank      │  ─────────────────
│                    │  Search 100 L2 entries
│  Each entry:       │  │
│  • atomic_problems │  ├─► Compute cosine similarity
│  • repair_traj     │  │
│  • repo            │  └─► Rank by score
└────────┬───────────┘       │
         │                   ▼
         │              Top 3 matches:
         │              1. score=0.75 (22 files type migration)
         │              2. score=0.68 (API upgrade issue)
         │              3. score=0.62 (mypy config change)

┌────────────────────┐
│   L3: Pattern-Lvl  │  Filter: NONE (cross-repo)
│   Memory Bank      │  ─────────────────────────
│                    │  Search 50 L3 entries
│  Each entry:       │  │
│  • principle       │  ├─► Compute cosine similarity
│  • pattern         │  │
│  • fix_guide       │  └─► Rank by score
└────────┬───────────┘       │
         │                   ▼
         │              Top 3 matches:
         │              1. score=0.68 (covariance pattern)
         │              2. score=0.65 (dependency upgrade)
         │              3. score=0.58 (type system evolution)

         │
         ├────────────┐
         │            │
         ▼            ▼
    ┌─────────────────────────┐
    │  Combined Ranked List   │
    │  (9 total candidates)   │
    └────────┬────────────────┘
             │
             ▼


STEP 4: LLM SYNTHESIS
═══════════════════════════════════════════════════════════════

Input to LLM:
┌───────────────────────────────────────────────────────────┐
│ CURRENT CI CONTEXT:                                       │
│ • Failure: "Type error in src/client.py"                 │
│ • File: src/client.py                                    │
│ • Error: "List[int] vs Sequence[int]"                   │
│                                                           │
│ RETRIEVED MEMORIES (9 candidates):                       │
│                                                           │
│ [L1 - score 0.82]                                        │
│ • file: src/client.py                                    │
│ • problem: "Type error at line 45..."                   │
│ • fix: "Update List → Sequence..."                      │
│                                                           │
│ [L1 - score 0.76]                                        │
│ • file: src/server.py                                    │
│ • problem: "Type error at line 67..."                   │
│ • fix: "Update Dict → Mapping..."                       │
│                                                           │
│ [L2 - score 0.75]                                        │
│ • issue: 22 files type migration                        │
│ • pattern: "API upgrade requires covariant types"       │
│ • trajectory: "Fix install → types → tests"             │
│                                                           │
│ [L3 - score 0.68]                                        │
│ • pattern: "Type covariance after dependency upgrade"   │
│ • guide: "Use Sequence, Mapping, AbstractSet..."        │
└───────────────────────────────────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────────────────────┐
│ LLM REASONING:                                            │
│                                                           │
│ 1. Current failure: client.py line 45 type error        │
│ 2. L1 memory shows: server.py also has same pattern     │
│ 3. L2 memory shows: 22 files affected by API upgrade    │
│ 4. L3 pattern matches: covariance requirement           │
│                                                           │
│ CONCLUSION:                                               │
│ • Root cause: API dependency upgrade                     │
│ • Hidden failures: server.py, utils.py (not in log)    │
│ • Pattern: List→Sequence, Dict→Mapping                  │
│ • Order: Fix all 3 files together                       │
└───────────────────────────────────────────────────────────┘
             │
             ▼
┌───────────────────────────────────────────────────────────┐
│ LLM OUTPUT (guidance_document):                          │
│                                                           │
│ {                                                         │
│   "diagnosis": "API upgrade requires type updates...",   │
│   "primary_files": [                                     │
│     {"file": "src/client.py", "fix": "..."}            │
│   ],                                                      │
│   "additional_files": [                                  │
│     {"file": "src/server.py", "fix": "..."},           │
│     {"file": "src/utils.py", "fix": "..."}             │
│   ],                                                      │
│   "linked_issues": [...],                               │
│   "fix_approach": ["step 1", "step 2", "step 3"],      │
│   "verification": {"command": "mypy src/"},             │
│   "confidence": "high"                                   │
│ }                                                         │
└───────────────────────────────────────────────────────────┘


STEP 5: FORMAT AS MARKDOWN
═══════════════════════════════════════════════════════════════

guidance_document
│
└─► format_memory_context()
    │
    └─► Markdown Output
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ ## Memory Context — Repair Guidance from Past Experience│
│                                                          │
│ **Confidence:** 🟢 HIGH                                 │
│                                                          │
│ ### What is really happening                            │
│ Root cause: API upgrade changed type signatures...     │
│                                                          │
│ ### Primary files to inspect first                      │
│   - `src/client.py` — Mentioned in CI log              │
│     → **prior fix:** Update List → Sequence            │
│                                                          │
│ ### Files to fix — including those NOT in the log      │
│   - `src/server.py` — Not in log but needs fix        │
│     → **fix:** Update Dict → Mapping at line 67       │
│   - `src/utils.py` — Not in log but needs fix         │
│     → **fix:** Update Set → AbstractSet               │
│                                                          │
│ ### Fix Approach                                        │
│   1. Fix all 3 files together                          │
│   2. Pattern: List→Sequence, Dict→Mapping             │
│   3. Run mypy src/ to verify                           │
│                                                          │
│ ### How to Verify the Fix                              │
│ **Command:** `mypy src/`                               │
│ **Expected:** Success: no issues found                 │
└─────────────────────────────────────────────────────────┘
```

---

## Memory Content Impact

```
┌─────────────────────────────────────────────────────────────┐
│  WITHOUT ENHANCED CONTENT (Current - Vague)                 │
└─────────────────────────────────────────────────────────────┘

L1 Memory Entry:
{
  "problem": "Type error",
  "fix_strategy": "Fixed types"
}
         │
         ▼
LLM Synthesis gets vague input
         │
         ▼
Guidance Document:
{
  "diagnosis": "Some type error occurred",
  "fix_approach": ["Fix types in the file"]
}
         │
         ▼
Agent receives: Generic, not actionable


┌─────────────────────────────────────────────────────────────┐
│  WITH ENHANCED CONTENT (After Our Changes)                  │
└─────────────────────────────────────────────────────────────┘

L1 Memory Entry:
{
  "problem": "Type error at line 45 in client.py. Symptom: 
              mypy 'incompatible type List[int]; expected 
              Sequence[int]'. Root cause: API upgrade requires 
              covariant types per PEP-484.",
              
  "fix_strategy": "Update annotation at line 45. Step 1: 
                   Change 'List[int]' to 'Sequence[int]'. 
                   Step 2: Import Sequence from typing. 
                   Before: 'def process(data: List[int])'. 
                   After: 'def process(data: Sequence[int])'. 
                   This works because Sequence is covariant. 
                   Verification: Run 'mypy src/'."
}
         │
         ▼
LLM Synthesis gets concrete input
         │
         ▼
Guidance Document:
{
  "diagnosis": "API upgrade requires covariant type changes",
  "primary_files": [
    {
      "file": "src/client.py",
      "reason": "Line 45: List[int] → Sequence[int]",
      "fix": "Update type annotation with covariant type"
    }
  ],
  "fix_approach": [
    "1. Update line 45: List[int] → Sequence[int]",
    "2. Import Sequence from typing",
    "3. Verify with mypy src/"
  ]
}
         │
         ▼
Agent receives: Concrete, actionable, specific
```

---

## Summary

**Memory helps by:**
1. ✅ Revealing hidden failures not in CI log
2. ✅ Providing concrete fix patterns
3. ✅ Suggesting repair order
4. ✅ Giving verification commands
5. ✅ Showing confidence level

**Enhanced content improves:**
1. ✅ Similarity matching (concrete symptoms match better)
2. ✅ LLM synthesis (better input → better output)
3. ✅ Agent repairs (actionable guidance)

**Next optimization:**
- Pattern-based L1 for large changes (110 files)
- Workflow-aware retrieval (validation step context)
