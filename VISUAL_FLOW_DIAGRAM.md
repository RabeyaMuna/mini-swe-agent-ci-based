# Visual Flow Diagram - L2 Analysis with File Frequency Fix

## The Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Current CI Failure + L2 Memories (Past Repair Records)  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  Current CI Failure           │
         │  problem_1 = {                │
         │    files: ["current_file.py"] │
         │    problem: "Type error"      │
         │  }                            │
         └───────────────┬───────────────┘
                         │
                         │  (Passed to consecutive selection)
                         │
         ┌───────────────▼───────────────────────────────────┐
         │  L2 Memories (Past Repair Trajectories)           │
         │                                                    │
         │  flower_126:                                      │
         │    Step 1 [PRIMARY]: "Pylint in exit_code_test.py"│
         │    Step 2 [CONSEC]:  "Import error in utils.py"  │
         │                                                    │
         │  flower_120:                                      │
         │    Step 1 [PRIMARY]: "Docstring in exit_code_test.py"│
         │    Step 2 [CONSEC]:  "Type error in typing.py"   │
         │                                                    │
         │  flower_117:                                      │
         │    Step 1 [PRIMARY]: "Merge conflict exit_code_test.py"│
         │    Step 2 [CONSEC]:  "Test failure in test_*.py" │
         └────────────────────┬──────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │  STEP 1: Flatten L2       │
              │  Extract ALL problems     │
              │  from ALL steps           │
              └──────┬────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────┐
    │  Flattened Problem Rows:                       │
    │  1. {problem: "Pylint...", is_primary: True}   │
    │  2. {problem: "Import...", is_primary: False}  │
    │  3. {problem: "Docstring...", is_primary: True}│
    │  4. {problem: "Type error...", is_primary: False}│
    │  5. {problem: "Merge...", is_primary: True}    │
    │  6. {problem: "Test fail...", is_primary: False}│
    └────────────┬───────────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────────────────┐
    │  STEP 2: Split into Two Candidate Pools        │
    └─────┬──────────────────────────────────┬───────┘
          │                                  │
          ▼                                  ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│ COMMON CANDIDATES       │    │ CONSECUTIVE CANDIDATES  │
│ (include_primary=True)  │    │ (include_primary=False) │
├─────────────────────────┤    ├─────────────────────────┤
│ • Pylint exit_code_test │    │ • Import utils.py       │
│ • Import utils.py       │    │ • Type error typing.py  │
│ • Docstring exit_code_test│  │ • Test failure test_*.py│
│ • Type error typing.py  │    │                         │
│ • Merge exit_code_test  │    │ (Only Step 2+ problems) │
│ • Test failure test_*.py│    │                         │
│                         │    │                         │
│ (All problems from L2)  │    │                         │
└──────────┬──────────────┘    └──────────┬──────────────┘
           │                              │
           │                              │
    ┌──────▼──────────────┐      ┌────────▼────────────┐
    │ STEP 3a: Cluster    │      │ STEP 3b: Group      │
    │ by Similarity       │      │ Consecutive         │
    │ (embedding 0.6)     │      │                     │
    └──────┬──────────────┘      └────────┬────────────┘
           │                              │
           ▼                              ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ Cluster 1:      │           │ Group 1:        │
    │ - Pylint exit_  │           │ - Import utils  │
    │ - Docstring exit│           │ Group 2:        │
    │ - Merge exit_   │           │ - Type typing   │
    │ Cluster 2:      │           │ Group 3:        │
    │ - Import utils  │           │ - Test fail     │
    │ Cluster 3:      │           │                 │
    │ - Type typing   │           └────────┬────────┘
    │ Cluster 4:      │                    │
    │ - Test fail     │                    │
    └──────┬──────────┘                    │
           │                               │
           │ ┌─────────────────────────────┘
           │ │
           ▼ ▼
    ┌──────────────────────────────────────────┐
    │ *** OUR FIX: File Frequency Analysis *** │
    │                                          │
    │ Analyze which files appear in multiple   │
    │ clusters/groups:                         │
    │                                          │
    │ exit_code_test.py: 3 occurrences        │
    │   - Cluster 1 (Pylint)                  │
    │   - Cluster 1 (Docstring)               │
    │   - Cluster 1 (Merge)                   │
    │   Priority: HIGH ⚠️                      │
    │                                          │
    │ utils.py: 1 occurrence                  │
    │ typing.py: 1 occurrence                 │
    │ test_*.py: 1 occurrence                 │
    └──────────┬───────────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────────┐
    │ STEP 4: LLM Selection (with file freq)   │
    └─────┬─────────────────────────────┬──────┘
          │                             │
          ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│ LLM Select Common    │    │ LLM Select Consecutive│
│                      │    │                      │
│ Prompt includes:     │    │ Prompt includes:     │
│ ┌──────────────────┐ │    │ ┌──────────────────┐ │
│ │ FILE FREQUENCY:  │ │    │ │ CURRENT CI:      │ │
│ │ exit_code_test.py│ │    │ │ problem_1        │ │
│ │ 3 problems (HIGH)│ │    │ │                  │ │
│ │ ⚠️ PRIORITY      │ │    │ │ FILE FREQUENCY:  │ │
│ └──────────────────┘ │    │ │ (if any in consec)│ │
│                      │    │ └──────────────────┘ │
│ [All clusters...]    │    │ [All groups...]      │
│                      │    │                      │
│ Selection:           │    │ Selection:           │
│ ✓ exit_code_test    │    │ ✓ utils.py (maybe)  │
│ ✓ utils.py          │    │ ✗ typing.py         │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           └────────┬──────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ STEP 5: Merge &      │
         │ Deduplicate          │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────────────┐
         │ *** OUR FIX: Validation *** │
         │                              │
         │ Check high-frequency files:  │
         │ exit_code_test.py (freq=3)   │
         │ ✓ In selection? YES          │
         │                              │
         │ If NO → Force-add best       │
         │ candidate for that file      │
         └──────────┬───────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ STEP 6: Normalize    │
         │ to Final Format      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ OUTPUT: Final        │
         │ Problems to Fix      │
         │                      │
         │ 1. exit_code_test.py │
         │ 2. utils.py          │
         │ ...                  │
         └──────────────────────┘
```

## Key Points Illustrated

### 1. Two Candidate Pools

**Common Candidates:**
- Contains ALL problems (primary + consecutive) from L2
- Purpose: Find patterns that appear across multiple L2 records
- exit_code_test.py appears here because it's in primary steps

**Consecutive Candidates:**
- Contains ONLY non-primary (Step 2+) problems from L2
- Purpose: Find problems that may appear after fixing current CI failure
- exit_code_test.py does NOT appear here (it's primary, not consecutive)

### 2. Where Our Fix Applies

**File Frequency Analysis** happens in BOTH paths:
```
Common Selection:
  ┌─────────────────────────┐
  │ Clusters → File Freq   │
  │ exit_code_test.py: 3   │  ← HIGH PRIORITY
  │ → LLM sees priority    │
  └─────────────────────────┘

Consecutive Selection:
  ┌─────────────────────────┐
  │ Groups → File Freq     │
  │ (analyze consecutive    │
  │  problems only)         │
  │ → LLM sees priority    │
  └─────────────────────────┘
```

### 3. The Safeguard (Validation)

After LLM selection, check:
```
For each high-frequency file (3+ occurrences):
  ✓ Is it in the selection?
  ✗ If not → Force-add best candidate

Example:
  exit_code_test.py appears 3 times
  LLM selected 0/3 problems for it
  → Safeguard adds 1 problem (best candidate)
```

## exit_code_test.py Case Flow

```
L2 Records:
  flower_126 Step 1: exit_code_test.py (Pylint)
  flower_120 Step 1: exit_code_test.py (Docstring)  
  flower_117 Step 1: exit_code_test.py (Merge)
            │
            ▼
    All are PRIMARY (Step 1)
            │
            ▼
    Go to COMMON candidates
    (not CONSECUTIVE)
            │
            ▼
    Clustered separately:
    - Cluster A: Pylint issue
    - Cluster B: Docstring issue
    - Cluster C: Merge conflict
            │
            ▼
    *** OUR FIX ***
    File Frequency Analysis:
    exit_code_test.py appears in 3 clusters
    → HIGH PRIORITY
            │
            ▼
    LLM sees:
    "FILE FREQUENCY: exit_code_test.py 3 problems"
    → More likely to select
            │
            ▼
    If LLM still misses it:
    Validation adds it (freq >= 3)
            │
            ▼
    ✓ exit_code_test.py in final selection!
```

## Before vs After Fix

### BEFORE (Broken)
```
Clusters:
  A: [exit_code_test.py] freq=1
  B: [exit_code_test.py] freq=1
  C: [exit_code_test.py] freq=1
  D: [ndarrays.py] freq=3      ← Selected (high freq)

LLM sees: Individual clusters, doesn't know same file
Result: Selects D (freq=3), skips A,B,C (freq=1 each)
```

### AFTER (Fixed)
```
Clusters:
  A: [exit_code_test.py] freq=1
  B: [exit_code_test.py] freq=1
  C: [exit_code_test.py] freq=1
  D: [ndarrays.py] freq=3

File Frequency:
  exit_code_test.py: 3 clusters ← HIGH PRIORITY
  ndarrays.py: 1 cluster

LLM sees: "exit_code_test.py HIGH PRIORITY (3 problems)"
Result: Selects problems for exit_code_test.py + D

If LLM misses: Validation force-adds exit_code_test.py
```

## Summary

✅ **Common vs Consecutive:** Both improved, exit_code_test.py is in common path
✅ **File Frequency:** Counted across clusters, identifies cascading failures  
✅ **Two Fixes:** LLM prompt enhancement + validation safeguard
✅ **Backward Compatible:** No breaking changes, graceful degradation
✅ **Addresses Root Cause:** LLM now sees file-level frequency signal
