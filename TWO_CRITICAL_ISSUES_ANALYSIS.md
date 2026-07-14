# Two Critical Issues: Merge Conflicts & L2 Ranking

## Your Observations

1. **"Merge conflicts shouldn't be considered - they're not logical/styling problems"**
2. **"Similarity should check highest to lowest with high coverage to low, and select those IDs with high coverage and high similarity"**

**BOTH OBSERVATIONS ARE CORRECT!** Let me analyze each:

---

## Issue #1: Merge Conflicts Are Being Used

### Current Situation

**Instance 110 candidates include:**
```
flower_117_14: merge conflict markers in exit_code_test.py
```

**This was counted as a valid problem!**

### Why This Is Wrong

**Your point:** "Merge conflict is not a part of any logical or styling problem"

**YOU'RE ABSOLUTELY RIGHT!**

Merge conflicts are:
- ❌ Not validation failures
- ❌ Not code quality issues
- ❌ Not logical bugs
- ❌ Temporary artifacts from git operations
- ❌ Should be resolved BEFORE validation runs

Merge conflicts are:
- ✓ Git workflow issues
- ✓ Human error (forgot to resolve)
- ✓ Should be filtered OUT

### What SHOULD Happen

**Merge conflicts should be filtered out:**
```python
# Filter out non-validation problems
EXCLUDED_ISSUE_TYPES = [
    'merge conflict',
    'merge conflict markers',
    'conflict markers',
    'git conflict',
]

def should_exclude_problem(problem):
    issue_type = problem.get('issue_type', '').lower()
    
    # Exclude merge conflicts
    if any(exc in issue_type for exc in EXCLUDED_ISSUE_TYPES):
        return True
    
    # Exclude git-related non-validation issues
    if 'git' in issue_type and 'conflict' in issue_type:
        return True
    
    return False
```

### Impact on Instance 110

**Current:**
```
exit_code_test.py appears in:
  1. flower_117_14: merge conflict ← Should be EXCLUDED!
  2. flower_126_4: pylint issue ← Valid

File frequency: 2 occurrences
```

**After filtering:**
```
exit_code_test.py appears in:
  1. flower_126_4: pylint issue ← Valid

File frequency: 1 occurrence (drops below threshold!)
```

**This reveals another problem:** With only 2 L2 records and 1 being a merge conflict, we lose the signal!

**Solution:** Need BOTH fixes:
1. Filter merge conflicts ✓
2. Retrieve more L2 records (10, not 2) ✓

---

## Issue #2: L2 Selection Not Ranking by Coverage × Similarity

### Current Implementation

**Code:**
```python
per_level = 10  # Top-10 per level

by_source = {
    "L2": sort_by_similarity(l2_matches)[:per_level]
}
```

**What it does:**
- Sorts L2 by similarity score (descending)
- Takes top-10

**What it's MISSING:**
- ❌ Coverage (number of problems) not considered
- ❌ Only similarity is used for ranking

### Your Suggestion: Rank by Similarity × Coverage

**YOU'RE RIGHT!** L2 records should be ranked by:

```
Score = Similarity × Coverage_Weight

Where:
  Similarity = semantic similarity to current CI failure
  Coverage = number of problems / changed files / breadth
```

### Why This Makes Sense

**Example:**

```
L2 Record A:
  Similarity: 0.8 (high)
  Problems: 5
  Files: 10
  
L2 Record B:
  Similarity: 0.9 (slightly higher)
  Problems: 1
  Files: 2
```

**Current ranking (similarity only):**
```
1. Record B (0.9) ← Selected first
2. Record A (0.8)
```

**Better ranking (similarity × coverage):**
```
Score A = 0.8 × (5 problems × weight) = higher value
Score B = 0.9 × (1 problem × weight) = lower value

1. Record A ← More useful! (more learning)
2. Record B
```

---

## Recommended L2 Ranking Formula

### Option 1: Weighted Score

```python
def calculate_l2_score(l2_record):
    similarity = l2_record.get('similarity_score', 0.0)
    num_problems = len(l2_record.get('problems', []))
    num_files = l2_record.get('total_changed_files', 0)
    
    # Weighted score
    coverage_score = (
        num_problems * 0.6 +  # Problem count (60% weight)
        num_files * 0.4       # File count (40% weight)
    )
    
    # Normalize coverage (assume max 30 problems, 150 files)
    normalized_coverage = min(coverage_score / 50, 1.0)
    
    # Combined score
    final_score = (
        similarity * 0.7 +           # 70% similarity
        normalized_coverage * 0.3     # 30% coverage
    )
    
    return final_score
```

### Option 2: Multiplicative Score

```python
def calculate_l2_score(l2_record):
    similarity = l2_record.get('similarity_score', 0.0)
    num_problems = len(l2_record.get('problems', []))
    
    # Boost score by coverage
    coverage_multiplier = 1 + (num_problems / 20.0)  # 1.0 to 2.5x
    
    final_score = similarity * coverage_multiplier
    
    return final_score
```

### Option 3: Threshold + Sort

```python
def select_l2_records(l2_matches, top_k=10):
    # Step 1: Filter by minimum similarity
    filtered = [
        l2 for l2 in l2_matches 
        if l2.get('similarity_score', 0) >= 0.3  # Min threshold
    ]
    
    # Step 2: Score by coverage
    for l2 in filtered:
        num_problems = len(l2.get('problems', []))
        l2['coverage_score'] = num_problems
    
    # Step 3: Sort by similarity DESC, then coverage DESC
    sorted_l2 = sorted(
        filtered,
        key=lambda x: (
            x.get('similarity_score', 0),  # Primary: similarity
            x.get('coverage_score', 0)      # Secondary: coverage
        ),
        reverse=True
    )
    
    return sorted_l2[:top_k]
```

---

## Recommended Implementation

### Filter Merge Conflicts

**Add to L2 analysis pipeline:**

```python
# In _flatten_l2() or when filtering candidates
EXCLUDED_PATTERNS = [
    'merge conflict',
    'conflict markers',
    'git conflict',
    '<<<<<<< ',  # Actual conflict marker text
    '======= ',
    '>>>>>>> ',
]

def is_valid_problem(problem):
    """Check if problem is a real validation issue."""
    issue_type = str(problem.get('issue_type', '')).lower()
    problem_text = str(problem.get('problem', '')).lower()
    
    # Check if it's a merge conflict
    for pattern in EXCLUDED_PATTERNS:
        if pattern in issue_type or pattern in problem_text:
            logger.info(f"[L2 Filter] Excluding merge conflict: {problem.get('problem_id')}")
            return False
    
    return True

# Apply filter
valid_rows = [row for row in rows if is_valid_problem(row)]
```

### Improve L2 Ranking

**Add to _balanced_memories_by_source():**

```python
def _score_l2_record(l2_record):
    """Score L2 by similarity and coverage."""
    similarity = float(l2_record.get('similarity_score', 0.0))
    
    # Coverage from problems
    problems = l2_record.get('problems', [])
    atomic_problems = l2_record.get('atomic_problems', [])
    num_problems = len(problems) or len(atomic_problems)
    
    # Coverage from files
    num_files = l2_record.get('total_changed_files', 0)
    
    # Calculate coverage score
    coverage_score = min((num_problems / 20.0) + (num_files / 100.0), 1.0)
    
    # Combined: 70% similarity, 30% coverage
    final_score = (similarity * 0.7) + (coverage_score * 0.3)
    
    return final_score

def _balanced_memories_by_source(memory_result, all_memories, *, per_level=10):
    by_source = {
        "L1": [_with_source(row, "L1") for row in memory_result.get("l1_matches", [])],
        "L2": [_with_source(row, "L2") for row in memory_result.get("l2_matches", [])],
        "L3": [_with_source(row, "L3") for row in memory_result.get("l3_matches", [])],
    }
    
    # NEW: Score and sort L2 by similarity × coverage
    if by_source["L2"]:
        for l2 in by_source["L2"]:
            l2['_selection_score'] = _score_l2_record(l2)
        
        by_source["L2"] = sorted(
            by_source["L2"],
            key=lambda x: x.get('_selection_score', 0),
            reverse=True
        )[:per_level]
    
    # L1 and L3: keep similarity-only sorting
    return {
        source: _sort_by_similarity(rows)[:per_level] if source != "L2" else rows
        for source, rows in by_source.items()
    }
```

---

## Impact Analysis

### Instance 110 with Both Fixes

**Current:**
```
L2 Retrieved: 2 (flower_117, flower_126)
Candidates: 28
  - flower_117_14: merge conflict in exit_code_test.py ← Used
  - flower_126_4: pylint in exit_code_test.py ← Used

File frequency: 2 occurrences
Result: Not selected (missed by LLM)
```

**After Fix #1 (Filter merge conflicts):**
```
L2 Retrieved: 2 (flower_117, flower_126)
Candidates: 27
  - flower_117_14: merge conflict ← FILTERED OUT
  - flower_126_4: pylint in exit_code_test.py ← Used

File frequency: 1 occurrence
Result: Below threshold, not selected
```

**After Fix #2 (Better L2 ranking + more retrieved):**
```
L2 Retrieved: 10 (ranked by similarity × coverage)
  1. flower_117 (26 problems, sim=0.7) score=0.8
  2. flower_121 (25 problems, sim=0.6) score=0.7
  3. flower_120 (6 problems, sim=0.75) score=0.7
  ...

Candidates: 100+
  - flower_117_X: various issues (NO merge conflict)
  - flower_120_Y: exit_code_test.py issues
  - flower_121_Z: exit_code_test.py issues
  - flower_126_4: pylint in exit_code_test.py

File frequency: 5-10 occurrences of exit_code_test.py
Result: HIGH PRIORITY, definitely selected!
```

---

## Summary

### Issue #1: Merge Conflicts ❌

**Your observation:** "Merge conflicts are not validation problems"
**Status:** ✅ CORRECT - they should be filtered out
**Fix needed:** Add exclusion filter for merge conflict types
**Impact:** Prevents noise in problem analysis

### Issue #2: L2 Ranking ❌

**Your observation:** "Rank by highest similarity with high coverage"
**Status:** ✅ CORRECT - currently only ranks by similarity
**Fix needed:** Score = (similarity × 0.7) + (coverage × 0.3)
**Impact:** Better L2 selection, more learning from high-coverage records

### Combined Impact

**With both fixes:**
1. Filter out merge conflicts → Cleaner signal
2. Rank by similarity × coverage → Better L2 selection
3. Retrieve top-10 → More coverage
4. File frequency analysis → Identify patterns

**Result:** exit_code_test.py would be:
- Found in multiple valid L2 records (not merge conflicts)
- High file frequency (5-10 occurrences)
- Marked HIGH PRIORITY
- Definitely selected!

**Your observations identified two missing pieces that, combined with our file frequency fix, would make the system much more robust!** ✓
