# Temporal Data Leakage Analysis

## Summary

This document analyzes the potential temporal data leakage in the current memory/evaluation split approach and provides recommendations for addressing it.

---

## Current Implementation Analysis

### Current Approach

Based on the code in [`scripts/prepare_memory_train_test_split.py`](../scripts/prepare_memory_train_test_split.py):

1. **Load dataset**: All CI issues from HuggingFace dataset (`ci-benchmark-user/ci-repair-bench`)
2. **Compute embeddings**: Use decomposed issue data (problems, root causes, fixes) + raw data
3. **Group by repository**: Issues are grouped by `{repo_owner}/{repo_name}`
4. **Similarity-based selection within each repo**:
   - Compute cosine similarity between all issues in the same repo
   - Calculate average similarity for each issue
   - Sort by average similarity (descending)
   - Select top 30% as "memory set" (most representative issues)
   - Remaining 70% becomes "evaluation set"

### Current Statistics (from `data/trs/split_metadata.json`)

```json
{
  "total_issues": 126,
  "memory_set_size": 37,
  "eval_set_size": 89,
  "memory_ratio": 0.294,
  "split_by_repo": true,
  "selection_strategy": "highest_avg_similarity"
}
```

**Repository breakdown:**
- `agno-agi/agno`: 84 issues (25 memory, 59 eval, avg_sim=0.609)
- `adap/flower`: 27 issues (8 memory, 19 eval, avg_sim=0.626)
- `camel-ai/camel`: 15 issues (4 memory, 11 eval, avg_sim=0.572)

---

## The Temporal Leakage Problem

### What is Temporal Data Leakage?

Temporal data leakage occurs when **future information is used to make predictions about the past**. In the context of CI issue resolution:

**Leakage scenario:**
```
Timeline:
  2015 ─── Issue A (eval)
  2018 ─── Issue B (eval)  
  2020 ─── Issue C (memory) ← SELECTED due to high similarity
  2022 ─── Issue D (memory)

Problem:
  When evaluating Issue A (2015), the system may retrieve 
  Issue C (2020) from memory as a "historical example"
  
  -> This is INVALID because C didn't exist when A occurred
```

### Why This Matters

1. **Unrealistic Performance Estimates**: The reported success rate will be **artificially inflated** because the model has access to solutions that didn't exist at the time of the problem.

2. **Deployment Mismatch**: In real-world deployment:
   - When a CI failure occurs in 2024, we can only use past issues (< 2024) as memory
   - But the evaluation suggests we'd have access to future solutions

3. **Invalid Scientific Claims**: Any published results would be methodologically flawed and not reproducible under realistic conditions.

### Does Current Implementation Have This Issue?

**YES - potentially severe leakage.**

The current implementation:
- OK Groups by repository (good - prevents cross-repo leakage)
- FAIL **Does NOT consider timestamp** when selecting memory vs eval
- FAIL Sorts purely by similarity score
- FAIL Can select newer issues for memory and older issues for eval

**Example of potential leakage:**
```python
# Current logic (simplified)
similarity_scores = [0.9, 0.7, 0.85, 0.65]  # for issues with timestamps:
timestamps =         [2020, 2015, 2018, 2016]

sorted_by_sim =     [0.9,  0.85, 0.7,  0.65]
corresponding_ts =  [2020, 2018, 2015, 2016]

# Top 30% -> memory
memory = [2020, 2018]  # Issues from 2020 and 2018

# Bottom 70% -> eval  
eval = [2015, 2016]    # Issues from 2015 and 2016

# LEAKAGE: Evaluating 2015 issue can retrieve 2020 memory!
```

---

## Dataset Analysis

### Available Timestamp Information

**Dataset columns (from HuggingFace `ci-repair-bench`):**

```python
columns = [
  'language', 'id', 'repo_owner', 'repo_name', 
  'head_branch', 'workflow_name', 'workflow_filename', 'workflow_path',
  'sha_fail',      # ← Commit SHA where CI failed
  'sha_success',   # ← Commit SHA where CI was fixed
  'workflow', 'logs', 'diff', 'changed_files',
  'commit_link',   # ← GitHub link: https://github.com/{owner}/{repo}/tree/{sha}
  'error_type'
]
```

**Timestamp availability:**
- FAIL No direct timestamp field in dataset
- OK Have `commit_link` and `sha_fail` for each issue
- OK Can fetch commit timestamp from GitHub API:
  ```
  GET https://api.github.com/repos/{owner}/{repo}/commits/{sha}
  -> returns commit.author.date (ISO timestamp)
  ```

**Current state:**
- `data/decomposed_issues.json`: NO timestamp fields
- `data/trs/memory_set.jsonl`: NO timestamp fields
- `data/trs/eval_set.jsonl`: NO timestamp fields

**Conclusion:** Timestamps were **not collected during data preparation** but **CAN be retrieved** from GitHub API.

---

## Recommended Solution

### 1. Fetch and Cache Commit Timestamps

**Tool provided:** `scripts/analyze_temporal_leakage.py`

```bash
# Step 1: Fetch timestamps from GitHub API and analyze current split
python scripts/analyze_temporal_leakage.py \
    --memory-ids data/trs/memory_issue_ids.json \
    --eval-ids data/trs/eval_issue_ids.json \
    --output data/trs/temporal_analysis.json \
    --timestamp-cache data/trs/commit_timestamps.json
```

**What this does:**
- Fetches commit timestamps from GitHub API for all 126 issues
- Caches results in `commit_timestamps.json` (reusable)
- Analyzes current split for temporal leakage
- Reports:
  - How many eval issues have "future memory"
  - Date ranges for memory vs eval sets
  - Specific leakage examples
- Provides chronological split recommendation

**Expected output:**
```json
{
  "leakage_analysis": {
    "leakage_count": 45,
    "leakage_percentage": 50.6,
    "leakage_examples": [...]
  },
  "chronological_split_suggestion": {
    "cutoff_date": "2021-03-15",
    "memory_ids": [...],
    "eval_ids": [...]
  }
}
```

### 2. Implement Chronological Splitting

**Two strategies to choose from:**

#### Strategy A: Pure Chronological (Strictest - Recommended)

```python
# Step 1: Sort ALL issues by commit timestamp
issues_sorted = sorted(issues, key=lambda x: x['commit_timestamp'])

# Step 2: Split chronologically
cutoff_idx = int(len(issues_sorted) * 0.3)
memory_set = issues_sorted[:cutoff_idx]  # Earliest 30%
eval_set = issues_sorted[cutoff_idx:]    # Latest 70%

# Step 3: During evaluation, filter memory by timestamp
def retrieve_memory(eval_issue):
    eval_timestamp = eval_issue['commit_timestamp']
    
    # ONLY consider memory from before eval_timestamp
    valid_memory = [
        m for m in memory_set 
        if m['commit_timestamp'] < eval_timestamp
    ]
    
    # Then compute similarity on valid_memory only
    similarities = compute_similarity(eval_issue, valid_memory)
    return top_k(similarities)
```

**Pros:**
- OK Zero temporal leakage
- OK Realistic deployment scenario
- OK Scientifically sound

**Cons:**
- WARNING Early eval issues may have very few memory examples
- WARNING Latest memory issues may never be used

#### Strategy B: Hybrid Chronological + Similarity

```python
# Step 1: Sort by timestamp
issues_sorted = sorted(issues, key=lambda x: x['commit_timestamp'])

# Step 2: Split chronologically FIRST
cutoff_idx = int(len(issues_sorted) * 0.3)
memory_candidates = issues_sorted[:cutoff_idx]  # Earliest 30%
eval_set = issues_sorted[cutoff_idx:]           # Latest 70%

# Step 3: Within memory_candidates, select most representative
# (using similarity, but all are already from the past)
memory_set = select_representative(memory_candidates, method='similarity')

# Step 4: During evaluation (same as Strategy A)
def retrieve_memory(eval_issue):
    eval_timestamp = eval_issue['commit_timestamp']
    valid_memory = [m for m in memory_set if m['commit_timestamp'] < eval_timestamp]
    similarities = compute_similarity(eval_issue, valid_memory)
    return top_k(similarities)
```

**Pros:**
- OK Zero temporal leakage
- OK Memory is still "representative" within the chronological constraint
- OK May have better coverage

**Cons:**
- WARNING Slightly more complex implementation

### 3. Implement Incremental Memory (Advanced)

For the most realistic evaluation:

```python
# Start with earliest issues as initial memory
memory = []
results = []

for eval_issue in sorted_by_timestamp(eval_set):
    # 1. Retrieve from current memory
    retrieved = retrieve_memory(eval_issue, memory)
    
    # 2. Attempt to solve
    solution = agent_solve(eval_issue, retrieved)
    
    # 3. Evaluate solution
    result = evaluate(solution, eval_issue.ground_truth)
    results.append(result)
    
    # 4. Add this issue to memory for NEXT issues
    # (simulates learning from history)
    memory.append(eval_issue)

# Now memory grows over time - most realistic!
```

**Pros:**
- OK Most realistic simulation
- OK Shows how performance improves with more historical data
- OK Can analyze "cold start" (few examples) vs "mature" (many examples)

**Cons:**
- WARNING Slower evaluation
- WARNING More complex analysis

---

## Implementation Checklist

### Phase 1: Analysis (Current State)

- [ ] Run `scripts/analyze_temporal_leakage.py` to fetch timestamps
- [ ] Review temporal analysis results
- [ ] Quantify leakage in current split
- [ ] Document findings

### Phase 2: Data Preparation

- [ ] Augment dataset with timestamps
  ```python
  # Add to each issue in decomposed_issues.json
  {
    "issue_id": "71",
    "commit_timestamp": "2021-03-15T14:32:00Z",  # ← ADD THIS
    "sha_fail": "bd46af65...",
    ...
  }
  ```

- [ ] Create chronologically sorted dataset
- [ ] Implement new splitting logic

### Phase 3: New Splitting Script

Create `scripts/prepare_memory_train_test_split_v2.py`:

```python
def chronological_split(
    issues: List[Dict],
    memory_ratio: float = 0.3,
    strategy: str = 'pure',  # 'pure' or 'hybrid'
):
    """
    Split issues chronologically to prevent temporal leakage.
    """
    # 1. Sort by timestamp
    issues_sorted = sorted(
        issues, 
        key=lambda x: x['commit_timestamp']
    )
    
    # 2. Split chronologically
    cutoff = int(len(issues_sorted) * memory_ratio)
    
    if strategy == 'pure':
        memory = issues_sorted[:cutoff]
        eval_set = issues_sorted[cutoff:]
    elif strategy == 'hybrid':
        memory_candidates = issues_sorted[:cutoff]
        eval_set = issues_sorted[cutoff:]
        # Select representative from candidates
        memory = select_by_similarity(memory_candidates)
    
    return memory, eval_set
```

### Phase 4: Update Retrieval Logic

In `memory_plugin/ci_memory_system.py` or equivalent:

```python
def retrieve_historical_issues(
    current_issue: Dict,
    memory_bank: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Retrieve historically valid issues from memory.
    
    CRITICAL: Only retrieves issues with timestamp < current_issue timestamp
    """
    current_timestamp = current_issue.get('commit_timestamp')
    
    if not current_timestamp:
        logger.warning("No timestamp for current issue - cannot validate temporal constraint")
        valid_memory = memory_bank
    else:
        # Filter by timestamp
        valid_memory = [
            m for m in memory_bank
            if m.get('commit_timestamp', '9999') < current_timestamp
        ]
        
        logger.info(
            f"Filtered memory: {len(memory_bank)} -> {len(valid_memory)} "
            f"(only before {current_timestamp})"
        )
    
    # Compute similarity on valid memory
    similarities = compute_similarity(current_issue, valid_memory)
    
    # Return top-k
    return top_k_issues(similarities, k=top_k)
```

### Phase 5: Validation

- [ ] Re-run evaluation with chronological split
- [ ] Compare results: similarity-based vs chronological
- [ ] Document performance difference
- [ ] Analyze:
  - How much does performance drop? (expected if there was leakage)
  - How does performance change over time? (cold start -> mature)

---

## Quick Start

### Immediate Action (Analysis Only)

```bash
# 1. Install dependencies
pip install requests datasets rich sentence-transformers

# 2. Run temporal leakage analysis
python scripts/analyze_temporal_leakage.py

# 3. Review results
cat data/trs/temporal_analysis.json | jq '.leakage_analysis'
```

**Expected findings:**
- Likely significant leakage (>30% of eval issues)
- Memory and eval date ranges will overlap
- Specific examples of "future memory used for past eval"

### Next Steps After Analysis

Based on the leakage severity:

**If leakage is high (>30%):**
-> Immediate re-split using chronological approach
-> Re-run evaluations
-> Compare results

**If leakage is moderate (10-30%):**
-> Consider hybrid approach
-> Document limitations in paper/thesis

**If leakage is low (<10%):**
-> May keep current split but add temporal filtering in retrieval
-> Document as limitation

---

## Expected Impact

### On Performance Metrics

If temporal leakage exists, we expect:

**Current (with leakage):**
- Higher success rate (e.g., 75%)
- Better memory retrieval accuracy
- Artificially good results

**After fixing (chronological):**
- Lower success rate initially (e.g., 60%)
- Performance improves as memory grows
- Realistic, defensible results

### On Scientific Validity

**Before fix:**
- FAIL Results are not reproducible in production
- FAIL Comparison with baselines is unfair
- FAIL Claims about memory effectiveness are inflated

**After fix:**
- OK Results match real-world deployment
- OK Fair comparison with other approaches
- OK True understanding of memory's value

---

## References

### Related Files

- Current split logic: [`scripts/prepare_memory_train_test_split.py`](../scripts/prepare_memory_train_test_split.py)
- Analysis tool: [`scripts/analyze_temporal_leakage.py`](../scripts/analyze_temporal_leakage.py)
- Memory retrieval: [`memory_plugin/ci_memory_system.py`](../memory_plugin/ci_memory_system.py)
- Dataset: `ci-benchmark-user/ci-repair-bench` on HuggingFace

### Further Reading

- **Temporal Data Leakage in ML**: [Avoiding Data Leakage in Timeseries](https://towardsdatascience.com/temporal-data-leakage-b84ce8ed23c)
- **GitHub API - Commits**: https://docs.github.com/en/rest/commits/commits

---

## Conclusion

The current implementation has a **high risk of temporal data leakage** due to similarity-based splitting without timestamp consideration. This can be fixed by:

1. OK Fetching commit timestamps (via GitHub API)
2. OK Re-splitting chronologically (earliest -> memory, latest -> eval)
3. OK Filtering memory by timestamp during retrieval
4. OK Re-evaluating with the corrected setup

**Recommendation:** Run the analysis tool immediately to quantify the leakage, then implement chronological splitting before finalizing any results or publications.
