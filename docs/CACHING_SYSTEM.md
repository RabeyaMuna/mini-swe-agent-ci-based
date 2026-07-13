# Caching System - Avoid Redundant LLM Calls

The system automatically caches expensive LLM analysis to avoid reprocessing the same issues.

---

## **Cache Files**

### **1. Log Analysis Cache**
- **File:** `data/trs/log_details.json`
- **Purpose:** Caches CI log analysis (Phase A)
- **Saves:** ~3 LLM calls per issue
- **Contains:**
  - `error_types` - What failed
  - `relevant_files` - Files involved
  - `failed_jobs` - Failed CI jobs
  - `error_context` - Summarized error messages

### **2. Workflow Validation Cache**
- **File:** `data/trs/workflow_validation_cache.json`
- **Purpose:** Caches workflow validation sequences (Phase B)
- **Saves:** ~1 LLM call per issue
- **Contains:**
  - `validation_sequence` - Ordered validation steps
  - `installation_cmd` - Setup commands
  - `validation_cmd` - Check commands
  - `critical_steps` - Important workflow stages

---

## **How Caching Works**

### **Phase A: Log Analysis**

```python
# Step 1: Try to load from cache
cached = _load_log_analysis_cache(sha_fail=sha_fail, task_id=task_id)

if cached:
    # ✓ Cache HIT - use cached data (NO LLM calls!)
    logger.info("[Phase A] Loaded cached log analysis for %s", sha_fail[:12])
    return cached

# ✗ Cache MISS - run CILogAnalyzer (3 LLM calls)
analyzer = CILogAnalyzer(logs=logs, sha_fail=sha_fail, ...)
result = analyzer.run()

# Save to cache for next time
_save_log_analysis_cache(result)
return result
```

### **Phase B: Workflow Validation**

```python
# Step 1: Try to load from cache
cached = _load_workflow_validation_cache(sha_fail=sha_fail, issue_id=issue_id)

if cached:
    # ✓ Cache HIT - use cached data (NO LLM calls!)
    logger.info("[Phase B] Loaded cached workflow validation for %s", sha_fail[:12])
    return cached

# ✗ Cache MISS - run workflow analyzer (1 LLM call)
context = analyze_workflow_from_benchmark(workflow_content=workflow, ...)

# Save to cache for next time
_save_workflow_validation_cache(context)
return context
```

---

## **Cache Lookup Strategy**

Caches are looked up by **SHA (commit hash)** or **task ID**:

```python
def _load_log_analysis_cache(sha_fail: str, task_id: str):
    # Read cache file
    cache_data = json.load("data/trs/log_details.json")
    
    # Search for matching entry
    for entry in cache_data:
        # Match by SHA (preferred)
        if entry.get("sha_fail") == sha_fail:
            return entry
        
        # Or match by task ID
        if entry.get("id") == task_id:
            return entry
    
    # No match found
    return None
```

---

## **When Caches Are Used**

### **✓ Cache Hit (Logged as):**

```
[Phase A] Loaded cached log analysis for bd46af65 — error_types=1 files=2 jobs=1
[Phase B] Loaded cached workflow validation sequence for bd46af65
```

**Result:**
- ✅ No LLM calls
- ✅ Instant processing
- ✅ Cost: $0.00
- ✅ Time: <1 second

### **✗ Cache Miss (Logged as):**

```
[Phase A] Running CILogAnalyzer...
[Phase B] Extracting workflow validation sequence with ci_workflow_aware_retrieval
```

**Result:**
- ❌ 3-4 LLM calls needed
- ❌ Processing takes time
- ❌ Cost: ~$0.01-0.05 per issue
- ❌ Time: 10-30 seconds

---

## **Cost Savings**

### **Without Caching:**
- 89 issues × 4 LLM calls = **356 LLM calls**
- Cost: 356 × $0.02 = **~$7.12**
- Time: 356 × 5 sec = **~30 minutes**

### **With Caching (90% hit rate):**
- 89 issues × 10% miss × 4 calls = **36 LLM calls**
- Cost: 36 × $0.02 = **~$0.72**
- Time: 36 × 5 sec = **~3 minutes**

**Savings: $6.40 + 27 minutes!**

---

## **Viewing Cache Contents**

### **Check Log Analysis Cache:**

```bash
# Count cached entries
cat data/trs/log_details.json | jq '. | length'

# View entry for specific SHA
cat data/trs/log_details.json | jq '.[] | select(.sha_fail == "bd46af65")'

# List all cached SHAs
cat data/trs/log_details.json | jq '.[].sha_fail' | sort | uniq
```

### **Check Workflow Validation Cache:**

```bash
# Count cached entries
cat data/trs/workflow_validation_cache.json | jq '. | length'

# View entry for specific issue
cat data/trs/workflow_validation_cache.json | jq '.[] | select(.issue_id == "71")'

# List all cached workflows
cat data/trs/workflow_validation_cache.json | jq '.[].workflow_path' | sort | uniq
```

---

## **Cache Management**

### **Clear Caches (Force Regeneration):**

```bash
# Clear log analysis cache
rm data/trs/log_details.json

# Clear workflow validation cache
rm data/trs/workflow_validation_cache.json

# Clear both
rm data/trs/log_details.json data/trs/workflow_validation_cache.json
```

**When to clear:**
- After fixing bugs in log analyzer
- After updating workflow analysis logic
- When testing changes to Phase A/B

### **Rebuild Caches:**

```bash
# Run evaluation - will regenerate missing caches
python3 scripts/run_eval.py \
    --issue-ids-file data/trs/eval_issue_ids.json \
    --ablation L1+L2+L3 \
    --workers 4

# Caches will be rebuilt on first run
```

---

## **Cache File Structure**

### **log_details.json:**

```json
[
  {
    "id": "71",
    "sha_fail": "bd46af653e25",
    "repo": "agno",
    "workflow_path": ".github/workflows/test.yml",
    "error_types": [
      {
        "type": "ImportError",
        "pattern": "module_not_found",
        "tool": "pytest"
      }
    ],
    "relevant_files": [
      "src/models.py",
      "tests/test_models.py"
    ],
    "failed_jobs": [
      {
        "name": "test",
        "step": "Run tests",
        "cmd": "pytest"
      }
    ],
    "error_context": "Import error in test_models.py..."
  }
]
```

### **workflow_validation_cache.json:**

```json
[
  {
    "issue_id": "71",
    "sha_fail": "bd46af653e25",
    "repo": "agno",
    "workflow_path": ".github/workflows/test.yml",
    "validation_sequence": [
      {
        "order": 1,
        "validates": "Checkout repository",
        "installation_cmd": "",
        "validation_cmd": "",
        "source": ".github/workflows/test.yml"
      },
      {
        "order": 2,
        "validates": "Python setup",
        "installation_cmd": "pip install -r requirements.txt",
        "validation_cmd": "",
        "source": ".github/workflows/test.yml"
      },
      {
        "order": 3,
        "validates": "Code formatting",
        "installation_cmd": "",
        "validation_cmd": "ruff format .",
        "source": ".github/workflows/test.yml"
      }
    ]
  }
]
```

---

## **Cache Behavior in Different Scenarios**

### **Scenario 1: First Run (No Cache)**

```bash
# Run evaluation
python3 scripts/run_eval.py --issue-ids 71,72,73 --ablation L1+L2+L3

# Output:
[Phase A] Running CILogAnalyzer... (3 LLM calls)
[Phase B] Extracting workflow validation... (1 LLM call)
[Phase A] Saved log analysis to data/trs/log_details.json
[Phase B] Saved workflow validation to data/trs/workflow_validation_cache.json

# Time: ~30 seconds per issue
# Cost: ~$0.04 per issue
```

### **Scenario 2: Second Run (With Cache)**

```bash
# Run same evaluation again
python3 scripts/run_eval.py --issue-ids 71,72,73 --ablation L1+L2+L3

# Output:
[Phase A] Loaded cached log analysis for bd46af65
[Phase B] Loaded cached workflow validation sequence for bd46af65

# Time: <1 second per issue (instant!)
# Cost: $0.00 (no LLM calls!)
```

### **Scenario 3: Partial Cache (Some New Issues)**

```bash
# Run with 1 cached + 2 new issues
python3 scripts/run_eval.py --issue-ids 71,74,75 --ablation L1+L2+L3

# Output:
[Phase A] Loaded cached log analysis for bd46af65 (issue 71)
[Phase A] Running CILogAnalyzer... (issue 74)
[Phase A] Running CILogAnalyzer... (issue 75)

# Time: ~1 sec for 71, ~30 sec for 74+75
# Cost: $0.00 for 71, ~$0.08 for 74+75
```

---

## **Automatic Cache Population**

When you build memory with the workflow script, caches are automatically populated:

```bash
# This creates both memory AND caches
bash scripts/run_memory_split_workflow.sh

# Result:
# - data/trs/failure_memory.json (L1)
# - data/trs/repo_memory.json (L2)
# - data/trs/cross_memory.json (L3)
# - data/trs/log_details.json (cache!)
# - data/trs/workflow_validation_cache.json (cache!)
```

---

## **Cache Invalidation**

Caches are **NOT** automatically invalidated. You must manually clear them if:

1. **Code changes** - Updated log analyzer logic
2. **Prompt changes** - Modified LLM prompts
3. **Bug fixes** - Fixed parsing bugs
4. **Model changes** - Switched LLM models

**Best practice:**
```bash
# After code changes affecting Phase A/B
rm data/trs/log_details.json data/trs/workflow_validation_cache.json

# Re-run to regenerate
python3 scripts/run_eval.py --issue-ids-file data/trs/eval_issue_ids.json --ablation L1+L2+L3
```

---

## **Monitoring Cache Usage**

### **Check Cache Hit Rate:**

```bash
# Count total issues
TOTAL=$(cat data/trs/eval_issue_ids.json | jq '. | length')

# Count cached issues
CACHED=$(cat data/trs/log_details.json | jq '. | length')

# Calculate hit rate
echo "Cache hit rate: $CACHED / $TOTAL = $((CACHED * 100 / TOTAL))%"
```

### **View Cache Stats:**

```bash
# Log analysis cache stats
echo "Log Analysis Cache:"
cat data/trs/log_details.json | jq -r '
  "Total entries: \(. | length)",
  "Repos: \([.[].repo] | unique | length)",
  "Workflows: \([.[].workflow_path] | unique | length)"
'

# Workflow validation cache stats
echo "Workflow Validation Cache:"
cat data/trs/workflow_validation_cache.json | jq -r '
  "Total entries: \(. | length)",
  "Repos: \([.[].repo] | unique | length)",
  "Avg validation steps: \([.[].validation_sequence | length] | add / length)"
'
```

---

## **Summary**

✅ **Automatic:** Caching happens automatically, no configuration needed

✅ **Fast:** Cache hits are instant (<1 second)

✅ **Cost-effective:** Saves ~$0.04 per cached issue

✅ **Persistent:** Caches survive across runs

✅ **Transparent:** Logs show "Loaded cached" for cache hits

✅ **Safe:** Caches can be cleared anytime without breaking anything

**Bottom line:** The caching system saves time and money by avoiding redundant LLM calls for issues you've already processed!
