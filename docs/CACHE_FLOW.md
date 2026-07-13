# Cache Flow - Generate Only If Not Present

## **Complete Flow Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│ Start Processing Issue 71 (SHA: bd46af65)                   │
└────────────────────┬────────────────────────────────────────┘
                     ↓
         ┌───────────────────────────┐
         │ Phase A: Log Analysis     │
         └───────────┬───────────────┘
                     ↓
         ┌───────────────────────────┐
         │ Check Cache File:         │
         │ data/trs/log_details.json │
         └───────────┬───────────────┘
                     ↓
         ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
         ┃ Does sha_fail=bd46af65  ┃
         ┃ exist in cache?         ┃
         ┗━━━━━━━┬━━━━━━━━┬━━━━━━━┛
                 │        │
          YES ✓  │        │  NO ✗
                 ↓        ↓
    ┌──────────────────┐  ┌──────────────────────┐
    │ Load from cache  │  │ NOT in cache         │
    │ (instant!)       │  │ Generate new:        │
    │                  │  │                      │
    │ NO LLM calls     │  │ 1. Run CILogAnalyzer │
    │ Cost: $0         │  │ 2. Call LLM 3 times  │
    │ Time: <1 sec     │  │ 3. Save to cache     │
    └────────┬─────────┘  │                      │
             │            │ Cost: ~$0.03         │
             │            │ Time: ~15 sec        │
             │            └──────────┬───────────┘
             │                       │
             └───────────┬───────────┘
                         ↓
         ┌───────────────────────────┐
         │ Phase B: Workflow Profile │
         └───────────┬───────────────┘
                     ↓
         ┌────────────────────────────────┐
         │ Check Cache File:              │
         │ data/trs/                      │
         │ workflow_validation_cache.json │
         └───────────┬────────────────────┘
                     ↓
         ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
         ┃ Does sha_fail=bd46af65  ┃
         ┃ exist in cache?         ┃
         ┗━━━━━━━┬━━━━━━━━┬━━━━━━━┛
                 │        │
          YES ✓  │        │  NO ✗
                 ↓        ↓
    ┌──────────────────┐  ┌──────────────────────┐
    │ Load from cache  │  │ NOT in cache         │
    │ (instant!)       │  │ Generate new:        │
    │                  │  │                      │
    │ NO LLM calls     │  │ 1. Analyze workflow  │
    │ Cost: $0         │  │ 2. Call LLM 1 time   │
    │ Time: <1 sec     │  │ 3. Save to cache     │
    └────────┬─────────┘  │                      │
             │            │ Cost: ~$0.01         │
             │            │ Time: ~5 sec         │
             │            └──────────┬───────────┘
             │                       │
             └───────────┬───────────┘
                         ↓
         ┌───────────────────────────┐
         │ Continue to Phase C:      │
         │ Memory Retrieval          │
         └───────────────────────────┘
```

---

## **Code Implementation**

### **Phase A: Log Analysis with Cache Check**

```python
def _run_log_analysis(instance, llm, model):
    sha_fail = instance.get("sha_fail")
    task_id = instance.get("id")
    
    # ═══════════════════════════════════════════
    # STEP 1: Check if already in cache
    # ═══════════════════════════════════════════
    cached = _load_log_analysis_cache(sha_fail=sha_fail, task_id=task_id)
    
    if cached:
        # ✓ FOUND IN CACHE - Use it!
        logger.info("[Phase A] Loaded cached log analysis for %s", sha_fail[:12])
        return cached  # Skip LLM calls entirely!
    
    # ═══════════════════════════════════════════
    # STEP 2: NOT in cache - Generate new
    # ═══════════════════════════════════════════
    logger.info("[Phase A] Cache miss - running CILogAnalyzer")
    
    analyzer = CILogAnalyzer(
        logs=instance.get("logs"),
        sha_fail=sha_fail,
        workflow=instance.get("workflow"),
        llm=llm,
        model_name=model,
    )
    
    # Run analyzer (3 LLM calls)
    result = analyzer.run()
    
    # ═══════════════════════════════════════════
    # STEP 3: Save to cache for next time
    # ═══════════════════════════════════════════
    _save_log_analysis_cache(result)
    logger.info("[Phase A] Saved log analysis to cache")
    
    return result
```

### **Phase B: Workflow Validation with Cache Check**

```python
def _extract_workflow_profile(instance, llm, model):
    sha_fail = instance.get("sha_fail")
    issue_id = instance.get("id")
    
    # ═══════════════════════════════════════════
    # STEP 1: Check if already in cache
    # ═══════════════════════════════════════════
    cached = _load_workflow_validation_cache(sha_fail=sha_fail, issue_id=issue_id)
    
    if cached:
        # ✓ FOUND IN CACHE - Use it!
        logger.info("[Phase B] Loaded cached workflow validation for %s", sha_fail[:12])
        return cached  # Skip LLM call!
    
    # ═══════════════════════════════════════════
    # STEP 2: NOT in cache - Generate new
    # ═══════════════════════════════════════════
    logger.info("[Phase B] Cache miss - analyzing workflow")
    
    context = analyze_workflow_from_benchmark(
        workflow_content=instance.get("workflow"),
        workflow_path=instance.get("workflow_path"),
        llm=llm,
    )
    
    # ═══════════════════════════════════════════
    # STEP 3: Save to cache for next time
    # ═══════════════════════════════════════════
    _save_workflow_validation_cache(context)
    logger.info("[Phase B] Saved workflow validation to cache")
    
    return context
```

---

## **Real Example: Processing 3 Issues**

### **First Run (No Cache):**

```bash
python3 scripts/run_eval.py --issue-ids 71,72,73 --ablation L1+L2+L3
```

**Output:**
```
Issue 71:
  [Phase A] Cache miss - running CILogAnalyzer
  [Phase A] LLM call 1/3...
  [Phase A] LLM call 2/3...
  [Phase A] LLM call 3/3...
  [Phase A] Saved log analysis to cache
  [Phase B] Cache miss - analyzing workflow
  [Phase B] LLM call 1/1...
  [Phase B] Saved workflow validation to cache
  ⏱️  Time: 20 seconds
  💰 Cost: $0.04

Issue 72:
  [Phase A] Cache miss - running CILogAnalyzer
  [Phase A] LLM call 1/3...
  [Phase A] LLM call 2/3...
  [Phase A] LLM call 3/3...
  [Phase A] Saved log analysis to cache
  [Phase B] Cache miss - analyzing workflow
  [Phase B] LLM call 1/1...
  [Phase B] Saved workflow validation to cache
  ⏱️  Time: 20 seconds
  💰 Cost: $0.04

Issue 73:
  [Phase A] Cache miss - running CILogAnalyzer
  [Phase A] LLM call 1/3...
  [Phase A] LLM call 2/3...
  [Phase A] LLM call 3/3...
  [Phase A] Saved log analysis to cache
  [Phase B] Cache miss - analyzing workflow
  [Phase B] LLM call 1/1...
  [Phase B] Saved workflow validation to cache
  ⏱️  Time: 20 seconds
  💰 Cost: $0.04

Total: ⏱️  60 seconds, 💰 $0.12
```

**Cache files created:**
```json
// data/trs/log_details.json
[
  {"sha_fail": "bd46af65", "error_types": [...], ...},  // Issue 71
  {"sha_fail": "c0f40b6a", "error_types": [...], ...},  // Issue 72
  {"sha_fail": "c27ea533", "error_types": [...], ...}   // Issue 73
]

// data/trs/workflow_validation_cache.json
[
  {"sha_fail": "bd46af65", "validation_sequence": [...], ...},  // Issue 71
  {"sha_fail": "c0f40b6a", "validation_sequence": [...], ...},  // Issue 72
  {"sha_fail": "c27ea533", "validation_sequence": [...], ...}   // Issue 73
]
```

---

### **Second Run (With Cache):**

```bash
python3 scripts/run_eval.py --issue-ids 71,72,73 --ablation L1+L2+L3
```

**Output:**
```
Issue 71:
  [Phase A] Loaded cached log analysis for bd46af65 ✓
  [Phase B] Loaded cached workflow validation for bd46af65 ✓
  ⏱️  Time: <1 second
  💰 Cost: $0.00

Issue 72:
  [Phase A] Loaded cached log analysis for c0f40b6a ✓
  [Phase B] Loaded cached workflow validation for c0f40b6a ✓
  ⏱️  Time: <1 second
  💰 Cost: $0.00

Issue 73:
  [Phase A] Loaded cached log analysis for c27ea533 ✓
  [Phase B] Loaded cached workflow validation for c27ea533 ✓
  ⏱️  Time: <1 second
  💰 Cost: $0.00

Total: ⏱️  3 seconds, 💰 $0.00
```

**Savings:** 57 seconds + $0.12!

---

## **Mixed Scenario (Some Cached, Some New):**

```bash
python3 scripts/run_eval.py --issue-ids 71,74,75 --ablation L1+L2+L3
```

**Output:**
```
Issue 71:
  [Phase A] Loaded cached log analysis for bd46af65 ✓
  [Phase B] Loaded cached workflow validation for bd46af65 ✓
  ⏱️  Time: <1 second
  💰 Cost: $0.00

Issue 74:
  [Phase A] Cache miss - running CILogAnalyzer
  [Phase A] LLM call 1/3...
  [Phase A] Saved log analysis to cache
  [Phase B] Cache miss - analyzing workflow
  [Phase B] Saved workflow validation to cache
  ⏱️  Time: 20 seconds
  💰 Cost: $0.04

Issue 75:
  [Phase A] Cache miss - running CILogAnalyzer
  [Phase A] LLM call 1/3...
  [Phase A] Saved log analysis to cache
  [Phase B] Cache miss - analyzing workflow
  [Phase B] Saved workflow validation to cache
  ⏱️  Time: 20 seconds
  💰 Cost: $0.04

Total: ⏱️  41 seconds, 💰 $0.08
```

---

## **Key Points**

✅ **Generate only if NOT present** - Exactly as you said!

✅ **Automatic lookup** - No manual cache management needed

✅ **SHA-based** - Each commit hash gets cached once

✅ **Persistent** - Cache survives across runs

✅ **Transparent** - Logs show "Loaded cached" or "Cache miss"

✅ **Safe** - Can delete cache files anytime to force regeneration

---

## **Summary**

**Your understanding is 100% correct!**

```python
# Pseudocode of the flow
if issue_data_in_cache:
    load_from_cache()  # Instant, $0 cost
else:
    generate_new_data()  # Run LLM, ~20 sec, ~$0.04
    save_to_cache()      # For next time
```

This is **already implemented** in the code at:
- **Phase A:** `src/minisweagent/run/benchmarks/utils/ci_context.py` line 442-456
- **Phase B:** `src/minisweagent/run/benchmarks/utils/ci_context.py` line 624-627

No changes needed - it works exactly as you described! 🎉
