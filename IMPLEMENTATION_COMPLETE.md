# Implementation Complete: Sequential Multi-Problem CI Repair

## OK **All Changes Implemented**

### **Summary:**
Successfully implemented **validation-sequence-aware sequential repair planning** that organizes problems by CI validation stages and predicts consecutive failures.

---

## Changes **Changes Made:**

### **1. Added Helper Functions (10 new functions)**

All added after line 276 in `ci_memory_system.py`:

#### **Validation Sequence Helpers:**
- OK `_get_current_failure_stage()` - Get full stage info for current failure
- OK `_get_consecutive_stages()` - Get stages that run AFTER current failure
- OK `_get_validation_cmd()` - Get validation command for a stage name

#### **Memory Organization Helpers:**
- OK `_map_memory_to_validation_stage()` - Map memory's `failed_cmd` -> validation stage
- OK `_extract_files_from_memory()` - Extract file info in standardized format
- OK `_extract_identification_criteria()` - Extract/generate grep patterns
- OK `_organize_memories_by_validation_stage()` - Group memories by stage, ordered

#### **Problem Extraction & Prediction:**
- OK `_extract_current_failure_as_problem_1()` - Extract current CI failure as Problem #1
- OK `_analyze_interdependency()` - Analyze how problems relate
- OK `_predict_consecutive_problems()` - Predict Problem #2+ from consecutive stages

---

### **2. Updated Prompt 1 (Organization)**

**Before:**
```python
def _build_organization_prompt(memory_result, validation_sequence):
    # Sends raw memories to LLM
    # No pre-organization
    # LLM has to guess validation stages
```

**After:**
```python
def _build_organization_prompt(organized, validation_sequence):
    # Pre-organized memories by validation stage
    # LLM only refines (adds criteria, combines duplicates)
    # Returns None if no memories (skips Prompt 1)
```

**Key Changes:**
- Takes pre-organized memories as input
- Optional refinement (not required)
- Skips if no memories

---

### **3. Updated Prompt 2 (Reasoning)**

**Before:**
```python
def _build_reasoning_prompt(current_context, organized_problems):
    # Vague "current_context" dict
    # No specific Problem #1 details
    # No consecutive stages list
```

**After:**
```python
def _build_reasoning_prompt(problem_1, organized_problems, validation_sequence):
    # Fully-formed Problem #1 with all details
    # Explicit consecutive_stages list
    # Clear interdependency examples
```

**Key Changes:**
- Problem #1 is pre-extracted and fully formed
- Shows consecutive stages explicitly
- Better reasoning guidance

---

### **4. Completely Rewrote `_run_single_llm_synthesis()`**

**Before:**
```python
def _run_single_llm_synthesis(...):
    if llm is None:
        return _build_simple_fallback()  # JSON dump
    
    # Build vague current_context
    # Prompt 1: organize (raw memories)
    # If Prompt 1 fails -> fallback (gives up)
    # Prompt 2: reason (with vague context)
    # If Prompt 2 fails -> fallback (gives up)
```

**After:**
```python
def _run_single_llm_synthesis(...):
    if llm is None:
        return _build_simple_fallback()  # Now uses helpers!
    
    # Extract Problem #1 first
    problem_1 = _extract_current_failure_as_problem_1(...)
    
    if not matches:
        # No memories: still predict from validation sequence
        # Uses problem_1 + consecutive_stages
    
    # Step 1: Pre-organize deterministically
    organized = _organize_memories_by_validation_stage(...)
    # Optional: LLM refinement
    # If LLM fails: use pre-organized (no fallback!)
    
    # Step 2: Build repair plan
    reason_prompt = _build_reasoning_prompt(problem_1, organized, ...)
    # If LLM fails: use deterministic prediction
    consecutive_problems = _predict_consecutive_problems(...)
```

**Key Changes:**
- OK **Works with 0 memories** (still predicts from validation sequence)
- OK **Pre-organizes deterministically** (LLM refinement optional)
- OK **Always has fallback** (deterministic prediction)
- OK **Problem #1 always extracted** (not vague context)

---

### **5. Updated `_run_two_llm_gate()`**

**Before:**
```python
def _run_two_llm_gate(...):
    all_matches = memory_result.get("matches") or []
    if not all_matches:
        return result  # <- Early exit! No synthesis!

**After:**
```python
def _run_two_llm_gate(...):
    # ALWAYS synthesize (even if matches=0)
    
    if llm is None:
        guidance_document = _build_simple_fallback(...)  # Now structured!
    else:
        guidance_document = _run_single_llm_synthesis(...)  # Works with 0 matches
    
    result["use_memory"] = True  # Always usable
    # ... (no debug breakpoints)
```

**Key Changes:**
- OK **No early exit** - always synthesizes
- OK **Removed debug breakpoints**
- OK **Works with 0 memories**

---

### **6. Improved `_build_simple_fallback()`**

**Before:**
```python
def _build_simple_fallback(...):
    # Just dumps JSON
    fallback_statement = f"""# CI Repair Task
    
    ## Current Failure
    {json.dumps(q)}
    
    ## Memories
    {json.dumps(cleaned_matches)}
    """
    
    return {"agent_problem_statement": fallback_statement, "total_problems": 1}
```

**After:**
```python
def _build_simple_fallback(...):
    # Extract Problem #1
    problem_1 = _extract_current_failure_as_problem_1(...)
    problems = [problem_1]
    
    if matches:
        # Organize memories
        organized = _organize_memories_by_validation_stage(...)
        # Predict consecutive problems
        consecutive = _predict_consecutive_problems(problem_1, organized, ...)
        problems.extend(consecutive[:3])
    
    # Build structured markdown
    agent_statement = _format_problems_as_markdown(problems, len(problems))
    
    return {
        "total_problems": len(problems),
        "problems": problems,
        "agent_problem_statement": agent_statement
    }
```

**Key Changes:**
- OK **Structured output** (not JSON dump)
- OK **Uses helper functions** (same logic as LLM path)
- OK **Predicts consecutive problems** (even without LLM)

---

### **7. Removed Debug Breakpoints**

Removed from:
- `_run_two_llm_gate()` (2 breakpoints)
- `format_memory_context()` (1 breakpoint)

---

## Performance **Before vs After:**

### **Scenario 1: Retrieval Returns 0 Memories**

#### Before:
```
retrieve() -> 0 matches
  down
_run_two_llm_gate() -> if not matches: return {} <- STOPS
  down
Agent gets NOTHING
```

#### After:
```
retrieve() -> 0 matches
  down
_run_two_llm_gate() -> ALWAYS synthesize
  down
_run_single_llm_synthesis()
  down
Extract Problem #1 from current CI
  down
Predict Problem #2+ from validation sequence order
  down
Agent gets structured plan with N problems
```

---

### **Scenario 2: Retrieval Returns 30 Memories**

#### Before:
```
retrieve() -> 30 matches
  down
Prompt 1: Organize (LLM guesses validation stages - empty field!)
  down
If fails -> JSON dump fallback
  down
Prompt 2: Reason (vague current_context)
  down
If fails -> JSON dump fallback
  down
Agent might get plan OR JSON dump
```

#### After:
```
retrieve() -> 30 matches
  down
Pre-organize by validation stage (deterministic - maps failed_cmd -> stage)
  down
Extract Problem #1 (current failure)
  down
Prompt 1: Refine organization (optional)
  down
If fails -> use pre-organized (still works!)
  down
Prompt 2: Build plan (Problem #1 + consecutive stages + organized problems)
  down
If fails -> deterministic prediction
  down
Agent ALWAYS gets structured plan with N problems
```

---

### **Scenario 3: No LLM Available**

#### Before:
```
llm = None
  down
_build_simple_fallback()
  down
Returns JSON dump of raw data
  down
Agent has to parse JSON manually
```

#### After:
```
llm = None
  down
_build_simple_fallback() (improved!)
  down
Extract Problem #1
  down
Organize memories by validation stage
  down
Predict consecutive problems
  down
Build structured markdown
  down
Agent gets proper sequential plan
```

---

##  **What This Achieves:**

### **Your Requirements (All Met!):**

1. OK **Use validation_sequence to understand order**
   - `_get_current_failure_stage()`
   - `_get_consecutive_stages()`

2. OK **Organize memories by validation_cmd**
   - `_map_memory_to_validation_stage()` - maps `failed_cmd` -> stage
   - `_organize_memories_by_validation_stage()` - groups by stage

3. OK **Build repair plan with consecutive problems**
   - `_extract_current_failure_as_problem_1()` - current failure
   - `_predict_consecutive_problems()` - what fails AFTER
   - `_analyze_interdependency()` - how they relate

### **Additional Improvements:**

4. OK **Works with 0 memories** (still predicts from validation order)
5. OK **Deterministic fallbacks** (never gives up)
6. OK **Problem #1 always current failure** (enforced in code)
7. OK **No early exits** (always synthesizes)
8. OK **Clean code** (no debug breakpoints)

---

## Testing **Testing:**

To test the implementation:

```bash
cd /Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based

# Run on instance 102 (the test case from your debug session)
python -m minisweagent.run.benchmarks.cibench \
  --config config.yml \
  --split test \
  --instance_id 102

# Check results
cat results/*/preds.json | python -m json.tool
```

**Expected Output:**
- OK Memories organized by validation stage
- OK Problem #1 = F632 lint error (confirmed)
- OK Problem #2+ = predicted from consecutive validation stages
- OK Interdependency reasoning included
- OK Agent receives structured markdown with N problems

---

## Files **Files Modified:**

1. **`src/minisweagent/run/benchmarks/utils/ci_memory_system.py`**
   - Added 10 helper functions
   - Updated `_build_organization_prompt()`
   - Updated `_build_reasoning_prompt()`
   - Completely rewrote `_run_single_llm_synthesis()`
   - Updated `_run_two_llm_gate()`
   - Improved `_build_simple_fallback()`
   - Removed debug breakpoints

**Total Changes:**
- Lines added: ~300
- Functions added: 10
- Functions modified: 5
- Debug breakpoints removed: 3

---

##  **Next Steps:**

1. **Test the implementation** (run cibench on instance 102)
2. **Verify output** (check if problems are organized by validation stage)
3. **Monitor logs** (check if consecutive problems are predicted)
4. **Review agent behavior** (does it fix all problems sequentially?)

The implementation is **complete and ready to test**! 
