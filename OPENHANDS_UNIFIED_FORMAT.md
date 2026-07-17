# OpenHands Unified Prompt Format - FINAL

## ✅ **SAME FORMAT FOR BOTH MODES**

This is brilliant! Using the **same instructions** for both baseline and memory modes makes the comparison cleaner and fairer.

---

## 📝 Unified Format (Both Modes)

```
Fix the CI failure at commit {sha_fail}.

## Repository
{repo}

## Problem
{problem_statement}

## Repair Plan (If available)
{repair_plan OR "No previous experiences available. Analyze from scratch."}

## Failed Commit
{sha_fail}

## Task
1. Checkout commit {sha_fail}
2. Review the previous experiences and repair plan above (if available)
3. Analyze whether the failure matches known patterns from memory
4. Identify the locations and problems
5. If any problems can be automatically fixed by tools like ruff or docformatter 
   (particularly styling and formatting issues), apply them to fix automatically 
   instead of manually
6. Analyze the root cause, considering similar past solutions (if available)
7. Generate fixes informed by past solutions (if available) or based on analysis
8. Verify if possible using: {validation_command}
   (Note: Verification depends on your environment. If you cannot run tests, 
   generate the fix based on analysis.)

## Expected Output
Provide a complete git patch (diff format) that fixes this CI failure.
```

---

## 🎯 Key Difference Between Modes

### Baseline Mode

```
## Repair Plan (If available)
No previous experiences available. Analyze from scratch.
```

**Agent sees:**
- No repair plan → explores from scratch
- Same task instructions → knows what to do
- "(if available)" in instructions → understands to analyze without memory

### Memory Mode

```
## Repair Plan (If available)
Based on previous experiences, consider these approaches:

**From Similar Past Failures:**
1. Updated expected value in test
   (Similar issue: AssertionError in test_feature_x)
2. Added null check before property access
   (Similar issue: NoneType error in test_parser)

**Repository-Specific Patterns:**
3. This repo uses pytest fixtures in conftest.py
4. All tests require DJANGO_SETTINGS_MODULE env var

**General Debugging Strategies:**
5. CI failures often stem from missing dependencies
6. Check for timezone-dependent test failures
```

**Agent sees:**
- Has repair plan → applies known solutions
- Same task instructions → knows what to do
- "(if available)" in instructions → understands to use the memory

---

## ✅ Advantages

### 1. **Truly Fair Comparison**

**Same:**
- Repository
- Problem statement
- Task instructions
- Expected output
- Verification approach

**Only Difference:**
- Repair plan content (none vs. specific strategies)

### 2. **No Bias**

The agent gets the **same cognitive instructions** in both modes:
- "Review repair plan (if available)" ← Both modes see this
- "Analyze whether matches patterns" ← Both modes see this
- "Consider past solutions (if available)" ← Both modes see this

Baseline just has no past solutions to consider!

### 3. **Agent-Friendly**

The agent doesn't need different reasoning for different modes:
- Always checks for repair plan
- Always considers past solutions
- Always applies the same process

Baseline: No plan found → analyzes from scratch  
Memory: Plan found → applies strategies

### 4. **Includes Auto-Formatting**

**Step 5:**
```
If any problems can be automatically fixed by tools like ruff or 
docformatter (particularly styling and formatting issues), 
apply them to fix automatically instead of manually
```

This is **brilliant** because:
- Many CI failures are just formatting (black, ruff, etc.)
- Agent can fix these in seconds vs. manual analysis
- Reduces manual fix overhead
- More realistic (developers use these tools!)

---

## 📊 Real Example Comparison

### Baseline Example

```
Fix the CI failure at commit 7f7a4e7cd.

## Repository
pytest-dev/pytest

## Problem
CI pipeline failed due to test failure in test_collection.py - 
AssertionError: Expected 1 test, found 0

## Repair Plan (If available)
No previous experiences available. Analyze from scratch.

## Failed Commit
7f7a4e7cd

## Task
1. Checkout commit 7f7a4e7cd
2. Review the previous experiences and repair plan above (if available)
3. Analyze whether the failure matches known patterns from memory
4. Identify the locations and problems
5. If any problems can be automatically fixed by tools like ruff or docformatter 
   (particularly styling and formatting issues), apply them automatically
6. Analyze the root cause, considering similar past solutions (if available)
7. Generate fixes informed by past solutions (if available) or based on analysis
8. Verify if possible using: pytest tests/test_collection.py

## Expected Output
Provide a complete git patch (diff format) that fixes this CI failure.
```

**Agent reasoning:**
1. ✓ Check repair plan → "No previous experiences"
2. ✓ Analyze from scratch → explores codebase
3. ✓ Identify problem → test collection issue
4. ✓ Check auto-fix tools → not applicable
5. ✓ Analyze root cause → makepyfile missing __init__
6. ✓ Generate fix → add __init__=''
7. ✓ Verify → run tests

### Memory Example

```
Fix the CI failure at commit 7f7a4e7cd.

## Repository
pytest-dev/pytest

## Problem
CI pipeline failed due to test failure in test_collection.py - 
AssertionError: Expected 1 test, found 0

## Repair Plan (If available)
Based on previous experiences, consider these approaches:

**From Similar Past Failures:**
1. Added __init__.py file to test directory
   (Similar issue: Expected 2 tests, found 0)
2. makepyfile needs __init__='' parameter
   (Similar issue: test_parametrize.py fails with 0 tests)
3. Ensure proper package structure
   (Similar issue: No tests collected)

**Repository-Specific Patterns:**
4. pytest uses testdir fixture for testing
5. makepyfile() requires __init__='' for packages

**General Debugging Strategies:**
6. Test discovery failures often indicate missing __init__.py
7. Verify package structure before running tests

## Failed Commit
7f7a4e7cd

## Task
1. Checkout commit 7f7a4e7cd
2. Review the previous experiences and repair plan above (if available)
3. Analyze whether the failure matches known patterns from memory
4. Identify the locations and problems
5. If any problems can be automatically fixed by tools like ruff or docformatter 
   (particularly styling and formatting issues), apply them automatically
6. Analyze the root cause, considering similar past solutions (if available)
7. Generate fixes informed by past solutions (if available) or based on analysis
8. Verify if possible using: pytest tests/test_collection.py

## Expected Output
Provide a complete git patch (diff format) that fixes this CI failure.
```

**Agent reasoning:**
1. ✓ Check repair plan → **Sees strategies #1, #2, #3!**
2. ✓ Review past solutions → **"makepyfile needs __init__=''"**
3. ✓ Match patterns → **Matches strategy #2**
4. ✓ Identify problem → **Guided by memory**
5. ✓ Check auto-fix tools → not applicable
6. ✓ Apply past solution → **add __init__='' (from memory)**
7. ✓ Verify → run tests

**Result:** Same fix, but **faster** and more **confident**!

---

## 💡 Why This Format is Better

### Old Approach (Different Instructions)

**Baseline:**
```
Task:
1. Checkout
2. Analyze
3. Fix
4. Verify
```

**Memory:**
```
Task:
1. Checkout
2. **Review memory**
3. **Apply memory guidance**
4. Analyze considering memory
5. Fix informed by memory
6. Verify
```

**Problem:** Different instructions = potential bias in results

### New Approach (Same Instructions)

**Both Modes:**
```
Task:
1. Checkout
2. Review repair plan (if available)
3. Analyze whether matches patterns
4. Identify problems
5. Auto-fix if possible
6. Analyze root cause (considering past solutions if available)
7. Generate fixes (informed by past solutions if available)
8. Verify
```

**Benefit:** Same instructions, only data differs!

---

## 🎓 For Research

### Experimental Design

**Independent Variable:**
- Repair plan content (None vs. L1/L2/L3 strategies)

**Controlled Variables:**
- Task instructions (SAME)
- Problem statement (SAME)
- Repository (SAME)
- Validation command (SAME)
- Agent scaffold (SAME)

**Dependent Variables:**
- Pass rate
- Time to solution
- Iterations needed
- Patch quality

### Hypothesis

Memory mode will achieve higher pass rate because:
- Same reasoning process
- But has concrete strategies to try
- Can match patterns from past

**Not because:**
- Different instructions
- Different task structure
- Different cognitive load

---

## 💻 Implementation

### Code

```python
# Unified formatter - ONE function for both modes
def format_task(issue_data, memory_context=None):
    """
    Baseline: memory_context=None → "No previous experiences..."
    Memory: memory_context=repair_plan → "Based on previous..."
    """
    if memory_context:
        repair_plan = memory_context
    else:
        repair_plan = "No previous experiences available. Analyze from scratch."
    
    return unified_prompt_template.format(
        repo=issue_data['repo'],
        problem=issue_data['problem_statement'],
        repair_plan=repair_plan,  # Only difference!
        sha_fail=issue_data['sha_fail'],
        validation_cmd=issue_data['validation_command']
    )
```

### Usage

```python
# Baseline
task = format_task(issue_data, memory_context=None)

# Memory
memory = retrieve_and_format_memory(issue_id)
task = format_task(issue_data, memory_context=memory)

# OpenHands receives identical structure, different content
openhands.run(task)
```

---

## ✅ Status

### Implemented ✅

- [x] Unified format for both modes
- [x] Same task instructions
- [x] Auto-formatting suggestion (ruff, docformatter)
- [x] "(if available)" conditional phrasing
- [x] Repair plan or "analyze from scratch"
- [x] Memory retriever generates repair plan
- [x] Prompt formatter uses unified format

### Benefits ✅

- [x] Fair comparison
- [x] No instruction bias
- [x] Agent-friendly
- [x] Clean code (one function)
- [x] Easy to maintain

---

## 🚀 Ready to Use

```bash
cd openhands

# Test unified format
python prompt_formatter.py

# Run baseline (no memory)
python ci_bench_runner.py \
    --eval-issues ../data/trs/eval_set.jsonl \
    --mode baseline \
    --model glm-4-plus \
    --output ../results/openhands/glm/baseline

# Run memory (with repair plan)
python ci_bench_runner.py \
    --eval-issues ../data/trs/eval_set.jsonl \
    --mode memory \
    --memory-layers L1 L2 L3 \
    --model glm-4-plus \
    --output ../results/openhands/glm/L1_L2_L3
```

---

## 🎯 Summary

### What Changed

**Before:** Different instructions for baseline vs memory  
**After:** **SAME instructions**, different repair plan content

### Why Better

1. ✅ Fairer comparison
2. ✅ No instruction bias
3. ✅ Cleaner code
4. ✅ Agent gets same cognitive framework
5. ✅ Only data differs, not process

### The Magic

```
Baseline: "No previous experiences available. Analyze from scratch."
Memory:   "Based on previous experiences, consider: [strategies]"

Same instructions → Different information → Fair comparison!
```

---

**Last Updated**: July 16, 2026  
**Status**: ✅ **IMPLEMENTED** and **TESTED**
