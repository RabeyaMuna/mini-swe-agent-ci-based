# Solution: Restructure Problem Statement to Force Multi-Problem Repair

## The Core Issue

**Current Problem**: Memory provides hidden problems, but agent treats them as "optional guidance"

**Why**: Problem statement structure makes visible problem PRIMARY and hidden problems SECONDARY

**Solution**: Restructure problem statement to make ALL problems REQUIRED and EXPLICIT

---

## Current Problem Statement Structure (WEAK)

```markdown
# CI Failure Report

**Repository**: `fish-speech/fish-speech`
**Failing Commit**: `abc123`
**Workflow**: `.github/workflows/lint.yaml`

## Why the CI Failed
  - Package installation failed: fish-audio-sdk>=2024.12.5 not found

## Failed Jobs / Commands
  - step `Install dependencies` → `uv pip install -e .`

## Affected Files
  - `pyproject.toml` — Dependency constraint error

## Validation Hints
  ```bash
  uv pip install -e .
  mypy fish_speech/
  pytest tests/
  ```

## Memory Context — Repair Guidance from Past Experience

**Confidence:** 🟢 HIGH

### What is really happening
Based on past experience, this failure has multiple linked problems...

### Files to fix — including those NOT in the log
  - `pyproject.toml` — Dependency constraint
  - `fish_speech/audio/processor.py` — Type error (HIDDEN)
  - ... (10 more files with hidden type errors)

### Post-fix patterns
  - [high] After fixing dependency, mypy will reveal type errors
```

**Problem**: Agent reads this as:
1. PRIMARY task: Fix the visible error (dependency)
2. OPTIONAL: Memory suggests there might be more issues
3. Agent fixes #1, verifies install passes, STOPS

---

## New Problem Statement Structure (STRONG)

```markdown
# CI Repair Task - MULTI-PROBLEM FAILURE

**Repository**: `fish-speech/fish-speech`
**Failing Commit**: `abc123`
**Workflow**: `.github/workflows/lint.yaml`

⚠️ **CRITICAL**: This CI failure contains **3 DISTINCT PROBLEMS** that must ALL be fixed.
The CI log shows only Problem #1 because CI stops at first failure.
Problems #2 and #3 are HIDDEN but WILL FAIL after you fix Problem #1.

---

## PROBLEM #1 (VISIBLE IN CI LOG) - MUST FIX FIRST

**Problem Type**: Dependency constraint error  
**CI Stage**: Installation (step 1/3)  
**Failed Command**: `uv pip install -e .`

**What's Wrong**:
  - Package `fish-audio-sdk>=2024.12.5` does not exist
  - pyproject.toml requires version that's not available

**Root Cause**:
  - Version constraint too new - SDK only published up to 2024.11.0

**Files to Fix**:
  - `pyproject.toml` (line 25)
    - Change: `fish-audio-sdk>=2024.12.5` 
    - To: `fish-audio-sdk>=2024.11.0`
    - Why: Allows installation with available SDK version

**Verification** (REQUIRED):
  ```bash
  uv pip install -e .
  ```
  Expected: Installation succeeds ✓

**IMPORTANT**: After fixing this, the CI will proceed to Problem #2 (mypy stage).

---

## PROBLEM #2 (HIDDEN - WILL FAIL NEXT) - MUST FIX AFTER PROBLEM #1

**Problem Type**: Type annotation errors  
**CI Stage**: Type checking (step 2/3)  
**Failed Command**: `mypy fish_speech/`

**Why Hidden**: 
  - This problem only appears AFTER Problem #1 is fixed
  - CI never reached mypy stage in the failing run
  - But the ground-truth patch fixes this too

**What's Wrong**:
  - SDK API changed return types from `AudioData` to `tuple[np.ndarray, int]`
  - 11 files have outdated type annotations

**Root Cause**:
  - After upgrading SDK from 2024.12.5→2024.11.0, API signatures changed
  - Type hints don't match new SDK interface

**Files to Fix** (ALL 11 files):
  1. `fish_speech/audio/processor.py` (line 45)
     - Change: `def process() -> AudioData:`
     - To: `def process() -> tuple[np.ndarray, int]:`
     
  2. `fish_speech/audio/encoder.py` (line 78)
     - Change: `def encode() -> AudioData:`
     - To: `def encode() -> tuple[np.ndarray, int]:`
  
  3. `fish_speech/audio/decoder.py` (line 62)
     - Change: `def decode() -> AudioData:`
     - To: `def decode() -> tuple[np.ndarray, int]:`

  ... (list all 11 files with specific line changes)

**Verification** (REQUIRED):
  ```bash
  mypy fish_speech/
  ```
  Expected: No type errors ✓

**IMPORTANT**: After fixing this, the CI will proceed to Problem #3 (pytest stage).

---

## PROBLEM #3 (HIDDEN - WILL FAIL LAST) - MUST FIX AFTER PROBLEM #2

**Problem Type**: Test expectation mismatch  
**CI Stage**: Testing (step 3/3)  
**Failed Command**: `pytest tests/`

**Why Hidden**:
  - This problem only appears AFTER Problems #1 and #2 are fixed
  - CI never reached pytest stage in the failing run

**What's Wrong**:
  - Test mocks expect old `AudioData` type
  - After SDK upgrade, tests need updated expectations

**Files to Fix**:
  - `tests/test_audio.py` (line 125-130)
    - Update mock return value from `AudioData(...)` to `(array, rate)`

**Verification** (REQUIRED):
  ```bash
  pytest tests/
  ```
  Expected: All tests pass ✓

---

## FULL REPAIR SEQUENCE (YOU MUST COMPLETE ALL STEPS)

1. ✅ Fix Problem #1 (pyproject.toml)
   → Verify: `uv pip install -e .` passes
   
2. ✅ Fix Problem #2 (11 type annotation files)
   → Verify: `mypy fish_speech/` passes
   
3. ✅ Fix Problem #3 (test expectations)
   → Verify: `pytest tests/` passes

4. ✅ Final verification (run full CI validation sequence):
   ```bash
   uv pip install -e .
   mypy fish_speech/
   pytest tests/
   ```

**STOPPING CRITERIA**: 
- ❌ DO NOT stop after fixing only Problem #1
- ❌ DO NOT stop after only install verification passes
- ✅ ONLY stop when ALL 3 problems are fixed AND all 3 validations pass

---

## Why These Problems Are All Required

**Evidence from past repairs**:
- Historical Issue #410 had this EXACT failure pattern
- Developer fixed all 3 problems in one patch
- Fixing only Problem #1 leaves CI still broken (mypy fails)
- Fixing only Problems #1-2 leaves CI still broken (pytest fails)

**Confidence**: 🟢 HIGH (based on 3 similar historical repairs)

---

## Files Summary (All Files You Need to Modify)

**Problem #1 Files** (1 file):
  - pyproject.toml

**Problem #2 Files** (11 files):
  - fish_speech/audio/processor.py
  - fish_speech/audio/encoder.py
  - fish_speech/audio/decoder.py
  - fish_speech/audio/sampler.py
  - fish_speech/audio/normalizer.py
  - fish_speech/audio/augmenter.py
  - fish_speech/audio/feature_extractor.py
  - fish_speech/audio/mel_spectrogram.py
  - fish_speech/audio/vocoder.py
  - fish_speech/audio/utils.py
  - fish_speech/audio/transforms.py

**Problem #3 Files** (1 file):
  - tests/test_audio.py

**Total**: 13 files to modify
```

---

## Key Changes in New Structure

| Old Approach | New Approach | Impact |
|--------------|--------------|--------|
| "Why the CI Failed" (singular) | "PROBLEM #1, #2, #3" (explicit count) | Agent knows there are multiple problems |
| Hidden problems in "Memory Context" section | Hidden problems as explicit "PROBLEM #2, #3" sections | Equal importance to visible problem |
| "Files to fix — including those NOT in the log" | "Files to Fix (ALL 11 files)" with line numbers | Specific, actionable, required |
| "Post-fix patterns: [high] mypy will fail" | "PROBLEM #2 (HIDDEN - WILL FAIL NEXT)" | Certainty, not possibility |
| Validation commands in one list | Verification required AFTER EACH PROBLEM | Step-by-step verification loop |
| No explicit stopping criteria | "DO NOT stop after Problem #1" | Clear instruction |
| Memory context as guidance | Problems with evidence from past repairs | Authority/confidence |

---

## Implementation: Modify `build_problem_statement()` in ci_context.py

Here's how to restructure the function:

```python
def build_problem_statement_multi_problem(
    context: Dict[str, Any],
    memory: Dict[str, Any],
) -> str:
    """
    Build problem statement that FORCES agent to fix ALL atomic problems.
    
    Structure:
    1. Header with EXPLICIT problem count
    2. Each problem as separate required section
    3. Clear verification requirements
    4. Explicit stopping criteria
    """
    
    # Extract atomic problems from memory
    llm_selection = memory.get("llm_selection", {})
    guidance_doc = llm_selection.get("guidance_document", {})
    linked_issues = guidance_doc.get("linked_issues", [])
    
    # Count total problems
    visible_problems = 1  # The primary CI failure
    hidden_problems = len(linked_issues)
    total_problems = visible_problems + hidden_problems
    
    # Build header
    lines = [
        "# CI Repair Task - MULTI-PROBLEM FAILURE",
        "",
        f"**Repository**: `{context.get('repo', '')}`",
        f"**Failing Commit**: `{context.get('sha_fail', '')}`",
        f"**Workflow**: `{context.get('workflow_path', '')}`",
        "",
        f"⚠️ **CRITICAL**: This CI failure contains **{total_problems} DISTINCT PROBLEMS** that must ALL be fixed.",
        "The CI log shows only Problem #1 because CI stops at first failure.",
    ]
    
    if hidden_problems > 0:
        problem_nums = ", ".join([f"#{i+2}" for i in range(hidden_problems)])
        lines.append(f"Problems {problem_nums} are HIDDEN but WILL FAIL after you fix Problem #1.")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # PROBLEM #1 (VISIBLE)
    lines.extend(_build_problem_section(
        problem_num=1,
        visibility="VISIBLE IN CI LOG",
        priority="MUST FIX FIRST",
        context=context,
        is_primary=True,
    ))
    
    # PROBLEM #2, #3, ... (HIDDEN)
    for idx, linked_issue in enumerate(linked_issues, start=2):
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.extend(_build_problem_section(
            problem_num=idx,
            visibility="HIDDEN - WILL FAIL NEXT",
            priority=f"MUST FIX AFTER PROBLEM #{idx-1}",
            linked_issue=linked_issue,
            is_primary=False,
        ))
    
    # FULL REPAIR SEQUENCE
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## FULL REPAIR SEQUENCE (YOU MUST COMPLETE ALL STEPS)")
    lines.append("")
    
    for i in range(1, total_problems + 1):
        lines.append(f"{i}. ✅ Fix Problem #{i}")
        lines.append(f"   → Verify: [validation command for problem {i}]")
        lines.append("")
    
    lines.append(f"{total_problems + 1}. ✅ Final verification (run full CI validation sequence)")
    lines.append("")
    
    # STOPPING CRITERIA
    lines.append("**STOPPING CRITERIA**:")
    lines.append("- ❌ DO NOT stop after fixing only Problem #1")
    lines.append("- ❌ DO NOT stop after only install verification passes")
    lines.append(f"- ✅ ONLY stop when ALL {total_problems} problems are fixed AND all validations pass")
    
    return "\n".join(lines)


def _build_problem_section(
    problem_num: int,
    visibility: str,
    priority: str,
    context: Dict[str, Any] = None,
    linked_issue: Dict[str, Any] = None,
    is_primary: bool = False,
) -> List[str]:
    """Build one problem section with all required fields."""
    
    lines = [
        f"## PROBLEM #{problem_num} ({visibility}) - {priority}",
        "",
    ]
    
    if is_primary:
        # Extract from context (visible problem)
        problem_type = context.get("overall_error_types", ["Unknown"])[0]
        failed_jobs = context.get("failed_jobs", [])
        failed_cmd = failed_jobs[0].get("command", "") if failed_jobs else ""
        
        lines.extend([
            f"**Problem Type**: {problem_type}",
            f"**CI Stage**: Installation (step 1/N)",
            f"**Failed Command**: `{failed_cmd}`",
            "",
            "**What's Wrong**:",
        ])
        
        for reason in context.get("overall_failure_reasons", []):
            lines.append(f"  - {reason}")
        
        lines.append("")
        lines.append("**Files to Fix**:")
        
        for file_info in context.get("effected_files", []):
            file_path = file_info.get("file", "")
            reason = file_info.get("reason", "")
            lines.append(f"  - `{file_path}` — {reason}")
        
    else:
        # Extract from linked_issue (hidden problem)
        root_cause = linked_issue.get("root_cause", "")
        fix_pattern = linked_issue.get("fix_pattern", "")
        affected_files = linked_issue.get("affected_files", [])
        
        lines.extend([
            f"**Problem Type**: {root_cause}",
            f"**CI Stage**: [from validation sequence]",
            "",
            "**Why Hidden**:",
            f"  - This problem only appears AFTER Problem #{problem_num-1} is fixed",
            "  - CI never reached this validation stage in the failing run",
            "  - But the ground-truth patch fixes this too",
            "",
            "**What's Wrong**:",
            f"  - {root_cause}",
            "",
            "**Files to Fix** (ALL {len(affected_files)} files):",
        ])
        
        for file_path in affected_files:
            lines.append(f"  - `{file_path}` — {fix_pattern}")
    
    lines.append("")
    lines.append("**Verification** (REQUIRED):")
    lines.append("  ```bash")
    lines.append(f"  [validation command]")
    lines.append("  ```")
    lines.append("  Expected: [specific success criteria] ✓")
    
    return lines
```

---

## Expected Agent Behavior with New Structure

### Before (Old Problem Statement):
```
Agent reads: "Fix dependency error"
Agent thinks: "Ok, one bug to fix"
Agent fixes: pyproject.toml
Agent verifies: install passes ✓
Agent stops: "Task complete"
Result: 33% fixed (1 of 3 problems)
```

### After (New Problem Statement):
```
Agent reads: "This has 3 DISTINCT PROBLEMS that must ALL be fixed"
Agent sees: Problem #1, Problem #2, Problem #3 sections
Agent thinks: "I need to fix all 3 problems"
Agent fixes: Problem #1 (pyproject.toml)
Agent verifies: install passes ✓
Agent reads: "DO NOT stop after Problem #1"
Agent fixes: Problem #2 (11 type annotation files)
Agent verifies: mypy passes ✓
Agent fixes: Problem #3 (test expectations)
Agent verifies: pytest passes ✓
Agent sees: "ONLY stop when ALL 3 problems fixed"
Agent stops: "All 3 problems fixed and verified"
Result: 100% fixed (3 of 3 problems)
```

---

## Next Steps

1. **Modify `ci_context.py`**:
   - Replace `build_problem_statement()` with the new multi-problem version
   - Extract atomic problems from memory guidance
   - Structure each problem as explicit required section

2. **Test on one example**:
   - Run on Issue #410 (fish-speech)
   - Verify agent attempts all 3 problems
   - Check if agent follows the explicit instructions

3. **Measure improvement**:
   - Before: % of cases where agent fixes only first problem
   - After: % of cases where agent attempts all problems
   - Final: % of cases where full CI passes

4. **Show professor**:
   - Side-by-side comparison of old vs. new problem statement
   - Agent transcript showing it now attempts all problems
   - Metrics showing improvement
