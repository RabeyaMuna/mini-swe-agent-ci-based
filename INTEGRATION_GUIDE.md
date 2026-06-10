# Integration Guide: Multi-Problem Problem Statement

## Quick Start: Enable Multi-Problem Mode

### Option 1: Minimal Change (Recommended)

Modify `src/minisweagent/run/benchmarks/cibench.py`:

```python
# Before (line ~XXX):
from minisweagent.run.benchmarks.utils.ci_context import build_ci_context

# After:
from minisweagent.run.benchmarks.utils.ci_context import build_ci_context
from minisweagent.run.benchmarks.utils.ci_context_multi_problem import (
    build_problem_statement_multi_problem
)

# Then find where build_ci_context() is called and modify:

# Before:
ci_context_result = build_ci_context(
    instance,
    memory_root=memory_root,
    memory_enabled=memory_enabled,
    model=model,
    llm=llm,
)
problem_statement = ci_context_result["problem_statement"]

# After:
ci_context_result = build_ci_context(
    instance,
    memory_root=memory_root,
    memory_enabled=memory_enabled,
    model=model,
    llm=llm,
)

# Use multi-problem version instead
problem_statement = build_problem_statement_multi_problem(
    ci_context_result["context"],
    ci_context_result["memory"],
)
```

### Option 2: Environment Variable Toggle

Add conditional logic to switch modes:

```python
import os

# Get multi-problem mode from environment
use_multi_problem = os.getenv("MEMCI_MULTI_PROBLEM", "true").lower() == "true"

ci_context_result = build_ci_context(...)

if use_multi_problem:
    from minisweagent.run.benchmarks.utils.ci_context_multi_problem import (
        build_problem_statement_multi_problem
    )
    problem_statement = build_problem_statement_multi_problem(
        ci_context_result["context"],
        ci_context_result["memory"],
    )
else:
    # Use default single-problem version
    problem_statement = ci_context_result["problem_statement"]
```

Then run with:
```bash
export MEMCI_MULTI_PROBLEM=true
python scripts/run_cibench.py --issue-id 410
```

---

## Test Before/After

### Test Script

Create `test_multi_problem.py`:

```python
#!/usr/bin/env python3
"""
Test multi-problem problem statement generation.

Usage:
    python test_multi_problem.py --issue-id 410
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minisweagent.run.benchmarks.utils.ci_context import build_ci_context
from minisweagent.run.benchmarks.utils.ci_context_multi_problem import (
    build_problem_statement_multi_problem
)


def load_issue(issue_id: str) -> dict:
    """Load issue from eval_issues.json."""
    eval_path = PROJECT_ROOT / "data" / "trs" / "eval_issues.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval issues not found: {eval_path}")
    
    with open(eval_path) as f:
        issues = json.load(f)
    
    for issue in issues:
        if str(issue.get("id")) == issue_id:
            return issue
    
    raise ValueError(f"Issue {issue_id} not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-id", required=True, help="Issue ID to test")
    args = parser.parse_args()
    
    # Load issue
    print(f"Loading issue {args.issue_id}...")
    issue = load_issue(args.issue_id)
    
    # Build CI context
    print("Building CI context with memory...")
    ci_result = build_ci_context(
        issue,
        memory_root=str(PROJECT_ROOT / "data" / "trs"),
        memory_enabled=True,
        memory_ablation_levels="L1+L2+L3",
        model="gpt-4o-mini",
    )
    
    # Generate BOTH versions
    print("\n" + "="*80)
    print("ORIGINAL PROBLEM STATEMENT (single-problem mode)")
    print("="*80)
    original_statement = ci_result["problem_statement"]
    print(original_statement)
    
    print("\n" + "="*80)
    print("NEW PROBLEM STATEMENT (multi-problem mode)")
    print("="*80)
    new_statement = build_problem_statement_multi_problem(
        ci_result["context"],
        ci_result["memory"],
    )
    print(new_statement)
    
    # Save both to files for comparison
    output_dir = PROJECT_ROOT / "output" / "problem_statements"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_file = output_dir / f"issue_{args.issue_id}_original.md"
    new_file = output_dir / f"issue_{args.issue_id}_multi_problem.md"
    
    original_file.write_text(original_statement)
    new_file.write_text(new_statement)
    
    print("\n" + "="*80)
    print("COMPARISON SAVED")
    print("="*80)
    print(f"Original: {original_file}")
    print(f"New:      {new_file}")
    print(f"\nRun: diff {original_file} {new_file}")
    
    # Stats
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    memory = ci_result["memory"]
    llm_selection = memory.get("llm_selection", {})
    guidance = llm_selection.get("guidance_document", {})
    
    linked_issues = guidance.get("linked_issues", [])
    additional_files = guidance.get("full_scope", {}).get("additional_files", [])
    
    print(f"Memory enabled: {memory.get('enabled', False)}")
    print(f"Linked issues (hidden problems): {len(linked_issues)}")
    print(f"Additional files (hidden): {len(additional_files)}")
    print(f"Primary files (visible): {len(guidance.get('primary_files', []))}")
    
    total_problems = 1 + len(linked_issues)
    print(f"\nTotal problems identified: {total_problems}")
    
    if total_problems > 1:
        print("\n✅ Multi-problem structure should be used")
        print(f"   - Problem #1: Visible in CI log")
        for i in range(len(linked_issues)):
            print(f"   - Problem #{i+2}: Hidden (from memory)")
    else:
        print("\n⚠️  Only 1 problem identified (single-problem CI failure)")


if __name__ == "__main__":
    main()
```

### Run Test

```bash
# Test on issue 410 (known multi-problem case)
python test_multi_problem.py --issue-id 410

# Compare the outputs
diff output/problem_statements/issue_410_original.md \
     output/problem_statements/issue_410_multi_problem.md
```

---

## Expected Differences

### Original Problem Statement:
```markdown
# CI Failure Report

## Why the CI Failed
  - Package installation failed

## Affected Files
  - pyproject.toml — Dependency constraint

## Memory Context — Repair Guidance from Past Experience
### Files to fix — including those NOT in the log
  - fish_speech/audio/processor.py — Type error (HIDDEN)
  - ... (10 more)
```

**Agent sees**: One primary problem + optional memory hints

---

### Multi-Problem Statement:
```markdown
# CI REPAIR TASK - MULTI-PROBLEM FAILURE

⚠️ **CRITICAL**: This has **3 DISTINCT PROBLEMS** that MUST ALL be fixed.

## PROBLEM #1: VISIBLE IN CI LOG (FIX THIS FIRST)
**Failed Command**: `uv pip install -e .`
**Files to Fix**:
  - pyproject.toml

## PROBLEM #2: HIDDEN PROBLEM (FIX AFTER PROBLEM #1)
**Why Hidden**: CI never reached mypy stage
**Files to Fix** (ALL 11 files):
  - fish_speech/audio/processor.py
  - ... (10 more)

## PROBLEM #3: HIDDEN PROBLEM (FIX AFTER PROBLEM #2)
**Files to Fix**:
  - tests/test_audio.py

## MANDATORY REPAIR SEQUENCE
**Step 1**: Fix Problem #1
**Step 2**: Fix Problem #2
**Step 3**: Fix Problem #3

## STOPPING CRITERIA
❌ DO NOT STOP if you only fixed Problem #1
✅ ONLY STOP when ALL 3 problems are fixed
```

**Agent sees**: Three explicit, required problems

---

## Evaluation Metrics

Track these before/after:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Agent attempts Problem #2 | ~10% | >80% |
| Agent attempts Problem #3 | ~5% | >60% |
| Full CI passes | ~35% | >70% |
| Agent stops after Problem #1 | ~85% | <20% |

---

## Debugging: Check What Agent Sees

Add logging to see what agent receives:

```python
# In cibench.py, after problem_statement is built:
print("\n" + "="*80)
print("PROBLEM STATEMENT SENT TO AGENT")
print("="*80)
print(problem_statement)
print("="*80 + "\n")

# Count problems
if "PROBLEM #2" in problem_statement:
    print("✅ Multi-problem mode active")
else:
    print("⚠️  Single-problem mode (agent may not fix all issues)")
```

---

## Rollback Plan

If multi-problem mode causes issues:

```python
# Simple: Just comment out the new import and use original
# from minisweagent.run.benchmarks.utils.ci_context_multi_problem import (
#     build_problem_statement_multi_problem
# )

# Use original:
problem_statement = ci_context_result["problem_statement"]
```

---

## Next Steps After Integration

1. **Run 10-20 test cases** comparing single vs. multi-problem mode
2. **Measure agent behavior**:
   - Does agent attempt hidden problems?
   - Does agent follow the repair sequence?
   - Does agent respect stopping criteria?

3. **Tune the prompts** based on results:
   - If agent still stops early: Make warnings stronger
   - If agent gets confused: Simplify structure
   - If agent ignores hidden problems: Make them more explicit

4. **Show professor**:
   - Side-by-side problem statements
   - Agent transcripts showing improved behavior
   - Metrics showing improvement
