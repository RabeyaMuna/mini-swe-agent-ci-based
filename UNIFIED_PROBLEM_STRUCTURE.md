# Unified Problem Structure - Best Approach

## Current Approach (Fragmented)

### Problem 1: Only CI Failure, No Related Issues
```python
def extract_problem_list(ci_failure):
    # Only extract from current CI failure
    files = ci_failure.get("relevant_files")
    problems = [{"file": f} for f in files]
    return problems
```

**Missing:**
- ❌ Related issues from memory
- ❌ Common follow-up problems
- ❌ Repair strategy from experience

### Problem 2: One Problem at a Time
```python
for problem in problems:
    prompt = compose_issue_document(problem)
    run_codex(prompt)  # 82 separate calls!
```

**Issues:**
- ❌ Fragmented context
- ❌ Doesn't see related issues
- ❌ Inefficient (82 calls)

---

## Proposed Approach: Unified Problem Structure

### Core Idea
**Combine:**
1. Current CI failure
2. Related issues from memory (that often occur together)
3. Repair strategy from experience
4. All in ONE comprehensive structure

### Structure

```python
@dataclass
class Problem:
    """Single problem to fix."""
    type: str  # "ci_failure", "memory_related", "potential"
    description: str
    files: List[str]
    error_type: str
    confidence: float  # 1.0 for CI failure, <1.0 for memory-based
    evidence: str
    from_memory: Optional[str] = None

@dataclass
class RepairStrategy:
    """How to fix the problems."""
    approach: str  # "tool_autofix", "manual_edit"
    primary_command: str
    success_rate: float
    estimated_time: str
    files_affected: int
    fallback: Optional[str] = None

@dataclass
class ProblemSet:
    """Complete problem context for Codex."""
    primary_problem: Problem  # The CI failure
    related_problems: List[Problem]  # From memory
    repair_strategy: RepairStrategy  # From memory analysis
    verification_steps: List[str]
    
    def to_prompt(self) -> str:
        """Generate comprehensive prompt."""
        ...
```

### Example

```python
problem_set = ProblemSet(
    primary_problem=Problem(
        type="ci_failure",
        description="Black formatting failed - Missing blank line after imports",
        files=["bbox_utils.py", "callback.py", ...],  # 82 files
        error_type="Code Formatting",
        confidence=1.0,
        evidence="pre-commit black-jupyter hook exit code 1"
    ),
    
    related_problems=[
        Problem(
            type="memory_related",
            description="Import sorting often needed with black formatting",
            files=["bbox_utils.py", "callback.py"],
            error_type="Import Order",
            confidence=0.85,
            evidence="5/8 similar cases also had isort issues",
            from_memory="L2: wandb/wandb history"
        ),
        Problem(
            type="potential",
            description="Type annotations might trigger mypy warnings",
            files=["bbox_utils.py"],
            error_type="Type Checking",
            confidence=0.60,
            evidence="3/8 similar cases had follow-up type issues",
            from_memory="L1: Cross-repo pattern"
        )
    ],
    
    repair_strategy=RepairStrategy(
        approach="tool_autofix",
        primary_command="pre-commit run --all-files",
        success_rate=0.93,
        estimated_time="30 seconds",
        files_affected=82,
        fallback="Manual edit: Add blank line after imports"
    ),
    
    verification_steps=[
        "pre-commit run black-jupyter --all-files",
        "pre-commit run --all-files"
    ]
)
```

---

## Prompt Generation

### Old Way (Fragmented)
```markdown
# CI Repair Task

## Problem To Fix
Problem 1 of 82: bbox_utils.py
File: bbox_utils.py
Reason: Missing blank line

## Previous Experience / Memory
L1 score=0.92, L2 score=0.85  ← Vague!
```

### New Way (Unified)
```markdown
# CI Repair Task

## Primary Problem
**Type:** CI Failure (Confidence: 100%)
**Description:** Black formatting failed - Missing blank line after imports
**Affected files:** 82 files
  - wandb/integration/ultralytics/bbox_utils.py
  - wandb/integration/ultralytics/callback.py
  - ... (80 more)
**Error type:** Code Formatting
**Evidence:** pre-commit black-jupyter hook exit code 1

## Related Issues (From Previous Experience)

### Likely to occur (85% confidence)
**Issue:** Import sorting often needed with black formatting
**Affected files:** bbox_utils.py, callback.py (same files)
**Evidence:** 5/8 similar cases also had isort issues after black fix
**Source:** L2 memory (wandb/wandb history)

### Possible follow-up (60% confidence)
**Issue:** Type annotations might trigger mypy warnings
**Affected files:** bbox_utils.py
**Evidence:** 3/8 similar cases had type-checking issues
**Source:** L1 memory (cross-repo pattern)

## Recommended Repair Strategy

**Based on 8 similar cases (93% success rate):**

**Primary approach: Automated tool fix**
```bash
# This handles ALL issues at once:
pre-commit run --all-files
```

**What this fixes:**
- ✅ Black formatting (82 files) - PRIMARY ISSUE
- ✅ Import sorting (if needed) - RELATED ISSUE
- ✅ Other pre-commit hooks - POTENTIAL ISSUES

**Why this works:**
- Used in: 7/8 successful cases
- Estimated time: 30 seconds
- Handles multiple issues: Yes (all pre-commit hooks)
- Files affected: All 82 files

**Fallback (if tool fails):**
Manual approach (used in 1/8 cases):
- Add blank line after last import statement
- Sort imports if needed

## CI Verification
1. Run: `pre-commit run --all-files`
2. Verify all checks pass
```

**Benefits:**
- ✅ Shows ALL problems (current + related)
- ✅ Clear repair strategy from experience
- ✅ One comprehensive view
- ✅ Confidence levels shown
- ✅ Handles related issues proactively

---

## Implementation

### Extract Comprehensive Problem Set

```python
def extract_comprehensive_problem_set(
    issue: dict,
    ci_failure: dict,
    verification: dict,
    memory_retrieval: dict,
) -> ProblemSet:
    """
    Extract complete problem context including:
    1. Primary CI failure
    2. Related issues from memory
    3. Repair strategy from experience
    """
    
    # 1. Primary problem from CI failure
    primary = extract_primary_problem(ci_failure)
    
    # 2. Related problems from memory
    related = extract_related_problems(memory_retrieval, primary)
    
    # 3. Repair strategy from memory
    strategy = synthesize_repair_strategy(memory_retrieval, primary)
    
    # 4. Verification from workflow
    verification_steps = extract_verification_steps(verification)
    
    return ProblemSet(
        primary_problem=primary,
        related_problems=related,
        repair_strategy=strategy,
        verification_steps=verification_steps
    )

def extract_primary_problem(ci_failure: dict) -> Problem:
    """Extract main CI failure."""
    files = ci_failure.get("relevant_files", [])
    error_context = ci_failure.get("error_context", [])
    error_types = ci_failure.get("error_types", [])
    
    return Problem(
        type="ci_failure",
        description=error_context[0] if error_context else "CI failure",
        files=[f.get("file") for f in files if isinstance(f, dict)],
        error_type=error_types[0].get("category") if error_types else "Unknown",
        confidence=1.0,
        evidence="\n".join(str(e) for e in error_context[:3])
    )

def extract_related_problems(
    memory_retrieval: dict,
    primary: Problem
) -> List[Problem]:
    """Extract related issues that often occur with primary problem."""
    related = []
    
    # Analyze L1/L2/L3 matches
    all_matches = (
        memory_retrieval.get("l1_matches", []) +
        memory_retrieval.get("l2_matches", []) +
        memory_retrieval.get("l3_matches", [])
    )
    
    # Find patterns of issues that occur together
    co_occurring = analyze_co_occurring_issues(all_matches, primary)
    
    for issue_pattern in co_occurring:
        if issue_pattern["frequency"] > 0.5:  # Occurs in >50% of cases
            related.append(Problem(
                type="memory_related",
                description=issue_pattern["description"],
                files=issue_pattern["files"],
                error_type=issue_pattern["error_type"],
                confidence=issue_pattern["frequency"],
                evidence=f"{issue_pattern['count']}/{len(all_matches)} similar cases",
                from_memory=f"L{issue_pattern['level']}"
            ))
    
    return related

def synthesize_repair_strategy(
    memory_retrieval: dict,
    primary: Problem
) -> RepairStrategy:
    """Synthesize repair strategy from successful past fixes."""
    all_matches = get_all_matches(memory_retrieval)
    
    # Group by approach
    strategies = group_by_strategy(all_matches)
    
    # Find most successful
    best = max(strategies.values(), key=lambda s: s["success_rate"])
    
    return RepairStrategy(
        approach=best["approach"],
        primary_command=best["command"],
        success_rate=best["success_rate"],
        estimated_time=best["avg_time"],
        files_affected=len(primary.files),
        fallback=best.get("fallback")
    )
```

---

## New Prompt Flow

### Single Comprehensive Prompt

```python
def compose_comprehensive_prompt(
    issue: dict,
    problem_set: ProblemSet
) -> str:
    """Generate one comprehensive prompt with all context."""
    
    prompt = f"""
# CI Repair Task

## Repository Context
Repository: {issue['repo']}
Commit: {issue['sha_fail']}
Workflow: {issue['workflow_path']}

## Primary Problem (CI Failure)
{format_problem(problem_set.primary_problem)}

## Related Issues (From Experience)
{format_related_problems(problem_set.related_problems)}

## Recommended Repair Strategy
{format_repair_strategy(problem_set.repair_strategy)}

## Verification Steps
{format_verification(problem_set.verification_steps)}

## Instructions
1. Address the PRIMARY PROBLEM first
2. Be aware of RELATED ISSUES that may surface
3. Follow REPAIR STRATEGY from successful past cases
4. Run VERIFICATION STEPS to confirm fix
5. Report all issues addressed
"""
    
    return prompt
```

**Result: ONE comprehensive prompt instead of 82 fragmented ones!**

---

## Comparison

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Problem scope** | CI failure only | CI + related issues |
| **Repair guidance** | Vague memory scores | Clear strategy from experience |
| **Number of prompts** | 82 (one per file) | 1 (comprehensive) |
| **Related issues** | Not considered | Proactively included |
| **Confidence levels** | No | Yes (0.0-1.0) |
| **Success rate** | Not shown | From memory (e.g., 93%) |
| **Time estimate** | No | Yes (e.g., "30 seconds") |

---

## Mini-SWE-Agent Comparison

### Mini-SWE-Agent approach:
```json
{
  "problem_statement": "The test suite is failing...",
  "hints": ["Check X", "Look at Y"],
  "test_patch": "diff of test...",
  "repo": "owner/repo"
}
```

**Key insights:**
- ✅ Single problem statement
- ✅ Includes hints
- ✅ Test-driven

### Our enhanced approach:
```python
{
  "primary_problem": {...},
  "related_problems": [...],  # ← Better than hints!
  "repair_strategy": {...},   # ← From actual experience!
  "verification": [...]
}
```

**Advantages:**
- ✅ More structured
- ✅ Confidence levels
- ✅ Evidence-based
- ✅ Success rates from memory

---

## Benefits

### 1. Comprehensive View
```
Before: "Fix file1.py" (no context)
After:  "Fix 82 files + watch for import sorting (85% likely)"
```

### 2. Proactive Problem Solving
```
Before: Fix formatting → CI fails again (import sorting) → Fix again
After:  Fix formatting + import sorting in one go → CI passes
```

### 3. Evidence-Based Strategy
```
Before: "L1 score=0.92" (what does this mean?)
After:  "Run tool X (93% success, 30 sec, used in 7/8 cases)"
```

### 4. Efficiency
```
Before: 82 separate Codex calls
After:  1 comprehensive Codex call
```

---

## Implementation Priority

1. **Phase 1: Structure**
   - Define ProblemSet, Problem, RepairStrategy classes
   - Update extract functions

2. **Phase 2: Memory Integration**
   - Extract related problems from memory
   - Synthesize repair strategy
   - Calculate confidence levels

3. **Phase 3: Prompt Generation**
   - Update compose_issue_document
   - Generate comprehensive prompt
   - Test with real issues

4. **Phase 4: Evaluation**
   - Compare old vs new approach
   - Measure success rates
   - Tune confidence thresholds

---

## Recommendation

**Implement the unified structure:**

```python
# New flow
problem_set = extract_comprehensive_problem_set(
    issue, ci_failure, verification, memory_retrieval
)

prompt = compose_comprehensive_prompt(issue, problem_set)

result = run_codex(prompt)  # ONE call handles everything!
```

**Benefits:**
- ✅ ONE prompt instead of 82
- ✅ Includes related issues
- ✅ Clear repair strategy
- ✅ Confidence levels
- ✅ Much more efficient

---

## Date
2026-07-30
