# Memory & Problem Format Design - Foundation First

## Your Insight: Work Backwards!

**Current approach (wrong):**
```
Fix prompts → Hope memory works → Try to format problems → ???
```

**Your approach (right):**
```
1. Design L1/L2/L3 memory usage ✓
2. Define problem + repair plan format ✓
3. Make agent-agnostic (Mini/Codex/etc.) ✓
4. THEN fix prompts to accept this format ✓
```

**This is the correct order!** Foundation → Interface → Implementation

---

## Step 1: L1/L2/L3 Memory Design

### What Each Level Should Provide

#### L1: Failure Memory (Cross-Repo)
**Purpose:** Learn from similar failures in ANY repo

**Data structure:**
```python
{
    "level": "L1",
    "match_type": "similar_failure",
    "similarity_score": 0.92,
    
    # What failed
    "failure": {
        "error_type": "Code Formatting",
        "error_subtype": "Black - Missing blank line",
        "files_affected": 50,
        "repo": "optuna/optuna"
    },
    
    # What worked to fix it
    "repair": {
        "approach": "tool_autofix",
        "command": "black .",
        "time_taken": "25 seconds",
        "success": true
    },
    
    # Evidence
    "evidence": {
        "issue_id": "28",
        "commit": "3137ef6...",
        "verification": "pre-commit run --all-files passed"
    }
}
```

**Key insight:** L1 shows **what approach worked** for similar failures

#### L2: Repo Memory (Same Repo)
**Purpose:** Repo-specific patterns and conventions

**Data structure:**
```python
{
    "level": "L2",
    "match_type": "same_repo_history",
    "similarity_score": 0.85,
    
    # Repo context
    "repo": {
        "name": "wandb/wandb",
        "conventions": {
            "formatter": "black-jupyter",
            "pre_commit": true,
            "setup_cmd": "./core/scripts/code-checks.sh update"
        }
    },
    
    # Previous fixes in this repo
    "repair_history": [
        {
            "pr": "#1234",
            "date": "2026-06-15",
            "issue": "Black formatting",
            "approach": "pre-commit run black-jupyter --all-files",
            "files_fixed": 5,
            "success": true
        }
    ],
    
    # Co-occurring issues in this repo
    "common_patterns": [
        {
            "primary": "Black formatting",
            "often_with": "Import sorting",
            "frequency": 0.70  # 70% of time
        }
    ]
}
```

**Key insight:** L2 shows **repo-specific conventions** and **co-occurring issues**

#### L3: Cross-Repo Patterns
**Purpose:** General best practices and tool patterns

**Data structure:**
```python
{
    "level": "L3",
    "match_type": "cross_repo_pattern",
    "confidence": 0.78,
    
    # General pattern
    "pattern": {
        "name": "Black/Prettier formatting failures",
        "category": "Automated tool formatting",
        "repos_seen": 15,
        "total_cases": 45
    },
    
    # Best practice approach
    "recommended_approach": {
        "strategy": "tool_autofix",
        "commands": [
            "black .",
            "prettier --write .",
            "pre-commit run --all-files"
        ],
        "success_rate": 0.95,
        "avg_time": "30 seconds"
    },
    
    # When it fails
    "failure_modes": [
        {
            "reason": "Tool not configured",
            "frequency": 0.03,
            "solution": "Run setup command first"
        },
        {
            "reason": "Manual changes needed",
            "frequency": 0.02,
            "solution": "Logical issues, edit manually"
        }
    ]
}
```

**Key insight:** L3 shows **general best practices** and **when to use tools**

---

## Step 2: Universal Problem Format

### Problem + Repair Plan Structure

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

class ProblemSource(Enum):
    CI_FAILURE = "ci_failure"          # From current CI logs
    MEMORY_L1 = "memory_l1"            # From L1: similar failures
    MEMORY_L2 = "memory_l2"            # From L2: repo patterns
    MEMORY_L3 = "memory_l3"            # From L3: best practices

class RepairApproach(Enum):
    TOOL_AUTOFIX = "tool_autofix"      # Run automated tool
    MANUAL_EDIT = "manual_edit"        # Requires code changes
    CONFIGURATION = "configuration"     # Config/YAML changes
    DEPENDENCY = "dependency"           # Install/update deps

@dataclass
class Problem:
    """Single problem to address."""
    
    # Identity
    id: str
    source: ProblemSource
    confidence: float  # 0.0-1.0
    
    # Description
    title: str
    description: str
    error_type: str
    error_subtype: Optional[str]
    
    # Scope
    files: List[str]
    lines: Optional[List[int]]
    
    # Evidence
    evidence: str
    from_memory: Optional[str]  # Which L1/L2/L3 match

@dataclass
class RepairInstruction:
    """How to fix a problem."""
    
    # Approach
    approach: RepairApproach
    primary_command: str
    alternate_commands: List[str]
    
    # Success metrics
    success_rate: float
    estimated_time: str
    cases_used: str  # e.g., "7/8 similar cases"
    
    # Context
    handles_problems: List[str]  # Problem IDs this fixes
    requires_setup: Optional[str]
    
    # Fallback
    fallback_approach: Optional[str]
    fallback_reason: Optional[str]

@dataclass
class RepairPlan:
    """Complete repair plan for an issue."""
    
    # Context
    issue_id: str
    repo: str
    commit: str
    workflow: str
    
    # Problems
    primary_problems: List[Problem]      # From CI failure
    related_problems: List[Problem]      # From memory
    
    # Instructions
    repair_instructions: List[RepairInstruction]
    
    # Verification
    verification_steps: List[str]
    
    # Metadata
    memory_used: Dict[str, Any]  # L1/L2/L3 details
    confidence: float  # Overall confidence

    def to_agent_format(self, agent_type: str) -> Dict[str, Any]:
        """Convert to agent-specific format."""
        if agent_type == "codex":
            return self._to_codex_format()
        elif agent_type == "mini-swe":
            return self._to_mini_swe_format()
        else:
            return self._to_generic_format()
```

---

## Step 3: Agent-Agnostic Interface

### Universal Interface

```python
class RepairAgent(ABC):
    """Base class for any repair agent."""
    
    @abstractmethod
    def prepare_prompt(self, plan: RepairPlan) -> str:
        """Convert RepairPlan to agent-specific prompt."""
        pass
    
    @abstractmethod
    def execute(self, prompt: str) -> AgentResult:
        """Execute the repair."""
        pass
    
    @abstractmethod
    def verify(self, result: AgentResult) -> VerificationResult:
        """Verify the fix worked."""
        pass

class CodexAgent(RepairAgent):
    """Codex-specific implementation."""
    
    def prepare_prompt(self, plan: RepairPlan) -> str:
        return f"""
# CI Repair Task

## Primary Issues
{self._format_problems(plan.primary_problems)}

## Related Issues (From Memory)
{self._format_problems(plan.related_problems)}

## Recommended Repair Instructions
{self._format_instructions(plan.repair_instructions)}

## Verification
{self._format_verification(plan.verification_steps)}
"""

class MiniSWEAgent(RepairAgent):
    """Mini-SWE-Agent specific implementation."""
    
    def prepare_prompt(self, plan: RepairPlan) -> str:
        return {
            "problem_statement": self._build_statement(plan),
            "hints": self._extract_hints(plan),
            "repo": plan.repo,
            "base_commit": plan.commit
        }

class CustomAgent(RepairAgent):
    """Your custom agent implementation."""
    pass
```

**Key benefit:** Change agent without changing memory or problem extraction!

---

## Step 4: Memory → Problem → Repair Plan Pipeline

### Complete Flow

```python
def build_repair_plan(
    issue: Dict[str, Any],
    ci_failure: Dict[str, Any],
    ablation: str
) -> RepairPlan:
    """
    Build complete repair plan from memory analysis.
    
    Flow:
    1. Extract primary problems from CI failure
    2. Query memory (L1/L2/L3 based on ablation)
    3. Extract related problems from memory
    4. Synthesize repair instructions from memory
    5. Build comprehensive plan
    """
    
    # 1. Primary problems from CI
    primary_problems = extract_primary_problems(ci_failure)
    
    # 2. Query memory
    if ablation == "baseline":
        memory = None
    else:
        memory = query_memory(
            issue=issue,
            ci_failure=ci_failure,
            levels=parse_ablation(ablation)  # ["L1", "L2", "L3"]
        )
    
    # 3. Related problems from memory
    related_problems = []
    if memory:
        related_problems = extract_related_problems(memory, primary_problems)
    
    # 4. Repair instructions from memory
    instructions = []
    if memory:
        instructions = synthesize_repair_instructions(
            memory=memory,
            primary_problems=primary_problems,
            related_problems=related_problems
        )
    else:
        # Baseline: Use heuristics
        instructions = generate_heuristic_instructions(primary_problems)
    
    # 5. Build plan
    plan = RepairPlan(
        issue_id=issue["id"],
        repo=issue["repo"],
        commit=issue["sha_fail"],
        workflow=issue["workflow_path"],
        primary_problems=primary_problems,
        related_problems=related_problems,
        repair_instructions=instructions,
        verification_steps=extract_verification_steps(issue),
        memory_used=memory or {},
        confidence=calculate_confidence(primary_problems, memory)
    )
    
    return plan
```

---

## Step 5: Concrete Examples

### Example 1: Baseline (No Memory)

```python
plan = RepairPlan(
    issue_id="43",
    repo="wandb/wandb",
    
    primary_problems=[
        Problem(
            id="p1",
            source=ProblemSource.CI_FAILURE,
            confidence=1.0,
            title="Black formatting failed",
            description="Missing blank line after imports",
            error_type="Code Formatting",
            files=["bbox_utils.py", "callback.py"],
            evidence="pre-commit black-jupyter hook exit code 1"
        )
    ],
    
    related_problems=[],  # No memory!
    
    repair_instructions=[
        RepairInstruction(
            approach=RepairApproach.TOOL_AUTOFIX,
            primary_command="pre-commit run black-jupyter --all-files",
            success_rate=0.0,  # No memory data
            estimated_time="unknown",
            cases_used="heuristic",
            handles_problems=["p1"]
        )
    ]
)
```

**Baseline:** Only CI failure, heuristic instructions

### Example 2: L1 (Similar Failures)

```python
plan = RepairPlan(
    issue_id="43",
    repo="wandb/wandb",
    
    primary_problems=[
        Problem(
            id="p1",
            source=ProblemSource.CI_FAILURE,
            confidence=1.0,
            title="Black formatting failed",
            files=["bbox_utils.py", "callback.py"],
            ...
        )
    ],
    
    related_problems=[],  # L1 doesn't predict related issues
    
    repair_instructions=[
        RepairInstruction(
            approach=RepairApproach.TOOL_AUTOFIX,
            primary_command="black .",
            alternate_commands=["pre-commit run black --all-files"],
            success_rate=0.92,  # From L1 memory!
            estimated_time="30 seconds",
            cases_used="12/15 similar cases",
            handles_problems=["p1"],
            fallback_approach="Manual edit: Add blank line after imports",
            fallback_reason="If tool not configured or conflicts"
        )
    ],
    
    memory_used={
        "l1_matches": 15,
        "best_match_score": 0.92,
        "approach_distribution": {
            "tool_autofix": 12,
            "manual_edit": 3
        }
    }
)
```

**L1:** Better instructions from similar failures

### Example 3: L1+L2 (+ Repo Context)

```python
plan = RepairPlan(
    issue_id="43",
    repo="wandb/wandb",
    
    primary_problems=[
        Problem(
            id="p1",
            source=ProblemSource.CI_FAILURE,
            confidence=1.0,
            title="Black formatting failed",
            files=["bbox_utils.py", "callback.py"],
            ...
        )
    ],
    
    related_problems=[
        Problem(
            id="p2",
            source=ProblemSource.MEMORY_L2,
            confidence=0.70,  # 70% chance
            title="Import sorting usually needed too",
            description="In wandb/wandb, formatting often requires import sorting",
            error_type="Import Order",
            files=["bbox_utils.py", "callback.py"],
            evidence="5/8 previous formatting fixes also sorted imports",
            from_memory="L2: wandb/wandb history"
        )
    ],
    
    repair_instructions=[
        RepairInstruction(
            approach=RepairApproach.TOOL_AUTOFIX,
            primary_command="pre-commit run --all-files",  # Handles both!
            success_rate=0.93,
            estimated_time="30 seconds",
            cases_used="7/8 similar cases in wandb/wandb",
            handles_problems=["p1", "p2"],  # Fixes both!
            requires_setup="./core/scripts/code-checks.sh update"
        )
    ],
    
    memory_used={
        "l1_matches": 15,
        "l2_matches": 8,
        "repo_conventions": {
            "formatter": "black-jupyter",
            "setup_cmd": "./core/scripts/code-checks.sh update"
        }
    }
)
```

**L1+L2:** Repo-specific patterns + related issues predicted!

### Example 4: L1+L2+L3 (Full Memory)

```python
plan = RepairPlan(
    issue_id="43",
    repo="wandb/wandb",
    
    primary_problems=[
        Problem(
            id="p1",
            source=ProblemSource.CI_FAILURE,
            confidence=1.0,
            title="Black formatting failed",
            files=["bbox_utils.py", "callback.py"],
            ...
        )
    ],
    
    related_problems=[
        Problem(
            id="p2",
            source=ProblemSource.MEMORY_L2,
            confidence=0.70,
            title="Import sorting usually needed",
            from_memory="L2: wandb/wandb (5/8 cases)"
        ),
        Problem(
            id="p3",
            source=ProblemSource.MEMORY_L3,
            confidence=0.40,
            title="Type hints might need updating",
            description="Black formatting changes sometimes expose mypy issues",
            from_memory="L3: Cross-repo pattern (6/45 cases)"
        )
    ],
    
    repair_instructions=[
        RepairInstruction(
            approach=RepairApproach.TOOL_AUTOFIX,
            primary_command="pre-commit run --all-files",
            alternate_commands=[
                "black . && isort . && mypy .",
                "pre-commit run black-jupyter --all-files"
            ],
            success_rate=0.95,  # L3 best practice!
            estimated_time="30 seconds",
            cases_used="43/45 cross-repo cases",
            handles_problems=["p1", "p2", "p3"],  # All three!
            requires_setup="./core/scripts/code-checks.sh update",
            fallback_approach="Manual edit if tool cannot fix logical issues",
            fallback_reason="2/45 cases needed manual changes for type issues"
        )
    ],
    
    memory_used={
        "l1_matches": 15,
        "l2_matches": 8,
        "l3_patterns": 3,
        "confidence_breakdown": {
            "approach": 0.95,  # From L3
            "related_issues": 0.55  # Avg of L2+L3 predictions
        }
    }
)
```

**L1+L2+L3:** Best practices + repo patterns + related issues = comprehensive!

---

## Step 6: Agent Prompt Generation

### Codex Agent

```python
def to_codex_prompt(plan: RepairPlan) -> str:
    """Convert RepairPlan to Codex prompt."""
    
    prompt = f"""
# CI Repair Task

Repository: {plan.repo}
Commit: {plan.commit}

## Primary Problems ({len(plan.primary_problems)})

"""
    
    # Format primary problems
    for p in plan.primary_problems:
        prompt += f"""
### {p.title}
- **Type:** {p.error_type}
- **Files:** {len(p.files)} files
  {chr(10).join(f"  - {f}" for f in p.files[:10])}
  {"  ... and " + str(len(p.files)-10) + " more" if len(p.files) > 10 else ""}
- **Evidence:** {p.evidence}
"""
    
    # Format related problems if any
    if plan.related_problems:
        prompt += f"""

## Related Issues (From Memory)

Based on analysis of similar failures, these issues often occur together:

"""
        for p in plan.related_problems:
            prompt += f"""
### {p.title} (Confidence: {p.confidence*100:.0f}%)
- **Source:** {p.from_memory}
- **Description:** {p.description}
- **Files:** {", ".join(p.files[:5])}
"""
    
    # Format repair instructions
    if plan.repair_instructions:
        instr = plan.repair_instructions[0]  # Primary instruction
        prompt += f"""

## Recommended Repair Approach

**Based on {instr.cases_used} (success rate: {instr.success_rate*100:.0f}%)**

**Primary command:**
```bash
{instr.primary_command}
```

**What this fixes:**
{chr(10).join(f"- {plan._get_problem_title(pid)}" for pid in instr.handles_problems)}

**Estimated time:** {instr.estimated_time}

{f"**Setup required first:**{chr(10)}```bash{chr(10)}{instr.requires_setup}{chr(10)}```" if instr.requires_setup else ""}

{f"**If that fails:**{chr(10)}{instr.fallback_approach}{chr(10)}Reason: {instr.fallback_reason}" if instr.fallback_approach else ""}
"""
    
    # Verification
    prompt += f"""

## Verification Steps

{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(plan.verification_steps))}

## Instructions

1. **Prefer automated tools** - Use the recommended command above
2. **Address all issues** - Primary problem + related issues (if they surface)
3. **Run verification** - Ensure all checks pass
4. **Report results** - What was fixed, verification output
"""
    
    return prompt
```

### Mini-SWE-Agent

```python
def to_mini_swe_format(plan: RepairPlan) -> Dict[str, Any]:
    """Convert RepairPlan to Mini-SWE-Agent format."""
    
    # Build problem statement
    statement = f"{plan.primary_problems[0].title}\n\n"
    statement += plan.primary_problems[0].description
    
    # Build hints
    hints = []
    
    # From repair instructions
    if plan.repair_instructions:
        instr = plan.repair_instructions[0]
        hints.append(f"Try: {instr.primary_command}")
        if instr.success_rate > 0:
            hints.append(f"This approach worked in {instr.cases_used}")
    
    # From related problems
    for p in plan.related_problems:
        if p.confidence > 0.5:
            hints.append(f"Watch for: {p.title} ({p.confidence*100:.0f}% likely)")
    
    return {
        "instance_id": plan.issue_id,
        "repo": plan.repo,
        "base_commit": plan.commit,
        "problem_statement": statement,
        "hints": hints,
        "test_patch": "",  # If available
        "version": "2.0"
    }
```

---

## Step 7: Implementation Plan

### Phase 1: Memory Layer (L1/L2/L3)
```
1. Update memory_plugin to return structured data
   - L1: Similar failures with repair approaches
   - L2: Repo patterns and co-occurring issues
   - L3: Best practices and tool patterns

2. Define memory query interface
   - query_memory(issue, ci_failure, levels=["L1", "L2", "L3"])
   - Returns: {l1_matches: [...], l2_matches: [...], l3_matches: [...]}

3. Test memory retrieval
   - Verify L1 returns similar failures
   - Verify L2 returns repo-specific patterns
   - Verify L3 returns best practices
```

### Phase 2: Problem & Repair Format
```
1. Define data classes
   - Problem, RepairInstruction, RepairPlan

2. Implement extraction
   - extract_primary_problems(ci_failure)
   - extract_related_problems(memory, primary)
   - synthesize_repair_instructions(memory, problems)

3. Test format
   - Baseline: Only primary problems
   - L1: + repair success rates
   - L1+L2: + related issues
   - L1+L2+L3: + best practices
```

### Phase 3: Agent Interface
```
1. Define base RepairAgent class

2. Implement Codex agent
   - prepare_prompt(plan) → Codex-format prompt
   - execute(prompt) → Run Codex
   - verify(result) → Check fix

3. Implement Mini-SWE agent (optional)
   - prepare_prompt(plan) → Mini-SWE format
   
4. Test agent-agnostic interface
   - Same plan → different agent formats
```

### Phase 4: Integration
```
1. Update run_codex_ci_repair.py
   - Use build_repair_plan() instead of extract_problem_list()
   - Use agent.prepare_prompt(plan) instead of compose_issue_document()

2. Test full pipeline
   - Issue → CI failure → Memory query → Plan → Prompt → Codex → Fix

3. Evaluate
   - Compare baseline vs L1 vs L1+L2 vs L1+L2+L3
   - Measure success rates, time, cost
```

---

## Summary

**Your approach is correct:**

1. ✅ **First:** Design L1/L2/L3 memory structure
   - L1: Similar failures, repair approaches
   - L2: Repo patterns, co-occurring issues
   - L3: Best practices, tool strategies

2. ✅ **Second:** Define universal format
   - Problem (from CI + memory)
   - RepairInstruction (from memory analysis)
   - RepairPlan (comprehensive)

3. ✅ **Third:** Make agent-agnostic
   - Base RepairAgent interface
   - Codex implementation
   - Mini-SWE implementation
   - Any future agent

4. ✅ **Fourth:** Fix prompts
   - plan.to_codex_prompt()
   - plan.to_mini_swe_format()
   - Clean, maintainable

**Result:** 
- Clear data model
- Agent-agnostic design
- Memory-driven repair
- Scalable and maintainable

---

## Next Steps

1. Review this design
2. Refine L1/L2/L3 structures
3. Implement data classes
4. Update memory plugin
5. Build agent interface
6. Test and iterate

**Foundation first, implementation second!** ✅

---

## Date
2026-07-30
