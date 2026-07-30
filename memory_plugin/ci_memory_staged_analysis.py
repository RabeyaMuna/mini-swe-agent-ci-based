"""
Staged Memory Analysis for CI Failure Repair

Multi-stage pipeline with smart filtering and structure:
- Stage 0: L1 analysis (direct + dependent problems)
- Stage 1: L2 analysis (common patterns + consecutive failures)
- Stage 2: L3 analysis (only if L1/L2 insufficient)
- Stage 3: Deduplication (remove duplicates)
- Stage 4: Structuring (proper format for agent)
"""

import json
import logging
import math
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None  # type: ignore

# Import optimal L2 analysis pipeline
from memory_plugin.ci_memory_l2_analysis import staged_l2_analysis

logger = logging.getLogger(__name__)


def _call_llm(llm: Any, prompt: str) -> str:
    """Universal LLM caller."""
    if llm is None:
        return ""
    try:
        try:
            from langchain_core.messages import HumanMessage

            result = llm.invoke([HumanMessage(content=prompt)])
            return (getattr(result, "content", None) or "").strip()
        except (ImportError, AttributeError):
            pass
        result = llm.invoke(prompt)
        if hasattr(result, "content"):
            return (result.content or "").strip()
        return str(result).strip()
    except Exception:
        pass
    try:
        return str(llm(prompt)).strip()
    except Exception:
        return ""


def _parse_json(response: str, context: str = "") -> Dict:
    """Parse JSON object from LLM response."""
    import re

    try:
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*", "", response)
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(response)
    except Exception as e:
        logger.warning(f"[{context}] JSON parse error: {e}")
        return {}


def _parse_json_array(response: str, context: str = "") -> List[Dict]:
    """Parse JSON array from LLM response."""
    import re

    try:
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*", "", response)
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        return []
    except Exception as e:
        logger.warning(f"[{context}] JSON array parse error: {e}")
        return []


def _compute_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    try:
        if cosine_similarity is None or np is None:
            dot = sum(float(a) * float(b) for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(float(a) * float(a) for a in vec1))
            norm2 = math.sqrt(sum(float(b) * float(b) for b in vec2))
            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0
            return dot / (norm1 * norm2)
        v1 = np.array(vec1).reshape(1, -1)
        v2 = np.array(vec2).reshape(1, -1)
        return float(cosine_similarity(v1, v2)[0][0])
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0: L1 Analysis - Direct + Dependent Problems
# ══════════════════════════════════════════════════════════════════════════════


def stage0_analyze_l1(
    ci_failure: Dict[str, Any],
    l1_memories: List[Dict[str, Any]],
    llm: Any,
    current_workflow_sequence: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    STAGE 0: Analyze L1 memories based on CI failure.

    Steps:
    1. Select L1 problems matching CI failure
    2. Fetch dependent problems (enables/enabled_by chains)
    3. List all L1 problems

    Returns:
        List of L1-sourced problems with dependencies
    """

    if not l1_memories:
        logger.info("[Stage 0] No L1 memories available")
        return []

    logger.info(
        f"[Stage 0] Analyzing {len(l1_memories)} L1 memories against CI failure"
    )

    prompt = f"""Analyze L1 file-level failures against current CI failure.

**CURRENT CI FAILURE:**
```json
{json.dumps(ci_failure, indent=2)}
```

**L1 FILE-LEVEL MEMORIES (with dependency chains):**
```json
{json.dumps(l1_memories, indent=2)}
```

**YOUR TASK:**

1. **Select relevant L1 problems** that match the CI failure:
   - Same or similar validation_cmd
   - Same failure_type or issue_type
   - Same or related files

2. **Fetch dependent problems** from L1:
   - For each selected problem, check `enables` field (problem IDs it enables)
   - For each selected problem, check `enabled_by` field (problem IDs that enable it)
   - Find those problem IDs in the L1 memories list
   - Extract their full details

3. **List ALL L1 problems** (direct matches + dependencies)

**OUTPUT FORMAT:**

Return JSON object:

{{
  "direct_problems": [
    {{
      "problem_id": "l1_1",
      "source": "L1-direct",
      "description": "exact 'problem' field from L1",
      "root_cause": "exact 'root_cause' field from L1",
      "fix_strategy": "exact 'how_fixed' + 'why_fix_works' from L1",
      "files": ["file from L1"],
      "validation_cmd": "CHECK SOURCE field from current workflow - use that instead of L1 validation_cmd",
      "failure_type": "failure_type from L1",
      "issue_type": "issue_type from L1"
    }}
  ],
  "dependent_problems": [
    {{
      "problem_id": "l1_2",
      "source": "L1-dependent",
      "enabled_by": "l1_1",
      "description": "...",
      "root_cause": "...",
      "fix_strategy": "...",
      "files": ["..."],
      "validation_cmd": "CHECK SOURCE field from current workflow",
      "failure_type": "...",
      "issue_type": "..."
    }}
  ]
}}

**RULES:**
- Use EXACT text from L1 fields
- Include validation_cmd, failure_type, issue_type for each problem
- If no dependencies found, return empty array for dependent_problems

Return ONLY the JSON object:
"""

    try:
        response = _call_llm(llm, prompt)
        result = _parse_json(response, "Stage0-L1")

        direct = result.get("direct_problems", [])
        dependent = result.get("dependent_problems", [])

        # Fix validation_cmd from current workflow (not L1 memory)
        all_problems = direct + dependent
        if current_workflow_sequence:
            for prob in all_problems:
                memory_val_cmd = prob.get("validation_cmd", "")
                correct_val_cmd = _match_validation_cmd_from_workflow(
                    memory_val_cmd, current_workflow_sequence
                )
                prob["validation_cmd"] = correct_val_cmd

        logger.info(
            f"[Stage 0] L1 Analysis: {len(direct)} direct + {len(dependent)} dependent = {len(all_problems)} total"
        )

        return all_problems

    except Exception as e:
        logger.warning(f"[Stage 0] L1 analysis failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: L2 Analysis - Common Patterns + Consecutive Failures
# ══════════════════════════════════════════════════════════════════════════════


def stage1_phase1_identify_common_patterns(
    ci_failure: Dict[str, Any],
    l2_memories: List[Dict[str, Any]],
    llm: Any,
    current_workflow_sequence: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    STAGE 1 - Phase 1: Identify COMMON failure patterns from L2.

    Uses cosine similarity on:
    - validation_cmd
    - failure_type
    - issue_type

    Then LLM analyzes root causes to identify common patterns.
    """

    if not l2_memories:
        logger.info("[Stage 1.1] No L2 memories available")
        return []

    logger.info(
        f"[Stage 1.1] Checking common patterns in {len(l2_memories)} L2 trajectories"
    )

    # First, extract validation_cmd + failure_type + issue_type signatures
    signatures = {}
    for mem in l2_memories:
        trajectory = mem.get("repair_trajectory", [])
        for step in trajectory:
            validation_cmd = step.get("validation_cmd", "")

            # Get failure types from problems in this step
            problems = step.get("problems", [])
            for prob in problems:
                failure_type = prob.get("failure_type", "unknown")
                issue_type = prob.get("issue_type", "unknown")

                sig = f"{validation_cmd}|{failure_type}|{issue_type}"

                if sig not in signatures:
                    signatures[sig] = []
                signatures[sig].append(
                    {
                        "step": step,
                        "problem": prob,
                        "issue_id": mem.get("issue_id"),
                        "repo": mem.get("repo"),
                    }
                )

    # Find common signatures (appear in multiple issues)
    common_sigs = {sig: data for sig, data in signatures.items() if len(data) >= 2}

    logger.info(
        f"[Stage 1.1] Found {len(common_sigs)} common patterns (appear in >=2 issues)"
    )

    if not common_sigs:
        return []

    # Ask LLM to analyze common patterns
    prompt = f"""Identify COMMON failure patterns.

**CURRENT CI FAILURE:**
```json
{json.dumps(ci_failure, indent=2)}
```

**COMMON PATTERNS FOUND (appear in multiple issues):**
```json
{json.dumps(common_sigs, indent=2)}
```

**YOUR TASK:**

For each common pattern:
1. Check if it's relevant to the current CI failure
2. Extract the representative problem (most detailed one)
3. Include root_cause, how_fixed, why_fix_works

**OUTPUT FORMAT:**

Return JSON array of common problems:

[
  {{
    "problem_id": "l2_common_1",
    "source": "L2-common",
    "commonality": "Appears in X issues",
    "description": "exact 'problem' field from pattern",
    "root_cause": "exact 'root_cause' field",
    "fix_strategy": "exact 'how_fixed' + 'why_fix_works'",
    "files": ["affected files"],
    "validation_cmd": "validation command",
    "failure_type": "failure type",
    "issue_type": "issue type"
  }}
]

**RULES:**
- Only include patterns relevant to current CI failure
- Use exact text from L2 data
- If no relevant common patterns, return empty array: []

Return ONLY the JSON array:
"""

    try:
        response = _call_llm(llm, prompt)
        common_problems = _parse_json_array(response, "Stage1.1-Common")

        # Fix validation_cmd from current workflow
        if current_workflow_sequence:
            for prob in common_problems:
                memory_val_cmd = prob.get("validation_cmd", "")
                correct_val_cmd = _match_validation_cmd_from_workflow(
                    memory_val_cmd, current_workflow_sequence
                )
                prob["validation_cmd"] = correct_val_cmd

        logger.info(
            f"[Stage 1.1] Identified {len(common_problems)} relevant common patterns"
        )

        return common_problems

    except Exception as e:
        logger.warning(f"[Stage 1.1] Common pattern analysis failed: {e}")
        return []


def stage1_phase2_identify_consecutive_failures(
    ci_failure: Dict[str, Any],
    l2_memories: List[Dict[str, Any]],
    llm: Any,
    current_workflow_sequence: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    STAGE 1 - Phase 2: Identify CONSECUTIVE failures from L2 trajectories.

    Based on similarity to CI failure, extract what problems came AFTER
    in the repair sequence.
    """

    if not l2_memories:
        logger.info("[Stage 1.2] No L2 memories available")
        return []

    logger.info(
        f"[Stage 1.2] Checking consecutive failures in {len(l2_memories)} L2 trajectories"
    )

    prompt = f"""Identify CONSECUTIVE failures that may occur after fixing the primary problem.

**CURRENT CI FAILURE:**
```json
{json.dumps(ci_failure, indent=2)}
```

**L2 REPAIR TRAJECTORIES:**
```json
{json.dumps(l2_memories, indent=2)}
```

**YOUR TASK:**

1. **Find similar primary failures** in L2 trajectories:
   - Check step 1 of each trajectory
   - Match against current CI failure (validation_cmd, failure_type, issue_type)

2. **Extract consecutive problems** from those trajectories:
   - Look at steps 2, 3, 4... (hidden problems)
   - These are problems that appeared AFTER fixing the primary

3. **List consecutive problems** with full details

**OUTPUT FORMAT:**

Return JSON array:

[
  {{
    "problem_id": "l2_consecutive_1",
    "source": "L2-consecutive",
    "appears_after": "primary fix",
    "description": "exact 'problem' from step.problems",
    "root_cause": "exact 'root_cause'",
    "fix_strategy": "exact 'how_fixed' + 'why_fix_works'",
    "files": ["affected files"],
    "validation_cmd": "validation command",
    "failure_type": "failure type",
    "issue_type": "issue type",
    "revealed_by": "what validation revealed this"
  }}
]

**RULES:**
- Only include problems from trajectories with similar primary failures
- Use exact text from L2 trajectory data
- Include revealed_by field to show dependency
- If no similar trajectories found, return empty array: []

Return ONLY the JSON array:
"""

    try:
        response = _call_llm(llm, prompt)
        consecutive_problems = _parse_json_array(response, "Stage1.2-Consecutive")

        # Fix validation_cmd from current workflow
        if current_workflow_sequence:
            for prob in consecutive_problems:
                memory_val_cmd = prob.get("validation_cmd", "")
                correct_val_cmd = _match_validation_cmd_from_workflow(
                    memory_val_cmd, current_workflow_sequence
                )
                prob["validation_cmd"] = correct_val_cmd

        logger.info(
            f"[Stage 1.2] Identified {len(consecutive_problems)} consecutive problems"
        )

        return consecutive_problems

    except Exception as e:
        logger.warning(f"[Stage 1.2] Consecutive failure analysis failed: {e}")
        return []


def stage1_analyze_l2(
    ci_failure: Dict[str, Any],
    l2_memories: List[Dict[str, Any]],
    llm: Any,
    current_workflow_sequence: List[Dict[str, Any]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    STAGE 1: Complete L2 analysis.

    Returns:
        (common_problems, consecutive_problems)
    """

    # Phase 1: Common patterns
    common = stage1_phase1_identify_common_patterns(
        ci_failure, l2_memories, llm, current_workflow_sequence
    )

    # Phase 2: Consecutive failures
    consecutive = stage1_phase2_identify_consecutive_failures(
        ci_failure, l2_memories, llm, current_workflow_sequence
    )

    return common, consecutive


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: L3 Analysis - Universal Patterns (only if L1/L2 insufficient)
# ══════════════════════════════════════════════════════════════════════════════


def stage2_analyze_l3(
    ci_failure: Dict[str, Any], l3_memories: List[Dict[str, Any]], llm: Any
) -> List[Dict[str, Any]]:
    """
    STAGE 2: Analyze L3 universal patterns.

    Always called if L3 memories are available.
    Deduplication happens later in Stage 3.
    """

    if not l3_memories:
        logger.info("[Stage 2] No L3 memories available")
        return []

    logger.info(f"[Stage 2] Analyzing {len(l3_memories)} L3 universal patterns")

    prompt = f"""Find relevant universal patterns for CI failure.

**CURRENT CI FAILURE:**
```json
{json.dumps(ci_failure, indent=2)}
```

**L3 UNIVERSAL PATTERNS:**
```json
{json.dumps(l3_memories, indent=2)}
```

**YOUR TASK:**

1. Match CI failure against L3 patterns:
   - Same failure_type
   - Same issue_type
   - Similar failure patterns

2. Extract typical problems from matched patterns:
   - Check failure_patterns field
   - Check fix_approach field

**OUTPUT FORMAT:**

Return JSON array:

[
  {{
    "problem_id": "l3_1",
    "source": "L3-pattern",
    "pattern_name": "pattern name from L3",
    "description": "problem from pattern",
    "fix_strategy": "fix_approach from pattern",
    "failure_type": "failure type",
    "issue_type": "issue type"
  }}
]

**RULES:**
- Only include patterns matching current CI failure
- Use exact text from L3 data
- If no matches, return empty array: []

Return ONLY the JSON array:
"""

    try:
        response = _call_llm(llm, prompt)
        l3_problems = _parse_json_array(response, "Stage2-L3")

        logger.info(f"[Stage 2] Identified {len(l3_problems)} L3 pattern problems")

        return l3_problems

    except Exception as e:
        logger.warning(f"[Stage 2] L3 analysis failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Deduplication
# ══════════════════════════════════════════════════════════════════════════════


def stage3_deduplicate(
    all_problems: List[Dict[str, Any]], llm: Any
) -> List[Dict[str, Any]]:
    """
    STAGE 3: Remove duplicate problems.

    Deduplication criteria:
    - Same validation_cmd + similar description
    - Priority: L1 > L2-consecutive > L2-common > L3
    """

    if not all_problems:
        return []

    if len(all_problems) <= 10:
        logger.info(f"[Stage 3] {len(all_problems)} problems, minimal dedup needed")
        # Simple dedup by validation_cmd
        seen = {}
        unique = []
        for prob in all_problems:
            key = prob.get("validation_cmd", "")
            if key not in seen:
                seen[key] = prob
                unique.append(prob)
        return unique

    logger.info(f"[Stage 3] Deduplicating {len(all_problems)} problems")

    prompt = f"""Remove duplicate problems, keeping highest priority.

**ALL CANDIDATE PROBLEMS:**
```json
{json.dumps(all_problems, indent=2)}
```

**YOUR TASK:**

1. **Find duplicates:**
   - Same validation_cmd
   - Similar description
   - Same failure_type + issue_type

2. **Priority when duplicates found:**
   - L1-direct > L1-dependent > L2-consecutive > L2-common > L3-pattern
   - Keep the highest priority version

3. **Keep top 10** most relevant unique problems

**OUTPUT:**

Return JSON array of unique problems (max 10):

[same structure as input]

Return ONLY the JSON array:
"""

    try:
        response = _call_llm(llm, prompt)
        unique_problems = _parse_json_array(response, "Stage3-Dedup")

        logger.info(
            f"[Stage 3] Deduplicated: {len(all_problems)} → {len(unique_problems)} unique"
        )

        return unique_problems[:10]

    except Exception as e:
        logger.warning(f"[Stage 3] Deduplication failed: {e}")
        # Fallback: simple dedup
        seen = {}
        unique = []
        for prob in all_problems:
            key = f"{prob.get('validation_cmd', '')}|{prob.get('description', '')[:50]}"
            if key not in seen:
                seen[key] = prob
                unique.append(prob)
        return unique[:10]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Structuring - Proper Format for Agent
# ══════════════════════════════════════════════════════════════════════════════


def stage4_structure_problems(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    STAGE 4: Structure problems for mini-swe-agent.

    Mini-swe-agent has its own agentic loop (up to 250 steps):
    - Checks if problem exists by reading files
    - Applies fix based on guidance
    - Validates itself
    - Self-corrects if needed

    So we only need to provide:
    - problem description
    - root_cause
    - how_fixed guidance
    - why_fix_works reasoning
    - files to change

    Output format (simplified):
    {
      "problem_id": int,
      "is_primary": bool,
      "source": str,
      "problem": str,
      "root_cause": str,
      "how_fixed": str,
      "why_fix_works": str,
      "files": [str]
    }
      "verification": {
        "validation_cmd": str,
        "check_first": bool,
        "expected_result": str
      }
    }
    """

    if not problems:
        return []

    logger.info(f"[Stage 4] Structuring {len(problems)} problems for agent")

    # Direct structuring - no LLM needed
    # Don't add problem_id here - let caller number them serially after combining CI + memory
    structured = []
    for prob in problems:
        # Map to cibench expected field names
        problem_text = prob.get("problem") or prob.get("description", "")
        root_cause = prob.get("root_cause", "")
        how_fixed = prob.get("how_fixed", "")
        why_fix_works = prob.get("why_fix_works", "")

        # Combine how_fixed + why_fix_works into fix_strategy
        fix_strategy_parts = []
        if how_fixed:
            fix_strategy_parts.append(how_fixed)
        if why_fix_works:
            fix_strategy_parts.append(f"Why this works: {why_fix_works}")
        fix_strategy = "\n\n".join(fix_strategy_parts) if fix_strategy_parts else ""

        structured.append(
            {
                "problem_statement": problem_text,  # cibench expects this field name
                "root_cause": root_cause,
                "fix_strategy": fix_strategy,  # Combined how_fixed + why_fix_works
                "files": prob.get("files", []),
                # Context fields:
                "verification_cmd": prob.get(
                    "validation_cmd", ""
                ),  # cibench expects this field name
                "error_type": prob.get("failure_type", ""),
                "issue_type": prob.get("issue_type", ""),
            }
        )

    logger.info(f"[Stage 4] Structured {len(structured)} problems")
    return structured


# OLD COMPLEX LLM CODE BELOW - REMOVED
def _match_validation_cmd_from_workflow(
    memory_validation_cmd: str, current_workflow_sequence: List[Dict[str, Any]]
) -> str:
    """
    Match validation command from memory to current workflow.

    Memory may have: "python -m mypy py"
    Current workflow has: {"source": "framework/dev/test.sh", "validation_cmd": "python -m mypy py"}

    We should return the SOURCE command from current workflow, not the validation_cmd from memory.

    Args:
        memory_validation_cmd: Validation command from memory (may be wrong)
        current_workflow_sequence: Validation sequence from current CI workflow

    Returns:
        Correct validation command from current workflow (the "source" field)
    """
    if not current_workflow_sequence:
        return memory_validation_cmd

    # Try to match by validation_cmd similarity
    for step in current_workflow_sequence:
        workflow_cmd = step.get("validation_cmd", "")
        source = step.get("source", "")

        # If memory command is similar to workflow validation_cmd
        if workflow_cmd and memory_validation_cmd:
            # Check for substring match (e.g., "mypy py" in "python -m mypy py")
            memory_lower = memory_validation_cmd.lower()
            workflow_lower = workflow_cmd.lower()

            # Extract key parts (mypy, pytest, mdformat, etc.)
            key_tools = [
                "mypy",
                "pytest",
                "mdformat",
                "docformatter",
                "docstrfmt",
                "ruff",
                "pylint",
                "black",
                "isort",
                "flake8",
            ]

            for tool in key_tools:
                if tool in memory_lower and tool in workflow_lower:
                    # Match found - return the SOURCE command
                    if source:
                        logger.info(
                            f"[Validation Match] '{memory_validation_cmd}' → '{source}' (matched on '{tool}')"
                        )
                        return source
                    else:
                        # No source, use workflow validation_cmd
                        return workflow_cmd

    # No match found, return empty (don't validate)
    logger.warning(
        f"[Validation Match] No match for '{memory_validation_cmd}' in current workflow - skipping validation"
    )
    return ""  # Empty means don't validate


def staged_memory_analysis(
    problem_1_primary: Dict[str, Any],
    l1_memories: List[Dict[str, Any]],
    l2_memories: List[Dict[str, Any]],
    l3_memories: List[Dict[str, Any]],
    llm: Any,
    current_workflow_sequence: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    DEFAULT PIPELINE for consecutive problem extraction.

    IMPORTANT:
    - Problem #1 (PRIMARY) is ALWAYS from CI failure (passed as input)
    - This pipeline finds Problems #2-N (CONSECUTIVE) based on Problem #1
    - Uses memory (L1/L2/L3) to predict what problems will appear AFTER fixing Problem #1

    Pipeline Stages:
    - Stage 0: L1 analysis (find dependent problems from L1)
    - Stage 1: L2 analysis (common patterns + consecutive failures from trajectories)
    - Stage 2: L3 analysis (universal patterns - always if L3 available)
    - Stage 3: Deduplication (remove duplicates from L1+L2+L3)
    - Stage 4: Structuring (format for agent)

    Args:
        problem_1_primary: The PRIMARY CI failure (Problem #1) - already determined
        l1_memories: L1 file-level memories
        l2_memories: L2 repo-level trajectories
        l3_memories: L3 universal patterns
        llm: LLM for analysis

    Returns:
        List of CONSECUTIVE problems (Problems #2-N) ready for agent
        Note: Does NOT include Problem #1 (that's handled separately)
    """

    logger.info("=" * 80)
    logger.info("CONSECUTIVE PROBLEM EXTRACTION PIPELINE")
    logger.info("Priority: CI Failure (Problem #1) → Memory-based Consecutive Problems")
    logger.info("=" * 80)

    all_consecutive = []

    # Extract CI failure context for matching
    ci_failure_context = {
        "validation_cmd": problem_1_primary.get("validation_cmd", ""),
        "failure_type": problem_1_primary.get("failure_type", ""),
        "issue_type": problem_1_primary.get("issue_type", ""),
        "description": problem_1_primary.get("description", ""),
        "files": problem_1_primary.get("files", []),
    }

    # Extract current workflow sequence (if not passed, try to get from problem_1)
    if current_workflow_sequence is None:
        current_workflow_sequence = problem_1_primary.get("validation_sequence", [])

    # ── STAGE 0: L1 Analysis - Find Dependent Problems ──
    logger.info("\n[STAGE 0] Analyzing L1 for dependent problems...")
    l1_problems = stage0_analyze_l1(
        ci_failure_context, l1_memories, llm, current_workflow_sequence
    )
    all_consecutive.extend(l1_problems)
    logger.info(f"[STAGE 0] Found {len(l1_problems)} L1 dependent problems")

    # ── STAGE 1: L2 Analysis - OPTIMAL PIPELINE (Common + Consecutive) ──
    logger.info("\n[STAGE 1] Analyzing L2 trajectories with OPTIMAL pipeline...")
    l2_consecutive = staged_l2_analysis(ci_failure_context, l2_memories, llm)
    all_consecutive.extend(l2_consecutive)
    logger.info(
        f"[STAGE 1] Found {len(l2_consecutive)} L2 consecutive problems (prioritized by commonality)"
    )

    # ── STAGE 2: L3 Analysis (universal patterns) ──
    if l3_memories and llm:
        logger.info(
            f"\n[STAGE 2] Analyzing {len(l3_memories)} L3 universal patterns..."
        )
        l3_problems = stage2_analyze_l3(ci_failure_context, l3_memories, llm)
        all_consecutive.extend(l3_problems)
        logger.info(f"[STAGE 2] Found {len(l3_problems)} L3 pattern problems")
    else:
        logger.info("\n[STAGE 2] SKIPPED - No L3 memories or LLM available")

    if not all_consecutive:
        logger.info("\n[PIPELINE] No consecutive problems found in memory")
        logger.info("=" * 80)
        return []

    # ── STAGE 3: Deduplication ──
    logger.info(f"\n[STAGE 3] Deduplicating {len(all_consecutive)} problems...")
    unique_problems = stage3_deduplicate(all_consecutive, llm)
    logger.info(f"[STAGE 3] Reduced to {len(unique_problems)} unique problems")

    # ── STAGE 4: Structuring ──
    logger.info(f"\n[STAGE 4] Structuring {len(unique_problems)} problems for agent...")
    structured_problems = stage4_structure_problems(unique_problems)
    logger.info(f"[STAGE 4] Structured {len(structured_problems)} consecutive problems")

    logger.info("=" * 80)
    logger.info(
        f"PIPELINE COMPLETE: {len(structured_problems)} consecutive problems (Problems #2-{len(structured_problems) + 1})"
    )
    logger.info("=" * 80)

    return structured_problems
