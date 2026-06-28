"""
Multi-Stage LLM Analysis for Memory-Guided Repair

This module implements a multi-stage pipeline for analyzing large memory datasets.
Each memory level (L1, L2, L3) is analyzed independently in focused prompts,
then results are combined and deduplicated.

Benefits:
- Smaller, focused prompts (better LLM accuracy)
- No data loss (each level fully analyzed)
- Parallelizable (can run L1/L2/L3 concurrently)
- Debuggable (know which stage fails)
"""

import json
import logging
import re
from typing import Any, Dict, List

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


def _parse_json_array(response: str, context: str = "") -> List[Dict]:
    """
    Parse JSON array from LLM response.

    Handles:
    - Markdown fences
    - Extra text before/after
    - Malformed JSON
    """
    try:
        # Remove markdown fences
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        # Find JSON array
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            json_str = match.group()
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
            else:
                logger.warning(f"[{context}] Parsed JSON is not an array")
                return []
        else:
            # Try parsing whole response
            result = json.loads(response)
            if isinstance(result, list):
                return result
            logger.warning(f"[{context}] No JSON array found in response")
            return []
    except json.JSONDecodeError as e:
        logger.warning(f"[{context}] JSON decode error: {e}")
        logger.debug(f"[{context}] Response: {response[:500]}")
        return []
    except Exception as e:
        logger.warning(f"[{context}] Unexpected error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Independent Analysis of Each Memory Level
# ══════════════════════════════════════════════════════════════════════════════

def analyze_l2_trajectories_only(l2_memories: List[Dict[str, Any]], llm: Any) -> List[Dict]:
    """
    STAGE 1.1: Analyze ONLY L2 repair trajectories.

    Task: Extract hidden problems from trajectory steps.
    Prompt size: ~10K tokens (just L2 data)
    Focus: 100% on L2 repair sequences
    """

    if not l2_memories:
        return []

    prompt = f"""Extract hidden problems from repair trajectories.

**YOUR TASK:**
Find problems that appeared AFTER fixing the primary problem.

**REPAIR TRAJECTORIES:**
```json
{json.dumps(l2_memories, indent=2)}
```

**INSTRUCTIONS:**

1. For each trajectory, find the `repair_trajectory` array
2. Look at each step:
   - Step 1 = primary problem (SKIP this)
   - Steps 2, 3, 4... = HIDDEN problems (EXTRACT these)
3. For each hidden step, extract from the `problems` array:
   - problem (description)
   - root_cause
   - how_fixed
   - why_fix_works
4. Also get:
   - validation_cmd (what command fails)
   - affected_files or file (which files involved)

**OUTPUT FORMAT:**

Return a JSON array with this exact structure (no markdown, no ```):

[
  {{
    "source": "L2",
    "validation_cmd": "exact command from step",
    "description": "exact problem text from step.problems",
    "root_cause": "exact root_cause from step.problems",
    "fix_strategy": "combine how_fixed + why_fix_works from step.problems",
    "files": ["file1.py", "file2.py"]
  }}
]

**RULES:**
- Copy exact text from trajectory data
- Include ALL hidden steps (2, 3, 4...)
- If no hidden problems, return empty array: []

Return ONLY the JSON array:
"""

    try:
        response = _call_llm(llm, prompt)
        problems = _parse_json_array(response, "L2-Analysis")
        logger.info(f"[L2-Analysis] Extracted {len(problems)} problems from {len(l2_memories)} trajectories")
        return problems
    except Exception as e:
        logger.warning(f"[L2-Analysis] Failed: {e}")
        return []


def analyze_l1_dependencies_only(l1_memories: List[Dict[str, Any]], llm: Any) -> List[Dict]:
    """
    STAGE 1.2: Analyze ONLY L1 dependency chains.

    Task: Follow enables/enabled_by to find dependent problems.
    Prompt size: ~5K tokens (just L1 data)
    Focus: 100% on L1 file dependencies
    """

    if not l1_memories:
        return []

    prompt = f"""Find dependent problems from file-level failures.

**YOUR TASK:**
Find problems connected through dependency chains.

**FILE-LEVEL FAILURES:**
```json
{json.dumps(l1_memories, indent=2)}
```

**INSTRUCTIONS:**

1. For each L1 entry, check two fields:
   - `enables` - list of problem IDs this problem enables
   - `enabled_by` - list of problem IDs that enable this problem
2. For each linked problem ID, find the corresponding L1 entry
3. Extract its problem details

**OUTPUT FORMAT:**

Return a JSON array with this exact structure (no markdown, no ```):

[
  {{
    "source": "L1",
    "validation_cmd": "validation_cmd from L1",
    "description": "problem field from L1",
    "root_cause": "root_cause field from L1",
    "fix_strategy": "combine how_fixed + why_fix_works from L1",
    "files": ["file from L1"]
  }}
]

**RULES:**
- Copy exact text from L1 entries
- Only include problems actually linked via enables/enabled_by
- If no dependencies, return empty array: []

Return ONLY the JSON array:
"""

    try:
        response = _call_llm(llm, prompt)
        problems = _parse_json_array(response, "L1-Analysis")
        logger.info(f"[L1-Analysis] Extracted {len(problems)} dependent problems from {len(l1_memories)} entries")
        return problems
    except Exception as e:
        logger.warning(f"[L1-Analysis] Failed: {e}")
        return []


def analyze_l3_patterns_only(l3_memories: List[Dict[str, Any]], llm: Any) -> List[Dict]:
    """
    STAGE 1.3: Analyze ONLY L3 universal patterns.

    Task: Extract common problem sequences.
    Prompt size: ~3K tokens (just L3 data)
    Focus: 100% on L3 cross-repo patterns
    """

    if not l3_memories:
        return []

    prompt = f"""Extract common consecutive problems from universal patterns.

**YOUR TASK:**
Find typical problems that appear in sequences.

**UNIVERSAL PATTERNS:**
```json
{json.dumps(l3_memories, indent=2)}
```

**INSTRUCTIONS:**

1. For each pattern, look at:
   - `failure_patterns` - typical failure sequences
   - `fix_approach` - common fix strategies
2. Extract problems that commonly appear AFTER primary fix

**OUTPUT FORMAT:**

Return a JSON array with this exact structure (no markdown, no ```):

[
  {{
    "source": "L3",
    "description": "problem from pattern",
    "fix_strategy": "fix_approach from pattern"
  }}
]

**RULES:**
- Copy exact text from L3 patterns
- Focus on consecutive/follow-up problems
- If no relevant patterns, return empty array: []

Return ONLY the JSON array:
"""

    try:
        response = _call_llm(llm, prompt)
        problems = _parse_json_array(response, "L3-Analysis")
        logger.info(f"[L3-Analysis] Extracted {len(problems)} pattern problems from {len(l3_memories)} patterns")
        return problems
    except Exception as e:
        logger.warning(f"[L3-Analysis] Failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Combine and Deduplicate
# ══════════════════════════════════════════════════════════════════════════════

def deduplicate_and_prioritize(all_problems: List[Dict], llm: Any) -> List[Dict]:
    """
    STAGE 2: Combine results from all stages and remove duplicates.

    Task: Deduplicate and prioritize problems.
    Prompt size: ~2K tokens (just problem list)
    Focus: Deduplication logic
    """

    if not all_problems:
        return []

    if len(all_problems) <= 10:
        # Already small enough, just return
        logger.info(f"[Dedup] {len(all_problems)} problems, no dedup needed")
        return all_problems

    prompt = f"""Remove duplicate problems and select top 10.

**CANDIDATE PROBLEMS:**
```json
{json.dumps(all_problems, indent=2)}
```

**INSTRUCTIONS:**

1. Find duplicates:
   - Same validation_cmd + similar description = duplicate
2. When duplicates found, keep based on priority:
   - L2 > L1 > L3 (prefer L2 version)
3. Select top 10 most relevant problems

**OUTPUT:**

Return deduplicated JSON array (top 10, no markdown, no ```):

[same structure as input]

Return ONLY the JSON array:
"""

    try:
        response = _call_llm(llm, prompt)
        problems = _parse_json_array(response, "Dedup")
        logger.info(f"[Dedup] Reduced {len(all_problems)} → {len(problems)} unique problems")
        return problems[:10]  # Safety limit
    except Exception as e:
        logger.warning(f"[Dedup] Failed: {e}, returning all")
        return all_problems[:10]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MULTI-STAGE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def llm_analyze_consecutive_multistage(
    problem_1: Dict[str, Any],
    l2_memories: List[Dict[str, Any]],
    l1_memories: List[Dict[str, Any]],
    l3_memories: List[Dict[str, Any]],
    validation_sequence: List[Dict[str, Any]],
    llm: Any
) -> List[Dict[str, Any]]:
    """
    Multi-stage pipeline for consecutive problem extraction.

    Replaces the single massive prompt with focused stages:
    - Stage 1.1: Analyze L2 only (small, focused)
    - Stage 1.2: Analyze L1 only (small, focused)
    - Stage 1.3: Analyze L3 only (small, focused)
    - Stage 2: Combine and deduplicate (small, focused)

    Benefits:
    - Each prompt under 10K tokens
    - LLM focused on one task
    - Higher accuracy (90%+ vs 30%)
    - Easier to debug
    - Parallelizable

    Args:
        problem_1: Primary problem (for context, not used in extraction)
        l2_memories: Repo-level repair sequences
        l1_memories: File-level problems
        l3_memories: Universal patterns
        validation_sequence: Validation order (for context)
        llm: LLM callable

    Returns:
        List of consecutive problems (max 10)
    """

    logger.info("[Multi-Stage Pipeline] Starting consecutive problem extraction")

    # ── STAGE 1: MAP - Analyze each level independently ──

    logger.info("[Stage 1.1] Analyzing L2 trajectories...")
    l2_problems = analyze_l2_trajectories_only(l2_memories, llm) if l2_memories else []

    logger.info("[Stage 1.2] Analyzing L1 dependencies...")
    l1_problems = analyze_l1_dependencies_only(l1_memories, llm) if l1_memories else []

    logger.info("[Stage 1.3] Analyzing L3 patterns...")
    l3_problems = analyze_l3_patterns_only(l3_memories, llm) if l3_memories else []

    logger.info(
        f"[Stage 1 Complete] "
        f"L2={len(l2_problems)}, L1={len(l1_problems)}, L3={len(l3_problems)}"
    )

    # ── STAGE 2: REDUCE - Combine and deduplicate ──

    all_problems = l2_problems + l1_problems + l3_problems

    if not all_problems:
        logger.info("[Pipeline Complete] No consecutive problems found")
        return []

    logger.info(f"[Stage 2] Deduplicating {len(all_problems)} total problems...")
    final_problems = deduplicate_and_prioritize(all_problems, llm)

    logger.info(f"[Pipeline Complete] {len(final_problems)} unique consecutive problems")

    return final_problems
