"""
LLM-based Analysis for Memory-Guided Repair

This module contains LLM prompts for analyzing memories and selecting problems.
User requirement: "All analysis and selection should be done by PROMPT"
"""

import json
import logging
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


def llm_analyze_l1_for_problem_1(
  ci_problem: Dict[str, Any],
  l1_memories: List[Dict[str, Any]],
  llm: Any
) -> Dict[str, Any]:
  """
  LLM PROMPT 1: Analyze L1 memories to enrich Problem #1.

  Input:
  - ci_problem: Problem #1 extracted from CI failure
  - l1_memories: Top L1 memories (similar file-level failures)

  LLM Task:
  - Analyze L1 to find similar failures
  - Extract dependent files from L1
  - Get fix strategies from L1
  - Enrich Problem #1 with this information

  Output: Enriched Problem #1
  """

  if not l1_memories:
    return ci_problem

  # Build prompt
  prompt = f"""You are analyzing L1 file-level failure memories to enrich the current CI failure.

**CURRENT CI FAILURE (Problem #1):**
```json
{json.dumps(ci_problem, indent=2)}
```

**L1 MEMORIES (Top-10 similar past failures):**
```json
{json.dumps(l1_memories[:10], indent=2)}
```

**TASK:**
Analyze the L1 memories and enrich Problem #1:

1. **Check for similar failures in L1:**
   - Same validation_cmd?
   - Same error types (failure_type, issue_type)?
   - What changes were made (how_fixed field)?

2. **Extract dependent files from L1 enables/enabled_by:**
   - Look at L1 `enables` field - lists problem IDs this problem enables
   - Look at L1 `enabled_by` field - lists problem IDs that enable this problem
   - For each enabled problem ID, find corresponding L1 entries
   - Extract their affected files (file field)
   - Add ALL dependent files to the enriched problem

3. **Get fix strategy from L1:**
   - Use `root_cause` field - explains why it failed
   - Use `how_fixed` field - what specific changes were made
   - Use `why_fix_works` field - why the fix resolves the issue
   - Combine these into comprehensive fix_strategy

4. **Create enriched Problem #1:**
   Return a JSON object with these fields:
   - problem_id: 1
   - is_primary: true
   - status: "confirmed"
   - source: "CI+L1"
   - description: Problem statement (use L1 `problem` field + CI error)
   - root_cause: Why it fails (use L1 `root_cause` field)
   - files: [ALL files from CI + dependent files from L1 enables chain]
   - fix_strategy: How to fix (combine L1 `how_fixed` + `why_fix_works`)
   - validation_cmd: Command to verify fix (from L1 or CI)
   - check_first: false

**IMPORTANT:**
- Keep ALL CI error info
- EXPAND files by following L1 enables/enabled_by dependency chains
- Use ACTUAL L1 fields: problem, root_cause, how_fixed, why_fix_works
- Make description specific using L1 patterns
- NO hallucination - only use data from L1 memories

Return ONLY the JSON object, no markdown fences.
"""

  try:
    response = _call_llm(llm, prompt)

    # Parse JSON
    import re
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
      enriched = json.loads(match.group())
      logger.info(f"[LLM L1 Analysis] Enriched Problem #1: {len(enriched.get('files', []))} files, fix_strategy: {bool(enriched.get('fix_strategy'))}")
      return enriched
    else:
      logger.warning("[LLM L1 Analysis] Failed to parse JSON, using original CI problem")
      return ci_problem

  except Exception as e:
    logger.warning(f"[LLM L1 Analysis] Failed: {e}, using original CI problem")
    return ci_problem


def llm_analyze_l2_for_consecutive(
  problem_1: Dict[str, Any],
  l2_memories: List[Dict[str, Any]],
  l1_memories: List[Dict[str, Any]],
  validation_sequence: List[Dict[str, Any]],
  llm: Any,
  l3_memories: List[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
  """
  LLM PROMPT 2: Analyze L2/L1/L3 to select consecutive problems.

  ABLATION-AWARE: Only uses enabled levels based on what's passed.
  Priority: L2 > L1 > L3 (L2 checked first for sequences)

  Input:
  - problem_1: The confirmed CI failure (enriched)
  - l2_memories: Repo-level repair sequences (priority 1)
  - l1_memories: File-level problems (priority 2)
  - l3_memories: Universal patterns (priority 3, optional)
  - validation_sequence: What validations run next

  LLM Task:
  - Find sequences where similar problem appeared (L2 first)
  - Extract what problems came AFTER
  - Use L1 for dependent file problems
  - Use L3 for universal patterns if no L2/L1
  - SELECT relevant consecutive problems
  - Return array of atomic problems

  Output: [Problem #2, Problem #3, ...]
  """

  l3_memories = l3_memories or []

  if not (l2_memories or l1_memories or l3_memories):
    return []

  # Build prompt with enabled levels
  enabled_levels = []
  if l2_memories: enabled_levels.append("L2")
  if l1_memories: enabled_levels.append("L1")
  if l3_memories: enabled_levels.append("L3")

  prompt = f"""You are analyzing memory to predict what problems will appear AFTER fixing Problem #1.

**ENABLED MEMORY LEVELS**: {', '.join(enabled_levels) if enabled_levels else 'None'}
**PRIORITY ORDER**: L2 (repo sequences) > L1 (file problems) > L3 (universal patterns)

**PROBLEM #1 (Current CI Failure):**
```json
{json.dumps(problem_1, indent=2)}
```

**VALIDATION SEQUENCE (What runs next):**
```json
{json.dumps(validation_sequence or [], indent=2)}
```"""

  if l2_memories:
    prompt += f"""

**L2 MEMORIES (Top-10 repo-level repair sequences - PRIORITY 1):**
```json
{json.dumps(l2_memories[:10], indent=2)}
```"""

  if l1_memories:
    prompt += f"""

**L1 MEMORIES (Top-10 file-level problems - PRIORITY 2):**
```json
{json.dumps(l1_memories[:10], indent=2)}
```"""

  if l3_memories:
    prompt += f"""

**L3 MEMORIES (Top-10 universal patterns - PRIORITY 3):**
```json
{json.dumps(l3_memories[:10], indent=2)}
```"""

  prompt += """

**YOUR TASK:**
Find problems that will likely appear AFTER fixing Problem #1.

**STEP-BY-STEP INSTRUCTIONS:**

1. **Look at L2 repair_trajectory** (if provided):
   - Find the `repair_trajectory` array in L2 memories
   - Each trajectory has steps: [step 1, step 2, step 3, ...]
   - Step 1 = primary problem (already being fixed)
   - Steps 2, 3, 4... = HIDDEN problems that appeared after fixing step 1
   - For each hidden step, look at the `problems` array
   - Extract: problem, root_cause, how_fixed, why_fix_works, validation_cmd

2. **Look at L1 enables/enabled_by** (if provided):
   - Check L1 memories for `enables` and `enabled_by` fields
   - These link related problems
   - Find other L1 entries that appear in these dependency chains
   - Extract their problem details

3. **Look at L3 patterns** (if provided):
   - Find L3 memories with similar failure_type
   - Check their typical problem sequences
   - Extract common follow-up problems

4. **Create output**:
   Return a JSON array with this EXACT structure (no markdown, no ```):

[
  {
    "problem_id": 2,
    "is_primary": false,
    "status": "predicted",
    "source": "L2",
    "description": "Exact problem text from L2/L1/L3",
    "root_cause": "Exact root_cause from L2/L1/L3",
    "files": ["file1.py", "file2.py"],
    "fix_strategy": "Exact how_fixed + why_fix_works from L2/L1/L3",
    "validation_cmd": "exact validation command",
    "check_first": true
  }
]

**CRITICAL RULES:**
- Copy exact text from L2 trajectory problems, L1 entries, or L3 patterns
- NO hallucination - only use data from memories above
- Return ONLY the JSON array - no extra text
- If no consecutive problems found, return empty array: []
"""

  try:
    response = _call_llm(llm, prompt)

    # Log raw response for debugging
    logger.debug(f"[LLM L2 Analysis] Raw response length: {len(response)} chars")
    logger.debug(f"[LLM L2 Analysis] Response preview: {response[:500]}")

    # Parse JSON array
    import re
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
      json_str = match.group()
      logger.debug(f"[LLM L2 Analysis] Matched JSON length: {len(json_str)} chars")
      consecutive = json.loads(json_str)
      if isinstance(consecutive, list):
        logger.info(f"[LLM L2 Analysis] Found {len(consecutive)} consecutive problems")
        return consecutive[:10]  # Top 10 consecutive problems
      else:
        logger.warning("[LLM L2 Analysis] Response not an array")
        return []
    else:
      logger.warning(f"[LLM L2 Analysis] Failed to parse JSON array. Response: {response[:1000]}")
      return []

  except json.JSONDecodeError as e:
    logger.warning(f"[LLM L2 Analysis] JSON decode error: {e}. Response: {response[:1000]}")
    return []
  except Exception as e:
    logger.warning(f"[LLM L2 Analysis] Failed: {e}")
    return []
