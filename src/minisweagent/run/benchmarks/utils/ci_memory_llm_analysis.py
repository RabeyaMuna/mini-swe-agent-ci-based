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

**L1 MEMORIES (Similar past failures):**
```json
{json.dumps(l1_memories[:5], indent=2)}
```

**TASK:**
Analyze the L1 memories and enrich Problem #1:

1. **Check for similar failures in L1:**
   - Same validation_cmd?
   - Same error types?
   - What changes were made?

2. **Extract dependent files:**
   - What other files needed fixing together?
   - Add them to the files list

3. **Get fix strategy from L1:**
   - How was this fixed before?
   - What specific changes?

4. **Create enriched Problem #1:**
   Return a JSON object with these fields:
   - problem_id: 1
   - is_primary: true
   - status: "confirmed"
   - source: "CI+L1"
   - description: Clear problem statement (use L1 to make it more specific)
   - root_cause: Why it fails (from L1 if available)
   - files: [all files including dependent from L1]
   - fix_strategy: How to fix (from L1 experience)
   - validation_cmd: Command to verify fix
   - check_first: false

**IMPORTANT:**
- Keep CI error info
- ADD dependent files from L1
- ADD fix strategy from L1
- Make description more specific using L1 patterns

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

**L2 MEMORIES (Repo-level repair sequences - PRIORITY 1):**
```json
{json.dumps(l2_memories[:5], indent=2)}
```"""

  if l1_memories:
    prompt += f"""

**L1 MEMORIES (File-level problems - PRIORITY 2):**
```json
{json.dumps(l1_memories[:5], indent=2)}
```"""

  if l3_memories:
    prompt += f"""

**L3 MEMORIES (Universal patterns - PRIORITY 3):**
```json
{json.dumps(l3_memories[:5], indent=2)}
```"""

  prompt += """

**TASK:**
Predict what problems will appear AFTER fixing Problem #1 using ONLY the enabled memory levels above:

1. **Check L2 repair sequences:**
   - Find sequences where Problem #1's pattern appeared
   - What problems came AFTER?
   - In what order?

2. **Check validation sequence:**
   - What validations run after Problem #1's validation?
   - What errors commonly appear in those stages?

3. **SELECT relevant consecutive problems:**
   - Filter by relevance to this repo
   - Only include problems likely to occur
   - Max 3-5 consecutive problems

4. **Create atomic problems:**
   For each consecutive problem, return:
   - problem_id: (2, 3, 4...)
   - is_primary: false
   - status: "predicted"
   - source: "L2" or "L1"
   - description: Clear statement of what will fail
   - root_cause: Why it will fail
   - files: [predicted files that will fail]
   - fix_strategy: How to fix (from L2/L1)
   - validation_cmd: Command that will fail
   - check_first: true

**IMPORTANT:**
- Return array of problems: [Problem #2, Problem #3, ...]
- Each problem is ATOMIC (one validation, one fix)
- Based on actual L2/L1 patterns
- Only relevant to this repo

Return JSON array, no markdown fences:
```json
[
  {{problem_id: 2, ...}},
  {{problem_id: 3, ...}}
]
```
"""

  try:
    response = _call_llm(llm, prompt)

    # Parse JSON array
    import re
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
      consecutive = json.loads(match.group())
      if isinstance(consecutive, list):
        logger.info(f"[LLM L2 Analysis] Found {len(consecutive)} consecutive problems")
        return consecutive[:5]  # Max 5
      else:
        logger.warning("[LLM L2 Analysis] Response not an array")
        return []
    else:
      logger.warning("[LLM L2 Analysis] Failed to parse JSON array")
      return []

  except Exception as e:
    logger.warning(f"[LLM L2 Analysis] Failed: {e}")
    return []
