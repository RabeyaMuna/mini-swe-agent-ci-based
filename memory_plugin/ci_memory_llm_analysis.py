"""
LLM-based Analysis for Memory-Guided Repair

Clean version: Selects relevant L1 memories based on CURRENT CI problems.
Matching criteria: validation_cmd + failure_type + (file OR problem)
"""

import os
import re
import json
import logging
import signal
from typing import Any, Dict, List
from memory_plugin.ci_memory_l2_analysis import staged_l2_analysis

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when an LLM call exceeds timeout."""

    pass


def _timeout_handler(signum, frame):
    """Signal handler for LLM call timeout."""
    raise TimeoutError("LLM call timed out")


def _json_array_from_text(text: str) -> List[Any]:
    if not text:
        return []
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        pass
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        return []
    try:
        parsed = json.loads(match.group())
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _json_object_from_text(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group())
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _parse_l1_selection_ids(response: str) -> List[str]:
    obj = _json_object_from_text(response)
    if obj:
        ids = obj.get("selected_ids", [])
        if isinstance(ids, list):
            return [str(item) for item in ids if isinstance(item, str)]

    arr = _json_array_from_text(response)
    return [str(item) for item in arr if isinstance(item, str)]


def _call_llm(llm: Any, prompt: str, timeout: int = 120, max_retries: int = 2) -> str:
    """
    Universal LLM caller with timeout and retry logic.

    Args:
        llm: LLM client (LangChain, callable, or OpenRouter client)
        prompt: Prompt text
        timeout: Timeout in seconds (default: 120s = 2 minutes)
        max_retries: Maximum number of retries on failure (default: 2)

    Returns:
        str: LLM response, or empty string on failure
    """
    if llm is None:
        return ""

    for attempt in range(max_retries + 1):
        try:
            # Set alarm signal for timeout (Unix/Mac only)
            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)

            try:
                # Try LangChain interface
                try:
                    from langchain_core.messages import HumanMessage

                    result = llm.invoke([HumanMessage(content=prompt)])
                    response = (getattr(result, "content", None) or "").strip()
                    if response:
                        return response
                except (ImportError, AttributeError):
                    pass

                # Try generic invoke
                result = llm.invoke(prompt)
                if hasattr(result, "content"):
                    response = (result.content or "").strip()
                    if response:
                        return response

                # Try direct call
                response = str(result).strip()
                if response:
                    return response

            finally:
                # Cancel alarm
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

            # Fallback: direct callable
            response = str(llm(prompt)).strip()
            if response:
                return response

        except TimeoutError:
            logger.warning(
                f"[_call_llm] Timeout on attempt {attempt + 1}/{max_retries + 1}"
            )
            if attempt < max_retries:
                continue
            return ""

        except Exception as e:
            logger.warning(
                f"[_call_llm] Error on attempt {attempt + 1}/{max_retries + 1}: {e}"
            )
            if attempt < max_retries:
                continue
            return ""

    return ""


def _llm_select_relevant_l1(
    ci_problem: Dict[str, Any],
    l1_memories: List[Dict[str, Any]],
    llm: Any,
    chunk_size: int | None = None,
    model_name: str | None = None,
) -> List[str]:
    """
    Select L1s that match CI problem by processing in chunks.

    Matching: If ANY of (file, problem, root_cause) matches -> SELECT
    Ignores: L1s with empty problem AND root_cause

    Args:
      ci_problem: Current CI failure
      l1_memories: All L1 memories
      llm: LLM instance
      chunk_size: L1s per chunk (None = auto-detect from model)
      model_name: Model name for auto-detecting chunk size

    Returns: List of matching L1 IDs
    """

    # Auto-detect chunk size based on model if not specified
    if chunk_size is None:
        try:
            from utilities.model_token_config import get_l1_chunk_size

            # Try to get model name from environment or llm object
            if model_name is None:
                model_name = os.environ.get("MEMCI_LLM_MODEL", "")
            chunk_size = get_l1_chunk_size(model_name)
            logger.info(
                f"[L1 Chunk] Auto-detected chunk_size={chunk_size} for model={model_name}"
            )
        except Exception as e:
            logger.warning(
                f"[L1 Chunk] Could not auto-detect chunk size: {e}, using default=5"
            )
            chunk_size = 5

    if not l1_memories:
        return []

    # Filter out L1s with empty problem AND root_cause
    valid_l1s = []
    for mem in l1_memories:
        problem = mem.get("problem", "").strip()
        root_cause = mem.get("root_cause", "").strip()

        # Skip if BOTH are empty
        if not problem and not root_cause:
            logger.debug(
                f"[L1 Filter] Skipping {mem.get('id')} - empty problem/root_cause"
            )
            continue

        valid_l1s.append(mem)

    if not valid_l1s:
        logger.warning("[L1 Select] All L1s have empty problem/root_cause")
        return []

    logger.info(f"[L1 Select] Valid L1s: {len(valid_l1s)}/{len(l1_memories)}")

    # Collect all selected IDs
    all_selected_ids = []

    # Process L1s in chunks
    total_chunks = (len(valid_l1s) + chunk_size - 1) // chunk_size

    for chunk_idx, i in enumerate(range(0, len(valid_l1s), chunk_size), 1):
        chunk = valid_l1s[i : i + chunk_size]

        logger.info(
            f"[L1 Chunk {chunk_idx}/{total_chunks}] Processing {len(chunk)} L1s..."
        )

        # Build chunk summary
        chunk_summary = []
        for l1 in chunk:
            chunk_summary.append(
                {
                    "id": l1.get("id"),
                    "file": l1.get("file", ""),
                    "problem": l1.get("problem", ""),
                    "root_cause": l1.get("root_cause", ""),
                    "how_fixed": l1.get("how_fixed", ""),
                    "why_fix_works": l1.get("why_fix_works", ""),
                    "validation_cmd": l1.get("validation_cmd", ""),
                    "failure_type": l1.get("failure_type", ""),
                }
            )

        prompt = f"""You are selecting relevant L1 file-level repair memories for a Python CI failure.

L1 memories are previous concrete repair experiences. The current CI failure may contain MULTIPLE distinct sub-problems in one validation stage. Select an L1 memory if it is relevant to ANY one current sub-problem.

CURRENT CI FAILURE:
{json.dumps(ci_problem, indent=2)}

CANDIDATE L1 MEMORIES ({len(chunk)}):
```json
{json.dumps(chunk_summary, indent=2)}
```

TASK:
Return the IDs of L1 memories that could help diagnose or fix at least one current CI sub-problem.

HOW TO READ THE CURRENT FAILURE:
- Treat each entry in files/error_details as a separate sub-problem.
- Also use description, root_cause, error_type, validation_cmd, and fix_strategy as evidence.
- A candidate does not need to match every CI sub-problem. Matching one meaningful sub-problem is enough.

SELECT a memory when it matches ANY current sub-problem by one or more of these:
1. Same failure mode: same exception, validator complaint, failed assertion, type-checking error, lint rule, formatting rule, dependency/build error, or test failure.
2. Same root cause: same technical reason such as empty collection access, private/internal API usage, incompatible dependency/config, missing import/package, invalid annotation, schema mismatch, or formatter rule violation.
3. Same repair strategy: previous fix would plausibly apply, such as adding bounds checks, replacing private types, removing obsolete plugins, pinning/upgrading dependencies, fixing config keys, reformatting files, changing imports, or updating tests.
4. Same tool/API/technology: same validator, library, API symbol, config format, framework component, or protocol boundary involved in the sub-problem.
5. Same or dependent file: exact file, same module/package area, test/source pair, config controlling the failing tool, or file commonly changed together.

SKIP a memory when:
- It only shares the repository/language/broad category but not a concrete sub-problem, root cause, or repair strategy.
- It only shares a file with an unrelated issue.
- It is a common repo issue but not linked to the current failure evidence.
- Its fix would not help diagnose or repair any current sub-problem.

IMPORTANT:
- Select all L1s relevant to any current sub-problem.
- Problem/root-cause/fix similarity is more important than exact file matching.
- Include alternative solutions for the same failure family.
- Avoid noisy weak matches.

OUTPUT:
Return ONLY a valid JSON array of selected L1 IDs, ordered from most relevant to least relevant.
Example: ["L1-0012", "L1-0088"]
"""

        try:
            # Use shorter timeout for selection (60s per attempt, 1 retry = max 2 minutes)
            response = _call_llm(llm, prompt, timeout=60, max_retries=1)

            if not response or not response.strip():
                logger.warning(
                    f"  Chunk {chunk_idx}: LLM returned empty response, skipping"
                )
                continue

            chunk_ids = _parse_l1_selection_ids(response)
            all_selected_ids.extend(chunk_ids)
            logger.info(f"  Selected: {len(chunk_ids)} L1s from this chunk")

        except Exception as e:
            logger.warning(f"  Error in chunk {chunk_idx}: {e}")
            continue

    # Deduplicate
    all_selected_ids = list(dict.fromkeys(all_selected_ids))

    logger.info(
        f"[L1 Total] Selected {len(all_selected_ids)} unique L1s from {len(valid_l1s)} valid"
    )
    for l1_id in all_selected_ids:
        logger.info(f"  - {l1_id}")

    return all_selected_ids


def _expand_l1_dependencies(
    relevant_ids: List[str], all_l1_memories: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    STAGE 2: Expand dependencies (CODE-based, no LLM).

    Follows enables/enabled_by chains to get related L1s.
    """

    if not relevant_ids:
        return []

    # Build ID -> memory map
    id_to_mem = {m.get("id", ""): m for m in all_l1_memories}

    result = []
    seen_ids = set()

    # Add initially selected L1s
    for l1_id in relevant_ids:
        if l1_id in id_to_mem and l1_id not in seen_ids:
            selected = dict(id_to_mem[l1_id])
            selected["_memory_role"] = "selected"
            selected["_dependency_source"] = ""
            result.append(selected)
            seen_ids.add(l1_id)

    # Expand dependencies recursively
    to_expand = list(result)
    while to_expand:
        current = to_expand.pop(0)

        enables = current.get("enables", []) or []
        enabled_by = current.get("enabled_by", []) or []

        for dep_id in enables + enabled_by:
            if dep_id in id_to_mem and dep_id not in seen_ids:
                dep = dict(id_to_mem[dep_id])
                dep["_memory_role"] = "dependency"
                dep["_dependency_source"] = str(current.get("id", ""))
                result.append(dep)
                seen_ids.add(dep_id)
                to_expand.append(dep)

    logger.info(
        f"[L1 Expand] {len(relevant_ids)} selected -> {len(result)} total (with dependencies)"
    )
    return result


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _are_problems_related(prob1: Dict[str, Any], prob2: Dict[str, Any]) -> bool:
    """
    Check if two problems are related.

    Related = share same error/technology AND different fix types
    Example: config change + code change for same issue
    """

    # Extract key terms
    p1_text = " ".join(
        [
            prob1.get("problem", ""),
            prob1.get("root_cause", ""),
            prob1.get("issue_type", ""),
        ]
    ).lower()

    p2_text = " ".join(
        [
            prob2.get("problem", ""),
            prob2.get("root_cause", ""),
            prob2.get("issue_type", ""),
        ]
    ).lower()

    # Check for shared technology/error keywords
    shared_keywords = []
    keywords = [
        "numpy",
        "mypy",
        "dtype",
        "dtypelike",
        "plugin",
        "type",
        "import",
        "pytest",
        "test",
        "mock",
        "index",
        "error",
        "validation",
    ]

    for keyword in keywords:
        if keyword in p1_text and keyword in p2_text:
            shared_keywords.append(keyword)

    # Need at least 2 shared keywords
    if len(shared_keywords) < 2:
        return False

    # Check if different fix types (config vs code)
    p1_files = prob1.get("files", [])
    p2_files = prob2.get("files", [])

    p1_is_config = any(
        "pyproject.toml" in str(f) or "setup." in str(f) or ".cfg" in str(f)
        for f in p1_files
    )
    p2_is_config = any(
        "pyproject.toml" in str(f) or "setup." in str(f) or ".cfg" in str(f)
        for f in p2_files
    )

    # Related if: same technology + different fix types (config vs code)
    if p1_is_config != p2_is_config and len(shared_keywords) >= 2:
        return True

    # Also related if: same validation_cmd + same failure_type + different files
    if (
        prob1.get("validation_cmd") == prob2.get("validation_cmd")
        and prob1.get("failure_type") == prob2.get("failure_type")
        and p1_files != p2_files
        and len(shared_keywords) >= 2
    ):
        return True

    return False


def _combine_related_problems(problems: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine related problems into one merged problem.
    """

    if not problems:
        return {}

    # Start with first problem as base
    merged = problems[0].copy()

    # Merge IDs
    all_ids = []
    for p in problems:
        all_ids.extend(p.get("ids", []))
    merged["ids"] = list(set(all_ids))

    # Merge files
    all_files = []
    for p in problems:
        all_files.extend(p.get("files", []))
    merged["files"] = list(set(all_files))

    # Merge dependency sources
    all_deps = []
    for p in problems:
        all_deps.extend(p.get("dependency_sources", []))
    merged["dependency_sources"] = list(set(all_deps))

    # Combine problem descriptions
    unique_problems = list(
        set(p.get("problem", "") for p in problems if p.get("problem"))
    )
    if len(unique_problems) > 1:
        merged["problem"] = " AND ".join(unique_problems)

    # Combine fixes
    unique_fixes = list(
        set(p.get("how_fixed", "") for p in problems if p.get("how_fixed"))
    )
    if len(unique_fixes) > 1:
        merged["how_fixed"] = " PLUS ".join(unique_fixes)

    # Mark as merged
    merged["_merged_from"] = len(problems)

    return merged


def llm_analyze_l1_for_problem_1(
    ci_problem: Dict[str, Any], l1_memories: List[Dict[str, Any]], llm: Any
) -> List[Dict[str, Any]]:
    """
    3-STAGE L1 ANALYSIS to enrich Problem #1.

    Stage 1: LLM selects matching L1s (strict criteria)
    Stage 2: CODE expands dependencies
    Stage 3: LLM enriches CI problem with L1 details

    Returns: List of extracted problems (can be multiple if CI has multiple sub-problems)
    """

    if not l1_memories:
        logger.info("[L1 Analysis] No L1 memories provided")
        return [ci_problem]  # Return as list

    # Stage 1: Select matching L1s
    logger.info("[L1 Stage 1] Selecting L1s matching CI problem...")
    relevant_ids = _llm_select_relevant_l1(ci_problem, l1_memories, llm)

    if not relevant_ids:
        logger.info("[L1 Analysis] No matching L1s found, using original CI problem")
        return [ci_problem]  # Return as list

    # Stage 2: Expand dependencies
    logger.info("[L1 Stage 2] Expanding dependencies...")
    all_relevant = _expand_l1_dependencies(relevant_ids, l1_memories)

    if not all_relevant:
        logger.info("[L1 Analysis] No L1s after expansion, using original CI problem")
        return [ci_problem]  # Return as list

    # Don't compact - pass distinct L1s to LLM
    # LLM will extract multiple distinct problems
    compact_relevant = all_relevant

    # Stage 3: Enrich CI problem
    logger.info(
        "[L1 Stage 3] Enriching CI problem with %d L1 memories",
        len(compact_relevant),
    )

    # Extract CI problem details for clarity
    ci_files = ci_problem.get("files", [])
    ci_file_list = []
    for f in ci_files:
        if isinstance(f, dict):
            ci_file_list.append(f.get("path", ""))
        elif isinstance(f, str):
            ci_file_list.append(f)

    ci_description = ci_problem.get("description", "")
    ci_root_cause = ci_problem.get("root_cause", "")
    ci_error_details = ci_problem.get("error_details", [])

    prompt = f"""You are extracting structured repair problems from a Python CI failure using relevant L1 repair memories.

  CURRENT CI FAILURE - PRIMARY EVIDENCE:

  Files directly mentioned by CI:
  ```json
  {json.dumps(ci_file_list, indent=2)}

  Error details directly reported by CI:

  {json.dumps(ci_error_details, indent=2)}

  CI description:
  {ci_description}

  CI root cause:
  {ci_root_cause}

  RELEVANT L1 MEMORY GROUPS - PRIOR FIX EVIDENCE:

  {json.dumps(compact_relevant, indent=2)}

  TASK:
  Return a JSON array of actionable repair problems.

  Create one problem per distinct repair unit:

  - one directly failing file,
  - or one tightly related file group that shares the same root cause and same fix pattern.

  CLASSIFICATION:
  Set relevant_to_primary based only on whether the problem is directly supported by the current CI evidence.

  - relevant_to_primary = true:
    The file/problem is directly mentioned in ci_file_list, ci_error_details, CI description, or CI root cause.

  - relevant_to_primary = false:
    The file/problem comes from L1 memory only, such as a related config fix, dependency fix, alternative repair, or follow-up
    repair connected to the same failure family.

  Do not drop relevant L1-only problems. Keep them with relevant_to_primary=false when they are technically connected to the
  current CI failure.

  RULES:

  1. Include all primary CI problems.
  2. Include connected L1 problems when they add a non-redundant root cause, fix, config change, dependency change, or
     alternative repair path.

  3. Do not include unrelated L1 memories that only share the repo, broad category, or language.
  4. Merge redundant L1 memories into one problem when they have the same root cause and same fix pattern.
  5. Split problems when the fix strategy differs, even if the symptom is related.
  6. Mention concrete files, directories, or file patterns in the problem text.
  7. Root cause must explain why the old state fails technically.
  8. how_fixed must say exactly what to change.
  9. why_fix_works must explain why the change satisfies the failing validation.
  10. Preserve important details from CI and L1, but remove repeated duplicate wording.

  OUTPUT FORMAT - CRITICAL:

  You MUST return a JSON ARRAY (starts with [ and ends with ]).
  Even if there is only ONE problem, wrap it in an array: [{...}]

  CORRECT:
  [
    {{
      "problem_id": 1,
      "relevant_to_primary": true,
      "file": "exact/file/path.py",
      "problem": "Specific error with file context",
      "root_cause": "Technical reason",
      "how_fixed": "Exact changes to make",
      "why_fix_works": "Why this fixes it",
      "validation_cmd": "./framework/dev/test.sh"
    }},
    // more problems...
  ]

  Return ONLY the JSON array. No markdown fences. No explanatory text.
  Start with [ and end with ].
  """

    try:
        # Call LLM with timeout (2 minutes per attempt, 2 retries = max 6 minutes total)
        response = _call_llm(llm, prompt, timeout=120, max_retries=2)

        # EARLY FALLBACK: If LLM returned empty response after all retries
        if not response or not response.strip():
            logger.warning("[L1 Enrich] LLM returned empty response after all retries")
            logger.warning(
                "[L1 Enrich] Falling back to original CI problem without enrichment"
            )
            return [ci_problem]  # Return as array

        # Log response for debugging
        logger.debug(f"[L1 Enrich] Response length: {len(response)} chars")
        logger.debug(f"[L1 Enrich] Response preview: {response[:500]}")

        # Parse JSON array
        extracted_problems = _json_array_from_text(response)

        # FALLBACK: If LLM returned single object instead of array, wrap it
        if not extracted_problems:
            logger.warning(
                "[L1 Enrich] Failed to parse as array, trying as single object..."
            )
            single_obj = _json_object_from_text(response)

            if single_obj and isinstance(single_obj, dict):
                logger.info("[L1 Enrich] Got single object, wrapping in array")
                extracted_problems = [single_obj]
            else:
                logger.warning("[L1 Enrich] Failed to parse response completely")
                logger.warning(f"[L1 Enrich] Response: {response[:500]}")
                logger.warning("[L1 Enrich] Falling back to original CI problem")
                return [ci_problem]  # Return as array

        if extracted_problems and isinstance(extracted_problems, list):
            logger.info(
                f"[L1 Enrich] Success! Extracted {len(extracted_problems)} problems"
            )

            # Log each problem
            for idx, prob in enumerate(extracted_problems, 1):
                file_str = prob.get("file", "") or (
                    prob.get("files", [""])[0] if prob.get("files") else "unknown"
                )
                problem_str = prob.get("problem", "")[:80]
                logger.info(f"  Problem {idx}: {file_str} - {problem_str}")

            # Return problems array directly
            return extracted_problems

        else:
            logger.warning("[L1 Enrich] Unexpected type after parsing")
            logger.warning(f"[L1 Enrich] Response: {response[:500]}")
            return [ci_problem]  # Return as array

    except Exception as e:
        logger.warning(f"[L1 Enrich] Error: {e}")
        logger.warning(
            f"[L1 Enrich] Response: {response[:500] if 'response' in locals() else 'No response'}"
        )
        return ci_problem


def llm_analyze_l2_for_consecutive(
    problem_1: Dict[str, Any],
    l2_memories: List[Dict[str, Any]],
    l1_memories: List[Dict[str, Any]],
    validation_sequence: List[Dict[str, Any]],
    llm: Any,
    l3_memories: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Find consecutive problems using L2/L1/L3 memories.

    ONE clean implementation:
    - L2: Uses staged_l2_analysis (finds COMMON + CONSECUTIVE)
    - L1: CODE-based dependency extraction
    - L3: CODE-based pattern extraction

    Returns: List of consecutive problems (relevant_to_primary=false)
    """

    l3_memories = l3_memories or []
    consecutive = []

    # L2: Use staged analysis (COMMON + CONSECUTIVE)
    if l2_memories:
        try:
            from .ci_memory_l2_analysis import staged_l2_analysis

            logger.info(
                "[L2] Using staged L2 analysis (finds COMMON + CONSECUTIVE problems)"
            )
            l2_consecutive = staged_l2_analysis(problem_1, l2_memories, llm)
            consecutive.extend(l2_consecutive)
            logger.info(f"[L2] Found {len(l2_consecutive)} consecutive problems")
        except ImportError as e:
            logger.warning(f"[L2] staged_l2_analysis not available: {e}")
        except Exception as e:
            logger.warning(f"[L2] Error in staged_l2_analysis: {e}")

    # L1: CODE-based dependency extraction
    if l1_memories:
        logger.info("[L1] Extracting dependencies from L1 chains...")
        l1_consecutive = _extract_l1_dependencies(problem_1, l1_memories)
        consecutive.extend(l1_consecutive)
        logger.info(f"[L1] Found {len(l1_consecutive)} dependent problems")

    # L3: CODE-based pattern extraction
    if l3_memories:
        logger.info("[L3] Extracting patterns from L3...")
        l3_consecutive = _extract_l3_patterns(problem_1, l3_memories)
        consecutive.extend(l3_consecutive)
        logger.info(f"[L3] Found {len(l3_consecutive)} pattern-based problems")

    # Deduplicate
    consecutive = _deduplicate_consecutive(consecutive)

    logger.info(f"[Total] {len(consecutive)} consecutive problems after deduplication")

    return consecutive[:10]  # Top 10


def _extract_l1_dependencies(
    problem_1: Dict[str, Any], l1_memories: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract dependent problems from L1 enables/enabled_by chains.

    CODE-based extraction (deterministic, no LLM).
    """

    consecutive = []
    problem_1_file = problem_1.get("file", "")

    # Find L1s matching problem_1
    matching_l1s = []
    for l1 in l1_memories:
        if l1.get("file") == problem_1_file:
            matching_l1s.append(l1)

    # Extract enabled problems
    for l1 in matching_l1s:
        enables = l1.get("enables", [])

        for enabled_id in enables:
            # Find the enabled L1
            enabled_l1 = None
            for l1_candidate in l1_memories:
                if l1_candidate.get("id") == enabled_id:
                    enabled_l1 = l1_candidate
                    break

            if enabled_l1:
                consecutive.append(
                    {
                        "problem_id": len(consecutive) + 2,
                        "relevant_to_primary": False,
                        "source": "L1",
                        "dependency_type": "enables",
                        "file": enabled_l1.get("file", ""),
                        "problem": enabled_l1.get("problem", ""),
                        "root_cause": enabled_l1.get("root_cause", ""),
                        "how_fixed": enabled_l1.get("how_fixed", ""),
                        "why_fix_works": enabled_l1.get("why_fix_works", ""),
                        "validation_cmd": enabled_l1.get("validation_cmd", ""),
                    }
                )

    return consecutive


def _extract_l3_patterns(
    problem_1: Dict[str, Any], l3_memories: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract common patterns from L3.

    CODE-based extraction (deterministic, no LLM).
    """

    consecutive = []
    problem_1_failure_type = problem_1.get("failure_type", "")

    for l3 in l3_memories:
        # Match by failure type
        if l3.get("failure_type") != problem_1_failure_type:
            continue

        # Extract typical sequence steps (skip step 1, get 2+)
        typical_sequence = l3.get("typical_sequence", [])

        for seq_step in typical_sequence[1:]:  # Skip step 1
            consecutive.append(
                {
                    "problem_id": len(consecutive) + 2,
                    "relevant_to_primary": False,
                    "source": "L3",
                    "pattern_type": l3.get("pattern", ""),
                    "problem": seq_step.get("problem", ""),
                    "root_cause": seq_step.get("root_cause", ""),
                    "how_fixed": seq_step.get("how_fixed", ""),
                    "validation_cmd": problem_1.get("validation_cmd", ""),
                }
            )

    return consecutive


def _deduplicate_consecutive(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate consecutive problems.

    Deduplication by: (file, problem description)
    """

    seen = set()
    unique = []

    for prob in problems:
        # Create key from file + problem
        file = prob.get("file", "")
        problem = prob.get("problem", "")  # First 100 chars
        key = (file, problem)

        if key not in seen:
            seen.add(key)
            unique.append(prob)

    return unique


# =============================================================================
# UNIFIED PROBLEM EXTRACTION - Main Entry Point
# =============================================================================


def extract_all_problems_for_repair(
    ci_problem: Dict[str, Any],
    l1_memories: List[Dict[str, Any]],
    l2_memories: List[Dict[str, Any]],
    l3_memories: List[Dict[str, Any]],
    llm: Any,
    ablation_levels: str = "L1+L2+L3",
) -> List[Dict[str, Any]]:
    """
    Extract ALL problems in correct priority order for agent repair.

    This is the MAIN ENTRY POINT for problem extraction.

    Supports ablation study:
    - "L1": Only L1 enrichment (primary CI problems)
    - "L1+L2": L1 + L2 consecutive (common + consecutive)
    - "L1+L2+L3": L1 + L2 + L3 patterns (full pipeline)

    Returns: Single array ordered by priority:
    1. PRIMARY problems (relevant_to_primary: true) - CI failures
    2. COMMON problems (is_common: true) - High confidence
    3. CONSECUTIVE problems (from_step: 2, 3, 4...) - Linked to primary
    4. DEPENDENT problems (dependency_type: "enables") - L1 chains

    Agent receives this array and fixes problems one by one.

    Args:
      ci_problem: Current CI failure (from decomposition)
      l1_memories: File-level repair memories
      l2_memories: Repo-level repair trajectories
      l3_memories: Universal patterns
      llm: LLM instance
      ablation_levels: Which levels to use (for ablation study)

    Returns:
      List of problems ordered by priority, ready for agent
    """

    logger.info("=" * 80)
    logger.info(
        f"[ExtractAll] Starting unified problem extraction (levels: {ablation_levels})"
    )
    logger.info("=" * 80)

    all_problems = []

    # Step 1: Extract PRIMARY problems from CI (L1 enrichment)
    if "L1" in ablation_levels:
        logger.info("[ExtractAll] Step 1: L1 enrichment of primary CI problems")

        try:
            primary_problems = llm_analyze_l1_for_problem_1(
                ci_problem, l1_memories, llm
            )

            # Check if it's a list (new format) or dict (old format)
            if isinstance(primary_problems, list):
                # New format: array of individual problems
                all_problems.extend(primary_problems)
                logger.info(f" Added {len(primary_problems)} primary problems from L1")
            elif isinstance(primary_problems, dict):
                # Old format: single enriched problem - convert to list
                logger.info(" L1 returned single dict, converting to list")
                all_problems.append(primary_problems)
                logger.info("  Added 1 primary problem from L1")
            else:
                logger.warning(
                    f"  L1 enrichment returned unexpected type: {type(primary_problems)}"
                )

        except Exception as e:
            logger.error(f" L1 enrichment failed: {e}")

    # Step 2: Extract CONSECUTIVE from L2 (common + consecutive)
    if "L2" in ablation_levels and l2_memories:
        logger.info(
            "[ExtractAll] Step 2: L2 consecutive problems (COMMON + CONSECUTIVE)"
        )

        try:
            # Get first primary problem for matching
            problem_1 = all_problems[0] if all_problems else ci_problem

            l2_consecutive = staged_l2_analysis(problem_1, l2_memories, llm)

            all_problems.extend(l2_consecutive)
            logger.info(f"  Added {len(l2_consecutive)} L2 consecutive problems")

            # Log common vs unique
            common_count = sum(1 for p in l2_consecutive if p.get("is_common"))
            logger.info(
                f"     - {common_count} common problems (appear in multiple issues)"
            )
            logger.info(f"     - {len(l2_consecutive) - common_count} unique problems")

        except Exception as e:
            logger.error(f"  L2 analysis failed: {e}")

    # Step 3: Extract DEPENDENCIES from L1
    if "L1" in ablation_levels and l1_memories:
        logger.info("[ExtractAll] Step 3: L1 dependencies (enables chains)")

        try:
            problem_1 = all_problems[0] if all_problems else ci_problem

            l1_dependencies = _extract_l1_dependencies(problem_1, l1_memories)

            all_problems.extend(l1_dependencies)
            logger.info(f"  Added {len(l1_dependencies)} L1 dependency problems")

        except Exception as e:
            logger.error(f"  L1 dependency extraction failed: {e}")

    # Step 4: Extract PATTERNS from L3
    if "L3" in ablation_levels and l3_memories:
        logger.info("[ExtractAll] Step 4: L3 patterns (universal sequences)")

        try:
            problem_1 = all_problems[0] if all_problems else ci_problem

            l3_patterns = _extract_l3_patterns(problem_1, l3_memories)

            all_problems.extend(l3_patterns)
            logger.info(f"  Added {len(l3_patterns)} L3 pattern problems")

        except Exception as e:
            logger.error(f"  L3 pattern extraction failed: {e}")

    # Step 5: SORT by priority
    logger.info("[ExtractAll] Step 5: Sorting by priority")

    all_problems = _sort_problems_by_priority(all_problems)

    # Step 6: Add sequential problem_id
    for idx, prob in enumerate(all_problems, 1):
        prob["problem_id"] = idx

    # Final summary
    logger.info("=" * 80)
    logger.info(f"[ExtractAll] COMPLETE: {len(all_problems)} problems total")
    logger.info("=" * 80)

    primary_count = sum(1 for p in all_problems if p.get("relevant_to_primary"))
    common_count = sum(1 for p in all_problems if p.get("is_common"))
    consecutive_count = sum(1 for p in all_problems if p.get("from_step"))
    dependency_count = sum(1 for p in all_problems if p.get("dependency_type"))

    logger.info("  Breakdown:")
    logger.info(f"     - {primary_count} PRIMARY problems (CI failures)")
    logger.info(f"     - {common_count} COMMON problems (high confidence)")
    logger.info(f"     - {consecutive_count} CONSECUTIVE problems (linked to primary)")
    logger.info(f"     - {dependency_count} DEPENDENCY problems (L1 chains)")
    logger.info("=" * 80)

    return all_problems


def _sort_problems_by_priority(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort problems by priority.

    Priority order (lower = higher priority):
    1. relevant_to_primary: true (CI failures) - highest
    2. is_common: true (common problems) - high
       - More issues = higher priority
    3. from_step: 2, 3, 4... (consecutive) - medium
       - Earlier steps = higher priority
    4. Everything else - low

    Args:
      problems: Unsorted list of problems

    Returns:
      Sorted list (highest priority first)
    """

    def priority_score(prob):
        """
        Calculate priority score.

        Returns: (priority_tier, sub_priority)
        Lower values = higher priority
        """

        # Tier 1: Primary CI failures (highest)
        if prob.get("relevant_to_primary"):
            return (1, 0)

        # Tier 2: Common problems (high)
        # More common = higher priority (use negative to reverse sort)
        if prob.get("is_common"):
            num_issues = len(prob.get("appears_in_issues", []))
            return (2, -num_issues)

        # Tier 3: Consecutive problems (medium)
        # Earlier steps = higher priority
        if prob.get("from_step"):
            return (3, prob.get("from_step", 999))

        # Tier 4: Everything else (low)
        return (4, 0)

    return sorted(problems, key=priority_score)
