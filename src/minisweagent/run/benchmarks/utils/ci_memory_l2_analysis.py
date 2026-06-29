"""
L2 memory analysis for CI repair.

Code prepares compact trajectory evidence. The LLM decides which repeated
problems are common and which later problems are consecutive for the current CI
failure.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _call_llm(llm: Any, prompt: str) -> str:
    """Call either a LangChain-style object or a plain callable."""
    if llm is None:
        return ""
    try:
        try:
            from langchain_core.messages import HumanMessage

            result = llm.invoke([HumanMessage(content=prompt)])
            return (getattr(result, "content", None) or "").strip()
        except (AttributeError, ImportError):
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


def _parse_json_array(content: str) -> List[Dict[str, Any]]:
    """Parse a JSON array from an LLM response."""
    if not content:
        return []
    content = content.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()
    content = re.sub(r",(\s*[}\]])", r"\1", content)
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_text(value: Any, limit: Optional[int] = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit and len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _dedupe_keep_order(values: Iterable[Any], limit: Optional[int] = None) -> List[Any]:
    seen = set()
    result = []
    for value in values:
        if value in (None, ""):
            continue
        key = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if limit and len(result) >= limit:
            break
    return result


def _step_problems(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Support both current L2 schema variants."""
    problems = step.get("problems")
    if isinstance(problems, list) and problems:
        return [p for p in problems if isinstance(p, dict)]

    # Some L2 records store one problem directly at step level.
    if any(step.get(key) for key in ("problem", "root_cause", "how_fixed", "why_fix_works", "issue_type")):
        return [step]
    return []


def _flatten_l2(l2_memories: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Flatten L2 repair trajectories into problem rows."""
    rows: List[Dict[str, Any]] = []
    trajectory_ids = set()
    row_id = 1

    for memory_index, memory in enumerate(l2_memories, 1):
        issue_id = str(memory.get("issue_id") or memory.get("id") or f"memory_{memory_index}")
        trajectory_ids.add(issue_id)
        trajectory = memory.get("repair_trajectory") or memory.get("trajectory") or []
        if not isinstance(trajectory, list):
            continue

        for step_index, step in enumerate(trajectory, 1):
            if not isinstance(step, dict):
                continue
            step_number = step.get("step") or step_index
            is_primary = step_index == 1 or str(step.get("type", "")).lower() in {
                "primary",
                "primary_failure",
                "foundational",
            }
            validation_cmd = _clean_text(step.get("validation_cmd") or memory.get("validation_cmd"))
            failure_type = _clean_text(step.get("failure_type") or memory.get("failure_type"))

            for problem in _step_problems(step):
                problem_text = _clean_text(problem.get("problem") or problem.get("description"), 700)
                root_cause = _clean_text(problem.get("root_cause"), 700)
                how_fixed = _clean_text(problem.get("how_fixed") or problem.get("fix"), 700)
                why_fix_works = _clean_text(problem.get("why_fix_works"), 500)
                if not any((problem_text, root_cause, how_fixed, why_fix_works)):
                    continue

                rows.append(
                    {
                        "row_id": row_id,
                        "issue_id": issue_id,
                        "repo": memory.get("repo", ""),
                        "step": step_number,
                        "step_index": step_index,
                        "is_primary_step": is_primary,
                        "step_type": step.get("type", ""),
                        "depends_on": step.get("depends_on"),
                        "validation_cmd": validation_cmd,
                        "failure_type": failure_type,
                        "issue_type": _clean_text(problem.get("issue_type") or step.get("issue_type"), 160),
                        "problem": problem_text,
                        "root_cause": root_cause,
                        "how_fixed": how_fixed,
                        "why_fix_works": why_fix_works,
                        "files": _dedupe_keep_order(_as_list(problem.get("files") or problem.get("affected_files")), 20),
                    }
                )
                row_id += 1

    return rows, max(len(trajectory_ids), len(l2_memories))


def _candidate_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    problem_hint = _clean_text(row.get("problem", ""), 180).lower()
    fix_hint = _clean_text(row.get("how_fixed", ""), 180).lower()
    return (
        row.get("validation_cmd", "").lower(),
        row.get("failure_type", "").lower(),
        row.get("issue_type", "").lower(),
        f"{problem_hint}|{fix_hint}",
    )


def _group_candidate_rows(
    rows: List[Dict[str, Any]],
    *,
    total_trajectories: int,
    prefix: str,
    include_primary: bool,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not include_primary and row.get("is_primary_step"):
            continue
        grouped[_candidate_key(row)].append(row)

    candidates = []
    for index, group in enumerate(grouped.values(), 1):
        representative = group[0]
        issue_ids = _dedupe_keep_order(row.get("issue_id") for row in group)
        files = _dedupe_keep_order(
            file_path
            for row in group
            for file_path in _as_list(row.get("files"))
            if file_path
        )
        candidate = {
            "candidate_id": f"{prefix}{index:03d}",
            "validation_cmd": representative.get("validation_cmd", ""),
            "failure_type": representative.get("failure_type", ""),
            "issue_type": representative.get("issue_type", ""),
            "problem": representative.get("problem", ""),
            "root_cause": representative.get("root_cause", ""),
            "how_fixed": representative.get("how_fixed", ""),
            "why_fix_works": representative.get("why_fix_works", ""),
            "files": files[:15],
            "appears_in_issues": issue_ids,
            "frequency": len(issue_ids),
            "frequency_ratio": round(len(issue_ids) / max(total_trajectories, 1), 2),
            "steps": _dedupe_keep_order((row.get("step") for row in group), 10),
            "row_ids": [row.get("row_id") for row in group[:20]],
            "evidence_count": len(group),
        }
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -int(item.get("frequency", 0)),
            -int(item.get("evidence_count", 0)),
            str(item.get("validation_cmd", "")),
        )
    )
    return candidates


def _compact_problem(problem_1: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "validation_cmd": problem_1.get("validation_cmd", ""),
        "failure_type": problem_1.get("failure_type") or problem_1.get("error_type", ""),
        "description": _clean_text(problem_1.get("description") or problem_1.get("problem"), 900),
        "root_cause": _clean_text(problem_1.get("root_cause"), 1200),
        "files": [
            item.get("path", item) if isinstance(item, dict) else item
            for item in _as_list(problem_1.get("files"))
        ],
        "error_details": _as_list(problem_1.get("error_details"))[:8],
    }


def _limit_candidates(candidates: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Keep high-signal candidates while leaving semantic choice to the LLM."""
    return candidates[:limit]


def _llm_select_common_problems(
    common_candidates: List[Dict[str, Any]],
    llm: Any,
) -> List[Dict[str, Any]]:
    """
    Select COMMON problems: problems that appear repeatedly across multiple trajectories.

    Criteria: Repetition-based (appears in 2+ trajectories)
    """
    if not common_candidates:
        return []

    prompt = f"""Select COMMON problems from L2 trajectories.

COMMON CANDIDATES (grouped by similarity):
{json.dumps(_limit_candidates(common_candidates, 30), indent=2)}

TASK:
Select problems that appear REPEATEDLY across multiple trajectories (at least 2+ issues).

SELECTION CRITERIA:
1. Problem appears in multiple trajectories (check "trajectory_count" and "issue_ids")
2. Same validation/failure family
3. Same or similar fix strategy
4. If duplicate problems (same root cause + fix), select ONE

IGNORE:
- Problems that appear in only 1 trajectory (not common)
- Current CI failure matching (common problems don't need to match CI)

OUTPUT:
Return a JSON array of DISTINCT common problems. Deduplicate if same problem appears multiple times.

[
  {{
    "validation_cmd": "exact command",
    "failure_type": "category",
    "files": ["paths"],
    "problem": "description",
    "root_cause": "why",
    "how_fixed": "what to change",
    "why_fix_works": "why it works"
  }}
]
"""

    response = _call_llm(llm, prompt)
    selected = _parse_json_array(response)

    if not selected:
        logger.warning("[L2 Common] No common problems selected")
    else:
        logger.info(f"[L2 Common] Selected {len(selected)} common problems")
    return selected


def _llm_select_consecutive_problems(
    problem_1: Dict[str, Any],
    consecutive_candidates: List[Dict[str, Any]],
    llm: Any,
) -> List[Dict[str, Any]]:
    """
    Select CONSECUTIVE problems: problems that appear AFTER fixing the primary CI failure.

    Criteria: Dependency-based (revealed after fixing primary)
    """
    if not consecutive_candidates:
        return []

    current_problem = _compact_problem(problem_1)
    prompt = f"""Select CONSECUTIVE problems that may appear after fixing the primary CI failure.

CURRENT CI FAILURE (PRIMARY):
{json.dumps(current_problem, indent=2)}

CONSECUTIVE CANDIDATES (non-primary problems from trajectories):
{json.dumps(_limit_candidates(consecutive_candidates, 40), indent=2)}

TASK:
Select problems that can REASONABLY appear AFTER fixing the current CI failure.

SELECTION CRITERIA:
1. Technically connected to the current failure (same system/component/validation)
2. Revealed by the same validation sequence
3. Dependent on the current failure being fixed first
4. Config/dependency/plugin changes connected to the same failure family

IGNORE:
- Unrelated problems
- Problems from completely different validation stages
- Duplicates (same root cause + same fix)

OUTPUT:
Return a JSON array of DISTINCT consecutive problems.

[
  {{
    "validation_cmd": "exact command",
    "failure_type": "category",
    "files": ["paths"],
    "problem": "description",
    "root_cause": "why",
    "how_fixed": "what to change",
    "why_fix_works": "why it works"
  }}
]
"""

    response = _call_llm(llm, prompt)
    selected = _parse_json_array(response)

    if not selected:
        logger.warning("[L2 Consecutive] No consecutive problems selected")
    else:
        logger.info(f"[L2 Consecutive] Selected {len(selected)} consecutive problems")
    return selected


def _merge_and_deduplicate(
    common_problems: List[Dict[str, Any]],
    consecutive_problems: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge common and consecutive problems, removing duplicates.

    If a problem appears in both lists (same root_cause + how_fixed), keep ONE.
    """
    def problem_key(p):
        """Create a unique key for deduplication."""
        root = _clean_text(p.get("root_cause", ""), 200).lower()
        fix = _clean_text(p.get("how_fixed", ""), 200).lower()
        return f"{root}|{fix}"

    seen = set()
    merged = []

    # Add common problems first (higher priority)
    for p in common_problems:
        key = problem_key(p)
        if key not in seen:
            seen.add(key)
            merged.append(p)

    # Add consecutive problems (skip if already in common)
    for p in consecutive_problems:
        key = problem_key(p)
        if key not in seen:
            seen.add(key)
            merged.append(p)

    logger.info(
        f"[L2 Merge] {len(common_problems)} common + {len(consecutive_problems)} consecutive "
        f"→ {len(merged)} distinct problems (removed {len(common_problems) + len(consecutive_problems) - len(merged)} duplicates)"
    )

    return merged


def _normalize_selected_problem(problem: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Normalize L2 problem to minimal essential fields.

    Note: Deduplication already happened in LLM selection, so we don't need
    to track commonality or how many times a problem appeared.
    """
    return {
        "problem_id": problem.get("problem_id") or index,
        "is_primary": False,
        "source": "L2",
        "validation_cmd": _clean_text(problem.get("validation_cmd")),
        "failure_type": _clean_text(problem.get("failure_type")),
        "problem": _clean_text(problem.get("description") or problem.get("problem"), 900),
        "root_cause": _clean_text(problem.get("root_cause"), 900),
        "how_fixed": _clean_text(problem.get("how_fixed") or problem.get("fix"), 900),
        "why_fix_works": _clean_text(problem.get("why_fix_works"), 700),
        "files": _dedupe_keep_order(_as_list(problem.get("files")), 20),
    }


def staged_l2_analysis(
    problem_1: Dict[str, Any],
    l2_memories: List[Dict[str, Any]],
    llm: Any,
) -> List[Dict[str, Any]]:
    """
    Analyze fetched L2 trajectories.

    Common problems are selected from repeated grouped evidence and do not need
    direct CI matching. Consecutive problems are selected from later trajectory
    steps using the current CI failure as the anchor.
    """
    if not l2_memories:
        logger.info("[L2] No L2 memories provided")
        return []

    rows, total_trajectories = _flatten_l2(l2_memories)
    if not rows:
        logger.info("[L2] No trajectory problems found")
        return []

    common_candidates = _group_candidate_rows(
        rows,
        total_trajectories=total_trajectories,
        prefix="C",
        include_primary=True,
    )
    consecutive_candidates = _group_candidate_rows(
        rows,
        total_trajectories=total_trajectories,
        prefix="N",
        include_primary=False,
    )

    logger.info(
        "[L2] Prepared candidates: rows=%d common=%d consecutive=%d trajectories=%d",
        len(rows),
        len(common_candidates),
        len(consecutive_candidates),
        total_trajectories,
    )

    # Select common and consecutive problems SEPARATELY
    common_problems = _llm_select_common_problems(common_candidates, llm)
    consecutive_problems = _llm_select_consecutive_problems(problem_1, consecutive_candidates, llm)

    # Merge and deduplicate
    merged = _merge_and_deduplicate(common_problems, consecutive_problems)

    # Normalize to final format
    normalized = [_normalize_selected_problem(problem, index) for index, problem in enumerate(merged, 1)]

    logger.info(
        "[L2] Final: %d problems (%d common, %d consecutive, merged and deduplicated)",
        len(normalized),
        len(common_problems),
        len(consecutive_problems),
    )
    return normalized
