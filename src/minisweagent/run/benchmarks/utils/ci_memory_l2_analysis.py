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

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

try:
    from .memory_plugin import _EmbeddingProvider
    _EMBEDDING_AVAILABLE = True
except ImportError:
    _EmbeddingProvider = None  # type: ignore
    _EMBEDDING_AVAILABLE = False

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


def _cluster_problems_by_embedding(
    rows: List[Dict[str, Any]],
    total_trajectories: int,
    similarity_threshold: float = 0.5,
    frequency_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Cluster problems using embeddings (problem_statement + root_cause).

    Steps:
    1. Extract all problems and create signatures (problem + root_cause)
    2. Embed each signature
    3. Cluster by cosine similarity >= similarity_threshold
    4. Count frequency based on unique L2 records (not total problems)
    5. Filter by frequency_ratio >= frequency_threshold
    6. Merge problems in each cluster

    Args:
        rows: Flattened problem rows from L2 trajectories
        total_trajectories: Total number of L2 records fetched
        similarity_threshold: Cosine similarity threshold (default: 0.5)
        frequency_threshold: Frequency ratio based on L2 records (default: 0.3, i.e., 30%)

    Returns:
        List of clustered common problems with high frequency
    """
    if not _EMBEDDING_AVAILABLE or not _NUMPY_AVAILABLE:
        logger.warning("[L2 Clustering] Embedding not available, falling back to exact grouping")
        return []

    # Type guard: at this point we know these are available
    assert _EmbeddingProvider is not None
    assert np is not None

    if not rows:
        return []

    # Step 1: Create signatures for all problems
    problems_with_signatures = []
    for row in rows:
        problem_text = row.get("problem", "")
        root_cause = row.get("root_cause", "")
        # Combine problem + root_cause for better matching
        signature = f"{problem_text} | {root_cause}".strip()

        if signature and signature != "|":
            problems_with_signatures.append({
                "signature": signature,
                "row": row
            })

    if not problems_with_signatures:
        logger.warning("[L2 Clustering] No valid problem signatures found")
        return []

    logger.info(f"[L2 Clustering] Processing {len(problems_with_signatures)} problems")

    # Step 2: Embed all signatures
    embedder = _EmbeddingProvider.get()
    embeddings = []
    valid_problems = []

    for item in problems_with_signatures:
        emb = embedder.embed(item["signature"])
        if emb is not None:
            embeddings.append(emb)
            valid_problems.append(item)

    if not embeddings:
        logger.warning("[L2 Clustering] No valid embeddings generated")
        return []

    logger.info(f"[L2 Clustering] Generated {len(embeddings)} embeddings")

    # Step 3: Cluster by cosine similarity
    clusters = []
    assigned = [False] * len(embeddings)

    for i in range(len(embeddings)):
        if assigned[i]:
            continue

        # Start new cluster with problem i
        cluster = [valid_problems[i]]
        assigned[i] = True

        # Find all similar problems
        for j in range(i + 1, len(embeddings)):
            if assigned[j]:
                continue

            # Compute cosine similarity (dot product since embeddings are normalized)
            similarity = float(np.dot(embeddings[i], embeddings[j]))

            if similarity >= similarity_threshold:
                cluster.append(valid_problems[j])
                assigned[j] = True

        clusters.append(cluster)

    logger.info(f"[L2 Clustering] Created {len(clusters)} clusters")

    # Step 4: Count frequency based on unique L2 records
    common_problems = []

    for cluster in clusters:
        # Count unique L2 records (issue_ids) in this cluster
        unique_issue_ids = set()
        for item in cluster:
            issue_id = item["row"].get("issue_id")
            if issue_id:
                unique_issue_ids.add(issue_id)

        # Frequency = how many unique L2 records contain this problem
        frequency = len(unique_issue_ids)
        frequency_ratio = frequency / max(total_trajectories, 1)

        if frequency_ratio >= frequency_threshold:
            # This cluster represents a common problem pattern
            # Merge all problems in the cluster
            merged = _merge_cluster_problems(cluster)
            merged["frequency"] = frequency
            merged["frequency_ratio"] = round(frequency_ratio, 2)
            merged["unique_l2_records"] = frequency  # How many L2 records
            merged["total_instances"] = len(cluster)  # Total occurrences
            common_problems.append(merged)

    # Sort by frequency (highest first)
    common_problems.sort(key=lambda x: x["frequency"], reverse=True)

    logger.info(
        f"[L2 Clustering] Selected {len(common_problems)} common problems "
        f"(appears in >= {frequency_threshold*100}% of L2 records)"
    )

    return common_problems


def _merge_cluster_problems(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge all problems in a cluster into one representative problem.

    Combines:
    - problem statements (all unique ones)
    - root causes (all unique ones)
    - fixes (all unique ones)
    - files (all unique ones)
    - validation commands (first one)
    - failure types (first one)
    """
    if not cluster:
        return {}

    # Get representative (first) problem
    representative = cluster[0]["row"]

    # Collect all unique values from cluster
    all_problems = []
    all_root_causes = []
    all_fixes = []
    all_why_works = []
    all_files = []
    all_issue_ids = set()

    for item in cluster:
        row = item["row"]

        problem = row.get("problem", "").strip()
        if problem and problem not in all_problems:
            all_problems.append(problem)

        root_cause = row.get("root_cause", "").strip()
        if root_cause and root_cause not in all_root_causes:
            all_root_causes.append(root_cause)

        fix = row.get("how_fixed", "").strip()
        if fix and fix not in all_fixes:
            all_fixes.append(fix)

        why = row.get("why_fix_works", "").strip()
        if why and why not in all_why_works:
            all_why_works.append(why)

        files = _as_list(row.get("files"))
        for f in files:
            if f and f not in all_files:
                all_files.append(f)

        issue_id = row.get("issue_id")
        if issue_id:
            all_issue_ids.add(issue_id)

    # Merge into single problem
    merged = {
        "validation_cmd": representative.get("validation_cmd", ""),
        "failure_type": representative.get("failure_type", ""),
        "issue_type": representative.get("issue_type", ""),
        "problem": "\n".join(all_problems) if all_problems else "",
        "root_cause": "\n".join(all_root_causes) if all_root_causes else "",
        "how_fixed": "\n".join(all_fixes) if all_fixes else "",
        "why_fix_works": "\n".join(all_why_works) if all_why_works else "",
        "files": all_files,  # Include ALL files from cluster (no limit)
        "appears_in_issues": list(all_issue_ids),
        "evidence_count": len(cluster),
    }

    return merged


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
1. Problem appears in multiple trajectories (check "frequency", "frequency_ratio", "appears_in_issues")
2. Same validation/failure family
3. Same or similar fix strategy
4. If duplicate problems (same root cause + fix), select ONE

IGNORE:
- Problems that appear in only 1 trajectory (not common)
- Current CI failure matching (common problems don't need to match CI)

OUTPUT:
Return a JSON array of DISTINCT common problems. Deduplicate if same problem appears multiple times.
IMPORTANT: Include the "files" field from the candidates - copy the file paths exactly.

[
  {{
    "validation_cmd": "exact command",
    "failure_type": "category",
    "files": ["copy file paths from candidates"],
    "problem": "description",
    "root_cause": "why",
    "how_fixed": "what to change",
    "why_fix_works": "why it works"
  }},
  // ... more problems
]
"""

    response = _call_llm(llm, prompt)
    selected = _parse_json_array(response)
    
    # Post-process: Add back missing fields (files, frequency, etc.) from candidates
    for problem in selected:
        # Find matching candidate(s) by validation_cmd + failure_type
        validation_cmd = problem.get("validation_cmd", "").lower()
        failure_type = problem.get("failure_type", "").lower()

        # Look for ALL matching candidates (LLM might have merged multiple)
        matching_candidates = [
            c for c in common_candidates
            if c.get("validation_cmd", "").lower() == validation_cmd
            and c.get("failure_type", "").lower() == failure_type
        ]

        if matching_candidates:
            # Merge files from ALL matching candidates (LLM might have merged them)
            all_files = []
            all_issues = set()
            total_frequency = 0
            max_frequency_ratio = 0.0

            for candidate in matching_candidates:
                # Collect files from all candidates
                candidate_files = candidate.get("files", [])
                if isinstance(candidate_files, list):
                    all_files.extend(candidate_files)

                # Collect issue IDs
                candidate_issues = candidate.get("appears_in_issues", [])
                if isinstance(candidate_issues, list):
                    all_issues.update(candidate_issues)

                # Track highest frequency
                total_frequency = max(total_frequency, candidate.get("frequency", 0))
                max_frequency_ratio = max(max_frequency_ratio, candidate.get("frequency_ratio", 0.0))

            # Also include any files the LLM returned (merge with candidate files)
            llm_files = problem.get("files", [])
            if isinstance(llm_files, list):
                all_files.extend(llm_files)

            # Deduplicate files while preserving order
            seen = set()
            deduplicated_files = []
            for f in all_files:
                if f and f not in seen:
                    seen.add(f)
                    deduplicated_files.append(f)

            # ALWAYS set the complete merged files (even if LLM returned some)
            problem["files"] = deduplicated_files

            # Add frequency info if missing
            if "frequency" not in problem:
                problem["frequency"] = total_frequency
            if "frequency_ratio" not in problem:
                problem["frequency_ratio"] = max_frequency_ratio
            if "appears_in_issues" not in problem:
                problem["appears_in_issues"] = list(all_issues)

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
IMPORTANT: Include the "files" field from the candidates - copy the file paths exactly.

[
  {{
    "validation_cmd": "exact command",
    "failure_type": "category",
    "files": ["copy file paths from candidates"],
    "problem": "description",
    "root_cause": "why",
    "how_fixed": "what to change",
    "why_fix_works": "why it works"
  }}
]
"""

    response = _call_llm(llm, prompt)
    selected = _parse_json_array(response)

    # Post-process: Add back missing fields (files, frequency, etc.) from candidates
    for problem in selected:
        # Find matching candidate(s) by validation_cmd + failure_type
        validation_cmd = problem.get("validation_cmd", "").lower()
        failure_type = problem.get("failure_type", "").lower()

        # Look for ALL matching candidates (LLM might have merged multiple)
        matching_candidates = [
            c for c in consecutive_candidates
            if c.get("validation_cmd", "").lower() == validation_cmd
            and c.get("failure_type", "").lower() == failure_type
        ]

        if matching_candidates:
            # Merge files from ALL matching candidates (LLM might have merged them)
            all_files = []
            all_issues = set()
            total_frequency = 0
            max_frequency_ratio = 0.0

            for candidate in matching_candidates:
                # Collect files from all candidates
                candidate_files = candidate.get("files", [])
                if isinstance(candidate_files, list):
                    all_files.extend(candidate_files)

                # Collect issue IDs
                candidate_issues = candidate.get("appears_in_issues", [])
                if isinstance(candidate_issues, list):
                    all_issues.update(candidate_issues)

                # Track highest frequency
                total_frequency = max(total_frequency, candidate.get("frequency", 0))
                max_frequency_ratio = max(max_frequency_ratio, candidate.get("frequency_ratio", 0.0))

            # Also include any files the LLM returned (merge with candidate files)
            llm_files = problem.get("files", [])
            if isinstance(llm_files, list):
                all_files.extend(llm_files)

            # Deduplicate files while preserving order
            seen = set()
            deduplicated_files = []
            for f in all_files:
                if f and f not in seen:
                    seen.add(f)
                    deduplicated_files.append(f)

            # ALWAYS set the complete merged files (even if LLM returned some)
            problem["files"] = deduplicated_files

            # Add frequency info if missing
            if "frequency" not in problem:
                problem["frequency"] = total_frequency
            if "frequency_ratio" not in problem:
                problem["frequency_ratio"] = max_frequency_ratio
            if "appears_in_issues" not in problem:
                problem["appears_in_issues"] = list(all_issues)

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
    Analyze fetched L2 trajectories using embedding-based clustering.

    Steps:
    1. Flatten all L2 trajectories into problem rows
    2. Cluster problems by embedding similarity (problem + root_cause)
    3. Filter by frequency (>= threshold % of L2 records, not total problems)
    4. Pass high-frequency problems to LLM for final selection

    Frequency is L2-based: counts how many unique L2 records contain the problem,
    not total problem instances. This identifies problems that appear across
    multiple different CI issues.
    """
    if not l2_memories:
        logger.info("[L2] No L2 memories provided")
        return []

    rows, total_trajectories = _flatten_l2(l2_memories)
    if not rows:
        logger.info("[L2] No trajectory problems found")
        return []

    logger.info(f"[L2] Extracted {len(rows)} problems from {total_trajectories} trajectories")

    # Use embedding-based clustering to find common problems
    common_candidates = _cluster_problems_by_embedding(
        rows,
        total_trajectories=total_trajectories,
        similarity_threshold=0.6,
        frequency_threshold=0.3,
    )

    if not common_candidates:
        logger.warning("[L2] No common problems found with embedding clustering, falling back to old method")
        # Fallback to old grouping method if clustering fails
        common_candidates = _group_candidate_rows(
            rows,
            total_trajectories=total_trajectories,
            prefix="C",
            include_primary=True,
        )

    # Also get consecutive candidates (non-primary problems)
    consecutive_candidates = _group_candidate_rows(
        rows,
        total_trajectories=total_trajectories,
        prefix="N",
        include_primary=False,
    )

    logger.info(
        "[L2] Prepared candidates: common=%d consecutive=%d",
        len(common_candidates),
        len(consecutive_candidates),
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
