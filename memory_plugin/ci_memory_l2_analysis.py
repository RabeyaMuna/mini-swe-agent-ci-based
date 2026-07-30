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
import signal
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from utilities.model_token_config import get_l2_common_candidates
from utilities.model_token_config import get_l2_consecutive_candidates

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


def _extract_model_name_from_llm(llm: Any) -> str | None:
    """
    Extract model name from LLM object.

    Priority order:
    1. memci_model_key (set by utilities.llm_provider.get_llm)
    2. model_name, model, model_id (LangChain attributes)

    Returns None if not found.
    """
    if llm is None:
        return None

    # Priority 1: Check for memci_model_key (set by utilities.llm_provider)
    if hasattr(llm, "memci_model_key"):
        value = getattr(llm, "memci_model_key", None)
        if value and isinstance(value, str):
            return value

    # Priority 2: Try common LangChain attribute names
    for attr in ("model_name", "model", "model_id"):
        if hasattr(llm, attr):
            value = getattr(llm, attr, None)
            if value and isinstance(value, str):
                return value

    return None


class TimeoutError(Exception):
    """Raised when an LLM call exceeds timeout."""

    pass


def _timeout_handler(signum, frame):
    """Signal handler for LLM call timeout."""
    raise TimeoutError("LLM call timed out")


def _call_llm(llm: Any, prompt: str, timeout: int = 120, max_retries: int = 2) -> str:
    """
    Call either a LangChain-style object or a plain callable with timeout and retry.

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
                except (AttributeError, ImportError):
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
        key = (
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else str(value)
        )
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
    if any(
        step.get(key)
        for key in ("problem", "root_cause", "how_fixed", "why_fix_works", "issue_type")
    ):
        return [step]
    return []


def _calculate_multi_signal_similarity(
    prob1: Dict[str, Any],
    prob2: Dict[str, Any],
    emb1: Any,  # np.ndarray when numpy available
    emb2: Any,  # np.ndarray when numpy available
) -> float:
    """
    Calculate comprehensive similarity using ALL available data fields.

    Combines multiple signals:
    - File overlap (40%): Same files = highly related
    - Semantic similarity (30%): Meaning from all text fields
    - Issue type match (15%): E501 vs merge conflict
    - Validation command (10%): Same validator
    - Failure type (5%): Type error vs import error

    Args:
        prob1, prob2: Problem dictionaries with metadata
        emb1, emb2: Pre-computed semantic embeddings

    Returns:
        Combined similarity score (0.0-1.0)
    """
    row1 = prob1.get("row", {})
    row2 = prob2.get("row", {})

    # Signal 1: File Overlap (40% weight) - MOST RELIABLE
    files1 = set(row1.get("files", []))
    files2 = set(row2.get("files", []))

    if files1 and files2:
        intersection = len(files1 & files2)
        union = len(files1 | files2)
        file_overlap = intersection / union if union > 0 else 0.0

        # Bonus for exact same file set
        if files1 == files2 and len(files1) > 0:
            file_overlap = 1.0
    else:
        file_overlap = 0.0

    # Signal 2: Semantic Similarity (30% weight) - Already computed
    # Embeddings are normalized, so dot product = cosine similarity
    if np is not None:
        semantic_sim = float(np.dot(emb1, emb2))
    else:
        semantic_sim = 0.0

    # Signal 3: Issue Type Match (15% weight)
    type1 = str(row1.get("issue_type", "")).lower().strip()
    type2 = str(row2.get("issue_type", "")).lower().strip()

    if type1 and type2:
        if type1 == type2:
            type_match = 1.0  # Exact match
        elif type1 in type2 or type2 in type1:
            type_match = 0.7  # Partial match (e.g., "E501" in "E501 line-too-long")
        else:
            type_match = 0.0
    else:
        type_match = 0.0

    # Signal 4: Validation Command Match (10% weight)
    cmd1 = str(row1.get("validation_cmd", "")).lower().strip()
    cmd2 = str(row2.get("validation_cmd", "")).lower().strip()

    if cmd1 and cmd2:
        if cmd1 == cmd2:
            cmd_match = 1.0
        # Same validator family (e.g., "ruff" vs "ruff check")
        elif any(
            v in cmd1 and v in cmd2
            for v in ["ruff", "mypy", "pylint", "pytest", "doctest"]
        ):
            cmd_match = 0.5
        else:
            cmd_match = 0.0
    else:
        cmd_match = 0.0

    # Signal 5: Failure Type Match (5% weight)
    ftype1 = str(row1.get("failure_type", "")).lower().strip()
    ftype2 = str(row2.get("failure_type", "")).lower().strip()

    failure_match = 1.0 if (ftype1 and ftype2 and ftype1 == ftype2) else 0.0

    # Combined weighted similarity
    total_similarity = (
        file_overlap * 0.40  # 40% - file overlap (most reliable)
        + semantic_sim * 0.30  # 30% - semantic similarity
        + type_match * 0.15  # 15% - issue type
        + cmd_match * 0.10  # 10% - validation command
        + failure_match * 0.05  # 5% - failure type
    )

    return total_similarity


def _cluster_problems_by_embedding(
    rows: List[Dict[str, Any]],
    total_trajectories: int,
    similarity_threshold: float = 0.5,
    frequency_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Cluster problems using MULTI-SIGNAL similarity (not just semantic!).

    Steps:
    1. Extract all problems and create signatures (problem + root_cause)
    2. Embed each signature
    3. Cluster by MULTI-SIGNAL similarity >= threshold
       - File overlap (40%), semantic (30%), issue_type (15%), validation_cmd (10%), failure_type (5%)
    4. Count frequency based on unique L2 records (not total problems)
    5. Filter by frequency_ratio >= frequency_threshold
    6. Merge problems in each cluster

    Args:
        rows: Flattened problem rows from L2 trajectories
        total_trajectories: Total number of L2 records fetched
        similarity_threshold: Multi-signal similarity threshold (default: 0.5)
        frequency_threshold: Frequency ratio based on L2 records (default: 0.3, i.e., 30%)

    Returns:
        List of clustered common problems with high frequency
    """
    if not _EMBEDDING_AVAILABLE or not _NUMPY_AVAILABLE:
        logger.warning(
            "[L2 Clustering] Embedding not available, falling back to exact grouping"
        )
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
            problems_with_signatures.append({"signature": signature, "row": row})

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

    # Step 3: Cluster by MULTI-SIGNAL similarity (not just semantic!)
    # Combines: file overlap (40%), semantic (30%), issue type (15%),
    # validation cmd (10%), failure type (5%)
    clusters = []
    assigned = [False] * len(embeddings)

    for i in range(len(embeddings)):
        if assigned[i]:
            continue

        # Start new cluster with problem i
        cluster = [valid_problems[i]]
        assigned[i] = True

        # Find all similar problems using comprehensive similarity
        for j in range(i + 1, len(embeddings)):
            if assigned[j]:
                continue

            # Calculate multi-signal similarity
            similarity = _calculate_multi_signal_similarity(
                valid_problems[i], valid_problems[j], embeddings[i], embeddings[j]
            )

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
        f"(appears in >= {frequency_threshold * 100}% of L2 records)"
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


def _is_valid_validation_problem(problem: Dict[str, Any]) -> bool:
    """
    Check if a problem is a real validation issue (not git artifacts).

    IMPORTANT: Merge conflicts are excluded UNLESS the file changes are related
    to solving the validation failure. For example:
    - [FAIL] Exclude: "merge conflict markers in file.py" (pure git artifact)
    - OK Include: "type error in file.py" even if found during merge conflict resolution

    The deep analysis and classify prompts are already instructed to focus on
    validation-related changes, so this filter only removes PURE git workflow issues.

    Filters out:
    - Pure merge conflict markers (git workflow issues, not validation failures)
    - Git conflict artifacts that don't involve validation logic

    Args:
        problem: Problem dictionary with issue_type, problem text, how_fixed, root_cause

    Returns:
        True if valid validation problem, False if should be excluded
    """
    issue_type = str(problem.get("issue_type", "")).lower()
    problem_text = str(problem.get("problem", "")).lower()
    how_fixed = str(problem.get("how_fixed", "")).lower()
    root_cause = str(problem.get("root_cause", "")).lower()

    # Check if this is a PURE merge conflict (no validation logic involved)
    merge_conflict_keywords = ["merge conflict", "conflict markers", "git conflict"]

    is_merge_conflict = any(
        keyword in issue_type or keyword in problem_text
        for keyword in merge_conflict_keywords
    )

    if not is_merge_conflict:
        # Not a merge conflict at all - include
        return True

    # It's marked as merge conflict - check if it has validation-related changes
    validation_keywords = [
        "validation",
        "test",
        "lint",
        "type",
        "error",
        "failure",
        "check",
        "syntax",
        "import",
        "undefined",
        "missing",
        "invalid",
        "warning",
    ]

    has_validation_content = any(
        keyword in how_fixed or keyword in root_cause or keyword in problem_text
        for keyword in validation_keywords
    )

    if has_validation_content:
        # Merge conflict but involves validation fixes - include
        logger.debug(
            f"[L2 Filter] Including merge conflict with validation content: {issue_type}"
        )
        return True
    else:
        # Pure merge conflict with no validation logic - exclude
        logger.debug(f"[L2 Filter] Excluding pure merge conflict: {issue_type}")
        return False


def _flatten_l2(l2_memories: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Flatten L2 repair trajectories into problem rows, filtering out non-validation issues."""
    rows: List[Dict[str, Any]] = []
    trajectory_ids = set()
    row_id = 1
    excluded_count = 0

    for memory_index, memory in enumerate(l2_memories, 1):
        issue_id = str(
            memory.get("issue_id") or memory.get("id") or f"memory_{memory_index}"
        )
        trajectory_ids.add(issue_id)
        # L2 can have either "repair_trajectory", "trajectory", or "problems" (NEW!)
        trajectory = (
            memory.get("repair_trajectory")
            or memory.get("trajectory")
            or memory.get("problems")
            or []
        )
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
            validation_cmd = _clean_text(
                step.get("validation_cmd") or memory.get("validation_cmd")
            )
            failure_type = _clean_text(
                step.get("failure_type") or memory.get("failure_type")
            )

            for problem in _step_problems(step):
                # Filter out non-validation problems (merge conflicts, etc.)
                if not _is_valid_validation_problem(problem):
                    excluded_count += 1
                    continue

                problem_text = _clean_text(
                    problem.get("problem") or problem.get("description"), 700
                )
                root_cause = _clean_text(problem.get("root_cause"), 700)
                how_fixed = _clean_text(
                    problem.get("how_fixed") or problem.get("fix"), 700
                )
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
                        "issue_type": _clean_text(
                            problem.get("issue_type") or step.get("issue_type"), 160
                        ),
                        "problem": problem_text,
                        "root_cause": root_cause,
                        "how_fixed": how_fixed,
                        "why_fix_works": why_fix_works,
                        "files": _dedupe_keep_order(
                            _as_list(
                                problem.get("files") or problem.get("affected_files")
                            ),
                            20,
                        ),
                    }
                )
                row_id += 1

    if excluded_count > 0:
        logger.info(
            f"[L2 Filter] Excluded {excluded_count} non-validation problems (merge conflicts, etc.)"
        )

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
        "failure_type": problem_1.get("failure_type")
        or problem_1.get("error_type", ""),
        "description": _clean_text(
            problem_1.get("description") or problem_1.get("problem"), 900
        ),
        "root_cause": _clean_text(problem_1.get("root_cause"), 1200),
        "files": [
            item.get("path", item) if isinstance(item, dict) else item
            for item in _as_list(problem_1.get("files"))
        ],
        "error_details": _as_list(problem_1.get("error_details"))[:8],
    }


def _limit_candidates(
    candidates: List[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    """Keep high-signal candidates while leaving semantic choice to the LLM."""
    return candidates[:limit]


def _analyze_file_frequency(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze which files appear in multiple problems to identify high-priority targets.

    When a file appears in multiple distinct problems, it suggests:
    1. A root cause affecting multiple validations
    2. High ROI - fixing one file resolves multiple problems
    3. Should be prioritized in selection

    Args:
        candidates: List of problem candidates from clustering

    Returns:
        Dictionary containing:
        - high_frequency_files: Files appearing in 2+ problems with details
        - consolidation_suggestions: Recommendations for high-priority fixes
    """
    file_to_problems = defaultdict(list)

    # Map each file to all problems it appears in
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id", candidate.get("id", "unknown"))
        files = candidate.get("files", [])

        if not isinstance(files, list):
            files = [files] if files else []

        for file_path in files:
            if file_path and isinstance(file_path, str):
                file_to_problems[file_path].append(
                    {
                        "candidate_id": candidate_id,
                        "issue_type": candidate.get("issue_type", ""),
                        "failure_type": candidate.get("failure_type", ""),
                        "problem_summary": candidate.get("problem", "")[:150],
                        "frequency": candidate.get("frequency", 0),
                    }
                )

    # Identify high-frequency files (2+ occurrences)
    high_frequency_files = {}
    consolidation_suggestions = []

    for file_path, problems in file_to_problems.items():
        if len(problems) >= 2:
            # Collect unique issue types and failure types
            issue_types = list(
                set(p["issue_type"] for p in problems if p.get("issue_type"))
            )
            failure_types = list(
                set(p["failure_type"] for p in problems if p.get("failure_type"))
            )

            high_frequency_files[file_path] = {
                "frequency": len(problems),
                "problem_ids": [p["candidate_id"] for p in problems],
                "issue_types": issue_types,
                "failure_types": failure_types,
            }

            # Create consolidation suggestion
            consolidation_suggestions.append(
                {
                    "file": file_path,
                    "frequency": len(problems),
                    "issue_types": issue_types[:3],
                    "priority": "HIGH" if len(problems) >= 3 else "MEDIUM",
                    "recommendation": (
                        f"File '{file_path}' appears in {len(problems)} distinct problems. "
                        f"Likely cascading failure - fix this file to resolve multiple issues."
                    ),
                    "candidate_ids": [p["candidate_id"] for p in problems],
                }
            )

    # Sort by frequency (most frequent first)
    consolidation_suggestions.sort(key=lambda x: (-x["frequency"], x["file"]))

    if high_frequency_files:
        logger.info(
            f"[L2 File Frequency] Found {len(high_frequency_files)} high-frequency files "
            f"(appearing in 2+ problems)"
        )
        for suggestion in consolidation_suggestions[:5]:
            logger.info(
                f"  - {suggestion['file']}: {suggestion['frequency']} problems ({suggestion['priority']})"
            )

    return {
        "high_frequency_files": high_frequency_files,
        "consolidation_suggestions": consolidation_suggestions,
        "high_frequency_count": len(high_frequency_files),
    }


def _format_file_frequency_for_prompt(
    file_analysis: Dict[str, Any], max_files: int = 10
) -> str:
    """
    Format file frequency analysis as markdown for LLM prompt.

    Args:
        file_analysis: Output from _analyze_file_frequency()
        max_files: Maximum files to include (default: 10)

    Returns:
        Formatted markdown string
    """
    suggestions = file_analysis.get("consolidation_suggestions", [])

    if not suggestions:
        return ""

    lines = [
        "FILE FREQUENCY ANALYSIS - HIGH PRIORITY TARGETS:",
        f"The following {len(suggestions)} file(s) appear in MULTIPLE distinct problems.",
        "These are HIGH PRIORITY because fixing ONE file resolves MULTIPLE problems:",
        "",
    ]

    for i, suggestion in enumerate(suggestions[:max_files], 1):
        lines.append(
            f"{i}. {suggestion['file']} ({suggestion['priority']} PRIORITY): "
            f"{suggestion['frequency']} problems ({', '.join(suggestion['issue_types'])})"
        )

    if len(suggestions) > max_files:
        lines.append(f"... and {len(suggestions) - max_files} more files")

    lines.append("")
    lines.append(
        "IMPORTANT: When selecting problems, PRIORITIZE files with frequency >= 2 "
        "as they likely represent root causes affecting multiple validations."
    )
    lines.append("")

    return "\n".join(lines)


def _validate_file_frequency_coverage(
    selected: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    min_frequency_threshold: int = 3,
) -> List[Dict[str, Any]]:
    """
    Validate that high-frequency files are represented in selection.

    If a file appears in >= min_frequency_threshold problems but is NOT
    in any selected problem, force-add the best candidate for that file.

    Args:
        selected: Problems selected by LLM
        candidates: All candidate problems
        min_frequency_threshold: Minimum frequency to trigger force-add (default: 3)

    Returns:
        Updated selection with high-frequency files added if missing
    """
    file_analysis = _analyze_file_frequency(candidates)

    # Collect all files in selected problems
    selected_files = set()
    for problem in selected:
        files = problem.get("files", [])
        if not isinstance(files, list):
            files = [files] if files else []
        selected_files.update(files)

    # Check for high-frequency files missing from selection
    additions = []
    missed_files = []

    for file_path, info in file_analysis["high_frequency_files"].items():
        if (
            info["frequency"] >= min_frequency_threshold
            and file_path not in selected_files
        ):
            missed_files.append((file_path, info["frequency"]))

            # Find the best candidate for this file (highest frequency)
            best_candidate = None
            best_freq = 0

            for candidate in candidates:
                if file_path in candidate.get("files", []):
                    freq = candidate.get("frequency", 0)
                    if freq > best_freq:
                        best_freq = freq
                        best_candidate = candidate

            if best_candidate:
                logger.warning(
                    f"[L2 File Frequency Safeguard] Adding high-frequency file: "
                    f"'{file_path}' (appeared in {info['frequency']} problems but not selected by LLM)"
                )
                additions.append(best_candidate)

    if missed_files:
        logger.warning(
            f"[L2 File Frequency Safeguard] {len(missed_files)} high-frequency files "
            f"(>={min_frequency_threshold} occurrences) were missed by LLM selection"
        )
    else:
        logger.info(
            f"[L2 File Frequency Safeguard] All high-frequency files "
            f"(>={min_frequency_threshold} occurrences) are covered OK"
        )

    return selected + additions


def _llm_select_common_problems(
    common_candidates: List[Dict[str, Any]],
    llm: Any,
    model_name: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Select COMMON problems: problems that appear repeatedly across multiple trajectories.

    Criteria: Repetition-based (appears in 2+ trajectories)

    Args:
        common_candidates: List of common problem candidates
        llm: LLM client for selection
        model_name: Model name for determining candidate limits (None = use default)
    """
    if not common_candidates:
        return []

    # Analyze file frequency to identify high-priority targets
    file_analysis = _analyze_file_frequency(common_candidates)
    file_freq_section = _format_file_frequency_for_prompt(file_analysis)

    # Get model-aware limit for common problems
    try:
        common_limit = get_l2_common_candidates(model_name)
    except Exception:
        common_limit = 30  # Fallback

    prompt = f"""Select COMMON problems from L2 trajectories.

{file_freq_section}
COMMON CANDIDATES (grouped by similarity):
{json.dumps(_limit_candidates(common_candidates, common_limit), indent=2)}

TASK:
Select problems that appear REPEATEDLY across multiple trajectories (at least 2+ issues).

SELECTION CRITERIA (IN PRIORITY ORDER):
1. **HIGH PRIORITY**: Files appearing in MULTIPLE problems (frequency >= 2)
   - These likely represent root causes with cascading failures
   - Check the FILE FREQUENCY ANALYSIS section above
   - Prioritize selecting problems for these high-frequency files
2. Problem appears in multiple trajectories (check "frequency", "frequency_ratio", "appears_in_issues")
3. Same validation/failure family
4. Same or similar fix strategy
5. If duplicate problems (same root cause + fix), select ONE

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

    response = _call_llm(llm, prompt, timeout=60, max_retries=1)

    if not response or not response.strip():
        logger.warning("[L2 Common] LLM returned empty response, returning empty list")
        return []

    selected = _parse_json_array(response)

    # Post-process: Add back missing fields (files, frequency, etc.) from candidates
    for problem in selected:
        # Find matching candidate(s) by validation_cmd + failure_type
        validation_cmd = problem.get("validation_cmd", "").lower()
        failure_type = problem.get("failure_type", "").lower()

        # Look for ALL matching candidates (LLM might have merged multiple)
        matching_candidates = [
            c
            for c in common_candidates
            if c.get("validation_cmd", "").lower() == validation_cmd
            and c.get("failure_type", "").lower() == failure_type
        ]

        if matching_candidates:
            # Merge data from ALL matching candidates (LLM might have merged them)
            all_files = []
            all_issues = set()
            all_how_fixed = []
            all_why_works = []
            total_frequency = 0
            max_frequency_ratio = 0.0

            for candidate in matching_candidates:
                # Collect files from all candidates
                candidate_files = candidate.get("files", [])
                if isinstance(candidate_files, list):
                    all_files.extend(candidate_files)

                # Collect fix strategies from all candidates
                how_fixed = candidate.get("how_fixed", "").strip()
                if how_fixed and how_fixed not in all_how_fixed:
                    all_how_fixed.append(how_fixed)

                why_works = candidate.get("why_fix_works", "").strip()
                if why_works and why_works not in all_why_works:
                    all_why_works.append(why_works)

                # Collect issue IDs
                candidate_issues = candidate.get("appears_in_issues", [])
                if isinstance(candidate_issues, list):
                    all_issues.update(candidate_issues)

                # Track highest frequency
                total_frequency = max(total_frequency, candidate.get("frequency", 0))
                max_frequency_ratio = max(
                    max_frequency_ratio, candidate.get("frequency_ratio", 0.0)
                )

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

            # Merge fix strategies if LLM didn't provide them or to enhance what LLM provided
            if not problem.get("how_fixed") and all_how_fixed:
                problem["how_fixed"] = "\n\n".join(all_how_fixed)
            elif problem.get("how_fixed") and all_how_fixed:
                # LLM provided some, but ensure all candidate fixes are included
                llm_how_fixed = problem.get("how_fixed", "").strip()
                if llm_how_fixed not in all_how_fixed:
                    all_how_fixed.insert(0, llm_how_fixed)
                problem["how_fixed"] = "\n\n".join(all_how_fixed)

            if not problem.get("why_fix_works") and all_why_works:
                problem["why_fix_works"] = "\n\n".join(all_why_works)
            elif problem.get("why_fix_works") and all_why_works:
                # LLM provided some, but ensure all candidate reasons are included
                llm_why_works = problem.get("why_fix_works", "").strip()
                if llm_why_works not in all_why_works:
                    all_why_works.insert(0, llm_why_works)
                problem["why_fix_works"] = "\n\n".join(all_why_works)

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

        # Log file frequency coverage
        if file_analysis["high_frequency_count"] > 0:
            selected_files = set()
            for p in selected:
                selected_files.update(p.get("files", []))

            covered_count = sum(
                1
                for f in file_analysis["high_frequency_files"].keys()
                if f in selected_files
            )
            logger.info(
                f"[L2 Common] High-frequency file coverage: "
                f"{covered_count}/{file_analysis['high_frequency_count']}"
            )
    return selected


def _llm_select_consecutive_problems(
    problem_1: Dict[str, Any],
    consecutive_candidates: List[Dict[str, Any]],
    llm: Any,
    model_name: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Select CONSECUTIVE problems: problems that appear AFTER fixing the primary CI failure.

    Criteria: Dependency-based (revealed after fixing primary)

    Args:
        problem_1: Current CI failure (primary problem)
        consecutive_candidates: List of consecutive problem candidates
        llm: LLM client for selection
        model_name: Model name for determining candidate limits (None = use default)
    """
    if not consecutive_candidates:
        return []

    # Analyze file frequency to identify high-priority targets
    file_analysis = _analyze_file_frequency(consecutive_candidates)
    file_freq_section = _format_file_frequency_for_prompt(file_analysis)

    # Get model-aware limit for consecutive problems
    try:
        consecutive_limit = get_l2_consecutive_candidates(model_name)
    except Exception:
        consecutive_limit = 40  # Fallback

    current_problem = _compact_problem(problem_1)
    prompt = f"""Select CONSECUTIVE problems that may appear after fixing the primary CI failure.

CURRENT CI FAILURE (PRIMARY):
{json.dumps(current_problem, indent=2)}

{file_freq_section}
CONSECUTIVE CANDIDATES (non-primary problems from trajectories):
{json.dumps(_limit_candidates(consecutive_candidates, consecutive_limit), indent=2)}

TASK:
Select problems that can REASONABLY appear AFTER fixing the current CI failure.

SELECTION CRITERIA (IN PRIORITY ORDER):
1. **HIGH PRIORITY**: Files appearing in MULTIPLE problems (frequency >= 2)
   - Check the FILE FREQUENCY ANALYSIS section above
   - Prioritize selecting problems for these high-frequency files
2. Technically connected to the current failure (same system/component/validation)
3. Revealed by the same validation sequence
4. Dependent on the current failure being fixed first
5. Config/dependency/plugin changes connected to the same failure family

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

    response = _call_llm(llm, prompt, timeout=60, max_retries=1)

    if not response or not response.strip():
        logger.warning("[L2 Common] LLM returned empty response, returning empty list")
        return []

    selected = _parse_json_array(response)

    # Post-process: Add back missing fields (files, frequency, etc.) from candidates
    for problem in selected:
        # Find matching candidate(s) by validation_cmd + failure_type
        validation_cmd = problem.get("validation_cmd", "").lower()
        failure_type = problem.get("failure_type", "").lower()

        # Look for ALL matching candidates (LLM might have merged multiple)
        matching_candidates = [
            c
            for c in consecutive_candidates
            if c.get("validation_cmd", "").lower() == validation_cmd
            and c.get("failure_type", "").lower() == failure_type
        ]

        if matching_candidates:
            # Merge data from ALL matching candidates (LLM might have merged them)
            all_files = []
            all_issues = set()
            all_how_fixed = []
            all_why_works = []
            total_frequency = 0
            max_frequency_ratio = 0.0

            for candidate in matching_candidates:
                # Collect files from all candidates
                candidate_files = candidate.get("files", [])
                if isinstance(candidate_files, list):
                    all_files.extend(candidate_files)

                # Collect fix strategies from all candidates
                how_fixed = candidate.get("how_fixed", "").strip()
                if how_fixed and how_fixed not in all_how_fixed:
                    all_how_fixed.append(how_fixed)

                why_works = candidate.get("why_fix_works", "").strip()
                if why_works and why_works not in all_why_works:
                    all_why_works.append(why_works)

                # Collect issue IDs
                candidate_issues = candidate.get("appears_in_issues", [])
                if isinstance(candidate_issues, list):
                    all_issues.update(candidate_issues)

                # Track highest frequency
                total_frequency = max(total_frequency, candidate.get("frequency", 0))
                max_frequency_ratio = max(
                    max_frequency_ratio, candidate.get("frequency_ratio", 0.0)
                )

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

            # Merge fix strategies if LLM didn't provide them or to enhance what LLM provided
            if not problem.get("how_fixed") and all_how_fixed:
                problem["how_fixed"] = "\n\n".join(all_how_fixed)
            elif problem.get("how_fixed") and all_how_fixed:
                # LLM provided some, but ensure all candidate fixes are included
                llm_how_fixed = problem.get("how_fixed", "").strip()
                if llm_how_fixed not in all_how_fixed:
                    all_how_fixed.insert(0, llm_how_fixed)
                problem["how_fixed"] = "\n\n".join(all_how_fixed)

            if not problem.get("why_fix_works") and all_why_works:
                problem["why_fix_works"] = "\n\n".join(all_why_works)
            elif problem.get("why_fix_works") and all_why_works:
                # LLM provided some, but ensure all candidate reasons are included
                llm_why_works = problem.get("why_fix_works", "").strip()
                if llm_why_works not in all_why_works:
                    all_why_works.insert(0, llm_why_works)
                problem["why_fix_works"] = "\n\n".join(all_why_works)

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

        # Log file frequency coverage
        if file_analysis["high_frequency_count"] > 0:
            selected_files = set()
            for p in selected:
                selected_files.update(p.get("files", []))

            covered_count = sum(
                1
                for f in file_analysis["high_frequency_files"].keys()
                if f in selected_files
            )
            logger.info(
                f"[L2 Consecutive] High-frequency file coverage: "
                f"{covered_count}/{file_analysis['high_frequency_count']}"
            )
    return selected


def _merge_and_deduplicate(
    common_problems: List[Dict[str, Any]],
    consecutive_problems: List[Dict[str, Any]],
    llm: Any = None,
) -> List[Dict[str, Any]]:
    """
    Merge common and consecutive problems using semantic groups.

    The old exact-key rule (root_cause + how_fixed) was too brittle. This path
    first builds small similarity groups from structured fields and embeddings,
    then asks the LLM to decide whether each group should be merged, kept
    separate, or reduced to a representative. If the LLM is unavailable or
    returns invalid output, a conservative fallback only merges near-duplicates.
    """
    problems = [
        {**problem, "_dedupe_source": "common"}
        for problem in common_problems
        if isinstance(problem, dict)
    ] + [
        {**problem, "_dedupe_source": "consecutive"}
        for problem in consecutive_problems
        if isinstance(problem, dict)
    ]

    if not problems:
        return []

    groups = _group_similar_selected_problems(problems)
    merged: List[Dict[str, Any]] = []

    for group in groups:
        if len(group) == 1:
            merged.append(_strip_dedupe_metadata(group[0]))
            continue

        decided = _llm_decide_problem_group(group, llm)
        if decided is None:
            decided = _fallback_merge_problem_group(group)

        merged.extend(_strip_dedupe_metadata(problem) for problem in decided)

    logger.info(
        f"[L2 Merge] {len(common_problems)} common + {len(consecutive_problems)} consecutive "
        f"→ {len(merged)} distinct problems across {len(groups)} semantic groups "
        f"(removed {len(problems) - len(merged)} duplicates/merged items)"
    )

    return merged


def _dedupe_signature(problem: Dict[str, Any]) -> str:
    files = " ".join(str(path) for path in _as_list(problem.get("files")) if path)
    return " | ".join(
        text
        for text in [
            _clean_text(problem.get("validation_cmd"), 300),
            _clean_text(problem.get("issue_type"), 200),
            _clean_text(problem.get("failure_type"), 200),
            _clean_text(files, 500),
            _clean_text(problem.get("problem") or problem.get("description"), 700),
            _clean_text(problem.get("root_cause"), 700),
            _clean_text(problem.get("how_fixed") or problem.get("fix"), 700),
        ]
        if text
    )


def _selected_problem_similarity(
    left: Dict[str, Any],
    right: Dict[str, Any],
    left_embedding: Any,
    right_embedding: Any,
) -> float:
    if np is not None and left_embedding is not None and right_embedding is not None:
        semantic = float(np.dot(left_embedding, right_embedding))
    else:
        semantic = 0.0

    left_files = {str(path).lower() for path in _as_list(left.get("files")) if path}
    right_files = {str(path).lower() for path in _as_list(right.get("files")) if path}
    if left_files and right_files:
        file_overlap = len(left_files & right_files) / len(left_files | right_files)
    else:
        file_overlap = 0.0

    left_cmd = _clean_text(left.get("validation_cmd")).lower()
    right_cmd = _clean_text(right.get("validation_cmd")).lower()
    if left_cmd and right_cmd:
        cmd_match = 1.0 if left_cmd == right_cmd else 0.0
    else:
        cmd_match = 0.0

    left_issue = _clean_text(left.get("issue_type")).lower()
    right_issue = _clean_text(right.get("issue_type")).lower()
    if left_issue and right_issue:
        issue_match = 1.0 if left_issue == right_issue else 0.0
    else:
        issue_match = 0.0

    left_failure = _clean_text(left.get("failure_type")).lower()
    right_failure = _clean_text(right.get("failure_type")).lower()
    failure_match = 1.0 if left_failure and left_failure == right_failure else 0.0

    return (
        0.45 * semantic
        + 0.25 * file_overlap
        + 0.15 * issue_match
        + 0.10 * cmd_match
        + 0.05 * failure_match
    )


def _group_similar_selected_problems(
    problems: List[Dict[str, Any]],
    similarity_threshold: float = 0.72,
) -> List[List[Dict[str, Any]]]:
    """Build small semantic groups for LLM merge decisions."""
    if len(problems) <= 1:
        return [problems] if problems else []

    embeddings: List[Any] = []
    if _EMBEDDING_AVAILABLE and _NUMPY_AVAILABLE and _EmbeddingProvider is not None:
        embedder = _EmbeddingProvider.get()
        for problem in problems:
            embeddings.append(embedder.embed(_dedupe_signature(problem)))
    else:
        embeddings = [None] * len(problems)

    assigned = [False] * len(problems)
    groups: List[List[Dict[str, Any]]] = []

    for i, problem in enumerate(problems):
        if assigned[i]:
            continue
        group = [problem]
        assigned[i] = True

        for j in range(i + 1, len(problems)):
            if assigned[j]:
                continue
            similarity = _selected_problem_similarity(
                problem,
                problems[j],
                embeddings[i],
                embeddings[j],
            )
            if similarity >= similarity_threshold:
                group.append(problems[j])
                assigned[j] = True

        groups.append(group)

    return groups


def _llm_decide_problem_group(
    group: List[Dict[str, Any]], llm: Any
) -> Optional[List[Dict[str, Any]]]:
    if llm is None or len(group) <= 1:
        return None

    compact_group = []
    for index, problem in enumerate(group, 1):
        compact_group.append(
            {
                "index": index,
                "source": problem.get("_dedupe_source", ""),
                "validation_cmd": problem.get("validation_cmd", ""),
                "failure_type": problem.get("failure_type", ""),
                "issue_type": problem.get("issue_type", ""),
                "files": _as_list(problem.get("files"))[:15],
                "problem": _clean_text(
                    problem.get("problem") or problem.get("description"), 500
                ),
                "root_cause": _clean_text(problem.get("root_cause"), 500),
                "how_fixed": _clean_text(
                    problem.get("how_fixed") or problem.get("fix"), 500
                ),
                "why_fix_works": _clean_text(problem.get("why_fix_works"), 300),
            }
        )

    prompt = f"""Decide whether these similar CI repair problems should be merged.

PROBLEMS:
{json.dumps(compact_group, indent=2)}

TASK:
Return STRICT JSON with one action:
- MERGE: same underlying validation problem/fix intent. Return one merged problem.
- KEEP_SEPARATE: related but distinct problems. Return the original problems that should remain.
- REMOVE_DUPLICATE: one or more entries duplicate another. Return the best representative problem(s).

Use validation_cmd, issue_type, affected files, root_cause, fix strategy, and semantic problem meaning.
Do not merge problems only because root_cause or how_fixed are generic.

OUTPUT JSON:
[
  {{
    "action": "MERGE|KEEP_SEPARATE|REMOVE_DUPLICATE",
    "validation_cmd": "exact command",
    "failure_type": "category",
    "issue_type": "type",
    "files": ["file paths"],
    "problem": "description",
    "root_cause": "why",
    "how_fixed": "what to change",
    "why_fix_works": "why it works"
  }}
]
"""
    response = _call_llm(llm, prompt, timeout=60, max_retries=1)
    if not response or not response.strip():
        logger.warning("[L2 Group Merge] LLM returned empty response")
        return None

    selected = _parse_json_array(response)
    if not selected:
        return None

    normalized = []
    for problem in selected:
        if not any(
            problem.get(key) for key in ("problem", "root_cause", "how_fixed", "files")
        ):
            continue
        normalized.append(
            {
                "validation_cmd": problem.get("validation_cmd", ""),
                "failure_type": problem.get("failure_type", ""),
                "issue_type": problem.get("issue_type", ""),
                "files": _dedupe_keep_order(_as_list(problem.get("files")), 20),
                "problem": problem.get("problem", ""),
                "root_cause": problem.get("root_cause", ""),
                "how_fixed": problem.get("how_fixed") or problem.get("fix", ""),
                "why_fix_works": problem.get("why_fix_works", ""),
            }
        )

    return normalized or None


def _fallback_merge_problem_group(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Conservative fallback when LLM cannot decide.

    It merges only when every grouped problem shares validation command or issue
    type and has overlapping files. Otherwise it keeps the group separate.
    """
    if len(group) <= 1:
        return group

    commands = {
        _clean_text(problem.get("validation_cmd")).lower()
        for problem in group
        if problem.get("validation_cmd")
    }
    issue_types = {
        _clean_text(problem.get("issue_type")).lower()
        for problem in group
        if problem.get("issue_type")
    }
    file_sets = [
        {str(path).lower() for path in _as_list(problem.get("files")) if path}
        for problem in group
    ]
    non_empty_file_sets = [files for files in file_sets if files]
    has_file_overlap = bool(
        non_empty_file_sets and set.intersection(*non_empty_file_sets)
    )
    same_command = bool(commands) and len(commands) == 1
    same_issue_type = bool(issue_types) and len(issue_types) == 1
    same_cmd_or_issue = same_command or same_issue_type

    if not (has_file_overlap and same_cmd_or_issue):
        return group

    representative = group[0]
    return [
        {
            **representative,
            "files": _dedupe_keep_order(
                file_path
                for problem in group
                for file_path in _as_list(problem.get("files"))
            ),
            "problem": "\n".join(
                _dedupe_keep_order(
                    problem.get("problem") or problem.get("description")
                    for problem in group
                )
            ),
            "root_cause": "\n".join(
                _dedupe_keep_order(problem.get("root_cause") for problem in group)
            ),
            "how_fixed": "\n".join(
                _dedupe_keep_order(
                    problem.get("how_fixed") or problem.get("fix") for problem in group
                )
            ),
            "why_fix_works": "\n".join(
                _dedupe_keep_order(problem.get("why_fix_works") for problem in group)
            ),
        }
    ]


def _strip_dedupe_metadata(problem: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(problem)
    cleaned.pop("_dedupe_source", None)
    return cleaned


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
        "problem": _clean_text(
            problem.get("description") or problem.get("problem"), 900
        ),
        "root_cause": _clean_text(problem.get("root_cause"), 900),
        "how_fixed": _clean_text(problem.get("how_fixed") or problem.get("fix"), 900),
        "why_fix_works": _clean_text(problem.get("why_fix_works"), 700),
        "files": _dedupe_keep_order(_as_list(problem.get("files")), 20),
    }


def _extract_consecutive_from_l2_after_match(
    problem_1: Dict[str, Any],
    l2_memories: List[Dict[str, Any]],
    similarity_threshold: float = 0.6,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extract consecutive problems from L2 records by finding where they match
    the current CI failure, then extracting everything AFTER that match.

    The matching problem can be at ANY step in the L2 trajectory (not just primary).
    Once a match is found, all subsequent steps are extracted as consecutive problems.

    Args:
        problem_1: Current CI failure (may contain multiple sub-problems)
        l2_memories: All L2 repair trajectories
        similarity_threshold: Cosine similarity threshold (0.6 = moderate)

    Returns:
        Tuple of (consecutive_problem_rows, number_of_matched_l2_records)
    """
    if not _EMBEDDING_AVAILABLE or not _NUMPY_AVAILABLE or not l2_memories:
        logger.warning(
            "[L2 Consecutive Extract] Embeddings not available, returning empty"
        )
        return [], 0

    # Type guards
    assert np is not None
    assert _EmbeddingProvider is not None

    embedder = _EmbeddingProvider.get()

    # Embed current CI failure (may have multiple aspects)
    ci_signatures = []

    # Overall description and root cause
    ci_desc = problem_1.get("description") or problem_1.get("problem", "")
    ci_root = problem_1.get("root_cause", "")
    if ci_desc or ci_root:
        ci_signatures.append(f"{ci_desc} | {ci_root}".strip())

    # Individual error details (sub-problems)
    ci_errors = problem_1.get("error_details", [])
    for error in ci_errors if isinstance(ci_errors, list) else []:
        if isinstance(error, dict):
            err_msg = error.get("message", "")
            err_file = error.get("file", "")
            if err_msg:
                ci_signatures.append(f"{err_file} {err_msg}".strip())
        elif isinstance(error, str):
            ci_signatures.append(error.strip())

    # Embed all CI signatures
    ci_embeddings = []
    for sig in ci_signatures:
        if sig and sig != "|":
            emb = embedder.embed(sig)
            if emb is not None:
                ci_embeddings.append(emb)

    if not ci_embeddings:
        logger.warning("[L2 Consecutive Extract] Failed to embed current problem")
        return [], 0

    all_consecutive_rows = []
    matched_l2_count = 0

    # For each L2 record, find where it matches current CI
    for l2_memory in l2_memories:
        trajectory = (
            l2_memory.get("repair_trajectory")
            or l2_memory.get("trajectory")
            or l2_memory.get("problems")
            or []
        )

        if not isinstance(trajectory, list) or not trajectory:
            continue

        issue_id = l2_memory.get("issue_id", "unknown")
        matching_step_index = None
        best_similarity = 0.0

        # Search through ALL steps to find a match
        for step_index, step in enumerate(trajectory):
            if not isinstance(step, dict):
                continue

            step_problems = _step_problems(step)
            if not step_problems:
                continue

            # Check each problem in this step
            for problem in step_problems:
                l2_desc = problem.get("problem") or problem.get("description", "")
                l2_root = problem.get("root_cause", "")
                l2_signature = f"{l2_desc} | {l2_root}".strip()

                if not l2_signature or l2_signature == "|":
                    continue

                l2_embedding = embedder.embed(l2_signature)
                if l2_embedding is None:
                    continue

                # Check similarity with ANY CI sub-problem
                for ci_embedding in ci_embeddings:
                    similarity = float(np.dot(ci_embedding, l2_embedding))

                    if (
                        similarity >= similarity_threshold
                        and similarity > best_similarity
                    ):
                        best_similarity = similarity
                        matching_step_index = step_index

        # If match found, extract all subsequent steps
        if matching_step_index is not None:
            matched_l2_count += 1
            logger.debug(
                f"[L2 Consecutive Extract] Matched L2 {issue_id} at step {matching_step_index + 1} "
                f"(similarity: {best_similarity:.3f})"
            )

            # Extract consecutive problems (everything AFTER the match)
            for step_index in range(matching_step_index + 1, len(trajectory)):
                step = trajectory[step_index]
                if not isinstance(step, dict):
                    continue

                for problem in _step_problems(step):
                    problem_text = _clean_text(
                        problem.get("problem") or problem.get("description"), 700
                    )
                    root_cause = _clean_text(problem.get("root_cause"), 700)
                    how_fixed = _clean_text(
                        problem.get("how_fixed") or problem.get("fix"), 700
                    )
                    why_fix_works = _clean_text(problem.get("why_fix_works"), 500)

                    if not any((problem_text, root_cause, how_fixed, why_fix_works)):
                        continue

                    all_consecutive_rows.append(
                        {
                            "row_id": len(all_consecutive_rows) + 1,
                            "issue_id": issue_id,
                            "repo": l2_memory.get("repo", ""),
                            "step": step_index + 1,
                            "step_index": step_index + 1,
                            "is_primary_step": False,  # All consecutive
                            "matched_at_step": matching_step_index + 1,
                            "match_similarity": best_similarity,
                            "validation_cmd": _clean_text(
                                step.get("validation_cmd")
                                or l2_memory.get("validation_cmd")
                            ),
                            "failure_type": _clean_text(
                                step.get("failure_type")
                                or l2_memory.get("failure_type")
                            ),
                            "issue_type": _clean_text(
                                problem.get("issue_type") or step.get("issue_type"), 160
                            ),
                            "problem": problem_text,
                            "root_cause": root_cause,
                            "how_fixed": how_fixed,
                            "why_fix_works": why_fix_works,
                            "files": _dedupe_keep_order(
                                _as_list(
                                    problem.get("files")
                                    or problem.get("affected_files")
                                ),
                                20,
                            ),
                        }
                    )

    logger.info(
        f"[L2 Consecutive Extract] Found {len(all_consecutive_rows)} consecutive problems "
        f"from {matched_l2_count}/{len(l2_memories)} matched L2 records "
        f"(similarity >= {similarity_threshold})"
    )

    return all_consecutive_rows, matched_l2_count


def staged_l2_analysis(
    problem_1: Dict[str, Any],
    l2_memories: List[Dict[str, Any]],
    llm: Any,
    model_name: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Analyze fetched L2 trajectories using embedding-based clustering.

    Steps:
    1. Filter L2 by similarity to current CI failure (for consecutive problems)
    2. Flatten all L2 trajectories into problem rows
    3. Cluster problems by embedding similarity (problem + root_cause)
    4. Filter by frequency (>= threshold % of L2 records, not total problems)
    5. Pass high-frequency problems to LLM for final selection

    Frequency is L2-based: counts how many unique L2 records contain the problem,
    not total problem instances. This identifies problems that appear across
    multiple different CI issues.

    Args:
        problem_1: Current CI failure
        l2_memories: List of L2 trajectory memories
        llm: LLM client for selection
        model_name: Model name for determining candidate limits (None = auto-detect from llm)
    """
    if not l2_memories:
        logger.info("[L2] No L2 memories provided")
        return []

    # Auto-detect model_name from llm if not provided
    if model_name is None:
        model_name = _extract_model_name_from_llm(llm)
        if model_name:
            logger.info(f"[L2] Auto-detected model_name='{model_name}' from LLM object")

    # Extract ALL problems for common selection (from ALL L2 records)
    all_rows, total_trajectories = _flatten_l2(l2_memories)
    if not all_rows:
        logger.info("[L2] No trajectory problems found")
        return []

    logger.info(
        f"[L2] Extracted {len(all_rows)} problems from {total_trajectories} trajectories"
    )

    # Use embedding-based clustering to find common problems (from ALL L2)
    common_candidates = _cluster_problems_by_embedding(
        all_rows,
        total_trajectories=total_trajectories,
        similarity_threshold=0.6,
        frequency_threshold=0.3,
    )

    if not common_candidates:
        logger.warning(
            "[L2] No common problems found with embedding clustering, falling back to old method"
        )
        # Fallback to old grouping method if clustering fails
        common_candidates = _group_candidate_rows(
            all_rows,
            total_trajectories=total_trajectories,
            prefix="C",
            include_primary=True,
        )

    # Extract consecutive problems by finding matches in L2 and taking what comes after
    # The match can be at ANY step in the L2 trajectory (not just primary)
    consecutive_rows, matched_l2_count = _extract_consecutive_from_l2_after_match(
        problem_1, l2_memories, similarity_threshold=0.6
    )

    if consecutive_rows:
        consecutive_candidates = _group_candidate_rows(
            consecutive_rows,
            total_trajectories=matched_l2_count,
            prefix="N",
            include_primary=False,
        )
    else:
        logger.warning("[L2] No matching L2 records found for consecutive problems")
        consecutive_candidates = []

    logger.info(
        "[L2] Prepared candidates: common=%d (from %d L2), consecutive=%d (from %d matched L2)",
        len(common_candidates),
        total_trajectories,
        len(consecutive_candidates),
        matched_l2_count,
    )

    # Select common and consecutive problems SEPARATELY
    common_problems = _llm_select_common_problems(common_candidates, llm, model_name)
    consecutive_problems = _llm_select_consecutive_problems(
        problem_1, consecutive_candidates, llm, model_name
    )

    # Merge and deduplicate
    merged = _merge_and_deduplicate(common_problems, consecutive_problems, llm)

    # Validate file frequency coverage (safeguard)
    # This ensures high-frequency files (2+ occurrences) are not missed
    all_candidates = common_candidates + consecutive_candidates
    merged = _validate_file_frequency_coverage(
        merged,
        all_candidates,
        min_frequency_threshold=2,  # Force-add files appearing 2+ times (lowered from 3)
    )

    # Normalize to final format
    normalized = [
        _normalize_selected_problem(problem, index)
        for index, problem in enumerate(merged, 1)
    ]

    logger.info(
        "[L2] Final: %d problems (%d common, %d consecutive, merged and deduplicated)",
        len(normalized),
        len(common_problems),
        len(consecutive_problems),
    )
    return normalized
