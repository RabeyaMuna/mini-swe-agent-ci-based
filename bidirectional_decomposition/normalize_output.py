"""
Normalize bidirectional decomposition output to match backward decomposition structure.

This ensures compatibility with memory building and consistent field names.
"""
from typing import Any, Dict, List
from collections import defaultdict


def _dependency_order(
    problems: List[Dict[str, Any]],
    dependencies: List[Dict[str, Any]],
    requested_sequence: List[int],
) -> List[int]:
    """Return a complete topological sequence; dependencies outrank CI order."""
    ids = [p.get("problem_id") for p in problems if p.get("problem_id") is not None]
    valid = set(ids)
    requested_rank = {pid: index for index, pid in enumerate(requested_sequence or [])}
    original_rank = {pid: index for index, pid in enumerate(ids)}
    outgoing: Dict[int, set[int]] = defaultdict(set)
    indegree = {pid: 0 for pid in ids}
    for edge in dependencies:
        source, target = edge.get("from"), edge.get("to")
        if source in valid and target in valid and source != target and target not in outgoing[source]:
            outgoing[source].add(target)
            indegree[target] += 1

    def key(pid: int) -> tuple[int, int]:
        return (requested_rank.get(pid, 10_000), original_rank.get(pid, 10_000))

    ready = sorted((pid for pid in ids if indegree[pid] == 0), key=key)
    ordered: List[int] = []
    while ready:
        source = ready.pop(0)
        ordered.append(source)
        for target in sorted(outgoing[source], key=key):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=key)

    # Preserve all problems if malformed model output contains a cycle.
    ordered.extend(sorted(valid - set(ordered), key=key))
    return ordered


def normalize_problem_structure(problem: Dict[str, Any], sequence_index: int) -> Dict[str, Any]:
    """
    Normalize a bidirectional problem to match backward decomposition structure.

    Backward structure:
    {
      "problem_id": 1,
      "problem_type": "primary",    # "primary" if from CI logs, "hidden" if inferred
      "is_cascading": false,
      "dependency_type": "",
      "cascade_explanation": "",
      "validation_cmd": "...",
      "validation_order": 5,
      "repair_sequence_index": 1,
      "affected_files": [...],
      "problem": "...",
      "root_cause": "...",
      "how_fixed": "...",
      "why_fix_works": "..."
    }
    """
    # Extract ci_analysis if present
    ci_analysis = problem.get("ci_analysis", {})

    # Determine problem_type: primary if CI-relevant, hidden otherwise
    ci_relevance = ci_analysis.get("ci_relevance", "")
    explains_jobs = ci_analysis.get("explains_jobs", [])

    # Primary if:
    # - Has high/medium CI relevance, OR
    # - Explains failed jobs, OR
    # - Already marked as primary, OR
    # - Has ci_analysis (means it was analyzed against CI)
    is_ci_relevant = (
        ci_relevance in ["high", "medium"] or
        len(explains_jobs) > 0 or
        problem.get("problem_type") == "primary" or
        bool(ci_analysis)
    )

    problem_type = "primary" if is_ci_relevant else "hidden"

    # Build normalized problem
    normalized = {
        "problem_id": problem.get("problem_id", 0),

        # Core fields from backward structure
        "problem_type": problem_type,  # Classify based on CI relevance
        "is_cascading": problem.get("is_cascading", False),
        "dependency_type": problem.get("dependency_type", ""),
        "cascade_explanation": problem.get("cascade_explanation", ""),

        # Validation fields (map from different names)
        "validation_cmd": (
            problem.get("validation_cmd") or
            problem.get("verification_cmd") or
            ""
        ),
        "validation_order": (
            problem.get("validation_order") or
            ci_analysis.get("step_order") or
            sequence_index
        ),
        "repair_sequence_index": sequence_index,

        # File and description fields
        "affected_files": problem.get("affected_files", problem.get("files", [])),
        "problem": problem.get("problem", ""),
        "root_cause": problem.get("root_cause", ""),
        "how_fixed": problem.get("how_fixed", ""),
        "why_fix_works": problem.get("why_fix_works", ""),

        # Classification
        "failure_type": problem.get("failure_type", ""),
        "issue_type": problem.get("issue_type", ""),

        # Bidirectional reconciliation provenance
        "source": problem.get("source", ""),
        "source_ids": problem.get("source_ids", []),
        "merge_reason": problem.get("merge_reason", ""),
    }

    # Remove any fields with None values
    return {k: v for k, v in normalized.items() if v is not None}


def normalize_bidirectional_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize entire bidirectional result to match backward structure.

    Args:
        result: Bidirectional decomposition result

    Returns:
        Normalized result matching backward decomposition format
    """
    problems = result.get("problems", [])
    dependencies = result.get("dependencies", [])
    repair_sequence = _dependency_order(
        problems,
        dependencies,
        result.get("repair_sequence", []),
    )

    # Create mapping from problem_id to sequence index
    sequence_map = {
        pid: idx + 1
        for idx, pid in enumerate(repair_sequence)
    }

    # Build dependency graph: problem_id -> list of enabled problem_ids
    enabled_map = {}  # {from_id: [to_id1, to_id2, ...]}
    dependency_types = {}  # {from_id: dependency_type}

    for dep in dependencies:
        from_id = dep.get("from")
        to_id = dep.get("to")
        dep_type = dep.get("dependency_type", "")

        if from_id and to_id:
            if from_id not in enabled_map:
                enabled_map[from_id] = []
            enabled_map[from_id].append(to_id)

            # Store dependency type (use first one if multiple)
            if from_id not in dependency_types:
                dependency_types[from_id] = dep_type

    # Normalize each problem
    normalized_problems = []
    for problem in problems:
        problem_id = problem.get("problem_id", 0)
        sequence_index = sequence_map.get(problem_id, len(normalized_problems) + 1)

        normalized = normalize_problem_structure(problem, sequence_index)

        # Add enabled list and dependency info from graph
        enabled_list = enabled_map.get(problem_id, [])
        normalized["enabled"] = enabled_list if enabled_list else []
        normalized["is_cascading"] = len(enabled_list) > 0

        # Set dependency_type if this problem enables others
        if problem_id in dependency_types:
            normalized["dependency_type"] = dependency_types[problem_id]

        normalized_problems.append(normalized)

    # Sort by repair_sequence_index
    normalized_problems.sort(key=lambda p: p.get("repair_sequence_index", 999))

    # Build result in backward format
    return {
        "original_issue_id": result.get("original_issue_id", ""),
        "issue_id": result.get("original_issue_id", ""),  # Duplicate for compatibility
        "sha_fail": result.get("sha_fail", ""),
        "repo": result.get("repo", ""),
        "repo_owner": result.get("repo_owner", ""),
        "repo_name": result.get("repo_name", ""),
        "workflow_path": result.get("workflow_path", ""),
        "workflow_name": result.get("workflow_name", ""),
        "original_error_type": result.get("original_error_type", []),

        # Problems in normalized structure
        "problems": normalized_problems,

        # Dependencies and repair sequence (critical for memory building!)
        "dependencies": result.get("dependencies", []),
        "repair_sequence": repair_sequence,

        # Reconciliation diagnostics
        "reconciliation_metadata": result.get("reconciliation_metadata", {}),

        # Summary fields
        "total_problems": len(normalized_problems),
        "total_changed_files": result.get("total_changed_files", 0),
        "changed_files": result.get("changed_files", []),

        # Keep CI context for reference
        "benchmark_ci_context": result.get("benchmark_ci_context"),
        "diff_analysis_context": result.get("diff_analysis_context"),
    }
