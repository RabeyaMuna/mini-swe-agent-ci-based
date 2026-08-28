"""
Extract clean problems from forward/backward decomposition.

This extracts ONLY the problem structure needed for clustering,
dropping all heavy context fields (diffs, logs, validation groups, etc.)
"""
from typing import Any, Dict, List


def extract_problem_fields(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only essential fields from a decomposed problem.

    This creates a clean, lightweight problem structure suitable for:
    - Clustering
    - Merging
    - LLM analysis

    Drops:
    - Heavy context (diffs, logs, traces)
    - Validation groups
    - Raw error messages
    - Full stack traces
    """
    return {
        # Core identification
        "problem_id": problem.get("problem_id"),

        # Problem description
        "problem": problem.get("problem", ""),
        "root_cause": problem.get("root_cause", ""),

        # Fix details
        "how_fixed": problem.get("how_fixed", ""),
        "why_fix_works": problem.get("why_fix_works", "") or problem.get("why_fixed_works", ""),

        # Files and commands
        "affected_files": problem.get("affected_files", []) or problem.get("files", []),
        "verification_cmd": problem.get("verification_cmd", "") or problem.get("validation_cmd", ""),

        # Classification
        "failure_type": problem.get("failure_type", ""),
        "issue_type": problem.get("issue_type", ""),
        "problem_type": problem.get("problem_type", ""),

        # Metadata (lightweight)
        "is_cascading": problem.get("is_cascading", False),
        "validation_order": problem.get("validation_order"),
    }


def extract_problems_from_forward(forward_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract clean problems from forward decomposition.

    Input: Full forward decomposition (with all heavy context)
    Output: List of clean problem structures
    """
    # Get problems (handle both field names)
    problems = (
        forward_data.get("decomposed_problems") or
        forward_data.get("problems", [])
    )

    return [extract_problem_fields(p) for p in problems]


def extract_problems_from_backward(backward_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract clean problems from backward decomposition.

    Input: Full backward decomposition (with all heavy context)
    Output: List of clean problem structures
    """
    # Get problems (handle both field names)
    problems = (
        backward_data.get("decomposed_problems") or
        backward_data.get("problems", [])
    )

    return [extract_problem_fields(p) for p in problems]


def extract_ci_context_essential(ci_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract essential CI context, dropping heavy fields.

    Keeps:
    - Failed jobs (names, reasons)
    - Validation sequence
    - Workflow path

    Drops:
    - Full logs
    - Validation groups (can be 2MB+)
    - Detailed error traces
    """
    return {
        "failed_jobs": [
            {
                "job_name": job.get("job_name", ""),
                "failure_reason": str(job.get("failure_reason", ""))[:500],  # Truncate
                "error_type": job.get("error_type", ""),
            }
            for job in ci_context.get("failed_jobs", [])
        ],

        "validation_sequence": [
            step if isinstance(step, str) else step.get("name", "")
            for step in ci_context.get("validation_sequence", [])
        ],

        "workflow_path": ci_context.get("workflow_path", ""),
        "original_error_type": ci_context.get("original_error_type", []),
    }


def extract_dependencies(decomp_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract dependencies from decomposition data.

    Returns list of {from, to, type, reason} dicts
    """
    deps = decomp_data.get("dependencies") or decomp_data.get("_dependencies", [])

    return [
        {
            "from": dep.get("from"),
            "to": dep.get("to"),
            "type": dep.get("type") or dep.get("dependency_type", ""),
            "reason": str(dep.get("reason", ""))[:200],  # Truncate
        }
        for dep in deps
    ]


def prepare_for_clustering(
    forward_data: Dict[str, Any],
    backward_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Prepare forward and backward data for clustering.

    Extracts only essential fields, dropping all heavy context.

    Returns:
        {
            "forward_problems": [...],
            "backward_problems": [...],
            "ci_context": {...},
            "forward_dependencies": [...],
            "backward_dependencies": [...],
        }
    """
    return {
        "forward_problems": extract_problems_from_forward(forward_data),
        "backward_problems": extract_problems_from_backward(backward_data),
        "ci_context": extract_ci_context_essential(
            backward_data.get("benchmark_ci_context") or
            forward_data.get("benchmark_ci_context", {})
        ),
        "forward_dependencies": extract_dependencies(forward_data),
        "backward_dependencies": extract_dependencies(backward_data),
        "metadata": {
            "issue_id": forward_data.get("issue_id") or forward_data.get("original_issue_id"),
            "sha_fail": forward_data.get("sha_fail") or backward_data.get("sha_fail"),
            "repo": forward_data.get("repo") or backward_data.get("repo"),
            "workflow_path": forward_data.get("workflow_path") or backward_data.get("workflow_path"),
        }
    }
