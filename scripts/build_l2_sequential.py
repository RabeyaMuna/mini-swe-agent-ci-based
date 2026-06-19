"""
Clean L2 builder - Transform decomposed data to simple, actionable format.

NO LLM calls - pure transformation.
NO duplication - each piece of info once.
NO over-nesting - flat and clear.

Output format:
{
  "atomic_problems": [
    {
      "problem_id": 1,
      "verification_cmd": "python -m mypy py",
      "files": ["a.py", "b.py"],
      "problem": {
        "what_wrong": "Uses private numpy API",
        "root_cause": "Importing from underscore-prefixed private module"
      },
      "fixes": [
        {
          "from": "from numpy.typing import DTypeLike, NDArray",
          "to": "from numpy.typing import NDArray",
          "why_this_fix": "DTypeLike is private API, must be removed"
        }
      ]
    }
  ]
}
"""

from typing import Dict, Any, List


def build_l2_with_sequential_reasoning(
    issue: Dict[str, Any],
    llm=None,
    verified_patch: str = None
) -> Dict[str, Any]:
    """
    Transform decomposed issue to clean L2 memory structure.

    Args:
        issue: Decomposed issue data
        llm: Not used (kept for compatibility)
        verified_patch: Not used (kept for compatibility)

    Returns:
        Clean L2 memory structure
    """

    # Get atomic problems from decomposition
    problems = issue.get("atomic_problems") or issue.get("problems", [])

    if not problems:
        return {}

    # Transform each problem to clean L2 format
    l2_problems = []
    for i, problem in enumerate(problems, 1):
        l2_problem = _transform_problem_to_l2(problem, i)
        if l2_problem:
            l2_problems.append(l2_problem)

    return {
        "atomic_problems": l2_problems
    }


def _transform_problem_to_l2(problem: Dict[str, Any], problem_id: int) -> Dict[str, Any]:
    """
    Transform one atomic problem to clean L2 format.

    Output structure:
    {
      "problem_id": 1,
      "verification_cmd": "...",
      "issue_type": "...",  // For L3 compatibility
      "files": [...],
      "problem": {"what_wrong": "...", "root_cause": "..."},
      "fixes": [{"from": "...", "to": "...", "why_this_fix": "..."}]
    }
    """

    # Verification command
    verification_cmd = (
        problem.get("validation_cmd") or
        problem.get("ci_command") or
        problem.get("failed_cmd") or
        ""
    )

    # Issue type (for L3 grouping/clustering)
    issue_type = (
        problem.get("failure_type") or
        problem.get("issue_type") or
        problem.get("problem_type") or
        ""
    )

    # Files affected
    files = _extract_files(problem)

    if not files:
        return None

    # Problem description
    what_wrong = (
        problem.get("problem") or
        problem.get("symptom") or
        ""
    )

    root_cause = (
        problem.get("root_cause") or
        problem.get("why") or
        ""
    )

    # Extract fixes
    fixes = _extract_fixes(problem)

    if not fixes:
        return None

    # Build clean L2 problem
    l2_problem = {
        "problem_id": problem_id,
        "verification_cmd": verification_cmd,
        "files": files,
        "problem": {
            "what_wrong": what_wrong,
            "root_cause": root_cause
        },
        "fixes": fixes
    }

    # Add issue_type for L3 compatibility (pattern extraction)
    if issue_type:
        l2_problem["issue_type"] = issue_type

    return l2_problem


def _extract_files(problem: Dict[str, Any]) -> List[str]:
    """Extract file list from problem, handling multiple formats."""

    files = []

    # Try different field names
    file_sources = [
        problem.get("files"),
        problem.get("affected_files"),
        problem.get("file_changes"),
    ]

    for source in file_sources:
        if not source:
            continue

        # Handle different formats
        if isinstance(source, list):
            for item in source:
                if isinstance(item, str):
                    files.append(item)
                elif isinstance(item, dict):
                    file_path = item.get("file", "")
                    if file_path:
                        files.append(file_path)

        elif isinstance(source, dict):
            # Dict with files list: {"files": [...], "total_count": N}
            nested_files = source.get("files", [])
            for f in nested_files:
                if isinstance(f, str):
                    files.append(f)
                elif isinstance(f, dict):
                    file_path = f.get("file", "")
                    if file_path:
                        files.append(file_path)

        if files:
            break

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f and f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


def _extract_fixes(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract fixes from problem.

    Returns list of fix objects:
    [
      {
        "from": "old code",
        "to": "new code",
        "why_this_fix": "explanation"
      }
    ]
    """

    fixes = []

    # Try to get from file_changes
    file_changes = problem.get("file_changes") or []

    if file_changes and isinstance(file_changes, list):
        for fc in file_changes:
            if not isinstance(fc, dict):
                continue

            fix = {}

            # Get change (from/to)
            change = fc.get("change")
            if change and isinstance(change, dict):
                from_code = change.get("from", "")
                to_code = change.get("to", "")
                why = change.get("why", "")

                if from_code and to_code:
                    fix["from"] = from_code
                    fix["to"] = to_code
                    if why:
                        fix["why_this_fix"] = why

            # Alternative: get from separate fields
            if not fix:
                from_code = fc.get("from", "")
                to_code = fc.get("to", "")

                if from_code and to_code:
                    fix["from"] = from_code
                    fix["to"] = to_code

                    # Get why from various fields
                    why = (
                        fc.get("why_this_fix") or
                        fc.get("why") or
                        fc.get("fix") or
                        fc.get("how_fix") or
                        ""
                    )
                    if why:
                        fix["why_this_fix"] = why

            if fix and "from" in fix and "to" in fix:
                fixes.append(fix)

    # If no fixes found in file_changes, try to build from how_fixed
    if not fixes:
        how_fixed = problem.get("how_fixed") or problem.get("fix_strategy") or ""
        root_cause = problem.get("root_cause") or problem.get("why") or ""

        if how_fixed:
            # Create generic fix entry
            fix = {
                "from": "(see problem description)",
                "to": "(see fix description)",
                "why_this_fix": how_fixed
            }

            if root_cause and "why_this_fix" not in fix:
                fix["why_this_fix"] = f"{root_cause}. Fix: {how_fixed}"

            fixes.append(fix)

    return fixes


# Backward compatibility
def build_l2_from_enhanced_decomposition(issue: Dict[str, Any], llm=None, verified_patch: str = None):
    """Alias for compatibility."""
    return build_l2_with_sequential_reasoning(issue, llm, verified_patch)
