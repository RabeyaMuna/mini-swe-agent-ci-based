#!/usr/bin/env python3
"""
Build structured problem payloads for OpenHands API from CI-Bench data
"""

from typing import Any, Optional


def build_problem_from_decomposed(
    decomposed_problem: dict[str, Any],
    issue_data: dict[str, Any],
    problem_index: int = 1,
) -> dict[str, Any]:
    """
    Convert decomposed CI problem into OpenHands API problem format

    Args:
        decomposed_problem: Problem from decomposed_issues.json
        issue_data: Full issue data with repo, sha_fail, etc.
        problem_index: Index of this problem in sequence

    Returns:
        Structured problem dict for OpenHands API
    """
    problem_id = f"{issue_data.get('instance_id', 'unknown')}_p{problem_index}"

    # Extract fields from decomposed problem
    summary = str(decomposed_problem.get("problem", "CI failure"))[:200]
    affected_files = decomposed_problem.get("affected_files", [])
    if isinstance(affected_files, str):
        affected_files = [affected_files]

    validation_cmd = (
        decomposed_problem.get("validation_cmd")
        or decomposed_problem.get("validation_command")
        or ""
    )

    # Build reproduction steps
    repo_url = f"https://github.com/{issue_data['repo']}"
    sha_fail = issue_data["sha_fail"]

    reproduction = [
        f"git clone {repo_url} repo",
        "cd repo",
        f"git fetch origin {sha_fail}",
        f"git checkout {sha_fail}",
    ]

    # Add validation as reproduction step if present
    if validation_cmd:
        reproduction.append(validation_cmd)

    # Build root causes
    root_causes = []
    root_cause_text = decomposed_problem.get("root_cause", "")
    if root_cause_text:
        root_causes.append(
            {
                "id": "rc1",
                "description": root_cause_text,
                "evidence": decomposed_problem.get("problem", ""),
                "confidence": 0.8,
            }
        )

    # Build suggested fixes
    suggested_fixes = []
    how_fixed = (
        decomposed_problem.get("how_fixed")
        or decomposed_problem.get("fixes")
        or decomposed_problem.get("changes_made")
        or ""
    )
    if how_fixed:
        suggested_fixes.append(
            {
                "id": "sf1",
                "description": how_fixed,
                "files_to_change": affected_files,
                "confidence": 0.8,
                "est_minutes": 15,
            }
        )

    # Build fix strategy
    fix_strategy = {
        "mode": "conservative",
        "steps": [
            {"name": "checkout", "cmd": f"git checkout {sha_fail}"},
            {"name": "reproduce", "cmd": validation_cmd if validation_cmd else ""},
            {"name": "apply_fix", "cmd": "edit files based on suggested_fixes"},
            {"name": "validate", "cmd": validation_cmd if validation_cmd else ""},
        ],
        "verify_after_each_change": True,
    }

    # Logs snippet
    logs_snippet = decomposed_problem.get("problem", "")[:500]

    # Build problem
    problem = {
        "problem_id": problem_id,
        "summary": summary,
        "severity": "high",
        "priority": 100 - (problem_index * 10),  # Earlier problems have higher priority
        "repo": repo_url,
        "sha_fail": sha_fail,
        "reproduction": reproduction,
        "relevant_files": affected_files,
        "logs_snippet": logs_snippet,
        "validation": [validation_cmd] if validation_cmd else [],
        "time_budget_minutes": 30,
        "allow_auto_apply": False,
        "root_causes": root_causes if root_causes else [],
        "suggested_fixes": suggested_fixes if suggested_fixes else [],
        "fix_strategy": fix_strategy,
    }

    return problem


def build_problem_from_ci_failure(
    issue_data: dict[str, Any], baseline_mode: bool = False
) -> dict[str, Any]:
    """
    Build problem from raw CI failure data (baseline mode)

    Args:
        issue_data: Issue data with ci_failure, problem_statement, etc.
        baseline_mode: If True, minimal problem with no guidance

    Returns:
        Structured problem dict
    """
    problem_id = issue_data.get("instance_id", "unknown")
    repo_url = f"https://github.com/{issue_data['repo']}"
    sha_fail = issue_data["sha_fail"]

    # Get CI failure info
    ci_failure = issue_data.get("ci_failure", {})
    error_messages = ci_failure.get("error_messages", [])
    if isinstance(error_messages, list):
        logs_snippet = "\n".join(error_messages[:5])
    else:
        logs_snippet = str(error_messages)[:500]

    # Fallback to problem_statement if no logs
    if not logs_snippet:
        logs_snippet = issue_data.get("problem_statement", "CI failure")[:500]

    # Get faulty files from ci_failure
    faulty_files = ci_failure.get("faulty_files", [])
    if not faulty_files:
        # Try to extract from problem_statement
        from fault_localization import extract_faulty_files

        faulty_locations = extract_faulty_files(
            issue_data.get("problem_statement", "")
        )
        faulty_files = [
            loc.get("file_path") for loc in faulty_locations if loc.get("file_path")
        ]

    # Reproduction steps
    reproduction = [
        f"git clone {repo_url} repo",
        "cd repo",
        f"git fetch origin {sha_fail}",
        f"git checkout {sha_fail}",
    ]

    # Add validation command
    validation_cmd = issue_data.get("validation_command", "")
    workflow = issue_data.get("workflow", {})
    if not validation_cmd and workflow:
        validation_cmd = workflow.get("validation_command", "")

    if validation_cmd:
        reproduction.append(validation_cmd)

    problem = {
        "problem_id": problem_id,
        "summary": issue_data.get("problem_statement", "CI failure")[:200],
        "severity": "high",
        "priority": 100,
        "repo": repo_url,
        "sha_fail": sha_fail,
        "reproduction": reproduction,
        "relevant_files": faulty_files,
        "logs_snippet": logs_snippet,
        "validation": [validation_cmd] if validation_cmd else [],
        "time_budget_minutes": 60,
        "allow_auto_apply": False,
    }

    # In baseline mode, don't add root_causes or suggested_fixes
    if not baseline_mode:
        # Add memory-based hints if available
        ci_failure_summary = issue_data.get("ci_failure_summary", "")
        if ci_failure_summary:
            problem["root_causes"] = [
                {
                    "id": "rc1",
                    "description": "CI failure detected",
                    "evidence": ci_failure_summary[:300],
                    "confidence": 0.7,
                }
            ]

    return problem


def build_problems_from_issue(
    issue_data: dict[str, Any],
    decomposed_issue: Optional[dict[str, Any]] = None,
    mode: str = "baseline",
) -> list[dict[str, Any]]:
    """
    Build all problems for an issue

    Args:
        issue_data: Full issue data
        decomposed_issue: Optional decomposed problems
        mode: 'baseline' or 'memory'

    Returns:
        List of structured problems
    """
    problems = []

    if mode == "baseline":
        # Baseline: single problem, no decomposition
        problem = build_problem_from_ci_failure(issue_data, baseline_mode=True)
        problems.append(problem)

    elif decomposed_issue and decomposed_issue.get("problems"):
        # Memory with decomposition: multiple problems
        decomposed_problems = decomposed_issue.get("problems", [])
        for idx, decomposed_problem in enumerate(decomposed_problems, 1):
            problem = build_problem_from_decomposed(
                decomposed_problem, issue_data, problem_index=idx
            )
            problems.append(problem)

    else:
        # Memory without decomposition: single problem with hints
        problem = build_problem_from_ci_failure(issue_data, baseline_mode=False)
        problems.append(problem)

    return problems


if __name__ == "__main__":
    # Test
    import json

    test_issue = {
        "instance_id": "test-001",
        "repo": "adap/flower",
        "sha_fail": "6aee1d58",
        "problem_statement": "Fix mypy errors",
        "validation_command": "cd py && mypy flwr/",
    }

    test_decomposed = {
        "problems": [
            {
                "problem_id": 1,
                "problem": "Missing return type annotation",
                "root_cause": "Function lacks -> None",
                "how_fixed": "Add -> None to function signature",
                "affected_files": ["py/flwr/common/inflatable_test.py"],
                "validation_cmd": "mypy flwr/common/inflatable_test.py",
            }
        ]
    }

    problems = build_problems_from_issue(test_issue, test_decomposed, mode="memory")
    print(json.dumps(problems[0], indent=2))
