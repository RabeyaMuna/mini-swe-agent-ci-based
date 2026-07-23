#!/usr/bin/env python3
"""
trajectory_builder.py - Build and organize repair trajectory
"""

import json
from typing import List, Dict


def build_repair_trajectory(per_commit_analysis: List[Dict]) -> List[Dict]:
    """
    Build simple sequential repair trajectory from all commits

    Args:
        per_commit_analysis: List of commit analyses

    Returns:
        Flat list of all problems in sequence
    """

    problem_sequence = []

    for commit_analysis in per_commit_analysis:
        commit_sha = commit_analysis.get("commit_sha", "unknown")

        for problem in commit_analysis.get("problems", []):
            # Simple flat structure
            problem_entry = {
                "commit_sha": problem.get("commit_sha") or commit_sha,
                "commit_message": problem.get("commit_message", ""),
                "files": problem.get("files", []),
                "failure_type": problem.get("failure_type", ""),
                "issue_type": problem.get("issue_type", ""),
                "problem": problem.get("problem", ""),
                "root_cause": problem.get("root_cause", ""),
                "changes_made": problem.get("changes_made", ""),
                "introduces": problem.get("introduces", ""),
                "fixes": problem.get("fixes", ""),
                "current_failed_jobs": problem.get("current_failed_jobs", []),
                "current_fixed_jobs": problem.get("current_fixed_jobs", []),
                "validation_cmd": problem.get("validation_cmd", ""),
                "commit": commit_sha,
                "sha_success": problem.get("sha_success"),
                "sha_fail": problem.get("sha_fail"),
            }

            problem_sequence.append(problem_entry)

    return problem_sequence


def organize_trajectory_with_llm(
    all_problems: List[Dict], validation_sequence: List[Dict], analyzer
) -> List[Dict]:
    """
    Final LLM step: Organize all problems into optimal repair trajectory

    Args:
        all_problems: All problems from all commits
        validation_sequence: CI validation sequence
        analyzer: CommitAnalyzer instance with LLM

    Returns:
        Organized problem sequence with dependencies and priorities
    """

    if not all_problems:
        return []

    # Format problems for LLM
    problems_json = json.dumps(all_problems, indent=2)
    validation_json = json.dumps(validation_sequence, indent=2)

    prompt = f"""You are organizing a repair trajectory for CI failures.

ALL PROBLEMS FOUND (from all commits):
{problems_json}

CI VALIDATION SEQUENCE:
{validation_json}

TASK:
Analyze all the problems and organize them into an optimal repair sequence.

Consider:
1. **Dependencies**: Some problems must be fixed before others (e.g., import before usage)
2. **Validation Order**: Follow CI validation sequence (lower order first)
3. **Impact**: Problems with current_failed_jobs block the current CI state
4. **Grouping**: Same failure family problems should stay together
5. **Evidence**: Do not invent install/setup failures from a dependency/config diff unless CI metadata, structured failure, or validation output shows that step failed. A config/dependency change can be a hidden fix, but its priority must reflect evidence.
6. **Job status fields**: current_fixed_jobs means jobs that passed in that commit. Preserve it from the input problem records; do not reinterpret it as jobs fixed by the problem.

For each problem, add:
- **problem_id**: integer starting at 1 in final repair order
- **depends_on**: List of problem indices this depends on (if any)

OUTPUT JSON (valid JSON only, no markdown):
{{
  "organized_problems": [
    {{
      "problem_id": 1,
      "commit_sha": "...",
      "commit_message": "...",
      "files": [...],
      "failure_type": "...",
      "issue_type": "...",
      "problem": "...",
      "root_cause": "...",
      "changes_made": "...",
      "introduces": "...",
      "fixes": "...",
      "current_failed_jobs": [
        {{
          "job": "...",
          "step": "...",
          "validation_cmd": "..."
        }}
      ],
      "current_fixed_jobs": [
        {{
          "job": "...",
          "step": "...",
          "validation_cmd": "..."
        }}
      ],
      "validation_cmd": "...",
      "commit": "...",
      "sha_success": "...",
      "sha_fail": "...",
      "depends_on": [],
      "reasoning": "Why this priority/ordering"
    }},
    // ... more problems in order
  ]
}}

IMPORTANT:
1. Keep ALL problems (don't drop any)
2. Order by blocking impact, then validation order and evidence strength
3. Mark dependencies clearly
4. Do not include a priority field in the output.
5. Return ONLY valid JSON"""

    try:
        # Call LLM
        response = analyzer._call_llm(prompt)
        result = analyzer._parse_response(response)

        organized = result.get("organized_problems", all_problems)

        # Ensure all original fields are preserved
        if organized:
            return _normalize_organized_problems(organized)
        else:
            return _normalize_organized_problems(all_problems)

    except Exception as e:
        print(f"    Warning: LLM organization failed: {e}")
        print("    Returning problems in original order")
        return _normalize_organized_problems(all_problems)


def _normalize_organized_problems(problems: List[Dict]) -> List[Dict]:
    """Ensure final problem sequence matches the requested output schema."""
    normalized = []
    for idx, problem in enumerate(problems or [], 1):
        if not isinstance(problem, dict):
            continue
        item = dict(problem)
        item.pop("priority", None)
        item["problem_id"] = idx
        item.setdefault("depends_on", [])
        item.setdefault("reasoning", "")
        item.setdefault("current_failed_jobs", [])
        item.setdefault("current_fixed_jobs", [])
        normalized.append(item)
    return normalized
