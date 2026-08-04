#!/usr/bin/env python3
"""
Regenerate failure_memory.json (L1) from reordered decomposed_issues.json.

This ensures L1 has problems in the correct sequence:
1. Dependencies/config first
2. Then by CI verification order
"""

import json
from pathlib import Path


def build_failure_memory(decomposed_issues: list[dict]) -> list[dict]:
    """Convert decomposed_issues to failure_memory format."""
    failure_memory = []

    for issue in decomposed_issues:
        # Extract metadata
        memory_entry = {
            "issue_id": issue.get("original_issue_id"),
            "repo": issue.get("repo"),
            "repo_owner": issue.get("repo", "").split("/")[0] if "/" in issue.get("repo", "") else None,
            "workflow": issue.get("workflow_path"),
            "workflow_name": issue.get("workflow_name"),
            "problems": []
        }

        # Convert problems (already in correct order from reordered file!)
        for problem in issue.get("problems", []):
            # Build fix_strategy from how_fixed + why_fix_works
            how_fixed = problem.get("how_fixed", "")
            why_fix_works = problem.get("why_fix_works", "")
            fix_strategy = f"{how_fixed}\n\nWhy this works: {why_fix_works}" if how_fixed else ""

            # Extract enabled[] from problem
            enabled = []
            if problem.get("is_cascading"):
                # Cascading problems may depend on earlier problems
                # This is approximate - ideally we'd parse dependency_type
                problem_id = problem.get("problem_id", 0)
                if problem_id > 1:
                    # Depend on previous problems (heuristic)
                    enabled = [problem_id - 1]

            memory_problem = {
                "problem_id": problem.get("problem_id"),
                "problem": problem.get("problem", ""),
                "root_cause": problem.get("root_cause", ""),
                "files": problem.get("affected_files", []),
                "failure_type": problem.get("failure_type", ""),
                "verification_cmd": problem.get("validation_cmd", ""),
                "fix_strategy": fix_strategy.strip(),
                "enabled": enabled
            }

            memory_entry["problems"].append(memory_problem)

        failure_memory.append(memory_entry)

    return failure_memory


def main():
    """Main entry point."""
    data_dir = Path(__file__).parent.parent / "data" / "back_trs"

    decomposed_file = data_dir / "decomposed_issues.json"
    output_file = data_dir / "failure_memory.json"
    backup_file = data_dir / "failure_memory_original.json"

    print(f"Loading {decomposed_file}...")
    with open(decomposed_file, "r", encoding="utf-8") as f:
        decomposed_issues = json.load(f)

    print(f"Building failure memory from {len(decomposed_issues)} issues...")
    failure_memory = build_failure_memory(decomposed_issues)

    # Backup original
    if output_file.exists():
        print(f"Backing up original to {backup_file}...")
        import shutil
        shutil.copy(output_file, backup_file)

    print(f"Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(failure_memory, f, indent=2, ensure_ascii=False)

    print(f"✓ Done! Generated failure_memory.json with {len(failure_memory)} issues.")
    print(f"\nProblems are now in CI verification order:")
    print(f"  1. Dependencies/config")
    print(f"  2. Build/generation")
    print(f"  3. Type checking")
    print(f"  4. Tests")
    print(f"  5. Documentation")


if __name__ == "__main__":
    main()
