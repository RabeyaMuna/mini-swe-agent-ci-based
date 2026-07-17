"""
Prompt formatter for OpenHands CI-Bench tasks
Uses SAME format for both baseline and memory modes
"""

from typing import Dict, Any, Optional


class PromptFormatter:
    """Format prompts for OpenHands - unified format for baseline and memory"""

    @staticmethod
    def format_task(
        issue_data: Dict[str, Any],
        memory_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format issue data for OpenHands (unified format)

        Works for both baseline and memory modes:
        - Baseline: memory_context = None → no repair plan
        - Memory: memory_context = formatted string → includes repair plan

        Args:
            issue_data: Issue data from data_loader
            memory_context: Formatted repair plan from memory_retriever (optional)

        Returns:
            Dict with repository, branch, and initial_message for OpenHands
        """
        repo_url = f"https://github.com/{issue_data['repo']}"

        # Repair plan section (empty for baseline, filled for memory)
        if memory_context:
            repair_plan_section = memory_context
        else:
            repair_plan_section = "No previous experiences available. Analyze from scratch."

        initial_message = f"""Fix the CI failure at commit {issue_data['sha_fail']}.

## Repository
{issue_data['repo']}

## Problem
{issue_data['problem_statement']}

## Repair Plan (If available)
{repair_plan_section}

## Failed Commit
{issue_data['sha_fail']}

## Task
1. Checkout commit {issue_data['sha_fail']}
2. Review the previous experiences and repair plan above (if available)
3. Analyze whether the failure matches known patterns from memory
4. Identify the locations and problems
5. If any problems can be automatically fixed by tools like ruff or docformatter (particularly styling and formatting issues), apply them to fix automatically instead of manually
6. Analyze the root cause, considering similar past solutions (if available)
7. Generate fixes informed by past solutions (if available) or based on analysis
8. Verify if possible using: {issue_data['validation_command']}
   (Note: Verification depends on your environment. If you cannot run tests, generate the fix based on analysis.)

## Expected Output
Provide a complete git patch (diff format) that fixes this CI failure.
"""

        return {
            "repository": repo_url,
            "selected_branch": issue_data.get('base_sha', 'main'),
            "initial_message": initial_message.strip(),
            "instance_id": issue_data['instance_id']
        }

    @staticmethod
    def format_baseline_task(issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format baseline task (no memory) - uses unified format

        Args:
            issue_data: Issue data from data_loader

        Returns:
            Dict with repository, branch, and initial_message for OpenHands
        """
        return PromptFormatter.format_task(issue_data, memory_context=None)

    @staticmethod
    def format_memory_task(
        issue_data: Dict[str, Any],
        memory_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format memory task (with repair plan) - uses unified format

        Args:
            issue_data: Issue data from data_loader
            memory_context: Formatted repair plan from memory_retriever

        Returns:
            Dict with repository, branch, and initial_message for OpenHands
        """
        return PromptFormatter.format_task(issue_data, memory_context)


if __name__ == "__main__":
    # Test formatter
    import json

    formatter = PromptFormatter()

    # Test data
    issue_data = {
        "instance_id": "owner__repo__abc123",
        "repo": "owner/repo",
        "sha_fail": "abc123def",
        "base_sha": "main",
        "problem_statement": "CI test is failing with AssertionError",
        "ci_logs": "ERROR: test_feature_x failed\nAssertionError: Expected 5, got 3",
        "validation_command": "pytest tests/",
        "workflow": {}
    }

    # Baseline format
    print("=== BASELINE FORMAT ===")
    baseline = formatter.format_baseline_task(issue_data)
    print(json.dumps(baseline, indent=2))

    # Memory format
    print("\n=== MEMORY FORMAT ===")
    memory_context = """Based on previous experiences, consider these approaches:

**From Similar Past Failures:**
1. Updated expected value in test
   (Similar issue: AssertionError in test_feature_x)

**Repository-Specific Patterns:**
2. This repo uses pytest fixtures in conftest.py
"""
    memory = formatter.format_memory_task(issue_data, memory_context)
    print(json.dumps(memory, indent=2))
