"""Prompt formatter for OpenHands CI-Bench tasks."""

import json
from typing import Any, Optional


class PromptFormatter:
    """Format prompts for OpenHands - unified format for baseline and memory"""

    @staticmethod
    def _format_workflow_validation(workflow: dict[str, Any]) -> str:
        """Format cached workflow validation context for the agent prompt."""
        if not workflow:
            return (
                'No cached workflow validation context was found. Inspect the '
                'repository CI workflow files directly and infer setup/validation.'
            )

        lines = []
        workflow_path = workflow.get('workflow_path')
        if workflow_path:
            lines.append(f'Workflow path: {workflow_path}')

        validation_sequence = workflow.get('validation_sequence')
        if isinstance(validation_sequence, list) and validation_sequence:
            lines.append('\nCI verification sequence:')
            for item in validation_sequence:
                if not isinstance(item, dict):
                    continue
                order = item.get('order', '?')
                validates = item.get('validates', '')
                install_cmd = item.get('installation_cmd', '')
                validation_cmd = item.get('validation_cmd', '')
                source = item.get('source', '')
                evidence = item.get('evidence', '')
                lines.append(f'{order}. {validates}'.rstrip())
                if install_cmd:
                    lines.append(f'   setup: {install_cmd}')
                if validation_cmd:
                    lines.append(f'   validate: {validation_cmd}')
                if source:
                    lines.append(f'   source: {source}')
                if evidence:
                    lines.append(f'   evidence: {evidence}')

        validation_command = workflow.get('validation_command')
        if validation_command:
            lines.append(f'\nPrimary validation command: {validation_command}')

        if len(lines) == 0:
            return json.dumps(workflow, indent=2, ensure_ascii=False)

        return '\n'.join(lines)

    @staticmethod
    def format_task(
        issue_data: dict[str, Any], memory_context: Optional[str] = None
    ) -> dict[str, Any]:
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
        repo_url = f'https://github.com/{issue_data["repo"]}'
        benchmark_id = issue_data.get('id') or issue_data['instance_id']
        workflow_validation = PromptFormatter._format_workflow_validation(
            issue_data.get('workflow', {})
        )
        ci_failure_summary = issue_data.get('ci_failure_summary') or (
            'No processed CI failure summary was provided.'
        )
        if ci_failure_summary == issue_data['problem_statement']:
            ci_failure_summary = 'Same as the Problem section above.'

        # Repair plan section (empty for baseline, filled for memory)
        if memory_context:
            repair_plan_section = memory_context
        else:
            repair_plan_section = (
                'No previous memory context is available. Analyze the problem '
                'and workflow validation context from scratch.'
            )

        initial_message = f"""Fix the CI failure at the failed commit.

## Repository
{issue_data['repo']}

## Failed Commit
The runtime will checkout this failed commit before the agent edits files:
{issue_data['sha_fail']}

## Problem
{issue_data['problem_statement']}

## Processed CI Failure Details
Use this processed failure analysis instead of raw CI logs:

{ci_failure_summary}

## Repair Plan and Relevant Prior Experience
{repair_plan_section}

## CI Workflow Validation Context
Use this cached workflow validation context as the source of setup and validation commands. Treat `validation_sequence` as CI verification information, not as files to edit. If a command is a GitHub Actions action or shorthand, inspect the workflow file in the repository to reconstruct the equivalent local setup.

{workflow_validation}

## Workflow File To Inspect
{issue_data.get('workflow_path') or issue_data.get('workflow', {}).get('workflow_path') or 'Inspect .github/workflows/ for the relevant workflow file.'}

## Task
1. Review the problem statement, repair plan/prior experience if available, workflow validation context, and workflow file.
2. Determine the necessary local setup from the CI workflow and validation context.
3. Reproduce or narrow the failing validation when practical using the relevant commands from the workflow validation context.
4. Identify the root cause and make the smallest correct code change.
5. If tools like ruff, black, isort, docformatter, or project-specific formatters are the appropriate fix, use them instead of hand-editing formatting.
6. Run the relevant validation command(s) if possible. Prefer the smallest command that validates the failing area, then broader workflow commands if practical.
7. If validation cannot run because of missing CI-only services, secrets, system packages, or time limits, state exactly what could not be run and what local checks you did run.

## Expected Output
Leave the repository with the fix applied. The benchmark runner will collect the final unified diff against {issue_data['sha_fail']}.
"""

        return {
            'repository': repo_url,
            'selected_branch': None,
            'initial_message': initial_message.strip(),
            'instance_id': issue_data['instance_id'],
        }

    @staticmethod
    def format_baseline_ci_failure(ci_failure_summary: str) -> str:
        """
        Format BASELINE task - just raw CI failure info, no guidance.

        Agent must figure out everything on its own:
        - What the errors mean
        - Which files to fix
        - How to fix them

        Args:
            ci_failure_summary: Formatted CI failure from data_loader

        Returns:
            Simple problem statement for baseline mode
        """
        return f"""The CI run failed. Fix the failures described below.

{ci_failure_summary}

Your task:
1. Analyze the CI failure information above
2. Identify which files need to be fixed
3. Make the necessary code changes to fix the errors
4. Ensure the fixes address all reported issues

The repository is already checked out at the failed commit. Make your changes and the benchmark runner will collect the final diff.
"""

    @staticmethod
    def format_baseline_task(issue_data: dict[str, Any]) -> dict[str, Any]:
        """
        Format baseline task (no memory, no decomposition, no guidance)

        Baseline = Raw CI failure info only. Agent figures it out.

        Args:
            issue_data: Issue data from data_loader

        Returns:
            Dict with repository, branch, and initial_message for OpenHands
        """
        return PromptFormatter.format_task(issue_data, memory_context=None)

    @staticmethod
    def format_memory_task(
        issue_data: dict[str, Any], memory_context: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Format memory task (with repair plan) - uses unified format

        Args:
            issue_data: Issue data from data_loader
            memory_context: Formatted repair plan from memory_retriever

        Returns:
            Dict with repository, branch, and initial_message for OpenHands
        """
        return PromptFormatter.format_task(issue_data, memory_context)


if __name__ == '__main__':
    # Test formatter
    import json

    formatter = PromptFormatter()

    # Test data
    issue_data = {
        'instance_id': 'owner__repo__abc123',
        'repo': 'owner/repo',
        'sha_fail': 'abc123def',
        'base_sha': 'main',
        'problem_statement': 'CI test is failing with AssertionError',
        'ci_logs': 'ERROR: test_feature_x failed\nAssertionError: Expected 5, got 3',
        'validation_command': 'pytest tests/',
        'workflow': {},
    }

    # Baseline format
    print('=== BASELINE FORMAT ===')
    baseline = formatter.format_baseline_task(issue_data)
    print(json.dumps(baseline, indent=2))

    # Memory format
    print('\n=== MEMORY FORMAT ===')
    memory_context = """Based on previous experiences, consider these approaches:

**From Similar Past Failures:**
1. Updated expected value in test
   (Similar issue: AssertionError in test_feature_x)

**Repository-Specific Patterns:**
2. This repo uses pytest fixtures in conftest.py
"""
    memory = formatter.format_memory_task(issue_data, memory_context)
    print(json.dumps(memory, indent=2))
