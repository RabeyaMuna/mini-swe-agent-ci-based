#!/usr/bin/env python3
"""
Simple Agent Executor for OpenHands CI-Bench

Implements a basic LLM-based agent that generates patches for CI failures.
"""

from __future__ import annotations

from typing import Any

import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def execute_agent(
    task: dict[str, Any],
    model: str,
    max_iterations: int = 5,
) -> dict[str, Any]:
    """
    Execute LLM-based agent to generate CI fix patch.

    Args:
        task: Formatted task from PromptFormatter containing:
            - repository: Repo URL
            - selected_branch: Branch name (or None)
            - initial_message: Full formatted task prompt
            - instance_id: Issue ID
        model: Model name (e.g., "zai/glm-5.2")
        max_iterations: Maximum agent iterations

    Returns:
        {
            "patch": str,  # Generated patch
            "trajectory": list,  # Agent actions
            "status": str,  # "success" | "failed" | "max_iterations"
            "total_cost": float,
        }
    """
    print(f'   Starting agent execution with {model}')

    # Extract initial_message (already formatted by PromptFormatter)
    initial_message = task.get('initial_message', '')
    task.get('repository', '')

    if not initial_message:
        print('   Error: No initial_message in task')
        return {
            'patch': '',
            'trajectory': [{'action': 'error', 'error': 'No initial_message in task'}],
            'status': 'failed',
            'total_cost': 0.0,
        }

    system_prompt = """You are an expert software engineer. Generate a git patch to fix the CI failure described below.

Output ONLY the patch in unified diff format. Do not include explanations or markdown fences.

Example output format:
diff --git a/example.py b/example.py
--- a/example.py
+++ b/example.py
@@ -10,3 +10,3 @@ def function():
     context line
-    old_code = "wrong"
+    new_code = "fixed"
     context line

For multiple files:
diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -5,2 +5,2 @@
-old line
+new line
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -10,2 +10,2 @@
-old line
+new line
"""

    # Append strong patch-only instruction to the task
    user_prompt = f"""{initial_message}

---
IMPORTANT: Based on the above information, generate ONLY a unified diff patch that fixes the CI failure.
Do NOT explain your reasoning. Do NOT include bash commands. ONLY output the patch.

Output format (example):
diff --git a/example/file.py b/example/file.py
--- a/example/file.py
+++ b/example/file.py
@@ -10,3 +10,3 @@
 context
-old code
+new code
 context
"""

    # Execute agent
    trajectory = []
    patch = ''
    status = 'failed'
    total_cost = 0.0

    try:
        # Call LLM
        response = litellm.completion(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.0,  # Deterministic for patch generation
            max_tokens=16000,  # Allow larger patches
        )

        # Extract patch
        patch = response.choices[0].message.content.strip()

        # Track trajectory
        trajectory.append(
            {
                'action': 'generate_patch',
                'input': user_prompt[:500] + '...',
                'output': patch[:500] + '...' if len(patch) > 500 else patch,
            }
        )

        # Calculate cost (rough estimate)
        usage = response.usage
        if usage:
            # Rough cost estimate: $0.01 per 1K tokens
            total_cost = (usage.total_tokens / 1000) * 0.01

        # Validate patch format
        if patch.startswith('diff --git') or patch.startswith('---'):
            status = 'success'
            print(f'   Generated patch ({len(patch)} chars)')
        else:
            print("    Response doesn't look like a patch, treating as failed")
            patch = f'# Agent output (not a valid patch):\n{patch}'
            status = 'failed'

    except Exception as e:
        print(f'   Agent execution failed: {e}')
        trajectory.append(
            {
                'action': 'error',
                'error': str(e),
            }
        )
        patch = ''
        status = 'failed'

    return {
        'patch': patch,
        'trajectory': trajectory,
        'status': status,
        'total_cost': total_cost,
    }


def execute_baseline_agent(
    task: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """
    Execute baseline agent (no memory).

    Args:
        task: Task dict (repair_plan should be None)
        model: Model name

    Returns:
        Agent result with patch
    """
    # Ensure no memory context
    task_copy = task.copy()
    task_copy['repair_plan'] = None

    return execute_agent(task_copy, model)


def execute_memory_agent(
    task: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """
    Execute memory-augmented agent.

    Args:
        task: Task dict (should have repair_plan)
        model: Model name

    Returns:
        Agent result with patch
    """
    if not task.get('repair_plan'):
        print('Warning: Memory agent called but no repair_plan provided')

    return execute_agent(task, model)


if __name__ == '__main__':
    # Test execution
    test_task = {
        'repository': 'https://github.com/example/repo',
        'selected_branch': 'main',
        'problem_statement': 'CI failed: Ruff linter error F632 at line 389',
        'repair_plan': None,
    }

    result = execute_baseline_agent(test_task, 'zai/glm-5.2')
    print(f'\nResult: {result["status"]}')
    print(f'Patch length: {len(result["patch"])}')
    print(f'Cost: ${result["total_cost"]:.4f}')
