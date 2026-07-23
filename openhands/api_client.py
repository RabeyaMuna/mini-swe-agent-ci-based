#!/usr/bin/env python3
"""
OpenHands API Client - Send structured problems to OpenHands server
"""

import time
from typing import Any, Optional

import requests


class OpenHandsAPIClient:
    """Client for interacting with OpenHands API server"""

    def __init__(
        self,
        base_url: str = 'http://localhost:3000',
        api_token: Optional[str] = None,
    ):
        """
        Initialize OpenHands API client

        Args:
            base_url: OpenHands server URL (default: http://localhost:3000)
            api_token: Optional API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f'{self.base_url}/api/v1'
        self.headers = {'Content-Type': 'application/json'}
        if api_token:
            self.headers['Authorization'] = f'Bearer {api_token}'

    def create_conversation(
        self, problem: dict[str, Any], metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Create new conversation with structured problem

        Args:
            problem: Structured problem object (see format_problem_message)
            metadata: Optional metadata (repo_url, sha_fail, etc.)

        Returns:
            conversation_id: ID of created conversation
        """
        payload = {
            'messages': [
                {
                    'role': 'user',
                    'content': {'type': 'problem', 'problem': problem},
                }
            ]
        }

        if metadata:
            payload['metadata'] = metadata

        print(f'Creating conversation at {self.api_base}/conversations')
        print(f'Problem ID: {problem.get("problem_id")}')

        response = requests.post(
            f'{self.api_base}/conversations', json=payload, headers=self.headers
        )

        if response.status_code != 200:
            raise RuntimeError(
                f'Failed to create conversation: {response.status_code} - {response.text}'
            )

        result = response.json()
        conversation_id = result.get('conversation_id') or result.get('id')

        if not conversation_id:
            raise RuntimeError(f'No conversation_id in response: {result}')

        print(f'Created conversation: {conversation_id}')
        return conversation_id

    def send_message(
        self, conversation_id: str, message: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Send message to existing conversation

        Args:
            conversation_id: Conversation ID
            message: Message dict with role and content

        Returns:
            Response from API
        """
        payload = {'content': [message]}

        response = requests.post(
            f'{self.api_base}/conversations/{conversation_id}/messages',
            json=payload,
            headers=self.headers,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f'Failed to send message: {response.status_code} - {response.text}'
            )

        return response.json()

    def get_conversation_state(self, conversation_id: str) -> dict[str, Any]:
        """Get current conversation state"""
        response = requests.get(
            f'{self.api_base}/conversations/{conversation_id}', headers=self.headers
        )

        if response.status_code != 200:
            raise RuntimeError(
                f'Failed to get conversation: {response.status_code} - {response.text}'
            )

        return response.json()

    def wait_for_completion(
        self, conversation_id: str, timeout: int = 600, poll_interval: int = 5
    ) -> dict[str, Any]:
        """
        Poll conversation until agent completes or timeout

        Args:
            conversation_id: Conversation ID
            timeout: Maximum wait time in seconds
            poll_interval: Time between polls in seconds

        Returns:
            Final conversation state
        """
        start_time = time.time()
        print(f'Waiting for conversation {conversation_id} to complete...')

        while time.time() - start_time < timeout:
            state = self.get_conversation_state(conversation_id)
            status = state.get('status', 'unknown')

            print(f'  Status: {status} ({int(time.time() - start_time)}s elapsed)')

            # Check for terminal states
            if status in ['completed', 'finished', 'success']:
                print('✓ Conversation completed successfully')
                return state
            elif status in ['failed', 'error', 'stopped']:
                print(f'✗ Conversation failed: {status}')
                return state

            time.sleep(poll_interval)

        raise TimeoutError(
            f'Conversation {conversation_id} did not complete within {timeout}s'
        )

    def get_patch(self, conversation_id: str) -> str:
        """
        Extract patch from completed conversation

        Args:
            conversation_id: Conversation ID

        Returns:
            Unified diff patch string
        """
        state = self.get_conversation_state(conversation_id)

        # Try different possible locations for patch
        patch = (
            state.get('patch')
            or state.get('model_patch')
            or state.get('diff')
            or state.get('result', {}).get('patch')
            or ''
        )

        return patch

    def get_trajectory(self, conversation_id: str) -> list[dict[str, Any]]:
        """Get agent trajectory (actions taken)"""
        state = self.get_conversation_state(conversation_id)
        return state.get('trajectory', []) or state.get('history', [])


def format_problem_message(
    problem_id: str,
    summary: str,
    repo: str,
    sha_fail: str,
    reproduction: list[str],
    relevant_files: list[str],
    logs_snippet: str,
    validation: list[str],
    root_causes: Optional[list[dict[str, Any]]] = None,
    suggested_fixes: Optional[list[dict[str, Any]]] = None,
    fix_strategy: Optional[dict[str, Any]] = None,
    time_budget_minutes: int = 120,
    allow_auto_apply: bool = False,
) -> dict[str, Any]:
    """
    Format a structured problem message for OpenHands

    Args:
        problem_id: Unique problem identifier
        summary: Brief problem description
        repo: Repository URL
        sha_fail: Failed commit SHA
        reproduction: List of commands to reproduce
        relevant_files: List of files involved
        logs_snippet: Relevant error logs
        validation: List of validation commands
        root_causes: Optional list of root cause hypotheses
        suggested_fixes: Optional list of suggested fixes
        fix_strategy: Optional fix strategy with steps
        time_budget_minutes: Time budget for solving
        allow_auto_apply: Whether to allow auto-apply

    Returns:
        Structured problem dict
    """
    problem = {
        'problem_id': problem_id,
        'summary': summary,
        'severity': 'high',
        'priority': 100,
        'repo': repo,
        'sha_fail': sha_fail,
        'reproduction': reproduction,
        'relevant_files': relevant_files,
        'logs_snippet': logs_snippet,
        'validation': validation,
        'time_budget_minutes': time_budget_minutes,
        'allow_auto_apply': allow_auto_apply,
    }

    if root_causes:
        problem['root_causes'] = root_causes

    if suggested_fixes:
        problem['suggested_fixes'] = suggested_fixes

    if fix_strategy:
        problem['fix_strategy'] = fix_strategy

    return problem


if __name__ == '__main__':
    # Test example
    client = OpenHandsAPIClient('http://localhost:3000')

    # Example problem
    problem = format_problem_message(
        problem_id='test-mypy-001',
        summary='Fix mypy type annotation errors',
        repo='https://github.com/adap/flower',
        sha_fail='6aee1d58e8ce6402c48325c8c479dae84596d352',
        reproduction=[
            'cd py',
            'python -m mypy --config-file=pyproject.toml flwr/common/inflatable_test.py',
        ],
        relevant_files=[
            'py/flwr/common/inflatable_test.py',
            'py/flwr/common/secure_aggregation/ndarrays_arithmetic.py',
        ],
        logs_snippet='py/flwr/common/inflatable_test.py:60: error: Function is missing a return type annotation',
        validation=['cd py && python -m mypy --config-file=pyproject.toml flwr/'],
        root_causes=[
            {
                'id': 'rc1',
                'description': 'Missing -> None return type on test function',
                'evidence': 'mypy error at line 60',
                'confidence': 0.95,
            }
        ],
    )

    try:
        conv_id = client.create_conversation(problem)
        final_state = client.wait_for_completion(conv_id)
        patch = client.get_patch(conv_id)

        print(f'\nPatch ({len(patch)} chars):')
        print(patch[:500])
    except Exception as e:
        print(f'Error: {e}')
