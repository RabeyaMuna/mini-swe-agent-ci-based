#!/usr/bin/env python3
"""
OpenHands Interactive Agent - Full implementation with environment access

This is a separate agent architecture from mini-swe-agent to enable
comparison of how different agents perform with the same memory plugin.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import litellm
from dotenv import load_dotenv

from bash_instructions import build_bash_instruction
from bash_parser import (
    extract_bash_command,
    extract_read_file_path,
    extract_written_file_path,
    is_completion_command,
    is_read_command,
    is_write_command,
)
from fault_localization import extract_faulty_files

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_CACHE_ROOT = Path(
    os.getenv('OPENHANDS_REPO_CACHE_ROOT')
    or os.getenv('MSWEA_REPO_CACHE_ROOT')
    or PROJECT_ROOT / 'repo'
).resolve()
BLOCKED_COMMANDS = {
    'docker',
    'docker-compose',
    'podman',
}


def _repo_slug(repo_url: str) -> str:
    """Return owner/repo from a GitHub URL or owner/repo string."""
    if not repo_url:
        raise ValueError('Missing repository')

    parsed = urlparse(repo_url)
    if parsed.scheme:
        path = parsed.path.strip('/')
    else:
        path = repo_url.strip('/')

    if path.endswith('.git'):
        path = path[:-4]

    parts = [part for part in path.split('/') if part]
    if len(parts) < 2:
        raise ValueError(f'Could not parse repository owner/name from: {repo_url}')
    return '/'.join(parts[-2:])


def _ensure_relative_path(repo_dir: Path, file_path: str) -> Path:
    """Resolve a repo-relative path and reject traversal outside the checkout."""
    if not file_path:
        raise ValueError('Missing file_path')

    full_path = (repo_dir / file_path).resolve()
    repo_root = repo_dir.resolve()
    if full_path != repo_root and repo_root not in full_path.parents:
        raise ValueError(f'Path escapes repository: {file_path}')
    return full_path


def _run_git(
    args: list[str], cwd: Path, timeout: int = 300
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _must_run_git(
    args: list[str], cwd: Path, error_label: str, timeout: int = 300
) -> None:
    result = _run_git(args, cwd, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f'{error_label}:\n{result.stderr[:800] or result.stdout[:800]}'
        )


def _ensure_commit_available(
    repo_path: Path, commit: str, remote: str = 'origin'
) -> None:
    verify = _run_git(['rev-parse', '--verify', f'{commit}^{{commit}}'], repo_path)
    if verify.returncode == 0:
        return

    fetch = _run_git(['fetch', '--quiet', remote, commit], repo_path)
    if fetch.returncode != 0:
        raise RuntimeError(
            f'git fetch failed for commit {commit}:\n{fetch.stderr[:800]}'
        )

    verify = _run_git(['rev-parse', '--verify', f'{commit}^{{commit}}'], repo_path)
    if verify.returncode != 0:
        raise RuntimeError(
            f'commit {commit} unavailable after fetch:\n{verify.stderr[:800]}'
        )


def _is_blocked_command(command: str) -> bool:
    """Return True if the shell command tries to invoke container runtimes."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    return any(Path(token).name in BLOCKED_COMMANDS for token in tokens)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from a model response."""
    text = text.strip()
    if '```json' in text:
        text = text.split('```json', 1)[1].split('```', 1)[0].strip()
    elif '```' in text:
        text = text.split('```', 1)[1].split('```', 1)[0].strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f'No valid JSON object found in response: {text[:200]}')


def _coerce_text_to_action(text: str) -> dict[str, Any] | None:
    """
    Simple fallback: if model returns prose, try basic exploration.
    This is a last resort - ideally model should return JSON.
    """
    # If model says it's done, accept it
    if 'done' in text.lower() or 'complete' in text.lower():
        return {'tool': 'done', 'args': {'notes': 'Marked complete'}}

    # Otherwise, safe fallback: list Python files to help model orient
    return {
        'tool': 'run_command',
        'args': {'command': 'find . -type f -name "*.py" | head -20', 'timeout': 10},
    }


class AgentEnvironment:
    """Environment for agent to interact with repository."""

    def __init__(self, repo_url: str, commit_sha: str, instance_id: str):
        """
        Initialize environment with cloned repository.

        Args:
            repo_url: GitHub repository URL
            commit_sha: Commit SHA to checkout
            instance_id: Issue instance ID
        """
        self.repo_url = repo_url
        self.commit_sha = commit_sha
        self.instance_id = instance_id
        self.work_dir = None
        self.repo_dir = None

    def setup(self) -> dict[str, Any]:
        """
        Setup working directory and clone repository.

        Returns:
            {"status": "success"/"failed", "message": str}
        """
        try:
            # Create temporary working directory
            self.work_dir = Path(
                tempfile.mkdtemp(prefix=f'openhands_{self.instance_id}_')
            )
            self.repo_dir = self.work_dir / 'repo'
            repo_slug = _repo_slug(self.repo_url)
            repo_owner, repo_name = repo_slug.split('/', 1)
            cache_dir = REPO_CACHE_ROOT / f'{repo_owner}__{repo_name}'
            clone_url = f'https://github.com/{repo_slug}.git'

            REPO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            if not (cache_dir / '.git').exists():
                result = subprocess.run(
                    ['git', 'clone', '--quiet', clone_url, str(cache_dir)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    return {
                        'status': 'failed',
                        'message': f'Clone failed: {result.stderr[:800]}',
                    }
            else:
                subprocess.run(
                    [
                        'git',
                        '-C',
                        str(cache_dir),
                        'fetch',
                        '--all',
                        '--tags',
                        '--prune',
                        '--quiet',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

            _ensure_commit_available(cache_dir, self.commit_sha)
            result = subprocess.run(
                [
                    'git',
                    'clone',
                    '--quiet',
                    '--shared',
                    str(cache_dir),
                    str(self.repo_dir),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                return {
                    'status': 'failed',
                    'message': f'Local clone failed: {result.stderr[:800]}',
                }

            _ensure_commit_available(self.repo_dir, self.commit_sha)
            _must_run_git(
                ['checkout', '--detach', '--force', self.commit_sha],
                self.repo_dir,
                'Checkout failed',
            )
            _must_run_git(
                ['reset', '--hard', self.commit_sha], self.repo_dir, 'Reset failed'
            )
            _must_run_git(['clean', '-fdx'], self.repo_dir, 'Clean failed')

            # Verify commit
            verify_result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )

            actual_sha = verify_result.stdout.strip()
            if actual_sha != self.commit_sha:
                return {
                    'status': 'failed',
                    'message': f'Checkout mismatch: expected {self.commit_sha}, got {actual_sha}',
                }

            return {
                'status': 'success',
                'message': f'Repository ready at {self.repo_dir}',
                'work_dir': str(self.repo_dir),
            }

        except Exception as e:
            return {'status': 'failed', 'message': f'Setup error: {e}'}

    def read_file(self, file_path: str) -> dict[str, Any]:
        """
        Read file from repository.

        Args:
            file_path: Relative path from repo root

        Returns:
            {"status": "success"/"failed", "content": str, "message": str}
        """
        try:
            full_path = _ensure_relative_path(self.repo_dir, file_path)
            if not full_path.exists():
                return {'status': 'failed', 'message': f'File not found: {file_path}'}

            content = full_path.read_text()
            return {
                'status': 'success',
                'content': content,
                'message': f'Read {len(content)} chars',
            }

        except Exception as e:
            return {'status': 'failed', 'message': f'Read error: {e}'}

    def write_file(self, file_path: str, content: str) -> dict[str, Any]:
        """
        Write file to repository.

        Args:
            file_path: Relative path from repo root
            content: File content

        Returns:
            {"status": "success"/"failed", "message": str}
        """
        try:
            full_path = _ensure_relative_path(self.repo_dir, file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            return {
                'status': 'success',
                'message': f'Wrote {len(content)} chars to {file_path}',
            }

        except Exception as e:
            return {'status': 'failed', 'message': f'Write error: {e}'}

    def search(self, query: str, file_glob: str = '') -> dict[str, Any]:
        """
        Search repository text with ripgrep.

        Args:
            query: Text or regex to search for
            file_glob: Optional glob such as "*.py"

        Returns:
            {"status": "success"/"failed", "output": str, "exit_code": int}
        """
        if not query:
            return {
                'status': 'failed',
                'message': 'Missing search query',
                'exit_code': 2,
            }

        command = ['rg', '-n', '--hidden', '--glob', '!.git', query]
        if file_glob:
            command.extend(['--glob', file_glob])

        try:
            result = subprocess.run(
                command,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                'status': 'success' if result.returncode in (0, 1) else 'failed',
                'output': result.stdout[:12000],
                'stderr': result.stderr[:2000],
                'exit_code': result.returncode,
            }
        except Exception as e:
            return {'status': 'failed', 'message': f'Search error: {e}', 'exit_code': 2}

    def run_command(self, command: str, timeout: int = 60) -> dict[str, Any]:
        """
        Run shell command in repository directory.

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds

        Returns:
            {"status": "success"/"failed", "stdout": str, "stderr": str, "exit_code": int}
        """
        try:
            if _is_blocked_command(command):
                return {
                    'status': 'failed',
                    'message': 'Docker/container commands are disabled for this CI-Bench OpenHands run.',
                    'exit_code': 126,
                }

            # Fix sed -i for Mac OS X (BSD sed requires backup extension)
            import platform

            if (
                platform.system() == 'Darwin'
                and 'sed -i' in command
                and "sed -i ''" not in command
            ):
                # Convert: sed -i 's/.../' file
                # To: sed -i '' 's/.../' file
                command = command.replace('sed -i ', "sed -i '' ")

            result = subprocess.run(
                command,
                shell=True,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {'status': 'failed', 'message': f'Command timeout after {timeout}s'}
        except Exception as e:
            return {'status': 'failed', 'message': f'Command error: {e}'}

    def get_diff(self) -> dict[str, Any]:
        """
        Get unified diff of all changes.

        Returns:
            {"status": "success"/"failed", "diff": str}
        """
        try:
            result = subprocess.run(
                ['git', 'diff', '--binary', self.commit_sha],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # If no diff against commit, try unstaged changes
            if not result.stdout.strip():
                result = subprocess.run(
                    ['git', 'diff', '--binary'],
                    cwd=self.repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            return {'status': 'success', 'diff': result.stdout}

        except Exception as e:
            return {'status': 'failed', 'message': f'Diff error: {e}'}

    def cleanup(self):
        """Clean up working directory."""
        if self.work_dir and self.work_dir.exists():
            import shutil

            shutil.rmtree(self.work_dir, ignore_errors=True)


class OpenHandsAgent:
    """
    Simple sequential CI repair wrapper around a local checkout.

    The production flow is intentionally small:
    1. Prepare the repository at sha_fail.
    2. Build a problem list.
    3. Solve each problem one at a time with repo search/read/write tools.
    4. Collect one final unified diff after all problems.
    """

    def __init__(
        self,
        model: str,
        max_steps: int = 15,
        max_cost: float = 5.0,
    ):
        """
        Initialize agent.

        Args:
            model: LLM model name (e.g., "zai/glm-5.2")
            max_steps: Maximum agent steps
            max_cost: Maximum cost in dollars
        """
        self.model = model
        self.max_steps = max_steps
        self.max_cost = max_cost
        self.total_cost = 0.0
        self.trajectory = []

    def run(
        self,
        task: dict[str, Any],
        env: AgentEnvironment,
    ) -> dict[str, Any]:
        """
        Run agent on task.

        Args:
            task: Formatted task from PromptFormatter
            env: Agent environment with repository access

        Returns:
            {
                "patch": str,
                "status": "success"/"failed"/"max_steps"/"max_cost",
                "trajectory": list,
                "total_cost": float,
            }
        """
        # Setup environment
        setup_result = env.setup()
        if setup_result['status'] != 'success':
            return {
                'patch': '',
                'status': 'failed',
                'trajectory': [{'action': 'setup', 'result': setup_result}],
                'total_cost': 0.0,
            }

        self.trajectory.append({'action': 'setup', 'result': setup_result})

        try:
            problems = self._task_problems(task)
            print(f'\n{"=" * 70}')
            print(f'Starting sequential problem solving: {len(problems)} problem(s)')
            print(f'{"=" * 70}')

            for index, problem in enumerate(problems, start=1):
                print(
                    f'\n Problem {index}/{len(problems)}: {problem.get("source", "unknown")}'
                )
                if problem.get('title'):
                    print(f'   Title: {problem["title"]}')

                self.trajectory.append(
                    {
                        'action': 'start_problem',
                        'problem_index': index,
                        'problem_count': len(problems),
                        'source': problem.get('source', ''),
                    }
                )
                self._solve_problem(problem, env, index=index, total=len(problems))
                print(f' Finished problem {index}/{len(problems)}')

            diff_result = env.get_diff()
            patch = diff_result.get('diff', '')
            status = 'success' if patch else 'no_patch'
            return {
                'patch': patch,
                'status': status,
                'trajectory': self.trajectory,
                'total_cost': self.total_cost,
            }
        finally:
            env.cleanup()

    def _format_initial_observation(
        self, task: dict[str, Any], setup_result: dict
    ) -> str:
        """Backward-compatible task summary for tests and diagnostics."""
        return self._build_problem_prompt(task)

    @staticmethod
    def _extract_section(initial_msg: str, section_name: str) -> str:
        """Extract a markdown section from the formatted CI task."""
        marker = f'## {section_name}'
        if marker not in initial_msg:
            return ''
        section_text = initial_msg.split(marker, 1)[1]
        next_section_idx = section_text.find('\n##')
        if next_section_idx > 0:
            section_text = section_text[:next_section_idx]
        return section_text.strip()

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f'{text[:limit]}\n...(truncated)'

    def _build_problem_prompt(self, task: dict[str, Any]) -> str:
        """Build one focused problem prompt for the model."""
        initial_msg = task.get('initial_message', '')
        problem = self._extract_section(initial_msg, 'Problem')
        processed_details = self._extract_section(
            initial_msg, 'Processed CI Failure Details'
        )
        repair_plan = self._extract_section(
            initial_msg, 'Repair Plan and Relevant Prior Experience'
        )
        workflow_context = self._extract_section(
            initial_msg, 'CI Workflow Validation Context'
        )

        task_parts = []

        if problem:
            task_parts.append(f'## Problem\n{self._truncate(problem, 1200)}')

        if (
            processed_details
            and processed_details != 'Same as the Problem section above.'
        ):
            task_parts.append(
                f'## Processed CI Failure Details\n{self._truncate(processed_details, 1200)}'
            )

        if repair_plan and 'No previous memory context is available' not in repair_plan:
            task_parts.append(f'## Repair Plan\n{self._truncate(repair_plan, 900)}')

        if workflow_context:
            task_parts.append(
                f'## Validation Context\n{self._truncate(workflow_context, 900)}'
            )

        if not task_parts:
            task_parts.append(self._truncate(initial_msg, 2500))

        commit_sha = task.get('commit_sha', 'unknown')
        repo_url = task.get('repository', 'unknown')
        task_body = '\n\n'.join(task_parts)

        return f"""Repository: {repo_url}
Failed commit already checked out: {commit_sha}

{task_body}
"""

    @staticmethod
    def _problem_text(problem: dict[str, Any]) -> str:
        parts = []
        for key, label in (
            ('title', 'Title'),
            ('source', 'Source'),
            ('description', 'Description'),
            ('root_cause', 'Root Cause'),
            ('fix_strategy', 'Fix Strategy'),
            ('validation_command', 'Validation Command'),
        ):
            value = problem.get(key)
            if value:
                parts.append(f'{label}: {value}')
        return '\n'.join(parts)

    def _build_dynamic_instruction(
        self,
        problem_context: str,
        phase: str,
        files_read: list[str],
        files_written: list[str],
    ) -> str:
        """
        Build dynamic, phase-specific instructions to guide the model.

        Phase 0: UNDERSTAND - Analyze problem and root causes
        Phase 1: INVESTIGATE - Find and read files
        Phase 2: ANALYZE - Identify specific errors and plan fixes
        Phase 3: FIX - Write corrected files
        Phase 4: VERIFY - Run tests (optional)
        Phase 5: COMPLETE - Submit patch
        """
        # Determine current phase based on progress
        if not files_read:
            # Start: understand then investigate
            current_phase = 'understand'
        elif len(files_read) == 1:
            # Read first file, analyze it
            current_phase = 'analyze'
        elif len(files_read) >= 2 and not files_written:
            # Read enough files, now fix
            current_phase = 'fix'
        elif len(files_written) > 0:
            # Written files, can verify or complete
            current_phase = 'verify'
        else:
            current_phase = phase

        instructions = {
            'understand': f"""
{problem_context}

=== PHASE 0: UNDERSTAND THE PROBLEM ===

Your task: Analyze the problem and decide the fix strategy.

CHECKLIST:
□ What is the problem? (mypy error, linting, formatting, syntax, etc.)
□ Where is it? (specific files or whole repo?)
□ Can it be AUTO-FIXED by a tool?
□ Is there a repair plan given?

AUTO-FIXABLE PROBLEMS (try these first!):
- Formatting issues → run: black . or autopep8
- Import sorting → run: isort .
- Type checking → run: mypy --install-types
- Linting → run: pylint --fix or ruff check --fix
- Missing deps → run: pip install <package>

STRATEGY:
1. If problem can be AUTO-FIXED → Try automated tool FIRST
2. If automation fails or not applicable → Manual fix

NEXT ACTIONS:

Option A - Try automated fix:
{{"tool": "run_command", "args": {{"command": "black path/to/files/", "timeout": 60}}}}
{{"tool": "run_command", "args": {{"command": "isort .", "timeout": 60}}}}
{{"tool": "run_command", "args": {{"command": "mypy --install-types --non-interactive", "timeout": 60}}}}

Option B - Manual investigation:
{{"tool": "read_file", "args": {{"file_path": "file/mentioned/in/problem.py"}}}}
{{"tool": "search", "args": {{"query": "error_pattern", "file_glob": "*.py"}}}}

TRY AUTOMATION FIRST IF APPLICABLE!

Respond with JSON only.
""",
            'investigate': f"""
{problem_context}

=== PHASE 1: INVESTIGATE & LOCATE ===

Files read: {len(files_read)}
{chr(10).join(f'  ✓ {f}' for f in files_read) if files_read else ''}

Your task: Identify WHERE the problem is in the repository.

LOCATION STRATEGIES:

1. Error messages with file:line → Read those exact files
   Example: "error in file.py:42" → read file.py

2. No specific files → Search for error pattern in repo
   {{"tool": "search", "args": {{"query": "error_class_name|error_pattern", "file_glob": "*.py"}}}}

3. Repo-wide issue (formatting, imports) → Find all affected files
   {{"tool": "run_command", "args": {{"command": "find . -name '*.py' -type f | head -20", "timeout": 30}}}}

4. Parse stack traces → Extract file paths from error output

IDENTIFY EXACT LOCATIONS:
- Which files have the error?
- Which lines in those files?
- Is it one file or multiple?
- Is it entire repo or specific directory?

Next action - Find problem location:
{{"tool": "search", "args": {{"query": "specific_error_or_function", "file_glob": "*.py"}}}}

OR read known problematic file:
{{"tool": "read_file", "args": {{"file_path": "exact/path/from/error.py"}}}}

Respond with JSON only.
""",
            'analyze': f"""
{problem_context}

=== PHASE 2: ANALYZE & PLAN FIX ===

Files you've read: {len(files_read)}
{chr(10).join(f'  ✓ {f}' for f in files_read) if files_read else ''}

Your task: ANALYZE the errors and DECIDE fix approach.

ANALYSIS CHECKLIST:
□ What is the EXACT error? (type error, missing import, syntax, formatting)
□ Which files and lines are affected?
□ Can automated tools fix this?
□ What manual fixes are needed?

ERROR TYPE → FIX APPROACH:

1. Code formatting issues (indentation, line length)
   → AUTO: {{"tool": "run_command", "args": {{"command": "black path/to/file.py", "timeout": 30}}}}

2. Import sorting issues
   → AUTO: {{"tool": "run_command", "args": {{"command": "isort path/to/file.py", "timeout": 30}}}}

3. Missing type hints (mypy errors)
   → MANUAL: Add type annotations with write_file

4. Missing imports (NameError, ImportError)
   → MANUAL: Add import statements with write_file

5. Logic/syntax errors
   → MANUAL: Fix code with write_file

DECISION:
- If can auto-fix → Run tool in next step
- If must manual-fix → Prepare to write_file in next phase
- If need more info → Read more files

Next action - Try automated fix OR prepare for manual fix:
{{"tool": "run_command", "args": {{"command": "black file.py", "timeout": 30}}}}

Respond with JSON only.
""",
            'fix': f"""
{problem_context}

=== PHASE 3: FIX THE CODE (WRITE FILES NOW!) ===

Files READ: {len(files_read)} ✓
{chr(10).join(f'  📖 {f}' for f in files_read[-5:]) if files_read else ''}

Files WRITTEN: {len(files_written)} {'❌ ZERO - MUST WRITE NOW!' if not files_written else '✓'}
{chr(10).join(f'  ✏️  {f}' for f in files_written) if files_written else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CRITICAL: You MUST use write_file RIGHT NOW!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have read the files. You know what's broken.
The ONLY way to fix code is: write_file

MANDATORY ACTION:
{{"tool": "write_file", "args": {{"file_path": "exact/path/from/files/read.py", "content": "COMPLETE FILE CONTENT WITH FIXES APPLIED"}}}}

COMMON FIXES TO APPLY:
1. Type annotation: def func(): → def func() -> ReturnType:
2. Import missing: Add "from typing import X" at top
3. Fix syntax: Add missing : or ) or ,

EXAMPLE (copy this pattern):
{{"tool": "write_file", "args": {{"file_path": "framework/py/flwr/common/inflatable_test.py", "content": "# Original file content\\n# with type hints added:\\ndef test_function() -> None:\\n    pass\\n"}}}}

⚠️  IF YOU DON'T WRITE, NOTHING GETS FIXED!
⚠️  write_file requires COMPLETE file (all lines, not just changes)
⚠️  You MUST write EVERY file that has errors

RESPOND NOW WITH write_file JSON!
""",
            'verify': f"""
{problem_context}

=== PHASE 4: VERIFY (Optional) ===

Files written: {len(files_written)}
{chr(10).join(f'  ✏️  {f}' for f in files_written) if files_written else ''}

You've written fixes! Optionally verify them:

Options:
1. Run tests: {{"tool": "run_command", "args": {{"command": "pytest test_file.py"}}}}
2. Skip verification: {{"tool": "done", "args": {{"notes": "Fixed type errors in X files"}}}}

Most likely: Use 'done' now.

Respond with JSON only.
""",
            'complete': f"""
{problem_context}

=== PHASE 5: COMPLETE ===

Summary:
- Files read: {len(files_read)}
- Files written: {len(files_written)}
{chr(10).join(f'  ✏️  {f}' for f in files_written) if files_written else '  ⚠️  NO FILES WRITTEN!'}

FINAL STEP: Mark this problem as complete.

{{"tool": "done", "args": {{"notes": "Fixed [what you fixed]"}}}}

Respond with JSON only.
""",
        }

        return instructions.get(current_phase, instructions['investigate'])

    def _task_problems(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        """Return baseline CI problem plus optional memory follow-up problems."""
        problems = task.get('problems')
        if isinstance(problems, list) and problems:
            return [problem for problem in problems if isinstance(problem, dict)]

        return [
            {
                'source': 'ci failure',
                'title': 'Current CI failure',
                'description': self._build_problem_prompt(task),
                'validation_command': task.get('validation_command', ''),
            }
        ]

    def _solve_problem(
        self,
        problem: dict[str, Any],
        env: AgentEnvironment,
        *,
        index: int,
        total: int,
    ) -> None:
        """Run a small local tool loop for one problem."""
        # Give problem ALL remaining steps (one problem might need 100 files!)
        # No artificial limit - let it uses what it needs
        steps_per_problem = self.max_steps

        # Track progress through workflow phases
        files_read = []
        files_written = []
        last_actions = []  # Track recent actions to detect loops

        # Build problem context (shown in every step)
        problem_context = f"""Problem {index} of {total}:
{self._problem_text(problem)}
"""

        # Automatic fault localization
        problem_description = problem.get('description', '')
        faulty_files = extract_faulty_files(problem_description)

        # Initial observation with bash-based instructions (works with any model)
        observation = build_bash_instruction(
            problem_context=problem_context,
            files_read=files_read,
            files_written=files_written,
            step_count=0,
            faulty_files=faulty_files,
        )
        consecutive_errors = 0
        for step in range(steps_per_problem):
            action = self._next_action(observation)
            result = self._execute_action(action, env)

            # Smart fallback: If read_file fails, auto-search for the file
            if (
                action.get('tool') == 'read_file'
                and result.get('status') == 'failed'
                and 'File not found' in result.get('message', '')
            ):
                file_path = action.get('args', {}).get('file_path', '')
                if file_path:
                    file_name = file_path.split('/')[-1]
                    print(f'Auto-searching for: {file_name}')
                    # Use find command to locate file by name
                    find_result = env.run_command(
                        f'find . -type f -name "{file_name}" 2>/dev/null | head -5',
                        timeout=10,
                    )
                    if find_result.get('status') == 'success' and find_result.get(
                        'stdout'
                    ):
                        # Get first matching file path
                        found_paths = [
                            p.strip().lstrip('./')
                            for p in find_result['stdout'].strip().split('\n')
                            if p.strip()
                        ]
                        if found_paths:
                            actual_path = found_paths[0]
                            print(f'   Found at: {actual_path}')
                            # Retry read with correct path
                            result = env.read_file(actual_path)
                            # Update action to reflect what actually happened
                            action['args']['file_path'] = actual_path

            # Track progress through workflow
            if action.get('tool') == 'read_file' and result.get('status') == 'success':
                file_path = action['args'].get('file_path', '')
                if file_path and file_path not in files_read:
                    files_read.append(file_path)
                    print(f'  Read: {file_path}')

            if action.get('tool') == 'write_file' and result.get('status') == 'success':
                file_path = action['args'].get('file_path', '')
                if file_path and file_path not in files_written:
                    files_written.append(file_path)
                    print(f'  Wrote: {file_path}')

            # Track in-place edits (sed, perl, patch)
            if (
                action.get('tool') == 'run_command'
                and result.get('status') == 'success'
            ):
                modifies_file = action['args'].get('modifies_file', '')
                if modifies_file and modifies_file not in files_written:
                    files_written.append(modifies_file)
                    print(f'  Modified: {modifies_file}')

            # LOOP DETECTION: Check if model is repeating same action
            action_signature = (
                f'{action.get("tool")}:{action.get("args", {}).get("file_path", "")}'
            )
            last_actions.append(action_signature)
            if len(last_actions) > 3:
                last_actions.pop(0)  # Keep only last 3 actions

            # If model tried to read same file 2+ times, force write
            if (
                len(last_actions) >= 2
                and action.get('tool') == 'read_file'
                and last_actions[-1] == last_actions[-2]
                and files_read
                and not files_written
            ):
                print('  WARNING: Loop detected - model keeps reading same file.')
                print('  Stopping problem - model stuck in read loop.')
                # Stop this problem - model is stuck
                self.trajectory.append(
                    {
                        'problem_index': index,
                        'action': 'stop_problem',
                        'reason': 'loop detected: model repeatedly reading same file without writing fix',
                    }
                )
                return

            self.trajectory.append(
                {
                    'problem_index': index,
                    'step': step,
                    'action': self._redact_large_action(action),
                    'result': self._shorten_result(result),
                }
            )
            if action.get('tool') == 'done':
                print('  ✓ Problem marked complete')
                return
            if result.get('status') == 'failed':
                consecutive_errors += 1
            else:
                consecutive_errors = 0
            if consecutive_errors >= 3:
                self.trajectory.append(
                    {
                        'problem_index': index,
                        'action': 'stop_problem',
                        'reason': 'three consecutive tool failures',
                    }
                )
                return

            # Generate next observation with bash-based instructions (works with any model)
            result_str = json.dumps(self._shorten_result(result), indent=2)
            observation = build_bash_instruction(
                problem_context=problem_context,
                files_read=files_read,
                files_written=files_written,
                step_count=step + 1,
                previous_result=result_str,
                faulty_files=faulty_files,
            )

    def _next_action(self, observation: str) -> dict[str, Any]:
        """Ask the model for the next bash command."""
        system_prompt = """You are OpenHands - an agent that fixes code using bash commands.

OUTPUT FORMAT:
- You MUST output bash commands in ```bash code blocks
- One command per response
- The environment will execute your bash commands

CRITICAL WORKFLOW:
1. Read files to understand the problem (1-2 files max)
2. Write the fix immediately (sed -i OR cat >)
3. Mark complete (echo COMPLETE_TASK)

DO NOT:
- Read the same file multiple times
- Read more than 2 files before writing a fix
- Output prose instead of commands

EXAMPLES:
Read a file:
```bash
cat path/to/file.py
```

Write a fix (in-place edit):
```bash
sed -i 's/old_code/new_code/g' path/to/file.py
```

Write a fix (full file):
```bash
cat > path/to/file.py << 'EOF'
complete fixed file content here
EOF
```

Mark complete:
```bash
echo COMPLETE_TASK
```
"""
        response = litellm.completion(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': observation},
            ],
            temperature=0,
            max_tokens=12000,
        )
        usage = response.usage
        if usage:
            self.total_cost += (usage.total_tokens / 1000) * 0.01

        response_text = str(response.choices[0].message.content or '').strip()

        # Parse bash command from response
        bash_command = extract_bash_command(response_text)

        if not bash_command:
            # No bash block found - model might have output prose
            print('    No bash command found in response')
            print(f'  Response: {response_text[:200]}...')
            # Fallback: try to find file mentioned and read it
            action = {
                'tool': 'search',
                'args': {'query': 'test', 'file_glob': '*.py'},
            }
            return action

        # DEBUG: Show what model generated
        if bash_command and len(bash_command) < 100:
            print(f'  Model: {bash_command}')
        elif bash_command:
            print(f'  Model: {bash_command[:80]}...')

        # Convert bash command to action
        action = self._bash_to_action(bash_command)
        return action

    def _bash_to_action(self, bash_command: str) -> dict[str, Any]:
        """
        Convert bash command to action dict - DYNAMIC conversion.

        Handles 567+ different issue patterns by being flexible and robust.
        """
        # Check for completion (highest priority)
        if is_completion_command(bash_command):
            return {'tool': 'done', 'args': {'notes': 'Task marked complete'}}

        # Check for write command (heredoc, redirect, or in-place edit)
        if is_write_command(bash_command):
            file_path = extract_written_file_path(bash_command)

            # IN-PLACE EDITS (sed, perl, patch) - execute directly
            if any(cmd in bash_command for cmd in ['sed -i', 'perl -i', 'patch ']):
                # These modify files in-place, so run as command
                # but track the file as written
                return {
                    'tool': 'run_command',
                    'args': {
                        'command': bash_command,
                        'timeout': 60,
                        'modifies_file': file_path,  # Track for progress
                    },
                }

            # FULL FILE WRITE (heredoc)
            if file_path:
                # Extract content from heredoc
                content = self._extract_heredoc_content(bash_command)
                if content:  # Only write if we have content
                    return {
                        'tool': 'write_file',
                        'args': {'file_path': file_path, 'content': content},
                    }
                else:
                    # No content extracted - maybe it's a simple redirect
                    # Execute as command and let bash handle it
                    return {
                        'tool': 'run_command',
                        'args': {'command': bash_command, 'timeout': 60},
                    }

        # Check for read command
        if is_read_command(bash_command):
            file_path = extract_read_file_path(bash_command)
            if file_path:
                return {'tool': 'read_file', 'args': {'file_path': file_path}}

        # DYNAMIC: Handle find commands specially (for file discovery)
        if bash_command.strip().startswith('find '):
            return {
                'tool': 'run_command',
                'args': {'command': bash_command, 'timeout': 30},
            }

        # DYNAMIC: Handle grep commands (for code search)
        if 'grep ' in bash_command:
            return {
                'tool': 'run_command',
                'args': {'command': bash_command, 'timeout': 30},
            }

        # Otherwise, run as generic command with reasonable timeout
        return {'tool': 'run_command', 'args': {'command': bash_command, 'timeout': 60}}

    def _extract_heredoc_content(self, bash_command: str) -> str:
        """
        Extract content from heredoc - DYNAMIC parsing for 567+ issue patterns.

        Handles:
        - cat > file << 'EOF' ... EOF
        - cat > file << EOF ... EOF
        - cat > file <<EOF ... EOF
        - Different delimiter names (EOF, END, MARKER, etc.)
        """
        import re

        # Try standard heredoc with delimiter
        match = re.search(
            r"<<\s*['\"]?(\w+)['\"]?\s*\n(.*?)^\1\s*$",
            bash_command,
            re.MULTILINE | re.DOTALL,
        )
        if match:
            return match.group(2).rstrip('\n')

        # DYNAMIC: Try to find content between << and the end
        # This handles malformed heredocs
        match = re.search(r"<<\s*['\"]?\w+['\"]?\s*\n(.*)", bash_command, re.DOTALL)
        if match:
            content = match.group(1)
            # Remove the delimiter line at the end if present
            lines = content.split('\n')
            # Check if last line is a single word (likely the delimiter)
            if lines and lines[-1].strip() and len(lines[-1].strip().split()) == 1:
                return '\n'.join(lines[:-1]).rstrip('\n')
            return content.rstrip('\n')

        # If no heredoc pattern, return empty
        return ''

    def _execute_action(
        self, action: dict[str, Any], env: AgentEnvironment
    ) -> dict[str, Any]:
        """Execute a model-selected repository action."""
        tool = action.get('tool')
        args = action.get('args') or {}

        if tool == 'search':
            return env.search(
                str(args.get('query') or ''),
                file_glob=str(args.get('file_glob') or ''),
            )
        if tool == 'read_file':
            return env.read_file(str(args.get('file_path') or ''))
        if tool == 'write_file':
            return env.write_file(
                str(args.get('file_path') or ''),
                str(args.get('content') or ''),
            )
        if tool == 'run_command':
            return env.run_command(
                str(args.get('command') or ''),
                timeout=int(args.get('timeout') or 60),
            )
        if tool == 'done':
            return {'status': 'success', 'message': str(args.get('notes') or 'done')}
        return {'status': 'failed', 'message': f'Unknown tool: {tool}'}

    @staticmethod
    def _redact_large_action(action: dict[str, Any]) -> dict[str, Any]:
        redacted = dict(action)
        args = dict(redacted.get('args') or {})
        if isinstance(args.get('content'), str):
            args['content'] = f'<{len(args["content"])} chars>'
        redacted['args'] = args
        return redacted

    @staticmethod
    def _shorten_result(result: dict[str, Any]) -> dict[str, Any]:
        shortened = dict(result)
        for key in ('stdout', 'stderr', 'message'):
            if isinstance(shortened.get(key), str):
                shortened[key] = OpenHandsAgent._truncate(shortened[key], 2000)
        return shortened


def execute_openhands_agent(
    task: dict[str, Any],
    model: str,
    max_steps: int = 15,
) -> dict[str, Any]:
    """
    Main entry point for OpenHands agent execution.

    Args:
        task: Formatted task from PromptFormatter
        model: Model name
        max_steps: Maximum agent steps

    Returns:
        {
            "patch": str,
            "status": str,
            "trajectory": list,
            "total_cost": float,
        }
    """
    # Create environment
    env = AgentEnvironment(
        repo_url=task.get('repository', ''),
        commit_sha=task.get('commit_sha', ''),
        instance_id=task.get('instance_id', 'unknown'),
    )
    # Create and run agent
    agent = OpenHandsAgent(model=model, max_steps=max_steps)
    result = agent.run(task=task, env=env)

    return result


if __name__ == '__main__':
    # Test
    test_task = {
        'repository': 'https://github.com/adap/flower',
        'commit_sha': '6aee1d58e8ce6402c48325c8c479dae84596d352',
        'instance_id': '113',
        'initial_message': 'Fix mypy errors in the repository.',
    }

    result = execute_openhands_agent(test_task, 'zai/glm-5.2', max_steps=5)
    print(f'Status: {result["status"]}')
    print(f'Patch length: {len(result["patch"])}')
    print(f'Steps: {len(result["trajectory"])}')
    print(f'Cost: ${result["total_cost"]:.4f}')
