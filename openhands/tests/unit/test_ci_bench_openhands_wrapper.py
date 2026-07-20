import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

OPENHANDS_ROOT = Path(__file__).resolve().parents[2]
if str(OPENHANDS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENHANDS_ROOT))

import interactive_agent  # noqa: E402
from interactive_agent import (  # noqa: E402
    AgentEnvironment,
    OpenHandsAgent,
    _ensure_relative_path,
    _extract_json_object,
    _is_blocked_command,
    _repo_slug,
)
from prompt_formatter import PromptFormatter  # noqa: E402


def test_repo_slug_accepts_github_url_and_plain_slug():
    assert _repo_slug('https://github.com/example/project.git') == 'example/project'
    assert _repo_slug('example/project') == 'example/project'


def test_extract_json_object_handles_nested_args_and_fences():
    response = """```json
{"thought": "read", "tool": "read_file", "args": {"file_path": "src/app.py"}}
```"""

    assert _extract_json_object(response) == {
        'thought': 'read',
        'tool': 'read_file',
        'args': {'file_path': 'src/app.py'},
    }


@pytest.mark.parametrize(
    'command',
    [
        'docker build .',
        'python -m docker compose up',
        'podman run image',
        '/usr/local/bin/docker-compose up',
    ],
)
def test_container_commands_are_blocked(command):
    assert _is_blocked_command(command)


def test_path_guard_rejects_traversal(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()

    with pytest.raises(ValueError, match='escapes repository'):
        _ensure_relative_path(repo, '../outside.txt')


def test_baseline_prompt_has_no_memory_context():
    task = PromptFormatter.format_task(
        {
            'instance_id': '1',
            'repo': 'owner/repo',
            'sha_fail': 'abc123',
            'problem_statement': 'Tests fail',
            'workflow': {},
        },
        memory_context=None,
    )

    assert 'No previous memory context is available' in task['initial_message']
    assert 'Repair Plan and Relevant Prior Experience' in task['initial_message']


def test_memory_prompt_includes_repair_plan():
    task = PromptFormatter.format_task(
        {
            'instance_id': '1',
            'repo': 'owner/repo',
            'sha_fail': 'abc123',
            'problem_statement': 'Tests fail',
            'workflow': {},
        },
        memory_context='Use the prior pytest fixture fix.',
    )

    assert 'Use the prior pytest fixture fix.' in task['initial_message']


def test_agent_run_command_rejects_docker(tmp_path):
    env = AgentEnvironment(
        repo_url='https://github.com/example/project',
        commit_sha='abc123',
        instance_id='1',
    )
    env.repo_dir = tmp_path

    result = env.run_command('docker build .')

    assert result['status'] == 'failed'
    assert result['exit_code'] == 126


def test_agent_run_solves_problem_with_tools_and_returns_final_diff(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.writes = []
            self.cleaned = False

        def setup(self):
            return {
                'status': 'success',
                'message': 'ready',
                'work_dir': '/tmp/repo',
            }

        def read_file(self, file_path):
            assert file_path == 'src/app.py'
            return {
                'status': 'success',
                'content': 'VALUE = 1\n',
            }

        def search(self, query, file_glob=''):
            assert query == 'VALUE'
            return {
                'status': 'success',
                'output': 'src/app.py:1:VALUE = 1\n',
                'stderr': '',
                'exit_code': 0,
            }

        def write_file(self, file_path, content):
            self.writes.append((file_path, content))
            return {'status': 'success', 'message': 'wrote'}

        def run_command(self, command, timeout=60):
            return {'status': 'success', 'stdout': '', 'stderr': '', 'exit_code': 0}

        def get_diff(self):
            return {
                'status': 'success',
                'diff': 'diff --git a/src/app.py b/src/app.py\n',
            }

        def cleanup(self):
            self.cleaned = True

    responses = iter(
        [
            '{"tool": "search", "args": {"query": "VALUE", "file_glob": "*.py"}}',
            '{"tool": "read_file", "args": {"file_path": "src/app.py"}}',
            '{"tool": "write_file", "args": {"file_path": "src/app.py", "content": "VALUE = 2\\n"}}',
            '{"tool": "done", "args": {"notes": "fixed"}}',
        ]
    )

    def fake_completion(**kwargs):
        content = next(responses)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(total_tokens=100),
        )

    monkeypatch.setattr(interactive_agent.litellm, 'completion', fake_completion)
    env = FakeEnv()
    agent = OpenHandsAgent(model='test-model')

    result = agent.run(
        {
            'repository': 'https://github.com/owner/repo',
            'commit_sha': 'abc123',
            'initial_message': '## Problem\nFix src/app.py\n',
        },
        env,
    )

    assert env.writes == [('src/app.py', 'VALUE = 2\n')]
    assert env.cleaned
    assert result['status'] == 'success'
    assert result['patch'].startswith('diff --git')


def test_agent_runs_multiple_problems_in_one_checkout(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.setup_count = 0
            self.cleanup_count = 0
            self.writes = []

        def setup(self):
            self.setup_count += 1
            return {'status': 'success', 'message': 'ready', 'work_dir': '/tmp/repo'}

        def search(self, query, file_glob=''):
            return {'status': 'success', 'output': '', 'stderr': '', 'exit_code': 1}

        def read_file(self, file_path):
            return {'status': 'success', 'content': f'{file_path}: old\n'}

        def write_file(self, file_path, content):
            self.writes.append((file_path, content))
            return {'status': 'success', 'message': 'wrote'}

        def run_command(self, command, timeout=60):
            return {'status': 'success', 'stdout': '', 'stderr': '', 'exit_code': 0}

        def get_diff(self):
            return {'status': 'success', 'diff': 'diff --git combined\n'}

        def cleanup(self):
            self.cleanup_count += 1

    responses = iter(
        [
            '{"tool": "write_file", "args": {"file_path": "a.py", "content": "A = 1\\n"}}',
            '{"tool": "done", "args": {"notes": "first"}}',
            '{"tool": "write_file", "args": {"file_path": "b.py", "content": "B = 1\\n"}}',
            '{"tool": "done", "args": {"notes": "second"}}',
        ]
    )

    def fake_completion(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))],
            usage=SimpleNamespace(total_tokens=50),
        )

    monkeypatch.setattr(interactive_agent.litellm, 'completion', fake_completion)
    env = FakeEnv()
    agent = OpenHandsAgent(model='test-model', max_steps=3)

    result = agent.run(
        {
            'repository': 'https://github.com/owner/repo',
            'commit_sha': 'abc123',
            'problems': [
                {'source': 'ci failure', 'description': 'fix first'},
                {'source': 'previous experience', 'description': 'fix second'},
            ],
        },
        env,
    )

    assert env.setup_count == 1
    assert env.cleanup_count == 1
    assert env.writes == [('a.py', 'A = 1\n'), ('b.py', 'B = 1\n')]
    assert result['patch'] == 'diff --git combined\n'


def test_agent_setup_uses_existing_cache_without_docker(tmp_path, monkeypatch):
    cache_root = tmp_path / 'cache'
    source = tmp_path / 'source'
    source.mkdir()
    interactive_agent.subprocess.run(['git', 'init'], cwd=source, check=True)
    interactive_agent.subprocess.run(
        ['git', 'config', 'user.email', 'test@example.com'], cwd=source, check=True
    )
    interactive_agent.subprocess.run(
        ['git', 'config', 'user.name', 'Test User'], cwd=source, check=True
    )
    (source / 'README.md').write_text('hello\n')
    interactive_agent.subprocess.run(
        ['git', 'add', 'README.md'], cwd=source, check=True
    )
    interactive_agent.subprocess.run(
        ['git', 'commit', '-m', 'init'], cwd=source, check=True
    )
    commit = interactive_agent.subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    cache = cache_root / 'owner__repo'
    cache_root.mkdir()
    interactive_agent.subprocess.run(
        ['git', 'clone', str(source), str(cache)], check=True
    )
    monkeypatch.setattr(interactive_agent, 'REPO_CACHE_ROOT', cache_root)

    env = AgentEnvironment(
        repo_url='https://github.com/owner/repo',
        commit_sha=commit,
        instance_id='1',
    )

    result = env.setup()
    try:
        assert result['status'] == 'success'
        assert (env.repo_dir / 'README.md').read_text() == 'hello\n'
    finally:
        env.cleanup()
