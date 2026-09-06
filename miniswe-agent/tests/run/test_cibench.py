# These integration tests use real Git snapshots and local validation commands.
import json
import shlex
import subprocess
import sys

import pytest

from minisweagent.environments.local import LocalEnvironment
from minisweagent.run.benchmarks.cibench import _run_sequential_repair
from minisweagent.run.benchmarks.utils.repair_session import RepairSession, validation_commands


from types import SimpleNamespace
from unittest.mock import patch

from minisweagent.run.benchmarks.cibench import _make_context_llm, _resolve_context_model


def test_context_model_defaults_to_repair_model():
    config = {"model": {"model_name": "minimax/minimax-m2.5"}}

    assert _resolve_context_model(None, config) == "minimax/minimax-m2.5"


def test_context_model_can_be_overridden():
    config = {"model": {"model_name": "minimax/minimax-m2.5"}}

    assert _resolve_context_model("openai/gpt-4.1", config) == "openai/gpt-4.1"


def test_context_llm_uses_configured_model():
    config = {
        "model": {
            "model_name": "minimax/minimax-m2.5",
            "model_class": "openrouter",
            "model_kwargs": {"temperature": 0.0, "drop_params": True},
        }
    }
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="answer"),
            )
        ]
    )

    with patch("minisweagent.run.benchmarks.cibench.litellm.completion", return_value=response) as completion:
        context_llm = _make_context_llm(config)
        assert context_llm("prompt") == "answer"

    completion.assert_called_once_with(
        model="minimax/minimax-m2.5",
        messages=[{"role": "user", "content": "prompt"}],
        temperature=0.0,
        drop_params=True,
    )




def repair_git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True).stdout


@pytest.fixture
def repair_repo(tmp_path):
    repair_git(tmp_path, "init", "-q")
    repair_git(tmp_path, "config", "user.name", "Test")
    repair_git(tmp_path, "config", "user.email", "test@example.invalid")
    (tmp_path / "first.txt").write_text("old\n")
    (tmp_path / "second.txt").write_text("old\n")
    repair_git(tmp_path, "add", ".")
    repair_git(tmp_path, "commit", "-qm", "failed commit")
    return tmp_path, repair_git(tmp_path, "rev-parse", "HEAD").decode().strip()


def check_file_command(name, value="fixed\n"):
    code = f"from pathlib import Path; assert Path({name!r}).read_text() == {value!r}"
    return shlex.quote(sys.executable) + " -c " + shlex.quote(code)


class SequentialTestAgent:
    def __init__(self, repo, actions):
        self.env = LocalEnvironment(cwd=str(repo))
        self.config = SimpleNamespace(wall_time_limit_seconds=42)
        self._start_time = 123
        self.actions = iter(actions)
        self.prompts = []

    def run(self, prompt):
        self.prompts.append(prompt)
        next(self.actions)(prompt)
        return {"exit_status": "Submitted", "submission": ""}


def run_repair(repo, base, agent, problems):
    progress = SimpleNamespace(update_instance_status=lambda *args: None)
    return _run_sequential_repair(agent, problems, repo, progress, "test", sha_fail=base)


def two_problems():
    return [
        {"problem_statement": "First behavior must remain fixed", "verification_cmd": check_file_command("first.txt")},
        {"problem_statement": "Second behavior must remain fixed", "verification_cmd": check_file_command("second.txt")},
    ]


def test_repair_session_carries_previous_requirements_and_runs_fast_problems(repair_repo):
    repo, base = repair_repo
    def second(prompt):
        records = next((repo.parent / f"{repo.name}-patch-reconciliation").glob("attempt-*/repair-record.json"))
        assert str(records) in prompt
        record = json.loads(records.read_text())
        assert record["problems"][0]["problem_statement"] == "First behavior must remain fixed"
        assert record["repairs"][0]["validation"]["status"] == "passed"
        assert record["repairs"][0]["changed_files"] == ["first.txt"]
        (repo / "second.txt").write_text("fixed\n")

    agent = SequentialTestAgent(repo, [lambda _: (repo / "first.txt").write_text("fixed\n"), second])
    index = (repo / ".git/index").read_bytes()
    info, diff = run_repair(repo, base, agent, two_problems())
    assert len(agent.prompts) == 2
    assert info["exit_status"] == "submitted"
    assert info["sequential_repair"]["fixed_problems"] == 2
    assert info["sequential_repair"]["integration_validation"]["status"] == "passed"
    assert "first.txt" in diff and "second.txt" in diff
    assert (repo / ".git/index").read_bytes() == index
    assert agent.config.wall_time_limit_seconds == 42 and agent._start_time == 123


def test_repair_session_recovers_an_earlier_fix_overwritten_without_git_conflict(repair_repo):
    repo, base = repair_repo

    def second(_):
        (repo / "first.txt").write_text("old\n")
        (repo / "second.txt").write_text("fixed\n")

    def reconcile(prompt):
        assert "Combined validation failed" in prompt
        assert "repair-record.json" in prompt
        assert (repo / "second.txt").read_text() == "fixed\n"
        (repo / "first.txt").write_text("fixed\n")

    agent = SequentialTestAgent(repo, [lambda _: (repo / "first.txt").write_text("fixed\n"), second, reconcile])
    info, diff = run_repair(repo, base, agent, two_problems())
    assert len(agent.prompts) == 3
    assert info["exit_status"] == "submitted"
    assert info["sequential_repair"]["fixed_problems"] == 2
    assert (repo / "first.txt").read_text() == (repo / "second.txt").read_text() == "fixed\n"
    assert "+fixed" in diff
    record = json.loads(open(info["sequential_repair"]["repair_record"]).read())
    assert [r["status"] for r in record["validation_history"] if r["label"].startswith("final")] == ["failed", "passed"]


def test_repair_session_preserves_candidate_when_combined_checks_still_fail(repair_repo):
    repo, base = repair_repo
    agent = SequentialTestAgent(repo, [lambda _: (repo / "second.txt").write_text("fixed\n"), lambda _: None, lambda _: None])
    problems = [{"problem_statement": "Fix first", "verification_cmd": check_file_command("first.txt")}]
    info, diff = run_repair(repo, base, agent, problems)
    assert len(agent.prompts) == 3
    assert info["exit_status"] == "validation_failed"
    assert info["sequential_repair"]["fixed_problems"] == 0
    assert open(info["sequential_repair"]["candidate_patch"], "rb").read() == diff.encode()
    assert (repo / "second.txt").read_text() == "fixed\n"


@pytest.mark.parametrize("command", [None, "nonexistent-ci-check-47"])
def test_repair_session_unavailable_checks_remain_unverified(repair_repo, command):
    repo, base = repair_repo
    agent = SequentialTestAgent(repo, [lambda _: (repo / "first.txt").write_text("fixed\n")])
    info, diff = run_repair(repo, base, agent, [{"problem_statement": "Fix first", "verification_cmd": command}])
    assert len(agent.prompts) == 1
    assert info["exit_status"] == "submitted"
    assert info["sequential_repair"]["integration_validation"]["status"] == "unverified"
    assert info["sequential_repair"]["fixed_problems"] == 0
    assert diff


def test_repair_session_deduplicates_checks_and_keeps_full_output(repair_repo):
    repo, base = repair_repo
    problems = [{"verification_cmd": "check"}, {"verification": {"validation_cmd": "check"}}]
    session = RepairSession(repo, base, problems, repo.parent / f"{repo.name}-record")
    calls = []
    output = "evidence " * 10000

    def execute(command):
        calls.append(command)
        return {"returncode": 0, "output": output}

    report = session.validate([0, 1], execute, "final")
    assert calls == ["check"]
    assert report["status"] == "passed"
    assert open(report["checks"][0]["output_path"]).read() == output
    assert len(session.context()) < 5000


def test_repair_session_does_not_reuse_checks_after_validation_changes_files(repair_repo):
    repo, base = repair_repo
    session = RepairSession(repo, base, [{"verification_cmd": "mutating-check"}], repo.parent / f"{repo.name}-record")
    calls = []

    def execute(command):
        calls.append(command)
        (repo / "first.txt").write_text(f"changed by check {len(calls)}\n")
        return {"returncode": 0, "output": ""}

    patch, report = session.finish(execute, lambda *args: pytest.fail("No agent needed to rerun checks"))
    assert report["status"] == "failed"
    assert "changed the final tree" in report["reason"]
    assert len(calls) == 3
    assert "+changed by check 3" in patch


def test_repair_session_handles_existing_validation_schemas():
    assert validation_commands({"verification": {"validation_sequence": [{"validation_cmd": "one"}, {"validation_cmd": "two"} ]},
                                "repair_strategy": {"validation_cmd": "one"}}) == ["one", "two"]


def test_repair_session_retry_keeps_previous_checkpoints(repair_repo):
    repo, base = repair_repo
    artifacts = repo.parent / f"{repo.name}-record"
    first = RepairSession(repo, base, [{"problem_statement": "first attempt"}], artifacts)
    original_record = first.record_path.read_bytes()
    second = RepairSession(repo, base, [{"problem_statement": "second attempt"}], artifacts)
    assert first.record_path != second.record_path
    assert first.record_path.read_bytes() == original_record
