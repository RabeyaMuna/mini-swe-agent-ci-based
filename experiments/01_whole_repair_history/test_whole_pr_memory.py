"""Unit tests for the no-attribution whole-PR memory control."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("build_whole_pr_memory.py")
SPEC = importlib.util.spec_from_file_location("whole_pr_builder", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class FakeGitHubFetcher:
    def get_commits_between(self, owner, name, sha_success, sha_fail):
        assert (owner, name) == ("upstream", "project")
        assert (sha_success, sha_fail) == ("pass-sha", "fail-sha")
        return [
            {
                "sha": "repair-1",
                "message": "repair complete episode",
                "author": "Developer",
                "date": "2026-01-01T00:00:00Z",
                "html_url": "https://github.com/upstream/project/commit/repair-1",
            }
        ]

    def get_commit_ci_metadata(self, owner, name, sha):
        return {"current_jobs_fixed": [{"job": "tests", "conclusion": "success"}]}


def sample_record():
    return {
        "id": "100",
        "repo_owner": "benchmark",
        "repo_name": "project",
        "commit_link": "https://github.com/upstream/project/tree/fail-sha",
        "workflow_path": ".github/workflows/ci.yml",
        "workflow_name": "CI",
        "workflow": "jobs: {test: {}}",
        "sha_fail": "fail-sha",
        "sha_success": "pass-sha",
        "logs": [{"job_name": "tests", "log": "AssertionError"}],
        "failed_jobs": [{"job_name": "tests", "steps": ["pytest"]}],
        "failure_types": ["Test Failure", "Dependency Issues"],
        "diff": "diff --git a/a.py b/a.py\n+fixed\n",
        "changed_files": ["a.py", "requirements.txt"],
    }


def sample_abstraction():
    return {
        "l1": {
            "failure_type": "test and dependency failure",
            "problem": "The complete CI episode failed.",
            "root_cause": "The PR required coordinated repository changes.",
            "fix_strategy": "Apply the cumulative patch and validate the workflow.",
            "why_fix_works": "The complete change set restores the CI contract.",
            "verification_cmd": "pytest",
        },
        "l2": {
            "failure_identify": "A repository-level CI repair was required.",
            "strategy": {
                "summary": "Repair and validate the complete episode.",
                "key_actions": ["Apply coordinated changes", "Run CI"],
            },
        },
        "l3": {
            "pattern_id": "coordinated-ci-repair",
            "failure_type": "CI failure",
            "failure_pattern": "CI requires a coordinated repair.",
            "problem": "A complete repair spans related repository changes.",
            "reasoning": "The cumulative repair restores consistency.",
            "when_to_apply": "When CI evidence and the cumulative patch align.",
            "signals": ["multiple failing checks"],
            "universal_fix": {
                "approach": "Apply a coordinated repair.",
                "steps": ["Inspect CI", "Apply changes", "Validate"],
                "applies_to": ["CI repair"],
            },
            "examples": [],
            "dependent_changes": [],
        },
    }


def test_collects_one_complete_episode_without_attribution():
    episode = builder.collect_whole_pr_episode(
        sample_record(),
        structured_failure={"error_context": ["AssertionError"]},
        github_fetcher=FakeGitHubFetcher(),
        allow_incomplete=False,
        fetch_final_ci=True,
    )

    assert episode["issue_id"] == "100"
    assert episode["repo"] == "benchmark/project"
    assert episode["source_repository"] == "upstream/project"
    assert len(episode["ordered_commits"]) == 1
    assert episode["cumulative_patch"].startswith("diff --git")
    assert all(episode["completeness"].values())
    assert "problems" not in episode
    assert "dependencies" not in episode


def test_builds_exactly_one_compatible_item_per_level():
    episode = builder.collect_whole_pr_episode(
        sample_record(),
        structured_failure={"error_context": ["AssertionError"]},
        github_fetcher=FakeGitHubFetcher(),
        allow_incomplete=False,
        fetch_final_ci=False,
    )
    abstraction = sample_abstraction()
    builder.validate_abstraction(abstraction, "100")
    l1, l2, l3 = builder.build_memory_records(
        episode, abstraction, prompt_truncated=False
    )

    assert len(l1["problems"]) == 1
    assert l1["dependencies"] == []
    assert l1["repair_sequence"] == [1]
    assert len(l2["repair_strategies"]) == 1
    assert l2["total_problems"] == 1
    assert l3["source_issue_id"] == "100"
    assert l3["pattern_id"].startswith("whole-pr-100-")
    builder.validate_memory_sets(["100"], [l1], [l2], [l3])


def test_rejects_plural_problem_output():
    abstraction = sample_abstraction()
    abstraction["l1"]["problems"] = [{"problem": "leaked decomposition"}]
    with pytest.raises(ValueError, match="reintroduced plural records"):
        builder.validate_abstraction(abstraction, "100")


def test_strict_collection_rejects_missing_commit_sequence():
    with pytest.raises(ValueError, match="ordered_commit_sequence"):
        builder.collect_whole_pr_episode(
            sample_record(),
            structured_failure={"error_context": ["AssertionError"]},
            github_fetcher=None,
            allow_incomplete=False,
            fetch_final_ci=False,
        )
