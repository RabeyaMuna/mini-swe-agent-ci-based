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
        allow_incomplete=False,
    )

    assert episode["issue_id"] == "100"
    assert episode["repo"] == "benchmark/project"
    assert episode["evidence_policy"] == "benchmark_only"
    assert episode["repair_transition"]["sha_fail"] == "fail-sha"
    assert episode["repair_transition"]["sha_success"] == "pass-sha"
    assert episode["cumulative_patch"].startswith("diff --git")
    assert all(episode["completeness"].values())
    assert "problems" not in episode
    assert "dependencies" not in episode


def test_builds_exactly_one_compatible_item_per_level():
    episode = builder.collect_whole_pr_episode(
        sample_record(),
        structured_failure={"error_context": ["AssertionError"]},
        allow_incomplete=False,
    )
    abstraction = sample_abstraction()
    builder.validate_abstraction(abstraction, "100")
    l1, l2, l3 = builder.build_memory_records(episode, abstraction)

    assert len(l1["problems"]) == 1
    assert l1["problems"][0]["record_scope"] == "whole_pr"
    assert "problem_id" not in l1["problems"][0]
    assert "enabled" not in l1["problems"][0]
    assert "dependencies" not in l1
    assert "repair_sequence" not in l1
    assert "whole_pr_history" not in l1
    assert "abstraction_processing" not in l1
    assert set(l1) == {
        "issue_id",
        "repo",
        "repo_owner",
        "repo_name",
        "workflow",
        "workflow_path",
        "workflow_name",
        "problems",
        "representation",
    }
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


def test_builds_l1_then_l2_then_l3_sequentially(monkeypatch):
    episode = builder.collect_whole_pr_episode(
        sample_record(),
        structured_failure={"error_context": ["AssertionError"]},
        allow_incomplete=False,
    )
    expected = sample_abstraction()
    responses = [expected["l1"], expected["l2"], expected["l3"]]
    prompts = []

    def fake_invoke(**kwargs):
        prompts.append(kwargs["prompt"])
        return responses[len(prompts) - 1]

    monkeypatch.setattr(builder, "invoke_llm_with_retry", fake_invoke)
    actual, processing = builder.generate_whole_pr_abstractions(
        episode, object(), 120_000
    )

    assert actual == expected
    assert processing["chunking_applied"] is False
    assert processing["evidence_truncated"] is False
    assert len(prompts) == 3
    assert "WHOLE PR EVIDENCE FOR L1" in prompts[0]
    assert expected["l1"]["problem"] in prompts[1]
    assert expected["l2"]["strategy"]["summary"] in prompts[2]


def test_retries_incomplete_l1_before_generating_l2(monkeypatch):
    episode = builder.collect_whole_pr_episode(
        sample_record(),
        structured_failure={"error_context": ["AssertionError"]},
        allow_incomplete=False,
    )
    expected = sample_abstraction()
    incomplete_l1 = dict(expected["l1"])
    incomplete_l1.pop("why_fix_works")
    stages = []
    l1_attempts = 0

    def fake_invoke(**kwargs):
        nonlocal l1_attempts
        prompt = kwargs["prompt"]
        if "Build one repository-level L2" in prompt:
            stages.append("l2")
            return expected["l2"]
        if "Build one transferable L3" in prompt:
            stages.append("l3")
            return expected["l3"]
        if "WHOLE PR EVIDENCE FOR L1" in prompt:
            l1_attempts += 1
            stages.append("l1")
            if l1_attempts == 1:
                return incomplete_l1
            assert "VALIDATION ERROR" in prompt
            assert "why_fix_works" in prompt
            return expected["l1"]
        raise AssertionError(f"Unexpected prompt: {prompt[:120]}")

    monkeypatch.setattr(builder, "invoke_llm_with_retry", fake_invoke)
    actual, _ = builder.generate_whole_pr_abstractions(episode, object(), 120_000)

    assert actual == expected
    assert stages == ["l1", "l1", "l2", "l3"]


def test_strict_collection_rejects_missing_cumulative_patch():
    record = sample_record()
    record["diff"] = ""
    with pytest.raises(ValueError, match="cumulative_patch"):
        builder.collect_whole_pr_episode(
            record,
            structured_failure={"error_context": ["AssertionError"]},
            allow_incomplete=False,
        )


def test_large_episode_uses_dependency_graph_chunks_without_truncation(monkeypatch):
    record = sample_record()
    record["workflow"] = "jobs:\n" + "  test: pytest\n" * 500
    record["logs"] = [{"job_name": "tests", "log": "AssertionError\n" * 500}]
    record["diff"] = """diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
diff --git a/pkg/b.py b/pkg/b.py
--- a/pkg/b.py
+++ b/pkg/b.py
@@ -1 +1,2 @@
 from pkg.a import VALUE
+RESULT = VALUE
"""
    record["changed_files"] = ["pkg/a.py", "pkg/b.py"]
    episode = builder.collect_whole_pr_episode(
        record,
        structured_failure={"error_context": ["AssertionError"]},
        allow_incomplete=False,
    )
    expected = sample_abstraction()
    prompts = []

    def fake_invoke(**kwargs):
        prompt = kwargs["prompt"]
        prompts.append(prompt)
        if "PATCH CHUNK:" in prompt:
            return {
                "files": ["pkg/a.py", "pkg/b.py"],
                "observed_changes": ["Updated related source files"],
                "dependency_relationships": ["pkg/b.py imports pkg/a.py"],
                "repair_contribution": "The coordinated source update repairs the PR.",
                "validation_relevance": "The updated behavior is exercised by CI.",
            }
        if "CONTEXT CHUNK:" in prompt:
            return {
                "evidence_source": "CI/workflow",
                "failure_signals": ["AssertionError"],
                "workflow_constraints": ["pytest must pass"],
                "validation_commands": ["pytest"],
                "summary": "CI and workflow evidence",
            }
        if "REDUCE WHOLE-PR SUMMARIES" in prompt:
            return {
                "files": ["pkg/a.py", "pkg/b.py"],
                "observed_changes": ["Coordinated update"],
                "failure_signals": ["AssertionError"],
                "workflow_constraints": ["pytest must pass"],
                "validation_commands": ["pytest"],
                "cross_file_relationships": ["pkg/b.py imports pkg/a.py"],
                "summary": "Complete evidence summary",
            }
        if "WHOLE PR CHUNK SUMMARIES FOR L1" in prompt:
            return expected["l1"]
        if "Build one repository-level L2" in prompt:
            return expected["l2"]
        if "Build one transferable L3" in prompt:
            return expected["l3"]
        raise AssertionError(f"Unexpected prompt: {prompt[:120]}")

    monkeypatch.setattr(builder, "invoke_llm_with_retry", fake_invoke)
    actual, processing = builder.generate_whole_pr_abstractions(
        episode, object(), 4_000
    )

    assert actual == expected
    assert processing["chunking_applied"] is True
    assert processing["chunking_method"] == "dependency_graph_then_token_bound"
    assert processing["all_files_covered"] is True
    assert processing["all_hunks_covered"] is True
    assert processing["evidence_truncated"] is False
    assert processing["parsed_file_count"] == 2
    assert processing["parsed_change_count"] == 2
    assert any("PATCH CHUNK:" in prompt for prompt in prompts)
    assert any("CONTEXT CHUNK:" in prompt for prompt in prompts)
    assert any("WHOLE PR CHUNK SUMMARIES FOR L1" in prompt for prompt in prompts)


def test_patch_chunking_recovers_an_omitted_graph_file(monkeypatch):
    record = sample_record()
    record["diff"] = """diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
diff --git a/pkg/b.py b/pkg/b.py
--- a/pkg/b.py
+++ b/pkg/b.py
@@ -1 +1,2 @@
 from pkg.a import VALUE
+RESULT = VALUE
"""
    record["changed_files"] = ["pkg/a.py", "pkg/b.py"]
    episode = builder.collect_whole_pr_episode(
        record,
        structured_failure={"error_context": ["AssertionError"]},
        allow_incomplete=False,
    )
    structured = builder.parse_diff_to_structured(record["diff"])

    def omit_second_file(*args, **kwargs):
        return [
            {
                "files": [structured["files"][0]],
                "dependency_cluster": ["pkg/a.py"],
                "dependency_contexts": [],
            }
        ]

    monkeypatch.setattr(builder, "chunk_structured_diff", omit_second_file)
    payloads, metadata = builder._dependency_aware_patch_payloads(
        episode,
        model_name="deepseek-v4-flash",
        payload_chars=50_000,
    )

    assert metadata["coverage_recovery_applied"] is True
    assert metadata["coverage_recovery_change_count"] == 1
    assert metadata["all_files_covered"] is True
    assert metadata["all_hunks_covered"] is True
    assert any(
        "pkg/b.py" in payload["structured_changes_fragment"] for payload in payloads
    )
