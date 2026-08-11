from __future__ import annotations

from codex.scripts.ci_repair_prompts import (
    DYNAMIC_CONTEXT_BOUNDARY,
    prompt_cache_info,
)
from codex.scripts.run_codex_ci_repair import (
    compose_baseline_prompt,
    compose_memory_prompt,
)


def _issue(issue_id: str) -> dict[str, str]:
    return {
        "id": issue_id,
        "repo": f"owner/repository-{issue_id}",
        "sha_fail": f"sha-{issue_id}",
        "workflow_path": f".github/workflows/ci-{issue_id}.yml",
    }


def _problem(number: int, marker: str) -> dict[str, object]:
    return {
        "number": number,
        "problem": f"dynamic problem {marker}",
        "root_cause": f"dynamic root cause {marker}",
        "files": [f"config-{marker}.toml", f"source-{marker}.py"],
        "failure_signals": [f"dynamic signal {marker}"],
    }


def test_dynamic_issue_data_does_not_change_baseline_cache_prefix() -> None:
    first = compose_baseline_prompt(
        _issue("first"),
        {"error_context": ["failure-first"]},
        {"validation_cmd": "check-first"},
        _problem(1, "first"),
        2,
    )
    second = compose_baseline_prompt(
        _issue("second"),
        {"error_context": ["failure-second"]},
        {"validation_cmd": "check-second"},
        _problem(2, "second"),
        2,
    )

    first_prefix, _, first_dynamic = first.partition(DYNAMIC_CONTEXT_BOUNDARY)
    second_prefix, _, second_dynamic = second.partition(DYNAMIC_CONTEXT_BOUNDARY)

    assert first_prefix == second_prefix
    assert prompt_cache_info(first)["template_fingerprint"] == prompt_cache_info(
        second
    )["template_fingerprint"]
    assert "dynamic problem first" not in first_prefix
    assert "dynamic problem second" not in second_prefix
    assert "dynamic problem first" in first_dynamic
    assert "dynamic problem second" in second_dynamic
    assert first_dynamic != second_dynamic


def test_memory_repair_plan_remains_in_dynamic_context() -> None:
    problem = _problem(1, "package")
    problem["repair_strategy"] = {
        "summary": "upgrade the constrained package",
        "actions": ["update pyproject.toml", "refresh the lockfile"],
        "validation_cmd": "uv sync",
        "pitfalls": ["preserve the supported Python range"],
    }

    prompt = compose_memory_prompt(
        _issue("memory"),
        {"validation_cmd": "uv sync"},
        problem,
        1,
    )
    stable_prefix, boundary, dynamic_context = prompt.partition(
        DYNAMIC_CONTEXT_BOUNDARY
    )

    assert boundary == DYNAMIC_CONTEXT_BOUNDARY
    assert "upgrade the constrained package" not in stable_prefix
    assert "upgrade the constrained package" in dynamic_context
    assert "update pyproject.toml" in dynamic_context
    assert "refresh the lockfile" in dynamic_context
    assert "preserve the supported Python range" in dynamic_context


def test_editing_stable_instructions_automatically_changes_fingerprint() -> None:
    prompt = compose_baseline_prompt(
        _issue("edit"),
        {"error_context": ["failure"]},
        {"validation_cmd": "check"},
        _problem(1, "edit"),
        1,
    )
    edited_prompt = prompt.replace(
        "You are fixing one CI failure problem",
        "You are repairing one CI failure problem",
        1,
    )

    original_info = prompt_cache_info(prompt)
    edited_info = prompt_cache_info(edited_prompt)

    assert original_info["template_fingerprint"] != edited_info[
        "template_fingerprint"
    ]
    assert original_info["dynamic_context_chars"] == edited_info[
        "dynamic_context_chars"
    ]


def test_stable_prefix_has_a_conservative_cacheable_size() -> None:
    prompt = compose_baseline_prompt(
        _issue("size"),
        {"error_context": ["failure"]},
        {"validation_cmd": "check"},
        _problem(1, "size"),
        1,
    )

    info = prompt_cache_info(prompt)
    assert info["layout"] == "stable_prefix_dynamic_suffix_v1"
    assert info["stable_prefix_chars"] >= 4096
