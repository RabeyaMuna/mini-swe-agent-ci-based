import json

from minisweagent.run.benchmarks.utils import ci_memory_system as cms


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, _prompt):
        return json.dumps(self.payload)


def test_final_memory_merge_uses_llm_for_cross_level_similarity_group(monkeypatch):
    l1_problem = {
        "source": "L1",
        "problem": "unused import in pkg/a.py",
        "root_cause": "refactor left unused imports",
        "how_fixed": "remove unused import",
        "files": ["pkg/a.py"],
        "validation_cmd": "ruff check",
        "issue_type": "F401",
        "failure_type": "lint",
    }
    l3_problem = {
        "source": "L3",
        "problem": "unused imports usually need removal across related files",
        "root_cause": "refactor left unused imports",
        "how_fixed": "remove unused imports from related files",
        "files": ["pkg/a.py", "pkg/b.py"],
        "validation_cmd": "ruff check",
        "issue_type": "F401",
        "failure_type": "lint",
    }

    monkeypatch.setattr(cms, "_group_memory_problems_by_similarity", lambda problems: [problems])
    fake_llm = FakeLLM([
        {
            "action": "MERGE",
            "validation_cmd": "ruff check",
            "error_type": "lint",
            "issue_type": "F401",
            "files": ["pkg/a.py", "pkg/b.py"],
            "problem_statement": "Unused imports in related files",
            "root_cause": "Refactor left unused imports",
            "fix_strategy": "Remove the unused imports from both files.",
        }
    ])

    merged = cms._finalize_serial_problems([], [l1_problem, l3_problem], fake_llm)

    assert len(merged) == 1
    assert merged[0]["problem_id"] == 1
    assert merged[0]["source"] == "previous experience"
    assert merged[0]["problem_statement"] == "Unused imports in related files"
    assert merged[0]["files"] == [{"path": "pkg/a.py"}, {"path": "pkg/b.py"}]


def test_final_memory_merge_fallback_keeps_distinct_groups_separate(monkeypatch):
    lint_problem = {
        "source": "L1",
        "problem": "unused import",
        "root_cause": "unused import",
        "how_fixed": "remove import",
        "files": ["pkg/a.py"],
        "validation_cmd": "ruff check",
        "issue_type": "F401",
    }
    type_problem = {
        "source": "L2",
        "problem": "wrong argument type",
        "root_cause": "call signature mismatch",
        "how_fixed": "pass expected type",
        "files": ["pkg/b.py"],
        "validation_cmd": "mypy",
        "issue_type": "arg-type",
    }

    monkeypatch.setattr(cms, "_group_memory_problems_by_similarity", lambda problems: [problems])

    merged = cms._finalize_serial_problems([], [lint_problem, type_problem], llm=None)

    assert len(merged) == 2
    assert {problem["verification_cmd"] for problem in merged} == {"ruff check", "mypy"}
