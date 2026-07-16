import json

from minisweagent.run.benchmarks.utils import ci_memory_l2_analysis as l2


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, _prompt):
        return json.dumps(self.payload)


def test_l2_merge_uses_llm_decision_for_similarity_group(monkeypatch):
    common = {
        "validation_cmd": "ruff check",
        "failure_type": "lint",
        "issue_type": "F401",
        "files": ["pkg/a.py"],
        "problem": "unused import in pkg/a.py",
        "root_cause": "unused import remained after refactor",
        "how_fixed": "remove unused import",
        "why_fix_works": "ruff no longer reports F401",
    }
    consecutive = {
        "validation_cmd": "ruff check",
        "failure_type": "lint",
        "issue_type": "F401",
        "files": ["pkg/a.py", "pkg/b.py"],
        "problem": "same unused import pattern in related files",
        "root_cause": "unused import remained after refactor",
        "how_fixed": "remove unused imports",
        "why_fix_works": "ruff no longer reports F401",
    }

    monkeypatch.setattr(l2, "_group_similar_selected_problems", lambda problems: [problems])
    fake_llm = FakeLLM([
        {
            "action": "MERGE",
            "validation_cmd": "ruff check",
            "failure_type": "lint",
            "issue_type": "F401",
            "files": ["pkg/a.py", "pkg/b.py"],
            "problem": "unused imports in related files",
            "root_cause": "unused imports remained after refactor",
            "how_fixed": "remove unused imports",
            "why_fix_works": "ruff no longer reports F401",
        }
    ])

    merged = l2._merge_and_deduplicate([common], [consecutive], fake_llm)

    assert len(merged) == 1
    assert merged[0]["files"] == ["pkg/a.py", "pkg/b.py"]
    assert merged[0]["how_fixed"] == "remove unused imports"


def test_l2_merge_fallback_keeps_distinct_problems_separate(monkeypatch):
    common = {
        "validation_cmd": "ruff check",
        "failure_type": "lint",
        "issue_type": "F401",
        "files": ["pkg/a.py"],
        "problem": "unused import",
        "root_cause": "unused import",
        "how_fixed": "remove import",
    }
    consecutive = {
        "validation_cmd": "mypy",
        "failure_type": "type",
        "issue_type": "arg-type",
        "files": ["pkg/b.py"],
        "problem": "wrong argument type",
        "root_cause": "call signature mismatch",
        "how_fixed": "pass expected type",
    }

    monkeypatch.setattr(l2, "_group_similar_selected_problems", lambda problems: [problems])

    merged = l2._merge_and_deduplicate([common], [consecutive], llm=None)

    assert len(merged) == 2
    assert {problem["validation_cmd"] for problem in merged} == {"ruff check", "mypy"}
