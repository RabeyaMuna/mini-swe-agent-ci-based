import pytest

from memory_plugin.decomposition_cache import (
    DecompositionCache,
    DecompositionCacheError,
)
from memory_plugin.memory_plugin import MemoryPlugin
from memory_plugin.stair_retrieval import (
    DecompositionGenerationError,
    STAIRRetrieval,
)
import memory_plugin.stair_retrieval as stair_module


class RecordingCache:
    def __init__(self):
        self.saved = {}

    def has(self, sha_fail):
        return sha_fail in self.saved

    def get_problems(self, sha_fail):
        return self.saved.get(sha_fail)

    def set(self, sha_fail, query, problems, model="unknown"):
        self.saved[sha_fail] = problems


def _valid_query():
    return {
        "sha_fail": "9cf62fab15cce046bb01baf9d0d312ba4da659c3",
        "repo": "axolotl",
        "workflow_path": ".github/workflows/lint.yml",
        "error_context": ["The pre-commit job failed because pylint reported R0801."],
        "failure_signals": ["pylint R0801 duplicate-code"],
        "relevant_files": [{"file": "tests/test_validation_dataset.py"}],
        "error_types": [{"category": "Code Quality / Linting"}],
        "failed_cmd": ["pre-commit run --all-files"],
    }


def test_empty_llm_decomposition_is_reported_and_not_cached(monkeypatch):
    cache = RecordingCache()
    monkeypatch.setattr(stair_module, "get_global_cache", lambda: cache)
    monkeypatch.setattr(
        stair_module, "invoke_llm_with_retry", lambda **kwargs: []
    )

    retrieval = object.__new__(STAIRRetrieval)
    retrieval.llm = object()

    with pytest.raises(DecompositionGenerationError, match="returned malformed"):
        retrieval._stage_0_decompose_ci_failure(_valid_query())

    assert cache.saved == {}


def test_valid_llm_decomposition_is_cached(monkeypatch):
    cache = RecordingCache()
    problem = {
        "problem": "Pylint reported duplicate code",
        "root_cause": "Two implementations contain the same block",
        "files": ["tests/test_validation_dataset.py"],
    }
    monkeypatch.setattr(stair_module, "get_global_cache", lambda: cache)
    monkeypatch.setattr(
        stair_module,
        "invoke_llm_with_retry",
        lambda **kwargs: {"problems": [problem]},
    )

    retrieval = object.__new__(STAIRRetrieval)
    retrieval.llm = object()
    problems = retrieval._stage_0_decompose_ci_failure(_valid_query())

    assert problems == [problem]
    assert cache.saved[_valid_query()["sha_fail"]] == [problem]


def test_memory_query_accepts_singular_failed_job():
    plugin = object.__new__(MemoryPlugin)
    failed_job = [{"job": "pre-commit", "command": "pre-commit/action@v3"}]

    query = plugin._build_query(
        {"failed_job": failed_job},
        verification={},
        issue_metadata={},
    )

    assert query["failed_jobs"] == failed_job


def test_stale_cache_writers_merge_instead_of_overwriting(tmp_path):
    first = DecompositionCache(tmp_path)
    second = DecompositionCache(tmp_path)

    first.set("sha-one", {}, [{"problem": "first"}])
    second.set("sha-two", {}, [{"problem": "second"}])

    reloaded = DecompositionCache(tmp_path)
    assert reloaded.get_problems("sha-one") == [{"problem": "first"}]
    assert reloaded.get_problems("sha-two") == [{"problem": "second"}]


def test_malformed_cache_is_reported(tmp_path):
    cache_file = tmp_path / "decomposition_cache.json"
    cache_file.write_text("[]", encoding="utf-8")

    with pytest.raises(DecompositionCacheError, match="must contain a JSON object"):
        DecompositionCache(tmp_path)
