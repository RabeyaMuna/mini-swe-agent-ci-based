from memory_plugin.memory_plugin import MemoryPlugin
from memory_plugin.stair_retrieval import STAIRRetrieval
import importlib


def _retrieval_without_init() -> STAIRRetrieval:
    return STAIRRetrieval.__new__(STAIRRetrieval)


def test_full_query_preserves_target_failure_evidence():
    plugin = MemoryPlugin.__new__(MemoryPlugin)
    query = plugin._build_query(
        ci_failure={
            "error_context": ["ruff F401 unused import in agno/utils/models/claude.py"],
            "failure_signals": ["F401 log_info imported but unused"],
            "relevant_files": [
                {
                    "file": "agno/utils/models/claude.py",
                    "failed_tool": "ruff",
                }
            ],
            "error_types": [{"category": "linting", "subcategory": "F401"}],
            "failed_job": [{"command": "ruff check ."}],
        },
        verification={
            "validation_sequence": [
                {"validation_cmd": "ruff check .", "order": 1}
            ]
        },
        issue_metadata={"repo": "agno-agi/agno", "sha_fail": "abc"},
    )

    assert query["repo"] == "agno"
    assert query["failure_signals"] == ["F401 log_info imported but unused"]
    assert query["relevant_files"][0]["file"] == "agno/utils/models/claude.py"
    assert query["failed_cmd"] == ["ruff check ."]


def test_gate_rejects_unrelated_recurring_problem():
    retrieval = _retrieval_without_init()
    query = {
        "error_context": ["ruff F401 unused import in agno/utils/models/claude.py"],
        "failure_signals": ["log_info imported but unused"],
        "error_types": [{"category": "linting", "subcategory": "F401"}],
        "relevant_files": [
            {"file": "agno/utils/models/claude.py", "failed_tool": "ruff"}
        ],
        "failed_cmd": ["ruff check ."],
        "failed_jobs": [{"command": "ruff check ."}],
    }
    problems = [
        {
            "problem": "Current unused import",
            "problem_type": "ci_failure",
            "files": ["agno/utils/models/claude.py"],
        },
        {
            "problem": "MCP ClientSession type error",
            "problem_type": "dependency",
            "failure_type": "type_checking",
            "files": ["libs/agno/agno/tools/mcp.py"],
            "failure_signals": ["ClientSession has no attribute __aenter__"],
            "validation_cmd": "mypy .",
        },
    ]

    kept, rejected = retrieval._filter_by_target_evidence(problems, query)

    assert [problem["problem"] for problem in kept] == ["Current unused import"]
    assert [problem["problem"] for problem in rejected] == [
        "MCP ClientSession type error"
    ]


def test_gate_accepts_memory_candidate_confirmed_in_target_failure():
    retrieval = _retrieval_without_init()
    query = {
        "error_context": [
            "mypy: incompatible return value in agno/utils/whatsapp.py"
        ],
        "failure_signals": ["return value has type bytes, expected dict"],
        "error_types": [
            {"category": "type checking", "subcategory": "return-value"}
        ],
        "relevant_files": [
            {"file": "agno/utils/whatsapp.py", "failed_tool": "mypy"}
        ],
        "failed_cmd": ["mypy ."],
        "failed_jobs": [{"command": "mypy ."}],
    }
    candidate = {
        "problem": "WhatsApp helper returns bytes instead of dict",
        "problem_type": "common",
        "failure_type": "type checking",
        "files": ["libs/agno/agno/utils/whatsapp.py"],
        "failure_signals": ["return value has type bytes, expected dict"],
        "validation_cmd": "mypy .",
    }

    kept, rejected = retrieval._filter_by_target_evidence([candidate], query)

    assert len(kept) == 1
    assert kept[0]["evidence_status"] == "confirmed_in_target_failure"
    assert rejected == []


def _source_plugin() -> MemoryPlugin:
    plugin = MemoryPlugin.__new__(MemoryPlugin)
    plugin.enable_source_validation = True
    plugin.llm = object()
    return plugin


def test_source_gate_keeps_observed_problem_and_rejects_missing_location(tmp_path):
    plugin = _source_plugin()
    retrieval = {
        "problems": [
            {
                "problem": "Observed lint error",
                "problem_type": "ci_failure",
                "files": ["src/current.py"],
            },
            {
                "problem": "Remembered dependency defect",
                "problem_type": "dependency",
                "files": ["src/does_not_exist.py"],
            },
        ]
    }

    result = plugin.validate_candidates_against_checkout(retrieval, tmp_path)

    assert [problem["problem"] for problem in result["problems"]] == [
        "Observed lint error"
    ]
    rejected = result["source_rejected_memory_problems"]
    assert rejected[0]["source_validation"]["decision"] == "rejected"
    assert result["source_validation"]["actionable_count"] == 1


def test_source_gate_only_promotes_llm_confirmed_candidate(tmp_path, monkeypatch):
    source = tmp_path / "libs" / "agno" / "agno" / "tools" / "mcp.py"
    source.parent.mkdir(parents=True)
    source.write_text("async with ClientSession() as session:\n    pass\n")
    plugin = _source_plugin()
    memory_module = importlib.import_module("memory_plugin.memory_plugin")

    monkeypatch.setattr(
        memory_module,
        "invoke_llm_with_retry",
        lambda **kwargs: {
            "decision": "confirmed",
            "reason": "The claimed async context-manager construct is present.",
            "evidence": ["libs/agno/agno/tools/mcp.py: async with ClientSession"],
        },
    )
    retrieval = {
        "problems": [
            {
                "problem": "ClientSession is incorrectly used as an async context manager",
                "root_cause": "ClientSession lacks the required context-manager protocol",
                "problem_type": "dependency",
                "files": ["agno/tools/mcp.py"],
            }
        ]
    }

    result = plugin.validate_candidates_against_checkout(retrieval, tmp_path)

    assert len(result["problems"]) == 1
    assert result["problems"][0]["source_validation"]["decision"] == "confirmed"
    assert result["source_rejected_memory_problems"] == []


def test_source_gate_withholds_unverifiable_model_result(tmp_path, monkeypatch):
    source = tmp_path / "src" / "feature.py"
    source.parent.mkdir(parents=True)
    source.write_text("def feature():\n    return True\n")
    plugin = _source_plugin()
    memory_module = importlib.import_module("memory_plugin.memory_plugin")
    monkeypatch.setattr(
        memory_module,
        "invoke_llm_with_retry",
        lambda **kwargs: {
            "decision": "unverifiable",
            "reason": "The alleged failure depends on runtime state.",
            "evidence": [],
        },
    )

    result = plugin.validate_candidates_against_checkout(
        {
            "problems": [
                {
                    "problem": "Runtime-only dependency failure",
                    "problem_type": "common",
                    "files": ["src/feature.py"],
                }
            ]
        },
        tmp_path,
    )

    assert result["problems"] == []
    assert result["source_rejected_memory_problems"][0]["source_validation"][
        "decision"
    ] == "unverifiable"


def test_source_gate_can_promote_hidden_candidate_absent_from_ci_log(
    tmp_path, monkeypatch
):
    source = tmp_path / "src" / "hidden.py"
    source.parent.mkdir(parents=True)
    source.write_text("BROKEN_DEPENDENCY_API = True\n")
    plugin = _source_plugin()
    memory_module = importlib.import_module("memory_plugin.memory_plugin")
    monkeypatch.setattr(
        memory_module,
        "invoke_llm_with_retry",
        lambda **kwargs: {
            "decision": "confirmed",
            "reason": "The obsolete dependency API marker is present.",
            "evidence": ["src/hidden.py: BROKEN_DEPENDENCY_API"],
        },
    )

    result = plugin.validate_candidates_against_checkout(
        {
            "problems": [
                {
                    "problem": "Current observed failure",
                    "problem_type": "ci_failure",
                    "files": ["src/current.py"],
                }
            ],
            "rejected_memory_problems": [
                {
                    "problem": "Hidden obsolete dependency API",
                    "problem_type": "dependency",
                    "files": ["src/hidden.py"],
                    "evidence_status": "withheld_unconfirmed_hypothesis",
                }
            ],
        },
        tmp_path,
    )

    assert [problem["problem"] for problem in result["problems"]] == [
        "Current observed failure",
        "Hidden obsolete dependency API",
    ]
    assert result["problems"][1]["source_validation"]["decision"] == "confirmed"


def test_lightweight_command_pass_rejects_memory_problem(tmp_path):
    source = tmp_path / "src" / "lint.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    plugin = _source_plugin()
    result = plugin.validate_candidates_against_checkout(
        {
            "problems": [
                {
                    "problem": "Remembered F401 lint failure",
                    "problem_type": "common",
                    "files": ["src/lint.py"],
                    "failure_signals": ["F401 unused import"],
                    "verification_cmd": "ruff check src/lint.py",
                }
            ]
        },
        tmp_path,
        command_runner=lambda command, timeout: {
            "returncode": 0,
            "output": "All checks passed!",
        },
    )

    assert result["problems"] == []
    validation = result["source_rejected_memory_problems"][0]["source_validation"]
    assert validation["decision"] == "rejected"
    assert validation["lightweight_check"]["status"] == "problem_absent"


def test_lightweight_command_requires_matching_failure_output(tmp_path):
    source = tmp_path / "src" / "lint.py"
    source.parent.mkdir(parents=True)
    source.write_text("import unused\n")
    plugin = _source_plugin()
    result = plugin.validate_candidates_against_checkout(
        {
            "problems": [
                {
                    "problem": "Remembered F401 lint failure",
                    "problem_type": "dependency",
                    "files": ["src/lint.py"],
                    "failure_signals": ["F401 unused import"],
                    "verification_cmd": "ruff check src/lint.py",
                }
            ]
        },
        tmp_path,
        command_runner=lambda command, timeout: {
            "returncode": 1,
            "output": "src/lint.py:1:8: F401 unused import",
        },
    )

    assert len(result["problems"]) == 1
    validation = result["problems"][0]["source_validation"]
    assert validation["decision"] == "confirmed"
    assert validation["lightweight_check"]["status"] == "failure_confirmed"


def test_unsafe_or_broad_commands_are_not_executed(tmp_path):
    plugin = _source_plugin()
    source = tmp_path / "src" / "feature.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = True\n")
    calls = []

    check = plugin._run_lightweight_verification(
        {
            "verification_cmd": "pytest",
            "files": ["src/feature.py"],
        },
        tmp_path,
        [source],
        lambda command, timeout: calls.append(command),
    )

    assert check["status"] == "skipped"
    assert calls == []
