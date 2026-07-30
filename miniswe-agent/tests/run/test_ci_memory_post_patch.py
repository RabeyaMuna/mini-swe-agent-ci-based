import json

from minisweagent.run.benchmarks.utils.memory_plugin import MemoryPlugin


def test_post_patch_save_builds_l1_from_generated_diff(tmp_path):
    plugin = MemoryPlugin(
        {
            "memory_enabled": True,
            "memory_backend": "json",
            "memory_ablation_levels": "L1+L2+L3",
        },
        str(tmp_path),
    )

    diff = """diff --git a/pkg/module.py b/pkg/module.py
index 1111111..2222222 100644
--- a/pkg/module.py
+++ b/pkg/module.py
@@ -1,2 +1,3 @@
 import os
+from typing import Any
 VALUE = 1
"""
    log_analysis = {
        "error_types": [
            {
                "category": "Type Checking",
                "subcategory": "missing-typing-import",
                "evidence": "Name 'Any' is not defined",
            }
        ],
        "error_context": ["mypy failed because Any was used without an import"],
        "failed_job": [{"command": "mypy pkg", "step": "type check"}],
        "relevant_files": [
            {
                "file": "pkg/module.py",
                "issue_type": "missing-typing-import",
                "reason": "Uses Any without importing it",
                "failed_cmd": "mypy pkg",
                "failed_tool": "mypy",
            }
        ],
    }

    plugin.save_memory_entry(
        task_id="issue-1",
        sha_fail="abc123",
        repo_name="demo",
        repo_owner="owner",
        workflow_path=".github/workflows/ci.yml",
        workflow="ci",
        log_analysis_result=log_analysis,
        changed_files_info=None,
        fault_localizer=None,
        patch_generator={"diff": diff},
    )

    failure_memory = json.loads((tmp_path / "failure_memory.json").read_text())
    repo_memory = json.loads((tmp_path / "repo_memory.json").read_text())
    cross_memory = json.loads((tmp_path / "cross_memory.json").read_text())

    assert len(failure_memory) == 1
    assert failure_memory[0]["file"] == "pkg/module.py"
    assert failure_memory[0]["fix_direction"] == "from typing import Any"
    assert failure_memory[0]["failure_reason"] == "Uses Any without importing it"
    assert repo_memory[0]["changed_files"] == ["pkg/module.py"]
    assert cross_memory[0]["changed_files"] == ["pkg/module.py"]
