"""
Test workflow filtering in memory retrieval (L1 and L2)
This test verifies that L1 and L2 properly filter by workflow,
while L3 remains generalized across all workflows.
"""
import json
from minisweagent.run.benchmarks.utils.memory_plugin import MemoryPlugin


def test_l1_filters_by_workflow(tmp_path):
    """L1 should only return memories from the same repo AND workflow"""
    plugin = MemoryPlugin(
        {
            "memory_enabled": True,
            "memory_backend": "json",
            "memory_ablation_levels": "L1+L2+L3",
            "memory_top_k": 10,
        },
        str(tmp_path),
    )

    # Create L1 memories from different workflows in the same repo
    memories = [
        {
            "sha_fail": "sha1",
            "repo": "owner/demo",
            "workflow_path": ".github/workflows/ci.yml",
            "file": "src/module.py",
            "error_type": "Type Checking",
            "failure_pattern": "missing-import",
            "issue_type": "missing-import",
            "failed_tool": ["mypy"],
            "failed_cmd": ["mypy src"],
        },
        {
            "sha_fail": "sha2",
            "repo": "owner/demo",
            "workflow_path": ".github/workflows/test.yml",  # Different workflow
            "file": "src/module.py",
            "error_type": "Type Checking",
            "failure_pattern": "missing-import",
            "issue_type": "missing-import",
            "failed_tool": ["mypy"],
            "failed_cmd": ["mypy src"],
        },
        {
            "sha_fail": "sha3",
            "repo": "owner/other-repo",  # Different repo
            "workflow_path": ".github/workflows/ci.yml",
            "file": "src/module.py",
            "error_type": "Type Checking",
            "failure_pattern": "missing-import",
            "issue_type": "missing-import",
            "failed_tool": ["mypy"],
            "failed_cmd": ["mypy src"],
        },
    ]

    # Save memories to failure_memory.json
    (tmp_path / "failure_memory.json").write_text(json.dumps(memories))
    plugin.failure_memory = memories

    # Query for memories with specific repo and workflow
    query = {
        "task_id": "test-1",
        "sha_fail": "new_sha",
        "repo": "owner/demo",
        "workflow_path": ".github/workflows/ci.yml",
        "error_type": "Type Checking",
        "failure_pattern": "missing-import",
        "relevant_files": ["src/module.py"],
        "changed_files": [],
        "failed_tool": ["mypy"],
        "failed_cmd": ["mypy src"],
    }

    # Retrieve L1 memories
    l1_results = plugin._retrieve_l1(query)

    # Should only return sha1 (same repo AND workflow)
    # sha2 excluded: different workflow
    # sha3 excluded: different repo
    assert len(l1_results) == 1, f"Expected 1 L1 result, got {len(l1_results)}: {[r.get('sha_fail') for r in l1_results]}"
    assert l1_results[0]["sha_fail"] == "sha1"
    assert l1_results[0]["workflow_path"] == ".github/workflows/ci.yml"


def test_l2_filters_by_workflow(tmp_path):
    """L2 should only return memories from the same repo AND workflow"""
    plugin = MemoryPlugin(
        {
            "memory_enabled": True,
            "memory_backend": "json",
            "memory_ablation_levels": "L1+L2+L3",
            "memory_top_k": 10,
        },
        str(tmp_path),
    )

    # Create L2 memories from different workflows
    memories = [
        {
            "sha_fail": "sha1",
            "repo": "owner/demo",
            "workflow_path": ".github/workflows/ci.yml",
            "error_type": "Type Checking",
            "failure_pattern": "type-error",
            "issue_type": "type-error",
            "overall_failure_reason": "Missing type imports",
            "files": [{"file": "src/module.py"}],
            "failed_tool": ["mypy"],
            "failed_cmd": ["mypy src"],
        },
        {
            "sha_fail": "sha2",
            "repo": "owner/demo",
            "workflow_path": ".github/workflows/deploy.yml",  # Different workflow
            "error_type": "Type Checking",
            "failure_pattern": "type-error",
            "issue_type": "type-error",
            "overall_failure_reason": "Missing type imports",
            "files": [{"file": "src/module.py"}],
            "failed_tool": ["mypy"],
            "failed_cmd": ["mypy src"],
        },
    ]

    (tmp_path / "repo_memory.json").write_text(json.dumps(memories))
    plugin.repo_memory = memories

    query = {
        "task_id": "test-1",
        "sha_fail": "new_sha",
        "repo": "owner/demo",
        "workflow_path": ".github/workflows/ci.yml",
        "error_type": "Type Checking",
        "failure_pattern": "type-error",
        "relevant_files_details": [{"file": "src/module.py"}],
        "failed_tool": ["mypy"],
        "failed_cmd": ["mypy src"],
        "failure_reason": "Missing type imports",
    }

    l2_results = plugin._retrieve_l2(query)

    # Should only return sha1 (same workflow)
    assert len(l2_results) == 1, f"Expected 1 L2 result, got {len(l2_results)}"
    assert l2_results[0]["sha_fail"] == "sha1"
    assert l2_results[0]["workflow_path"] == ".github/workflows/ci.yml"


def test_l3_no_workflow_filtering(tmp_path):
    """L3 should return memories from ALL repos and workflows (generalized)"""
    plugin = MemoryPlugin(
        {
            "memory_enabled": True,
            "memory_backend": "json",
            "memory_ablation_levels": "L1+L2+L3",
            "memory_top_k": 10,
        },
        str(tmp_path),
    )

    # Create L3 memories from different repos and workflows
    memories = [
        {
            "error_type": "Type Checking",
            "issue_type": "missing-import",
            "principle": "Always import types before use",
            "failure_patterns": ["missing-import"],
            "failure_reasons": ["Type not imported"],
            "example_files": [
                {"file": "src/a.py", "repo": "owner/repo1"},
                {"file": "src/b.py", "repo": "owner/repo2"},
            ],
            "failed_tool": ["mypy"],
        },
        {
            "error_type": "Linting",
            "issue_type": "unused-import",
            "principle": "Remove unused imports",
            "failure_patterns": ["unused-import"],
            "failure_reasons": ["Import not used"],
            "example_files": [{"file": "src/c.py", "repo": "owner/repo3"}],
            "failed_tool": ["ruff"],
        },
    ]

    (tmp_path / "cross_memory.json").write_text(json.dumps(memories))
    plugin.cross_memory = memories

    query = {
        "task_id": "test-1",
        "sha_fail": "new_sha",
        "repo": "owner/demo",  # Different repo
        "workflow_path": ".github/workflows/ci.yml",  # Different workflow
        "error_type": "Type Checking",
        "failure_pattern": "missing-import",
        "relevant_files_details": [{"file": "src/module.py"}],
        "failed_tool": ["mypy"],
        "failed_cmd": ["mypy src"],
        "failure_reason": "Type not imported",
    }

    l3_results = plugin._retrieve_l3(query)

    # Should return matching L3 memories regardless of repo/workflow
    # The Type Checking one should match better than Linting
    assert len(l3_results) >= 1, f"Expected at least 1 L3 result, got {len(l3_results)}"
    # First result should be the Type Checking one (better semantic match)
    assert l3_results[0]["error_type"] == "Type Checking"


def test_workflow_filtering_with_missing_workflow_field(tmp_path):
    """Test backward compatibility when workflow field is missing"""
    plugin = MemoryPlugin(
        {
            "memory_enabled": True,
            "memory_backend": "json",
            "memory_ablation_levels": "L1+L2+L3",
            "memory_top_k": 10,
        },
        str(tmp_path),
    )

    # Create L1 memory without workflow field (backward compatibility)
    memories = [
        {
            "sha_fail": "sha1",
            "repo": "owner/demo",
            # No workflow_path field
            "file": "src/module.py",
            "error_type": "Type Checking",
            "failure_pattern": "missing-import",
            "issue_type": "missing-import",
            "failed_tool": ["mypy"],
            "failed_cmd": ["mypy src"],
        },
    ]

    plugin.failure_memory = memories

    # Query WITH workflow (should still return the record since filtering is skipped when empty)
    query = {
        "task_id": "test-1",
        "sha_fail": "new_sha",
        "repo": "owner/demo",
        "workflow_path": ".github/workflows/ci.yml",  # Has workflow
        "error_type": "Type Checking",
        "failure_pattern": "missing-import",
        "relevant_files": ["src/module.py"],
        "changed_files": [],
        "failed_tool": ["mypy"],
        "failed_cmd": ["mypy src"],
    }

    l1_results = plugin._retrieve_l1(query)

    # Should return the memory even though it lacks workflow field
    # (backward compatibility - if row has no workflow, filtering is skipped)
    assert len(l1_results) == 1
    assert l1_results[0]["sha_fail"] == "sha1"


if __name__ == "__main__":
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path_obj = type("Path", (), {"__truediv__": lambda s, p: os.path.join(tmpdir, p)})()

        print("Testing L1 workflow filtering...")
        test_l1_filters_by_workflow(tmp_path_obj)
        print("OK L1 workflow filtering works correctly")

        print("\nTesting L2 workflow filtering...")
        test_l2_filters_by_workflow(tmp_path_obj)
        print("OK L2 workflow filtering works correctly")

        print("\nTesting L3 no filtering...")
        test_l3_no_workflow_filtering(tmp_path_obj)
        print("OK L3 generalized search works correctly")

        print("\nTesting backward compatibility...")
        test_workflow_filtering_with_missing_workflow_field(tmp_path_obj)
        print("OK Backward compatibility maintained")

        print("\n" + "="*50)
        print("OK All workflow filtering tests passed!")
