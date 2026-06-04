"""
Test fuzzy repo name matching to handle format inconsistencies.

This addresses the issue where queries use "owner/repo" format
but memory records use "repo" format (or vice versa).
"""
import json
from minisweagent.run.benchmarks.utils.memory_plugin import MemoryPlugin, _repo_matches


def test_repo_matches_function():
    """Test the _repo_matches helper function"""
    # Exact matches
    assert _repo_matches("camel", "camel") is True
    assert _repo_matches("camel-ai/camel", "camel-ai/camel") is True

    # Full vs short format
    assert _repo_matches("camel-ai/camel", "camel") is True
    assert _repo_matches("camel", "camel-ai/camel") is True
    assert _repo_matches("owner/repo", "repo") is True
    assert _repo_matches("repo", "owner/repo") is True

    # Different repos
    assert _repo_matches("camel-ai/camel", "other/repo") is False
    assert _repo_matches("camel", "flower") is False
    assert _repo_matches("owner/repo1", "owner/repo2") is False

    # Empty strings
    assert _repo_matches("", "camel") is False
    assert _repo_matches("camel", "") is False
    assert _repo_matches("", "") is False


def test_l1_fuzzy_repo_matching(tmp_path):
    """L1 should match memories using fuzzy repo name matching"""
    plugin = MemoryPlugin(
        {
            "memory_enabled": True,
            "memory_backend": "json",
            "memory_ablation_levels": "L1+L2+L3",
            "memory_top_k": 10,
        },
        str(tmp_path),
    )

    # Create L1 memory with SHORT repo format (just "camel")
    memories = [
        {
            "sha_fail": "sha1",
            "repo": "camel",  # SHORT FORMAT
            "workflow_path": ".github/workflows/ci.yml",
            "file": "src/module.py",
            "error_type": "Import Error",
            "failure_pattern": "circular-import",
            "issue_type": "circular-import",
            "failed_tool": ["pytest"],
            "failed_cmd": ["pytest test"],
        },
    ]

    plugin.failure_memory = memories

    # Query with FULL repo format ("camel-ai/camel")
    query = {
        "task_id": "test-1",
        "sha_fail": "new_sha",
        "repo": "camel-ai/camel",  # FULL FORMAT
        "workflow_path": ".github/workflows/ci.yml",
        "error_type": "Import Error",
        "failure_pattern": "circular-import",
        "relevant_files": ["src/module.py"],
        "changed_files": [],
        "failed_tool": ["pytest"],
        "failed_cmd": ["pytest test"],
    }

    l1_results = plugin._retrieve_l1(query)

    # Should match despite format difference
    assert len(l1_results) == 1, f"Expected 1 L1 result (fuzzy match), got {len(l1_results)}"
    assert l1_results[0]["sha_fail"] == "sha1"
    assert l1_results[0]["repo"] == "camel"


def test_l1_reverse_fuzzy_matching(tmp_path):
    """L1 should also work when query is short and memory is full format"""
    plugin = MemoryPlugin(
        {
            "memory_enabled": True,
            "memory_backend": "json",
            "memory_ablation_levels": "L1+L2+L3",
            "memory_top_k": 10,
        },
        str(tmp_path),
    )

    # Create L1 memory with FULL repo format
    memories = [
        {
            "sha_fail": "sha1",
            "repo": "camel-ai/camel",  # FULL FORMAT
            "workflow_path": ".github/workflows/ci.yml",
            "file": "src/module.py",
            "error_type": "Import Error",
            "failure_pattern": "circular-import",
            "issue_type": "circular-import",
            "failed_tool": ["pytest"],
            "failed_cmd": ["pytest test"],
        },
    ]

    plugin.failure_memory = memories

    # Query with SHORT repo format
    query = {
        "task_id": "test-1",
        "sha_fail": "new_sha",
        "repo": "camel",  # SHORT FORMAT
        "workflow_path": ".github/workflows/ci.yml",
        "error_type": "Import Error",
        "failure_pattern": "circular-import",
        "relevant_files": ["src/module.py"],
        "changed_files": [],
        "failed_tool": ["pytest"],
        "failed_cmd": ["pytest test"],
    }

    l1_results = plugin._retrieve_l1(query)

    # Should match despite format difference
    assert len(l1_results) == 1
    assert l1_results[0]["repo"] == "camel-ai/camel"


def test_l2_fuzzy_repo_matching(tmp_path):
    """L2 should also use fuzzy repo matching"""
    plugin = MemoryPlugin(
        {
            "memory_enabled": True,
            "memory_backend": "json",
            "memory_ablation_levels": "L1+L2+L3",
            "memory_top_k": 10,
        },
        str(tmp_path),
    )

    # Create L2 memory with SHORT format
    memories = [
        {
            "sha_fail": "sha1",
            "repo": "camel",  # SHORT FORMAT
            "workflow_path": ".github/workflows/ci.yml",
            "error_type": "Import Error",
            "failure_pattern": "circular-import",
            "issue_type": "circular-import",
            "overall_failure_reason": "Circular import between modules",
            "files": [{"file": "src/a.py"}, {"file": "src/b.py"}],
            "failed_tool": ["pytest"],
            "failed_cmd": ["pytest test"],
        },
    ]

    plugin.repo_memory = memories

    # Query with FULL format
    query = {
        "task_id": "test-1",
        "sha_fail": "new_sha",
        "repo": "camel-ai/camel",  # FULL FORMAT
        "workflow_path": ".github/workflows/ci.yml",
        "error_type": "Import Error",
        "failure_pattern": "circular-import",
        "relevant_files_details": [{"file": "src/a.py"}],
        "failed_tool": ["pytest"],
        "failed_cmd": ["pytest test"],
        "failure_reason": "Circular import",
    }

    l2_results = plugin._retrieve_l2(query)

    # Should match despite format difference
    assert len(l2_results) == 1
    assert l2_results[0]["repo"] == "camel"


def test_fuzzy_matching_excludes_different_repos(tmp_path):
    """Fuzzy matching should still exclude actually different repos"""
    plugin = MemoryPlugin(
        {
            "memory_enabled": True,
            "memory_backend": "json",
            "memory_ablation_levels": "L1+L2+L3",
            "memory_top_k": 10,
        },
        str(tmp_path),
    )

    # Create memories for different repos
    memories = [
        {
            "sha_fail": "sha1",
            "repo": "camel",
            "workflow_path": ".github/workflows/ci.yml",
            "file": "src/module.py",
            "error_type": "Import Error",
            "failure_pattern": "circular-import",
            "issue_type": "circular-import",
            "failed_tool": ["pytest"],
            "failed_cmd": ["pytest test"],
        },
        {
            "sha_fail": "sha2",
            "repo": "flower",  # DIFFERENT REPO
            "workflow_path": ".github/workflows/ci.yml",
            "file": "src/module.py",
            "error_type": "Import Error",
            "failure_pattern": "circular-import",
            "issue_type": "circular-import",
            "failed_tool": ["pytest"],
            "failed_cmd": ["pytest test"],
        },
    ]

    plugin.failure_memory = memories

    # Query for camel repo
    query = {
        "task_id": "test-1",
        "sha_fail": "new_sha",
        "repo": "camel-ai/camel",
        "workflow_path": ".github/workflows/ci.yml",
        "error_type": "Import Error",
        "failure_pattern": "circular-import",
        "relevant_files": ["src/module.py"],
        "changed_files": [],
        "failed_tool": ["pytest"],
        "failed_cmd": ["pytest test"],
    }

    l1_results = plugin._retrieve_l1(query)

    # Should only match camel, NOT flower
    assert len(l1_results) == 1
    assert l1_results[0]["repo"] == "camel"
    assert l1_results[0]["sha_fail"] == "sha1"


if __name__ == "__main__":
    import tempfile

    print("Testing fuzzy repo matching...")

    print("\n1. Testing _repo_matches function...")
    test_repo_matches_function()
    print("✅ _repo_matches works correctly")

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        tmp_path = Path(tmpdir)

        print("\n2. Testing L1 fuzzy matching (full → short)...")
        test_l1_fuzzy_repo_matching(tmp_path)
        print("✅ L1 matches 'camel-ai/camel' query with 'camel' memory")

        print("\n3. Testing L1 reverse fuzzy matching (short → full)...")
        test_l1_reverse_fuzzy_matching(tmp_path)
        print("✅ L1 matches 'camel' query with 'camel-ai/camel' memory")

        print("\n4. Testing L2 fuzzy matching...")
        test_l2_fuzzy_repo_matching(tmp_path)
        print("✅ L2 uses fuzzy repo matching")

        print("\n5. Testing exclusion of different repos...")
        test_fuzzy_matching_excludes_different_repos(tmp_path)
        print("✅ Fuzzy matching correctly excludes different repos")

    print("\n" + "=" * 50)
    print("✅ All fuzzy repo matching tests passed!")
    print("\nThis fix addresses the issue where L1/L2 returned 0 results")
    print("due to 'camel-ai/camel' (query) != 'camel' (memory) mismatch.")
