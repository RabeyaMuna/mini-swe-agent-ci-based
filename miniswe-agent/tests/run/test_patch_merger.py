"""
Tests for patch_merger utility - merging duplicate file patches.
"""

import os
import subprocess
from pathlib import Path

import pytest

from utilities.git_patch import PatchValidationError, collect_workspace_patch
from minisweagent.run.benchmarks.utils.patch_merger import (
    filter_corrupted_patches,
    parse_unified_diff,
    group_patches_by_file,
    merge_duplicate_patches,
    detect_duplicate_patches,
    validate_patch,
)


def test_parse_unified_diff_single_file():
    """Test parsing a simple single-file diff."""
    diff = """diff --git a/file.py b/file.py
index 123..456 100644
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2_modified
 line3
"""
    patches = parse_unified_diff(diff)

    assert len(patches) == 1
    assert patches[0]['file'] == 'file.py'
    assert '@@' in patches[0]['hunks']


def test_parse_unified_diff_multiple_files():
    """Test parsing diff with multiple different files."""
    diff = """diff --git a/file1.py b/file1.py
index 111..222 100644
--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old1
+new1

diff --git a/file2.py b/file2.py
index 444..555 100644
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old2
+new2
"""
    patches = parse_unified_diff(diff)

    assert len(patches) == 2
    assert patches[0]['file'] == 'file1.py'
    assert patches[1]['file'] == 'file2.py'


def test_parse_unified_diff_duplicate_files():
    """Test parsing diff with multiple patches for the same file."""
    diff = """diff --git a/file.py b/file.py
index 111..222 100644
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new

diff --git a/file.py b/file.py
index 333..444 100644
--- a/file.py
+++ b/file.py
@@ -5 +5 @@
-old2
+new2
"""
    patches = parse_unified_diff(diff)

    assert len(patches) == 2
    assert patches[0]['file'] == 'file.py'
    assert patches[1]['file'] == 'file.py'


def test_group_patches_by_file():
    """Test grouping patches by file path."""
    patches = [
        {'file': 'a.py', 'full_patch': 'patch1'},
        {'file': 'b.py', 'full_patch': 'patch2'},
        {'file': 'a.py', 'full_patch': 'patch3'},
    ]

    grouped = group_patches_by_file(patches)

    assert len(grouped) == 2
    assert len(grouped['a.py']) == 2
    assert len(grouped['b.py']) == 1


def test_detect_duplicate_patches_no_duplicates():
    """Test detecting duplicates when there are none."""
    diff = """diff --git a/file1.py b/file1.py
index 111..222 100644
--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old
+new

diff --git a/file2.py b/file2.py
index 333..444 100644
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old2
+new2
"""
    duplicates = detect_duplicate_patches(diff)

    assert duplicates['file1.py'] == 1
    assert duplicates['file2.py'] == 1


def test_detect_duplicate_patches_with_duplicates():
    """Test detecting duplicates when they exist."""
    diff = """diff --git a/file.py b/file.py
index 111..222 100644
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new

diff --git a/file.py b/file.py
index 333..444 100644
--- a/file.py
+++ b/file.py
@@ -5 +5 @@
-old2
+new2

diff --git a/other.py b/other.py
index 666..777 100644
--- a/other.py
+++ b/other.py
@@ -1 +1 @@
-x
+y
"""
    duplicates = detect_duplicate_patches(diff)

    assert duplicates['file.py'] == 2  # Duplicate!
    assert duplicates['other.py'] == 1


def test_validate_patch_clean():
    """Test validating a clean patch (no duplicates)."""
    diff = """diff --git a/file.py b/file.py
index 111..222 100644
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
"""
    is_valid, message = validate_patch(diff)

    assert is_valid
    assert message == "OK"


def test_validate_patch_with_duplicates():
    """Test validating a patch with duplicates (should fail)."""
    diff = """diff --git a/file.py b/file.py
index 111..222 100644
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new

diff --git a/file.py b/file.py
index 333..444 100644
--- a/file.py
+++ b/file.py
@@ -5 +5 @@
-old2
+new2
"""
    is_valid, message = validate_patch(diff)

    assert not is_valid
    assert 'Duplicate patches' in message
    assert 'file.py' in message


def test_validate_patch_empty():
    """Test validating an empty patch."""
    is_valid, message = validate_patch("")

    assert not is_valid
    assert message == "Empty patch"


def test_merge_duplicate_patches_no_duplicates():
    """Test merging when there are no duplicates (should return original)."""
    diff = """diff --git a/file.py b/file.py
index 111..222 100644
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
"""
    merged = merge_duplicate_patches(diff)

    # Should be essentially unchanged (maybe minor formatting)
    assert 'diff --git a/file.py b/file.py' in merged
    assert '-old' in merged
    assert '+new' in merged


def git(repo, *args, input=None, check=True):
    return subprocess.run(
        ["git", *args], cwd=repo, input=input, capture_output=True, check=check
    )


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Patch Test")
    git(tmp_path, "config", "user.email", "patch@example.invalid")
    (tmp_path / "file.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "failed commit")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD").stdout.strip().decode()


def patch_for(repo, contents):
    (repo / "file.txt").write_text(contents)
    return git(repo, "diff", "--binary", "--full-index", "-U20").stdout.decode()


def apply_at_base(repo, base, patch):
    git(repo, "reset", "--hard", base)
    git(repo, "clean", "-fd")
    git(repo, "apply", "--check", "--3way", input=patch.encode())
    git(repo, "apply", "--3way", input=patch.encode())


@pytest.mark.parametrize("kind", ["cumulative", "sequential", "independent"])
def test_merge_preserves_all_edits(repository, kind):
    repo, base = repository
    original = (repo / "file.txt").read_text()
    first = original.replace("line2\n", "first fix\n")
    final = first.replace("line8\n", "second fix\n")
    p1 = patch_for(repo, first)
    if kind == "sequential":
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "first fix")
    p2 = patch_for(
        repo,
        final if kind != "independent" else original.replace("line8\n", "second fix\n"),
    )
    merged = merge_duplicate_patches(p1 + p2, repo, base)
    apply_at_base(repo, base, merged)
    assert (repo / "file.txt").read_text() == final


def test_conflicting_edits_are_reported_and_source_preserved(repository):
    repo, base = repository
    original = (repo / "file.txt").read_text()
    p1 = patch_for(repo, original.replace("line2", "first"))
    p2 = patch_for(repo, original.replace("line2", "conflicting"))
    before = (repo / "file.txt").read_bytes()
    index = (repo / ".git/index").read_bytes()
    with pytest.raises(PatchValidationError, match="could not be merged"):
        merge_duplicate_patches(p1 + p2, repo, base)
    assert (repo / "file.txt").read_bytes() == before
    assert (repo / ".git/index").read_bytes() == index


def test_duplicate_addition_preserves_quoted_paths(repository):
    repo, base = repository
    path = "quoted\tname.txt"
    (repo / path).write_text("Collecting package\n")
    patch = collect_workspace_patch(repo, base)
    assert parse_unified_diff(patch)[0]["file"] == path
    assert filter_corrupted_patches(patch) == patch
    merged = merge_duplicate_patches(patch + patch, repo, base)
    apply_at_base(repo, base, merged)
    assert (repo / path).read_text() == "Collecting package\n"


def test_all_workspace_changes_and_index_are_preserved(repository):
    repo, base = repository
    original = (repo / "file.txt").read_text()
    patch_for(repo, original.replace("line2", "committed"))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "agent commit")
    (repo / "staged.txt").write_bytes(b"staged\r\n")
    git(repo, "add", ".")
    (repo / "staged.txt").write_bytes(b"staged plus unstaged\r\n")
    (repo / "new.txt").write_bytes(b"no final newline")
    (repo / "file.txt").chmod(0o755)
    index = (repo / ".git/index").read_bytes()
    patch = collect_workspace_patch(repo, base)
    assert (repo / ".git/index").read_bytes() == index
    apply_at_base(repo, base, patch)
    assert (repo / "file.txt").read_text() == original.replace("line2", "committed")
    assert (repo / "staged.txt").read_bytes() == b"staged plus unstaged\r\n"
    assert (repo / "new.txt").read_bytes() == b"no final newline"
    assert (repo / "file.txt").stat().st_mode & 0o111


@pytest.mark.parametrize(
    "contents",
    [b"\0binary\xff\n", b"non-utf8 \xff\r\n", b"text\r\n\r\n", b"no newline"],
)
def test_file_bytes_round_trip(repository, contents):
    repo, base = repository
    (repo / "file.txt").write_bytes(contents)
    patch = collect_workspace_patch(repo, base)
    apply_at_base(repo, base, patch)
    assert (repo / "file.txt").read_bytes() == contents


def test_binary_patch_is_not_reformatted(repository):
    repo, base = repository
    (repo / "file.txt").write_bytes(b"\0binary\n")
    raw = git(
        repo, "diff", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", base
    ).stdout
    assert collect_workspace_patch(repo, base).encode() == raw


def test_rename_deletion_symlink_and_ignored_staged_file(repository):
    repo, base = repository
    git(repo, "mv", "file.txt", "renamed file.txt")
    (repo / "link").symlink_to("renamed file.txt")
    (repo / "tracked.log").write_text("keep me\n")
    git(repo, "add", "tracked.log")
    (repo / ".gitignore").write_text("*.log\n")
    (repo / "ignored.log").write_text("build artifact\n")
    patch = collect_workspace_patch(repo, base)
    apply_at_base(repo, base, patch)
    assert not (repo / "file.txt").exists()
    assert (repo / "renamed file.txt").exists()
    assert os.readlink(repo / "link") == "renamed file.txt"
    assert (repo / "tracked.log").read_text() == "keep me\n"
    (repo / "renamed file.txt").unlink()
    deleted = collect_workspace_patch(repo, base)
    apply_at_base(repo, base, deleted)
    assert not (repo / "file.txt").exists()
    assert not (repo / "renamed file.txt").exists()


def test_empty_final_tree_does_not_resurrect_an_earlier_commit(repository):
    repo, base = repository
    original = (repo / "file.txt").read_bytes()
    (repo / "file.txt").write_text("temporary fix\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "attempt")
    (repo / "file.txt").write_bytes(original)
    assert collect_workspace_patch(repo, base) == ""


def test_missing_base_and_invalid_patch_fail(repository):
    repo, base = repository
    with pytest.raises(PatchValidationError):
        collect_workspace_patch(repo, "missing-commit")
    assert validate_patch("not a patch\n")[0] is False
    patch = patch_for(repo, "changed\n")
    with pytest.raises(PatchValidationError, match="requires a repository"):
        merge_duplicate_patches(patch + patch)
    with pytest.raises(PatchValidationError):
        merge_duplicate_patches(patch + patch[:-3], repo, base)


def test_cibench_save_preserves_verified_binary_patch(repository):
    import json
    from minisweagent.run.benchmarks.cibench import (
        _collect_final_workspace_diff,
        _extract_diff,
        update_preds_file,
    )

    repo, base = repository
    (repo / "file.txt").write_bytes(b"\0binary\xff\n")
    patch = _collect_final_workspace_diff(repo, base)
    submitted = _extract_diff("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" + patch)
    assert submitted == patch
    output = repo.parent / (repo.name + "-preds.json")
    update_preds_file(output, "issue", base, submitted)
    saved = json.loads(output.read_text())["issue"]["diff"]
    assert saved == patch
    apply_at_base(repo, base, saved)
    assert (repo / "file.txt").read_bytes() == b"\0binary\xff\n"
    with pytest.raises(PatchValidationError):
        update_preds_file(output, "issue", base, "malformed patch\n")
    assert json.loads(output.read_text())["issue"]["diff"] == patch


def test_codex_save_captures_new_files_and_preserves_binary_patch(repository):
    import json
    from codex.scripts.run_codex_ci_repair import save_patch_and_result

    repo, base = repository
    (repo / "file.txt").write_bytes(b"\0binary\xff\n")
    (repo / "new.txt").write_text("new file\n")
    output = repo.parent / (repo.name + "-result")
    save_patch_and_result(
        output,
        repo,
        {"id": "issue", "repo": "owner/repo", "sha_fail": base},
        "baseline",
        "test",
        [],
        {},
        None,
        original_sha=base,
    )
    patch = (output / "patch.diff").read_bytes()
    result = json.loads((output / "result.json").read_text())
    assert result["changed_files"] == ["file.txt", "new.txt"]
    assert result["patch_bytes"] == len(patch)
    assert patch.decode() == collect_workspace_patch(repo, base)
    apply_at_base(repo, base, patch.decode())
    assert (repo / "file.txt").read_bytes() == b"\0binary\xff\n"
    assert (repo / "new.txt").read_text() == "new file\n"


def test_unresolved_then_resolved_merge_conflict(repository):
    repo, base = repository
    git(repo, "checkout", "-qb", "other")
    (repo / "file.txt").write_text("other side\n")
    git(repo, "commit", "-qam", "other change")
    git(repo, "checkout", "--detach", base)
    (repo / "file.txt").write_text("our side\n")
    git(repo, "commit", "-qam", "our change")
    assert git(repo, "merge", "other", check=False).returncode != 0
    with pytest.raises(PatchValidationError, match="merge conflicts"):
        collect_workspace_patch(repo, base)
    (repo / "file.txt").write_text("resolved with both changes\n")
    git(repo, "add", ".")
    patch = collect_workspace_patch(repo, base)
    apply_at_base(repo, base, patch)
    assert (repo / "file.txt").read_text() == "resolved with both changes\n"


def test_non_utf8_binary_fallback_overrides_text_attributes(repository):
    repo, base = repository
    (repo / ".gitattributes").write_text("*.txt diff\n")
    (repo / "file.txt").write_bytes(b"non-UTF-8 \xff\n")
    patch = collect_workspace_patch(repo, base)
    apply_at_base(repo, base, patch)
    assert (repo / "file.txt").read_bytes() == b"non-UTF-8 \xff\n"


def test_tree_comparison_rejects_applicable_patch_with_missing_change(
    repository, monkeypatch
):
    import utilities.git_patch as patches

    repo, base = repository
    partial = patch_for(repo, "first fix\n")
    (repo / "new.txt").write_text("second fix\n")
    monkeypatch.setattr(patches, "diff_trees", lambda *args: partial)
    with pytest.raises(PatchValidationError, match="complete final Git tree"):
        collect_workspace_patch(repo, base)
    assert (repo / "new.txt").read_text() == "second fix\n"


def test_split_index_is_unchanged(repository):
    repo, base = repository
    git(repo, "update-index", "--split-index")
    before = (repo / ".git/index").read_bytes()
    (repo / "file.txt").write_text("fixed\n")
    patch = collect_workspace_patch(repo, base)
    assert (repo / ".git/index").read_bytes() == before
    apply_at_base(repo, base, patch)
    assert (repo / "file.txt").read_text() == "fixed\n"


def test_dirty_submodule_is_not_silently_omitted(repository):
    repo, base = repository
    submodule = repo / "nested"
    submodule.mkdir()
    git(submodule, "init", "-q")
    git(submodule, "config", "user.name", "Patch Test")
    git(submodule, "config", "user.email", "patch@example.invalid")
    (submodule / "file.txt").write_text("original\n")
    git(submodule, "add", ".")
    git(submodule, "commit", "-qm", "original")
    (submodule / "file.txt").write_text("uncommitted fix\n")
    with pytest.raises(PatchValidationError, match="Uncommitted submodule changes"):
        collect_workspace_patch(repo, base)


def make_conflicting_checkout(repo, base):
    git(repo, "checkout", "-qb", "conflicting-side")
    (repo / "file.txt").write_text("other requirement\n")
    git(repo, "commit", "-qam", "other change")
    git(repo, "checkout", "--detach", base)
    (repo / "file.txt").write_text("our requirement\n")
    git(repo, "commit", "-qam", "our change")
    assert git(repo, "merge", "conflicting-side", check=False).returncode != 0


def test_reconciliation_only_calls_agent_for_actual_conflicts(repository):
    from utilities.patch_reconciliation import collect_with_reconciliation

    repo, base = repository
    calls = []
    (repo / "file.txt").write_text("valid fix\n")
    patch = collect_with_reconciliation(repo, base, lambda *args: calls.append(args))
    assert patch
    assert calls == []
    with pytest.raises(PatchValidationError):
        collect_with_reconciliation(repo, "missing-commit", lambda *args: calls.append(args))
    assert calls == []


def test_reconciliation_retries_and_preserves_other_repairs(repository):
    from utilities.patch_reconciliation import collect_with_reconciliation

    repo, base = repository
    make_conflicting_checkout(repo, base)
    (repo / "earlier.txt").write_text("earlier problem fixed\n")
    attempts = []

    def resolve(error, attempt):
        attempts.append(attempt)
        if attempt == 2:
            (repo / "file.txt").write_text("our requirement and other requirement\n")
            git(repo, "add", "file.txt")

    patch = collect_with_reconciliation(repo, base, resolve)
    assert attempts == [1, 2]
    apply_at_base(repo, base, patch)
    assert (repo / "file.txt").read_text() == "our requirement and other requirement\n"
    assert (repo / "earlier.txt").read_text() == "earlier problem fixed\n"


def test_reconciliation_exhaustion_never_saves_conflicted_patch(repository):
    from codex.scripts.run_codex_ci_repair import save_patch_and_result

    repo, base = repository
    make_conflicting_checkout(repo, base)
    before = (repo / "file.txt").read_bytes()
    attempts = []
    output = repo.parent / (repo.name + "-failed-result")
    with pytest.raises(PatchValidationError, match="merge conflicts"):
        save_patch_and_result(
            output, repo, {"id": "issue", "repo": "owner/repo", "sha_fail": base},
            "baseline", "test", [], {}, None, original_sha=base,
            resolve_conflict=lambda error, attempt: attempts.append(attempt),
        )
    assert attempts == [1, 2]
    assert not (output / "patch.diff").exists()
    assert (repo / "file.txt").read_bytes() == before


def test_codex_final_save_resolves_before_generating_patch(repository):
    import json
    from codex.scripts.run_codex_ci_repair import save_patch_and_result

    repo, base = repository
    make_conflicting_checkout(repo, base)
    records = []

    def resolve(error, attempt):
        (repo / "file.txt").write_text("both requirements resolved\n")
        git(repo, "add", "file.txt")
        records.append({"attempt": attempt, "returncode": 0})

    output = repo.parent / (repo.name + "-resolved-result")
    save_patch_and_result(
        output, repo, {"id": "issue", "repo": "owner/repo", "sha_fail": base},
        "baseline", "test", [], {}, None, original_sha=base,
        resolve_conflict=resolve, reconciliation_results=records,
    )
    assert json.loads((output / "result.json").read_text())["patch_reconciliation"] == records
    apply_at_base(repo, base, (output / "patch.diff").read_text())
    assert (repo / "file.txt").read_text() == "both requirements resolved\n"


def test_sequential_runner_returns_conflicts_to_agent(repository):
    import json
    from types import SimpleNamespace
    from minisweagent.run.benchmarks.cibench import _run_sequential_repair

    repo, base = repository
    problems = [{"problem_statement": "Preserve both requirements", "verification_cmd": "check-both"}]
    prompts = []

    class Agent:
        config = SimpleNamespace(wall_time_limit_seconds=42)
        _start_time = 123

        def run(self, prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                make_conflicting_checkout(repo, base)
                (repo / "earlier.txt").write_text("earlier fix\n")
            else:
                assert "ALL supplied problems" in prompt
                (repo / "file.txt").write_text("both requirements resolved\n")
                git(repo, "add", "file.txt")
            return {"exit_status": "submitted", "submission": "agent response"}

    agent = Agent()
    progress = SimpleNamespace(update_instance_status=lambda *args: None)
    info, patch = _run_sequential_repair(agent, problems, repo, progress, "issue", sha_fail=base)
    assert len(prompts) == 2
    assert agent.config.wall_time_limit_seconds == 42
    assert agent._start_time == 123
    assert len(info["sequential_repair"]["patch_reconciliation"]) == 1
    context = Path(info["sequential_repair"]["repair_record"]).parent / "problems.json"
    assert json.loads(context.read_text())["problems"] == problems
    apply_at_base(repo, base, patch)
    assert (repo / "file.txt").read_text() == "both requirements resolved\n"
    assert (repo / "earlier.txt").read_text() == "earlier fix\n"


def test_staging_and_committing_conflict_markers_is_not_a_resolution(repository):
    from utilities.patch_reconciliation import collect_with_reconciliation

    repo, base = repository
    make_conflicting_checkout(repo, base)
    attempts = []

    def resolve(error, attempt):
        attempts.append(attempt)
        if attempt == 1:
            git(repo, "add", "file.txt")
            git(repo, "commit", "-qm", "incorrectly staged unresolved markers")
        else:
            (repo / "file.txt").write_text("both fixes resolved\n")
            git(repo, "add", "file.txt")

    patch = collect_with_reconciliation(repo, base, resolve)
    assert attempts == [1, 2]
    apply_at_base(repo, base, patch)
    assert (repo / "file.txt").read_text() == "both fixes resolved\n"


def test_legitimate_conflict_marker_fixture_outside_a_merge_is_preserved(repository):
    repo, base = repository
    fixture = "<<<<<<< expected\nleft\n=======\nright\n>>>>>>> expected\n"
    (repo / "fixture.txt").write_text(fixture)
    patch = collect_workspace_patch(repo, base)
    apply_at_base(repo, base, patch)
    assert (repo / "fixture.txt").read_text() == fixture


def test_diagnosis_returns_original_failure_for_malformed_patch():
    from minisweagent.run.benchmarks.utils.patch_merger import diagnose_patch_failure

    result = diagnose_patch_failure("diff --git a/f b/f\n@@ -1 +1 @@\n-old\n", "error: corrupt patch at line 4")
    assert result["failure_type"] == "corrupt_patch"
    assert result["affected_files"] == []
    assert "_is_valid_patch_structure" not in result["suggested_fix"]


def test_patch_merger_imports_without_monorepo_utilities(tmp_path):
    import sys
    from minisweagent.run.benchmarks.utils import patch_merger

    src = Path(patch_merger.__file__).parents[4]
    script = f"import sys; sys.path.insert(0, {str(src)!r}); from minisweagent.run.benchmarks.utils import patch_merger"
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
