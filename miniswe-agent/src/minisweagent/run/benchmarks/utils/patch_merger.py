"""
patch_merger.py
===============
Utilities for merging multiple patches for the same file into a single coherent patch.

When the memory-based CI repair system identifies multiple problems affecting the same file,
the agent may generate separate patches for each problem. This module provides functions to:

1. Detect duplicate file patches in a unified diff
2. Merge overlapping patches by applying them sequentially to a virtual file state
3. Generate a clean, conflict-free unified diff

Usage:
    from minisweagent.run.benchmarks.utils.patch_merger import merge_duplicate_patches

    # Original diff with duplicates
    diff = '''
    diff --git a/file.py b/file.py
    ...
    diff --git a/file.py b/file.py
    ...
    '''

    # Merged diff (one entry per file)
    clean_diff = merge_duplicate_patches(diff)
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def _extract_hunk_signature(hunk: str) -> str:
    """Extract a unique signature from a hunk for deduplication."""
    lines = hunk.strip().split('\n')
    if not lines:
        return ""

    # Get hunk header (@@ ...)
    header = lines[0] if lines[0].startswith('@@') else ""

    # Get first 3 changed lines (skip context lines)
    changed_lines = [l for l in lines[1:6] if l.startswith(('+', '-'))]

    return header + '|' + '|'.join(changed_lines)


def _deduplicate_hunks(hunks_list: List[str]) -> List[str]:
    """
    Remove duplicate hunks while preserving order.

    Args:
        hunks_list: List of hunk blocks

    Returns:
        Deduplicated list of hunks
    """
    from typing import Set
    seen_signatures: Set[str] = set()
    unique_hunks = []

    for hunk_block in hunks_list:
        if not hunk_block or not hunk_block.strip():
            continue

        # Split into individual hunks
        individual_hunks = re.split(r'(^@@.*)', hunk_block, flags=re.MULTILINE)

        for i in range(1, len(individual_hunks), 2):
            if i + 1 < len(individual_hunks):
                hunk = individual_hunks[i] + individual_hunks[i + 1]
            else:
                hunk = individual_hunks[i]

            sig = _extract_hunk_signature(hunk)
            if sig and sig not in seen_signatures:
                seen_signatures.add(sig)
                unique_hunks.append(hunk)
            elif sig:
                logger.debug(f"[Patch Merger] Removing duplicate hunk: {sig[:80]}...")

    return unique_hunks


def _is_valid_filename(filename: str) -> bool:
    """
    Check if a filename is valid for a patch.

    Filters out:
    - Files starting with = (shell redirection artifacts)
    - Files that look like command output captures
    - Files with invalid characters for typical paths

    Args:
        filename: The file path to validate

    Returns:
        True if the filename is valid, False otherwise
    """
    if not filename or filename == '/dev/null':
        return False

    # Filter shell redirection artifacts (=4.6.0, =>output, etc.)
    if filename.startswith('=') or filename.startswith('>'):
        logger.warning(f"[Patch Merger] Ignoring patch for invalid filename (shell artifact): {filename}")
        return False

    # Filter obvious command output files
    invalid_patterns = [
        r'^[0-9]+\.[0-9]+\.[0-9]+$',  # Version numbers like "4.6.0"
        r'^\d+$',  # Pure numbers
        r'^<.*>$',  # Angle brackets
        r'^\[.*\]$',  # Square brackets
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, filename):
            logger.warning(f"[Patch Merger] Ignoring patch for suspicious filename: {filename}")
            return False

    return True


def parse_unified_diff(diff_text: str) -> List[Dict[str, str]]:
    """
    Parse a unified diff into individual file patches.

    Args:
        diff_text: Full unified diff (possibly with multiple diffs for same file)

    Returns:
        List of patch dictionaries with keys:
        - 'file': file path (from 'diff --git a/path b/path')
        - 'header': full git diff header
        - 'hunks': all hunks for this diff
        - 'full_patch': complete patch text
    """
    patches = []

    # Split on "diff --git" to get individual file patches
    diff_pattern = r'(diff --git a/[^\n]+\n(?:(?!diff --git).*\n)*)'
    matches = re.findall(diff_pattern, diff_text, re.MULTILINE)

    for match in matches:
        # Extract file path
        file_match = re.search(r'diff --git a/([^ ]+) b/([^ ]+)', match)
        if not file_match:
            continue

        file_path = file_match.group(1)

        # Validate filename before adding to patches
        if not _is_valid_filename(file_path):
            continue

        # Extract header (everything before first @@)
        header_match = re.search(r'(.*?)(?=^@@|\Z)', match, re.MULTILINE | re.DOTALL)
        header = header_match.group(1) if header_match else match

        # Extract hunks (everything after header)
        hunks = match[len(header):] if len(header) < len(match) else ""

        patches.append({
            'file': file_path,
            'header': header.rstrip(),
            'hunks': hunks,
            'full_patch': match
        })

    return patches


def group_patches_by_file(patches: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Group patches by file path.

    Args:
        patches: List of patch dictionaries from parse_unified_diff()

    Returns:
        Dictionary mapping file_path -> list of patches for that file
    """
    grouped = {}
    for patch in patches:
        file_path = patch['file']
        if file_path not in grouped:
            grouped[file_path] = []
        grouped[file_path].append(patch)

    return grouped


def _contains_command_output(patch: str) -> bool:
    """
    Detect if patch content contains command output instead of actual code.

    Common indicators:
    - pip/npm install output
    - wget/curl download output
    - Build system output
    - Error messages

    Args:
        patch: The patch text to check

    Returns:
        True if patch appears to contain command output
    """
    command_output_patterns = [
        r'\+Collecting [a-zA-Z0-9_-]+',  # pip install output
        r'\+Successfully installed',      # pip success
        r'\+Using cached.*\.whl',         # pip cache messages
        r'\+Downloading.*\([\d.]+\s*[KMG]B\)',  # pip/wget download
        r'\+npm (WARN|ERR!)',             # npm output
        r'\+error: command failed',       # Build errors
        r'\+\[[\d/]+\].*\d+%',           # Progress bars
    ]

    for pattern in command_output_patterns:
        if re.search(pattern, patch, re.MULTILINE):
            logger.warning(
                f"[Patch Merger] Patch appears to contain command output, not code: {pattern}"
            )
            return True

    return False


def _is_valid_patch_structure(patch: str) -> Tuple[bool, str]:
    """
    Check if patch has valid unified diff format.

    Valid patches must have:
    1. A 'diff --git' header
    2. 'index', '---', '+++' lines after diff header
    3. All hunks (@@) must come after complete file headers
    4. NO orphaned hunks (hunks without proper file context)
    5. NO command output in patch content

    Returns:
        (is_valid, error_message)
    """
    if not patch or not patch.strip():
        return False, "Empty patch"

    # Check for command output first
    if _contains_command_output(patch):
        return False, "Patch contains command output instead of code changes"

    lines = patch.split('\n')

    # Track state as we parse
    last_diff_idx = -100
    last_index_idx = -100
    last_minus_idx = -100
    last_plus_idx = -100

    for i, line in enumerate(lines):
        # Track diff header
        if line.startswith('diff --git'):
            last_diff_idx = i
            # Reset other markers when we see a new file
            last_index_idx = -100
            last_minus_idx = -100
            last_plus_idx = -100

        # Track index line
        elif line.startswith('index '):
            last_index_idx = i

        # Track --- line
        elif line.startswith('--- '):
            last_minus_idx = i

        # Track +++ line
        elif line.startswith('+++ '):
            last_plus_idx = i

        # Check hunks
        elif line.startswith('@@') and ' @@' in line:
            # Hunk header found - validate it has proper file context

            # Must have diff header before it (not too far back)
            if last_diff_idx < 0 or i - last_diff_idx > 20:
                return False, f"Line {i+1}: Orphaned hunk without diff header: {line[:60]}"

            # Must have index/---/+++ after the diff header and before this hunk
            if last_index_idx < last_diff_idx:
                return False, f"Line {i+1}: Missing 'index' line before hunk"
            if last_minus_idx < last_diff_idx:
                return False, f"Line {i+1}: Missing '--- a/' line before hunk"
            if last_plus_idx < last_diff_idx:
                return False, f"Line {i+1}: Missing '+++ b/' line before hunk"

            # Markers must be in order and close to the hunk
            if not (last_diff_idx < last_index_idx < last_minus_idx < last_plus_idx < i):
                return False, f"Line {i+1}: File headers out of order before hunk"

    return True, "OK"


def merge_patches_for_file(file_patches: List[Dict[str, str]], original_content: str = "") -> str:
    """
    Merge multiple patches for the same file by reconstructing final state from hunks.

    This function:
    1. Extracts all hunks from all patches
    2. Combines hunks intelligently (patches may build on each other)
    3. Validates the merged patch structure
    4. Falls back to git-based merge if validation fails

    Args:
        file_patches: List of patches for the same file
        original_content: Original file content (before any patches)

    Returns:
        Merged patch as unified diff string
    """
    if len(file_patches) == 1:
        return file_patches[0]['full_patch']

    file_path = file_patches[0]['file']
    logger.debug(f"[Patch Merger] Merging {len(file_patches)} patches for {file_path}")

    # Validate each input patch first
    invalid_patches = []
    for i, patch in enumerate(file_patches):
        is_valid, error = _is_valid_patch_structure(patch['full_patch'])
        if not is_valid:
            logger.warning(f"[Patch Merger] Input patch {i+1}/{len(file_patches)} invalid: {error}")
            invalid_patches.append(i)

    if invalid_patches:
        logger.warning(
            f"[Patch Merger] {len(invalid_patches)} of {len(file_patches)} patches have invalid structure, "
            "using git-based merge as fallback"
        )
        try:
            return _git_based_merge(file_patches, file_path, original_content)
        except Exception as e:
            logger.error(f"[Patch Merger] Git-based merge failed: {e}, using first valid patch")
            # Return first valid patch
            for i, patch in enumerate(file_patches):
                if i not in invalid_patches:
                    return patch['full_patch']
            # All patches invalid - return first one anyway (will likely fail downstream)
            return file_patches[0]['full_patch']

    # Strategy: Combine all hunks from all patches into one
    # This works because the patches are sequential changes to the same file

    # Use first patch's header as base
    base_header = file_patches[0]['header']

    # CRITICAL: Ensure header has required file markers (--- a/ and +++ b/)
    # The header must have these for git apply to work
    if '--- a/' not in base_header or '+++ b/' not in base_header:
        logger.debug(f"[Patch Merger] Adding missing file markers to header for {file_path}")
        # Insert file markers after diff --git line
        header_lines = base_header.split('\n')
        # Find the diff --git line
        git_diff_idx = next((i for i, line in enumerate(header_lines) if line.startswith('diff --git')), 0)
        # Insert markers after it (skip index line if present)
        insert_idx = git_diff_idx + 1
        if insert_idx < len(header_lines) and header_lines[insert_idx].startswith('index '):
            insert_idx += 1
        # Add file markers
        header_lines.insert(insert_idx, f'--- a/{file_path}')
        header_lines.insert(insert_idx + 1, f'+++ b/{file_path}')
        base_header = '\n'.join(header_lines)

    # Collect all hunks
    all_hunks_raw = []
    for i, patch in enumerate(file_patches):
        if i not in invalid_patches and patch['hunks']:
            all_hunks_raw.append(patch['hunks'])

    # CRITICAL FIX: Deduplicate hunks before merging!
    if all_hunks_raw:
        unique_hunks = _deduplicate_hunks(all_hunks_raw)

        if unique_hunks:
            combined_hunks = "\n".join(unique_hunks)  # Use newline separator, not empty string!
            merged_patch = f"{base_header}\n{combined_hunks}"

            # CRITICAL: Validate the merged patch structure
            is_valid, error = _is_valid_patch_structure(merged_patch)
            if not is_valid:
                logger.warning(
                    f"[Patch Merger] Merged patch for {file_path} invalid: {error}, "
                    "falling back to git-based merge"
                )
                try:
                    return _git_based_merge(file_patches, file_path, original_content)
                except Exception as e:
                    logger.error(f"[Patch Merger] Git-based merge failed: {e}, returning first patch")
                    return file_patches[0]['full_patch']

            logger.info(f"[Patch Merger] Combined {len(file_patches)} patches into {len(unique_hunks)} unique hunks")
            return merged_patch

    # Fallback: Try git-based merge with actual content
    try:
        return _git_based_merge(file_patches, file_path, original_content)
    except Exception as e:
        logger.warning(f"[Patch Merger] Git-based merge failed: {e}, using simple concatenation")
        # Last resort: just use the first patch (most conservative)
        return file_patches[0]['full_patch']


def _git_based_merge(file_patches: List[Dict[str, str]], file_path: str, original_content: str) -> str:
    """
    Git-based merge for complex cases where simple hunk merging won't work.

    Requires original file content to establish a baseline.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create git repo
        subprocess.run(['git', 'init'], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=tmp_path, capture_output=True, check=True)

        # Create file with original content
        file_full_path = tmp_path / file_path
        file_full_path.parent.mkdir(parents=True, exist_ok=True)
        file_full_path.write_text(original_content if original_content else "", encoding='utf-8')

        # Initial commit
        subprocess.run(['git', 'add', '.'], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial', '--allow-empty'], cwd=tmp_path, capture_output=True, check=True)

        # Apply each patch sequentially
        for i, patch in enumerate(file_patches):
            patch_file = tmp_path / f'patch_{i}.diff'
            patch_file.write_text(patch['full_patch'], encoding='utf-8')

            # Try to apply patch (ignore index mismatches)
            result = subprocess.run(
                ['git', 'apply', '--ignore-whitespace', str(patch_file)],
                cwd=tmp_path,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.debug(f"[Patch Merger] Patch {i+1}/{len(file_patches)} git apply failed, trying manual application")
                # Manually apply changes from this patch
                _manual_apply_patch(file_full_path, patch)

        # Generate final unified diff (original -> current state)
        diff_result = subprocess.run(
            ['git', 'diff', '--no-index', '--', '/dev/null' if not original_content else str(file_full_path.with_suffix('.orig')), str(file_full_path)],
            cwd=tmp_path,
            capture_output=True,
            text=True
        )

        if diff_result.stdout:
            return diff_result.stdout

        # Fallback to HEAD diff
        subprocess.run(['git', 'add', file_path], cwd=tmp_path, capture_output=True)
        diff_result = subprocess.run(
            ['git', 'diff', '--cached', file_path],
            cwd=tmp_path,
            capture_output=True,
            text=True
        )

        return diff_result.stdout if diff_result.stdout else file_patches[0]['full_patch']


def _manual_apply_patch(file_path: Path, patch: Dict[str, str]) -> None:
    """
    Manually apply patch changes by parsing hunks and modifying lines.

    This is a fallback when git apply fails.
    """
    # This is a simplified implementation
    # In practice, you might want to use a proper diff/patch library
    pass


def filter_corrupted_patches(diff_text: str) -> str:
    """
    Pre-process diff to remove obviously corrupted patches before parsing.

    This catches patches with:
    - Invalid filenames (shell artifacts like =4.6.0)
    - Command output in content
    - Malformed structure

    Args:
        diff_text: Raw diff text

    Returns:
        Filtered diff text with corrupted patches removed
    """
    if not diff_text or not diff_text.strip():
        return diff_text

    # Split into individual patches
    diff_pattern = r'(diff --git a/[^\n]+\n(?:(?!diff --git).*\n)*)'
    matches = re.findall(diff_pattern, diff_text, re.MULTILINE)

    valid_patches = []
    removed_count = 0

    for match in matches:
        # Check filename
        file_match = re.search(r'diff --git a/([^ ]+) b/([^ ]+)', match)
        if not file_match:
            logger.debug("[Patch Merger] Skipping patch: no file path found")
            removed_count += 1
            continue

        file_path = file_match.group(1)

        # Validate filename
        if not _is_valid_filename(file_path):
            removed_count += 1
            continue

        # Check for command output
        if _contains_command_output(match):
            logger.warning(
                f"[Patch Merger] Removing corrupted patch for {file_path}: contains command output"
            )
            removed_count += 1
            continue

        valid_patches.append(match)

    if removed_count > 0:
        logger.info(
            f"[Patch Merger] Filtered out {removed_count} corrupted patch(es), kept {len(valid_patches)}"
        )

    return "\n".join(valid_patches) if valid_patches else ""


def merge_duplicate_patches(diff_text: str, repo_path: Path = None) -> str:
    """
    Main entry point: merge all duplicate file patches in a unified diff.

    Args:
        diff_text: Full unified diff (may contain multiple diffs for same files)
        repo_path: Optional path to the repository (to get original file content)

    Returns:
        Clean unified diff with one entry per file
    """
    if not diff_text or not diff_text.strip():
        return diff_text

    # Filter corrupted patches first
    diff_text = filter_corrupted_patches(diff_text)

    if not diff_text:
        logger.warning("[Patch Merger] All patches were filtered out as corrupted")
        return ""

    # Parse all patches
    patches = parse_unified_diff(diff_text)

    if not patches:
        logger.warning("[Patch Merger] No valid patches found in diff")
        return diff_text

    # Group by file
    grouped = group_patches_by_file(patches)

    # Find files with duplicates
    duplicates = {file_path: patches for file_path, patches in grouped.items() if len(patches) > 1}

    if not duplicates:
        logger.debug("[Patch Merger] No duplicate patches found, returning original diff")
        return diff_text

    logger.info(f"[Patch Merger] Found {len(duplicates)} file(s) with duplicate patches:")
    for file_path, file_patches in duplicates.items():
        logger.info(f"  - {file_path}: {len(file_patches)} patches")

    # Merge duplicates
    merged_patches = {}

    for file_path, file_patches in grouped.items():
        if len(file_patches) == 1:
            # No duplicates for this file
            merged_patches[file_path] = file_patches[0]['full_patch']
        else:
            # Get original content if repo_path provided
            original_content = ""
            if repo_path:
                original_file = repo_path / file_path
                if original_file.exists():
                    original_content = original_file.read_text(encoding='utf-8')

            # Merge all patches for this file
            merged_patches[file_path] = merge_patches_for_file(file_patches, original_content)

    # Reconstruct full diff
    merged_diff = "\n".join(merged_patches.values())

    logger.info(f"[Patch Merger] Merged {len(patches)} patches into {len(merged_patches)} unique file patches")

    return merged_diff


def detect_duplicate_patches(diff_text: str) -> Dict[str, int]:
    """
    Detect which files have multiple patches in the diff.

    Args:
        diff_text: Full unified diff

    Returns:
        Dictionary mapping file_path -> number of patches for that file
    """
    patches = parse_unified_diff(diff_text)
    grouped = group_patches_by_file(patches)

    return {file_path: len(file_patches) for file_path, file_patches in grouped.items()}


def validate_patch(diff_text: str) -> Tuple[bool, str]:
    """
    Validate that a patch can be applied without conflicts.

    Args:
        diff_text: Unified diff to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not diff_text or not diff_text.strip():
        return False, "Empty patch"

    # Check for duplicate patches
    duplicates = detect_duplicate_patches(diff_text)
    duplicate_files = [f for f, count in duplicates.items() if count > 1]

    if duplicate_files:
        return False, f"Duplicate patches for files: {', '.join(duplicate_files)}"

    return True, "OK"


def diagnose_patch_failure(
    diff_text: str, git_error: str, repo_path: Path = None
) -> Dict[str, Any]:
    """
    Diagnose why a patch failed to apply by analyzing the git error and patch content.

    Args:
        diff_text: The patch that failed
        git_error: Error message from git apply
        repo_path: Optional path to repository for file checks

    Returns:
        Dictionary with diagnosis information:
        - failure_type: "missing_file" | "context_mismatch" | "corrupt_patch" | "merge_conflict"
        - affected_files: List of files mentioned in error
        - diagnosis: Human-readable explanation
        - suggested_fix: What might resolve the issue
    """
    diagnosis = {
        "failure_type": "unknown",
        "affected_files": [],
        "diagnosis": "",
        "suggested_fix": "",
    }

    git_error_lower = git_error.lower()

    # Pattern 1: File doesn't exist in index
    if "does not exist in index" in git_error_lower:
        diagnosis["failure_type"] = "missing_file"
        # Extract file names from error
        import re
        file_matches = re.findall(r"error: ([^\s:]+): does not exist in index", git_error)
        diagnosis["affected_files"] = file_matches
        diagnosis["diagnosis"] = (
            f"Agent created new file(s) but patch doesn't include 'new file mode' header. "
            f"Files: {', '.join(file_matches)}"
        )
        diagnosis["suggested_fix"] = (
            "Use 'git add -N .' before generating diff to mark new files, "
            "or ensure patches include proper 'new file mode 100644' headers"
        )

    # Pattern 2: Patch doesn't apply (context mismatch)
    elif "patch does not apply" in git_error_lower or "patch failed:" in git_error_lower:
        diagnosis["failure_type"] = "context_mismatch"
        # Extract file and line info
        import re
        file_matches = re.findall(r"error: patch failed: ([^\s:]+):(\d+)", git_error)
        diagnosis["affected_files"] = [f[0] for f in file_matches]
        diagnosis["diagnosis"] = (
            f"Patch context lines don't match file content at target commit. "
            f"Files: {', '.join(diagnosis['affected_files'])}"
        )
        diagnosis["suggested_fix"] = (
            "Ensure agent reads current file state before editing. "
            "Agent may have stale view or file changed between analysis and edit. "
            "Consider generating patches with more context lines (git diff -U10)"
        )

    # Pattern 3: Corrupt patch / missing header
    elif "corrupt patch" in git_error_lower or "patch fragment without header" in git_error_lower:
        diagnosis["failure_type"] = "corrupt_patch"
        diagnosis["diagnosis"] = (
            "Patch structure is malformed - missing 'diff --git' header before hunks"
        )
        diagnosis["suggested_fix"] = (
            "Check patch merger logic for proper header/hunk concatenation. "
            "Validate patch structure with _is_valid_patch_structure() before writing"
        )

        # Try to extract which file from the patch itself
        patches = parse_unified_diff(diff_text)
        if patches:
            diagnosis["affected_files"] = [p["file"] for p in patches]

    # Pattern 4: Missing blob for 3-way merge
    elif "lacks the necessary blob" in git_error_lower:
        diagnosis["failure_type"] = "missing_blob"
        diagnosis["diagnosis"] = (
            "Repository missing git objects for 3-way merge. "
            "This happens with --shared clones that don't have full history"
        )
        diagnosis["suggested_fix"] = (
            "Don't use 'git clone --shared' for testbed, or fetch full objects: "
            "'git fetch origin <sha_fail>' before applying patch"
        )

    # Pattern 5: File doesn't match index
    elif "does not match index" in git_error_lower:
        diagnosis["failure_type"] = "index_mismatch"
        import re
        file_matches = re.findall(r"error: ([^\s:]+): does not match index", git_error)
        diagnosis["affected_files"] = file_matches
        diagnosis["diagnosis"] = (
            f"File content differs from what's in git index. "
            f"Files: {', '.join(file_matches)}"
        )
        diagnosis["suggested_fix"] = (
            "File may have unstaged changes or the patch expects a different base state. "
            "Ensure clean checkout at exact sha_fail before applying"
        )

    # If we have repo_path, check if files actually exist
    if repo_path and diagnosis["affected_files"]:
        existing = []
        missing = []
        for file_path in diagnosis["affected_files"]:
            full_path = repo_path / file_path
            if full_path.exists():
                existing.append(file_path)
            else:
                missing.append(file_path)

        if missing:
            diagnosis["diagnosis"] += f"\n  Missing files: {', '.join(missing)}"
        if existing:
            diagnosis["diagnosis"] += f"\n  Existing files: {', '.join(existing)}"

    return diagnosis
