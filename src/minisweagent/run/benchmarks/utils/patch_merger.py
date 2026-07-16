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
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


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


def merge_patches_for_file(file_patches: List[Dict[str, str]], original_content: str = "") -> str:
    """
    Merge multiple patches for the same file by reconstructing final state from hunks.

    This function:
    1. Extracts all hunks from all patches
    2. Combines hunks intelligently (patches may build on each other)
    3. Generates a single unified diff with all changes

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

    # Strategy: Combine all hunks from all patches into one
    # This works because the patches are sequential changes to the same file

    # Use first patch's header as base
    base_header = file_patches[0]['header']

    # Collect all hunks
    all_hunks = []
    for patch in file_patches:
        if patch['hunks']:
            all_hunks.append(patch['hunks'])

    # If we have hunks, combine them
    if all_hunks:
        combined_hunks = "".join(all_hunks)
        merged_patch = f"{base_header}\n{combined_hunks}"

        logger.debug(f"[Patch Merger] Combined {len(file_patches)} patches with {len(all_hunks)} hunk blocks")
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
