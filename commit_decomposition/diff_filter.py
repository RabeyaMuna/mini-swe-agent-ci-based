#!/usr/bin/env python3
"""
diff_filter.py - Filter out irrelevant files from diffs before analysis
"""

import re
from typing import List


# Files/directories to ignore before CI relevance analysis.
IGNORED_PATTERNS = [
    r"^\.github/",  # GitHub workflow/action files
    r"(^|/)[^/]+\.json$",  # JSON files
    r"(^|/)[^/]+\.md$",  # Markdown files
]


def should_ignore_file(file_path: str) -> bool:
    """
    Check if a file should be ignored based on patterns

    Args:
        file_path: Path to check

    Returns:
        True if file should be ignored, False otherwise
    """
    for pattern in IGNORED_PATTERNS:
        if re.search(pattern, file_path):
            return True
    return False


def filter_diff(diff: str) -> str:
    """
    Filter out files from diff that match ignored patterns

    Args:
        diff: Full git diff string

    Returns:
        Filtered diff with ignored files removed
    """
    if not diff or diff.startswith("Error"):
        return diff

    # Split diff by file markers
    file_pattern = r"(diff --git a/(.*?) b/.*?\n)"
    parts = re.split(file_pattern, diff)

    filtered_parts = []
    i = 0

    while i < len(parts):
        if i == 0:
            # Header before first file
            filtered_parts.append(parts[i])
            i += 1
            continue

        if i + 2 < len(parts) and parts[i].startswith("diff --git"):
            # This is a file header
            file_header = parts[i]
            file_path = parts[i + 1]

            # Find the content for this file (everything until next diff or end)
            content_start = i + 2
            content_end = content_start

            # Find where this file's diff ends
            while content_end < len(parts) and not (
                content_end > i + 2 and parts[content_end].startswith("diff --git")
            ):
                content_end += 1

            # Get all content for this file
            file_content = "".join(parts[content_start:content_end])

            # Check if we should include this file
            if not should_ignore_file(file_path):
                filtered_parts.append(file_header)
                filtered_parts.append(file_path)
                filtered_parts.append(file_content)

            # Move to next file
            i = content_end
        else:
            i += 1

    return "".join(filtered_parts)


def _matches_target_file(file_path: str, target_files: List[str]) -> bool:
    """Return True when file_path matches one of the target paths."""
    normalized_path = file_path.strip().lstrip("/")
    for target in target_files:
        normalized_target = str(target or "").strip().lstrip("/")
        if not normalized_target:
            continue
        if normalized_path == normalized_target:
            return True
        if normalized_path.endswith(f"/{normalized_target}"):
            return True
        if normalized_path.endswith(normalized_target):
            return True
        if normalized_target.endswith(f"/{normalized_path}"):
            return True
    return False


def filter_diff_to_files(diff: str, target_files: List[str]) -> str:
    """
    Keep only diff sections for target files.

    Path matching allows prefixes such as framework/py/... to match CI log paths
    such as py/....
    """
    if not diff or diff.startswith("Error") or not target_files:
        return ""

    file_pattern = r"(diff --git a/(.*?) b/.*?\n)"
    parts = re.split(file_pattern, diff)

    filtered_parts = []
    i = 0

    while i < len(parts):
        if i == 0:
            i += 1
            continue

        if i + 2 < len(parts) and parts[i].startswith("diff --git"):
            file_header = parts[i]
            file_path = parts[i + 1]
            content_start = i + 2
            content_end = content_start

            while content_end < len(parts) and not (
                content_end > i + 2 and parts[content_end].startswith("diff --git")
            ):
                content_end += 1

            if _matches_target_file(file_path, target_files) and not should_ignore_file(
                file_path
            ):
                filtered_parts.append(file_header)
                filtered_parts.append(file_path)
                filtered_parts.append("".join(parts[content_start:content_end]))

            i = content_end
        else:
            i += 1

    return "".join(filtered_parts)


def get_filtered_file_count(diff: str) -> tuple:
    """
    Count total files and filtered files in diff

    Returns:
        (total_files, included_files, ignored_files)
    """
    if not diff or diff.startswith("Error"):
        return (0, 0, 0)

    # Find all file paths
    file_pattern = r"diff --git a/(.*?) b/"
    all_files = re.findall(file_pattern, diff)

    total_files = len(all_files)
    ignored_files = sum(1 for f in all_files if should_ignore_file(f))
    included_files = total_files - ignored_files

    return (total_files, included_files, ignored_files)


def get_changed_files(diff: str, include_ignored: bool = True) -> List[str]:
    """
    Extract changed file paths from a git diff.

    Args:
        diff: Full git diff string
        include_ignored: Include files that match ignored patterns

    Returns:
        Ordered list of changed file paths
    """
    if not diff or diff.startswith("Error"):
        return []

    file_pattern = r"diff --git a/(.*?) b/"
    files = re.findall(file_pattern, diff)
    if include_ignored:
        return files
    return [file_path for file_path in files if not should_ignore_file(file_path)]
