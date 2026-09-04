#!/usr/bin/env python3
"""
diff_filter.py - Filter out irrelevant files from diffs before analysis
"""

import re
from typing import List


# Files/directories to ignore before CI relevance analysis.
# IMPORTANT: Only ignore files that are NEVER validated by CI.
# Most files should be analyzed by LLM, not hard-filtered.
IGNORED_PATTERNS = [
    # DO NOT filter .github/ - workflow changes can break CI!
    # DO NOT filter .json - package.json, tsconfig.json affect builds!
    # DO NOT filter .md - README.md can affect docs validation!

    # Only filter truly irrelevant files:
    r"^\.git/",  # Git internal files (not .github!)
    r"(^|/)\.DS_Store$",  # macOS metadata
    r"(^|/)\.vscode/",  # Editor configs (unless CI validates them)
    # Add more ONLY if you're certain they're never validated by CI
]

# Binary file patterns to ignore (these cause GitHub push failures)
BINARY_PATTERNS = [
    r"\.dylib$",  # macOS dynamic libraries
    r"\.a$",  # Static libraries
    r"\.so$",  # Shared objects
    r"\.dll$",  # Windows libraries
    r"\.exe$",  # Executables
    r"\.bin$",  # Binary files
    r"\.o$",  # Object files
    r"bin/grype$",  # Specific large binary
]

# Large generated files to ignore (not needed for CI testing)
GENERATED_FILE_PATTERNS = [
    r"coverage.*\.json$",  # Coverage reports
    r"\.coverage$",  # Python coverage data
    r"htmlcov/",  # Coverage HTML reports
    r"node_modules/",  # Node dependencies
    r"vendor/",  # Vendor dependencies
    r"\.lock$",  # Lock files (package-lock.json, poetry.lock, etc.)
]


def is_binary_file(file_path: str) -> bool:
    """
    Check if a file is a binary file based on extension patterns

    Args:
        file_path: Path to check

    Returns:
        True if file is binary, False otherwise
    """
    for pattern in BINARY_PATTERNS:
        if re.search(pattern, file_path):
            return True
    return False


def is_generated_file(file_path: str) -> bool:
    """
    Check if a file is a generated/large file that should be skipped

    Args:
        file_path: Path to check

    Returns:
        True if file is generated/large, False otherwise
    """
    for pattern in GENERATED_FILE_PATTERNS:
        if re.search(pattern, file_path):
            return True
    return False


def should_ignore_file(file_path: str, skip_binaries: bool = True, skip_generated: bool = True) -> bool:
    """
    Check if a file should be ignored based on patterns

    Args:
        file_path: Path to check
        skip_binaries: Whether to skip binary files (default: True)
        skip_generated: Whether to skip generated/large files (default: True)

    Returns:
        True if file should be ignored, False otherwise
    """
    for pattern in IGNORED_PATTERNS:
        if re.search(pattern, file_path):
            return True

    if skip_binaries and is_binary_file(file_path):
        return True

    if skip_generated and is_generated_file(file_path):
        return True

    return False


def is_binary_patch(content: str) -> bool:
    """
    Check if diff content contains a binary patch

    Args:
        content: Diff content for a single file

    Returns:
        True if content contains binary patch, False otherwise
    """
    return "GIT binary patch" in content or "Binary files" in content


def filter_diff(diff: str, skip_binaries: bool = True) -> str:
    """
    Filter out files from diff that match ignored patterns

    Args:
        diff: Full git diff string
        skip_binaries: Whether to skip binary files (default: True)

    Returns:
        Filtered diff with ignored files and binary patches removed
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
            if not should_ignore_file(file_path, skip_binaries):
                # Also check if content is binary
                if skip_binaries and is_binary_patch(file_content):
                    # Skip binary patches
                    pass
                else:
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


def filter_diff_to_files(diff: str, target_files: List[str], skip_binaries: bool = True) -> str:
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

            file_content = "".join(parts[content_start:content_end])

            if _matches_target_file(file_path, target_files) and not should_ignore_file(
                file_path, skip_binaries
            ):
                # Also check if content is binary
                if skip_binaries and is_binary_patch(file_content):
                    # Skip binary patches
                    continue
                filtered_parts.append(file_header)
                filtered_parts.append(file_path)
                filtered_parts.append(file_content)

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


def get_changed_files(diff: str, include_ignored: bool = True, skip_binaries: bool = True) -> List[str]:
    """
    Extract changed file paths from a git diff.

    Args:
        diff: Full git diff string
        include_ignored: Include files that match ignored patterns
        skip_binaries: Skip binary files (default: True)

    Returns:
        Ordered list of changed file paths
    """
    if not diff or diff.startswith("Error"):
        return []

    file_pattern = r"diff --git a/(.*?) b/"
    files = re.findall(file_pattern, diff)
    if include_ignored:
        return files
    return [file_path for file_path in files if not should_ignore_file(file_path, skip_binaries)]
