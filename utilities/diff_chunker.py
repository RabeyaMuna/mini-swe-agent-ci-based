# utilities/diff_chunker.py
"""
Utilities for chunking large git diffs to fit within LLM token limits.

This module provides functions to:
1. Estimate token counts for text
2. Split large diffs by file boundaries
3. Summarize changes when diffs are too large
4. Decide which strategy to use based on diff size
"""

from typing import Dict, List, Tuple


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses rough heuristic: ~4 characters per token.
    This is approximate but good enough for deciding chunking strategy.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4


def chunk_diff_by_files(
    commit_diff: str, max_tokens_per_chunk: int = 12000
) -> List[Dict[str, any]]:
    """
    Split a large git diff into chunks, keeping file diffs together.

    This ensures each file's complete diff stays in one chunk,
    making it easier for LLMs to analyze files in context.

    Args:
        commit_diff: Full git diff output
        max_tokens_per_chunk: Maximum tokens per chunk (default 12000)

    Returns:
        List of chunks, where each chunk is a dict with:
            - diff: the diff text for this chunk
            - files: list of file paths in this chunk
            - tokens: estimated token count for this chunk

    Example:
        >>> diff = "diff --git a/file1.py...\\ndiff --git a/file2.py..."
        >>> chunks = chunk_diff_by_files(diff, max_tokens=5000)
        >>> len(chunks)
        2
        >>> chunks[0]['files']
        ['file1.py']
    """
    if not commit_diff:
        return []

    chunks = []
    current_chunk = {"diff": "", "files": [], "tokens": 0}

    current_file_diff = []
    current_file = None

    for line in commit_diff.split("\n"):
        if line.startswith("diff --git"):
            # Save previous file's diff
            if current_file and current_file_diff:
                file_diff_text = "\n".join(current_file_diff)
                file_tokens = estimate_tokens(file_diff_text)

                # Check if adding this file would exceed chunk limit
                if (
                    current_chunk["tokens"] + file_tokens > max_tokens_per_chunk
                    and current_chunk["files"]
                ):
                    # Start new chunk
                    chunks.append(current_chunk)
                    current_chunk = {"diff": "", "files": [], "tokens": 0}

                # Add file to current chunk
                current_chunk["diff"] += file_diff_text + "\n"
                current_chunk["files"].append(current_file)
                current_chunk["tokens"] += file_tokens

            # Parse new file path from diff header
            parts = line.split()
            if len(parts) >= 4:
                # Format: diff --git a/path/to/file b/path/to/file
                current_file = parts[3].lstrip("b/")
            else:
                current_file = None
            current_file_diff = [line]
        else:
            current_file_diff.append(line)

    # Save last file
    if current_file and current_file_diff:
        file_diff_text = "\n".join(current_file_diff)
        file_tokens = estimate_tokens(file_diff_text)

        if (
            current_chunk["tokens"] + file_tokens > max_tokens_per_chunk
            and current_chunk["files"]
        ):
            chunks.append(current_chunk)
            current_chunk = {"diff": "", "files": [], "tokens": 0}

        current_chunk["diff"] += file_diff_text + "\n"
        current_chunk["files"].append(current_file)
        current_chunk["tokens"] += file_tokens

    # Add last chunk if it has content
    if current_chunk["files"]:
        chunks.append(current_chunk)

    return chunks


def summarize_file_changes(file_path: str, diff_text: str) -> Dict[str, any]:
    """
    Summarize what changed in a file instead of showing full diff.

    Useful when diff is too large to include in prompt.

    Args:
        file_path: Path to the file
        diff_text: Diff text for this file

    Returns:
        Dict with:
            - file: file path
            - additions: number of added lines
            - deletions: number of deleted lines
            - changes: list of detected change types
            - sample_diff: first 300 chars of diff for context
    """
    lines = diff_text.split("\n")
    additions = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    deletions = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))

    changes = set()
    diff_lower = diff_text.lower()

    # Detect change types from diff content
    if "import" in diff_lower or "from " in diff_lower:
        changes.add("imports")
    if "def " in diff_text or "class " in diff_text:
        changes.add("definitions")
    if "-> " in diff_text:  # Type hints
        changes.add("type_hints")
    if "return" in diff_lower:
        changes.add("returns")
    if "assert" in diff_lower:
        changes.add("assertions")

    # Detect from file extension/path
    if file_path.endswith((".toml", ".ini", ".cfg", ".yaml", ".yml", ".json")):
        changes.add("config")
    if "test" in file_path.lower():
        changes.add("tests")
    if file_path.endswith((".txt", ".lock")) or "requirements" in file_path:
        changes.add("dependencies")
    if file_path.endswith(".md") or "readme" in file_path.lower():
        changes.add("docs")

    return {
        "file": file_path,
        "additions": additions,
        "deletions": deletions,
        "changes": list(changes) if changes else ["code"],
        "sample_diff": diff_text[:300],  # First 300 chars
    }


def summarize_commit_changes(
    commit_diff: str, changed_files: List[str], max_files: int = 30
) -> str:
    """
    Create a concise summary of all changes in a commit.

    Use this when commit diff is too large to include in prompt.

    Args:
        commit_diff: Full git diff
        changed_files: List of changed file paths
        max_files: Maximum files to include in summary

    Returns:
        Multi-line string summary like:
            - file1.py (+15, -8): imports, type_hints
            - file2.py (+3, -1): config
            ...
    """
    chunks = chunk_diff_by_files(commit_diff, max_tokens_per_chunk=100000)

    summaries = []
    for chunk in chunks:
        for file in chunk["files"]:
            # Extract this file's diff
            file_diff = extract_file_diff_from_chunk(chunk["diff"], file)
            summary = summarize_file_changes(file, file_diff)
            summaries.append(summary)

    # Format as readable lines
    lines = []
    for i, s in enumerate(summaries):
        if i >= max_files:
            lines.append(f"  ... and {len(summaries) - max_files} more files")
            break
        changes_str = ", ".join(s["changes"])
        lines.append(
            f"  - {s['file']} (+{s['additions']}, -{s['deletions']}): {changes_str}"
        )

    return "\n".join(lines)


def extract_file_diff_from_chunk(chunk_diff: str, file_path: str) -> str:
    """
    Extract diff for a specific file from a chunk.

    Args:
        chunk_diff: Diff text for entire chunk
        file_path: File to extract

    Returns:
        Diff text for just this file
    """
    lines = chunk_diff.split("\n")
    in_file = False
    file_lines = []

    for line in lines:
        if line.startswith("diff --git") and file_path in line:
            in_file = True
            file_lines.append(line)
        elif line.startswith("diff --git") and in_file:
            # Hit next file, stop
            break
        elif in_file:
            file_lines.append(line)

    return "\n".join(file_lines)


def decide_chunking_strategy(
    commit_diff: str, small_threshold: int = 10000, medium_threshold: int = 40000
) -> Tuple[str, any]:
    """
    Decide how to handle a diff based on its size.

    Three strategies:
    - 'full': Include entire diff in prompt (small diffs)
    - 'chunked': Split into chunks, process separately (medium diffs)
    - 'summarized': Use summary instead of full diff (huge diffs)

    Args:
        commit_diff: Full git diff
        small_threshold: Token limit for 'full' strategy (default 10k)
        medium_threshold: Token limit for 'chunked' strategy (default 40k)

    Returns:
        Tuple of (strategy_name, strategy_data) where:
        - strategy_name: 'full' | 'chunked' | 'summarized'
        - strategy_data:
            - For 'full': the diff itself
            - For 'chunked': list of chunk dicts
            - For 'summarized': None (caller should call summarize_commit_changes)

    Example:
        >>> diff = "..." # Large diff
        >>> strategy, data = decide_chunking_strategy(diff)
        >>> strategy
        'chunked'
        >>> len(data)  # Number of chunks
        6
    """
    total_tokens = estimate_tokens(commit_diff)

    if total_tokens < small_threshold:
        # Small: use full diff
        return ("full", commit_diff)

    elif total_tokens < medium_threshold:
        # Medium: chunk it
        chunks = chunk_diff_by_files(commit_diff, max_tokens_per_chunk=12000)
        return ("chunked", chunks)

    else:
        # Huge: caller should summarize
        return ("summarized", None)


def merge_groups(groups: List[Dict]) -> List[Dict]:
    """
    Merge groups with the same key attributes.

    Useful when processing chunks separately and need to merge results.
    Groups are merged when they have same:
    - validation_cmd
    - failure_type
    - issue_type

    Args:
        groups: List of group dicts from LLM responses

    Returns:
        Merged list with deduplicated files

    Example:
        >>> groups = [
        ...     {'files': ['a.py'], 'failure_type': 'test', 'validation_cmd': 'pytest'},
        ...     {'files': ['b.py'], 'failure_type': 'test', 'validation_cmd': 'pytest'},
        ... ]
        >>> merged = merge_groups(groups)
        >>> merged[0]['files']
        ['a.py', 'b.py']
    """
    merged = {}

    for group in groups:
        if not isinstance(group, dict):
            continue

        # Create key from identifying attributes
        key = (
            group.get("validation_cmd", ""),
            group.get("failure_type", ""),
            group.get("issue_type", ""),
        )

        if key not in merged:
            merged[key] = {
                "files": [],
                "failure_type": group.get("failure_type", ""),
                "issue_type": group.get("issue_type", ""),
                "validation_cmd": group.get("validation_cmd", ""),
                "reason": group.get("reason", ""),
            }

        # Add files (avoid duplicates)
        for f in group.get("files", []):
            if f not in merged[key]["files"]:
                merged[key]["files"].append(f)

    return list(merged.values())
