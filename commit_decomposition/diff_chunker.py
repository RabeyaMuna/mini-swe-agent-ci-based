#!/usr/bin/env python3
"""
diff_chunker.py - Split large diffs into manageable chunks
"""

import re
from typing import List, Dict


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count (rough approximation)
    ~1 token per 4 characters for English text
    """
    return len(text) // 4


def split_diff_by_files(diff: str) -> List[Dict]:
    """
    Split diff into per-file chunks

    Returns list of:
    {
        "file": "path/to/file.py",
        "diff": "diff content for this file",
        "tokens": estimated_tokens
    }
    """
    chunks = []

    # Split by file markers
    file_pattern = r'diff --git a/(.*?) b/(.*?)\n'
    parts = re.split(file_pattern, diff)

    current_file = None
    current_diff = ""

    for i, part in enumerate(parts):
        if i == 0:
            # Skip header
            continue

        if i % 3 == 1:
            # This is file path (a/...)
            if current_file and current_diff:
                # Save previous chunk
                chunks.append({
                    "file": current_file,
                    "diff": current_diff,
                    "tokens": count_tokens_estimate(current_diff)
                })

            current_file = part
            current_diff = f"diff --git a/{part} b/"

        elif i % 3 == 2:
            # This is file path (b/...)
            current_diff += f"{part}\n"

        elif i % 3 == 0:
            # This is the diff content
            current_diff += part

    # Add last chunk
    if current_file and current_diff:
        chunks.append({
            "file": current_file,
            "diff": current_diff,
            "tokens": count_tokens_estimate(current_diff)
        })

    return chunks


def split_large_file_diff(file_chunk: Dict, max_tokens: int = 8000) -> List[Dict]:
    """
    Split a single file's diff into smaller chunks by hunks

    If a file diff is too large, split it by hunks (@@ markers)
    """
    if file_chunk["tokens"] <= max_tokens:
        return [file_chunk]

    diff = file_chunk["diff"]
    file_path = file_chunk["file"]

    # Split by hunks
    hunk_pattern = r'(@@ -\d+,\d+ \+\d+,\d+ @@[^\n]*\n)'
    parts = re.split(hunk_pattern, diff)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    # Keep file header
    header = parts[0] if parts else ""

    for i, part in enumerate(parts[1:]):
        part_tokens = count_tokens_estimate(part)

        if current_tokens + part_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append({
                "file": file_path,
                "diff": header + current_chunk,
                "tokens": current_tokens,
                "hunk": len(chunks) + 1
            })
            current_chunk = part
            current_tokens = part_tokens
        else:
            current_chunk += part
            current_tokens += part_tokens

    # Add last chunk
    if current_chunk:
        chunks.append({
            "file": file_path,
            "diff": header + current_chunk,
            "tokens": current_tokens,
            "hunk": len(chunks) + 1
        })

    return chunks


def chunk_commit_diff(diff: str, max_tokens: int = 12000, max_files_per_chunk: int = 40) -> List[Dict]:
    """
    Main function: Split commit diff into chunks that fit within token limit

    Strategy:
    - Max 40 files per chunk
    - Max 12k tokens per chunk
    - Each chunk contains complete hunks (no partial changes)

    Returns list of chunks, each with:
    {
        "files": ["file1.py", "file2.py", ...],  # List of files in this chunk
        "diff": "diff content",
        "tokens": estimated_tokens,
        "file_count": number of files
    }
    """
    if not diff or diff.startswith("Error"):
        return []

    # First split by files
    file_chunks = split_diff_by_files(diff)

    if not file_chunks:
        return []

    # Group files into chunks (max 40 files, max 12k tokens each)
    final_chunks = []
    current_chunk_files = []
    current_chunk_diff = ""
    current_chunk_tokens = 0

    for file_chunk in file_chunks:
        file_tokens = file_chunk["tokens"]

        # If single file is too large, split it by hunks
        if file_tokens > max_tokens:
            # Save current chunk if any
            if current_chunk_files:
                final_chunks.append({
                    "files": current_chunk_files.copy(),
                    "diff": current_chunk_diff,
                    "tokens": current_chunk_tokens,
                    "file_count": len(current_chunk_files)
                })
                current_chunk_files = []
                current_chunk_diff = ""
                current_chunk_tokens = 0

            # Split large file by hunks
            sub_chunks = split_large_file_diff(file_chunk, max_tokens)
            for sub_chunk in sub_chunks:
                final_chunks.append({
                    "files": [sub_chunk["file"]],
                    "diff": sub_chunk["diff"],
                    "tokens": sub_chunk["tokens"],
                    "file_count": 1,
                    "hunk": sub_chunk.get("hunk")
                })
            continue

        # Check if adding this file would exceed limits
        would_exceed_tokens = (current_chunk_tokens + file_tokens) > max_tokens
        would_exceed_files = len(current_chunk_files) >= max_files_per_chunk

        if would_exceed_tokens or would_exceed_files:
            # Save current chunk
            if current_chunk_files:
                final_chunks.append({
                    "files": current_chunk_files.copy(),
                    "diff": current_chunk_diff,
                    "tokens": current_chunk_tokens,
                    "file_count": len(current_chunk_files)
                })

            # Start new chunk with this file
            current_chunk_files = [file_chunk["file"]]
            current_chunk_diff = file_chunk["diff"]
            current_chunk_tokens = file_tokens
        else:
            # Add to current chunk
            current_chunk_files.append(file_chunk["file"])
            current_chunk_diff += "\n" + file_chunk["diff"]
            current_chunk_tokens += file_tokens

    # Add last chunk if any
    if current_chunk_files:
        final_chunks.append({
            "files": current_chunk_files.copy(),
            "diff": current_chunk_diff,
            "tokens": current_chunk_tokens,
            "file_count": len(current_chunk_files)
        })

    return final_chunks
