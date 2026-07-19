#!/usr/bin/env python3
"""
Deterministic Diff Parser - Parse git diff into structured data

Instead of sending raw diff to LLM, pre-process it into clean JSON structure.
This prevents JSON parsing errors from code/quotes in strings.

Usage:
    from deterministic_diff_parser import parse_diff_to_structured

    structured = parse_diff_to_structured(diff_text)
    # structured = {
    #   "files": [
    #     {
    #       "path": "path/to/file.py",
    #       "changes": [
    #         {"line": 42, "before": "old code", "after": "new code"}
    #       ]
    #     }
    #   ]
    # }
"""

import re
from typing import Dict, List, Any


def parse_diff_to_structured(diff: str) -> Dict[str, Any]:
    """
    Parse git diff into structured format.

    Returns:
        {
          "files": [
            {
              "path": "path/to/file.py",
              "file_type": ".py",
              "changes": [
                {
                  "line": 42,
                  "before": "old code",
                  "after": "new code",
                  "change_type": "modified"  # or "added", "deleted"
                }
              ],
              "total_changes": 5
            }
          ],
          "total_files": 10,
          "total_changes": 150
        }
    """
    files = []
    current_file = None
    current_hunk_start = 0

    lines = diff.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # New file header
        if line.startswith('diff --git'):
            # Save previous file if exists
            if current_file:
                files.append(current_file)

            # Extract file path
            match = re.search(r'diff --git a/(.*?) b/', line)
            if match:
                file_path = match.group(1)
                file_type = _get_file_extension(file_path)
                current_file = {
                    "path": file_path,
                    "file_type": file_type,
                    "changes": [],
                    "total_changes": 0
                }

        # Hunk header (@@ -X,Y +A,B @@)
        elif line.startswith('@@'):
            match = re.match(r'@@ -(\d+),?\d* \+(\d+),?\d* @@', line)
            if match:
                current_hunk_start = int(match.group(2))  # New line number

        # Changed lines
        elif current_file:
            if line.startswith('-') and not line.startswith('---'):
                # Deletion
                before_code = line[1:]  # Remove '-' prefix

                # Check if next line is addition (modification)
                if i + 1 < len(lines) and lines[i + 1].startswith('+') and not lines[i + 1].startswith('+++'):
                    after_code = lines[i + 1][1:]  # Remove '+' prefix
                    current_file["changes"].append({
                        "line": current_hunk_start,
                        "before": before_code,
                        "after": after_code,
                        "change_type": "modified"
                    })
                    current_file["total_changes"] += 1
                    i += 1  # Skip next line since we processed it
                else:
                    # Pure deletion
                    current_file["changes"].append({
                        "line": current_hunk_start,
                        "before": before_code,
                        "after": "",
                        "change_type": "deleted"
                    })
                    current_file["total_changes"] += 1

            elif line.startswith('+') and not line.startswith('+++'):
                # Addition (not paired with deletion above)
                after_code = line[1:]
                current_file["changes"].append({
                    "line": current_hunk_start,
                    "before": "",
                    "after": after_code,
                    "change_type": "added"
                })
                current_file["total_changes"] += 1
                current_hunk_start += 1

            elif not line.startswith(('-', '+', '@', 'diff', 'index', '---', '+++')):
                # Context line (unchanged)
                current_hunk_start += 1

        i += 1

    # Save last file
    if current_file:
        files.append(current_file)

    total_changes = sum(f["total_changes"] for f in files)

    return {
        "files": files,
        "total_files": len(files),
        "total_changes": total_changes
    }


def _get_file_extension(file_path: str) -> str:
    """Extract file extension."""
    if '.' in file_path:
        return '.' + file_path.rsplit('.', 1)[1]
    return ""


def chunk_structured_diff(
    structured: Dict[str, Any],
    max_files_per_chunk: int = 10,
    dependency_graph: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Split structured diff into chunks.

    Strategy:
    1. If dependency_graph provided, use dependency-aware chunking (keep related files together)
    2. Otherwise, fall back to simple file-count chunking

    Args:
        structured: Output from parse_diff_to_structured()
        max_files_per_chunk: Maximum files per chunk
        dependency_graph: Optional dependency graph from dependency_detector.build_dependency_graph()

    Returns:
        List of chunks, each with:
        - files: List of file info dicts
        - total_files: Count
        - total_changes: Sum of changes
        - chunk_index: 1-indexed
        - dependency_cluster: List of file paths in this cluster (if using dependencies)
        - dependency_explanation: Human-readable dependency info (if using dependencies)
    """
    files = structured["files"]

    # Use dependency-aware chunking if graph provided
    if dependency_graph and dependency_graph.get("clusters"):
        return _chunk_by_dependency_clusters(
            files,
            dependency_graph,
            max_files_per_chunk
        )

    # Fall back to simple chunking
    return _chunk_by_file_count(files, max_files_per_chunk)


def _chunk_by_file_count(files: List[Dict], max_files_per_chunk: int) -> List[Dict[str, Any]]:
    """Simple chunking by file count (original logic)."""
    chunks = []

    for i in range(0, len(files), max_files_per_chunk):
        chunk_files = files[i:i + max_files_per_chunk]
        chunk_total_changes = sum(f["total_changes"] for f in chunk_files)

        chunks.append({
            "files": chunk_files,
            "total_files": len(chunk_files),
            "total_changes": chunk_total_changes,
            "chunk_index": len(chunks) + 1,
            "total_chunks": (len(files) + max_files_per_chunk - 1) // max_files_per_chunk
        })

    return chunks


def _chunk_by_dependency_clusters(
    files: List[Dict],
    dependency_graph: Dict[str, Any],
    max_files_per_chunk: int
) -> List[Dict[str, Any]]:
    """
    Dependency-aware chunking - keep related files together.

    Strategy:
    1. Group files by dependency cluster
    2. If cluster > max_files, split it (but warn)
    3. Keep dependencies within same chunk when possible
    """
    from dependency_detector import explain_dependencies

    chunks = []
    file_path_to_info = {f["path"]: f for f in files}

    # Process each dependency cluster
    for cluster in dependency_graph.get("clusters", []):
        # Get file info for files in this cluster
        cluster_files = [
            file_path_to_info[path]
            for path in cluster
            if path in file_path_to_info
        ]

        if not cluster_files:
            continue

        # If cluster fits in one chunk, keep it together
        if len(cluster_files) <= max_files_per_chunk:
            chunk_total_changes = sum(f["total_changes"] for f in cluster_files)

            chunks.append({
                "files": cluster_files,
                "total_files": len(cluster_files),
                "total_changes": chunk_total_changes,
                "chunk_index": len(chunks) + 1,
                "dependency_cluster": cluster,
                "dependency_explanation": explain_dependencies(cluster, dependency_graph),
            })
        else:
            # Cluster too large - split it (but keep track it's from same cluster)
            for i in range(0, len(cluster_files), max_files_per_chunk):
                sub_chunk_files = cluster_files[i:i + max_files_per_chunk]
                chunk_total_changes = sum(f["total_changes"] for f in sub_chunk_files)

                chunks.append({
                    "files": sub_chunk_files,
                    "total_files": len(sub_chunk_files),
                    "total_changes": chunk_total_changes,
                    "chunk_index": len(chunks) + 1,
                    "dependency_cluster": cluster,
                    "dependency_explanation": f"Part of larger cluster: {explain_dependencies(cluster, dependency_graph)}",
                    "is_partial_cluster": True,
                })

    # Add total_chunks to all
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk["total_chunks"] = total_chunks

    return chunks


def format_structured_for_llm(chunk: Dict[str, Any]) -> str:
    """
    Format structured chunk for LLM prompt.

    Shows file path, change counts, and 2-3 example changes so LLM can
    understand the type of changes (imports, types, formatting, etc.)
    """
    lines = []

    for file_info in chunk["files"]:
        file_path = file_info["path"]
        changes = file_info["changes"]

        if not changes:
            continue

        # Count change types
        modified = sum(1 for c in changes if c["change_type"] == "modified")
        added = sum(1 for c in changes if c["change_type"] == "added")
        deleted = sum(1 for c in changes if c["change_type"] == "deleted")

        # File header with counts
        lines.append(f"{file_path}: {len(changes)} changes ({modified}M, {added}A, {deleted}D)")

        # Show up to 3 example changes (helps LLM understand change type)
        examples = changes[:3]
        for change in examples:
            change_type = change.get("change_type", "")
            before = (change.get("before", "") or "")[:80]  # Limit to 80 chars
            after = (change.get("after", "") or "")[:80]

            if change_type == "modified" and before and after:
                lines.append(f"  - Modified: '{before}' -> '{after}'")
            elif change_type == "added" and after:
                lines.append(f"  - Added: '{after}'")
            elif change_type == "deleted" and before:
                lines.append(f"  - Deleted: '{before}'")

        lines.append("")  # Blank line between files

    return "\n".join(lines)


# Test function
def test_parser():
    """Test the diff parser."""
    sample_diff = """diff --git a/dev/pyproject.toml b/dev/pyproject.toml
index 2015a28c3fbe..d9d3b0c7703d 100644
--- a/dev/pyproject.toml
+++ b/dev/pyproject.toml
@@ -18,7 +18,7 @@ python = "^3.9"
 isort = "==5.13.2"
 black = { version = "==24.2.0" }
-#taplo = "==0.9.3"
+taplo = "==0.9.3"
 docformatter = "==1.7.5"
diff --git a/framework/py/flwr/common/exit/exit_code_test.py b/framework/py/flwr/common/exit/exit_code_test.py
index abc123..def456 100644
--- a/framework/py/flwr/common/exit/exit_code_test.py
+++ b/framework/py/flwr/common/exit/exit_code_test.py
@@ -10,6 +10,7 @@ import unittest

 def test_something():
+    # New comment
     pass
"""

    import json
    structured = parse_diff_to_structured(sample_diff)
    print("Structured diff:")
    print(json.dumps(structured, indent=2))

    print("\n" + "="*80)
    print("LLM-friendly format:")
    print("="*80)
    chunks = chunk_structured_diff(structured, max_files_per_chunk=5)
    for chunk in chunks:
        print(format_structured_for_llm(chunk))


if __name__ == "__main__":
    test_parser()
