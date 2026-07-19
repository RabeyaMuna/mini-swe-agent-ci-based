#!/usr/bin/env python3
"""
Dependency Detector - Build file dependency graph from diff.

This module extracts dependencies between changed files to enable
dependency-aware chunking and analysis.
"""

from __future__ import annotations

import re
from typing import Any


def build_dependency_graph(structured_diff: dict[str, Any]) -> dict[str, Any]:
    """
    Build dependency graph from structured diff.

    Args:
        structured_diff: Output from parse_diff_to_structured()

    Returns:
        {
            "nodes": {
                "file.py": {
                    "type": "code",
                    "imports": ["module.py"],
                    "reads": [],
                    "tests": [],
                    "configures": []
                }
            },
            "edges": [
                {"from": "test.py", "to": "file.py", "type": "tests"},
                {"from": "test.py", "to": "data.rst", "type": "reads"}
            ],
            "clusters": [
                ["test.py", "file.py", "data.rst"]  # Files that should be analyzed together
            ]
        }
    """
    nodes = {}
    edges = []

    # Step 1: Extract dependencies for each file
    for file_info in structured_diff.get("files", []):
        file_path = file_info["path"]
        changes = file_info.get("changes", [])

        # Detect file type
        file_type = _classify_file_type(file_path)

        # Extract dependencies
        deps = _extract_dependencies(file_path, changes)

        nodes[file_path] = {"type": file_type, **deps}

        # Create edges
        for dep_type in ["imports", "reads", "tests", "configures"]:
            for dep_file in deps.get(dep_type, []):
                edges.append({"from": file_path, "to": dep_file, "type": dep_type})

    # Step 2: Build dependency clusters
    clusters = _build_dependency_clusters(nodes, edges)

    return {"nodes": nodes, "edges": edges, "clusters": clusters}


def _classify_file_type(file_path: str) -> str:
    """Classify file by type."""
    if file_path.endswith("_test.py") or "/tests/" in file_path:
        return "test"
    elif file_path.endswith((".toml", ".yaml", ".yml", ".json", ".ini", ".cfg")):
        return "config"
    elif file_path.endswith((".rst", ".md")):
        return "docs"
    elif file_path.endswith(".py"):
        return "code"
    else:
        return "other"


def _extract_dependencies(file_path: str, changes: list[dict]) -> dict[str, list[str]]:
    """
    Extract dependencies from file changes.

    Returns:
        {
            "imports": ["module.py"],       # Python imports
            "reads": ["data.rst"],          # Files read at runtime
            "tests": ["source.py"],         # Files this test tests
            "configures": ["*.py"]          # Files this config affects
        }
    """
    deps = {"imports": [], "reads": [], "tests": [], "configures": []}

    # Combine all changes into single text for analysis
    all_code = "\n".join(
        [change.get("after", "") for change in changes if change.get("after")]
    )

    # Detect imports
    if file_path.endswith(".py"):
        deps["imports"] = _extract_python_imports(all_code)

    # Detect file reads
    deps["reads"] = _extract_file_reads(all_code, file_path)

    # Detect test relationships
    if "_test.py" in file_path or "/tests/" in file_path:
        deps["tests"] = _infer_tested_files(file_path)

    # Detect config affects
    if file_path.endswith((".toml", ".yaml", ".yml")):
        deps["configures"] = _infer_config_affects(file_path, all_code)

    return deps


def _extract_python_imports(code: str) -> list[str]:
    """Extract Python imports from code."""
    imports = []

    # Match: from X import Y, import X
    patterns = [
        r"from\s+([\w.]+)\s+import",  # from module import
        r"import\s+([\w.]+)",  # import module
    ]

    for pattern in patterns:
        matches = re.findall(pattern, code)
        imports.extend(matches)

    return list(set(imports))


def _extract_file_reads(code: str, current_file: str) -> list[str]:
    """Extract file paths that are read at runtime."""
    reads = []

    # Pattern 1: Path("file.rst").read_text()
    path_patterns = re.findall(r'Path\(["\']([^"\']+)["\']\)', code)
    reads.extend(path_patterns)

    # Pattern 2: glob("*.rst")
    glob_patterns = re.findall(r'glob\(["\']([^"\']+)["\']\)', code)
    reads.extend(glob_patterns)

    # Pattern 3: open("file.txt")
    open_patterns = re.findall(r'open\(["\']([^"\']+)["\']\)', code)
    reads.extend(open_patterns)

    # Expand glob patterns to actual files (simple heuristic)
    expanded = []
    for pattern in reads:
        if "*" in pattern:
            # Keep the pattern for now (will expand against actual files later)
            expanded.append(pattern)
        else:
            expanded.append(pattern)

    return list(set(expanded))


def _infer_tested_files(test_file_path: str) -> list[str]:
    """Infer which source file a test file tests."""
    tested = []

    # Pattern 1: test_exit_code.py → exit_code.py
    if test_file_path.endswith("_test.py"):
        source_file = test_file_path.replace("_test.py", ".py")
        tested.append(source_file)

    # Pattern 2: tests/test_module.py → module.py
    elif "/test_" in test_file_path:
        source_file = test_file_path.replace("/test_", "/").replace("/tests/", "/")
        tested.append(source_file)

    return tested


def _infer_config_affects(config_path: str, content: str) -> list[str]:
    """Infer which files a config file affects."""
    affects = []

    # pyproject.toml affects all Python files
    if "pyproject.toml" in config_path:
        affects.append("*.py")

    # .yaml workflow affects CI steps
    elif config_path.endswith((".yaml", ".yml")):
        affects.append("*")  # Affects all files in CI context

    return affects


def _build_dependency_clusters(
    nodes: dict[str, dict], edges: list[dict]
) -> list[list[str]]:
    """
    Build dependency clusters - groups of files that should be analyzed together.

    Strategy:
    1. Start with each file as a cluster
    2. Merge clusters connected by dependencies
    3. Return final clusters
    """
    # Build adjacency list
    graph = {}
    for node in nodes:
        graph[node] = set()

    for edge in edges:
        from_file = edge["from"]
        to_file = edge["to"]

        # Only cluster if to_file actually exists in nodes
        if to_file in nodes:
            graph[from_file].add(to_file)
            graph[to_file].add(from_file)  # Bidirectional

    # Find connected components using DFS
    visited = set()
    clusters = []

    def dfs(node: str, cluster: list[str]):
        visited.add(node)
        cluster.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, cluster)

    for node in nodes:
        if node not in visited:
            cluster = []
            dfs(node, cluster)
            if cluster:
                clusters.append(cluster)

    return clusters


def expand_glob_patterns(pattern: str, all_files: list[str]) -> list[str]:
    """Expand glob pattern against list of actual files."""
    import fnmatch

    matched = []

    for file_path in all_files:
        if fnmatch.fnmatch(file_path, pattern):
            matched.append(file_path)

    return matched


def get_dependency_cluster_for_file(
    file_path: str, dependency_graph: dict[str, Any]
) -> list[str]:
    """Get the dependency cluster containing this file."""
    for cluster in dependency_graph.get("clusters", []):
        if file_path in cluster:
            return cluster
    return [file_path]  # Singleton cluster


def explain_dependencies(cluster: list[str], dependency_graph: dict[str, Any]) -> str:
    """
    Generate human-readable explanation of dependencies in a cluster.

    Returns string like:
    "exit_code_test.py READS ref-exit-codes/*.rst files"
    "config.toml CONFIGURES all Python files"
    """
    edges = dependency_graph.get("edges", [])
    dependency_graph.get("nodes", {})

    explanations = []

    for file in cluster:
        # Find outgoing edges from this file
        file_edges = [e for e in edges if e["from"] == file and e["to"] in cluster]

        if file_edges:
            for edge in file_edges:
                explanations.append(
                    f"{edge['from']} {edge['type'].upper()} {edge['to']}"
                )

    return "\n".join(explanations) if explanations else "No dependencies within cluster"


if __name__ == "__main__":
    # Test with example
    test_diff = {
        "files": [
            {
                "path": "exit_code_test.py",
                "changes": [{"after": 'Path("ref-exit-codes/").glob("*.rst")'}],
            },
            {"path": "ref-exit-codes/000.rst", "changes": [{"after": "##### heading"}]},
        ]
    }

    graph = build_dependency_graph(test_diff)

    print("Nodes:", graph["nodes"].keys())
    print("Edges:", graph["edges"])
    print("Clusters:", graph["clusters"])
    print("\nExplanations:")
    for cluster in graph["clusters"]:
        print(f"Cluster: {cluster}")
        print(explain_dependencies(cluster, graph))
