#!/usr/bin/env python3
"""
Dependency Detector - Build file dependency graph from diff.

This module extracts dependencies between changed files to enable
dependency-aware chunking and analysis.
"""

from __future__ import annotations

import re
import fnmatch
from pathlib import PurePosixPath
from pathlib import Path
from typing import Any


def build_dependency_graph(
    structured_diff: dict[str, Any], repo_path: str | None = None
) -> dict[str, Any]:
    """
    Build dependency graph from structured diff.

    Args:
        structured_diff: Output from parse_diff_to_structured()
        repo_path: Optional failed-checkout path. When present, full changed
            file contents are inspected so dependencies in unchanged lines can
            still be detected.

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
    raw_edges = []

    # Step 1: Extract dependencies for each file
    for file_info in structured_diff.get("files", []):
        file_path = file_info["path"]
        changes = file_info.get("changes", [])
        file_content = _read_repo_file(repo_path, file_path)

        # Detect file type
        file_type = _classify_file_type(file_path)

        # Extract dependencies
        deps = _extract_dependencies(file_path, changes, file_content=file_content)

        nodes[file_path] = {"type": file_type, **deps}

        # Create raw edges. They are resolved against changed files below, so
        # glob patterns, directory reads, and repo-relative guesses can become
        # concrete file-to-file relationships.
        for dep_type in ["imports", "reads", "tests", "configures"]:
            for dep_file in deps.get(dep_type, []):
                raw_edges.append({"from": file_path, "to": dep_file, "type": dep_type})

    # Step 2: Resolve dependency patterns against changed files
    edges = _resolve_dependency_edges(nodes, raw_edges)
    edges.extend(_infer_changed_file_relationships(nodes))
    edges = _dedupe_edges(edges)

    # Step 3: Build dependency clusters
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


def _read_repo_file(
    repo_path: str | None, file_path: str, max_chars: int = 200_000
) -> str:
    """Read a changed file from the failed checkout when available."""
    if not repo_path:
        return ""

    root = Path(repo_path).resolve()
    path = (root / _normalize_path(file_path)).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return ""

    if not path.exists() or not path.is_file():
        return ""

    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def _extract_dependencies(
    file_path: str, changes: list[dict], file_content: str = ""
) -> dict[str, list[str]]:
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
    changed_code = "\n".join(
        [
            text
            for change in changes
            for text in (change.get("before", ""), change.get("after", ""))
            if text
        ]
    )
    all_code = "\n".join(text for text in [file_content, changed_code] if text)

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

    # Pattern 4: quoted path fragments, including chained pathlib expressions
    # like Path(__file__).parents[4] / "docs/source/ref-exit-codes".
    literal_patterns = re.findall(
        r'["\']([^"\']*(?:/|\*|\.(?:rst|md|py|txt|json|toml|ya?ml))[^"\']*)["\']', code
    )
    reads.extend(literal_patterns)

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

    lowered = content.lower()

    # pyproject.toml affects tool-specific file families depending on the
    # changed keys/packages. Keep this conservative enough to avoid pulling an
    # entire repository into one cluster for every config touch.
    if "pyproject.toml" in config_path:
        if any(tool in lowered for tool in ["mdformat", "docformatter", "sphinx"]):
            affects.extend(["*.md", "*.rst"])
        if any(
            tool in lowered for tool in ["mypy", "ruff", "pytest", "black", "isort"]
        ):
            affects.append("*.py")
        if not affects:
            affects.append("*.py")

    # .yaml workflow affects CI steps
    elif config_path.endswith((".yaml", ".yml")):
        affects.append("*")  # Affects all files in CI context

    return affects


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    """Remove duplicate resolved edges while keeping insertion order."""
    seen = set()
    deduped = []
    for edge in edges:
        key = (
            edge.get("from"),
            edge.get("to"),
            edge.get("type"),
            edge.get("via", ""),
        )
        if key in seen or edge.get("from") == edge.get("to"):
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _normalize_path(path: str) -> str:
    path = str(path or "").replace("\\", "/").strip()
    while path.startswith("./"):
        path = path[2:]
    return path


def _path_variants(pattern: str, from_file: str) -> list[str]:
    """Return repo-relative candidates for a dependency pattern."""
    pattern = _normalize_path(pattern)
    if not pattern:
        return []

    variants = [pattern]
    parent = str(PurePosixPath(from_file).parent)
    if parent and parent != ".":
        variants.append(_normalize_path(str(PurePosixPath(parent) / pattern)))

    if pattern.endswith("/"):
        variants.append(pattern + "*")

    return list(dict.fromkeys(variants))


def _matches_dependency_pattern(
    changed_file: str, pattern: str, from_file: str, dep_type: str
) -> bool:
    changed_file = _normalize_path(changed_file)
    variants = _path_variants(pattern, from_file)

    for variant in variants:
        if dep_type == "imports":
            module_path = variant.replace(".", "/")
            if changed_file.endswith(module_path + ".py") or changed_file.endswith(
                module_path + "/__init__.py"
            ):
                return True

        if "*" in variant and fnmatch.fnmatch(changed_file, variant):
            return True

        if variant.endswith("/") and changed_file.startswith(variant):
            return True

        if changed_file == variant:
            return True

        # Directory reads often appear as Path("ref-exit-codes/") while the
        # actual changed files are nested elsewhere, e.g.
        # framework/docs/source/ref-exit-codes/000.rst.
        stripped = variant.rstrip("/")
        if stripped and f"/{stripped}/" in f"/{changed_file}/":
            return True

    return False


def _resolve_dependency_edges(
    nodes: dict[str, dict], raw_edges: list[dict]
) -> list[dict]:
    """Resolve raw dependency targets to concrete changed files."""
    changed_files = list(nodes)
    resolved = []

    for edge in raw_edges:
        from_file = edge["from"]
        target = edge["to"]
        dep_type = edge["type"]
        for changed_file in changed_files:
            if _matches_dependency_pattern(changed_file, target, from_file, dep_type):
                resolved.append(
                    {
                        "from": from_file,
                        "to": changed_file,
                        "type": dep_type,
                        "via": target,
                    }
                )

    return resolved


def _path_tokens(file_path: str) -> set[str]:
    """Extract meaningful tokens for conservative changed-file relation hints."""
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9]+", file_path.lower()))
    normalized = set(tokens)
    for token in tokens:
        if token.endswith("s") and len(token) > 3:
            normalized.add(token[:-1])
    return {
        token
        for token in normalized
        if len(token) >= 3
        and token
        not in {
            "source",
            "docs",
            "doc",
            "test",
            "tests",
            "framework",
            "common",
            "code",
            "py",
            "rst",
            "md",
        }
    }


def _infer_changed_file_relationships(nodes: dict[str, dict]) -> list[dict]:
    """
    Infer relationships visible from the changed-file set itself.

    This intentionally stays file-family based: tests may depend on docs/data
    files with overlapping domain tokens, and formatter/tool config changes may
    affect matching file types. It avoids broad arbitrary source-to-source
    token clustering.
    """
    edges = []
    files = list(nodes)

    for from_file, from_node in nodes.items():
        from_type = from_node.get("type")
        from_tokens = _path_tokens(from_file)

        for to_file, to_node in nodes.items():
            if from_file == to_file:
                continue

            to_type = to_node.get("type")
            to_tokens = _path_tokens(to_file)

            if from_type == "test" and to_type in {"docs", "other"}:
                overlap = from_tokens & to_tokens
                if overlap:
                    edges.append(
                        {
                            "from": from_file,
                            "to": to_file,
                            "type": "related_changes",
                            "via": "shared changed-file tokens: "
                            + ", ".join(sorted(overlap)[:5]),
                        }
                    )

    # Config wildcard edges are resolved in _resolve_dependency_edges, but this
    # extra pass lets pyproject formatter changes attach to docs even when the
    # changed line only names a plugin package.
    for from_file, from_node in nodes.items():
        if from_node.get("type") != "config":
            continue
        for pattern in from_node.get("configures", []):
            for to_file in files:
                if _matches_dependency_pattern(
                    to_file, pattern, from_file, "configures"
                ):
                    edges.append(
                        {
                            "from": from_file,
                            "to": to_file,
                            "type": "configures",
                            "via": pattern,
                        }
                    )

    return edges


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
                via = f" via {edge['via']}" if edge.get("via") else ""
                explanations.append(
                    f"{edge['from']} {edge['type'].upper()} {edge['to']}{via}"
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
