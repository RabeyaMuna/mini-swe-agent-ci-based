#!/usr/bin/env python3
"""
validation_matcher.py - Match files to CI validation commands
"""

import fnmatch
from pathlib import Path
from typing import List, Dict, Set


# Files/directories to IGNORE (not CI-validated)
IGNORED_PATTERNS = [
    ".github/**",
    "docs/**",
    "*.md",
    "LICENSE*",
    ".gitignore",
    ".editorconfig",
    ".vscode/**",
    ".idea/**",
    "README*",
    "CONTRIBUTING*",
    "CHANGELOG*",
    ".pre-commit-config.yaml",
    ".readthedocs.yml",
    "mkdocs.yml",
]


def should_ignore_file(filepath: str) -> bool:
    """Check if file should be ignored (not CI-validated)"""
    for pattern in IGNORED_PATTERNS:
        if fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(
            f"**/{filepath}", pattern
        ):
            return True
    return False


def extract_validated_patterns(validation_sequence: List[Dict]) -> Set[str]:
    """
    Extract file patterns that are actually validated by CI

    Returns set of patterns like: {'*.py', '*.toml', 'tests/**'}
    """
    patterns = set()

    for validation in validation_sequence:
        validates = validation.get("validates", "").lower()

        # Python validations
        if any(
            x in validates
            for x in ["mypy", "isort", "black", "ruff", "pylint", "flake8", "python"]
        ):
            patterns.add("*.py")

        # Python config
        if "poetry" in validates or "pyproject" in validates or "setup.py" in validates:
            patterns.add("pyproject.toml")
            patterns.add("setup.py")
            patterns.add("setup.cfg")

        # Tests
        if "pytest" in validates or "unittest" in validates or "test" in validates:
            patterns.add("test_*.py")
            patterns.add("*_test.py")
            patterns.add("tests/**/*.py")

        # JavaScript/TypeScript
        if any(
            x in validates for x in ["eslint", "prettier", "javascript", "typescript"]
        ):
            patterns.add("*.js")
            patterns.add("*.ts")
            patterns.add("*.jsx")
            patterns.add("*.tsx")

        # JSON/YAML configs
        if "json" in validates or "package.json" in validates:
            patterns.add("*.json")
            patterns.add("package.json")

        if "yaml" in validates or "yml" in validates:
            patterns.add("*.yml")
            patterns.add("*.yaml")

        # Java
        if "java" in validates or "maven" in validates or "gradle" in validates:
            patterns.add("*.java")
            patterns.add("pom.xml")
            patterns.add("build.gradle")

        # Go
        if "go" in validates or "golang" in validates:
            patterns.add("*.go")
            patterns.add("go.mod")
            patterns.add("go.sum")

        # Rust
        if "rust" in validates or "cargo" in validates:
            patterns.add("*.rs")
            patterns.add("Cargo.toml")

    return patterns


def matches_pattern(filepath: str, patterns: Set[str]) -> bool:
    """Check if file matches any of the patterns"""
    for pattern in patterns:
        if fnmatch.fnmatch(filepath, pattern):
            return True
        if fnmatch.fnmatch(Path(filepath).name, pattern):
            return True
        if "**" in pattern:
            # Handle glob patterns like tests/**/*.py
            parts = pattern.split("**")
            if len(parts) == 2:
                if parts[0] in filepath and filepath.endswith(parts[1].lstrip("/")):
                    return True
    return False


def filter_to_validated_files(
    all_files: List[str], validation_sequence: List[Dict]
) -> List[str]:
    """
    Filter file list to only those validated by CI

    Args:
        all_files: List of file paths
        validation_sequence: CI validation sequence

    Returns:
        List of validated files only
    """
    validated_patterns = extract_validated_patterns(validation_sequence)

    validated_files = []
    for filepath in all_files:
        # Skip ignored files
        if should_ignore_file(filepath):
            continue

        # Check if matches validated patterns
        if matches_pattern(filepath, validated_patterns):
            validated_files.append(filepath)

    return validated_files


def filter_validations_for_files(
    validation_sequence: List[Dict], files: List[str]
) -> List[Dict]:
    """
    Get only the validation commands relevant to these files

    Args:
        validation_sequence: Full CI validation sequence
        files: List of file paths

    Returns:
        List of relevant validation commands
    """
    relevant = []

    for validation in validation_sequence:
        validates = validation.get("validates", "").lower()

        # Check if any file matches this validation type
        for filepath in files:
            ext = Path(filepath).suffix
            name = Path(filepath).name

            # Python files
            if ext == ".py":
                if any(
                    x in validates
                    for x in ["mypy", "isort", "black", "ruff", "pylint", "flake8"]
                ):
                    if validation not in relevant:
                        relevant.append(validation)
                if "test" in name and "pytest" in validates:
                    if validation not in relevant:
                        relevant.append(validation)

            # Config files
            elif filepath.endswith("pyproject.toml") or filepath.endswith("setup.py"):
                if "poetry" in validates or "pyproject" in validates:
                    if validation not in relevant:
                        relevant.append(validation)

            # JS/TS files
            elif ext in [".js", ".ts", ".jsx", ".tsx"]:
                if "eslint" in validates or "prettier" in validates:
                    if validation not in relevant:
                        relevant.append(validation)

            # YAML files
            elif ext in [".yml", ".yaml"]:
                if "yaml" in validates or "yml" in validates:
                    if validation not in relevant:
                        relevant.append(validation)

    return relevant
