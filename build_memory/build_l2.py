"""
L2 repair strategies builder.

Generates repair strategies from L1 problems using LLM.
"""

from typing import Any, Dict
import json

# Import L2 prompt from centralized location
from prompt_template.memory_build import build_l2_prompt
from utilities.llm_invoker import invoke_llm_with_retry


# ========================================
# AUTOMATION TOOLS REGISTRY
# ========================================
AUTOMATED_TOOLS = [
    {
        "tool": "ruff",
        "purpose": "Unified Python linter and formatter (combines flake8, isort, black)",
        "install_command": "pip install ruff",
        "fix_command": "ruff check --fix {{file_or_dir}}",
        "format_command": "ruff format {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["linting errors", "import sorting", "code formatting"],
    },
    {
        "tool": "black",
        "purpose": "Python code formatter (opinionated, PEP 8 compliant)",
        "install_command": "pip install black",
        "fix_command": "black {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["code formatting", "line length", "indentation"],
    },
    {
        "tool": "isort",
        "purpose": "Python import statement sorter",
        "install_command": "pip install isort",
        "fix_command": "isort {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["import order", "import grouping"],
    },
    {
        "tool": "docstrfmt",
        "purpose": "RST file formatter (mainly for .rst documentation files, also formats RST in docstrings)",
        "install_command": "pip install docstrfmt",
        "fix_command": "docstrfmt {{file_or_dir}}",
        "file_pattern": ["*.rst", "*.py"],
        "fixes": ["RST heading style", "RST formatting", "RST syntax"],
        "note": "Primarily for *.rst files. For Python docstrings use docformatter instead.",
    },
    {
        "tool": "docformatter",
        "purpose": "Python docstring formatter (PEP 257 compliant, for docstrings inside .py files)",
        "install_command": "pip install docformatter",
        "fix_command": "docformatter --in-place --recursive {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["docstring formatting", "PEP 257 compliance", "docstring wrapping"],
        "note": "For Python docstrings. For RST files use docstrfmt instead.",
    },
    {
        "tool": "mdformat",
        "purpose": "Markdown file formatter",
        "install_command": "pip install mdformat",
        "fix_command": "python -m mdformat {{file_or_dir}}",
        "file_pattern": "*.md",
        "fixes": ["markdown formatting", "heading style", "list formatting"],
    },
    {
        "tool": "codespell",
        "purpose": "Spell checker for code and documentation",
        "install_command": "pip install codespell",
        "fix_command": "codespell -w {{file_or_dir}}",
        "file_pattern": ["*.py", "*.md", "*.rst", "*.txt"],
        "fixes": ["spelling errors", "typos"],
    },
    {
        "tool": "taplo",
        "purpose": "TOML file formatter",
        "install_command": "cargo install taplo-cli",
        "fix_command": "taplo fmt {{file_or_dir}}",
        "file_pattern": "*.toml",
        "fixes": ["TOML formatting", "pyproject.toml formatting"],
        "note": "Requires Rust/Cargo to install",
    },
    {
        "tool": "autopep8",
        "purpose": "Python code formatter (less opinionated than black)",
        "install_command": "pip install autopep8",
        "fix_command": "autopep8 --in-place --recursive {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["PEP 8 violations", "code formatting"],
    },
    {
        "tool": "autoflake",
        "purpose": "Removes unused imports and variables",
        "install_command": "pip install autoflake",
        "fix_command": "autoflake --in-place --remove-all-unused-imports --recursive {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["unused imports", "unused variables"],
    },
    {
    "tool": "yamllint",
    "purpose": "YAML linter for syntax, indentation, whitespace, and style violations",
    "install_command": "pip install yamllint",
    "fix_command": None,
    "file_pattern": ["*.yml", "*.yaml"],
    "fixes": [],
    "note": "Detects YAML violations but does not automatically modify files.",
},
{
    "tool": "yamlfmt",
    "purpose": "YAML formatter",
    "install_command": "go install github.com/google/yamlfmt/cmd/yamlfmt@latest",
    "fix_command": "yamlfmt {{file_or_dir}}",
    "file_pattern": ["*.yml", "*.yaml"],
    "fixes": [
        "YAML formatting",
        "indentation",
        "trailing whitespace",
    ],
},
{
    "tool": "prettier",
    "purpose": "YAML, JSON, Markdown, and other file formatter",
    "install_command": "npm install -g prettier",
    "fix_command": "prettier --write {{file_or_dir}}",
    "file_pattern": ["*.yml", "*.yaml", "*.json", "*.md"],
    "fixes": [
        "YAML indentation",
        "trailing whitespace",
        "YAML formatting",
        "YAML style"
    ],
    "note": "Can automatically format GitHub Actions workflow YAML files.",
},
{
    "tool": "yapf",
    "purpose": "Python code formatter",
    "install_command": "pip install yapf",
    "fix_command": "yapf -i {{file_or_dir}}",
    "file_pattern": "*.py",
    "fixes": ["code formatting", "style violations"],
},
]


# ========================================
# L1 ANALYSIS & SAMPLING
# ========================================


def analyze_l1_size(l1_memory: Dict) -> Dict[str, Any]:
    """
    Analyze L1 size and determine if sampling is needed.

    Returns:
        {
            "total_size": <estimated tokens>,
            "needs_sampling": <bool>,
            "large_problems": [...]
        }
    """
    # Estimate size (rough: 1 char ≈ 0.25 tokens)
    l1_json = json.dumps(l1_memory)
    estimated_chars = len(l1_json)
    estimated_tokens = estimated_chars // 4

    # Check for problems with many files
    large_problems = []
    for problem in l1_memory.get("problems", []):
        files = problem.get("files", [])
        if len(files) > 10:
            large_problems.append(
                {
                    "problem_id": problem.get("problem_id"),
                    "file_count": len(files),
                }
            )

    return {
        "total_size": estimated_tokens,
        "needs_sampling": estimated_tokens > 12000 or len(large_problems) > 0,
        "large_problems": large_problems,
    }


def sample_l1_for_prompt(l1_memory: Dict, analysis: Dict) -> Dict:
    """
    Create sampled version of L1 for prompt if needed.

    - Keep first 5 files per problem if >10 files
    - Add metadata showing total count and pattern
    """
    if not analysis["needs_sampling"]:
        return l1_memory

    sampled = dict(l1_memory)
    sampled_problems = []

    for problem in l1_memory.get("problems", []):
        sampled_problem = dict(problem)
        files = problem.get("files", [])

        if len(files) > 10:
            # Sample: first 5 files + metadata
            sampled_files = files[:5]
            remaining = len(files) - 5

            # Extract directory pattern
            if files:
                first_file = files[0]
                if "/" in first_file:
                    parts = first_file.split("/")
                    common_dir = "/".join(parts[:-1]) if len(parts) > 1 else ""
                    pattern = f"{common_dir}/**/*" if common_dir else "*"
                else:
                    pattern = "*"
            else:
                pattern = "*"

            sampled_problem["files"] = sampled_files
            sampled_problem["files_metadata"] = {
                "sampled": True,
                "total_count": len(files),
                "showing": len(sampled_files),
                "remaining": remaining,
                "pattern": pattern,
                "note": f"Showing {len(sampled_files)} of {len(files)} files. Pattern: {pattern}",
            }

        sampled_problems.append(sampled_problem)

    sampled["problems"] = sampled_problems
    return sampled


def generate_l2_with_llm(l1_memory: Dict, llm: Any) -> Dict[str, Any]:
    """
    Use LLM to analyze L1 and generate complete L2.

    Returns:
        {
            "failure_identify": [...],
            "repair_strategies": [...]
        }
    """
    if llm is None:
        raise ValueError("LLM is required for L2 generation")

    # Analyze L1 size and sample if needed
    analysis = analyze_l1_size(l1_memory)
    l1_for_prompt = sample_l1_for_prompt(l1_memory, analysis)

    # Build sampling info for prompt
    sampling_info = (
        {
            "was_sampled": analysis["needs_sampling"],
            "total_size": analysis["total_size"],
        }
        if analysis["needs_sampling"]
        else None
    )

    # Build prompt with sampled data
    prompt = build_l2_prompt(l1_for_prompt, AUTOMATED_TOOLS, sampling_info)

    # Call LLM with retry, lenient parsing, and repair-prompt fallback
    l2_data = invoke_llm_with_retry(llm=llm, prompt=prompt, parse_json=True)

    if not isinstance(l2_data, dict):
        raise ValueError("LLM failed to produce valid L2 JSON after retries")

    return l2_data


def build_l2_memory(
    l1_memory: Dict[str, Any],
    llm: Any = None,
) -> Dict[str, Any]:
    """
    Build L2 repair strategies from L1 problems using LLM.

    The LLM analyzes L1 and generates:
    1. failure_identify: Summary of failure types
    2. repair_strategies: Reusable patterns

    Args:
        l1_memory: L1 memory with problems (from build_l1.py)
        llm: LLM instance for analysis and generation

    Returns:
        L2 memory with repair strategies:
        {
          "issue_id": "102",
          "repo": "flower-ai/flower",
          "workflow": ".github/workflows/framework.yml",
          "total_problems": 7,
          "failure_identify": ["type_checking (mypy) - 3 problems", ...],
          "repair_strategies": [
            {
              "step": 1,
              "applies_to_failures": [...],
              "summary": "...",
              "intent": "...",
              "reasoning": "...",
              "rationale": "...",
              "when_to_apply": "...",
              "signals": [...],
              "key_actions": [...],
              "pitfalls": [...],
              "example_phrasing": "..."
            }
          ]
        }
    """

    if llm is None:
        raise ValueError("LLM is required for L2 generation")

    # LLM analyzes L1 and generates complete L2 data.
    # Falls back to empty strategies on failure (same pattern as L1/L3) so a
    # bad LLM response degrades gracefully instead of crashing the pipeline.
    try:
        l2_data = generate_l2_with_llm(l1_memory, llm)
    except Exception as e:
        print(f"  WARNING: L2 generation failed: {e}")
        print("  Falling back to empty L2 structure")
        l2_data = {}

    # Build final L2 structure
    l2_memory = {
        "issue_id": l1_memory.get("issue_id"),
        "repo": l1_memory.get("repo"),
        "workflow": l1_memory.get("workflow"),
        "total_problems": len(l1_memory.get("problems", [])),
        "failure_identify": l2_data.get("failure_identify", []),
        "repair_strategies": l2_data.get("repair_strategies", []),
    }

    return l2_memory


# Public API
__all__ = [
    "build_l2_memory",
    "AUTOMATED_TOOLS",
]
