# Memory Format Specification

## Overview

The pipeline generates 3 memory files with distinct purposes:

```
data/trs/
├── failure_memory.json    (L1 - File-level problems)
├── repo_memory.json       (L2 - Repair sequences)  
└── cross_memory.json      (L3 - Universal patterns)
```

---

## L1: failure_memory.json (File-Level Problems)

**Purpose**: Detailed file-level problems from actual diff changes

**Format**: Array of file problems

```json
[
  {
    "repo": "flower",
    "workflow": ".github/workflows/framework.yml",
    "validation_cmd": "python -m mypy py",
    "failure_type": "type checking",
    "issue_type": "mypy_numpy_type_annotation_error",
    "file": "framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py",
    "problem": "Validation failed. Root cause: The code imported DTypeLike...",
    "fixes": "Two changes: (1) Line 21: 'from numpy.typing import DTypeLike, NDArray' → 'from numpy.typing import NDArray'...",
    "why_fix": "This fix resolves the validation failure by...",
    "dependent_files": [
      {
        "file": "framework/pyproject.toml",
        "change": "Changed from 'plugins = [\"numpy.typing.mypy_plugin\"]' to ''",
        "link_to": "Both files needed for same validation. Config change enables code fix."
      }
    ]
  }
]
```

**Key Points**:
- ✅ One entry per file
- ✅ Only actual files from diff (NO speculation)
- ✅ Clear dependent files with specific changes
- ✅ No underscores in failure_type

---

## L2: repo_memory.json (Repair Sequences)

**Purpose**: Ordered repair steps with clear file patterns

**Format**: Array of repair sequences

```json
[
  {
    "repo": "flower",
    "workflow": ".github/workflows/framework.yml",
    "problems": [
      {
        "problem_id": 1,
        "verification_cmd": "python -m mypy py",
        "failure_type": "type checking",
        "problem": "mypy type annotation failure...",
        "root_cause": "DTypeLike is private numpy type...",
        "pattern_detected": null,
        "files": [
          "framework/py/flwr/common/secure_aggregation/*.py (1 files)",
          "framework/pyproject.toml"
        ],
        "file_count": 2,
        "actual_files": [
          "framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py",
          "framework/pyproject.toml"
        ],
        "fix_strategy": "Approach: direct_fix | What: Replace private types | How: ... | Why: ... | Time: 15min"
      },
      {
        "problem_id": 2,
        "verification_cmd": "python -m mdformat --check",
        "failure_type": "formatting",
        "problem": "RST header formatting issues...",
        "pattern_detected": {
          "type": "bulk_formatting",
          "rule": "RST header formatting",
          "scope": "75 files"
        },
        "files": [
          "framework/docs/source/contributor-*.rst (6 files)",
          "framework/docs/source/docker/*.rst (2 files)",
          "framework/docs/source/*.rst (67 files)"
        ],
        "file_count": 75,
        "actual_files": [
          "framework/docs/source/contributor-how-to-release-flower.rst",
          "framework/docs/source/contributor-how-to-set-up-a-virtual-env.rst",
          ...
        ],
        "fix_strategy": "Approach: bulk_formatting_fix | What: Fix RST headers | Time: 30min"
      }
    ]
  }
]
```

**Key Points**:
- ✅ Clear file patterns: `"contributor-*.rst (6 files)"`
- ✅ Explicit `file_count` field
- ✅ `actual_files` list (max 50)
- ✅ NO speculation - only files from diff
- ✅ Ordered by repair sequence

---

## L3: cross_memory.json (Universal Patterns)

**Purpose**: Universal problem patterns with reusable fixes

**Format**: Array of distinct problem patterns

```json
[
  {
    "pattern_id": "numpy_private_type_annotation",
    "failure_type": "type checking",
    "verification_cmd": "python -m mypy py",
    "failure_pattern": "Private numpy type annotations fail without plugin",
    "problem": "mypy fails on private types like DTypeLike. Common root cause: Plugin removed from config but code still uses private types.",
    "universal_fix": {
      "approach": "Replace private types with public equivalents",
      "steps": [
        "1. Identify private type imports (from numpy.typing import DTypeLike)",
        "2. Find public equivalent (np.dtype[Any] or type[Any])",
        "3. Update type annotations in code",
        "4. Remove/update plugin config in pyproject.toml"
      ],
      "applies_to": ["numpy private types", "pandas private types", "typing module internals"]
    },
    "examples": [
      {
        "file": "ndarrays_arithmetic.py",
        "before": "dtype: DTypeLike = np.int64",
        "after": "dtype: np.dtype[Any] = np.int64"
      }
    ],
    "dependent_problems": [
      {
        "pattern_id": "pyproject_mypy_plugin_config",
        "relationship": "requires_config_change",
        "rationale": "Code change requires matching pyproject.toml plugin removal"
      }
    ]
  },
  {
    "pattern_id": "rst_header_formatting",
    "failure_type": "formatting",
    "verification_cmd": "python -m mdformat --check",
    "failure_pattern": "RST headers with mismatched underlines",
    "problem": "mdformat fails on RST files with improper section header formatting. Common root cause: Title underline length doesn't match title.",
    "universal_fix": {
      "approach": "Standardize RST header formatting",
      "steps": [
        "1. Add blank line before section title",
        "2. Ensure title format is correct",
        "3. Match underline length to title length",
        "4. Use consistent underline characters"
      ],
      "applies_to": [".rst documentation", "sphinx docs"]
    },
    "examples": [
      {
        "file": "contributor-how-to-release-flower.rst",
        "before": "Release Flower\\n##############",
        "after": "\\n Release Flower\\n################"
      }
    ],
    "dependent_problems": []
  }
]
```

**Key Points**:
- ✅ Each independent problem = separate entry
- ✅ Only link if actual dependency exists
- ✅ mypy ≠ mdformat (separate entries)
- ✅ Universal fixes applicable to similar future problems
- ✅ NO occurrences/issues fields

---

## Pipeline Command

```bash
# Full pipeline: Load → Decompose → Build Memory
python scripts/decompose_ci_failure.py --issue-id 121 --output-dir data/trs && python scripts/build_memory.py
```

**Output**:
- `decomposed_issues.json` - Intermediate decomposition data
- `failure_memory.json` - L1 (128 file problems)
- `repo_memory.json` - L2 (5 repair sequences)
- `cross_memory.json` - L3 (5 universal patterns)

---

## Key Principles

1. **No Speculation**: Only files from actual diff
2. **Clear Patterns**: `contributor-*.rst (6 files)` not vague paths
3. **Independent Problems**: Separate entries unless dependent
4. **Explicit Counts**: Always include `file_count`
5. **Universal Fixes**: Reusable solutions for similar problems
