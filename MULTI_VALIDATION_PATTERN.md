# Handling 110 Files with 21 Different Failed Validations

## Scenario

- **110 files changed**
- **21 different CI validation failures**
- Need concrete, actionable memory without overwhelming detail

## Strategy: Group by Validation Step + Pattern

### Structure Overview

```
L2 (Issue Level):
├── atomic_problem_1 (validation step 1: dependency install)
│   ├── pattern_group_1: pyproject.toml updates (3 files)
│   └── pattern_group_2: requirements.txt updates (2 files)
├── atomic_problem_2 (validation step 2: ruff check)
│   ├── pattern_group_3: unused import removal (15 files)
│   └── pattern_group_4: line length fixes (8 files)
├── atomic_problem_3 (validation step 3: mypy check)
│   ├── pattern_group_5: type hint additions (22 files)
│   └── pattern_group_6: Optional type fixes (12 files)
...
└── atomic_problem_21 (validation step 21: pytest)
    └── pattern_group_N: test fixture updates (10 files)

Total: 21 atomic problems, N pattern groups, 110 files
```

---

## L2 Structure for Multi-Validation Issues

```json
{
  "repo": "flower",
  "issue_id": "125",
  "total_changed_files": 110,
  "total_validation_failures": 21,
  
  "atomic_problems": [
    {
      "problem_id": 1,
      "visibility": "visible_in_log",
      "issue_type": "dependency_error",
      "failed_cmd": "cd framework && poetry install",
      "ci_stage": "install",
      "validation_step": 1,
      
      "problem": {
        "symptoms": [
          "Package mdformat-beautysh==0.1.1 not found in PyPI",
          "Poetry lock file incompatible with new dependencies"
        ],
        "root_causes": [
          "mdformat-beautysh 0.1.1 was deprecated and removed from PyPI",
          "Need to upgrade to 1.0.0"
        ],
        "why_ci_stopped_here": "Installation must succeed before any code validation can run"
      },
      
      "file_patterns": [
        {
          "pattern_id": "dep_version_bump",
          "pattern": "Update mdformat-beautysh version in pyproject.toml files",
          "files_count": 3,
          "example_file": "dev/pyproject.toml",
          
          "concrete_example": {
            "file": "dev/pyproject.toml",
            "line_location": "line 45",
            "before": "mdformat-beautysh = \"0.1.1\"",
            "after": "mdformat-beautysh = \"1.0.0\"",
            "change_type": "version_update"
          },
          
          "all_files": [
            {
              "file": "dev/pyproject.toml",
              "line": 45,
              "same_change": "0.1.1 → 1.0.0"
            },
            {
              "file": "framework/pyproject.toml",
              "line": 67,
              "same_change": "0.1.1 → 1.0.0"
            },
            {
              "file": "baselines/pyproject.toml",
              "line": 34,
              "same_change": "0.1.1 → 1.0.0"
            }
          ],
          
          "fix_instructions": "Update mdformat-beautysh dependency from 0.1.1 to 1.0.0 in all pyproject.toml files. Run 'poetry lock --no-update' to regenerate lock file."
        }
      ],
      
      "verification": {
        "command": "cd framework && poetry install",
        "expected_result": "Dependencies installed successfully",
        "next_problem_exposed": 2
      }
    },
    
    {
      "problem_id": 2,
      "visibility": "hidden",
      "issue_type": "linting_error",
      "failed_cmd": "ruff check .",
      "ci_stage": "lint",
      "validation_step": 4,
      
      "problem": {
        "symptoms": [
          "F401: 'module.Foo' imported but unused (15 occurrences)",
          "E501: line too long (>120 characters) (8 occurrences)"
        ],
        "root_causes": [
          "Unused imports left from refactoring",
          "Some docstring lines exceed new line length limit"
        ],
        "why_hidden": "Problem 1 (install) blocks this validation from running"
      },
      
      "file_patterns": [
        {
          "pattern_id": "remove_unused_imports",
          "pattern": "Remove unused import statements flagged by F401",
          "files_count": 15,
          "example_file": "framework/py/flwr/common/exit/exit_code.py",
          
          "concrete_example": {
            "file": "framework/py/flwr/common/exit/exit_code.py",
            "line_location": "line 12",
            "before": "from foo import Bar, Baz, Qux",
            "after": "from foo import Bar, Baz",
            "change_type": "remove_unused_import",
            "explanation": "Qux was imported but never used in this file"
          },
          
          "all_files": [
            "framework/py/flwr/common/exit/exit_code.py (line 12: removed Qux)",
            "framework/py/flwr/common/secure_aggregation/ndarrays_arithmetic.py (line 8: removed List)",
            "... (13 more files, each removing 1-2 unused imports)"
          ],
          
          "fix_instructions": "For each F401 error, remove the unused import. Use 'ruff check --select F401 .' to list all, then manually remove or use 'ruff --fix'."
        },
        
        {
          "pattern_id": "line_length_fixes",
          "pattern": "Break long lines (>120 chars) in docstrings and comments",
          "files_count": 8,
          "example_file": "framework/py/flwr/server/strategy/fedavg.py",
          
          "concrete_example": {
            "file": "framework/py/flwr/server/strategy/fedavg.py",
            "line_location": "lines 156-157",
            "before": "        # This is a very long comment that exceeds the maximum line length and should be split across multiple lines for better readability (125 chars)",
            "after": "        # This is a very long comment that exceeds the maximum line length\n        # and should be split across multiple lines for better readability",
            "change_type": "line_break_insertion"
          },
          
          "all_files": [
            "framework/py/flwr/server/strategy/fedavg.py (line 156-157)",
            "framework/py/flwr/client/client.py (line 89-90)",
            "... (6 more files)"
          ],
          
          "fix_instructions": "Break lines over 120 characters. For docstrings: break at natural phrase boundaries. For code: use implicit line continuation inside parentheses."
        }
      ],
      
      "verification": {
        "command": "ruff check .",
        "expected_result": "No linting errors",
        "next_problem_exposed": 3
      }
    },
    
    {
      "problem_id": 3,
      "visibility": "hidden",
      "issue_type": "type_error",
      "failed_cmd": "mypy framework/",
      "ci_stage": "type_check",
      "validation_step": 8,
      
      "problem": {
        "symptoms": [
          "error: Argument 1 has incompatible type (22 occurrences across multiple files)",
          "error: Missing return type annotation (12 occurrences)"
        ],
        "root_causes": [
          "Dependency upgrade changed API signatures",
          "Stricter type checking after mypy config update"
        ],
        "why_hidden": "Problems 1 (install) and 2 (lint) must be fixed first"
      },
      
      "file_patterns": [
        {
          "pattern_id": "update_type_signatures",
          "pattern": "Update function signatures to match new API",
          "files_count": 22,
          "example_file": "framework/py/flwr/common/typing.py",
          
          "concrete_example": {
            "file": "framework/py/flwr/common/typing.py",
            "line_location": "line 45",
            "before": "def process(data: List[int]) -> None:",
            "after": "def process(data: Sequence[int]) -> None:",
            "change_type": "type_annotation_update",
            "explanation": "New version expects Sequence (more general) instead of List (specific)"
          },
          
          "all_files": [
            "framework/py/flwr/common/typing.py (List → Sequence)",
            "framework/py/flwr/server/client_manager.py (Dict → Mapping)",
            "... (20 more files with similar covariance/contravariance fixes)"
          ],
          
          "fix_instructions": "Update type annotations to use more general types: List → Sequence, Dict → Mapping, Set → AbstractSet. Use mypy error messages to identify specific locations."
        },
        
        {
          "pattern_id": "add_return_type_annotations",
          "pattern": "Add explicit return type annotations to functions",
          "files_count": 12,
          "example_file": "framework/py/flwr/client/app.py",
          
          "concrete_example": {
            "file": "framework/py/flwr/client/app.py",
            "line_location": "line 78",
            "before": "def get_client():",
            "after": "def get_client() -> Client:",
            "change_type": "add_return_type"
          },
          
          "all_files": [
            "framework/py/flwr/client/app.py (line 78: → Client)",
            "framework/py/flwr/server/server.py (line 134: → Server)",
            "... (10 more files)"
          ],
          
          "fix_instructions": "Add return type annotations using '-> ReturnType:' syntax. If function returns None, use '-> None:'. For complex returns, import types from typing module."
        }
      ],
      
      "verification": {
        "command": "mypy framework/",
        "expected_result": "Success: no type errors",
        "next_problem_exposed": 4
      }
    },
    
    {
      "problem_id": 4,
      "visibility": "hidden",
      "issue_type": "documentation_formatting",
      "failed_cmd": "./dev/test.sh",
      "ci_stage": "lint",
      "validation_step": 6,
      
      "problem": {
        "symptoms": [
          "Title underline too short (47 occurrences in RST files)",
          "Missing overline for document titles"
        ],
        "root_causes": [
          "mdformat-beautysh 1.0.0 enforces strict RST title formatting",
          "Previous version (0.1.1) was lenient about underline length"
        ],
        "why_hidden": "Problem 1 (install mdformat-beautysh 1.0.0) must succeed first"
      },
      
      "file_patterns": [
        {
          "pattern_id": "rst_title_symmetric_formatting",
          "pattern": "Add overline and fix underline length for RST document titles",
          "files_count": 47,
          "example_file": "framework/docs/source/docker/set-environment-variables.rst",
          
          "concrete_example": {
            "file": "framework/docs/source/docker/set-environment-variables.rst",
            "line_location": "lines 4-6",
            "before": "Set Environment Variables\n=========================",
            "after": "=========================\nSet Environment Variables\n=========================",
            "change_type": "rst_title_formatting",
            "explanation": "Document title requires symmetric overline+underline using '=' character, each exactly 25 chars to match title length"
          },
          
          "all_files": [
            "framework/docs/source/docker/set-environment-variables.rst (title: 25 chars)",
            "framework/docs/source/docker/enable-tls.rst (title: 18 chars)",
            "... (45 more .rst files in docs/source/ directory)"
          ],
          
          "fix_instructions": "For each RST file: (1) Count title character length, (2) Add overline with N '=' chars above title, (3) Ensure underline also has N '=' chars. Apply to all document-level titles (top of file or after only directive comments)."
        }
      ],
      
      "verification": {
        "command": "./dev/test.sh",
        "expected_result": "All RST files pass mdformat validation",
        "next_problem_exposed": 5
      }
    }
    
    // ... problems 5-21 follow same pattern ...
  ],
  
  "repair_trajectory": {
    "sequence": [
      {
        "step": 1,
        "problem_ids": [1],
        "validation_step": 1,
        "what": "Fix dependency installation",
        "files": 3,
        "patterns": 1,
        "why_first": "All subsequent validations require dependencies to be installed"
      },
      {
        "step": 2,
        "problem_ids": [2, 4],
        "validation_step": 4,
        "what": "Fix linting errors (unused imports + RST formatting)",
        "files": 70,
        "patterns": 3,
        "why_next": "Linting must pass before type checking"
      },
      {
        "step": 3,
        "problem_ids": [3],
        "validation_step": 8,
        "what": "Fix type errors",
        "files": 34,
        "patterns": 2,
        "why_next": "Type checking must pass before tests run"
      }
      // ... more steps ...
    ],
    
    "summary": "110 files changed across 21 validation failures. Repair order: (1) install dependencies [3 files], (2) fix linting [70 files in 3 patterns], (3) fix types [34 files in 2 patterns], (4) fix tests [remaining files]. Each step unblocks the next validation stage in CI pipeline."
  },
  
  "file_index": {
    "by_validation": {
      "step_1_install": 3,
      "step_4_lint": 70,
      "step_8_type_check": 34,
      "step_12_test": 3
    },
    "by_pattern": {
      "dep_version_bump": 3,
      "remove_unused_imports": 15,
      "line_length_fixes": 8,
      "rst_title_formatting": 47,
      "update_type_signatures": 22,
      "add_return_type_annotations": 12,
      "... more patterns ...": "..."
    }
  }
}
```

---

## Key Strategies

### 1. **Hierarchical Grouping**
```
Issue (110 files)
  ↓
21 Atomic Problems (by validation step)
  ↓
N Pattern Groups (by fix type)
  ↓
Individual Files (with concrete examples)
```

### 2. **Pattern Recognition**
For each atomic problem:
- **Identify patterns**: "47 RST files all need same fix"
- **Extract one concrete example**: Show full before/after for one file
- **List all files**: Reference others with brief note
- **Provide pattern instructions**: How to apply the pattern

### 3. **Smart Truncation**
```json
{
  "all_files": [
    {
      "file": "file1.py",
      "line": 45,
      "change": "detailed example"
    },
    {
      "file": "file2.py",
      "line": 67,
      "same_pattern_as": "file1.py"
    },
    "... (45 more files with same pattern)"
  ]
}
```

### 4. **Validation-Centric View**
Group files by **which validation they fix**, not by alphabetical order:
- Problem 1 → files fixing validation step 1
- Problem 2 → files fixing validation step 4
- etc.

---

## Enhanced Decomposition for Multi-Validation

Update the decomposition prompt to extract patterns:

```python
# In decompose_ci_failure.py, enhance the consolidation prompt:

prompt = f"""Consolidate {len(chunk_findings)} chunk findings into validation-step-ordered problems.

CRITICAL for large repairs (100+ files):
1. Group changes by CI validation step (not by file)
2. Within each validation, identify PATTERNS (if 10+ files have same fix)
3. For each pattern: ONE concrete example + list of all files
4. Keep track: which files map to which validation step

INPUTS:
- Validation sequence: {validation_sequence}
- Chunk findings: {chunk_findings}

OUTPUT:
{{
  "validation_problems": [
    {{
      "validation_step": 1,
      "validation_cmd": "poetry install",
      "files_count": 3,
      "patterns": [
        {{
          "pattern_id": "dep_version_update",
          "pattern": "Update mdformat-beautysh 0.1.1 → 1.0.0",
          "files_count": 3,
          "example": {{
            "file": "dev/pyproject.toml",
            "line": 45,
            "before": "mdformat-beautysh = \\"0.1.1\\"",
            "after": "mdformat-beautysh = \\"1.0.0\\""
          }},
          "all_files": ["dev/pyproject.toml", "framework/pyproject.toml", "baselines/pyproject.toml"]
        }}
      ]
    }},
    {{
      "validation_step": 4,
      "validation_cmd": "ruff check .",
      "files_count": 70,
      "patterns": [
        {{
          "pattern_id": "remove_unused_imports",
          "files_count": 15,
          "example": {{...}},
          "all_files": [...]
        }},
        {{
          "pattern_id": "rst_title_formatting",
          "files_count": 47,
          "example": {{...}},
          "all_files": [...]
        }}
      ]
    }}
  ]
}}
"""
```

---

## L1 Memory for Multi-Validation

### Option A: Pattern-Based L1 (Recommended)
One L1 entry per pattern (not per file):

```json
{
  "memory_level": "L1",
  "pattern_id": "rst_title_symmetric_formatting",
  "applies_to_files": 47,
  "repo": "flower",
  "validation_step": 6,
  "failed_cmd": "./dev/test.sh",
  
  "problem_pattern": {
    "symptom_pattern": "Error: 'Title underline too short at line N'",
    "root_cause": "mdformat-beautysh 1.0.0 requires symmetric overline+underline for RST document titles",
    "affects_file_type": "*.rst files in docs/",
    "common_locations": "First heading in .rst files (document title)"
  },
  
  "fix_pattern": {
    "steps": [
      "1. Count title character length (e.g., 'My Title' = 8 chars)",
      "2. Add overline with N '=' characters above title",
      "3. Ensure underline also has exactly N '=' characters",
      "4. Verify: overline and underline should be identical"
    ],
    "example": {
      "title": "Set Environment Variables",
      "title_length": 25,
      "before": "Set Environment Variables\\n=========================",
      "after": "=========================\\nSet Environment Variables\\n========================="
    }
  },
  
  "example_files": [
    {
      "file": "framework/docs/source/docker/set-environment-variables.rst",
      "lines": "4-6",
      "title_length": 25
    },
    {
      "file": "framework/docs/source/docker/enable-tls.rst",
      "lines": "3-5",
      "title_length": 18
    },
    {
      "file": "framework/docs/source/tutorial/quickstart.rst",
      "lines": "1-3",
      "title_length": 32
    }
  ],
  
  "all_affected_files": [
    "framework/docs/source/docker/*.rst (12 files)",
    "framework/docs/source/tutorial/*.rst (8 files)",
    "framework/docs/source/how-to/*.rst (15 files)",
    "framework/docs/source/ref/*.rst (12 files)"
  ],
  
  "verification": "./dev/test.sh # should pass RST validation for all files"
}
```

### Option B: File-Specific L1 with Pattern Reference
One L1 per file, but reference the pattern:

```json
{
  "memory_level": "L1",
  "file": "framework/docs/source/docker/set-environment-variables.rst",
  "pattern_id": "rst_title_symmetric_formatting",
  "pattern_instance": 1,
  "total_instances": 47,
  
  "problem_in_this_file": "...",
  "concrete_change": "...",
  
  "see_pattern": {
    "pattern_id": "rst_title_symmetric_formatting",
    "same_fix_applies_to": 46,
    "other_examples": ["enable-tls.rst", "quickstart.rst", "..."]
  }
}
```

---

## Summary

**For 110 files with 21 validations:**

1. **Decomposition** → 21 atomic problems (by validation step)
2. **Pattern extraction** → N patterns within each problem (e.g., 3 patterns in problem 2)
3. **L1 memory** → Either:
   - Pattern-based (N entries for N patterns) 
   - File-based with pattern refs (110 entries referencing N patterns)
4. **L2 memory** → One entry with 21 atomic_problems, each containing pattern groups
5. **L3 memory** → Extract N universal patterns from the L2 patterns

This keeps memory **concrete and actionable** without exploding into 110 separate L1 entries with redundant information.

---

## Recommendation

Use **Pattern-Based L1** when:
- 10+ files have identical or very similar fixes
- Changes are systematic (version updates, formatting, import changes)

Use **File-Based L1** when:
- Each file has unique issues
- Fixes require file-specific context
- < 10 files per pattern

For your case (110 files, 21 validations): **Use pattern-based L1** with ~20-30 pattern entries instead of 110 file entries.
