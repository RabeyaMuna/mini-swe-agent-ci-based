# Commit-Based Decomposition

Alternative decomposition strategy that treats each commit as an individual repair unit.

## Overview

This module decomposes pull requests by analyzing commits individually (when independent) or in groups (when overlapping), producing detailed repair plans with:

1. **Problem identification** - What specific issues exist
2. **Root cause analysis** - Why the problems exist
3. **Fix strategy** - How to fix them
4. **Repair plan** - Exact file edits and validation commands

## Key Features

✅ **Only processes CI-validated files** - Ignores `.github/`, `docs/`, `*.md`, etc.
✅ **Commit-level granularity** - Treats each commit as repair unit
✅ **Smart grouping** - Groups overlapping commits together
✅ **Validation matching** - Maps changes to relevant CI checks
✅ **Complete repair plans** - Actionable file edits and validation commands

## Files

- `validation_matcher.py` - Filters files and matches validation commands
- `commit_analyzer.py` - LLM-based analysis of commits
- `commit_based_decomposer.py` - Main decomposition logic
- `run_commit_decomposition.py` - CLI entry point
- `test_run.sh` - Test script

## Usage

### Quick Test (3 issues)

```bash
bash commit_decomposition/test_run.sh
```

### Run on All Issues

```bash
python3 commit_decomposition/run_commit_decomposition.py \
    --dataset data/trs/filtered_issues.jsonl \
    --output data/trs/commit_decomposed_issues.json
```

### Run on Specific Issue

```bash
python3 commit_decomposition/run_commit_decomposition.py \
    --issue-id 113 \
    --output data/trs/commit_decomposed_issues.json
```

### Limit Number of Issues

```bash
python3 commit_decomposition/run_commit_decomposition.py \
    --limit 10 \
    --output data/trs/commit_decomposed_issues.json
```

## Output Format

```json
{
  "issue_id": "113",
  "sha_fail": "6aee1d58...",
  "sha_success": "714647c8...",
  "decomposition_type": "commit_based",

  "summary": {
    "total_commits": 2,
    "commit_groups": 1,
    "ci_failure_problems": 3,
    "indirect_problems": 0,
    "non_problems": 0
  },

  "ci_failures": [
    {
      "problem": {
        "file": "src/client.py",
        "lines": [35],
        "description": "Missing Optional import causes mypy error"
      },
      "root_cause": {
        "explanation": "Function returns None but type hint says str",
        "introduced_by": "commit_abc123",
        "why_not_caught_earlier": "Test coverage didn't include error path"
      },
      "fix_strategy": {
        "approach": "Add Optional import and update return type",
        "steps": [
          "Import Optional from typing",
          "Change return type from str to Optional[str]"
        ],
        "validation_cmd": "python -m mypy src/client.py",
        "validation_order": 7
      },
      "repair_plan": {
        "file_edits": [
          {
            "file": "src/client.py",
            "line": 1,
            "action": "add_import",
            "code": "from typing import Optional"
          },
          {
            "file": "src/client.py",
            "line": 35,
            "action": "replace",
            "old": "def func() -> str:",
            "new": "def func() -> Optional[str]:"
          }
        ],
        "validation_sequence": [
          "Run: python -m mypy src/client.py",
          "Expected: No errors"
        ]
      },
      "source_commits": ["abc123"],
      "relevance": "YES",
      "confidence": "high"
    }
  ],

  "repair_trajectory": [
    {
      "step": 1,
      "problem": "Missing Optional import causes mypy error",
      "root_cause": "Function returns None but type hint says str",
      "fix_strategy": "Add Optional import and update return type",
      "file_edits": [...],
      "validation_cmd": "python -m mypy src/client.py",
      "validation_order": 7
    }
  ]
}
```

## Algorithm

1. **Extract commits** between `sha_success` and `sha_fail`
2. **Filter to validated files** (ignore `.github/`, `docs/`, etc.)
3. **Group commits** by file overlap
4. **For each group:**
   - Get commit diff (only validated files)
   - Match to relevant CI validation commands
   - Use LLM to analyze:
     - What problems exist?
     - Why do they exist (root cause)?
     - How to fix them (strategy)?
     - Exact file edits (repair plan)
5. **Combine all problems** into sequential repair trajectory

## Comparison to PR-Level Decomposition

| Aspect | PR-Level | Commit-Level |
|--------|----------|--------------|
| Granularity | Entire PR at once | Per commit/group |
| Attribution | No commit info | Knows which commit |
| Files processed | All changed files | Only CI-validated files |
| Grouping | None | Smart overlap detection |
| Output | Problems list | Problems + root cause + fix strategy + repair plan |

## Dependencies

- `gitpython` - Git operations
- `litellm` - LLM API calls
- `python-dotenv` - Environment variables

## Configuration

Set environment variables:

```bash
export MEMCI_LLM_MODEL="openrouter/minimax/minimax-m2.5"
export OPENROUTER_API_KEY="your-key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

## Notes

- Requires `workflow_validation_cache.json` for validation sequence
- Requires cloned repositories in `repo/` directory
- Only processes commits that change CI-validated files
- Groups overlapping commits to handle dependencies
