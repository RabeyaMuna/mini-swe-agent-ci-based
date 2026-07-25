"""
L2 repair sequence generation prompts.

Generates optimal repair sequences for CI failures.
"""

import json
from typing import Any, Dict, List


def build_l2_full_sequence_prompt(
    issue_id: str,
    repo: str,
    problems: List[Dict[str, Any]],
    dependencies: Dict[str, Any],
    strict_json_rules: str,
) -> str:
    """Build L2 full repair sequence prompt."""

    problems_json = json.dumps(problems, indent=2)
    dependencies_json = json.dumps(dependencies, indent=2)

    prompt = f"""Generate L2 REPAIR SEQUENCE for CI failure resolution.

Issue: {issue_id}
Repo: {repo}

Problems to fix:
{problems_json}

Dependencies:
{dependencies_json}

Create optimal repair sequence in this EXACT format:
{{
  "problems": [
    {{
      "problem_id": 1,
      "verification_cmd": "python -m mypy py",
      "failure_type": "type_checking",
      "problem": "Clear description [error_code]",
      "root_cause": "Technical explanation. Scope: affected area",
      "fix_strategy": "Single comprehensive paragraph explaining: what to change, how to fix it, and why it works. Use the how_fixed and why_fixed_works from input data, combining them naturally into one flowing explanation.",
      "pattern_detected": null or {{
        "type": "bulk_formatting",
        "rule": "Pattern description",
        "scope": "X files"
      }},
      "files": ["path/to/file-1.ext", "path/to/file-2.ext", ...],
      "estimated_time_minutes": 5
    }},
    // Repeat for each problem in repair order
  ],
  "total_problems": <number of problems in repair sequence>,
}}

CRITICAL: fix_strategy must be a natural, flowing paragraph that:
1. Starts with WHAT and HOW to fix (from input how_fixed)
2. Continues with WHY it works (from input why_fixed_works)
3. NO labels like "Why this works:" - just natural sentences
4. Example: "Added .strip() method call to ensure whitespace is removed. The .strip() method removes leading/trailing whitespace, ensuring the value satisfies type expectations."

CRITICAL ORDERING RULES (MUST FOLLOW):
1. PRIMARY problems (problem_type="primary") MUST come FIRST
   - These are CI failures visible in logs
   - Fix these before any hidden problems

2. HIDDEN problems (problem_type="hidden") come AFTER primary
   - These are consecutive validations that never ran
   - Fix in validation_order sequence

3. Within each group (primary/hidden):
   - Respect validation_order (lower order = earlier in sequence)
   - Use dependencies to break ties (if problem A depends on B, do B first)
   - Independent problems at same validation can be in any order

4. NEVER reorder such that hidden comes before primary
5. NEVER violate validation sequence (validation_order 8 must come before 11)

OTHER RULES:
- "files": Array of ACTUAL file paths from diff (max 50), NO speculation or patterns
- Detect patterns for bulk operations (>10 files)
- Be specific and actionable

{strict_json_rules}
"""

    return prompt


def build_l2_validation_group_prompt(
    validation_cmd: str,
    problems: List[Dict[str, Any]],
    strict_json_rules: str,
) -> str:
    """Build L2 validation group prompt."""

    problems_json = json.dumps(problems, indent=2)
    problem_count = len(problems)

    prompt = f"""Organize repair sequence for validation: {validation_cmd}

Problems in this validation ({problem_count}):
{problems_json}

Generate L2 entries for these problems in this EXACT format:
{{
  "problems": [
    {{
      "problem_id": <original_id>,
      "verification_cmd": "{validation_cmd}",
      "failure_type": "...",
      "problem": "Clear description",
      "root_cause": "Technical explanation. Scope: affected area",
      "fix_strategy": "Single comprehensive paragraph: what to change, how to fix it, and why it works. Combine how_fixed and why_fixed_works from input naturally.",
      "pattern_detected": null or {{
        "type": "bulk_formatting",
        "rule": "Pattern description",
        "scope": "X files"
      }},
      "files": ["path/to/file.ext", ...],
      "estimated_time_minutes": 5
    }}
  ]
}}

CRITICAL: fix_strategy must combine input data naturally:
- Take how_fixed: "Added .strip() method call..."
- Take why_fixed_works: "The .strip() method removes whitespace..."
- Combine: "Added .strip() method call to ensure whitespace is removed. The .strip() method removes leading/trailing whitespace, ensuring the value satisfies type expectations."
- NO labels, just natural flowing text

{strict_json_rules}
"""

    return prompt
