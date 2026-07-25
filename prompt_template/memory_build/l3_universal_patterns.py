"""
L3 universal pattern extraction prompts.

Extracts generalizable problem patterns from repair sequences.
"""

import json
from typing import Any, Dict, List


def build_l3_full_extraction_prompt(
    l2_problems: List[Dict[str, Any]],
    strict_json_rules: str,
) -> str:
    """Build L3 full pattern extraction prompt."""

    l2_problems_json = json.dumps(l2_problems, indent=2)

    prompt = f"""Analyze CI failure patterns and extract UNIVERSAL PROBLEM PATTERNS (L3).

L2 Repair Sequence:
{l2_problems_json}

Task: Extract distinct, independent problem patterns with universal fixes.

CRITICAL RULES:
1. Each INDEPENDENT problem = separate entry
2. Only link problems if there's ACTUAL dependency
3. No forced grouping - mypy != mdformat (separate entries)
4. Extract universal fixes that apply to similar future problems
5. **IGNORE Git workflow issues** - DO NOT create patterns for merge conflicts
   - Merge conflict markers are Git problems, NOT CI validation problems
   - Focus on ACTUAL validation issues (type errors, formatting, imports, etc.)

Output JSON ARRAY of distinct patterns:
[
  {{
    "pattern_id": "numpy_private_type_annotation",
    "failure_type": "type checking",
    "verification_cmd": "python -m mypy py",
    "failure_pattern": "Private numpy type annotations fail without plugin",
    "problem": "< what is the problem, why it occurs, root cause >",
    "universal_fix": {{
      "approach": "Replace private types with public equivalents",
      "steps": [
        "1. Identify private type imports",
        "2. Find public equivalent (e.g., DTypeLike -> np.dtype[Any])",
        "3. Update annotations",
        "4. Remove plugin config"
      ],
      "applies_to": ["numpy private types", "pandas private types"]
    }},
    "examples": [
      {{
        "file": "ndarrays_arithmetic.py",
        "before": "dtype: DTypeLike = np.int64",
        "after": "dtype: np.dtype[Any] = np.int64"
      }}
    ],
    "dependent_problems": [
      {{
        "pattern_id": "pyproject_plugin_config",
        "relationship": "requires_config_change",
        "rationale": "Code change requires matching config update"
      }}
    ]
  }}
]

For EACH distinct validation/problem type, create separate entry.
Include dependencies ONLY if they actually exist.

{strict_json_rules}
"""

    return prompt


def build_l3_validation_group_prompt(
    validation_cmd: str,
    problems: List[Dict[str, Any]],
    strict_json_rules: str,
) -> str:
    """Build L3 validation group pattern extraction prompt."""

    problems_json = json.dumps(problems, indent=2)

    prompt = f"""Extract universal patterns from this validation group.

Validation: {validation_cmd}
Problems in this validation:
{problems_json}

Task: Identify DISTINCT problem patterns within this validation.

For EACH distinct pattern, extract:
1. Pattern ID (unique identifier)
2. Failure pattern (what breaks)
3. Problem (why it occurs, root cause)
4. Universal fix (approach + steps)
5. Examples (before/after)

Rules:
- Each pattern should be generalizable to similar future cases
- Use real examples from the problems provided
- Be specific about the fix approach
- Include actionable steps

Output JSON array:
[
  {{
    "pattern_id": "unique_identifier",
    "failure_type": "category",
    "verification_cmd": "{validation_cmd}",
    "failure_pattern": "what breaks",
    "problem": "why it occurs, root cause",
    "universal_fix": {{
      "approach": "high-level strategy",
      "steps": ["specific", "actionable", "steps"],
      "applies_to": ["applicable", "scenarios"]
    }},
    "examples": [
      {{
        "file": "actual/file/path.py",
        "before": "code before fix",
        "after": "code after fix"
      }}
    ]
  }}
]

{strict_json_rules}
"""

    return prompt


def build_l3_cross_validation_deps_prompt(
    patterns: List[Dict[str, Any]],
    strict_json_rules: str,
) -> str:
    """Build L3 cross-validation dependency analysis prompt."""

    patterns_json = json.dumps(patterns, indent=2)

    prompt = f"""Analyze dependencies between these patterns from different validations.

Patterns from multiple validations:
{patterns_json}

Task: Identify ACTUAL dependencies between patterns.

CRITICAL: Only link patterns if there's genuine causality:
- Code change requires config change
- Setup change enables code fix
- One fix cascades to another

Do NOT link if:
- Just happened in same commit (timing ≠ dependency)
- Same failure type (similarity ≠ dependency)
- No clear cause-effect relationship

For each pattern with dependencies, add:
{{
  "pattern_id": "pattern_name",
  "dependent_problems": [
    {{
      "pattern_id": "other_pattern",
      "relationship": "requires_config_change|requires_setup|cascades_to|blocks|enables",
      "rationale": "Clear explanation of WHY this dependency exists"
    }}
  ]
}}

Return updated patterns array with dependencies added.

{strict_json_rules}
"""

    return prompt
