"""
Dependency analysis prompts.

Analyzes dependencies between atomic problems to determine repair order.
"""

import json
from typing import Any, Dict, List


def build_full_dependency_prompt(
    problems: List[Dict[str, Any]],
    strict_json_rules: str,
) -> str:
    """Build full dependency analysis prompt for all problems."""

    problems_json = json.dumps(problems, indent=2)

    prompt = f"""Analyze the dependencies between these atomic CI repair problems.

Problems:
{problems_json}

TASK: Identify ALL dependencies and propose optimal repair order.

DEPENDENCY TYPES:

- "blocks": A must be fixed before B (blocker dependency)
  Examples: Missing import blocks type checking, broken setup blocks tests

- "enables": Fixing A allows B to be fixed (enabler dependency)
  Examples: Config change enables linter to run, dependency update enables feature

- "reveals": Fixing A reveals problem B (cascading dependency)
  Examples: Fixing formatter config reveals formatting issues in code

- "affects": Fixing A changes B (side effect)

- "independent": No relationship (can fix in any order)

For EACH relationship, provide:
- Type (one of above)
- Direction (from -> to)
- Reason (WHY this relationship exists, based on root_cause/how_fixed)
- Strength (strong/medium/weak based on how critical the relationship is)

REPAIR ORDER ANALYSIS:

CRITICAL: Validation order and problem_type define the base repair sequence:
1. PRIMARY problems (problem_type="primary") ALWAYS come first - these are CI failures
2. HIDDEN problems (problem_type="hidden") ALWAYS come after - these are consecutive validations
3. Within each group, RESPECT validation_order (validation 8 before validation 11)
4. Dependencies can ONLY reorder within same validation_order + problem_type group

Example correct order:
  Problem A: validation_order=8, problem_type="primary"     <- 1st (primary, earliest validation)
  Problem B: validation_order=8, problem_type="primary"     <- 2nd (primary, same validation, use deps)
  Problem C: validation_order=11, problem_type="hidden"     <- 3rd (hidden, later validation)
  Problem D: validation_order=11, problem_type="hidden"     <- 4th (hidden, same validation, use deps)

WRONG order (NEVER do this):
  FAIL Hidden before primary
  FAIL validation_order=11 before validation_order=8
  FAIL Ignoring validation sequence for "semantic dependencies"

Based on dependencies, determine WITHIN each (problem_type, validation_order) group:
1. Which problems should be fixed first (block others in same group)
2. Which are intermediate (depend on some, enable others in same group)
3. Which can be in any order (independent within same group)

OUTPUT FORMAT:
{{
  "dependency_edges": [
    {{
      "from": 1,
      "to": 3,
      "type": "blocks",
      "reason": "Config file must declare tool before validation can run it",
      "strength": "strong"
    }},
    {{
      "from": 2,
      "to": 4,
      "type": "reveals",
      "reason": "Fixing formatter config reveals formatting issues in code",
      "strength": "medium"
    }},
    // More edges...
  ],
  "repair_order": [1, 2, 3, 4, ...],
  "reasoning": "Detailed explanation of the dependency structure and repair strategy"
}}

IMPORTANT:
- Analyze based on actual content (root_cause, how_fixed), not just keywords
- Think about what a developer would need to know about problem interactions
- Use cascading metadata as evidence, but infer direction and relationship type
  from the problem content
- Identify both obvious (validation order) and subtle (semantic) dependencies
- Be specific in reasons - explain WHY the relationship exists
- Consider the real-world implications of fix order

{strict_json_rules}"""

    return prompt


def build_validation_group_dependency_prompt(
    validation_cmd: str,
    problems_data: List[Dict[str, Any]],
    strict_json_rules: str,
) -> str:
    """Build dependency analysis prompt for a validation group."""

    problems_json = json.dumps(problems_data, indent=2)

    prompt = f"""Analyze dependencies within this validation group.

Validation: {validation_cmd}
Problems in this validation:
{problems_json}

TASK: Identify dependencies between these problems.

CASCADING METADATA:
Some problems include is_cascading, dependency_type, and cascade_explanation.
These fields are evidence from earlier classification/deep analysis. Use them
to reason about interdependency, but do not treat them as automatic edges.

When cascading metadata is present:
- Identify which problem is the source/enabler that changed behavior,
  configuration, tooling, dependency, docs, tests, or code.
- Identify which problem is the dependent/adaptation problem.
- Create a dependency only when problem/root_cause/how_fixed supports it.
- Direction should be source/enabler -> dependent/adaptation.
- Keep the existing output schema exactly as shown below.

Analyze TWO types of relationships:

1. FILE-BASED DEPENDENCIES:
   - How do file changes link to each other?
   - Does changing file A affect file B?
   - Do files share context (same module, same functionality)?

   Examples:
   - Config file (pyproject.toml) affects code files
   - Import changes affect files that import from them
   - Files in same module/package interact

2. PROBLEM-BASED DEPENDENCIES:
   - How does one problem link to another?
   - Does fixing problem A enable/reveal/block problem B?
   - Are they part of same logical fix?

   Examples:
   - Fixing imports enables type checking
   - Config change enables linter to run
   - One problem reveals another (cascade)

OUTPUT format:
{{
  "dependencies": [
    {{
      "from": 1,
      "to": 2,
      "type": "blocks|enables|reveals|affects",
      "reason": "Explain how problem 'from' relates to problem 'to'",
      "file_link": "Optional: Explain file-based connection",
      "strength": "strong|medium|weak"
    }}
  ]
}}

Rules:
- Only include ACTUAL dependencies (not forced)
- Explain both file-based AND problem-based reasoning
- Use cascading metadata as evidence, but infer direction and relationship type
  from the problem content
- If no dependencies exist, return: {{"dependencies": []}}
- Be specific - explain WHY the dependency exists
- ALWAYS return valid JSON object with "dependencies" key

{strict_json_rules}"""

    return prompt
