"""
Dependency analysis prompts.

Analyzes dependencies between atomic problems to determine repair order.
"""

import json
from typing import Any, Dict, List


def build_full_dependency_prompt(
    problems: List[Dict[str, Any]],
    graph_info: Dict[str, Any],
    strict_json_rules: str,
) -> str:
    """
    Build unified dependency analysis + repair sequencing prompt.

    This combines dependency detection and repair order generation into ONE LLM call.
    The output is lightweight (just dependencies + sequence), then programmatically
    populated into the problems array.
    """

    problems_json = json.dumps(problems, indent=2)
    graph_info_json = json.dumps(graph_info, indent=2) if graph_info else "{}"

    prompt = f"""Analyze dependencies between problems and generate repair sequence.

PROBLEMS (semantic information):
{problems_json}

GRAPH INFO (structural CI information):
{graph_info_json}

TASK:
Detect dependencies AND generate optimal repair sequence in ONE analysis.

ORGANIZE BY:
1. CI validation sequence (validation_order: 1 → 8 → 11)
2. File dependencies from graph_info (config → code)
3. Semantic dependencies (blocks, reveals, enables)
4. Problem type (primary before hidden)

DEPENDENCY TYPES:

- "blocks": A must be fixed before B (blocker dependency)
  Examples: Missing import blocks type checking, broken setup blocks tests

- "reveals": Fixing A reveals problem B (cascading dependency)
  Examples: Fixing type annotation reveals incompatible assignment

- "enables": Fixing A allows B to be fixed (enabler dependency)
  Examples: Config change enables linter to run

- "affects": Fixing A changes B (side effect)

For EACH dependency, provide:
- from: problem_id
- to: problem_id
- reason: WHY this dependency exists (based on problem content + graph_info)

REPAIR ORDER RULES:

1. RESPECT validation_order (earlier validations first)
2. RESPECT dependencies (from before to)
3. Within same validation, use semantic dependencies
4. If no dependencies, use file relationships from graph_info
5. Problem type: primary before hidden

OUTPUT FORMAT:
{{
  "dependencies": [
    {{
      "from": 2,
      "to": 3,
      "reason": "Fixing type annotation at line 587 reveals incompatible assignment at line 598"
    }}
  ],
  "repair_sequence": [1, 2, 3, 4]
}}

RULES:
- dependencies: Array of edges (empty if all independent)
- repair_sequence: ALL problem IDs in repair order
- repair_sequence MUST respect all dependencies (from before to)
- repair_sequence MUST include ALL problem IDs exactly once
- If problems are independent, order by validation_order

EXAMPLES:

Example 1 - With Dependencies:
Input: [{{id:1, validation_order:8}}, {{id:2, validation_order:8}}, {{id:3, validation_order:11}}]
Dependencies: 2 reveals 3
Output: {{
  "dependencies": [{{"from": 2, "to": 3, "reason": "..."}]},
  "repair_sequence": [1, 2, 3]
}}

Example 2 - All Independent:
Input: [{{id:1, validation_order:8}}, {{id:2, validation_order:11}}]
Dependencies: None
Output: {{
  "dependencies": [],
  "repair_sequence": [1, 2]
}}

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
