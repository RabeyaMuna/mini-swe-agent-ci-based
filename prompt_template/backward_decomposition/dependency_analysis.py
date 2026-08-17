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

NOTE: This analysis uses updated dependency detection logic that looks beyond surface
validation types to identify actual changes (e.g., package upgrades in formatting fixes).

PROBLEMS (semantic information):
{problems_json}

GRAPH INFO (structural CI information):
{graph_info_json}

TASK:
Detect dependencies AND generate optimal repair sequence in ONE analysis.

## CROSS-CHUNK DEPENDENCY RECONCILIATION

The problems may have been extracted from different change chunks. Treat this
as the global reconciliation pass: compare every problem with every other
problem and recover semantic relationships that earlier file-graph or chunking
stages may not have discovered. Absence of an edge in GRAPH INFO does not prove
that two problems are independent.

Use exact subjects and before/after evidence in problem, root_cause, how_fixed,
why_fix_works, dependency_type, and cascade_explanation. Look for relationships
such as configuration or dependency changes requiring source/test adaptations,
API or symbol changes requiring caller updates, generated artifacts following
source changes, and one repaired validation revealing or enabling another.
Create an edge only when the supplied problem evidence supports the causal
direction; do not invent a relationship merely because files or validations are
similar.

## CASCADE_EXPLANATION IS STRONG DEPENDENCY EVIDENCE

When a problem has is_cascading=true and cascade_explanation is not empty, analyze
the explanation text to identify WHAT changed that triggered this cascade.

Look for mentions of:
- File paths (absolute or relative)
- Package/library names
- Version transitions (X.Y.Z → A.B.C)
- Configuration changes
- API/symbol changes

Then search ALL other problems to find which one made that change.

CRITICAL MATCHING RULE: Look beyond surface validation types. A problem described
as "TOML formatting" that actually upgrades a package IS a package upgrade problem.
Match on the ACTUAL CHANGE in how_fixed, not the validation failure type in problem.

Example:
- Problem A: "taplo formatting failed", how_fixed: "Updated mdformat-beautysh from 0.1.1 to 1.0.0"
  → Treat as PACKAGE UPGRADE, not formatting
- Problem B: cascade_explanation: "mdformat-beautysh 1.0.0 requires RST changes"
  → Match B to A based on package + version, create dependency A → B

CRITICAL: Semantic causality overrides CI execution order. If Problem A's changes
cause Problem B to fail (even if B runs first in CI), the dependency is A → B.

ORGANIZE BY:
1. **Semantic dependencies** (blocks, reveals, enables) - **HIGHEST PRIORITY**
2. File dependencies from graph_info (config → code)
3. CI validation sequence (validation_order) - use for independent problems only
4. Problem type (primary before hidden)

DEPENDENCY TYPES:

- "blocks": A must be fixed before B can be addressed
  Pattern: B's validation cannot run or will fail differently until A is fixed

- "reveals": Fixing A exposes problem B that was hidden
  Pattern: A's failure masked B; fixing A makes B visible to validation

- "enables": Fixing A makes B's fix possible or necessary
  Pattern: A changes environment/config/tools, B adapts to new state

- "affects": Fixing A changes B's behavior or requirements
  Pattern: A and B interact; A's fix has side effects on B

For EACH dependency, provide:
- from: problem_id
- to: problem_id
- reason: WHY this dependency exists (based on problem content + graph_info)

## DEPENDENCY DETECTION ALGORITHM

For each problem P:

1. **Extract CHANGE SIGNALS** from P (not just validation names):

   From cascade_explanation and root_cause, extract:
   - File paths mentioned (e.g., "dev/pyproject.toml", "src/foo.py")
   - Package/library names (e.g., "mdformat-beautysh", "numpy", "click")
   - Version transitions (e.g., "0.1.1 → 1.0.0", "7.x → 8.x")
   - Config values changed (e.g., "timeout: 30 → 60")
   - APIs/symbols changed (e.g., "DTypeLike removed", "function signature changed")
   - Action verbs ("upgraded", "changed", "added", "removed", "updated", "fixed")

2. **Extract ACTUAL CHANGES** from other problems Q:

   For each problem Q, identify the REAL CHANGE (not validation type):
   - From Q's how_fixed/root_cause: What VALUE changed? (versions, configs, code)
   - From Q's affected_files: What FILE was modified?
   - Example: Problem says "taplo formatting" but how_fixed says "upgraded X from 1.0 → 2.0"
     → The REAL CHANGE is the upgrade, not the formatting

   Build a change signature: (file, entity, before → after)

3. **Match signals to changes**:

   For each signal S from P:
   - Does any Q's change signature match S?
   - File match: S mentions file F, Q modified F
   - Entity match: S mentions package/version, Q's how_fixed mentions same package/version
   - Transition match: S says "X from A to B", Q's how_fixed contains "X" and "A" or "B"
   - If match: Candidate dependency Q → P

4. **Verify causation direction**:

   - Does Q's change logically cause/enable P's change?
   - Does P adapt to Q's new state?
   - If yes: Create dependency edge Q → P

Example:
- Problem Q: "taplo validation failed", how_fixed: "upgraded mdformat from 0.1.1 to 1.0.0"
  → REAL CHANGE: (dev/pyproject.toml, mdformat, 0.1.1 → 1.0.0)
- Problem P: cascade_explanation: "mdformat 1.0.0 requires RST formatting"
  → SIGNAL: (mdformat, 1.0.0)
- MATCH: Q's change introduced mdformat 1.0.0, P adapts to it
- DEPENDENCY: Q → P

REPAIR ORDER RULES (in priority order):

1. **RESPECT dependencies (from before to)** - HIGHEST PRIORITY
   Dependencies define causal order and MUST be respected first
2. Use validation_order for independent problems (no dependencies between them)
3. Within same validation, use file relationships from graph_info
4. Problem type: primary before hidden when all else is equal

CRITICAL: If semantic dependencies conflict with validation_order, the
dependencies win. Example: If Problem A (validation_order=15) causes Problem B
(validation_order=14), the repair sequence is [A, B] because A must be fixed first.

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
  "dependencies": [{{"from": 2, "to": 3, "reason": "..."}}],
  "repair_sequence": [1, 2, 3]
}}

Example 2 - All Independent:
Input: [{{id:1, validation_order:8}}, {{id:2, validation_order:11}}]
Dependencies: None
Output: {{
  "dependencies": [],
  "repair_sequence": [1, 2]
}}

Example 3 - Semantic Dependency Overrides Validation Order:
Input: [
  {{id:A, validation_order:N+1, affected_files:["config.file"]}},
  {{id:B, validation_order:N, is_cascading:true, cascade_explanation:"config.file changed X from Y to Z"}}
]
Analysis: Problem B mentions "config.file changed", Problem A modifies "config.file"
Output: {{
  "dependencies": [{{"from": A, "to": B, "reason": "config.file change requires B adaptation"}}],
  "repair_sequence": [A, B]
}}
Note: Repair sequence follows causality, not validation_order

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
