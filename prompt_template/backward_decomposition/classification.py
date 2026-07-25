"""
File classification prompts for CI validation analysis.

Classifies changed files by the CI validation step that would catch them.
"""

import json
from typing import Any, Dict, List


def build_classification_prompt_with_dependencies(
    ci_failure_context: Dict[str, Any],
    ci_visible_files: List[str],
    formatted_validations: List[Dict],
    chunk_index: int,
    total_chunks: int,
    files_in_chunk: int,
    formatted_chunk: str,
    dependency_info: str,
    strict_json_rules: str,
) -> str:
    """Build classification prompt with dependency context."""

    ci_failure_context_json = json.dumps(ci_failure_context, indent=2)
    ci_visible_files_json = json.dumps(ci_visible_files, indent=2) if ci_visible_files else "[]"
    formatted_validations_json = json.dumps(formatted_validations, indent=2)

    prompt = f"""Classify each changed file by the CI step that would catch or require the fixed issue.

## INPUT

CI failure context:
{ci_failure_context_json}

FILES VISIBLE IN CI FAILURE LOGS (primary errors):
{ci_visible_files_json}

Available validations:
{formatted_validations_json}

Changed files, chunk {chunk_index}/{total_chunks} ({files_in_chunk} files):
{formatted_chunk}

## DEPENDENCY CONTEXT

The chunk contains files with caller → callee relationships.
Use this to decide if related files should be grouped together.

{dependency_info}

## TASK

Classify files by CI validation step, USING dependency context for better decisions.

CLASSIFICATION BASIS:
1. CI failure context: visible/primary failures
2. Ground-truth diff: complete repair
3. Workflow validation sequence: all CI steps
4. Dependency relationships: caller → callee connections

For every file:
1. Inspect the provided before/after change data
2. Decide what CI validation the change fixes
3. Choose validation_order from VALIDATIONS
4. Set validation_cmd to the exact effective_cmd from VALIDATIONS
5. Group files with the same CI step + failure_type + issue_type
6. Determine visibility as primary or hidden

DEPENDENCY-AWARE DECISIONS:

CRITICAL: Dependency context helps UNDERSTAND the problem, but each file is STILL
classified by the CI validation that catches its specific change!

Use dependency context to:
1. Understand WHY changes happened (root cause)
2. Identify cascading relationships, their changes before and after
3. Link related problems across validations

BUT: Each file must be classified by WHICH CI validation would catch it!

Example:
  Dependency: exit_code_test.py READS ref-exit-codes/*.rst

  Analysis:
  - RST files changed format (caught by docstrfmt - validation 17)
  - Test file adapted assertions (caught by Black/pytest - validation 7)
  - They are RELATED (cascading), but different validations!

  Correct classification:
  - Group 1: RST files → validation 17 (docstrfmt)
    * is_cascading: true
    * cascade_explanation: "docstrfmt upgrade triggered test adaptation"

  - Group 2: Test file → validation 7 (Black/test validation)
    * is_cascading: true
    * cascade_explanation: "Test adapted to new RST format from validation 17"
    * DO NOT put test in docstrfmt validation!

Rules for file → validation mapping:
- Config files → Config validation (taplo, etc.)
- Test files → Test validation (Black, pytest, etc.)
- RST/docs → Doc validation (docstrfmt)
- Python code → Python validation (Black, mypy, ruff)

Cascading means: Related problems across different validations
- Mark related groups with is_cascading=true
- Use cascade_explanation to explain relationship
- But classify each file by its ACTUAL CI validation!

VISIBILITY RULE:
- visibility="primary" if at least one file appears in FILES VISIBLE IN CI FAILURE LOGS
- visibility="hidden" otherwise

## OUTPUT FORMAT

Return ONLY a JSON array with this format:

[
  {{
    "validation_order": <INT>,
    "validation_cmd": "<exact command from VALIDATIONS>",
    "failure_type": "<category>",
    "issue_type": "<specific>",
    "change_type": "<code|dependency|config>",
    "visibility": "<primary|hidden>",
    "files": [...],
    "total_files": <int>,
    "is_cascading": <true|false>,
    "dependency_type": "<dependency relationship type or empty string>",
    "cascade_explanation": "<explanation or empty string>"
  }}
]

REQUIREMENTS:
- Return valid JSON only. Do not include markdown or commentary.
- validation_order must be an INTEGER from VALIDATIONS.
- validation_cmd must exactly match an effective_cmd from VALIDATIONS.
- visibility must be "primary" or "hidden".
- Every changed file in this chunk must appear exactly once.
- Do not include files that are not in this chunk.
- For cascading groups, explain the trigger in cascade_explanation.
- For independent groups, dependency_type and cascade_explanation must be empty strings.
- If uncertain, prefer independent unless dependency context clearly shows one change triggered the other.

{strict_json_rules}
"""

    return prompt


def build_classification_prompt_regular(
    ci_failure_context: Dict[str, Any],
    ci_visible_files: List[str],
    formatted_validations: List[Dict],
    chunk_index: int,
    total_chunks: int,
    files_in_chunk: int,
    formatted_chunk: str,
    strict_json_rules: str,
) -> str:
    """Build regular classification prompt without dependencies."""

    ci_failure_context_json = json.dumps(ci_failure_context, indent=2)
    ci_visible_files_json = json.dumps(ci_visible_files, indent=2) if ci_visible_files else "[]"
    formatted_validations_json = json.dumps(formatted_validations, indent=2)

    prompt = f"""Classify each changed file by the CI step that would catch or require the fixed issue.

## INPUT

CI failure context:
{ci_failure_context_json}

FILES VISIBLE IN CI FAILURE LOGS (primary errors):
{ci_visible_files_json}

Available validations:
{formatted_validations_json}

Changed files, chunk {chunk_index}/{total_chunks} ({files_in_chunk} files):
{formatted_chunk}

## TASK

CLASSIFICATION BASIS:
Use all three evidence sources together:
1. CI failure context: identifies visible/primary failures, but may stop at the
   first failure and may not show later broken steps.
2. Ground-truth diff: shows the complete repair, including hidden setup,
   dependency, tooling, config, source, docs, test, build, and workflow fixes.
3. Full workflow validation sequence: shows setup, install, dependency,
   tooling, validation, docs, test, build, and workflow-local CI steps.

Do NOT classify only from CI logs. CI logs are incomplete.
A changed file absent from logs can still be a required hidden fix if the diff
and workflow show it supports any CI step.

For every file:
1. Inspect before/after changes
2. Decide what CI setup, installation, dependency, or validation failure the change fixes
3. Choose validation_order from VALIDATIONS
4. Set validation_cmd to effective_cmd from the chosen VALIDATIONS item
5. Group files with same effective CI step + failure_type + issue_type
6. Determine if file was VISIBLE in CI logs or HIDDEN

IMPORTANT CONTEXT:
- CI logs often show only the first failure, but the ground-truth diff is the
  complete repair needed for the whole CI workflow.
- A file absent from CI logs can still be a required hidden fix for setup,
  installation, dependency resolution, tool behavior, formatting, linting,
  typing, tests, docs, build, or workflow execution.
- Do not discard a changed file unless the before/after change is completely
  unrelated to this project's CI setup, dependencies, tooling, validations, or
  repair path.

Use two levels:
- failure_type: broad category (Type Checking, Linting, Formatting, etc.)
- issue_type: specific failure (missing annotation, unused import, etc.)

Config/dependency/tooling files must be classified dynamically:
- Treat package metadata, dependency files, lockfiles, workflow setup,
  environment config, tool config, and tool/plugin version changes as
  CI-relevant unless the diff proves otherwise.
- Infer the supported CI step from the changed package/tool/config key, nearby
  source/docs/test changes, and each VALIDATIONS item's effective_cmd,
  validates, source, and evidence.

VISIBILITY CLASSIFICATION:
- "primary" = file appears in "FILES VISIBLE IN CI FAILURE LOGS" above
- "hidden" = file does NOT appear in CI logs (enablement fix, cascaded fix)

## OUTPUT

Return JSON array:
[
  {{
    "validation_order": <INT>,
    "validation_cmd": "<exact>",
    "failure_type": "<category>",
    "issue_type": "<specific>",
    "change_type": "<code|dependency|config>",
    "visibility": "<primary|hidden>",
    "files": [...],
    "total_files": <int>
  }}
]

{strict_json_rules}
"""

    return prompt
