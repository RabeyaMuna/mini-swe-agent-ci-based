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
    ci_visible_files_json = (
        json.dumps(ci_visible_files, indent=2) if ci_visible_files else "[]"
    )
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

DEPENDENCY-AWARE ANALYSIS:

The dependency context shows caller → callee relationships with BEFORE/AFTER changes.
Your task: Analyze caller and callee changes TOGETHER to understand if they create dependency problems.

## Analysis Steps:

1. **Analyze CALLER and CALLEE Changes TOGETHER**
   For each dependency, examine BEFORE/AFTER of BOTH sides:

   a) CALLER side:
      - What did CALLER do BEFORE? (old behavior/logic/usage)
      - What does CALLER do AFTER? (new behavior/logic/usage)
      - What does CALLER expect/require from CALLEES?

   b) CALLEE side:
      - What did CALLEES provide BEFORE? (old interface/format/behavior)
      - What do CALLEES provide AFTER? (new interface/format/behavior)
      - Did this change break what CALLER expected?

   c) JOINT ANALYSIS (critical!):
      - Compare CALLER expectations vs CALLEE changes
      - Does CALLEE's AFTER state break CALLER's BEFORE expectations?
      - Does CALLER's AFTER state adapt to CALLEE's AFTER state?
      - Example: If test reads "title from line 1" (CALLER BEFORE) but docs removed
        title (CALLEE AFTER), then CALLER must adapt → this is a dependency problem!

2. **Identify ROOT CAUSE and CASCADE Direction**
   Based on the TOGETHER analysis above:

   a) Which change happened FIRST conceptually?
      - CALLER changed and CALLEES had to follow? (CALLER is root cause)
      - CALLEES changed and CALLER had to adapt? (CALLEES are root cause)

   b) What PROBLEM was created?
      - CALLEE change broke CALLER's assumptions → "CALLER adaptation to CALLEE change"
      - CALLER change required CALLEE updates → "CALLEE alignment with CALLER change"

   c) Determine cascade direction:
      - Root cause file → "triggers" or "configures" → dependent files
      - Dependent files → "adapts to" or "follows" → root cause file

3. **Determine TRUE ISSUE TYPE**
   - Don't just use the validation name (e.g., "Code Formatting")
   - Analyze WHAT PROBLEM is actually being fixed
   - Examples of semantic issue types:
     * "Test adaptation to breaking API change"
     * "Import path update after module restructure"
     * "Type annotation fix after dependency upgrade"
     * "Documentation format migration"
     * "Configuration alignment with new library version"

4. **Map to CI VALIDATION (Separate from Issue Type!)**
   - ISSUE TYPE = What problem is being fixed (semantic)
   - VALIDATION = Which CI step would catch this change (mechanical)
   - These are DIFFERENT concepts!

   File type → Validation mapping:
   - Config files (.toml, .yaml, .json) → Config validation (taplo, etc.)
   - Test files (*_test.py, test_*.py) → Test validation (pytest, black on tests)
   - Documentation (.rst, .md) → Doc validation (docstrfmt, mdformat)
   - Python source → Code validation (black, mypy, ruff, pytest)
   - Proto/generated → Build validation (protoc, codegen)

5. **Classify as Cascading or Independent**
   Cascading means: Related changes across DIFFERENT validations
   - Caller in validation A, callees in validation B → cascading=true
   - Explain which change triggered the other in cascade_explanation
   - Use semantic analysis from steps 1-2 to explain the trigger

   Independent means: Unrelated changes
   - No semantic dependency despite file relationship
   - Both dependency_type and cascade_explanation should be empty strings

VISIBILITY RULE:
- visibility="primary" if at least one file appears in FILES VISIBLE IN CI FAILURE LOGS
- visibility="hidden" otherwise

## CONFIGURATION AND PACKAGE CHANGE COVERAGE

Classification is file-level, but your analysis must be operation-aware. For
every configuration or dependency file, inspect EVERY supplied before/after
entry, not only the first visible CI error. A single manifest may contain an
independent tool-key removal, package replacement, version-bound change, source
change, extra/group change, and related environment adaptation.

- BEFORE is the pre-fix state; AFTER is the applied repair. Never reverse them.
- Do not let one configuration failure hide package additions, removals,
  replacements, or version changes in the same file.
- In change_scope_summary, enumerate every exact changed key and every exact
  old/new package specification, including repeated declaration scopes.
- For package changes, note supported candidate impacts involving project
  runtime/tool versions, resolver constraints, environment compatibility,
  package replacement/deprecation, or related API/source adaptations.
- Do not assert deprecation or incompatibility without evidence. When the diff
  proves the package operation but not its reason, state that the causal
  constraint must be resolved during atomic analysis.
- A file still appears exactly once in classification; change_scope_summary
  preserves all independent operations for later atomic splitting.

## OUTPUT FORMAT

Return ONLY a JSON array with this format:

[
  {{
    "validation_order": <INT>,
    "validation_cmd": "<exact command from VALIDATIONS>",
    "failure_type": "<category>",
    "issue_type": "<specific>",
    "change_type": "<code|dependency|config>",
    "change_scope_summary": ["<exact before -> after operation>", "..."],
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
- change_scope_summary must cover every semantic operation represented by the
  files in this classification entry. For config/dependency files, it must name
  every changed key/package and exact before/after value.

CRITICAL - Semantic Issue Type:
- issue_type must describe the ACTUAL PROBLEM being fixed (semantic meaning)
- DO NOT just repeat the validation command name
- Analyze BEFORE/AFTER changes in dependency context to understand root cause
- Examples: "Test adaptation to API change" NOT "Code formatting"
- Examples: "Import fix after module restructure" NOT "Type checking"

CRITICAL - Cascading Analysis:
- For cascading groups, explain the trigger in cascade_explanation based on ACTUAL dependency context
- Use the BEFORE/AFTER changes provided to determine which change triggered the other
- For independent groups, dependency_type and cascade_explanation must be empty strings
- If uncertain, prefer independent unless dependency context clearly shows one change triggered the other

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
    ci_visible_files_json = (
        json.dumps(ci_visible_files, indent=2) if ci_visible_files else "[]"
    )
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

CONFIGURATION AND PACKAGE CHANGE COVERAGE:
- Inspect EVERY before/after entry for each configuration/dependency file.
- BEFORE is the pre-fix state and AFTER is the applied repair; never reverse the
  direction when describing the change.
- Do not classify a manifest only by its first visible configuration failure.
  Retain every package addition, removal, replacement, version/source/extra
  change, environment constraint, and independent configuration-key change.
- Populate change_scope_summary with one exact before -> after statement for
  every semantic operation. Include exact package names, complete constraints,
  and declaration scopes.
- Identify supported candidate package problems from project setup, runtime or
  tool versions, resolver/environment constraints, deprecation/replacement, and
  related source/API adaptations. If evidence does not establish the cause,
  say the causal constraint requires atomic analysis; do not invent it.

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
    "change_scope_summary": ["<exact before -> after operation>", "..."],
    "visibility": "<primary|hidden>",
    "files": [...],
    "total_files": <int>
  }}
]

Every classification entry must include a non-empty change_scope_summary that
collectively covers all semantic changes in its files. Listing a config file
without its package/key operations is incomplete classification.

{strict_json_rules}
"""

    return prompt
