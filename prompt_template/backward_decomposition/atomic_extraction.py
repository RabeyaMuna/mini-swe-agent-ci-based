"""
Atomic problem extraction prompt.

Analyzes a validation group and extracts atomic CI repair problems.
"""

import json
from typing import Any, Dict


# Change type context templates
CHANGE_TYPE_CONTEXTS = {
    "config": "These are CONFIGURATION file changes (.toml, .yaml, .json, .ini). Pay special attention to CI setup, installation commands, tool settings, plugin configurations, and dependency specifications.",
    "dependency": "These are DEPENDENCY-related changes (imports, packages, requirements). Focus on package installations, version updates, and import fixes.",
    "code": "These are SOURCE CODE changes (.py, .rst, .md). Focus on code logic, formatting, type annotations, and documentation fixes.",
}


def build_atomic_prompt(
    ci_context: Dict[str, Any],
    failed_validation_order: Any,
    validation_order: Any,
    val_info: Dict[str, Any],
    chunk: Dict[str, Any],
    changes_data: Dict[str, Any],
    dependency_context: str = "",
    cascading_context: str = "",
    strict_json_rules: str = "",
) -> str:
    """Build atomic problem extraction prompt."""

    change_type = chunk.get("change_type", "unknown")
    change_type_context = CHANGE_TYPE_CONTEXTS.get(change_type, "")

    effective_validation_cmd = (
        val_info.get("validation_cmd") or chunk.get("validation_cmd") or ""
    )
    failure_type = chunk.get("failure_type", "")

    # Compact CI context if needed (limit to ~6000 chars)
    ci_context_str = json.dumps(ci_context, indent=2)
    if len(ci_context_str) > 6000:
        ci_context_str = ci_context_str[:6000] + "\n... (truncated)"

    changes_json = json.dumps(changes_data, indent=2)
    validation_cmd_json = json.dumps(effective_validation_cmd)
    failure_type_json = json.dumps(failure_type)
    is_cascading_json = json.dumps(chunk.get("is_cascading", False))
    dependency_type_json = json.dumps(chunk.get("dependency_type", ""))
    cascade_explanation_json = json.dumps(chunk.get("cascade_explanation", ""))

    prompt = f"""Analyze this validation group and create atomic CI repair problems.

CI FAILURE CONTEXT:
{ci_context_str}

VALIDATION CONTEXT:
- validation_order: {validation_order}
- validation_cmd: {effective_validation_cmd}
- validates: {val_info.get("validates", "Code quality/formatting")}
- failure_type: {failure_type}
- issue_type_hint: {chunk.get("issue_type", "")}
- change_type: {change_type.upper()}
- FAILED_VALIDATION_ORDER: {failed_validation_order} (CI stopped here)
- is_cascading: {chunk.get("is_cascading", False)}
- dependency_type: {chunk.get("dependency_type", "")}
- cascade_explanation: {chunk.get("cascade_explanation", "")}

{change_type_context}

CHANGES:
{changes_json}

{dependency_context}

{cascading_context}

TASK:
Infer the actual CI step problem fixed by these before/after changes.

This group contains only {change_type.upper()} changes. Preserve concrete details from those changes. CI steps may include setup, installation,
dependency resolution, environment preparation, formatting, linting, type checking, tests, docs checks, build steps, and workflow-local commands.

DECISION PROCESS:

1. Identify the CI step being repaired.
- validation_cmd may be an install/setup command, not only a final checker.
- Package metadata, dependency files, lockfiles, workflow setup, environment config, and tool installation changes belong to the relevant setup/
install CI step.
- Source, docs, and test changes belong to the validator that directly checks them.
- Prefer the CI step that would fail without this specific change.

2. Decide merge vs split.
Merge changes into one atomic problem when one explanation clearly covers all affected files:
- same validation_cmd, validator, or tool family
- same repair family or developer mental model
- variants of the same failure family
- repeated instances of the same validator problem across multiple files

Split changes into separate atomic problems when one explanation would hide important differences:
- different CI step or validator concern
- materially different repair strategy
- setup/config/dependency enablement mixed with source/docs/test fixes
- same directory or validator, but different problem family, root cause, or repair family

3. Handle repeated failures across files dynamically.
- Same validator plus same repair family across many files is one repeated problem pattern, even when files have variants.
- Formatter/linter/doc-style variants are usually one problem when the same tool normalizes them, such as RST heading underline length, trailing
whitespace, blank-line spacing, list/table spacing, import ordering, docstring style, quote style, or repeated lint codes.
- For bulk changes, group by directory scope, file type, validator, and repair family.
- Mention directory scope and important variants in problem/root_cause/how_fixed.
- Do not list every file in prose because affected_files already contains exact paths.

4. Keep setup/install enablement separate.
- Examples: invalid pyproject metadata, missing dependency, wrong extras, incompatible tool version, broken pip/poetry install config, workflow
setup command mismatch.
- If setup changes only enable a later formatter, linter, type checker, or test command, report the setup/install issue separately from later
validation violations.

5. Handle cascading fixes.
- Cascading means one change caused or required another related change.
- If all affected files share the same CI validation and repair family, they may be one atomic problem.
- If related files are caught by different CI validations or require different repair strategies, split them into separate atomic problems.
- For cascading problems, explain the triggering relationship in problem, root_cause, how_fixed, or why_fix_works.

6. Handle merge-conflict cleanup correctly.
- Git conflict markers and conflict resolution mechanics are not CI problems.
- Never use "merge conflict" or "conflict resolution" as failure_type, issue_type, problem, root_cause, or how_fixed.
- Do not discard real fixes because they appear near removed conflict markers.
- Analyze the final before/after content around the conflict and classify the CI-relevant change that remained after resolution.
- If conflict resolution selected or combined code/docs/config that fixes a formatter, linter, type, test, setup, dependency, or docs validation
issue, report that real validation/setup problem.
- If the only change is conflict marker removal with no CI-relevant behavior, formatting, config, docs, dependency, or workflow change, do not
create a problem for that change.

QUALITY RULES:
- Each problem must have a specific root cause and fix that applies to every affected file.
- Be specific about packages, symbols, config keys, validators, commands, before/after states, directories, and affected file kinds.
- Do not mention line numbers.
- Do not use vague phrasing like "fixed issues" or "updated files".
- **CRITICAL: affected_files must list EVERY SINGLE file from the CHANGES section that has ANY direct or indirect relation with the CI verification**
  * If a file change is related to fixing this CI validation failure, it MUST be in affected_files.
  * Do NOT omit files even if they seem similar or repetitive - LIST EVERY FILE.
  * Config files, dependency files, and code files ALL must be included if they contribute to fixing this validation.
  * Example: If 26 pyproject.toml files all have the same fix for the same validator, list ALL 26 files in affected_files.
- If no valid CI problem can be extracted, return {{"atomic_problems": []}}.

EXAMPLES:
- MERGE: RST formatting failures in docs/api/*.rst with variants including section underline length mismatches, trailing whitespace, and blank-
line spacing normalization.
- MERGE: Ruff unused imports removed across several Python modules.
- SEPARATE: Ruff source-code errors and pyproject Ruff configuration changes.
- SEPARATE: RST formatting cleanup and broken docs reference/import targets.
- SEPARATE: Dependency/setup change that enables the formatter and formatter violations in docs files.
- SEPARATE: Formatter plugin version bump in pyproject.toml that enables the docs formatter, and RST formatting violations in docs files.

FIELD GUIDANCE:
- problem: 1-2 sentences describing what failed. Mention directory scope, file types, and important variants when relevant. If this is a cascading
problem, explain the relationship.
- root_cause: 1-2 sentences explaining what violated which rule, requirement, or expectation. For cascading problems, explain how the dependency
change triggered this fix.
- how_fixed: 1-2 sentences describing what changed and why it was necessary. Include variants when one atomic problem covers multiple variants.
For cascading problems, explain what format/behavior changed in the dependency.
- why_fix_works: 1-2 sentences explaining how the new state satisfies the CI step or validator or handles the new format/behavior from
dependencies.
- issue_type: Specific failure subtype, error code, rule, or validator-specific category. Be precise, such as "RST Formatter: Section Underline
Length Mismatch", "Ruff: Unsorted Import", "Dependency: Package Version Mismatch", "Test Parser Logic Update (Cascading)", or "Test Failure:
Assertion Mismatch".
- **affected_files: MUST contain EVERY file from CHANGES that relates to this problem. Do NOT omit, truncate, or sample files.**
- problem_type: "primary" when affected_files are visible in CI failure context; otherwise "hidden".

OUTPUT FORMAT:
{{
  "atomic_problems": [
    {{
      "problem_id": 1,
      "validation_order": {validation_order},
      "validation_cmd": {validation_cmd_json},
      "failure_type": {failure_type_json},
      "issue_type": "specific_error_code_or_type",
      "problem": "Brief description of what broke",
      "root_cause": "Why it failed",
      "how_fixed": "What changed",
      "why_fix_works": "Why the fix solves it",
      "affected_files": ["file1.py", "file2.py"],
      "problem_type": "primary",
      "is_cascading": {is_cascading_json},
      "dependency_type": {dependency_type_json},
      "cascade_explanation": {cascade_explanation_json}
    }},
    // Additional problems if any
  ]
}}

OUTPUT REQUIREMENTS:
- Return only a JSON object with the "atomic_problems" array.
- If no problems can be extracted, return {{"atomic_problems": []}}.
- problem_id must be an integer starting at 1 and incrementing by 1.
- validation_order must be the integer validation_order from VALIDATION CONTEXT.
- validation_cmd must exactly match validation_cmd from VALIDATION CONTEXT.
- failure_type must match failure_type from VALIDATION CONTEXT unless the value is empty.
- **CRITICAL: affected_files must be an array containing EVERY SINGLE file path from the CHANGES section above that is related to this problem.**
  * Count the files in CHANGES and ensure affected_files has the SAME COUNT if all files share the same problem.
  * DO NOT truncate, sample, or omit files - include EVERY file that has this issue.
  * If you see 26 files with the same formatter issue, affected_files MUST contain all 26 file paths.
- problem_type must be either "primary" or "hidden".
- is_cascading must be a boolean matching the value from CLASSIFICATION CONTEXT.
- dependency_type must be a string (empty string if not cascading).
- cascade_explanation must be a string (empty string if not cascading).
- String fields must be non-empty for every returned problem: issue_type, problem, root_cause, how_fixed, why_fix_works.
- Do not include JavaScript-style comments in JSON.
- Do not include markdown, explanations, or text outside the JSON object.

{strict_json_rules}
"""

    return prompt
