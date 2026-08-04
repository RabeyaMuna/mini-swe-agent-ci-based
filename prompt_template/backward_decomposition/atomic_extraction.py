"""
Atomic problem extraction prompt.

Analyzes a validation group and extracts atomic CI repair problems.
"""

import json
from typing import Any, Dict, Optional


# Change type context templates
CHANGE_TYPE_CONTEXTS = {
    "config": "These are CONFIGURATION file changes (.toml, .yaml, .json, .ini). Pay special attention to CI setup, installation commands, tool settings, plugin configurations, and dependency specifications.",
    "dependency": "These are DEPENDENCY-related changes (imports, packages, requirements). Focus on package installations, version updates, and import fixes.",
    "code": "These are SOURCE CODE changes (.py, .rst, .md). Focus on code logic, formatting, type annotations, and documentation fixes.",
}


def build_atomic_prompt(
    ci_context: Dict[str, Any],
    failed_validation_order: Any,
    val_info: Dict[str, Any],
    chunk: Dict[str, Any],
    changes_data: Dict[str, Any],
    dependency_context: str = "",
    cascading_context: str = "",
    strict_json_rules: str = "",
    all_validation_groups: Optional[list[Dict[str, Any]]] = None,
) -> str:
    """Build atomic problem extraction prompt."""

    validation_order = chunk.get("validation_order", "?")
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

    # Format validation group boundaries
    validation_groups_context = ""
    if all_validation_groups:
        groups_summary = []
        current_validation = validation_order

        for group in all_validation_groups:
            group_validation = group.get("validation_order", "?")
            group_files = group.get("files", [])
            group_issue = group.get("issue_type", "")
            is_current = group_validation == current_validation

            marker = ">>> THIS GROUP <<<" if is_current else ""
            groups_summary.append(
                f"  - Validation {group_validation}: {len(group_files)} files - {group_issue} {marker}"
            )

        validation_groups_context = f"""
## VALIDATION GROUP BOUNDARIES

ALL validation groups identified by classification:
{chr(10).join(groups_summary)}

CRITICAL: You are analyzing the group marked ">>> THIS GROUP <<<" only.
Files in OTHER validation groups should NOT be included in affected_files.
"""

    # Extract CI-visible files for problem_type classification
    ci_visible_files = set()
    if isinstance(ci_context, dict):
        # From relevant_files (CI log analysis)
        relevant_files = ci_context.get("relevant_files", [])
        for rf in relevant_files:
            if isinstance(rf, dict) and rf.get("file"):
                ci_visible_files.add(rf["file"])
            elif isinstance(rf, str):
                ci_visible_files.add(rf)

        # From effected_files (alternate naming)
        effected_files = ci_context.get("effected_files", [])
        for ef in effected_files:
            if isinstance(ef, dict) and ef.get("file"):
                ci_visible_files.add(ef["file"])
            elif isinstance(ef, str):
                ci_visible_files.add(ef)

    ci_visible_files_str = json.dumps(sorted(ci_visible_files), indent=2) if ci_visible_files else "[]"

    prompt = f"""Analyze this validation group and create atomic CI repair problems.

CI FAILURE CONTEXT:
{ci_context_str}

CI-VISIBLE FILES (files mentioned in CI logs/errors):
{ci_visible_files_str}

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

{validation_groups_context}

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

2. Decide merge vs split (CRITICAL: Based on ROOT CAUSE analysis).

## ANALYSIS PROCESS (Dynamic - apply to each issue):

STEP 2.1: For EACH file in CHANGES, answer these questions:
  a) What changed? (BEFORE -> AFTER)
  b) WHY did it need to change? (root cause)
  c) What triggered this need? (dependency context, cascade explanation)

STEP 2.2: Identify ROOT CAUSE patterns:
  - List unique root causes found across all files
  - Root cause = the underlying WHY that triggered the change
  - Root cause is about REASON, not about WHAT changed

STEP 2.3: Group files by ROOT CAUSE:
  - Files with IDENTICAL root cause -> candidate for merging
  - Files with DIFFERENT root causes -> must be separate problems

STEP 2.4: Apply MERGE/SPLIT decision:

  MERGE into ONE atomic problem when:
  OK SAME root cause (WHY) across all files
  OK Different changes (WHAT) are acceptable - these are variants
  OK One root_cause explanation covers why ALL files changed
  OK Example: 10 files, same root cause (package version upgrade),
    different changes (header/trailer/spacing/imports) -> 1 problem

  SPLIT into SEPARATE atomic problems when:
  FAIL DIFFERENT root causes (even if same validation!)
  FAIL Cannot explain all files with one root cause
  FAIL Mixing unrelated changes

## KEY INSIGHT: Same ROOT CAUSE + Different CHANGES = MERGE with variants

Root cause is about WHY (reason), not WHAT (manifestation):
- WHY same + WHAT different = MERGE (variants of same problem)
- WHY different = SPLIT (separate problems)

Think semantically:
- "Why did these files need to change?"
- If answer is SAME -> merge
- If answer is DIFFERENT -> split

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
- **CRITICAL - ONE ROOT CAUSE per problem:**
  * Each atomic problem must have ONE specific root cause that applies to EVERY affected file
  * If files have DIFFERENT root causes, they MUST be SEPARATE problems (even in same validation)
  * Root cause = the underlying reason WHY all these files needed to change
  * Example: "RST format upgrade" is one root cause for 10 files
  * Example: File A (API change) + File B (doc format) = TWO problems (different root causes)
- Be specific about packages, symbols, config keys, validators, commands, before/after states, directories, and affected file kinds.
- Do not mention line numbers.
- Do not use vague phrasing like "fixed issues" or "updated files".
- **CRITICAL - affected_files scope (VALIDATION GROUP BOUNDARIES):**
  * affected_files must ONLY include files from the CHANGES section (THIS validation group)
  * DO NOT include files from OTHER validation groups, even if they are cascading-related
  * If cascade_explanation mentions files from other validations, those are SEPARATE problems
  * Example: If pyproject.toml (validation 3) triggered exit_code_test.py (validation 13) adaptation:
    - Problem for validation 3: affected_files = ["pyproject.toml"] only
    - Problem for validation 13: affected_files = ["exit_code_test.py"] only (created separately)
  * List EVERY file from CHANGES section, but NEVER add files from other validation groups
  * Config files, dependency files, and code files in CHANGES ALL must be included
  * Example: If 26 pyproject.toml files all in CHANGES, list ALL 26 - but don't add files from other validations
- If no valid CI problem can be extracted, return {{"atomic_problems": []}}.

## DECISION PRINCIPLES (Dynamic - apply to YOUR specific changes):

PRINCIPLE 1: Root cause determines grouping
- Analyze WHY each file changed (not just WHAT changed)
- Group files that share the SAME underlying reason
- Split files that have DIFFERENT underlying reasons

PRINCIPLE 2: Variants are acceptable in ONE problem
- If 10 files changed for the SAME reason but in different ways, that's ONE problem
- Variant changes (header vs trailer vs spacing) with SAME root cause = ONE problem
- Describe variants in problem/root_cause/how_fixed, list ALL files in affected_files

PRINCIPLE 3: Validation boundary enforcement
- Files in DIFFERENT validations = ALWAYS separate problems (already enforced)
- Files in SAME validation but DIFFERENT root causes = MUST be separate problems

PRINCIPLE 4: Test your grouping
- Ask: "Can I explain why ALL these files changed with ONE root_cause statement?"
- If YES -> merge into one problem
- If NO -> split into separate problems

FIELD GUIDANCE (Dynamic - based on YOUR analysis):

- **root_cause**: The underlying WHY that applies to ALL affected files
  * Must be specific to THIS issue's context
  * Explain what changed/broke that required these files to adapt
  * For variants: describe the common root cause, not individual changes
  * For cascading: explain the dependency change that triggered adaptation

- **problem**: What failed and scope
  * Describe the failure based on root_cause
  * Mention file count, directory scope if multiple files
  * For variants: "X files with variant changes" (don't list all variants)
  * For cascading: explain the triggering relationship

- **how_fixed**: What changed to address root_cause
  * Describe the fix that applies across affected files
  * For variants: "Fixed via X approach with variants (header/trailer/spacing changes)"
  * For cascading: explain adaptation to new format/behavior

- **issue_type**: Specific semantic issue type (NOT just validation name!)
  * Based on root_cause analysis, not validation command
  * Be specific to THIS issue's actual problem
  * Avoid generic terms - make it semantic and meaningful

- **affected_files**: EVERY file from CHANGES with this root cause
  * List ALL files that share this root cause
  * Do NOT omit files even if many
  * Do NOT include files from other validation groups
  * Do NOT include files with different root causes

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
      "problem_type": "primary or hidden - see rules below",
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
- **CRITICAL: problem_type classification:**
  * "primary" = At least ONE affected file is in CI-VISIBLE FILES list above
  * "hidden" = NONE of the affected files are in CI-VISIBLE FILES list (discovered only through ground truth analysis)
  * Algorithm: If any(file in affected_files is in CI-VISIBLE FILES) -> "primary", else -> "hidden"
  * This distinguishes problems that CI logs revealed vs problems only found in the diff
- is_cascading must be a boolean matching the value from CLASSIFICATION CONTEXT.
- dependency_type must be a string (empty string if not cascading).
- cascade_explanation must be a string (empty string if not cascading).
- String fields must be non-empty for every returned problem: issue_type, problem, root_cause, how_fixed, why_fix_works.
- Do not include JavaScript-style comments in JSON.
- Do not include markdown, explanations, or text outside the JSON object.

{strict_json_rules}
"""

    return prompt
