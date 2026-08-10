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
- classification_change_scope_summary: {json.dumps(chunk.get("change_scope_summary", []), ensure_ascii=False)}
- change_type: {change_type.upper()}
- FAILED_VALIDATION_ORDER: {failed_validation_order} (CI stopped here)
- is_cascading: {chunk.get("is_cascading", False)}
- dependency_type: {chunk.get("dependency_type", "")}
- cascade_explanation: {chunk.get("cascade_explanation", "")}

{change_type_context}

CHANGES:
{changes_json}

## MANDATORY CONFIGURATION PRESERVATION AND CAUSAL REASONING

Preserve EVERY configuration-related operation shown in CHANGES. Configuration
coverage is operation-level, not file-level: mentioning a manifest or config
path in affected_files does not cover the keys, packages, groups, commands,
versions, sources, extras, markers, or environment constraints changed inside
that file.

For EACH configuration operation:
1. State the exact BEFORE value/state and exact AFTER value/state.
2. Identify whether it was added, removed, replaced, renamed, enabled/disabled,
   or had a constraint/value changed.
3. Explain the most plausible reason the modification was required using the
   supplied project and CI evidence: project setup, installer/resolver behavior,
   runtime/language/tool versions, operating system or architecture constraints,
   package availability/deprecation/replacement, transitive constraints, or
   correlated source/test/API adaptations.
4. Clearly distinguish evidence from inference. Use wording such as "the
   supplied changes support..." for supported reasoning. If multiple causes are
   possible, list the supported possibilities and say which evidence would
   distinguish them. If none is supported, state that the exact causal reason
   is not supplied; never invent one.
5. Keep an existing configuration problem and create additional atomic problems
   for package/environment operations when they have different failure patterns
   or causes. Never replace the config problem with the package problem, or the
   package problem with the config problem.

FINAL FIELD REQUIREMENTS FOR CONFIGURATION CHANGES:
- problem: name the exact invalid/risky prior configuration and its possible
  install, resolution, build, validation, API, or runtime consequence.
- root_cause: explain why EACH changed key/package/constraint needed modification
  based on issue, project setup, environment, dependency, and adaptation evidence.
- how_fixed: enumerate ALL applied BEFORE -> AFTER operations exactly, including
  removals. Do not summarize them as "updated configuration" or "updated
  dependencies" and do not reverse the diff direction.
- why_fix_works: connect every applied operation to the failure or compatibility
  risk it resolves; when causality is not supplied, state the bounded expected
  effect without presenting speculation as fact.

{dependency_context}

{cascading_context}

{validation_groups_context}

TASK:
Infer the actual CI step problem fixed by these before/after changes.

## MANDATORY WHOLE-DIFF COVERAGE

Reason over every semantic before/after operation in CHANGES, regardless of
file type or technology. Do not stop after explaining the first CI-visible
failure. That failure may only be the first blocker, while other configuration,
dependency, workflow, source, test, build, or documentation changes repair
problems that become visible afterward.

Before returning JSON, perform an internal coverage audit:
1. Enumerate every independent changed subject in every change.
2. Map each subject to an atomic problem whose problem/root_cause/how_fixed
   states the exact old and new value or behavior.
3. Preserve a valid configuration problem when other changes in the same file
   require additional dependency, compatibility, API, build, or runtime problems.
4. Split changes into separate problems when their failure pattern or root cause
   differs; never erase one problem merely because another fails earlier.
5. For dependency changes, reason about deprecation or replacement, version and
   environment compatibility, resolver constraints, extras, transitive effects,
   and correlated source adaptations only when supplied evidence supports them.
   If the causal reason is not established, still report the exact change and
   explicitly state that the causal constraint is not supplied.

The final atomic_problems array must collectively cover every semantic operation
in CHANGES. Do not output the internal checklist.

The classification_change_scope_summary is an internal coverage handoff, not a
replacement for CHANGES. Every operation named there must be represented in the
final problem/root_cause/how_fixed text. Preserve all valid configuration-key
problems while adding separate package or environment problems when their
failure pattern or cause differs.

DIFF DIRECTION IS AUTHORITATIVE:
- BEFORE is the broken/pre-fix project state.
- AFTER is the ground-truth repair that was actually applied.
- how_fixed must describe BEFORE -> AFTER exactly. Never claim that an AFTER
  value was restored to BEFORE, or that a removed key/package was added.

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

STEP 2.1: For EACH individual before/after change in CHANGES, answer these questions:
  a) What changed? (BEFORE -> AFTER)
  b) What exact failure signal or violated contract does the BEFORE state show?
  c) What concrete problem would that failure signal cause in this validation?
  d) WHY did it need to change? (root cause)
  e) What triggered this need? (CI context, dependency context, cascade explanation)
  f) What exact repair strategy does the AFTER state implement?
  - A single file may contain multiple independent changes with different root
    causes. Analyze every change in the file; mentioning the file once does not
    mean all of its changes have been covered.

STEP 2.2: Identify FAILURE and ROOT CAUSE patterns:
  - List the exact failure signal/pattern for every affected file
  - List the concrete problem represented by that signal
  - List unique root causes and repair strategies across all files
  - Failure signal/pattern = the observable invalid state, violated contract,
    diagnostic, assertion mismatch, invalid value, or incompatible behavior
  - Root cause = the underlying WHY that triggered the change
  - Root cause is about REASON, not about WHAT changed

STEP 2.3: Group files by FAILURE PATTERN + PROBLEM + ROOT CAUSE:
  - Files are candidates for merging only when they have the SAME failure
    pattern, SAME concrete problem, SAME root cause, and compatible repair strategy
  - A shared validator or broad failure_type alone is NEVER sufficient to merge
  - Files with different failure patterns, problems, root causes, or materially
    different repair strategies must be separate problems
  - Different changes within the SAME file must also be split when they have
    different root causes and problems. The same file may appear in multiple problems.

STEP 2.4: Apply MERGE/SPLIT decision:

  MERGE into ONE atomic problem when:
  - SAME failure signal/pattern across all files
  - SAME concrete problem and SAME root cause (WHY) across all files
  - The fixes use the same repair strategy or true variants of that strategy
  - One precise problem/root_cause/how_fixed explanation covers ALL files
  - Example: 10 files violate the same formatter rule for the same reason and
    are normalized by the same formatter operation -> 1 repeated problem

  SPLIT into SEPARATE atomic problems when:
  - DIFFERENT failure signals or violated contracts
  - DIFFERENT concrete problems or root causes (even if same validation!)
  - Materially different repair strategies
  - Cannot explain all files precisely with one problem and root cause
  - Mixing unrelated changes

## KEY INSIGHT: Merge only repeated instances of the SAME failure

The validator name or broad category does not define atomicity:
- SAME failure pattern + SAME problem + SAME root cause + compatible fix = MERGE
- Any material difference in those dimensions = SPLIT

Think semantically:
- "What exact failure would each BEFORE state produce?"
- "Why did it occur, and what repair does each AFTER state implement?"
- Merge only when those answers describe the same repeatable problem pattern.

3. Handle repeated failures across files dynamically.
- Repeated failures across many files are one problem only when the validator,
  failure pattern, concrete problem, root cause, and repair strategy match.
- Formatter/linter/doc-style occurrences may be one problem when the same exact
  rule or violated convention and the same normalization strategy apply, such
  as repeated RST heading, trailing-whitespace, import-order, or quote-style
  violations. Different rules or contracts remain separate problems.
- For bulk changes, group by directory scope, file type, validator, and repair family.
- Mention directory scope in problem and describe every distinct file-specific
  failure and repair in root_cause/how_fixed. `affected_files` gives paths only;
  it does not replace the evidence an agent needs to repair each file.
- For each affected file, name the exact import statement, key, symbol,
  annotation, expression, assertion, command, or document construct involved.
  Use these stable identifiers instead of numeric line numbers.
- If a merged problem would make the per-file explanation vague or excessively
  compressed, split it into smaller problems by violated contract or repair
  strategy.

## DYNAMIC VALIDATOR CAPABILITY CHECK:
- Derive what the named validation can detect from its command, `validates`,
  source, evidence, CI output, and configuration; do not rely only on its name.
- Ask whether reverting this specific change would make that validation fail.
  If not, it does not belong to this validation's root cause.
- Separate the substantive repair from incidental cleanup. Formatting caused by
  editing code, an import added to support a new symbol, or a generated-file
  refresh is not automatically an independent formatter, lint, or build issue.
- Apply this reasoning to any setup, dependency, schema, compiler, build,
  generated-code, lint, format, type, API, test, documentation, security, or
  workflow validation.
- Do not invent tool rules, error codes, constraints, or failure signals that
  are absent from the supplied evidence.

4. Keep setup/install enablement separate.
- Examples: invalid pyproject metadata, missing dependency, wrong extras, incompatible tool version, broken pip/poetry install config, workflow
setup command mismatch.
- If setup changes only enable a later formatter, linter, type checker, or test command, report the setup/install issue separately from later
validation violations.

5. Handle cascading fixes.
- Cascading means one change caused or required another related change.
- If all affected files share the same CI validation, failure pattern, concrete
  problem, root cause, and compatible repair strategy, they may be one problem.
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
- **BE EXTREMELY SPECIFIC** - include exact names, versions, values, commands:
  * Package changes: "package_name: version_before → version_after" with FULL version constraints
  * Config changes: "key_name: old_value → new_value" with exact values
  * Symbol changes: exact function/class/variable names before and after
  * Command changes: full command strings, not summaries
  * Type changes: exact type annotations before and after
  * Import changes: full import paths and what changed
- Do not mention line numbers.
- **NEVER use vague phrasing** like "fixed issues", "updated files", "addressed problems", "made changes"
  * BAD: "Updated dependencies", GOOD: "Updated numpy from >=1.24.0 to >=2.0.0,<2.5.0"
  * BAD: "Fixed type errors", GOOD: "Added cast(str, value) for type narrowing in 3 files"
  * BAD: "Improved formatting", GOOD: "Applied black formatting: added trailing commas, normalized quotes"

## REPAIR EVIDENCE SPECIFICITY (CRITICAL FOR EVERY FAILURE FAMILY):
- Never use only category labels such as "type issues", "API incompatibility",
  "test fixes", "configuration problem", "formatting changes", "build update",
  or "code modernization". A repair agent must be able to locate and reproduce
  the fix from the description.
- For EACH atomic problem, place all repair evidence directly in
  problem/root_cause/how_fixed (there is no separate change-details field):
  1. the exact affected subject: package/key/command/target/rule/symbol/function/
     parameter/expression/test assertion/document construct/artifact;
  2. the exact BEFORE state or behavior that violated a requirement;
  3. the exact expected contract or observable failure signal supported by the
     CI, configuration, dependency context, test, or before/after evidence; and
  4. the exact AFTER repair operation and important resulting value or behavior.
- Adapt the evidence to the failure family:
  * dependency/config/setup: exact package constraints, keys, values, commands,
    missing/invalid references, and resolver/schema expectation;
  * type/API/symbol/compiler: exact symbol or expression, old and required
    types/signatures/shapes, and the annotation/guard/cast/import/call change;
  * test/runtime: exact scenario, input or fixture, expected versus previous
    output/state/exception, and behavior change;
  * lint/format/docs: exact construct and transformation; include a rule/code
    only when supplied by evidence;
  * build/workflow/generated artifacts: exact target, command, path, option,
    source-to-generated relationship, or environment assumption that changed.
- Treat supporting edits as supporting details rather than root causes: for
  example, an added import supports use of its symbol, and line wrapping may be
  incidental to a semantic code repair.
- If files violate different contracts, exhibit different failure signals, or
  require materially different repair strategies, split them unless one precise
  root-cause statement explains every file without category-level wording.
- When an exact diagnostic is unavailable because CI stopped earlier, describe
  only the mismatch demonstrated by BEFORE/AFTER and other supplied evidence;
  do not fabricate a diagnostic, rule code, dependency constraint, or intent.
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

## DEPENDENCY VERSION SPECIFICITY (CRITICAL):
- **NEVER generalize dependency version changes** - preserve EXACT versions from before/after
- **BAD**: "Updated datacommons packages"
- **GOOD**: "Updated datacommons>=1.4.3,<2 and datacommons_pandas>=0.0.3,<0.0.4 to datacommons-client[pandas]>=2,<3"
- **BAD**: "Downgraded fish-audio-sdk for compatibility"
- **GOOD**: "Downgraded fish-audio-sdk from >=2024.12.5,<2025 to >=1.0.0,<2"
- In root_cause/how_fixed: Include EXACT package names and version constraints
- Format: "package_name: old_version → new_version"
- For multiple dependencies: List ALL with exact versions
- Inspect EVERY dependency declaration in both before and after, including
  multiple dependency changes within the same manifest file.
- Classify each dependency operation accurately:
  * added: package absent before → exact package and constraint after
  * removed: exact package and constraint before → package absent after
  * constraint changed: same package, exact old constraint → exact new constraint
  * replaced: list ALL removed package specifications → list ALL added package specifications
- When the same dependency operation appears in multiple groups/extras, name
  every affected group/extra and do not treat the repetitions as new packages.
- Do not let a separate config change in the same file (for example,
  tool.uv.default-groups) hide or absorb package additions, removals,
  replacements, or constraint changes.
- Only claim that a package change caused a CI failure when CI context,
  dependency context, or a related source adaptation supports that causal link.
  Otherwise describe the exact manifest change without inventing a cause.
- EVERY package operation in CHANGES must be retained in atomic_problems. A
  package operation may not be omitted merely because another configuration
  change in the same manifest is the CI-visible failure.
- If a package operation has a different failure pattern, problem, or root cause
  from the other changes in that manifest, create a separate atomic problem for
  it even though affected_files contains the same manifest path.
- If the diff proves the exact package operation but the supplied evidence does
  not establish the underlying resolver/API constraint, keep it as a separate
  manifest-change problem and state that the exact causal constraint is not
  supplied. Do not merge it into an unrelated config problem and do not invent
  incompatibility, deprecation, availability, or replacement claims.
- This applies to ALL dependency files: pyproject.toml, requirements.txt, package.json, Cargo.toml, etc.

## DEPENDENCY AND CONFIG CONSTRAINT ANALYSIS:
- For every changed package or config entry, analyze the evidence available for:
  * exact declaration scope: dependency group, extra/feature, runtime/dev/build/
    optional scope, workspace/member, source/index, or lockfile section;
  * exact constraint dimensions: lower/upper bounds, pins, exclusions, extras/
    features, environment markers, source URLs, checksums, and resolver settings;
  * relevant environment dimensions: language/runtime version, operating system,
    architecture, build backend/toolchain, installer/resolver version, and the CI
    command that consumes the manifest or configuration;
  * dependency relationships: direct versus transitive dependency, replacement/
    rename, shared constraints, and source/API adaptations caused by the change.
- Report only dimensions supported by CHANGES, CI context, dependency context, or
  related adaptations. Mark an unsupported causal constraint as not supplied;
  never guess it.
- A removal is a substantive package/config operation and must remain in the
  output. Do not retain only additions or AFTER values.
- For multiline or structured config, preserve the complete key path and value,
  including list/table members. Analyze each independent key separately.
- Distinguish observed facts from inferred impact: exact before/after values are
  facts; resolver, environment, API, build, or runtime consequences require
  supporting evidence.

## DECISION PRINCIPLES (Dynamic - apply to YOUR specific changes):

PRINCIPLE 1: Failure pattern + problem + root cause determine grouping
- Analyze WHAT fails, what problem it represents, and WHY each file changed
- Group only files sharing the SAME failure pattern, problem, and root cause
- Split files when any of those materially differ

PRINCIPLE 2: Variants are acceptable in ONE problem
- Variants may share one problem only when they are manifestations of the same
  failure pattern and root cause and use a compatible repair strategy
- Different annotations, guards, APIs, assertions, rules, or behavior contracts
  are not variants merely because the validator is the same
- Describe every retained variant in problem/root_cause/how_fixed and list all
  corresponding paths in affected_files

PRINCIPLE 3: Validation boundary enforcement
- Files in DIFFERENT validations = ALWAYS separate problems (already enforced)
- Files in SAME validation but different failure patterns, problems, root
  causes, or repair strategies = separate problems

PRINCIPLE 4: Test your grouping
- Ask: "Can one precise failure-pattern, problem, root-cause, and repair
  explanation cover every affected file?"
- If YES -> merge; if any part requires category-level or vague wording -> split

FIELD GUIDANCE (Dynamic - based on YOUR analysis):

- **root_cause**: The underlying WHY, derived from the issue constraints,
  dependency evidence, CI context, and before/after changes, that applies to ALL
  affected files
  * Must be specific to THIS issue's context with EXACT details
  * Include EXACT package names, versions, config values, symbols, commands
  * For dependency changes: List EXACT version constraints (before → after)
  * For config changes: Include EXACT key names and values (old → new)
  * For code changes: Include EXACT symbols, types, imports changed
  * Explain what changed/broke that required these files to adapt
  * For variants: state the common root cause and the exact failure manifestation
    in every affected file
  * For cascading: explain the dependency change that triggered adaptation with exact details

- **problem**: What failed and scope
  * Describe the failure based on root_cause
  * Mention file count, directory scope if multiple files
  * A summary such as "X files with variant changes" is allowed only when
    root_cause/how_fixed enumerate the exact file-specific manifestations
  * For cascading: explain the triggering relationship

- **how_fixed**: The exact actionable fixes needed to address root_cause
  * Describe every required fix across affected files with EXACT specifics so a
    repair agent can reproduce it without another details field
  * For dependency changes: "Updated package_name from old_version to new_version"
  * For config changes: "Changed key_name from old_value to new_value"
  * For code changes: "Added/removed/changed exact_symbol_name"
  * For variants: name the exact repair applied in every affected file; do not
    replace the details with a generic list of repair categories
  * For cascading: explain adaptation to new format/behavior with exact changes
  * NEVER use generic terms like "updated dependencies" - list exact package names and versions

- **issue_type**: Specific semantic issue type (NOT just validation name!)
  * Based on root_cause analysis, not validation command
  * Be specific to THIS issue's actual problem
  * Avoid generic terms - make it semantic and meaningful

- **affected_files**: EVERY file from CHANGES with this root cause
  * List ALL files that share this root cause
  * Do NOT omit files even if many
  * Do NOT include files from other validation groups
  * Do NOT include files with different root causes

## CHANGE COVERAGE CHECK (MANDATORY BEFORE OUTPUT):
- Re-read every individual before/after entry in CHANGES after drafting the
  atomic problems.
- Confirm that each dependency addition, removal, replacement, and constraint
  change is explicitly described in at least one problem's root_cause or
  how_fixed with exact package names and constraints.
- Count the distinct package operations in CHANGES and compare them with the
  operations described in atomic_problems. The counts and exact before/after
  specifications must match before returning JSON.
- Confirm that each independent config change is also explicitly described.
- File coverage is not change coverage: listing pyproject.toml in
  affected_files is insufficient when one or more package changes inside it
  are missing from the explanation.
- If changes within one file have different root causes, create separate
  problems even though affected_files may contain the same path.

OUTPUT FORMAT:
{{
  "atomic_problems": [
    {{
      "problem_id": 1,
      "validation_order": {validation_order},
      "validation_cmd": {validation_cmd_json},
      "failure_type": {failure_type_json},
      "issue_type": "specific_error_code_or_type",
      "problem": "What broke, including the violated constraint and exact affected subjects",
      "root_cause": "Why it failed based on issue constraints, dependency evidence, CI context, and before/after evidence",
      "how_fixed": "Exact actionable fixes needed, including old and new values or behavior for every affected subject",
      "why_fix_works": "Why the fix solves it",
      "affected_files": ["file1.py", "file2.py"],
      "problem_type": "primary or hidden - see rules below",
      "is_cascading": {is_cascading_json},
      "dependency_type": {dependency_type_json},
      "cascade_explanation": {cascade_explanation_json}
    }}
  ]
}}

OUTPUT REQUIREMENTS:
- Return only a JSON object with the "atomic_problems" array.
- If no problems can be extracted, return {{"atomic_problems": []}}.
- problem_id must be an integer starting at 1 and incrementing by 1.
- validation_order must be the integer validation_order from VALIDATION CONTEXT.
- validation_cmd must exactly match validation_cmd from VALIDATION CONTEXT.
- failure_type must match failure_type from VALIDATION CONTEXT unless the value is empty.
- problem, root_cause, and how_fixed together must cover every semantic change.
  For config/dependency files, how_fixed MUST name every package or key added,
  removed, replaced, or constraint-changed with exact before/after values and
  declaration scopes. Separate unrelated operations into different problems
  when they do not share one root cause.
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
