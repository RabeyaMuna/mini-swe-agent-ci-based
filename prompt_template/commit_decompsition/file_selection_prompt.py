"""Prompt template for selecting CI-relevant changed files from a commit diff."""

import json
from typing import Dict, List


def build_file_selection_prompt(
    commit_metadata: Dict,
    changed_files: List[str],
    commit_diff: str,
    structured_ci_failure: Dict,
    relevant_validations: List[Dict],
    relationship_context: List[Dict] = None,
) -> str:
    """
    Build the file selection prompt.

    The prompt keeps a stable JSON output contract while making the selection
    criteria broad enough to include direct and indirect CI impact.
    """
    relationship_context_str = json.dumps(relationship_context or [], indent=2)
    changed_files_str = json.dumps(changed_files or [], indent=2)

    prompt = f"""You are selecting changed files for CI failure analysis.

TASK: Select ALL files whose changes are relevant to the CI validation that failed.

SELECTION RULES:
1. If a file is mentioned in the CI failure → SELECT IT
2. If a file's changes could affect the failing validation command  directly or indirectly→ SELECT IT
3. If a file is imported/used by a file that affects CI → SELECT IT
4. When in doubt → SELECT IT (be inclusive, not selective)
5. Any configuration related changes that could affect the CI validation → SELECT IT

DEFAULT: If any file looks remotely related to CI testing, linting, formatting,
building, or documentation → SELECT IT.

COMMIT METADATA:
{json.dumps(commit_metadata, indent=2)}

CHANGED FILES IN THIS DIFF CHUNK:
{changed_files_str}

COMMIT DIFF:
{commit_diff}

STRUCTURED CI FAILURE:
{json.dumps(structured_ci_failure, indent=2)}

RELEVANT CI VALIDATION STEPS:
{json.dumps(relevant_validations, indent=2)}

DEPENDENCY AND CALL GRAPH CONTEXT FOR CHANGED FILES:
{relationship_context_str}

This context may include config/dependency relationships, imports, changed functions, calls, and caller/callee edges extracted from repository source at this commit when available. Use it to notice indirectly related changed files that should be selected together. The final decision must still be based on the actual changed hunks plus their relationship to the listed CI steps.

TASK:
Analyze every changed file in COMMIT DIFF and select all files whose changed hunks can affect at least one CI setup, install, validation, tool configuration, dependency, generated artifact, test, source path, package, or documentation path used by the listed validation steps.
Use CHANGED FILES IN THIS DIFF CHUNK as the complete candidate list for this prompt. Select only files from that list, and verify each selected file against its actual hunks in COMMIT DIFF.

GENERIC SCOPE-DERIVATION PROCEDURE:
For each listed CI setup/install/validation step:
1. Extract the effective command text from all available fields, including
   validation_cmd, installation_cmd, validates, source, job, and step names.
2. Identify path scopes from command arguments and step descriptions. Treat
   explicit directories, file paths, glob-like fragments, package/workspace
   roots, and relative path fragments as validation scopes.
3. Normalize path comparisons conceptually: a command scope can match a changed
   file when one path is a suffix, prefix, workspace-relative equivalent, or
   shared subpath of the other. Do not require exact string equality when the
   same repository subtree is clearly referenced.
4. Infer file-kind scope from the command or step description only when the
   validation text itself implies it. If a command validates a document tree,
   package tree, source tree, generated tree, config file, or test tree, select
   changed files under that tree whose hunks can alter validated content.
5. Infer configuration/dependency scope from setup/install/build/tool commands:
   select changed manifest, lock, configuration, environment, script, or
   tool-setting files only when the command could read them or when relationship
   context links them to a selected validation.
6. Infer indirect scope from relationship context: select changed files that are
   imported, called, generated from, configured by, installed from, or used as
   inputs/fixtures/helpers for files already selected for a validation scope.

PATTERN TO DETECT:
When many changed files share the same directory, suffix/file kind, generated
output pattern, template relationship, or repeated hunk shape, and a listed CI
step validates that same path scope or file kind, select every changed file in
that repeated pattern. Do not select only one representative file. Group those
files together when they share the same validation command and reason.

STRUCTURED CI FAILURE is only the first known failing step at sha_fail. It is useful evidence, but it is NOT the only relevance criterion.

Do NOT stop after selecting the file named in STRUCTURED CI FAILURE.
Do NOT require a current failing log line to select a changed file.
Do NOT assume only the current failing validator matters.

CI-relevant changes can include:
- a direct fix or introduction for the structured failure,
- a fix for an earlier, hidden, later, or secondary validation issue,
- source behavior executed by tests or checked by static analysis,
- test or assertion changes for behavior that CI validates,
- configuration or dependency changes read during install, lint, format, type-check, test, build, package, docs, or generated-project validation,
- templates or generated-code inputs that CI later checks.

A changed file is CI-relevant if its visible diff can affect:
- the structured CI failure,
- any listed validation command,
- any listed setup or install command,
- any tool configuration used by those commands,
- any dependency used by those commands,
- any source, test, template, or generated-code path checked by those commands.

Select files when their changed hunks:
- directly fix, introduce, or modify the issue mentioned in STRUCTURED CI FAILURE
- are inside a path scope checked by any listed validation command
- change code executed by a listed test command, directly or through imports/calls
- change code imported by changed tests, fixtures, scripts, package entry points, docs builds, or generated-code validation
- change source or content checked by a listed type-check, lint, format, test, build, packaging, documentation, generated-code, or metadata validation
- change tests, assertions, imports, type annotations, function signatures, executable scripts, CLI behavior, or formatting-sensitive code
- change test expectations, parsing logic, fixtures, snapshots, or helper code
  used by tests to validate source, generated, documentation, or packaged files
- change project configuration, dependency versions, lock/config files, or tool settings used by any listed CI command
- change project manifests, setup/install config, lint config, type-check config, formatter config, packaging config, workflow config, docs config, or generated-project config that can be read by a listed CI step
- change templates, schemas, snapshots, or generated-file inputs that produce project files later checked by CI
- change dependency versions for tools used by listed validation commands, even if that dependency change is not the structured failure
- change package metadata, entry points, optional extras, build backends, scripts, or environment markers that install/build/package validation can read

Indirect relevance rule:
Select a changed file when relationship context or the diff shows that it is imported by, called by, configured by, generated from, or used as a fixture/helper/input for another changed file that is selected for CI relevance.
Examples:
- If a changed test is selected, also select changed source/helpers/fixtures it imports or exercises.
- If changed source behavior is selected, also select changed tests/assertions that validate that behavior.
- If a changed template/schema/generator is selected, also select changed generated outputs or tests that validate those outputs when they are in the diff.
- If a changed config/dependency file alters how a CI command runs, also select changed files whose validation behavior depends on that config/dependency change.

For config/manifest/dependency files:
Select them when a listed setup, install, lint, format, type-check, test, build, packaging, documentation, generated-code, or metadata command could read them.
Do not exclude them merely because they are not the file named in STRUCTURED CI FAILURE.

For repeated identical config edits:
Select every changed config file that is in scope for any listed validation or generated project validation.
Group them together in selected_groups when they share the same CI relevance.

GROUPING RULE:
Group selected files by CI problem family, not by directory alone.

Files belong in the same group when they share the same:
- validation_cmd or setup/install command
- failure_type
- issue_type
- root_cause
- reason for CI relevance

Split groups when:
- the validation command is different
- the failure type is different
- the issue subtype is different
- the root cause is different
- the repair strategy is different
- source/test changes are separate from config/dependency/tooling changes
- dependency/tool version changes are separate from source behavior changes

Do not select files when:
- their changed hunks are unrelated to every listed CI step
- they are outside every validation/setup/install scope and do not configure or feed any listed tool
- they are comments/docs-only and no listed docs, packaging, formatting, or metadata validation can read them
- the file name looks relevant but the actual hunks cannot affect CI behavior
- they are purely product/user-facing content changes and no listed test, docs, snapshot, packaging, lint, format, or metadata validation reads them

When uncertain:
- Select the file if the diff shows a realistic path for a listed CI command to read, execute, inspect, import, lint, type-check, format, build, package, or test it.
- Skip the file only when you can explain why no listed CI setup/install/validation command, config, dependency, or related selected changed file could be affected by its changed hunks.

IMPORTANT:
- Analyze every changed file in COMMIT DIFF.
- Match both file paths and actual changed hunks against all relevant validation steps.
- CI metadata is supporting evidence only; do not require CI metadata to select a file.
- Do not select only the structured-failure file when other changed files affect listed CI validations.
- Do not drop a changed test file just because the current failing job is a
  different validator. If a listed test command would execute the changed test
  or helper, select it and explain the validation relationship.
- If a listed CI command could read, execute, validate, install from, format, lint, type-check, test, build, package, or generate from a changed file, select it.
- Output only files that appear in COMMIT DIFF.
- Output only files listed in CHANGED FILES IN THIS DIFF CHUNK.
- Use exact file paths from COMMIT DIFF.
- Keep reasons brief but specific to the diff and validation relationship.

OUTPUT JSON ONLY:
{{
  "selected_groups": [
    {{
      "files": ["path/to/file.py", "path/to/other_file.py"],  ← **REQUIRED** - MUST include actual file paths
      "failure_type": "type_check",
      "issue_type": "tool_configuration",
      "validation_cmd": "exact command if known, otherwise empty string",
      "reason": "brief reason these files are CI-relevant as one group"
    }}
  ]
}}

**CRITICAL**: Every group MUST have the "files" array populated with actual file paths from the diff.
If you select a file, it MUST appear in the "files" array. Never return empty "files": []."""

    return prompt
