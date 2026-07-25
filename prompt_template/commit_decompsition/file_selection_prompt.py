"""
Original file selection prompt template.

This is the working prompt that selects CI-relevant files from commit diffs.
"""

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
    Build the original file selection prompt.

    This is the proven, working prompt. Don't over-complicate it.
    """
    relationship_context_str = json.dumps(relationship_context or [], indent=2)

    prompt = f"""Analyze the ACTUAL CHANGES in this commit's diff and select every changed file whose changed hunks can affect CI.

COMMIT METADATA:
{json.dumps(commit_metadata, indent=2)}

COMMIT DIFF:
{commit_diff}

STRUCTURED CI FAILURE:
{json.dumps(structured_ci_failure, indent=2)}

RELEVANT CI VALIDATION STEPS:
{json.dumps(relevant_validations, indent=2)}

DEPENDENCY AND CALL GRAPH CONTEXT FOR CHANGED FILES:
{relationship_context_str}

This context may include config/dependency relationships, imports, changed
functions, calls, and caller/callee edges extracted from repository source at
this commit when available. Use it to notice dependent changed files that should
be selected together, but verify selection against the actual diff and
validation steps.

TASK:
Select all changed files that are relevant to any CI setup, install, validation, lint, type-check, format, test, build, packaging, documentation, or tool-configuration step.

STRUCTURED CI FAILURE is only the known first failing step at sha_fail.
It is NOT the only relevance criterion.
Do not stop after selecting the file that directly matches STRUCTURED CI FAILURE.
Do not require a current failing log line to select a file. CI-relevant changes
include changes that fix an earlier/hidden/later validation issue, update tests
or assertions for changed validated behavior, or alter behavior that a listed
validation command would execute or inspect.

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
- change source code checked by mypy, pylint, ruff, black, isort, pytest, build, packaging, or docs validation
- change tests, assertions, imports, type annotations, function signatures, executable scripts, or formatting-sensitive code
- change test expectations, parsing logic, fixtures, snapshots, or helper code
  used by tests to validate source, generated, documentation, or packaged files
- change project configuration, dependency versions, lock/config files, or tool settings used by any listed CI command
- change pyproject.toml, setup config, lint config, type-check config, formatter config, packaging config, or docs config that can be read by a listed CI step
- change templates that generate project files later checked by CI
- change dependency versions for tools used by listed validation commands, even if that dependency change is not the structured failure

For TOML/config files:
Select them when a listed setup, install, lint, format, type-check, test, build, packaging, or docs command could read them.
Do not exclude them merely because they are not the file named in STRUCTURED CI FAILURE.

For repeated identical config edits:
Select every changed config file that is in scope for any listed validation or generated project validation.
Group them together in selected_groups when they share the same CI relevance.

GROUPING RULE:
Group selected files by CI problem family.

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

IMPORTANT:
- Analyze every changed file in COMMIT DIFF.
- Match both file paths and actual changed hunks against all relevant validation steps.
- CI metadata is supporting evidence only; do not require CI metadata to select a file.
- Do not select only the structured-failure file when other changed files affect listed CI validations.
- Do not drop a changed test file just because the current failing job is a
  different validator. If a listed test command would execute the changed test
  or helper, select it and explain the validation relationship.
- If a listed CI command could read, execute, validate, install from, format, lint, type-check, test, build, package, or generate from a changed file, select it.

OUTPUT JSON ONLY:
{{
  "selected_groups": [
    {{
      "files": ["path/to/file.py", "path/to/other_file.py"],
      "failure_type": "type_check",
      "issue_type": "tool_configuration",
      "validation_cmd": "exact command if known, otherwise empty string",
      "reason": "brief reason these files are CI-relevant as one group"
    }}
  ]
}}"""

    return prompt
