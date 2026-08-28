"""
Commit analysis prompt template.

This prompt analyzes commits to extract problems, root causes, and repair plans.
"""

import json
from typing import Dict, List


def build_commit_analysis_prompt(
    commit_info: str,
    commit_diff: str,
    commit_sha: str,
    sha_success: str,
    sha_fail: str,
    commit_number: int,
    total_commits: int,
    chunk_number: int,
    total_chunks: int,
    files_in_chunk: List[str],
    selected_groups: List[Dict],
    ci_failure_info: Dict,
    ci_metadata: Dict,
    relevant_validations: List[Dict],
) -> str:
    """Build prompt for analyzing a commit to extract CI problem events."""
    chunk_info = ""
    if total_chunks > 1:
        chunk_info = f"\n- Chunk {chunk_number}/{total_chunks} of this commit (files: {', '.join(files_in_chunk[:5])}{'...' if len(files_in_chunk) > 5 else ''})"

    ci_failure_info_str = json.dumps(ci_failure_info, indent=2)
    ci_metadata_str = json.dumps(ci_metadata, indent=2)
    selected_groups_str = json.dumps(selected_groups, indent=2)
    validations_str = json.dumps(relevant_validations, indent=2)
    commit_sha_json = json.dumps(commit_sha)

    prompt = f"""You are analyzing commits SEQUENTIALLY to understand how CI behavior changed between a known passing commit and a known failing commit.

Your job is NOT just to find "problems" or "failures".
Your job is to find EVERY change that affects CI validation, whether it:
- Fixes a CI failure (fixed=true)
- Introduces a CI failure (introduced=true)
- Fixed any CI validation
- Or both (rare)
- Or any changes that has affect on CI validation (formatting, linting, type checking, tests, config, dependencies, generated artifacts, etc.)

Report formatting fixes, test updates, import changes, type annotations - ANYTHING validated by CI.

Your output must describe ALL CI-RELEVANT CHANGES in this commit - any change that affects CI validation behavior, whether it fixes a problem, introduces a problem, or both.

A CI-relevant change is any modification to:
- Code validated by CI (formatting, linting, type checking, tests)
- Test expectations, assertions, or test data
- Configuration, dependencies, or tool settings used by CI
- Documentation or generated artifacts checked by CI
- Any code path executed during CI validation

For EACH CI-relevant change, determine:
- Does it FIX a CI problem? (fixed=true)
- Does it INTRODUCE a CI problem? (introduced=true)
- Or both?

DO NOT only report problems - report ALL changes relevant to CI validation.

CONTEXT:
- Repository: Known to PASS CI at {sha_success} and FAIL CI at {sha_fail}
- Total commits between success and failure: {total_commits}
- Currently analyzing: Commit {commit_number}/{total_commits}{chunk_info}

COMMIT BEING ANALYZED:
{commit_info}

RELEVANT CHANGES IN THIS {"CHUNK" if total_chunks > 1 else "COMMIT"}:
{commit_diff}

SELECTED CI-RELEVANT FILE GROUPS FOR THIS COMMIT:
{selected_groups_str}

Each selected group is a validation/failure cluster and may include:
- validation_order: order of this validation in the CI sequence
- validation_cmd: CI command shared by this cluster
- failure_type: failure family shared by this cluster
- groups: subgroups selected before clustering, each with files, issue_type, and reason
- diff: combined diff hunks for all files in this validation/failure cluster
- dependency_context: dependency/call/config evidence for this cluster when available

Use this grouped context to analyze related files together. The top-level cluster
defines the likely CI validation/failure family. The nested groups preserve issue
hints and reasons from file selection. Treat these as routing hints, not proof:
verify every problem, fix, and introduced risk from the actual diff,
dependency_context, and validation rules.

STRUCTURED CI FAILURE:
{ci_failure_info_str}

CI METADATA FOR THIS COMMIT (optional runtime evidence; may be missing, partial, or unavailable):
{ci_metadata_str}

Use these CI metadata fields when present:
- workflow_run_exists: whether any workflow run metadata exists for this commit
- workflow_names: workflow names found for this commit
- jobs_executed: jobs that executed and their conclusions
- failed_jobs: failed jobs for this commit
- step_names_executed: steps that executed and their conclusions
- failed_steps: failed steps for this commit
- current_jobs_fixed: jobs that passed in this commit
- current_failed_jobs: jobs that failed in this commit

**DATASET FAILED JOBS (ground truth from benchmark):**
When present, these fields provide AUTHORITATIVE EVIDENCE of which jobs/steps failed at sha_fail:
- dataset_failed_jobs: Array of job objects, each with job_name and steps list - specific jobs and steps that failed
- dataset_error_types: Array of error categories (e.g., "Code Formatting", "Test Failure", "Type Check")
- dataset_total_failed_jobs: Total number of failed jobs
- dataset_total_failed_steps: Total number of failed steps

**How to use dataset_failed_jobs:**
1. PRIORITIZE analyzing changes related to the failed jobs and error types
2. If error_type includes "Code Formatting" → look for formatting changes (whitespace, imports, quotes)
3. If error_type includes "Test Failure" → look for test changes, assertions, test data
4. Match changed files against the failed job names and steps
5. Use this as GROUND TRUTH - these jobs/steps DID fail at sha_fail

Example: If dataset_failed_jobs contains job_name "pre-commit" with steps "Run pre-commit"
and dataset_error_types includes "Code Formatting", then changes to formatting (blank lines, quotes, imports)
are HIGHLY RELEVANT and should be reported as fixing the pre-commit formatting failure.

If CI METADATA is missing or empty, do not stop and do not assume the commit is irrelevant.
Analyze the actual diff against STRUCTURED CI FAILURE and RELEVANT CI VALIDATION STEPS.
CI metadata is supporting evidence only; the diff and validation rules are the source of truth for whether this commit affects CI.

RELEVANT CI VALIDATION STEPS:
{validations_str}

TASK - ANALYZE ALL CI-RELEVANT CHANGES:

For EVERY changed hunk in this commit, ask:
1. Does this change affect ANY CI validation command listed below?
2. If YES → Create a problem object describing the change

For each CI-relevant change, determine:
- **introduced=true** if the change creates/adds a new issue that causes CI to fail
- **fixed=true** if the change resolves/fixes an issue that was causing CI to fail
- Both true if the change replaces one problem with another
- Both false only in rare cases (refactoring that preserves CI behavior)

CRITICAL: Do not limit output to "failures" or "bugs". Include:
- ✅ Formatting/style fixes (blank lines, quotes, imports)
- ✅ Test updates (assertions, test data, expected values)
- ✅ Type annotation additions/fixes
- ✅ Documentation formatting fixes
- ✅ Dependency/configuration updates
- ✅ Any code change validated by a listed CI command

If a change affects CI validation → Report it as a problem object.

1. Identify the CURRENT CI failure at sha_fail.
   CI stops at the first failing validation step. Use STRUCTURED CI FAILURE only as primary first-failure evidence: failed job, failed step, exact error, failed tool, validation command, and implicated file/line when available.
   Do not treat "current failure" as the only reportable event. The output
   "problems" array represents CI problem events from this commit: unresolved
   failures, introduced validation risks, and fixes for earlier/hidden/later
   validation issues.

2. Read the full CI validation sequence.
   RELEVANT CI VALIDATION STEPS are all checks that can apply to these files, including steps CI may not have reached because it stopped at the first failure. Analyze the diff against both the known CI failure and every listed CI verification step. A commit can fix the logged failure, fix a hidden/later validation, introduce a different validation problem, or be relevant only because another CI verification would check it.
   If CI metadata is absent, still use these validation steps to decide which changed hunks matter.

3. Analyze this commit's diff only and group changed hunks by CI problem family.
   Use SELECTED CI-RELEVANT FILE GROUPS, diff, and dependency_context as supporting grouping context, but verify every claim from the actual diff and validation rules.
   For each selected group, decide whether the changed hunks:
   - introduce a validation problem,
   - fix or adapt a validation that would otherwise fail,
   - update a validation contract, expected output, fixture, snapshot, or helper,
   - change config/dependencies/tooling/install behavior for a validation,
   - participate as support/config/test/generated-code for such a problem,
   - or are CI-relevant but do not provide enough evidence for a problem event.

   Report every case that can change CI pass/fail behavior, validated content,
   configuration, dependencies, generated output, runtime behavior, assertions,
   or tool execution as a problem object. Omit the last case only when
   you explain to yourself that the hunks are checked by CI but do not change an
   assertion, rule, behavior, configuration, dependency, generated output, or
   validated content in a way that can pass/fail a listed validation.

   If the changed hunks are support/config/test/generated-code for another
   selected file, include them in the same problem object when they share the
   same validation command, root cause, and repair strategy. Split them when
   they represent a separate validation contract or separate root cause.

GROUPING RULE:
Create one problem object per distinct CI problem family, not per file.

A CI problem family means the changed hunks share the same:
- validation_cmd
- failure_type
- issue_type
- root_cause
- repair strategy
- introduced/fixed behavior

If multiple files face the same kind of CI problem and are introduced or fixed in the same way, put all those files together in the same "files" list.

Split into separate problem objects only when:
- the validation command is different,
- the failure type is different,
- the issue subtype is different,
- the root cause is different,
- the repair strategy is different,
- one change fixes an issue while another introduces a different issue,
- the same file participates in two different validation failures,
- or config/dependency/tooling changes are separate from source/test fixes.

4. Treat sha_success as the known passing baseline and sha_fail as the failing target.
   The final successful repair trajectory should explain all commits needed to move from the failing state back to passing CI. For this single commit, report only CI-relevant problem events supported by the diff and validation rules.
   A problem event can be fixed=true even when current_failed_jobs is empty or
   points to a different validator, as long as the diff shows a concrete
   validation repair and a listed CI command would verify it.

5. Classify relevance from the changed hunks and validation commands.
   Source, test, dependency, project configuration, tool configuration, generated-code setup, import paths, type annotations, formatting-sensitive files, and executable scripts can all be CI-relevant when a listed validation would check them.
   Documentation-only or metadata-only changes should be ignored unless a listed documentation or packaging validation checks them.
   Test changes are CI-relevant when they alter assertions, expected values,
   parsing of validated artifacts, fixtures, snapshots, or helper logic executed
   by a listed test command. For such changes, describe the validation contract
   being updated from the diff itself instead of relying on a hardcoded file or
   domain-specific rule.

DIRECT AND INDIRECT IMPACT RULES:
- Direct impact: the changed hunk itself violates, satisfies, configures, or is
  executed by a listed CI command.
- Indirect impact: the changed hunk affects a file, function, fixture, generated
  artifact, dependency, template, script, config, or package metadata that a
  listed CI command reaches through imports, calls, installation, generation,
  test setup, docs build, packaging, lint/type-check discovery, or tool config.
- If changed source and changed tests describe the same behavior under the same
  validation command, group them together and explain both the code behavior and
  the validation contract.
- If changed config/dependency/tooling causes the same source/test validation
  behavior, include it with the source/test files only when the root cause and
  repair strategy are the same. Otherwise create a separate problem object.
- Do not invent a problem from a file name alone. The changed hunk must show a
  concrete CI relationship.
- Do not omit a selected file whose hunk changes a CI-relevant assertion, rule,
  behavior, dependency, configuration, generated output, or validated content.

INTRODUCED VS FIXED:
- introduced=true, fixed=false when this commit adds or changes something that
  can make a listed CI validation fail or creates a validation challenge that
  later commits must repair.
- introduced=false, fixed=true when this commit repairs or adapts code, tests,
  config, dependencies, generated files, or expectations so a listed CI
  validation can pass.
- introduced=true, fixed=true only when the same problem object genuinely
  describes a replacement that both removes one validation issue and introduces
  another in the same validation family. Prefer separate problem objects when
  the issue type, root cause, or repair strategy differs.
- introduced=false, fixed=false should be rare. Use it only when the diff is
  CI-relevant support/context for a validation problem but does not itself
  introduce or fix the problem.

CRITICAL - PACKAGE/DEPENDENCY VERSION SPECIFICITY:

For dependency/config file changes (pyproject.toml, requirements.txt, etc.), root_cause and fix_strategy MUST include:
1. Exact package name + old version → new version (EXACT constraints)
2. Config file changed
3. Technical reason WHY old version failed and WHY new version fixes
4. List EVERY package operation (added, removed, upgraded, downgraded)

Example: "click 8.2.0 broke TyperOption causing TypeError in py/flwr/cli/app.py. Changed click from >=8.0.0 to <8.2.0 in framework/pyproject.toml to maintain API compatibility."

For each problem object, provide:

1. "files": **REQUIRED** - List of EXACT file paths from the diff that are affected by this problem.
   NEVER leave this empty. If you detect a problem, you MUST list the file(s).
   Example: ["path/to/file.py"] or ["file1.py", "file2.py"]
2. "failure_type": Broad validation family, such as "format", "lint", "type_check", "test", "build", "install", "import", "docs", or "unknown".
3. "issue_type": Specific issue family, such as "missing_return_annotation", "import_order", "dependency_version", or "assertion_update".
4. "problem": The CI problem being described, with step/job and line numbers when available.
5. "root_cause": The underlying technical cause. **FOR DEPENDENCY CHANGES: Include exact package names, old version, new version, WHY old version failed, technical incompatibility details.**
6. "changes_made": What this commit changed in the code. **FOR DEPENDENCY CHANGES: List EVERY package operation with exact versions.**
7. "introduced": true if this commit introduced the problem or validation challenge; otherwise false.
8. "fixed": true if this commit fixed this problem; otherwise false.
9. "fix_strategy": When fixed is true, describe the concrete repair. **FOR DEPENDENCY CHANGES: Explain exact version change (old → new), which config file, WHY new version fixes the issue technically.**
10. "why_this_fix_works": When fixed is true, explain why the repair satisfies the validation. **FOR DEPENDENCY CHANGES: Explain technical compatibility restored, APIs/symbols now available, behavior aligned.**
11. "current_failed_jobs": Failed job/step records from CI METADATA for this commit, including validation_cmd when available. Use [] when CI metadata is missing or has no failed jobs.
12. "current_fixed_jobs": Jobs from CI METADATA that passed in this commit. Use [] when CI metadata is missing or has no passed jobs.
13. "validation_cmd": The exact CI command that verifies the problem or repair.

OUTPUT JSON FORMAT (valid JSON only, no markdown):
{{
  "commit_sha": {commit_sha_json},
  "problems": [
    {{
      "files": ["path/to/file.py"],
      "failure_type": "format",  ← Use "format" for formatting tools (black/yapf/prettier/ruff)
      "issue_type": "code_formatting",
      "problem": "File had formatting issues detected by yapf",
      "root_cause": "Code did not match yapf's formatting requirements",
      "changes_made": "Added blank line after docstring to match yapf style",
      "introduced": false,
      "fixed": true,  ← Set to true when fixing a CI failure
      "fix_strategy": "Modified whitespace/formatting to match tool requirements",
      "why_this_fix_works": "The formatting now matches what yapf expects",
      "current_failed_jobs": [],
      "current_fixed_jobs": [],
      "validation_cmd": "yapf --diff"
    }}
  ]
}}

EXAMPLES OF CI-RELEVANT CHANGES TO ALWAYS REPORT:

**Fixes (fixed=true, introduced=false):**
1. Added blank line → Fixed yapf/black formatting
2. Changed 'png' to "png" → Fixed quote style
3. Reordered imports → Fixed import-order linting
4. Added type annotation → Fixed mypy type check
5. Updated test assertion → Fixed failing test
6. Fixed docstring format → Fixed sphinx build

**Introductions (fixed=false, introduced=true):**
1. Removed blank line → Broke formatting
2. Added untyped parameter → Broke type checking
3. Changed test without updating assertion → Broke test

**Both (fixed=true, introduced=true):**
1. Fixed one linter but broke another
2. Updated code and test, but test is now wrong

IF the diff changes ANYTHING that a listed CI command validates, report it.

IMPORTANT RULES:
1. **ALWAYS populate the "files" array** - If you detect a problem, you MUST list which file(s) it affects. Use exact paths from the diff.
2. **ALWAYS detect formatting/linting fixes** - If the CI failure mentions formatting (black, yapf, ruff, prettier, etc.) and the diff shows ANY changes (blank lines, quotes, spacing, indentation), this is a FIX. Set fixed=true.
3. **ALWAYS detect test fixes** - If the CI failure mentions test failures and the diff changes test code or test data, this is a FIX. Set fixed=true.
4. STRUCTURED CI FAILURE shows only the known first failing step at sha_fail. Do not assume later validation steps passed or are irrelevant.
5. Check every changed hunk against the structured CI failure and every relevant validation step, not only the logged failure.
6. Emit a problem object for every changed hunk with direct or indirect impact on a listed CI validation, unless it cannot affect pass/fail behavior or validated content.
4. Set fixed=true when the diff directly or indirectly explains why a listed validation would now pass. Put the repair in "fix_strategy" and the validation reasoning in "why_this_fix_works".
5. Set introduced=true when the diff directly or indirectly violates a validation rule, changes behavior that can fail a listed validation, or explains the current CI failure.
6. **DEFAULT TO REPORTING**: If a change touches code/config/tests that ANY listed validation checks, report it. When uncertain, report it.
7. If the commit is completely unrelated to CI (e.g., only user-facing content with no CI validation), return "problems": [].
8. Same failure family must be one problem object, even if many files changed.
9. Different validators, root causes, or repair strategies must be separate problem objects.
10. Use plural field names exactly: "current_failed_jobs" and "current_fixed_jobs".
11. Do not invent current_failed_jobs/current_fixed_jobs. They must come from CI METADATA for this commit. If metadata is missing, use [].
12. Do not require CI metadata to identify a change. Use changed hunks plus validation steps.
13. Do not omit selected test/config/support files merely because another
    selected group better matches the current failed job. If their hunks change
    how a listed validation passes or fails, emit a separate problem object for
    that validation family.
14. Return ONLY valid JSON, no markdown, no comments, and no placeholder entries.
15. **LIBERAL REPORTING**: Report all CI-relevant changes. Better to over-report than miss changes that affect validation."""

    return prompt
