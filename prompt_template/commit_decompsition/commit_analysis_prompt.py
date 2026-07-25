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
    """
    Build prompt for analyzing a commit to extract CI problems.

    This is the original working prompt for commit analysis.
    """
    chunk_info = ""
    if total_chunks > 1:
        chunk_info = f"\n- Chunk {chunk_number}/{total_chunks} of this commit (files: {', '.join(files_in_chunk[:5])}{'...' if len(files_in_chunk) > 5 else ''})"

    ci_failure_info_str = json.dumps(ci_failure_info, indent=2)
    ci_metadata_str = json.dumps(ci_metadata, indent=2)
    selected_groups_str = json.dumps(selected_groups, indent=2)
    validations_str = json.dumps(relevant_validations, indent=2)
    commit_sha_json = json.dumps(commit_sha)

    prompt = f"""You are analyzing commits SEQUENTIALLY to understand how a CI failure evolved.

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
defines the CI validation/failure family. The nested groups preserve issue hints
and reasons from selection, but the actual problem/fix must be verified from the
diff, dependency_context, and validation rules.

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

If CI METADATA is missing or empty, do not stop and do not assume the commit is irrelevant.
Analyze the actual diff against STRUCTURED CI FAILURE and RELEVANT CI VALIDATION STEPS.
CI metadata is supporting evidence only; the diff and validation rules are the source of truth for whether this commit affects CI.

RELEVANT CI VALIDATION STEPS:
{validations_str}

TASK - COMMIT-BASED CI TRANSITION ANALYSIS:

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
   - participate as support/config/test code for such a problem,
   - or are CI-relevant but do not provide enough evidence for a problem event.

   Report the first three cases as problem objects. Omit the last case only when
   you explain to yourself that the hunks are checked by CI but do not change an
   assertion, rule, behavior, configuration, dependency, generated output, or
   validated content in a way that can pass/fail a listed validation.

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

For each problem object, provide:

1. "files": List of files affected.
2. "failure_type": Broad validation family, such as "format", "lint", "type_check", "test", "build", "install", "import", "docs", or "unknown".
3. "issue_type": Specific issue family, such as "missing_return_annotation", "import_order", "dependency_version", or "assertion_update".
4. "problem": The CI problem being described, with step/job and line numbers when available.
5. "root_cause": The underlying technical cause.
6. "changes_made": What this commit changed in the code.
7. "introduced": true if this commit introduced the problem or validation challenge; otherwise false.
8. "fixed": true if this commit fixed this problem; otherwise false.
9. "fix_strategy": When fixed is true, describe the concrete repair. When fixed is false, use "".
10. "why_this_fix_works": When fixed is true, explain why the repair satisfies the validation. When fixed is false, use "".
11. "current_failed_jobs": Failed job/step records from CI METADATA for this commit, including validation_cmd when available. Use [] when CI metadata is missing or has no failed jobs.
12. "current_fixed_jobs": Jobs from CI METADATA that passed in this commit. Use [] when CI metadata is missing or has no passed jobs.
13. "validation_cmd": The exact CI command that verifies the problem or repair.

OUTPUT JSON FORMAT (valid JSON only, no markdown):
{{
  "commit_sha": {commit_sha_json},
  "problems": [
    {{
      "files": ["path/to/file.py", "path/to/other_file.py"],
      "failure_type": "type_check",
      "issue_type": "missing_return_annotation",
      "problem": "Multiple files contain functions missing required return annotations checked by mypy.",
      "root_cause": "The type-checking configuration requires explicit return annotations for these functions.",
      "changes_made": "Added -> None return annotations across the listed files.",
      "fixed": true/false,
      "fix_strategy": "<If only the changes fix the problem; otherwise empty string>",
      "why_this_fix_works": "<If fixed is true, explain why the changes fix the problem; otherwise empty string>",
      "current_failed_jobs": [],
      "current_fixed_jobs": [],
      "validation_cmd": "mypy ..."
    }},
    //  Additional problems if any
  ]
}}

IMPORTANT RULES:
1. STRUCTURED CI FAILURE shows only the known first failing step at sha_fail. Do not assume later validation steps passed or are irrelevant.
2. Check every changed hunk against the structured CI failure and every relevant validation step, not only the logged failure.
3. Set fixed=true only when the diff directly explains why the validation would now pass. Put the repair in "fix_strategy" and the validation reasoning in "why_this_fix_works".
4. Set introduced=true only when the diff directly violates a validation rule or explains the current CI failure.
5. If the commit is unrelated to validated CI behavior, return the top-level commit fields with "problems": [].
6. Same failure family must be one problem object, even if many files changed.
7. Different validators, root causes, or repair strategies must be separate problem objects.
8. Use plural field names exactly: "current_failed_jobs" and "current_fixed_jobs".
9. Do not invent current_failed_jobs/current_fixed_jobs. They must come from CI METADATA for this commit. If metadata is missing, use [].
10. Do not require CI metadata to identify a problem. Use changed hunks plus validation steps when metadata is unavailable.
11. Do not omit selected test/config/support files merely because another
    selected group better matches the current failed job. If their hunks change
    how a listed validation passes or fails, emit a separate problem object for
    that validation family.
12. Return ONLY valid JSON, no markdown, no comments, and no placeholder entries.
13. If no changed hunk maps to the structured failure or any relevant validation step, return {{"commit_sha": {commit_sha_json}, "problems": []}}"""

    return prompt
