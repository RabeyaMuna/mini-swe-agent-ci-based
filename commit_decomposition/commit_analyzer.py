#!/usr/bin/env python3
"""
commit_analyzer.py - Analyze commits to extract problems, root causes, and repair plans
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import litellm
from dotenv import load_dotenv

load_dotenv()


class CommitAnalyzer:
    """Analyzes commits to extract problems and repair plans using LLM"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "MEMCI_LLM_MODEL", "openrouter/minimax/minimax-m2.5"
        )

    def analyze_commit_group(
        self,
        group: Dict,
        commit_diff: str,
        ci_logs: Any,
        relevant_validations: List[Dict],
    ) -> Dict:
        """
        Analyze a commit group to extract:
        1. Problem identification
        2. Root cause
        3. Fix strategy
        4. Repair plan

        Args:
            group: Commit group info with commits, files, etc.
            commit_diff: Git diff for this group
            ci_logs: CI failure logs
            relevant_validations: Validation commands relevant to changed files

        Returns:
            Decomposition with problems, root causes, fix strategies, repair plans
        """
        # Format commit info
        commit_info = self._format_commit_info(group.get("commits", []))

        # Extract CI failure info
        ci_failure_info = self._extract_ci_failure(ci_logs)

        # Format validations
        validations_str = json.dumps(relevant_validations, indent=2)
        sha_success = group.get("sha_success") or "unknown"
        sha_fail = group.get("sha_fail") or "unknown"
        commit_number = group.get("commit_number", 1)
        total_commits = group.get("total_commits", 1)
        chunk_number = group.get("chunk_number", 1)
        total_chunks = group.get("total_chunks", 1)
        files_in_chunk = group.get("files_in_chunk", [])
        changed_files = group.get("changed_files", [])
        validated_changed_files = group.get("validated_changed_files", [])
        ci_metadata = group.get("ci_metadata") or {}
        current_commit = (group.get("commits") or [{}])[0]
        commit_sha = current_commit.get("sha", "unknown")
        commit_message = current_commit.get("message", "")
        commit_sha_json = json.dumps(commit_sha)
        commit_message_json = json.dumps(commit_message)
        commit_metadata = {
            "commit_sha": commit_sha,
            "commit_message": commit_message,
            "author": current_commit.get("author", ""),
            "date": current_commit.get("date", ""),
            "html_url": current_commit.get("html_url", ""),
            "commit_number": commit_number,
            "total_commits": total_commits,
            "chunk_number": chunk_number,
            "total_chunks": total_chunks,
            "changed_files": changed_files,
            "validated_changed_files": validated_changed_files,
            "sha_success": sha_success,
            "sha_fail": sha_fail,
        }

        chunk_info = ""
        if total_chunks > 1:
            chunk_info = f"\n- Chunk {chunk_number}/{total_chunks} of this commit (files: {', '.join(files_in_chunk[:5])}{'...' if len(files_in_chunk) > 5 else ''})"

        ci_metadata_str = json.dumps(ci_metadata, indent=2)

        # Build LLM prompt
        prompt = f"""You are analyzing commits SEQUENTIALLY to understand how a CI failure evolved.

CONTEXT:
- Repository: Known to PASS CI at {sha_success} and FAIL CI at {sha_fail}
- Total commits between success and failure: {total_commits}
- Currently analyzing: Commit {commit_number}/{total_commits}{chunk_info}

COMMIT BEING ANALYZED:
{commit_info}

COMMIT METADATA:
{json.dumps(commit_metadata, indent=2)}

RELEVANT CHANGES IN THIS {"CHUNK" if total_chunks > 1 else "COMMIT"}:
{commit_diff}

STRUCTURED CI FAILURE:
{ci_failure_info}

CI METADATA FOR THIS COMMIT:
{ci_metadata_str}

Use these CI metadata fields directly:
- workflow_run_exists: whether any workflow run metadata exists for this commit
- workflow_names: workflow names found for this commit
- jobs_executed: jobs that executed and their conclusions
- failed_jobs: failed jobs for this commit
- step_names_executed: steps that executed and their conclusions
- failed_steps: failed steps for this commit
- current_jobs_fixed: jobs that passed in this commit
- current_failed_jobs: jobs that failed in this commit

RELEVANT CI VALIDATION STEPS:
{validations_str}

TASK - COMMIT-BASED CI TRANSITION ANALYSIS:

1. Identify the CURRENT CI failure at sha_fail.
   CI stops at the first failing validation step. Use STRUCTURED CI FAILURE only as primary first-failure evidence: failed job, failed step, exact error, failed tool, validation command, and implicated file/line when available.

2. Read the full CI validation sequence.
   RELEVANT CI VALIDATION STEPS are all checks that can apply to these files, including steps CI may not have reached because it stopped at the first failure. Analyze the diff against both the known CI failure and every listed CI verification step. A commit can fix the logged failure, fix a hidden/later validation, introduce a different validation problem, or be relevant only because another CI verification would check it.

3. Analyze this commit's diff only and group changed hunks by failure family.
   Group changes together when they share the same validation command, failure type, root cause, repair strategy, and related files changed for the same reason.
   Split changes when they map to different validations, different root causes, config/dependency enablement versus source/test fixes, or a fix that also introduces another problem.

4. Treat sha_success as the known passing baseline and sha_fail as the failing target.
   The final successful repair trajectory should explain all commits needed to move from the failing state back to passing CI. For this single commit, report only problems/fixes supported by the diff and validation rules.

For each problem object, provide:

1. "files": List of files affected.
2. "failure_type": Broad validation family, such as "format", "lint", "type_check", "test", "build", "install", "import", "docs", or "unknown".
3. "issue_type": Specific issue family, such as "missing_return_annotation", "import_order", "dependency_version", or "assertion_update".
4. "problem": The CI problem or fix being described, with step/job and line numbers when available.
5. "root_cause": The underlying technical cause.
6. "changes_made": What this commit changed in the code.
7. "introduces": What problems or validation challenges this commit introduced, if any. Use "" if none.
8. "fixes": What this commit fixed and why these changes fix it. Use "" if it fixes nothing.
9. "current_failed_jobs": Failed job/step records from CI METADATA for this commit, including validation_cmd when available, or [].
10. "current_fixed_jobs": Jobs from CI METADATA that passed in this commit, or [].
11. "validation_cmd": The exact CI command that verifies the fixed or introduced issue.

OUTPUT JSON FORMAT (valid JSON only, no markdown):
{{
  "commit_sha": {commit_sha_json},
  "commit_message": {commit_message_json},
  "current_jobs_fixed": [],
  "current_failed_jobs": [],
  "problems": [
    {{
      "files": ["inflatable_test.py"],
      "failure_type": "type_check",
      "issue_type": "missing_return_annotation",
      "problem": "mypy fails because test_get_object_id lacks an explicit return type.",
      "root_cause": "The project type-checking configuration requires test functions to declare -> None.",
      "changes_made": "Added '-> None' to test_get_object_id.",
      "introduces": "",
      "fixes": "Fixes the mypy validation because the test function now satisfies strict return type requirements.",
      "current_failed_jobs": [],
      "current_fixed_jobs": [],
      "validation_cmd": "python -m mypy inflatable_test.py"
    }}
  ]
}}

IMPORTANT RULES:
1. STRUCTURED CI FAILURE shows only the known first failing step at sha_fail. Do not assume later validation steps passed or are irrelevant.
2. Check every changed hunk against the structured CI failure and every relevant validation step, not only the logged failure.
3. Do not claim a fix unless the diff directly explains why the validation would now pass.
4. Do not claim a new problem unless the diff directly violates a validation rule or explains the current CI failure.
5. If the commit is unrelated to validated CI behavior, return the top-level commit fields with "problems": [].
6. Same failure family must be one problem object, even if many files changed.
7. Different validators, root causes, or repair strategies must be separate problem objects.
8. Use plural field names exactly: "current_failed_jobs" and "current_fixed_jobs".
9. Do not invent current_failed_jobs/current_fixed_jobs. They must come from CI METADATA for this commit.
10. Return ONLY valid JSON, no markdown, no comments, and no placeholder entries.
11. If no relevant changes, return {{"commit_sha": {commit_sha_json}, "commit_message": {commit_message_json}, "current_jobs_fixed": [], "current_failed_jobs": [], "problems": []}}"""

        # Call LLM
        try:
            response = self._call_llm(prompt)
            result = self._parse_response(response)
            import pdb

            pdb.set_trace()
            return self._normalize_commit_analysis(result, group)
        except Exception as e:
            print(f"    ERROR: LLM analysis failed: {e}")
            import traceback

            traceback.print_exc()
            return {"problems": [], "error": str(e)}

    def select_relevant_files(
        self,
        *,
        commit_metadata: Dict,
        changed_files: List[str],
        commit_diff: str,
        structured_ci_failure: Dict,
        ci_metadata: Dict,
        relevant_validations: List[Dict],
    ) -> Dict:
        """Use LLM to select changed files relevant to CI failure or validation based on actual changes."""

        prompt = f"""Analyze the ACTUAL CHANGES in this commit's diff to select files whose changes are directly or indirectly relevant to the current CI failure or any CI verification step.

COMMIT METADATA:
{json.dumps(commit_metadata, indent=2)}

COMMIT DIFF (contains all changed files and their changes):
{commit_diff}

STRUCTURED CI FAILURE:
{json.dumps(structured_ci_failure, indent=2)}

RELEVANT CI VALIDATION STEPS:
{json.dumps(relevant_validations, indent=2)}

TASK:
Analyze the DIFF to determine which file changes are relevant for commit-based CI decomposition.

Select files when their changes (visible in the DIFF):
- DIRECTLY fix or introduce the issue mentioned in STRUCTURED CI FAILURE
- Are validated by the failed validation command or any CI validation step in RELEVANT CI VALIDATION STEPS
- Modify code, tests, config, dependencies, or tool settings that could affect any validation step's behavior
- Change type annotations, imports, function signatures, or code structure checked by CI tools (mypy, pylint, black, isort, pytest, etc.)

IMPORTANT:
- Analyze the actual changes in each file from the DIFF, not just file names
- A file changing only version numbers or unrelated content should NOT be selected if the change doesn't affect any validation
- Multiple pyproject.toml files with identical dependency bumps may all be relevant if they affect the same validation tools, or has direct or indirect impact on the CI validations.

Do not select files when:
- The changes are completely unrelated to any failed validation or listed CI verification command
- The file changes cannot affect any validation step's outcome (e.g., comments-only changes when no doc validator failed)

OUTPUT JSON ONLY:
{{
  "selected_files": ["path/to/file.py"],
  "reasoning": "brief explanation of why these files were selected based on their actual changes"
}}"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_response(response)
            import pdb

            pdb.set_trace()
            selected = self._resolve_selected_files(
                result.get("selected_files", []), changed_files
            )
            return {
                "selected_files": selected,
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            print(f"    Warning: relevant file selection failed: {e}")
            return {"selected_files": [], "reasoning": ""}

    def _resolve_selected_files(
        self, selected_files: List[str], changed_files: List[str]
    ) -> List[str]:
        """Map LLM-selected paths back to exact changed file paths."""
        resolved = []
        changed = [str(file_path) for file_path in changed_files]

        for selected in selected_files or []:
            selected_path = str(selected).strip().lstrip("/")
            if not selected_path:
                continue

            match = None
            for changed_path in changed:
                normalized_changed = changed_path.strip().lstrip("/")
                if normalized_changed == selected_path:
                    match = changed_path
                    break
                if normalized_changed.endswith(f"/{selected_path}"):
                    match = changed_path
                    break
                if selected_path.endswith(f"/{normalized_changed}"):
                    match = changed_path
                    break

            if match and match not in resolved:
                resolved.append(match)

        return resolved

    def _normalize_commit_analysis(self, result: Dict, group: Dict) -> Dict:
        """Normalize LLM output to the commit-level schema."""
        commits = group.get("commits") or [{}]
        commit = commits[0]
        commit_sha = result.get("commit_sha") or commit.get("sha", "unknown")
        commit_message = result.get("commit_message") or commit.get("message", "")
        ci_metadata = group.get("ci_metadata") or {}
        current_jobs_fixed = ci_metadata.get("current_jobs_fixed", [])
        current_failed_jobs = ci_metadata.get("current_failed_jobs", [])

        normalized = {
            "commit_sha": commit_sha,
            "commit_message": commit_message,
            "current_jobs_fixed": current_jobs_fixed,
            "current_failed_jobs": current_failed_jobs,
            "problems": [],
        }

        for problem in result.get("problems", []) or []:
            if not isinstance(problem, dict):
                continue
            normalized["problems"].append(
                {
                    "files": problem.get("files", []),
                    "failure_type": problem.get("failure_type", ""),
                    "issue_type": problem.get("issue_type", ""),
                    "problem": problem.get("problem", ""),
                    "root_cause": problem.get("root_cause", ""),
                    "changes_made": problem.get("changes_made", ""),
                    "introduces": problem.get("introduces", ""),
                    "fixes": problem.get("fixes", ""),
                    "current_failed_jobs": current_failed_jobs,
                    "current_fixed_jobs": current_jobs_fixed,
                    "validation_cmd": problem.get("validation_cmd", ""),
                    "commit_sha": commit_sha,
                    "commit_message": commit_message,
                    "sha_success": group.get("sha_success"),
                    "sha_fail": group.get("sha_fail"),
                }
            )

        return normalized

    def _format_commit_info(self, commits: List[Dict]) -> str:
        """Format commit information for prompt"""
        if not commits:
            return "No commit information available"

        lines = []
        for i, commit in enumerate(commits, 1):
            sha = commit.get("sha", "unknown")
            short_sha = sha[:8] if sha != "unknown" else sha
            message = commit.get("message", "No message")
            lines.append(f"Commit {i}: {short_sha}")
            lines.append(f"  SHA: {sha}")
            lines.append(f"  Message: {message}")

        return "\n".join(lines)

    def _extract_ci_failure(self, ci_logs: Any) -> str:
        """Extract relevant CI failure information"""
        if isinstance(ci_logs, str):
            # Simple string logs
            lines = ci_logs.split("\n")
            # Find error lines
            error_lines = [
                l for l in lines if "error" in l.lower() or "failed" in l.lower()
            ]
            if error_lines:
                return "\n".join(error_lines[:10])  # First 10 error lines
            return ci_logs[:2000]  # First 2000 chars

        elif isinstance(ci_logs, list):
            # List of log entries
            all_logs = []
            for entry in ci_logs:
                if isinstance(entry, dict):
                    log = entry.get("log", "")
                    all_logs.append(log)
                else:
                    all_logs.append(str(entry))

            combined = "\n".join(all_logs)
            lines = combined.split("\n")
            error_lines = [
                l for l in lines if "error" in l.lower() or "failed" in l.lower()
            ]
            if error_lines:
                return "\n".join(error_lines[:10])
            return combined[:2000]

        elif isinstance(ci_logs, dict):
            compact = {
                "error_context": ci_logs.get("error_context", []),
                "relevant_files": ci_logs.get("relevant_files", []),
                "error_types": ci_logs.get("error_types", []),
                "failed_job": ci_logs.get("failed_job", []),
                "sha_fail": ci_logs.get("sha_fail"),
                "id": ci_logs.get("id"),
            }
            return json.dumps(compact, indent=2)

        return str(ci_logs)[:2000]

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        try:
            messages = [{"role": "user", "content": prompt}]

            # Get credentials based on model type
            api_key, api_base = self._get_model_credentials(self.model_name)

            # Auto-detect max_tokens
            max_tokens = self._get_max_tokens(self.model_name)

            completion_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "timeout": int(os.getenv("LITELLM_TIMEOUT", "900")),
            }

            # Add API credentials
            if api_key:
                completion_kwargs["api_key"] = api_key
            if api_base:
                completion_kwargs["api_base"] = api_base

            response = litellm.completion(**completion_kwargs)
            return response.choices[0].message.content or ""

        except Exception as e:
            raise Exception(f"LLM API call failed: {e}")

    def _get_model_credentials(self, model_name: str) -> tuple:
        """Get API credentials based on model type"""
        lowered = str(model_name or "").lower()

        # OpenRouter models (MiniMax)
        if lowered.startswith("openrouter/") or "minimax" in lowered:
            return (
                os.getenv("OPENROUTER_API_KEY"),
                os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            )

        # GLM models
        if "glm" in lowered or "z-ai" in lowered or "zai" in lowered:
            return (
                os.getenv("GLM_API_KEY"),
                os.getenv("GLM_BASE_URL", "https://api.z.ai/api/paas/v4"),
            )

        # Default to OpenRouter
        return (
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    def _get_max_tokens(self, model_name: str) -> int:
        """Get max tokens based on model"""
        try:
            from scripts.model_token_config import get_output_safe_tokens

            return get_output_safe_tokens(model_name)
        except:
            # Fallback
            if "glm" in str(model_name).lower():
                return 120000  # GLM supports large output
            return 16000  # Default

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response as JSON"""
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start:end])
                except:
                    pass

            print(f"    WARNING: Could not parse LLM response as JSON: {e}")
            print(f"    Response preview: {response[:500]}")
            return {"problems": [], "parse_error": str(e)}
