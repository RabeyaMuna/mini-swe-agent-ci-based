#!/usr/bin/env python3
"""
commit_analyzer.py - Analyze commits to extract problems, root causes, and repair plans
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from utilities.model_token_config import get_input_chunk_tokens
from prompt_template import build_commit_analysis_prompt, build_file_selection_prompt
from utilities.diff_chunker import chunk_diff_by_files, estimate_tokens
from utilities.llm_invoker import invoke_llm_with_retry
from utilities.model_token_config import get_output_safe_tokens

load_dotenv()

# Safety threshold: chunk size should not exceed 45% of model's context window
# This ensures: 45% diff + 30% output + 25% prompt overhead = 100%
CHUNK_SAFETY_THRESHOLD = 0.45


class CommitAnalyzer:
    """Analyzes commits to extract problems and repair plans using LLM"""

    def __init__(self, llm):
        """
        Initialize analyzer with an LLM instance.

        Args:
            llm: LLM instance from utilities.llm_provider.get_llm()
        """
        self.llm = llm
        # Extract model name for token limit calculations
        self.model_name = getattr(llm, 'model_name', None) or getattr(llm, 'memci_model_key', 'minimax/minimax-m2.5')

    def select_relevant_files(
        self,
        *,
        commit_metadata: Dict,
        changed_files: List[str],
        commit_diff: str,
        structured_ci_failure: Dict,
        relevant_validations: List[Dict],
        relationship_context: List[Dict] = None,
    ) -> Dict:
        """Select changed files relevant to CI validation."""
        max_chunk_tokens = self._get_safe_chunk_size()
        chunks = self._diff_chunks(commit_diff, changed_files, max_chunk_tokens)
        print(
            f"    File selection diff: ~{estimate_tokens(commit_diff):,} tokens, "
            f"{len(chunks)} chunk(s)"
        )

        try:
            all_groups = []
            for i, chunk in enumerate(chunks, 1):
                print(
                    f"      Selecting files from chunk {i}/{len(chunks)}: "
                    f"{len(chunk['files'])} file(s)"
                )
                prompt = build_file_selection_prompt(
                    commit_metadata,
                    chunk["files"],
                    chunk["diff"],
                    structured_ci_failure,
                    relevant_validations,
                    relationship_context,
                )
                chunk_result = self._call_json(prompt)
                all_groups.extend(chunk_result.get("selected_groups", []))

            # Resolve and organize in one pass (no redundant merging)
            selected_groups = self._resolve_and_organize_groups(
                all_groups, changed_files, relevant_validations
            )
            return {
                "selected_groups": selected_groups,
                "reasoning": "",
            }

        except Exception as e:
            print(f"    ERROR: File selection failed: {e}")
            # Fallback: select all files (safer than returning empty)
            return {
                "selected_groups": [
                    {
                        "files": changed_files,
                        "failure_type": "unknown",
                        "issue_type": "unknown",
                        "validation_cmd": "",
                        "reason": f"Fallback due to error: {e}",
                    }
                ],
                "reasoning": f"Error during selection, selected all files: {e}",
            }

    def analyze_commit_group(
        self,
        group: Dict,
        commit_diff: str,
        relevant_validations: List[Dict],
    ) -> Dict:
        """Analyze a selected commit/file group into CI problem events."""
        commit_info = self._format_commit_info(group.get("commits", []))
        sha_success = group.get("sha_success") or "unknown"
        sha_fail = group.get("sha_fail") or "unknown"
        commit_number = group.get("commit_number", 1)
        total_commits = group.get("total_commits", 1)
        selected_groups = group.get("selected_groups", [])
        ci_metadata = group.get("ci_metadata") or {}
        structured_ci_failure = group.get("structured_ci_failure") or {}
        current_commit = (group.get("commits") or [{}])[0]
        commit_sha = current_commit.get("sha", "unknown")
        max_chunk_tokens = self._get_safe_chunk_size()
        chunks = self._diff_chunks(
            commit_diff, group.get("files_in_chunk", []), max_chunk_tokens
        )
        print(
            f"    Commit analysis diff: ~{estimate_tokens(commit_diff):,} tokens, "
            f"{len(chunks)} chunk(s)"
        )

        try:
            all_problems = []
            for i, chunk in enumerate(chunks, 1):
                print(f"      Analyzing commit chunk {i}/{len(chunks)}")
                prompt = build_commit_analysis_prompt(
                    commit_info=commit_info,
                    commit_diff=chunk["diff"],
                    commit_sha=commit_sha,
                    sha_success=sha_success,
                    sha_fail=sha_fail,
                    commit_number=commit_number,
                    total_commits=total_commits,
                    chunk_number=i,
                    total_chunks=len(chunks),
                    files_in_chunk=chunk["files"],
                    selected_groups=selected_groups,
                    ci_failure_info=structured_ci_failure,
                    ci_metadata=ci_metadata,
                    relevant_validations=relevant_validations,
                )
                chunk_result = self._call_json(prompt)
                all_problems.extend(chunk_result.get("problems", []))

            result = {
                "commit_sha": commit_sha,
                "problems": self._consolidate_problem_events(
                    all_problems, relevant_validations
                ),
            }
            return result

        except Exception as e:
            print(f"    ERROR: LLM analysis failed: {e}")
            import traceback

            traceback.print_exc()
            return {"problems": [], "error": str(e)}

    def _diff_chunks(
        self, commit_diff: str, files: List[str], max_tokens: int
    ) -> List[Dict]:
        """Return one full-diff chunk or file-based chunks when needed."""
        if estimate_tokens(commit_diff) <= max_tokens:
            return [{"files": files or [], "diff": commit_diff}]
        return chunk_diff_by_files(commit_diff, max_tokens_per_chunk=max_tokens)

    def _resolve_and_organize_groups(
        self,
        all_groups: List[Dict],
        changed_files: List[str],
        validation_sequence: List[Dict],
    ) -> List[Dict]:
        """
        Unified method: Resolve file paths AND organize by validation in one pass.

        Eliminates redundant merging by doing both operations together:
        1. Merge groups from multiple chunks
        2. Resolve fuzzy file paths to exact matches
        3. Order by validation sequence
        """
        validation_order = self._validation_order_by_cmd(validation_sequence)
        merged: Dict[tuple, Dict] = {}

        # Process all groups in one pass
        for group in all_groups or []:
            if not isinstance(group, dict):
                continue

            # Resolve file paths
            resolved_files = self._resolve_selected_files(
                group.get("files", []), changed_files
            )
            if not resolved_files:
                continue

            # Merge by validation + failure type
            validation_cmd = group.get("validation_cmd", "")
            failure_type = group.get("failure_type", "")
            key = (
                validation_order.get(validation_cmd, 9999),
                validation_cmd,
                failure_type,
            )

            if key not in merged:
                merged[key] = {
                    "validation_order": key[0],
                    "failure_type": failure_type,
                    "validation_cmd": validation_cmd,
                    "groups": [],
                }

            # Add resolved files as subgroup
            merged[key]["groups"].append(
                {
                    "files": resolved_files,
                    "issue_type": group.get("issue_type", ""),
                    "reason": group.get("reason", ""),
                }
            )

        return [merged[key] for key in sorted(merged)]

    def _organize_groups_by_validation(
        self, groups: List[Dict], validation_sequence: List[Dict]
    ) -> List[Dict]:
        """Merge and order selected groups by validation command/failure family."""
        validation_order = self._validation_order_by_cmd(validation_sequence)
        merged: Dict[tuple, Dict] = {}

        for group in groups or []:
            validation_cmd = group.get("validation_cmd", "")
            failure_type = group.get("failure_type", "")
            key = (
                validation_order.get(validation_cmd, 9999),
                validation_cmd,
                failure_type,
            )
            if key not in merged:
                merged[key] = {
                    "validation_order": key[0],
                    "failure_type": failure_type,
                    "validation_cmd": validation_cmd,
                    "groups": [],
                }

            subgroups = group.get("groups") or [
                {
                    "files": group.get("files", []),
                    "issue_type": group.get("issue_type", ""),
                    "reason": group.get("reason", ""),
                }
            ]
            for subgroup in subgroups:
                subgroup_files = subgroup.get("files", []) or []
                merged[key]["groups"].append(
                    {
                        "files": subgroup_files,
                        "issue_type": subgroup.get("issue_type", ""),
                        "reason": subgroup.get("reason", ""),
                    }
                )

        return [merged[key] for key in sorted(merged)]

    def _consolidate_problem_events(
        self, problems: List[Dict], validation_sequence: List[Dict]
    ) -> List[Dict]:
        """Merge chunk-level problem events within validation/failure groups."""
        validation_order = self._validation_order_by_cmd(validation_sequence)
        grouped: Dict[tuple, Dict] = {}

        for problem in problems or []:
            if not isinstance(problem, dict):
                continue
            validation_cmd = problem.get("validation_cmd", "")
            failure_type = problem.get("failure_type", "")
            issue_type = problem.get("issue_type", "")
            fixed = bool(problem.get("fixed"))
            introduced = bool(problem.get("introduced"))
            key = (
                validation_order.get(validation_cmd, 9999),
                validation_cmd,
                failure_type,
                issue_type,
                fixed,
                introduced,
            )
            item = grouped.setdefault(
                key,
                {
                    "files": [],
                    "failure_type": failure_type,
                    "issue_type": issue_type,
                    "problem": [],
                    "root_cause": [],
                    "changes_made": [],
                    "introduced": introduced,
                    "fixed": fixed,
                    "fix_strategy": [],
                    "why_this_fix_works": [],
                    "repair": [],
                    "validation_cmd": validation_cmd,
                },
            )
            for file_path in problem.get("files", []) or []:
                if file_path not in item["files"]:
                    item["files"].append(file_path)
            for field in [
                "problem",
                "root_cause",
                "changes_made",
                "fix_strategy",
                "why_this_fix_works",
                "repair",
            ]:
                value = problem.get(field, "")
                if value and value not in item[field]:
                    item[field].append(value)

        consolidated = []
        for key in sorted(grouped):
            item = grouped[key]
            fixed = item["fixed"]
            consolidated.append(
                {
                    "files": item["files"],
                    "failure_type": item["failure_type"],
                    "issue_type": item["issue_type"],
                    "problem": self._join_unique(item["problem"]),
                    "root_cause": self._join_unique(item["root_cause"]),
                    "changes_made": self._join_unique(item["changes_made"]),
                    "introduced": item["introduced"],
                    "fixed": fixed,
                    "fix_strategy": self._join_unique(item["fix_strategy"])
                    if fixed
                    else "",
                    "why_this_fix_works": self._join_unique(item["why_this_fix_works"])
                    if fixed
                    else "",
                    "repair": self._join_unique(item["repair"]) if fixed else "",
                    "validation_cmd": item["validation_cmd"],
                }
            )
        return consolidated

    def _validation_order_by_cmd(
        self, validation_sequence: List[Dict]
    ) -> Dict[str, int]:
        """Map validation commands to CI order."""
        order_by_cmd = {}
        for item in validation_sequence or []:
            order = item.get("order")
            if order is None:
                continue
            for field in ["validation_cmd", "installation_cmd"]:
                cmd = item.get(field, "")
                if cmd:
                    order_by_cmd[cmd] = int(order)
        return order_by_cmd

    def _join_unique(self, values: List[str]) -> str:
        """Join unique non-empty strings in original order."""
        return "\n\n".join(dict.fromkeys(v for v in values if v))

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

    def _resolve_selected_groups(
        self, selected_groups: List[Dict], changed_files: List[str]
    ) -> List[Dict]:
        """Map LLM-selected grouped paths back to exact changed file paths."""
        resolved_groups = []
        for group in selected_groups or []:
            if not isinstance(group, dict):
                continue
            files = self._resolve_selected_files(group.get("files", []), changed_files)
            if not files:
                continue
            resolved_groups.append(
                {
                    "files": files,
                    "failure_type": group.get("failure_type", ""),
                    "issue_type": group.get("issue_type", ""),
                    "validation_cmd": group.get("validation_cmd", ""),
                    "reason": group.get("reason", ""),
                }
            )

        return resolved_groups

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

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API using utilities.llm_invoker for consistent handling"""
        try:
            # Get model-specific output token limit
            max_tokens = self._get_output_tokens()

            # Use the centralized LLM invoker with retry logic, rate limiting, etc.
            response = invoke_llm_with_retry(
                llm=self.llm,
                prompt=prompt,
                max_tokens=max_tokens,  # Use model-aware token limit
                parse_json=False,  # We'll parse JSON separately in _call_json
                verbose=False,
            )
            return response if isinstance(response, str) else str(response)

        except Exception as e:
            raise Exception(f"LLM API call failed: {e}")

    def _call_json(self, prompt: str) -> Dict:
        """Call the model and parse a JSON object."""
        return self._parse_response(self._call_llm(prompt))

    def _get_safe_chunk_size(self) -> int:
        """
        Calculate safe chunk size based on 45% of model's context window.

        This ensures the total prompt (chunk + template + overhead) never exceeds
        the model's context window, preventing LLM confusion and JSON format errors.

        Formula: chunk_size = 45% × context_window
        Reasoning: 45% chunk + 30% output + 25% prompt overhead = 100%
        """
        try:
            from utilities.model_token_config import get_model_config

            config = get_model_config(self.model_name)
            context_window = config["input_context_window"]
            safe_chunk_size = int(context_window * CHUNK_SAFETY_THRESHOLD)

            return safe_chunk_size
        except Exception:
            # Fallback for unknown models: conservative 30k tokens
            return 30_000

    def _get_input_chunk_tokens(self, fallback: int) -> int:
        """Get model-aware input chunk size, capped by the caller's target."""
        try:

            return min(fallback, get_input_chunk_tokens(self.model_name))
        except Exception:
            return fallback

    def _get_output_tokens(self) -> int:
        """Get model-aware output token limit."""
        try:


            return get_output_safe_tokens(self.model_name)
        except Exception:
            # Fallback for unknown models
            if "glm" in str(self.model_name).lower():
                return 120000  # GLM supports large output
            return 16000  # Safe default for most models

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
                except json.JSONDecodeError:
                    pass

            print(f"    WARNING: Could not parse LLM response as JSON: {e}")
            print(f"    Response preview: {response[:500]}")
            return {"problems": [], "parse_error": str(e)}
