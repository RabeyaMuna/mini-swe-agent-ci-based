#!/usr/bin/env python3
"""
commit_based_decomposer.py - Simple commit-level decomposition

Simplified production-ready version:
1. Fetch diff for each commit
2. Use cached CI validation data (survives 90 days)
3. LLM decides what's relevant (no pre-filtering)
4. Final LLM organization step
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from commit_decomposition.commit_analyzer import CommitAnalyzer
from commit_decomposition.trajectory_builder import organize_trajectory_with_llm
from commit_decomposition.github_fetcher import GitHubFetcher
from commit_decomposition.ci_command_resolver import enrich_ci_metadata_commands
from commit_decomposition.diff_filter import (
    filter_diff,
    filter_diff_to_files,
    get_changed_files,
    get_filtered_file_count,
    should_ignore_file,
)
from utilities.ci_cache import (
    load_structured_ci_failure as load_cached_structured_ci_failure,
    load_validation_sequence as load_cached_validation_sequence,
)
from utilities.dependency_evidence import dependency_graph_evidence


def load_structured_ci_failure(sha_fail: str, issue_id: str = "") -> Dict:
    """Load structured CI failure details from data/log_details.json (standardized location)."""
    return load_cached_structured_ci_failure(sha_fail, issue_id)


def compact_ci_metadata(ci_metadata: Dict) -> Dict:
    """Keep only CI metadata useful for commit-level reasoning."""
    return {
        "workflow_run_exists": ci_metadata.get("workflow_run_exists", False),
        "workflow_names": ci_metadata.get("workflow_names", []),
        "jobs_executed": ci_metadata.get("jobs_executed", []),
        "job_conclusions": ci_metadata.get("job_conclusions", []),
        "step_names_executed": ci_metadata.get("step_names_executed", []),
        "failed_jobs": ci_metadata.get("failed_jobs", []),
        "failed_steps": ci_metadata.get("failed_steps", []),
        "current_failed_jobs": ci_metadata.get("current_failed_jobs", []),
        "current_jobs_fixed": ci_metadata.get("current_jobs_fixed", []),
    }


def structured_failure_files(structured_failure: Dict) -> List[str]:
    """Extract file paths directly mentioned by structured CI failure data."""
    files = []
    for item in structured_failure.get("relevant_files", []) or []:
        if isinstance(item, dict) and item.get("file"):
            files.append(str(item["file"]))
    return files


def filter_relevant_validations(
    validation_sequence: List[Dict], structured_failure: Dict
) -> List[Dict]:
    """Prefer validations matching failed commands/tools from structured failure."""
    failure_items = (structured_failure.get("relevant_files", []) or []) + (
        structured_failure.get("failed_job", []) or []
    )
    failed_commands = {
        str(item.get("failed_cmd") or item.get("command") or "").strip()
        for item in failure_items
        if isinstance(item, dict)
    }
    failed_tools = {
        str(item.get("failed_tool") or "").strip().lower()
        for item in structured_failure.get("relevant_files", []) or []
        if isinstance(item, dict) and item.get("failed_tool")
    }

    relevant = []
    for validation in validation_sequence:
        cmd = str(validation.get("validation_cmd") or "").strip()
        validates = str(validation.get("validates") or "").lower()
        if cmd and cmd in failed_commands:
            relevant.append(validation)
        elif failed_tools and any(
            tool in validates or tool in cmd.lower() for tool in failed_tools
        ):
            relevant.append(validation)

    return relevant or validation_sequence


def validation_candidate_files(
    changed_files: List[str], validation_sequence: List[Dict], structured_failure: Dict
) -> List[str]:
    """Return all changed files except hard-ignored paths.

    Relevance is decided by the LLM using CI failure and validation context.
    This deterministic layer only applies the user-specified hard ignore rules.
    """
    candidates = []
    for file_path in changed_files:
        path = str(file_path)
        if should_ignore_file(path):
            continue
        candidates.append(path)

    return candidates


def merge_file_lists(*file_lists: List[str]) -> List[str]:
    """Merge file lists while preserving order."""
    merged = []
    for file_list in file_lists:
        for file_path in file_list or []:
            if file_path not in merged:
                merged.append(file_path)
    return merged


def files_from_groups(groups: List[Dict]) -> List[str]:
    """Flatten grouped file selections while preserving group/file order."""
    return merge_file_lists(*(files_from_group(group) for group in groups or []))


def files_from_group(group: Dict) -> List[str]:
    """Return all files represented by a selected validation/failure group."""
    if group.get("files"):
        return group.get("files", [])
    return merge_file_lists(
        *(subgroup.get("files", []) for subgroup in group.get("groups", []) or [])
    )


def deterministic_validation_groups(
    files: List[str], validation_sequence: List[Dict]
) -> List[Dict]:
    """Group missed files by obvious validator scope from CI commands."""
    groups_by_key: Dict[tuple, Dict] = {}

    def find_validation(*needles: str) -> str:
        lowered_needles = [needle.lower() for needle in needles if needle]
        for validation in validation_sequence:
            text = " ".join(
                [
                    str(validation.get("validation_cmd") or ""),
                    str(validation.get("validates") or ""),
                    str(validation.get("source") or ""),
                ]
            ).lower()
            if all(needle in text for needle in lowered_needles):
                return str(validation.get("validation_cmd") or "")
        return ""

    def add_group(
        file_path: str, failure_type: str, issue_type: str, cmd: str, reason: str
    ):
        key = (failure_type, issue_type, cmd)
        group = groups_by_key.setdefault(
            key,
            {
                "files": [],
                "failure_type": failure_type,
                "issue_type": issue_type,
                "validation_cmd": cmd,
                "reason": reason,
            },
        )
        group["files"].append(file_path)

    for file_path in files:
        path = str(file_path)
        suffix = Path(path).suffix
        is_examples_or_benchmarks = path.startswith(("examples/", "benchmarks/"))
        is_framework_python = path.startswith(("framework/py/", "py/flwr/"))

        if is_examples_or_benchmarks and suffix == ".py":
            add_group(
                path,
                "format",
                "black_formatting",
                find_validation("black", "../examples")
                or find_validation("black", "../benchmarks"),
                "Python examples/benchmarks files are checked by the examples Black validation.",
            )
        elif is_examples_or_benchmarks and suffix in {".md", ".mdx"}:
            add_group(
                path,
                "format",
                "markdown_formatting",
                find_validation("mdformat", "../examples"),
                "Examples Markdown files are checked by the examples mdformat validation.",
            )
        elif is_examples_or_benchmarks and suffix == ".toml":
            add_group(
                path,
                "format",
                "toml_formatting",
                find_validation("taplo", "../examples")
                or find_validation("taplo", "../benchmarks"),
                "Examples/benchmarks TOML files are checked by the examples taplo validation.",
            )
        elif (
            path.startswith(("framework/docs/source/", "docs/source/"))
            and suffix == ".rst"
        ):
            add_group(
                path,
                "format",
                "rst_formatting",
                find_validation("docstrfmt", "docs/source"),
                "RST documentation files are checked by docstrfmt.",
            )
        elif path.startswith(("framework/docs/source/", "docs/source/")) and suffix in {
            ".md",
            ".mdx",
        }:
            add_group(
                path,
                "format",
                "markdown_formatting",
                find_validation("mdformat", "docs/source"),
                "Documentation Markdown files are checked by mdformat.",
            )
        elif suffix == ".toml":
            add_group(
                path,
                "format",
                "toml_formatting",
                find_validation("taplo"),
                "TOML files are checked by taplo formatting validation.",
            )
        elif is_framework_python and suffix == ".py":
            add_group(
                path,
                "unknown",
                "framework_python_validation",
                "",
                "Framework Python files are checked by multiple validators; detailed analysis must infer the exact validator from the diff.",
            )
        else:
            add_group(
                path,
                "unknown",
                "validation_scope_candidate",
                "",
                "Changed file passed hard filtering but was not selected by the LLM; keep it for detailed CI validation analysis.",
            )

    return list(groups_by_key.values())


def _bounded_text(text: str, limit: int = 8000) -> str:
    """Keep prompt context bounded while preserving the start of the evidence."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... <truncated group diff> ..."


def build_selected_group_context(
    full_diff: str,
    selected_groups: List[Dict],
    *,
    repo_path: str | None = None,
) -> List[Dict]:
    """Build LLM-facing validation/failure clusters with combined diffs."""
    enriched = []
    for group in selected_groups or []:
        files = files_from_group(group)
        group_diff = filter_diff_to_files(full_diff, files)
        dependency_context = dependency_graph_evidence(group_diff, repo_path=repo_path)
        validation_cmd = str(group.get("validation_cmd") or "").strip()
        enriched.append(
            {
                "validation_order": group.get("validation_order"),
                "validation_cmd": validation_cmd,
                "failure_type": group.get("failure_type", ""),
                "groups": group.get("groups")
                or [
                    {
                        "files": files,
                        "issue_type": group.get("issue_type", ""),
                        "reason": group.get("reason", ""),
                    }
                ],
                "diff": _bounded_text(group_diff),
                "dependency_context": dependency_context,
            }
        )

    return enriched


def extract_validation_sequence_from_ci_metadata(
    ci_metadata: Dict, workflow_path: str = ""
) -> List[Dict]:
    """
    Build validation sequence from CI metadata step information

    Args:
        ci_metadata: CI metadata with step_names_executed
        workflow_path: Workflow file path

    Returns:
        Validation sequence list (order, validates, validation_cmd, source)
    """
    if not ci_metadata or not ci_metadata.get("step_names_executed"):
        return []

    steps = ci_metadata.get("step_names_executed", [])
    validation_sequence = []
    seen_steps = set()

    for idx, step_info in enumerate(steps, 1):
        step_name = step_info.get("step", "")
        workflow_name = step_info.get("workflow", "")

        # Skip duplicates (same step name)
        if step_name in seen_steps:
            continue
        seen_steps.add(step_name)

        # Build validation entry
        validation_sequence.append(
            {
                "order": idx,
                "validates": step_name,
                "validation_cmd": "",  # Will be enriched later if needed
                "source": workflow_path or workflow_name,
            }
        )

    return validation_sequence


def generate_and_cache_validation(
    issue_id: str, issue_data: Dict, github_fetcher, analyzer
) -> Dict:
    """Generate validation sequence from workflow content and cache it."""
    repo_owner = issue_data.get("repo_owner", "")
    repo_name = issue_data.get("repo_name", "")
    sha_fail = issue_data.get("sha_fail", "")
    workflow_path = issue_data.get("workflow_path", "")

    if not all([repo_owner, repo_name, sha_fail, workflow_path]):
        print("    Missing required fields to generate validation sequence")
        return {
            "validation_sequence": [],
            "failure_info": "",
            "workflow_path": workflow_path,
        }

    print(f"    Generating validation sequence from workflow at {sha_fail[:8]}...")
    workflow_content = github_fetcher.get_file_content(
        repo_owner, repo_name, workflow_path, sha_fail
    )
    if not workflow_content:
        print(f"    Could not fetch workflow file: {workflow_path}")
        return {
            "validation_sequence": [],
            "failure_info": "",
            "workflow_path": workflow_path,
        }

    result = load_cached_validation_sequence(
        issue_id,
        sha_fail,
        workflow_content=workflow_content,
        workflow_path=workflow_path,
        repo_path=issue_data.get("repo_path") or issue_data.get("checkout_path") or "",
        llm=analyzer._call_llm,
        save=True,
    )
    validation_sequence = result.get("validation_sequence", [])
    print(f"    Generated {len(validation_sequence)} validation steps")

    return {
        "validation_sequence": validation_sequence,
        "failure_info": result.get("failure_info", ""),
        "workflow_path": workflow_path,
    }


def load_validation_cache(
    issue_id: str, issue_data: Dict | None = None, github_fetcher=None, analyzer=None
) -> Dict:
    """
    Load validation data from cache or generate and save it

    Returns:
        validation_sequence: List of CI validation steps
        failure_info: What failed (cached, survives log expiry)
    """
    issue_data = issue_data or {}
    sha_fail = str(issue_data.get("sha_fail") or "")

    # Try cache first
    cached = load_cached_validation_sequence(issue_id, sha_fail)
    if cached.get("validation_sequence"):
        print(
            "  Found validation sequence in cache: "
            f"{len(cached.get('validation_sequence', []))} steps"
        )
        return cached

    # Not in cache - generate and save it
    if issue_data and github_fetcher and analyzer:
        print(f"  Issue {issue_id} not in cache, generating validation sequence...")
        return generate_and_cache_validation(
            issue_id, issue_data, github_fetcher, analyzer
        )

    # Fallback: Return empty - will be populated from ci_metadata during analysis
    # The actual validation steps will come from the per-commit CI metadata
    if issue_data:
        print(f"  Issue {issue_id} not in cache, will use CI metadata from commits...")

        return {
            "validation_sequence": [],
            "failure_info": "",
            "workflow_path": issue_data.get("workflow_path", ""),
        }

    print(f"  Warning: No validation data found for issue {issue_id}")
    return {"validation_sequence": [], "failure_info": "", "workflow_path": ""}


def get_commits_between(
    repo_owner: str,
    repo_name: str,
    sha_success: str,
    sha_fail: str,
    github_fetcher: GitHubFetcher,
) -> List[Dict]:
    """
    Get all commits between success and fail using GitHub API

    No local repo needed!
    """
    return github_fetcher.get_commits_between(
        repo_owner, repo_name, sha_success, sha_fail
    )


def get_commit_diff(
    repo_owner: str, repo_name: str, commit_sha: str, github_fetcher: GitHubFetcher
) -> str:
    """
    Get FULL diff for a commit using GitHub API

    No local repo needed!
    """
    return github_fetcher.get_commit_diff(repo_owner, repo_name, commit_sha)


def decompose_issue(
    issue: Dict,
    validation_cache: Dict,
    analyzer: CommitAnalyzer,
    github_fetcher: GitHubFetcher,
) -> Dict:
    """
    Simple commit-based decomposition using GitHub API

    Steps:
    1. Get commits between sha_success and sha_fail (from GitHub)
    2. For each commit: get full diff + analyze with LLM
    3. Collect all problems
    4. Final LLM organization
    """
    issue_id = issue.get("id", "unknown")
    sha_fail = issue.get("sha_fail", "")
    sha_success = issue.get("sha_success", "")
    repo_owner = issue.get("repo_owner", "")
    repo_name = issue.get("repo_name", "")
    repo_path = issue.get("repo_path") or issue.get("checkout_path")
    print(f"\n  Processing issue {issue_id}")
    print(f"    Repository: {repo_owner}/{repo_name}")
    print(f"    sha_success: {sha_success[:12]}")
    print(f"    sha_fail: {sha_fail[:12]}")

    # 1. Get all commits from GitHub
    commits = get_commits_between(
        repo_owner, repo_name, sha_success, sha_fail, github_fetcher
    )
    print(f"    Total commits: {len(commits)}")

    # The benchmark stores the exact repair patch.  GitHub compare can return
    # no commits when either SHA is only available on a temporary benchmark
    # branch, a mirrored repository, or when the API is unavailable.  In that
    # case the stored diff is still authoritative and can be analyzed as one
    # repair commit.
    if not commits and issue.get("diff"):
        fallback_sha = sha_fail or sha_success or f"dataset-issue-{issue_id}"
        commits = [
            {
                "sha": fallback_sha,
                "message": f"Benchmark repair diff for issue {issue_id}",
                "author": "",
                "date": issue.get("commit_date", ""),
                "html_url": issue.get("commit_link", ""),
                "_diff": issue["diff"],
                "_changed_files": issue.get("changed_files", []),
                "_from_dataset": True,
            }
        ]
        print("    GitHub compare returned no commits; using stored benchmark diff")

    if not commits:
        return {
            "issue_id": issue_id,
            "error": "No commits found",
            "problem_sequence": [],
        }

    # 2. Analyze each commit
    all_problems = []
    commit_analyses = []
    validation_sequence = validation_cache.get("validation_sequence", [])
    workflow_path = validation_cache.get("workflow_path") or issue.get(
        "workflow_path", ""
    )
    structured_failure = load_structured_ci_failure(sha_fail, issue_id)
    if structured_failure:
        print("    Using structured CI failure info from data/log_details.json")
    primary_failure_files = structured_failure_files(structured_failure)

    # Flag to track if validation_sequence needs to be built from first commit's CI metadata
    needs_validation_from_metadata = not validation_sequence

    for i, commit in enumerate(commits, 1):
        commit_sha = commit["sha"]
        print(f"    Analyzing commit {i}/{len(commits)}: {commit_sha[:8]}")

        # First fetch commit diff and changed-file information.
        full_diff = commit.get("_diff") or get_commit_diff(
            repo_owner, repo_name, commit_sha, github_fetcher
        )
        changed_files = get_changed_files(full_diff, include_ignored=True)
        if commit.get("_changed_files"):
            changed_files = merge_file_lists(
                commit.get("_changed_files", []), changed_files
            )
        hard_filtered_diff = filter_diff(full_diff)
        validated_changed_files = get_changed_files(
            hard_filtered_diff, include_ignored=True
        )
        total_files, hard_remaining_files, hard_ignored_files = get_filtered_file_count(
            full_diff
        )
        print(
            f"      Changed files: {total_files}; hard-filtered ignored: "
            f"{hard_ignored_files}; remaining for LLM: {hard_remaining_files}"
        )

        # Then fetch workflow/job/step metadata for this commit.
        ci_metadata = github_fetcher.get_commit_ci_metadata(
            repo_owner, repo_name, commit_sha
        )

        # If validation_sequence is empty (not in cache), extract from first commit's CI metadata
        if needs_validation_from_metadata and ci_metadata.get("step_names_executed"):
            print(
                "      Building validation sequence from CI metadata (not in cache)..."
            )
            validation_sequence = extract_validation_sequence_from_ci_metadata(
                ci_metadata, workflow_path
            )
            print(
                f"      Extracted {len(validation_sequence)} validation steps from CI metadata"
            )
            needs_validation_from_metadata = False  # Only do this once

        ci_metadata = enrich_ci_metadata_commands(
            ci_metadata,
            structured_failure=structured_failure,
            validation_sequence=validation_sequence,
            github_fetcher=github_fetcher,
            repo_owner=repo_owner,
            repo_name=repo_name,
            commit_sha=commit_sha,
            workflow_path=workflow_path,
        )
        compact_metadata = compact_ci_metadata(ci_metadata)
        failed_jobs = compact_metadata.get("current_failed_jobs", [])
        fixed_jobs = compact_metadata.get("current_jobs_fixed", [])

        if ci_metadata.get("workflow_runs"):
            print(
                f"      CI Metadata: {len(failed_jobs)} failed jobs, "
                f"{len(fixed_jobs)} successful jobs "
                f"({len(ci_metadata.get('workflow_runs', []))} workflow runs)"
            )

        commit_analysis = {
            "commit_sha": commit_sha,
            "commit_message": commit.get("message", ""),
            "commit_metadata": {
                "commit_sha": commit_sha,
                "commit_message": commit.get("message", ""),
                "author": commit.get("author", ""),
                "date": commit.get("date", ""),
                "html_url": commit.get("html_url", ""),
                "commit_number": i,
                "total_commits": len(commits),
                "changed_files": changed_files,
                "validated_changed_files": validated_changed_files,
            },
            "changed_files": changed_files,
            "validated_changed_files": validated_changed_files,
            "ci_metadata_summary": {
                "workflow_run_exists": compact_metadata.get(
                    "workflow_run_exists", False
                ),
                "workflow_names": compact_metadata.get("workflow_names", []),
                "jobs_executed": compact_metadata.get("jobs_executed", []),
                "job_conclusions": compact_metadata.get("job_conclusions", []),
                "step_names_executed": compact_metadata.get("step_names_executed", []),
                "failed_jobs": compact_metadata.get("failed_jobs", []),
                "failed_steps": compact_metadata.get("failed_steps", []),
            },
            "current_jobs_fixed": fixed_jobs,
            "current_failed_jobs": failed_jobs,
            "problems": [],
        }

        selection_metadata = dict(commit_analysis["commit_metadata"])
        selection_metadata["changed_files"] = validated_changed_files
        selection_metadata["validated_changed_files"] = validated_changed_files
        preselection_context = build_selected_group_context(
            full_diff,
            [
                {
                    "files": validated_changed_files,
                    "failure_type": "unknown",
                    "issue_type": "unknown",
                    "validation_cmd": "",
                    "reason": (
                        "Pre-selection dependency context for all changed files "
                        "remaining after hard filtering."
                    ),
                }
            ],
            repo_path=repo_path,
        )

        selected = analyzer.select_relevant_files(
            commit_metadata=selection_metadata,
            changed_files=validated_changed_files,
            commit_diff=hard_filtered_diff,
            structured_ci_failure=structured_failure,
            relevant_validations=validation_sequence,
            relationship_context=preselection_context,
        )
        selected_groups = selected.get("selected_groups", [])
        llm_group_files = files_from_groups(selected_groups)
        print(
            f"      LLM-selected groups: {len(selected_groups)} "
            f"({len(llm_group_files)} file(s))"
        )
        candidate_files = validation_candidate_files(
            validated_changed_files, validation_sequence, structured_failure
        )

        # Debug: Log candidate files
        if len(llm_group_files) == 0:
            print(f"      DEBUG: LLM selected 0 files, using fallback")
            print(f"      DEBUG: Candidate files from validation: {candidate_files}")
            print(f"      DEBUG: Structured failure files: {structured_failure_files(structured_failure)}")

        missed_candidate_files = [
            file_path
            for file_path in candidate_files
            if file_path not in set(llm_group_files)
        ]
        deterministic_groups = deterministic_validation_groups(
            missed_candidate_files, validation_sequence
        )
        if not selected_groups:
            selected_groups = [
                {
                    "files": candidate_files,
                    "failure_type": "unknown",
                    "issue_type": "unknown",
                    "validation_cmd": "",
                    "reason": (
                        "Fallback group for CI-relevant candidate files when the "
                        "selector did not return explicit groups."
                    ),
                }
            ]
            selection_source = "fallback candidates"
        else:
            if deterministic_groups:
                selected_groups = selected_groups + deterministic_groups
                selection_source = "LLM groups + validation-scope groups"
            else:
                selection_source = "LLM groups"
        selected_groups = analyzer._organize_groups_by_validation(
            selected_groups, validation_sequence
        )
        selected_files = files_from_groups(selected_groups)
        print(
            f"      Candidate files after hard filter: {len(candidate_files)}; "
            f"final selected for analysis: {len(selected_files)} "
            f"({selection_source})"
        )
        selected_group_context = build_selected_group_context(
            full_diff,
            selected_groups,
            repo_path=repo_path,
        )
        if selected_files:
            commit_analysis["selected_groups"] = selected_group_context
            commit_analysis["selection_reasoning"] = selected.get("reasoning", "")
            print(
                f"      Selected {len(selected_files)} relevant file(s) "
                f"({len(llm_group_files)} LLM-grouped, "
                f"{len(files_from_groups(deterministic_groups))} validation-scope, "
                f"{len(candidate_files)} fallback candidates)"
            )

        # Filter out irrelevant files (.github/workflows, CI configs, etc.)
        included_files = hard_remaining_files
        ignored_files = hard_ignored_files
        primary_diff = filter_diff_to_files(full_diff, selected_files)
        if not primary_diff.strip():
            primary_diff = filter_diff_to_files(full_diff, primary_failure_files)
        if primary_diff.strip():
            filtered_diff = primary_diff
            included_files = len(get_changed_files(filtered_diff, include_ignored=True))
            hard_ignored = get_filtered_file_count(full_diff)[2]
            relevance_excluded = total_files - included_files - hard_ignored
            print(
                f"      Focused diff: {total_files} files → {included_files} "
                f"CI-relevant file(s)"
            )
            if relevance_excluded > 0:
                print(f"      Excluded {relevance_excluded} file(s) as not CI-relevant")
            ignored_files = hard_ignored
        else:
            filtered_diff = hard_filtered_diff
            included_files = len(get_changed_files(filtered_diff, include_ignored=True))
            ignored_files = total_files - included_files

        if ignored_files > 0:
            print(
                f"      Filtered: {total_files} files → {included_files} files "
                f"(ignored {ignored_files} hard-ignored files)"
            )

        if not filtered_diff or filtered_diff.strip() == "":
            print("      No relevant files after filtering, skipping")
            commit_analysis["skipped"] = True
            commit_analysis["skip_reason"] = "No relevant files after filtering"
            commit_analyses.append(commit_analysis)
            continue

        print(
            f"      Analyzing commit with {len(selected_group_context)} "
            "selected validation/failure group(s)"
        )
        print(f"      Validation sequence: {len(validation_sequence)} step(s)")
        print(f"      Structured failure files: {len(structured_failure.get('relevant_files', []))} file(s)")

        analysis_result = analyzer.analyze_commit_group(
            group={
                "commits": [commit],
                "type": "single_commit",
                "commit_number": i,
                "total_commits": len(commits),
                "files_in_chunk": selected_files,
                "changed_files": changed_files,
                "validated_changed_files": validated_changed_files,
                "selected_groups": selected_group_context,
                "sha_success": sha_success,
                "sha_fail": sha_fail,
                "ci_metadata": compact_metadata,
                "structured_ci_failure": structured_failure,
            },
            commit_diff=filtered_diff,
            relevant_validations=validation_sequence,
        )

        commit_problems = []
        for problem in analysis_result.get("problems", []):
            problem["commit"] = commit_sha[:8]
            problem["commit_sha"] = problem.get("commit_sha") or commit_sha
            problem["commit_message"] = problem.get("commit_message") or commit.get(
                "message", ""
            )
            problem["commit_number"] = i
            commit_problems.append(problem)

        # Debug: Log problem detection
        if len(commit_problems) == 0:
            print(f"      ⚠️  No problems detected for commit {commit_sha[:8]}")
            print(f"      Files analyzed: {len(selected_files)}")
            print(f"      Analysis result keys: {list(analysis_result.keys())}")
        else:
            print(f"      ✓ Detected {len(commit_problems)} problem(s)")

        # Add all problems from this commit
        commit_analysis["problems"] = commit_problems
        commit_analysis["total_problems"] = len(commit_problems)
        commit_analyses.append(commit_analysis)
        all_problems.extend(commit_problems)

    print(f"    Total problems found: {len(all_problems)}")

    # 3. Final LLM consolidation (if we have problems)
    # Keep all commit-level records until this point so the final pass can group
    # by validation sequence, failure type, and validation command with full
    # evidence from the repair trajectory.
    if all_problems:
        print("    Consolidating repair trajectory...")
        organized = organize_trajectory_with_llm(
            all_problems=all_problems,
            validation_sequence=validation_sequence,
            analyzer=analyzer,
        )
        problem_sequence = organized
    else:
        problem_sequence = all_problems

    # 4. Collect changed files from all commits
    all_changed_files = set()
    for commit_analysis in commit_analyses:
        all_changed_files.update(commit_analysis.get("changed_files", []))

    # 5. Return result - SAME STRUCTURE as backward decomposition
    return {
        # Core identification (matches backward decomposition EXACTLY)
        "issue_id": issue_id,
        "repo": f"{repo_owner}/{repo_name}",
        "workflow": workflow_path,
        "workflow_path": workflow_path,
        "problems": problem_sequence,  # ← Changed to "problems" to match backward decomposition
        # Metadata (not saved to decomposed_issues.json, only for L1/L2/L3 building)
        "sha_fail": sha_fail,
        "sha_success": sha_success,
        "repo_owner": repo_owner,
        "changed_files": list(all_changed_files),
        # NOTE: No diff field - structured problems contain all necessary information
        "commit_link": issue.get("commit_link", ""),
        "decomposition_type": "commit_based",
        "total_commits": len(commits),
        "total_problems": len(problem_sequence),
        # Internal data for L1 building (not in final output)
        "_changed_files": list(
            all_changed_files
        ),  # Prefixed with _ to mark as internal
        "_dependencies": {},  # Internal data for dependency analysis
    }
