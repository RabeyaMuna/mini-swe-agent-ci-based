#!/usr/bin/env python3
"""
commit_based_decomposer.py - Simple commit-level decomposition

Simplified production-ready version:
1. Fetch diff for each commit
2. Use cached CI validation data (survives 90 days)
3. LLM decides what's relevant (no pre-filtering)
4. Final LLM organization step
"""

import json
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
from commit_decomposition.problem_deduplicator import deduplicate_problems
from commit_decomposition.ci_command_resolver import enrich_ci_metadata_commands
from commit_decomposition.diff_filter import (
    filter_diff,
    filter_diff_to_files,
    get_changed_files,
    get_filtered_file_count,
)
from commit_decomposition.diff_chunker import chunk_commit_diff


def load_structured_ci_failure(sha_fail: str, issue_id: str = "") -> Dict:
    """Load structured CI failure details from data/log_details.json."""
    log_details_path = PROJECT_ROOT / "data" / "log_details.json"
    if not log_details_path.exists():
        return {}

    try:
        with open(log_details_path) as f:
            log_details = json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load log_details.json: {e}")
        return {}

    if not isinstance(log_details, list):
        return {}

    for entry in log_details:
        if not isinstance(entry, dict):
            continue
        if sha_fail and str(entry.get("sha_fail") or "") == str(sha_fail):
            return entry
        if issue_id and str(entry.get("id") or "") == str(issue_id):
            return entry

    return {}


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
        name = Path(path).name
        if path.startswith(".github/") or name.endswith(".json"):
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
    """
    Generate validation sequence using ci_workflow_aware_retrieval.py and cache it

    Args:
        issue_id: Issue ID
        issue_data: Issue data with repo info
        github_fetcher: GitHubFetcher instance
        analyzer: CommitAnalyzer instance (has LLM)

    Returns:
        Dict with validation_sequence, failure_info, workflow_path
    """
    from ci_workflow_aware_retrieval import (
        build_dependent_file_prompt,
        build_validation_sequence_prompt,
        _load_json,
        _normalize_dependent_files,
        _normalize_validation_sequence,
    )

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

    # Fetch workflow content from sha_fail
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

    # Generate validation sequence using LLM
    try:
        # Step 1: Ask LLM which dependent files are needed
        import litellm

        # Create a simple LLM caller that works with litellm
        class SimpleLLM:
            def __init__(self, model_name):
                self.model_name = model_name

            def invoke(self, prompt):
                response = litellm.completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return response.choices[0].message.content

        simple_llm = SimpleLLM(analyzer.model_name)

        # Get dependent files
        dependent_prompt = build_dependent_file_prompt(workflow_path, workflow_content)
        dependent_raw = simple_llm.invoke(dependent_prompt)
        dependent_files = _normalize_dependent_files(
            _load_json(dependent_raw, {"dependent_files": []})
        )

        print(f"    Found {len(dependent_files)} dependent files")

        # Step 2: Fetch dependent file contents from GitHub
        dependent_file_contents = []
        for dep in dependent_files:
            dep_path = dep["path"]
            content = github_fetcher.get_file_content(
                repo_owner, repo_name, dep_path, sha_fail
            )
            found = bool(content)
            status = "✓" if found else "✗"
            print(f"      {status} {dep_path}")

            dependent_file_contents.append(
                {
                    "path": dep_path,
                    "reason": dep.get("reason", ""),
                    "found": found,
                    "content": content or "",
                }
            )

        # Step 3: Generate validation sequence using prompts directly
        validation_prompt = build_validation_sequence_prompt(
            workflow_path, workflow_content, dependent_file_contents
        )
        validation_raw = simple_llm.invoke(validation_prompt)

        print("    [DEBUG] Validation LLM response (first 500 chars):")
        print(f"    {str(validation_raw)[:500]}")

        validation_sequence = _normalize_validation_sequence(
            _load_json(validation_raw, [])
        )
        print(f"    Generated {len(validation_sequence)} validation steps")

        # Save to cache
        cache_path = PROJECT_ROOT / "data" / "workflow_validation_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cache = json.load(f)
        else:
            cache = []

        # Add new entry
        cache_entry = {
            "issue_id": issue_id,
            "id": issue_id,
            "sha_fail": sha_fail,
            "workflow_path": workflow_path,
            "validation_sequence": validation_sequence,
            "failure_info": "",
            "generated_at": str(Path(__file__).stat().st_mtime),
        }

        # Remove old entry if exists
        cache = [
            c for c in cache if str(c.get("issue_id") or c.get("id")) != str(issue_id)
        ]
        cache.append(cache_entry)

        # Save
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

        print(f"    Saved validation sequence to cache: {cache_path}")

        return {
            "validation_sequence": validation_sequence,
            "failure_info": "",
            "workflow_path": workflow_path,
        }

    except Exception as e:
        print(f"    Error generating validation sequence: {e}")
        import traceback

        traceback.print_exc()
        return {
            "validation_sequence": [],
            "failure_info": "",
            "workflow_path": workflow_path,
        }


def load_validation_cache(
    issue_id: str, issue_data: Dict = None, github_fetcher=None, analyzer=None
) -> Dict:
    """
    Load validation data from cache or generate and save it

    Returns:
        validation_sequence: List of CI validation steps
        failure_info: What failed (cached, survives log expiry)
    """
    cache_path = PROJECT_ROOT / "data" / "workflow_validation_cache.json"
    issue_data = issue_data or {}
    sha_fail = str(issue_data.get("sha_fail") or "")

    # Try cache first
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

        # Find entry for this issue
        for entry in cache:
            entry_issue_id = str(entry.get("issue_id") or entry.get("id") or "")
            entry_sha_fail = str(entry.get("sha_fail") or "")
            if entry_issue_id == str(issue_id) or (
                sha_fail and entry_sha_fail == sha_fail
            ):
                print(
                    f"  Found validation sequence in cache: {len(entry.get('validation_sequence', []))} steps"
                )
                return {
                    "validation_sequence": entry.get("validation_sequence", []),
                    "failure_info": (
                        entry.get("failure_info") or entry.get("failure_summary", "")
                    ),
                    "workflow_path": entry.get("workflow_path", ""),
                }

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

    print(f"\n  Processing issue {issue_id}")
    print(f"    Repository: {repo_owner}/{repo_name}")
    print(f"    sha_success: {sha_success[:12]}")
    print(f"    sha_fail: {sha_fail[:12]}")

    # 1. Get all commits from GitHub
    commits = get_commits_between(
        repo_owner, repo_name, sha_success, sha_fail, github_fetcher
    )
    print(f"    Total commits: {len(commits)}")

    if not commits:
        return {
            "issue_id": issue_id,
            "error": "No commits found",
            "problem_sequence": [],
        }

    # 2. Analyze each commit
    all_problems = []
    validation_sequence = validation_cache.get("validation_sequence", [])
    workflow_path = validation_cache.get("workflow_path", "")
    ci_failure_info = validation_cache.get("failure_info", "")
    structured_failure = load_structured_ci_failure(sha_fail, issue_id)
    if structured_failure:
        print("    Using structured CI failure info from data/log_details.json")
    primary_failure_files = structured_failure_files(structured_failure)

    # Flag to track if validation_sequence needs to be built from first commit's CI metadata
    needs_validation_from_metadata = not validation_sequence

    # Use cached failure info if logs are missing (90-day expiry)
    if structured_failure:
        ci_logs = structured_failure
    elif not issue.get("logs") or len(str(issue.get("logs", ""))) < 100:
        print("    Using cached CI failure info (logs expired or missing)")
        ci_logs = ci_failure_info
    else:
        ci_logs = issue.get("logs", "")

    for i, commit in enumerate(commits, 1):
        commit_sha = commit["sha"]
        print(f"    Analyzing commit {i}/{len(commits)}: {commit_sha[:8]}")

        # First fetch commit diff and changed-file information.
        full_diff = get_commit_diff(repo_owner, repo_name, commit_sha, github_fetcher)
        changed_files = get_changed_files(full_diff, include_ignored=True)
        validated_changed_files = get_changed_files(full_diff, include_ignored=False)

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

        selected = analyzer.select_relevant_files(
            commit_metadata=commit_analysis["commit_metadata"],
            changed_files=changed_files,
            commit_diff=full_diff,
            structured_ci_failure=structured_failure,
            ci_metadata=compact_metadata,
            relevant_validations=validation_sequence,
        )
        selected_files = selected.get("selected_files", [])
        candidate_files = validation_candidate_files(
            changed_files, validation_sequence, structured_failure
        )
        selected_files = merge_file_lists(selected_files, candidate_files)
        if selected_files:
            commit_analysis["selected_files"] = selected_files
            commit_analysis["selection_reasoning"] = selected.get("reasoning", "")
            print(
                f"      Selected {len(selected_files)} relevant file(s) "
                f"({len(selected.get('selected_files', []))} LLM, "
                f"{len(candidate_files)} non-ignored candidates)"
            )

        # Filter out irrelevant files (.github/workflows, CI configs, etc.)
        total_files, included_files, ignored_files = get_filtered_file_count(full_diff)
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
            filtered_diff = filter_diff(full_diff)
            included_files = len(get_changed_files(filtered_diff, include_ignored=True))
            ignored_files = total_files - included_files

        if ignored_files > 0:
            print(
                f"      Filtered: {total_files} files → {included_files} files (ignored {ignored_files} .github/.json files)"
            )

        if not filtered_diff or filtered_diff.strip() == "":
            print("      No relevant files after filtering, skipping")
            continue

        # Check if we need to chunk this commit (if too large)
        # Chunk by files if needed, but keep context that this is ONE commit
        chunks = chunk_commit_diff(
            filtered_diff, max_tokens=12000, max_files_per_chunk=40
        )

        if len(chunks) > 1:
            print(f"      Large commit - split into {len(chunks)} chunks for analysis")
        else:
            print("      Analyzing commit changes...")

        # Analyze each chunk of this commit
        commit_problems = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_files = chunk.get("files", [])
            file_count = chunk.get("file_count", len(chunk_files))

            if len(chunks) > 1:
                print(
                    f"        Chunk {chunk_idx + 1}/{len(chunks)}: {file_count} file(s)"
                )

            # LLM analysis on this chunk
            analysis_result = analyzer.analyze_commit_group(
                group={
                    "commits": [commit],
                    "type": "single_commit",
                    "commit_number": i,
                    "total_commits": len(commits),
                    "chunk_number": chunk_idx + 1,
                    "total_chunks": len(chunks),
                    "files_in_chunk": chunk_files,
                    "changed_files": changed_files,
                    "validated_changed_files": validated_changed_files,
                    "sha_success": sha_success,
                    "sha_fail": sha_fail,
                    "ci_metadata": compact_metadata,
                    "structured_ci_failure": structured_failure,
                },
                commit_diff=chunk["diff"],
                ci_logs=ci_logs,
                relevant_validations=validation_sequence,
            )

            # Collect problems from this chunk
            for problem in analysis_result.get("problems", []):
                problem["commit"] = commit_sha[:8]
                problem["commit_sha"] = problem.get("commit_sha") or commit_sha
                problem["commit_message"] = problem.get("commit_message") or commit.get(
                    "message", ""
                )
                problem["commit_number"] = i
                commit_problems.append(problem)

        # Add all problems from this commit
        commit_analysis["problems"] = commit_problems
        all_problems.extend(commit_problems)

    print(f"    Total problems found: {len(all_problems)}")

    # 3. Deduplicate similar problems
    if len(all_problems) > 1:
        deduplicated = deduplicate_problems(
            all_problems, analyzer, similarity_threshold=0.7
        )
        all_problems = deduplicated

    # 4. Final LLM organization (if we have problems)
    if all_problems:
        print("    Organizing repair trajectory...")
        organized = organize_trajectory_with_llm(
            all_problems=all_problems,
            validation_sequence=validation_sequence,
            analyzer=analyzer,
        )
        problem_sequence = organized
    else:
        problem_sequence = all_problems

    # 4. Return result
    return {
        "issue_id": issue_id,
        "sha_fail": sha_fail,
        "sha_success": sha_success,
        "decomposition_type": "commit_based",
        "total_commits": len(commits),
        "total_problems": len(problem_sequence),
        "problem_sequence": problem_sequence,
    }
