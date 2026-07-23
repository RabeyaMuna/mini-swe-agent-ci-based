#!/usr/bin/env python3
"""
github_fetcher.py - Fetch commits and diffs directly from GitHub API

No local repos needed - everything from GitHub!
"""

import os
from typing import Dict, List, Optional
import requests


class GitHubFetcher:
    """Fetch commits and diffs from GitHub API"""

    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        self.headers = {}
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"
        self.base_url = "https://api.github.com"
        self._file_cache = {}
        self._job_log_cache = {}

    def get_commits_between(
        self, repo_owner: str, repo_name: str, sha_success: str, sha_fail: str
    ) -> List[Dict]:
        """
        Fetch commits between sha_fail and sha_success using GitHub API

        This gets commits that FIXED the failure (from failing state to passing state)

        Args:
            repo_owner: Repository owner (e.g., "adap")
            repo_name: Repository name (e.g., "flower")
            sha_success: Commit SHA where CI passes
            sha_fail: Commit SHA where CI fails

        Returns:
            List of commits from sha_fail to sha_success (in chronological order)
        """
        # GitHub API: compare sha_fail (base) to sha_success (head)
        # This shows commits that move FROM failure TO success
        url = f"{self.base_url}/repos/{repo_owner}/{repo_name}/compare/{sha_fail}...{sha_success}"

        try:
            print(f"    Fetching from: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 404:
                print("    Warning: Repository or commits not found (404)")
                print(f"    Response: {response.text[:200]}")
                return []

            if response.status_code == 403:
                print("    Warning: Rate limited or forbidden - need GITHUB_TOKEN")
                return []

            if response.status_code != 200:
                print(f"    Warning: HTTP {response.status_code}")
                print(f"    Response: {response.text[:200]}")
                return []

            response.raise_for_status()
            data = response.json()

            # Debug
            status = data.get("status", "unknown")
            total_commits = data.get("total_commits", 0)
            print(
                f"    GitHub compare status: {status}, total_commits: {total_commits}"
            )

            commits = data.get("commits", [])

            # If status is "behind", it means sha_success comes BEFORE sha_fail chronologically
            # which means we're going backwards (from passing to failing)
            # We want forward direction (from failing to passing), so swap
            if status == "behind":
                print(
                    f"    Warning: '{status}' status means sha_success is older than sha_fail"
                )
                print(
                    "    This would show commits that INTRODUCED the failure, not fixed it"
                )
                print("    Swapping to get commits that FIXED the failure...")
                url_reversed = f"{self.base_url}/repos/{repo_owner}/{repo_name}/compare/{sha_success}...{sha_fail}"
                print(f"    Fetching from: {url_reversed}")
                response = requests.get(url_reversed, headers=self.headers, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    total_commits = data.get("total_commits", 0)
                    commits = data.get("commits", [])
                    print(
                        f"    After swap - status: {status}, total_commits: {total_commits}"
                    )

            # Format commits
            formatted_commits = []
            for commit in commits:
                formatted_commits.append(
                    {
                        "sha": commit["sha"],
                        "message": commit["commit"]["message"],
                        "author": commit["commit"]["author"]["name"],
                        "date": commit["commit"]["author"]["date"],
                        "html_url": commit["html_url"],
                    }
                )

            return formatted_commits

        except requests.exceptions.RequestException as e:
            print(f"    Error fetching commits: {e}")
            return []

    def get_commit_diff(self, repo_owner: str, repo_name: str, commit_sha: str) -> str:
        """
        Fetch diff for a specific commit from GitHub

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            commit_sha: Commit SHA

        Returns:
            Diff as string
        """
        url = f"{self.base_url}/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"

        try:
            # Request diff format
            headers = self.headers.copy()
            headers["Accept"] = "application/vnd.github.v3.diff"

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 404:
                return f"Error: Commit {commit_sha} not found"

            if response.status_code == 403:
                return "Error: Rate limited - need GITHUB_TOKEN"

            response.raise_for_status()

            # Response is already in diff format
            return response.text

        except requests.exceptions.RequestException as e:
            return f"Error fetching diff: {e}"

    def get_commit_ci_status(
        self, repo_owner: str, repo_name: str, commit_sha: str
    ) -> Dict:
        """
        Fetch CI check runs and status for a specific commit

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            commit_sha: Commit SHA

        Returns:
            Dict with check_runs and overall status
        """
        # Fetch check runs (GitHub Actions, etc.)
        url = f"{self.base_url}/repos/{repo_owner}/{repo_name}/commits/{commit_sha}/check-runs"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code != 200:
                return {"check_runs": [], "total_count": 0}

            data = response.json()

            check_runs = []
            for check in data.get("check_runs", []):
                check_runs.append(
                    {
                        "name": check.get("name", ""),
                        "status": check.get(
                            "status", ""
                        ),  # queued, in_progress, completed
                        "conclusion": check.get(
                            "conclusion"
                        ),  # success, failure, neutral, cancelled, skipped, timed_out
                        "started_at": check.get("started_at"),
                        "completed_at": check.get("completed_at"),
                        "html_url": check.get("html_url", ""),
                    }
                )

            return {"check_runs": check_runs, "total_count": data.get("total_count", 0)}

        except requests.exceptions.RequestException as e:
            print(f"    Warning: Could not fetch CI status: {e}")
            return {"check_runs": [], "total_count": 0}

    def _get_paginated(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Fetch a paginated GitHub API collection."""
        items = []
        page = 1
        params = dict(params or {})
        params.setdefault("per_page", 100)

        while True:
            params["page"] = page
            response = requests.get(
                url, headers=self.headers, params=params, timeout=30
            )
            if response.status_code != 200:
                return {"items": items, "status_code": response.status_code}

            data = response.json()
            page_items = (
                data.get("workflow_runs")
                or data.get("jobs")
                or data.get("check_runs")
                or []
            )
            items.extend(page_items)

            if len(page_items) < params["per_page"]:
                break
            page += 1

        return {"items": items, "status_code": 200}

    def get_workflow_runs_for_commit(
        self, repo_owner: str, repo_name: str, commit_sha: str
    ) -> List[Dict]:
        """Fetch GitHub Actions workflow runs for a commit SHA."""
        url = f"{self.base_url}/repos/{repo_owner}/{repo_name}/actions/runs"
        result = self._get_paginated(url, params={"head_sha": commit_sha})
        if result.get("status_code") != 200:
            return []

        workflow_runs = []
        for run in result.get("items", []):
            workflow_runs.append(
                {
                    "run_id": run.get("id"),
                    "name": run.get("name", ""),
                    "workflow_name": run.get("name", ""),
                    "display_title": run.get("display_title", ""),
                    "workflow_id": run.get("workflow_id"),
                    "path": run.get("path", ""),
                    "head_sha": run.get("head_sha"),
                    "head_branch": run.get("head_branch"),
                    "event": run.get("event"),
                    "status": run.get("status", ""),
                    "conclusion": run.get("conclusion"),
                    "created_at": run.get("created_at"),
                    "updated_at": run.get("updated_at"),
                    "html_url": run.get("html_url", ""),
                }
            )

        return workflow_runs

    def get_workflow_jobs_for_run(
        self, repo_owner: str, repo_name: str, run_id: int
    ) -> List[Dict]:
        """Fetch jobs and step metadata for a workflow run."""
        url = (
            f"{self.base_url}/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}/jobs"
        )
        result = self._get_paginated(url, params={"filter": "all"})
        if result.get("status_code") != 200:
            return []

        jobs = []
        for job in result.get("items", []):
            steps = []
            for step in job.get("steps", []) or []:
                steps.append(
                    {
                        "number": step.get("number"),
                        "name": step.get("name", ""),
                        "status": step.get("status", ""),
                        "conclusion": step.get("conclusion"),
                    }
                )

            jobs.append(
                {
                    "job_id": job.get("id"),
                    "run_id": run_id,
                    "name": job.get("name", ""),
                    "workflow_name": job.get("workflow_name", ""),
                    "status": job.get("status", ""),
                    "conclusion": job.get("conclusion"),
                    "html_url": job.get("html_url", ""),
                    "steps": steps,
                }
            )

        return jobs

    def get_commit_ci_metadata(
        self, repo_owner: str, repo_name: str, commit_sha: str
    ) -> Dict:
        """Fetch compact workflow/job/step metadata for one commit."""
        workflow_runs = self.get_workflow_runs_for_commit(
            repo_owner, repo_name, commit_sha
        )
        current_failed_jobs = []
        current_jobs_fixed = []
        jobs_executed = []
        job_conclusions = []
        step_names_executed = []
        failed_jobs = []
        failed_steps = []

        for run in workflow_runs:
            run_id = run.get("run_id")
            jobs = (
                self.get_workflow_jobs_for_run(repo_owner, repo_name, run_id)
                if run_id
                else []
            )
            run["jobs"] = jobs

            for job in jobs:
                if job.get("status") or job.get("conclusion"):
                    jobs_executed.append(
                        {
                            "workflow": run.get("workflow_name") or run.get("name", ""),
                            "job": job.get("name", ""),
                            "status": job.get("status", ""),
                            "conclusion": job.get("conclusion"),
                        }
                    )
                    job_conclusions.append(
                        {
                            "workflow": run.get("workflow_name") or run.get("name", ""),
                            "job": job.get("name", ""),
                            "conclusion": job.get("conclusion"),
                        }
                    )

                executed_steps = [
                    step
                    for step in job.get("steps", [])
                    if step.get("status") or step.get("conclusion")
                ]
                for step in executed_steps:
                    step_names_executed.append(
                        {
                            "workflow": run.get("workflow_name") or run.get("name", ""),
                            "job": job.get("name", ""),
                            "number": step.get("number"),
                            "step": step.get("name", ""),
                            "status": step.get("status", ""),
                            "conclusion": step.get("conclusion"),
                        }
                    )

                if job.get("conclusion") == "failure":
                    failed_jobs.append(
                        {
                            "workflow": run.get("workflow_name") or run.get("name", ""),
                            "job": job.get("name", ""),
                            "status": job.get("status", ""),
                            "conclusion": job.get("conclusion"),
                            "html_url": job.get("html_url", ""),
                        }
                    )
                    failed_steps.extend(
                        [
                            {
                                "workflow": run.get("workflow_name")
                                or run.get("name", ""),
                                "job": job.get("name", ""),
                                "number": step.get("number"),
                                "step": step.get("name", ""),
                                "status": step.get("status", ""),
                                "conclusion": step.get("conclusion"),
                                "html_url": job.get("html_url", ""),
                                "job_id": job.get("id"),
                                "run_id": run_id,
                                "workflow_path": run.get("path", ""),
                            }
                            for step in job.get("steps", [])
                            if step.get("conclusion") == "failure"
                        ]
                    )
                    current_failed_jobs.extend(
                        [
                            {
                                "workflow": run.get("workflow_name")
                                or run.get("name", ""),
                                "job": job.get("name", ""),
                                "step": step.get("name", ""),
                                "number": step.get("number"),
                                "status": step.get("status", ""),
                                "conclusion": step.get("conclusion"),
                                "validation_cmd": "",
                                "command_source": "",
                                "html_url": job.get("html_url", ""),
                                "job_id": job.get("id"),
                                "run_id": run_id,
                                "workflow_path": run.get("path", ""),
                            }
                            for step in job.get("steps", [])
                            if step.get("conclusion") == "failure"
                        ]
                    )
                elif job.get("conclusion") == "success":
                    current_jobs_fixed.append(
                        {
                            "workflow": run.get("workflow_name") or run.get("name", ""),
                            "job": job.get("name", ""),
                            "step": "",
                            "validation_cmd": "",
                            "status": job.get("status", ""),
                            "conclusion": job.get("conclusion"),
                            "html_url": job.get("html_url", ""),
                            "job_id": job.get("id"),
                            "run_id": run_id,
                            "workflow_path": run.get("path", ""),
                        }
                    )

        return {
            "commit_sha": commit_sha,
            "workflow_run_exists": bool(workflow_runs),
            "workflow_names": sorted(
                {
                    str(run.get("workflow_name") or run.get("name") or "")
                    for run in workflow_runs
                    if run.get("workflow_name") or run.get("name")
                }
            ),
            "jobs_executed": jobs_executed,
            "job_conclusions": job_conclusions,
            "step_names_executed": step_names_executed,
            "failed_jobs": failed_jobs,
            "failed_steps": failed_steps,
            "workflow_runs": workflow_runs,
            "current_failed_jobs": current_failed_jobs,
            "current_jobs_fixed": current_jobs_fixed,
        }

    def get_file_content(
        self, repo_owner: str, repo_name: str, path: str, ref: str
    ) -> str:
        """Fetch raw file content from a repository at a specific ref."""
        if not path:
            return ""
        cache_key = (repo_owner, repo_name, path, ref)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        url = f"{self.base_url}/repos/{repo_owner}/{repo_name}/contents/{path.lstrip('/')}"
        headers = self.headers.copy()
        headers["Accept"] = "application/vnd.github.raw"

        try:
            response = requests.get(
                url, headers=headers, params={"ref": ref}, timeout=30
            )
            if response.status_code != 200:
                self._file_cache[cache_key] = ""
                return ""
            self._file_cache[cache_key] = response.text
            return response.text
        except requests.exceptions.RequestException:
            self._file_cache[cache_key] = ""
            return ""

    def get_job_log(self, repo_owner: str, repo_name: str, job_id: int) -> str:
        """Fetch raw log text for a GitHub Actions job."""
        if not job_id:
            return ""
        cache_key = (repo_owner, repo_name, job_id)
        if cache_key in self._job_log_cache:
            return self._job_log_cache[cache_key]

        url = (
            f"{self.base_url}/repos/{repo_owner}/{repo_name}/actions/jobs/{job_id}/logs"
        )
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code != 200:
                self._job_log_cache[cache_key] = ""
                return ""
            self._job_log_cache[cache_key] = response.text
            return response.text
        except requests.exceptions.RequestException:
            self._job_log_cache[cache_key] = ""
            return ""

    def check_rate_limit(self) -> Dict:
        """Check GitHub API rate limit"""
        url = f"{self.base_url}/rate_limit"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            core = data.get("rate", {})
            return {
                "limit": core.get("limit", 0),
                "remaining": core.get("remaining", 0),
                "reset": core.get("reset", 0),
            }

        except:
            return {"limit": 0, "remaining": 0, "reset": 0}
