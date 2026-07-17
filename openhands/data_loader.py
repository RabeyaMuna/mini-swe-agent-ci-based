"""
Data loader for CI-Bench evaluation
Handles caching of CI logs and workflow data
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import requests
from tqdm import tqdm


class CIBenchDataLoader:
    """Load and cache CI-Bench data"""

    def __init__(self, data_root: str = "../data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.log_details_path = self.data_root / "log_details.json"
        self.workflow_cache_path = self.data_root / "workflow_validation_cache.json"

    def load_eval_issues(self, eval_issues_path: str) -> List[Dict[str, Any]]:
        """Load evaluation issues from JSONL"""
        issues = []
        with open(eval_issues_path, 'r') as f:
            for line in f:
                if line.strip():
                    issues.append(json.loads(line))
        print(f"✓ Loaded {len(issues)} issues from {eval_issues_path}")
        return issues

    def ensure_data_files(self, eval_issues: List[Dict[str, Any]]) -> tuple:
        """
        Ensure log_details.json and workflow_cache.json exist
        Load if exists, generate if not
        """
        # Load or generate log_details
        if self.log_details_path.exists():
            print(f"✓ Loading cached CI logs from {self.log_details_path}")
            with open(self.log_details_path, 'r') as f:
                log_details = json.load(f)
        else:
            print(f"⚠️  {self.log_details_path} not found, generating...")
            log_details = self._fetch_all_ci_logs(eval_issues)
            with open(self.log_details_path, 'w') as f:
                json.dump(log_details, f, indent=2)
            print(f"✓ Saved CI logs to {self.log_details_path}")

        # Load or generate workflow_cache
        if self.workflow_cache_path.exists():
            print(f"✓ Loading cached workflow data from {self.workflow_cache_path}")
            with open(self.workflow_cache_path, 'r') as f:
                workflow_cache = json.load(f)
        else:
            print(f"⚠️  {self.workflow_cache_path} not found, generating...")
            workflow_cache = self._fetch_all_workflows(eval_issues)
            with open(self.workflow_cache_path, 'w') as f:
                json.dump(workflow_cache, f, indent=2)
            print(f"✓ Saved workflow data to {self.workflow_cache_path}")

        return log_details, workflow_cache

    def _fetch_all_ci_logs(self, eval_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fetch CI logs for all issues
        This is a placeholder - implement based on your data source
        """
        log_details = {}

        print("Fetching CI logs for all issues...")
        for issue in tqdm(eval_issues):
            instance_id = issue["instance_id"]

            # TODO: Implement actual CI log fetching
            # For now, use problem_statement as placeholder
            log_details[instance_id] = {
                "ci_logs": issue.get("problem_statement", ""),
                "error_summary": "CI failure detected",
                "failed_tests": []
            }

        return log_details

    def _fetch_all_workflows(self, eval_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fetch workflow validation info for all issues
        This is a placeholder - implement based on your data source
        """
        workflow_cache = {}

        print("Fetching workflow data for all issues...")
        for issue in tqdm(eval_issues):
            instance_id = issue["instance_id"]

            # TODO: Implement actual workflow fetching
            # For now, use validation_command as placeholder
            workflow_cache[instance_id] = {
                "validation_command": issue.get("validation_command", "pytest"),
                "workflow_name": "CI",
                "stages": ["lint", "test", "build"]
            }

        return workflow_cache

    def get_issue_data(
        self,
        issue: Dict[str, Any],
        log_details: Dict[str, Any],
        workflow_cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine all data for a single issue"""
        instance_id = issue["instance_id"]

        return {
            "instance_id": instance_id,
            "repo": issue["repo"],
            "sha_fail": issue["sha_fail"],
            "base_sha": issue.get("base_sha", "main"),
            "problem_statement": issue["problem_statement"],
            "ci_logs": log_details.get(instance_id, {}).get("ci_logs", ""),
            "error_summary": log_details.get(instance_id, {}).get("error_summary", ""),
            "workflow": workflow_cache.get(instance_id, {}),
            "validation_command": workflow_cache.get(instance_id, {}).get("validation_command", "pytest")
        }


if __name__ == "__main__":
    # Test data loader
    loader = CIBenchDataLoader(data_root="../data")

    # Load issues
    issues = loader.load_eval_issues("../data/trs/eval_set.jsonl")

    # Ensure data files exist
    log_details, workflow_cache = loader.ensure_data_files(issues)

    # Get data for first issue
    if issues:
        issue_data = loader.get_issue_data(issues[0], log_details, workflow_cache)
        print("\nSample issue data:")
        print(json.dumps(issue_data, indent=2))
