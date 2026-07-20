#!/usr/bin/env python3
"""
commit_decomposition.py - Decompose PRs by Commits
==================================================

Task #2: Initial Decomposition Implementation (TODO #1)
Decomposes pull requests into individual commits when they are independent.

Strategy:
1. Extract each commit in the PR
2. Check if commits touch overlapping files
3. If independent → treat separately
4. If overlapping → group together

Output:
- data/trs/commit_decomposed_issues.json - Decomposed by commit
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import git

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_pr_commits(repo_path: Path, sha_fail: str, sha_success: str) -> List[Dict]:
    """Get all commits in PR with their changed files."""
    try:
        repo = git.Repo(repo_path)

        # Get commits between success (base) and fail (head)
        commits = list(repo.iter_commits(f"{sha_success}..{sha_fail}"))

        commit_data = []
        for commit in reversed(commits):  # Chronological order
            # Get files changed in this commit
            changed_files = []
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit)
                changed_files = [d.a_path or d.b_path for d in diffs]

            commit_data.append(
                {
                    "sha": commit.hexsha,
                    "message": commit.message.strip(),
                    "changed_files": changed_files,
                    "author": str(commit.author),
                    "date": commit.committed_datetime.isoformat(),
                }
            )

        return commit_data

    except Exception as e:
        print(f"  Warning: Could not get commits for {sha_fail}: {e}")
        return []


def extract_changed_files_from_diff(diff: str) -> Set[str]:
    """Extract changed files from unified diff."""
    files = set()
    for line in diff.split("\n"):
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                files.add(match.group(1))
    return files


def check_commit_overlap(commits: List[Dict]) -> List[List[int]]:
    """
    Group commits by file overlap.

    Returns list of commit groups (each group is list of commit indices).
    Independent commits get their own group, overlapping commits are grouped together.
    """
    if len(commits) <= 1:
        return [[0]] if commits else []

    # Build file → commit mapping
    file_to_commits = defaultdict(set)
    for i, commit in enumerate(commits):
        for file in commit["changed_files"]:
            file_to_commits[file].add(i)

    # Find connected components (commits that share files)
    visited = set()
    groups = []

    def dfs(idx: int, group: List[int]):
        """DFS to find all commits connected through file overlap."""
        if idx in visited:
            return
        visited.add(idx)
        group.append(idx)

        # Find all commits that share files with this commit
        for file in commits[idx]["changed_files"]:
            for other_idx in file_to_commits[file]:
                if other_idx not in visited:
                    dfs(other_idx, group)

    for i in range(len(commits)):
        if i not in visited:
            group = []
            dfs(i, group)
            groups.append(sorted(group))

    return groups


def decompose_issue_by_commits(issue: Dict, repo_path: Path) -> Dict:
    """
    Decompose a single issue by its commits.

    Returns decomposition info with commit groups.
    """
    sha_fail = issue.get("sha_fail", "")
    sha_success = issue.get("sha_success", "")

    # Get commits
    commits = get_pr_commits(repo_path, sha_fail, sha_success)

    if not commits:
        return {
            "decomposition_type": "no_commits",
            "num_commits": 0,
            "num_groups": 0,
            "groups": [],
        }

    # Find overlapping commit groups
    groups = check_commit_overlap(commits)

    # Classify decomposition
    if len(commits) == 1:
        decomposition_type = "single_commit"
    elif len(groups) == len(commits):
        decomposition_type = "independent_commits"
    elif len(groups) == 1:
        decomposition_type = "fully_overlapping"
    else:
        decomposition_type = "partial_overlap"

    # Build group details
    group_details = []
    for group_idx, commit_indices in enumerate(groups):
        group_commits = [commits[i] for i in commit_indices]

        # Collect all files touched by this group
        all_files = set()
        for commit in group_commits:
            all_files.update(commit["changed_files"])

        group_details.append(
            {
                "group_id": group_idx,
                "commit_count": len(group_commits),
                "commit_shas": [c["sha"] for c in group_commits],
                "commit_messages": [c["message"] for c in group_commits],
                "changed_files": sorted(all_files),
                "file_count": len(all_files),
            }
        )

    return {
        "decomposition_type": decomposition_type,
        "num_commits": len(commits),
        "num_groups": len(groups),
        "groups": group_details,
        "all_commits": commits,
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("Commit-Level Decomposition")
    print("=" * 70)

    # Load eval set
    eval_path = PROJECT_ROOT / "data" / "trs" / "eval_set.jsonl"
    repo_base = PROJECT_ROOT / "repo"

    print("\nLoading eval_set.jsonl...")
    eval_issues = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                eval_issues.append(json.loads(line))

    print(f"Loaded {len(eval_issues)} eval issues")

    # Decompose each issue
    decomposed = []
    stats = {
        "total": len(eval_issues),
        "single_commit": 0,
        "independent_commits": 0,
        "fully_overlapping": 0,
        "partial_overlap": 0,
        "no_commits": 0,
    }

    print("\nDecomposing issues by commits...")
    for i, issue in enumerate(eval_issues, 1):
        issue_id = issue.get("id", "unknown")
        print(f"Processing {i}/{len(eval_issues)}: {issue_id}", end="\r")

        # Find repo path
        repo_owner = issue.get("repo_owner", "")
        repo_name = issue.get("repo_name", "")
        repo_path = repo_base / issue_id / repo_name

        if not repo_path.exists():
            repo_path = repo_base / f"{repo_owner}__{repo_name}"

        # Decompose
        if repo_path.exists():
            decomposition = decompose_issue_by_commits(issue, repo_path)
        else:
            decomposition = {
                "decomposition_type": "no_repo",
                "num_commits": 0,
                "num_groups": 0,
                "groups": [],
            }

        # Update stats
        decomp_type = decomposition["decomposition_type"]
        if decomp_type in stats:
            stats[decomp_type] += 1

        # Store result
        decomposed.append(
            {
                "original_issue_id": issue_id,
                "sha_fail": issue.get("sha_fail", ""),
                "sha_success": issue.get("sha_success", ""),
                "repo": f"{repo_owner}/{repo_name}",
                "decomposition": decomposition,
            }
        )

    print("\nDone decomposing!")

    # Save output
    output_dir = PROJECT_ROOT / "data" / "trs"
    output_file = output_dir / "commit_decomposed_issues.json"

    with open(output_file, "w") as f:
        json.dump(decomposed, f, indent=2)

    print(f"\n✓ Saved commit decomposition to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Decomposition Summary:")
    print("=" * 70)
    print(f"Total issues: {stats['total']}")
    print(
        f"Single commit: {stats['single_commit']} ({stats['single_commit'] / stats['total'] * 100:.1f}%)"
    )
    print(
        f"Independent commits: {stats['independent_commits']} ({stats['independent_commits'] / stats['total'] * 100:.1f}%)"
    )
    print(
        f"Fully overlapping: {stats['fully_overlapping']} ({stats['fully_overlapping'] / stats['total'] * 100:.1f}%)"
    )
    print(
        f"Partial overlap: {stats['partial_overlap']} ({stats['partial_overlap'] / stats['total'] * 100:.1f}%)"
    )
    print(
        f"No commits: {stats['no_commits']} ({stats['no_commits'] / stats['total'] * 100:.1f}%)"
    )
    print("=" * 70)

    # Save stats
    stats_file = PROJECT_ROOT / "statistics" / "commit_decomposition_summary.json"
    stats_file.parent.mkdir(exist_ok=True)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved decomposition stats to {stats_file}")

    # Show example
    print("\n" + "=" * 70)
    print("Example Decomposition:")
    print("=" * 70)
    for item in decomposed[:3]:
        decomp = item["decomposition"]
        print(f"\nIssue: {item['original_issue_id']}")
        print(f"  Type: {decomp['decomposition_type']}")
        print(f"  Commits: {decomp['num_commits']}")
        print(f"  Groups: {decomp['num_groups']}")
        if decomp.get("groups"):
            for group in decomp["groups"][:2]:  # Show first 2 groups
                print(
                    f"    Group {group['group_id']}: {group['commit_count']} commit(s), {group['file_count']} file(s)"
                )


if __name__ == "__main__":
    main()
