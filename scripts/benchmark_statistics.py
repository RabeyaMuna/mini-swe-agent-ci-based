#!/usr/bin/env python3
"""
benchmark_statistics.py - Generate Benchmark Characterization Numbers
=====================================================================

Task #1: Benchmark Characterization (TODO #3)
Generates statistics to prove this benchmark is different from SWE-bench:
- Number of commits per PR
- Number of modified files per PR
- Number of changed lines (additions + deletions)
- Number/types of CI failure categories
- Distinct problems per PR

Output:
- statistics/benchmark_stats.json - Raw statistics
- statistics/benchmark_comparison.md - Formatted comparison table
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import git

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def count_commits_in_pr(repo_path: Path, sha_fail: str, sha_success: str) -> int:
    """Count commits between failure and success SHA."""
    try:
        repo = git.Repo(repo_path)
        # Get commits between success (base) and fail (head)
        commits = list(repo.iter_commits(f"{sha_success}..{sha_fail}"))
        return len(commits)
    except Exception as e:
        print(f"  Warning: Could not count commits for {sha_fail}: {e}")
        return 0


def parse_diff_stats(diff: str) -> Dict[str, int]:
    """Parse diff to get file count and line changes."""
    if not diff:
        return {"files": 0, "additions": 0, "deletions": 0, "total_lines": 0}

    files = set()
    additions = 0
    deletions = 0

    for line in diff.split("\n"):
        # Track files
        if line.startswith("diff --git"):
            # Extract file path: diff --git a/path/file.py b/path/file.py
            match = re.search(r"b/(.+)$", line)
            if match:
                files.add(match.group(1))

        # Count additions
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1

        # Count deletions
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1

    return {
        "files": len(files),
        "additions": additions,
        "deletions": deletions,
        "total_lines": additions + deletions,
    }


def collect_benchmark_statistics():
    """Collect statistics from eval_set.jsonl and decomposed_issues.json."""

    # Load eval set
    eval_path = PROJECT_ROOT / "data" / "trs" / "eval_set.jsonl"
    decomposed_path = PROJECT_ROOT / "data" / "trs" / "decomposed_issues.json"
    repo_base = PROJECT_ROOT / "repo"

    print("Loading eval_set.jsonl...")
    eval_issues = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                eval_issues.append(json.loads(line))

    print(f"Loaded {len(eval_issues)} eval issues")

    # Load decomposed issues
    print("\nLoading decomposed_issues.json...")
    decomposed_map = {}
    if decomposed_path.exists():
        decomposed_issues = json.load(open(decomposed_path))
        decomposed_map = {item["sha_fail"]: item for item in decomposed_issues}
        print(f"Loaded {len(decomposed_map)} decomposed issues")

    # Collect statistics
    stats = {
        "total_prs": len(eval_issues),
        "commits_per_pr": [],
        "files_per_pr": [],
        "lines_per_pr": [],
        "additions_per_pr": [],
        "deletions_per_pr": [],
        "ci_failure_types": [],
        "problems_per_pr": [],
        "per_issue": [],
    }

    print("\nCollecting statistics...")
    for i, issue in enumerate(eval_issues, 1):
        print(
            f"Processing {i}/{len(eval_issues)}: {issue.get('id', 'unknown')}", end="\r"
        )

        sha_fail = issue.get("sha_fail", "")
        sha_success = issue.get("sha_success", "")
        diff = issue.get("diff", "")

        # Count commits
        repo_owner = issue.get("repo_owner", "")
        repo_name = issue.get("repo_name", "")
        repo_path = repo_base / issue.get("id", "") / repo_name

        if not repo_path.exists():
            repo_path = repo_base / f"{repo_owner}__{repo_name}"

        num_commits = 0
        if repo_path.exists() and sha_fail and sha_success:
            num_commits = count_commits_in_pr(repo_path, sha_fail, sha_success)

        # Parse diff
        diff_stats = parse_diff_stats(diff)

        # Get CI failure types
        failure_types = issue.get("overall_error_types", [])
        if isinstance(failure_types, str):
            failure_types = [failure_types]

        # Get decomposed problems count
        num_problems = 0
        if sha_fail in decomposed_map:
            num_problems = decomposed_map[sha_fail].get("total_problems", 0)

        # Store per-issue data
        issue_data = {
            "id": issue.get("id", ""),
            "sha_fail": sha_fail,
            "commits": num_commits,
            "files": diff_stats["files"],
            "additions": diff_stats["additions"],
            "deletions": diff_stats["deletions"],
            "total_lines": diff_stats["total_lines"],
            "failure_types": failure_types,
            "num_problems": num_problems,
        }
        stats["per_issue"].append(issue_data)

        # Aggregate statistics
        if num_commits > 0:
            stats["commits_per_pr"].append(num_commits)
        if diff_stats["files"] > 0:
            stats["files_per_pr"].append(diff_stats["files"])
        if diff_stats["total_lines"] > 0:
            stats["lines_per_pr"].append(diff_stats["total_lines"])
            stats["additions_per_pr"].append(diff_stats["additions"])
            stats["deletions_per_pr"].append(diff_stats["deletions"])

        stats["ci_failure_types"].extend(failure_types)

        if num_problems > 0:
            stats["problems_per_pr"].append(num_problems)

    print("\nDone collecting statistics!")
    return stats


def compute_summary(stats: Dict) -> Dict:
    """Compute summary statistics (mean, median, min, max)."""

    def summarize(values: List[int], name: str) -> Dict:
        if not values:
            return {
                "name": name,
                "mean": 0,
                "median": 0,
                "min": 0,
                "max": 0,
                "total": 0,
            }

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "name": name,
            "mean": sum(values) / n,
            "median": sorted_vals[n // 2],
            "min": min(values),
            "max": max(values),
            "total": n,
        }

    # Count failure type frequencies
    failure_type_counter = Counter(stats["ci_failure_types"])

    return {
        "total_prs": stats["total_prs"],
        "commits": summarize(stats["commits_per_pr"], "Commits per PR"),
        "files": summarize(stats["files_per_pr"], "Files per PR"),
        "lines": summarize(stats["lines_per_pr"], "Lines changed per PR"),
        "additions": summarize(stats["additions_per_pr"], "Additions per PR"),
        "deletions": summarize(stats["deletions_per_pr"], "Deletions per PR"),
        "problems": summarize(stats["problems_per_pr"], "Problems per PR"),
        "failure_types": {
            "unique": len(failure_type_counter),
            "distribution": dict(failure_type_counter.most_common()),
        },
    }


def format_comparison_table(summary: Dict) -> str:
    """Format summary as markdown comparison table."""

    md = "# Benchmark Characterization: CI-REPAIR-BENCH vs SWE-bench\n\n"
    md += "## Overview\n\n"
    md += f"**Total Pull Requests:** {summary['total_prs']}\n\n"

    md += "## Key Statistics\n\n"
    md += "| Metric | Mean | Median | Min | Max | Count |\n"
    md += "|--------|------|--------|-----|-----|-------|\n"

    for key in ["commits", "files", "lines", "additions", "deletions", "problems"]:
        s = summary[key]
        md += f"| {s['name']} | {s['mean']:.2f} | {s['median']} | {s['min']} | {s['max']} | {s['total']} |\n"

    md += "\n## CI Failure Type Distribution\n\n"
    md += f"**Unique failure types:** {summary['failure_types']['unique']}\n\n"
    md += "| Failure Type | Count |\n"
    md += "|--------------|-------|\n"

    for failure_type, count in sorted(
        summary["failure_types"]["distribution"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        md += f"| {failure_type} | {count} |\n"

    md += "\n## Why This Is Different From SWE-bench\n\n"
    md += "1. **Multiple Commits per PR**: "
    if summary["commits"]["mean"] > 1:
        md += f"Average {summary['commits']['mean']:.1f} commits per PR (SWE-bench: typically 1)\n"
    else:
        md += "Most PRs have single commits, but some have multiple\n"

    md += f"2. **Multiple Files Changed**: Average {summary['files']['mean']:.1f} files per PR\n"
    md += f"3. **Diverse Failure Types**: {summary['failure_types']['unique']} unique CI failure categories\n"

    if summary["problems"]["mean"] > 0:
        md += f"4. **Multiple Problems per PR**: Average {summary['problems']['mean']:.1f} distinct problems per PR\n"

    md += "5. **Real CI Pipeline**: Multi-stage validation (formatting, linting, type checking, tests)\n"

    return md


def main():
    """Main entry point."""
    print("=" * 70)
    print("Benchmark Statistics Generator")
    print("=" * 70)

    # Collect statistics
    stats = collect_benchmark_statistics()

    # Compute summary
    summary = compute_summary(stats)

    # Save outputs
    output_dir = PROJECT_ROOT / "statistics"
    output_dir.mkdir(exist_ok=True)

    # Save raw stats
    stats_file = output_dir / "benchmark_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nOK Saved raw statistics to {stats_file}")

    # Save summary
    summary_file = output_dir / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"OK Saved summary to {summary_file}")

    # Generate comparison table
    comparison = format_comparison_table(summary)
    comparison_file = output_dir / "benchmark_comparison.md"
    with open(comparison_file, "w") as f:
        f.write(comparison)
    print(f"OK Saved comparison table to {comparison_file}")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total PRs: {summary['total_prs']}")
    print(f"Avg Commits: {summary['commits']['mean']:.2f}")
    print(f"Avg Files: {summary['files']['mean']:.2f}")
    print(f"Avg Lines: {summary['lines']['mean']:.2f}")
    print(f"Avg Problems: {summary['problems']['mean']:.2f}")
    print(f"Unique Failure Types: {summary['failure_types']['unique']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
