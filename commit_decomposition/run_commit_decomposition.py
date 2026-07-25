#!/usr/bin/env python3
"""
run_commit_decomposition.py - Main entry point for commit-based decomposition

Usage:
    # Run on filtered_issues.jsonl
    python commit_decomposition/run_commit_decomposition.py

    # Run on specific issue
    python commit_decomposition/run_commit_decomposition.py --issue-id 113

    # Limit number of issues (for testing)
    python commit_decomposition/run_commit_decomposition.py --limit 5
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from commit_decomposition.commit_based_decomposer import (
    decompose_issue,
    load_validation_cache,
)
from commit_decomposition.commit_analyzer import CommitAnalyzer
from commit_decomposition.github_fetcher import GitHubFetcher
from utilities.model_registry import configure_model_environment


def load_issues_from_huggingface(issue_id: str = None) -> list:
    """Load issues directly from HuggingFace"""
    from datasets import load_dataset

    print("Loading dataset from HuggingFace: ci-benchmark-user/ci-repair-bench")

    try:
        ds = load_dataset("ci-benchmark-user/ci-repair-bench", split="train")
        print(f"  Loaded {len(ds)} issues from HuggingFace")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        print("  Ensure datasets library is up to date: pip install --upgrade datasets")
        raise

    issues = []
    for item in ds:
        issue_dict = dict(item)
        if issue_id is None or str(issue_dict.get("id")) == str(issue_id):
            issues.append(issue_dict)

    print(f"  Filtered to {len(issues)} issue(s)")
    return issues


def load_issues(dataset_path: Path, issue_id: str = None) -> list:
    """Load issues from JSONL file (fallback)"""
    issues = []

    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                issue = json.loads(line)
                if issue_id is None or str(issue.get("id")) == str(issue_id):
                    issues.append(issue)

    return issues


def main():
    parser = argparse.ArgumentParser(description="Commit-based decomposition")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "trs" / "filtered_issues.jsonl",
        help="Path to issues JSONL file (optional, will use HuggingFace if --use-huggingface)",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Load data directly from HuggingFace (always fresh)",
    )
    parser.add_argument("--issue-id", type=str, help="Process single issue by ID")
    parser.add_argument("--limit", type=int, help="Limit number of issues to process")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "commit_decomposed_issues.json",
        help="Output file path",
    )
    parser.add_argument(
        "--model", type=str, help="LLM model to use (default: from env MEMCI_LLM_MODEL)"
    )
    parser.add_argument("--repo-owner", type=str, help="Override issue repo owner")
    parser.add_argument("--repo-name", type=str, help="Override issue repo name")
    parser.add_argument("--sha-fail", type=str, help="Override issue failing SHA")
    parser.add_argument("--sha-success", type=str, help="Override issue passing SHA")

    args = parser.parse_args()

    print("=" * 70)
    print("Commit-Based Decomposition")
    print("=" * 70)

    # Load issues
    if args.use_huggingface:
        print("\nLoading issues from HuggingFace")
        issues = load_issues_from_huggingface(args.issue_id)
    else:
        print(f"\nLoading issues from {args.dataset}")
        issues = load_issues(args.dataset, args.issue_id)

    if args.limit:
        issues = issues[: args.limit]

    print(f"Loaded {len(issues)} issue(s)")

    if not issues:
        print("No issues to process!")
        return

    # Configure model using existing model registry
    resolved_model = configure_model_environment(args.model)
    print(f"Model input: {args.model}")
    print(f"Resolved model: {resolved_model}")

    # Initialize analyzer with resolved model
    analyzer = CommitAnalyzer(model_name=resolved_model)
    print(f"Using model: {analyzer.model_name}")

    # Initialize GitHub fetcher
    github_fetcher = GitHubFetcher()
    rate_limit = github_fetcher.check_rate_limit()
    print(
        f"GitHub API rate limit: {rate_limit['remaining']}/{rate_limit['limit']} remaining"
    )

    if rate_limit["remaining"] < 10:
        print("WARNING: Low GitHub API rate limit! Consider setting GITHUB_TOKEN")
        print("Without token: 60 requests/hour. With token: 5000 requests/hour")

    # Process each issue
    results = []
    stats = {
        "total": len(issues),
        "success": 0,
        "failed": 0,
        "no_commits": 0,
        "no_validated_commits": 0,
    }

    for i, issue in enumerate(issues, 1):
        issue = dict(issue)
        overrides = {
            "repo_owner": args.repo_owner,
            "repo_name": args.repo_name,
            "sha_fail": args.sha_fail,
            "sha_success": args.sha_success,
        }
        applied_overrides = {
            key: value for key, value in overrides.items() if value is not None
        }
        issue.update(applied_overrides)

        issue_id = issue.get("id", "unknown")
        print(f"\n[{i}/{len(issues)}] Issue {issue_id}")
        print("-" * 70)
        if applied_overrides:
            print(f"  Applied overrides: {json.dumps(applied_overrides)}")

        # Load validation cache (has validation_sequence + failure_info)
        # Pass issue data, github_fetcher, and analyzer to generate if not in cache
        validation_cache = load_validation_cache(
            issue_id, issue_data=issue, github_fetcher=github_fetcher, analyzer=analyzer
        )

        # Decompose using GitHub API
        try:
            result = decompose_issue(
                issue=issue,
                validation_cache=validation_cache,
                analyzer=analyzer,
                github_fetcher=github_fetcher,
            )

            # Update stats
            if "error" in result:
                if "No commits found" in result.get("error", ""):
                    stats["no_commits"] += 1
                else:
                    stats["failed"] += 1
            else:
                stats["success"] += 1
                print(f"  ✓ Success: {result.get('total_problems', 0)} problems found")

            results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

            results.append(
                {
                    "issue_id": issue_id,
                    "error": str(e),
                    "decomposition_type": "commit_based",
                }
            )
            stats["failed"] += 1

    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved to {args.output}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    print(f"Total issues: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"No commits: {stats['no_commits']}")
    print(f"No validated commits: {stats['no_validated_commits']}")

    # Show example
    if results and "error" not in results[0]:
        print("\n" + "=" * 70)
        print("Example Result (first issue)")
        print("=" * 70)
        example = results[0]
        print(f"Issue ID: {example['issue_id']}")
        print(f"Total commits: {example.get('total_commits', 0)}")
        print(f"Total problems: {example.get('total_problems', 0)}")
        if example.get("problem_sequence"):
            print("\nFirst problem:")
            print(json.dumps(example["problem_sequence"][0], indent=2))

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
