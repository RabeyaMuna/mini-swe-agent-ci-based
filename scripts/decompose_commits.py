#!/usr/bin/env python3
"""
decompose_commits.py - Commit-Based Forward Traces
===================================================

Forward trace approach: Commit → Problem
Uses commit_decomposition/ module to analyze commits and build L1/L2/L3 memory.

Pipeline:
1. commit_decomposition/ - Extract problems from commits
2. build_memory/build_l1.py - Build L1 (failure sequences)
3. build_memory/build_l2.py - Build L2 (repair strategies)
4. build_memory/build_l3.py - Build L3 (universal patterns)

Output:
- data/fwr_trs/decomposed_issues.json
- data/fwr_trs/failure_memory.json (L1)
- data/fwr_trs/repo_memory.json (L2)
- data/fwr_trs/cross_memory.json (L3)

Usage:
    python scripts/decompose_commits.py --batch --use-huggingface --model minimax2.5 --limit 10
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import commit decomposition module
from commit_decomposition.commit_based_decomposer import decompose_issue
from commit_decomposition.commit_analyzer import CommitAnalyzer
from commit_decomposition.github_fetcher import GitHubFetcher

# Import memory building functions
from build_memory.build_l1 import generate_l1_from_decomposed_problems
from build_memory.build_l2 import build_l2_memory
from build_memory.build_l3 import build_l3_memory

# Import utilities
from utilities.ci_cache import load_validation_sequence
from datasets import load_dataset
import argparse


def build_l1_l2_l3_for_commit_decomposition(
    decomposed_result: dict, llm, output_dir: str = "data/fwr_trs"
) -> dict:
    """
    Build L1/L2/L3 memory from commit-based decomposed result.

    Args:
        decomposed_result: Result from commit_based_decomposer.decompose_issue()
        llm: LLM instance
        output_dir: Output directory (default: data/fwr_trs)

    Returns:
        Result dict with l1_memory, l2_memory, l3_memory
    """
    issue_id = decomposed_result.get("issue_id", "unknown")

    # Check if decomposition has problems (using unified field name)
    problems = decomposed_result.get("decomposed_problems", [])
    if not problems:
        print(f"  No problems found for issue {issue_id}, skipping L1/L2/L3")
        return decomposed_result

    print(f"\n[Memory Building] Issue {issue_id}")

    # Build L1 memory (using unified structure - same as backward decomposition)
    print("  [1/3] Building L1 (failure sequences)...")
    l1_memory = generate_l1_from_decomposed_problems(
        issue_id=issue_id,
        repo=decomposed_result.get("repo", ""),
        repo_owner=decomposed_result.get("repo_owner", ""),
        workflow_path=decomposed_result.get("workflow", ""),  # ← Changed from workflow_path
        decomposed_problems=decomposed_result.get("decomposed_problems", []),  # ← Changed from problems
        dependencies=decomposed_result.get("dependencies", {}),
        ground_truth_files=decomposed_result.get("changed_files", []),
        llm=llm,
    )
    print(f"  ✓ L1 generated: {len(l1_memory.get('problems', []))} problems")

    # Build L2 memory
    print("  [2/3] Building L2 (repair strategies)...")
    l2_memory = build_l2_memory(l1_memory=l1_memory, llm=llm)
    print(f"  ✓ L2 generated: {len(l2_memory.get('repair_strategies', []))} strategies")

    # Build L3 memory
    print("  [3/3] Building L3 (universal patterns)...")
    l3_memory = build_l3_memory(l1_memory=l1_memory, l2_memory=l2_memory, llm=llm)
    num_patterns = len(l3_memory.get("universal_patterns", []))
    print(f"  ✓ L3 generated: {num_patterns} patterns")

    # Add to result
    result = dict(decomposed_result)
    result["l1_memory"] = l1_memory
    result["l2_memory"] = l2_memory
    result["l3_memory"] = l3_memory

    # Save immediately to fwr_trs/ (append mode)
    _append_to_fwr_trs(result, output_dir)

    return result


def _append_to_fwr_trs(result: dict, output_dir: str):
    """Append result to forward traces memory files (with duplicate filtering)."""
    from pathlib import Path

    fwr_trs_dir = Path(output_dir)
    fwr_trs_dir.mkdir(parents=True, exist_ok=True)

    issue_id = result.get("issue_id", "unknown")
    l1_memory = result.get("l1_memory", {})
    l2_memory = result.get("l2_memory", {})
    l3_memory = result.get("l3_memory", {})

    # Append L1 (filter out duplicates by issue_id)
    if l1_memory:
        failure_memory_path = fwr_trs_dir / "failure_memory.json"
        existing = []
        if failure_memory_path.exists():
            with open(failure_memory_path) as f:
                existing = json.load(f)

        # Remove any existing entry with same issue_id to avoid duplicates
        existing = [e for e in existing if e.get("issue_id") != issue_id]
        existing.append(l1_memory)

        with open(failure_memory_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  ✓ Appended issue {issue_id} to failure_memory.json")

    # Append L2 (filter out duplicates by issue_id)
    if l2_memory:
        repo_memory_path = fwr_trs_dir / "repo_memory.json"
        existing = []
        if repo_memory_path.exists():
            with open(repo_memory_path) as f:
                existing = json.load(f)

        # Remove any existing entry with same issue_id to avoid duplicates
        existing = [e for e in existing if e.get("issue_id") != issue_id]
        existing.append(l2_memory)

        with open(repo_memory_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  ✓ Appended 1 issue to repo_memory.json")

    # Append L3 (filter out duplicates from same issue_id)
    if l3_memory and l3_memory.get("universal_patterns"):
        cross_memory_path = fwr_trs_dir / "cross_memory.json"
        existing = []
        if cross_memory_path.exists():
            with open(cross_memory_path) as f:
                existing = json.load(f)

        # Remove any existing patterns from this issue_id to avoid duplicates
        existing = [e for e in existing if e.get("source_issue_id") != issue_id]

        # Extract patterns with metadata
        for pattern in l3_memory["universal_patterns"]:
            pattern_with_meta = dict(pattern)
            pattern_with_meta["source_issue_id"] = issue_id
            pattern_with_meta["source_repo"] = l3_memory.get("repo", "")
            existing.append(pattern_with_meta)

        with open(cross_memory_path, "w") as f:
            json.dump(existing, f, indent=2)
        num_patterns = len(l3_memory["universal_patterns"])
        print(f"  ✓ Appended {num_patterns} patterns to cross_memory.json")


def main():
    """Main entry point for commit-based decomposition."""
    print("=" * 80)
    print("COMMIT-BASED DECOMPOSITION (Forward Traces: Commit → Problem)")
    print("=" * 80)
    print("Using: commit_decomposition/ module")
    print("Output: data/fwr_trs/")
    print("=" * 80)
    print()

    parser = argparse.ArgumentParser(
        description="Commit-based decomposition with L1/L2/L3"
    )
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--issue-id", type=str, help="Process specific issue ID")
    parser.add_argument(
        "--use-huggingface", action="store_true", help="Use HuggingFace dataset"
    )
    parser.add_argument("--model", type=str, default="minimax2.5", help="LLM model")
    parser.add_argument("--limit", type=int, help="Limit number of issues")
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Output directory"
    )

    args = parser.parse_args()

    # Initialize LLM
    from utilities.llm_provider import get_llm

    llm = get_llm(args.model)

    # Load issues
    if args.use_huggingface:
        print("Loading issues from HuggingFace...")
        dataset = load_dataset("ci-benchmark-user/ci-repair-bench", split="train")
        issues = [dict(item) for item in dataset]
        print(f"Loaded {len(issues)} issues")
    else:
        print("Error: --use-huggingface is required")
        return

    if args.limit:
        issues = issues[: args.limit]
        print(f"Limited to {len(issues)} issues")

    # Filter by specific issue ID if provided
    if args.issue_id:
        issues = [issue for issue in issues if str(issue.get("id")) == str(args.issue_id)]
        if not issues:
            print(f"Error: Issue {args.issue_id} not found in dataset")
            return
        print(f"Filtered to issue {args.issue_id}")

    # Load existing results to avoid reprocessing
    # Check repo_memory.json (L2) to see which issues have L1/L2/L3 built
    output_dir_path = Path(args.output_dir)
    repo_memory_path = output_dir_path / "repo_memory.json"
    decomposed_file = output_dir_path / "decomposed_issues.json"

    existing_results = []
    processed_ids = set()

    # Load from repo_memory.json (issues with L1/L2/L3)
    if repo_memory_path.exists():
        try:
            with open(repo_memory_path) as f:
                repo_memory = json.load(f)
            processed_ids = {
                r.get("issue_id") for r in repo_memory if r.get("issue_id")
            }
            print(f"Loaded {len(processed_ids)} existing L1/L2/L3 results (will skip)")
        except Exception as e:
            print(f"Warning: Could not load existing L2 results: {e}")

    # Also load decomposed results for the final save
    if decomposed_file.exists():
        try:
            with open(decomposed_file) as f:
                existing_results = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load decomposed results: {e}")

    # Initialize analyzers
    analyzer = CommitAnalyzer(llm)  # Pass LLM object (now uses utilities pattern)
    github_fetcher = GitHubFetcher()

    # Process each issue
    results = list(existing_results)  # Start with existing results
    for i, issue in enumerate(issues, 1):
        issue_id = issue.get("id", f"issue_{i}")

        # Skip if already processed
        if issue_id in processed_ids:
            print(f"\n[{i}/{len(issues)}] {issue_id} - ✓ Already processed (skipping)")
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing {i}/{len(issues)}: {issue_id}")
        print(f"{'=' * 80}")

        try:
            # Get validation cache
            sha_fail = issue.get("sha_fail", "")
            validation_cache = load_validation_sequence(issue_id, sha_fail) or {}

            # Decompose issue using commit_decomposition module
            decomposed = decompose_issue(
                issue, validation_cache, analyzer, github_fetcher
            )

            # Build L1/L2/L3 memory
            if not args.batch or "error" not in decomposed:
                result = build_l1_l2_l3_for_commit_decomposition(
                    decomposed, llm, args.output_dir
                )
                results.append(result)

        except Exception as e:
            print(f"Error processing {issue_id}: {e}")
            import traceback

            traceback.print_exc()

    # Save decomposed issues
    fwr_trs_dir = Path(args.output_dir)
    fwr_trs_dir.mkdir(parents=True, exist_ok=True)

    decomposed_file = fwr_trs_dir / "decomposed_issues.json"
    with open(decomposed_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("✓ Commit-based decomposition complete!")
    print(f"{'=' * 80}")
    print(f"Processed: {len(results)} issues")
    print("\nOutput saved to:")
    print(f"  - {fwr_trs_dir}/decomposed_issues.json")
    print(f"  - {fwr_trs_dir}/failure_memory.json (L1)")
    print(f"  - {fwr_trs_dir}/repo_memory.json (L2)")
    print(f"  - {fwr_trs_dir}/cross_memory.json (L3)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
