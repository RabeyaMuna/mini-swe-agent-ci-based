#!/usr/bin/env python3
"""
decompose_commits.py - Commit-Based Forward Traces
===================================================

Forward trace approach: Commit -> Problem
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

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import commit decomposition module
import argparse

from datasets import load_dataset

# Import memory building functions
from build_memory.build_l1 import generate_l1_from_decomposed_problems
from build_memory.build_l2 import build_l2_memory
from build_memory.build_l3 import build_l3_memory
from commit_decomposition.commit_analyzer import CommitAnalyzer
from commit_decomposition.commit_based_decomposer import decompose_issue
from commit_decomposition.github_fetcher import GitHubFetcher

# Import utilities
from utilities.ci_cache import load_validation_sequence
from utilities.error_handler import (
    clear_error_from_execption,
    save_error_to_execption,
)


def _issue_id(record: dict) -> str:
    return str(
        record.get("issue_id")
        or record.get("id")
        or record.get("instance_id")
        or record.get("original_issue_id")
        or ""
    )


def _load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not load {path}: {e}")
        return []


def _complete_memory_issue_ids(output_dir: Path) -> set[str]:
    l1 = _load_json_list(output_dir / "failure_memory.json")
    l2 = _load_json_list(output_dir / "repo_memory.json")
    l3 = _load_json_list(output_dir / "cross_memory.json")

    l1_ids = {str(item["issue_id"]) for item in l1 if item.get("issue_id")}
    l2_ids = {str(item["issue_id"]) for item in l2 if item.get("issue_id")}
    l3_ids = {
        str(item["source_issue_id"]) for item in l3 if item.get("source_issue_id")
    }
    return l1_ids & l2_ids & l3_ids


def _save_decomposed_results(results: list[dict], output_path: Path) -> None:
    by_issue_id: dict[str, dict] = {}
    for result in results:
        clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
        issue_id = _issue_id(clean_result)
        if issue_id:
            by_issue_id[issue_id] = clean_result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(list(by_issue_id.values()), f, indent=2)


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
    problems = decomposed_result.get("problems", [])
    if not problems:
        print(
            f"  No problems found for issue {issue_id}, saving empty L1/L2/L3 records"
        )
        workflow_path = decomposed_result.get("workflow", "")
        workflow_name = workflow_path.split("/")[-1] if workflow_path else ""
        l1_memory = {
            "issue_id": issue_id,
            "repo": decomposed_result.get("repo", ""),
            "repo_owner": decomposed_result.get("repo_owner", ""),
            "workflow": workflow_path,
            "workflow_name": workflow_name,
            "changed_files": decomposed_result.get("_changed_files", []),
            "problems": [],
            "note": decomposed_result.get("error")
            or "Forward decomposition found no problems",
        }
        l2_memory = {
            "issue_id": issue_id,
            "repo": decomposed_result.get("repo", ""),
            "workflow": workflow_path,
            "total_problems": 0,
            "failure_identify": [],
            "repair_strategies": [],
            "note": decomposed_result.get("error")
            or "Forward decomposition found no problems",
        }
        l3_memory = {
            "issue_id": issue_id,
            "repo": decomposed_result.get("repo", ""),
            "workflow": workflow_path,
            "universal_patterns": [
                {
                    "pattern_id": f"no-forward-problems-{issue_id}",
                    "failure_type": "no_forward_problems",
                    "failure_pattern": "",
                    "problem": "",
                    "reasoning": decomposed_result.get("error")
                    or "Forward decomposition found no commit-level problems",
                    "when_to_apply": "",
                    "signals": [],
                    "universal_fix": {"approach": "", "steps": []},
                    "examples": [],
                    "no_decomposed_problems": True,
                    "no_forward_problems": True,
                }
            ],
        }
        _append_to_fwr_trs(
            l1_memory=l1_memory,
            l2_memory=l2_memory,
            l3_memory=l3_memory,
            issue_id=issue_id,
            output_dir=output_dir,
        )
        return decomposed_result

    print(f"\n[Memory Building] Issue {issue_id}")

    # Build L1 memory (using unified structure - same as backward decomposition)
    print("  [1/3] Building L1 (failure sequences)...")
    l1_memory = generate_l1_from_decomposed_problems(
        issue_id=issue_id,
        repo=decomposed_result.get("repo", ""),
        repo_owner=decomposed_result.get("repo_owner", ""),
        workflow_path=decomposed_result.get("workflow", ""),
        decomposed_problems=problems,  # Use the problems we already extracted
        dependencies=decomposed_result.get("_dependencies", {}),
        ground_truth_files=decomposed_result.get("_changed_files", []),
        llm=llm,
    )
    print(f"  OK L1 generated: {len(l1_memory.get('problems', []))} problems")

    # Build L2 memory
    print("  [2/3] Building L2 (repair strategies)...")
    l2_memory = build_l2_memory(l1_memory=l1_memory, llm=llm)
    print(
        f"  OK L2 generated: {len(l2_memory.get('repair_strategies', []))} strategies"
    )

    # Build L3 memory
    print("  [3/3] Building L3 (universal patterns)...")
    l3_memory = build_l3_memory(l1_memory=l1_memory, l2_memory=l2_memory, llm=llm)
    num_patterns = len(l3_memory.get("universal_patterns", []))
    print(f"  OK L3 generated: {num_patterns} patterns")

    # Save L1/L2/L3 to their own files (NOT added to decomposed_result)
    _append_to_fwr_trs(
        l1_memory=l1_memory,
        l2_memory=l2_memory,
        l3_memory=l3_memory,
        issue_id=issue_id,
        output_dir=output_dir,
    )

    # Return clean decomposed result (no L1/L2/L3 embedded)
    return decomposed_result


def _append_to_fwr_trs(
    l1_memory: dict,
    l2_memory: dict,
    l3_memory: dict,
    issue_id: str,
    output_dir: str,
):
    """Append L1/L2/L3 memory to their respective files (with duplicate filtering)."""
    from pathlib import Path

    fwr_trs_dir = Path(output_dir)
    fwr_trs_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"  OK Appended issue {issue_id} to failure_memory.json")

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
        print("  OK Appended 1 issue to repo_memory.json")

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
        print(f"  OK Appended {num_patterns} patterns to cross_memory.json")


def main():
    """Main entry point for commit-based decomposition."""
    print("=" * 80)
    print("COMMIT-BASED DECOMPOSITION (Forward Traces: Commit -> Problem)")
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
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to JSONL dataset file. Use this for data/memory_set.jsonl.",
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
    if args.dataset:
        print(f"Loading issues from {args.dataset}...")
        issues = []
        with open(args.dataset) as f:
            for line in f:
                if line.strip():
                    issues.append(json.loads(line))
        print(f"Loaded {len(issues)} issues")
    elif args.use_huggingface:
        print("Loading issues from HuggingFace...")
        dataset = load_dataset("ci-benchmark-user/ci-repair-bench", split="train")
        issues = [dict(item) for item in dataset]
        print(f"Loaded {len(issues)} issues")
    else:
        print("Error: --dataset or --use-huggingface is required")
        return

    if args.limit:
        issues = issues[: args.limit]
        print(f"Limited to {len(issues)} issues")

    # Filter by specific issue ID if provided
    if args.issue_id:
        issues = [
            issue for issue in issues if str(issue.get("id")) == str(args.issue_id)
        ]
        if not issues:
            print(f"Error: Issue {args.issue_id} not found in dataset")
            return
        print(f"Filtered to issue {args.issue_id}")

    # Load existing results to avoid reprocessing
    # Check complete L1/L2/L3 files to see which issues are already fully built.
    output_dir_path = Path(args.output_dir)
    decomposed_file = output_dir_path / "decomposed_issues.json"

    existing_results = _load_json_list(decomposed_file)
    # Error-only cache entries are stale/incomplete: they were created when a
    # GitHub compare request returned no commits even though the dataset has a
    # repair diff.  Drop them so the issue is reprocessed through the fallback.
    decomposed_cache = {
        _issue_id(result): result
        for result in existing_results
        if _issue_id(result) and not result.get("error")
    }
    stale_error_ids = {
        _issue_id(result)
        for result in existing_results
        if _issue_id(result) and result.get("error")
    }
    if stale_error_ids:
        print(
            f"Discarding {len(stale_error_ids)} stale error-only decomposition "
            "records; they will be rebuilt from the stored dataset diffs"
        )
    processed_ids = _complete_memory_issue_ids(output_dir_path)
    processed_ids -= stale_error_ids
    partial_ids = (
        {
            str(item.get("issue_id"))
            for path in ("failure_memory.json", "repo_memory.json")
            for item in _load_json_list(output_dir_path / path)
            if item.get("issue_id")
        }
        | {
            str(item.get("source_issue_id"))
            for item in _load_json_list(output_dir_path / "cross_memory.json")
            if item.get("source_issue_id")
        }
    ) - processed_ids
    print(f"Loaded {len(decomposed_cache)} decomposed issues (can reuse)")
    print(f"Loaded {len(processed_ids)} complete L1/L2/L3 results (will skip)")
    if partial_ids:
        print(f"Found {len(partial_ids)} partial memory issues (will rebuild/replace)")

    # Initialize analyzers
    analyzer = CommitAnalyzer(llm)  # Pass LLM object (now uses utilities pattern)
    github_fetcher = GitHubFetcher()

    for i, issue in enumerate(issues, 1):
        issue_id = _issue_id(issue) or f"issue_{i}"

        # Skip if already processed
        if issue_id in processed_ids:
            print(f"\n[{i}/{len(issues)}] {issue_id} - OK Already processed (skipping)")
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing {i}/{len(issues)}: {issue_id}")
        print(f"{'=' * 80}")

        try:
            if issue_id in decomposed_cache:
                print("  Found in decomposed_issues.json - Building L1/L2/L3 directly")
                decomposed = decomposed_cache[issue_id]
            else:
                # Get validation cache
                sha_fail = issue.get("sha_fail", "")
                validation_cache = load_validation_sequence(issue_id, sha_fail) or {}

                # Decompose issue using commit_decomposition module
                decomposed = decompose_issue(
                    issue, validation_cache, analyzer, github_fetcher
                )
                if decomposed.get("error"):
                    save_error_to_execption(
                        issue_id,
                        RuntimeError(decomposed["error"]),
                        error_type="DECOMPOSITION_ERROR",
                        additional_context={
                            "repo": f"{issue.get('repo_owner', '')}/{issue.get('repo_name', '')}",
                            "sha_fail": issue.get("sha_fail", ""),
                            "sha_success": issue.get("sha_success", ""),
                            "changed_files": issue.get("changed_files", []),
                        },
                    )
                    print(
                        "  Decomposition failed; saved details to "
                        f"execption/{issue_id}.json"
                    )
                    continue
                decomposed_cache[issue_id] = decomposed
                _save_decomposed_results(
                    list(decomposed_cache.values()), decomposed_file
                )

            # Build L1/L2/L3 memory. Even decompositions with no commit-level
            # problems get empty records so memory coverage matches the
            # decomposed issue set.
            result = build_l1_l2_l3_for_commit_decomposition(
                decomposed, llm, args.output_dir
            )
            decomposed_cache[issue_id] = result
            clear_error_from_execption(issue_id)

        except Exception as e:  # noqa: BLE001
            print(f"Error processing {issue_id}: {e}")
            import traceback

            traceback.print_exc()
            save_error_to_execption(
                issue_id,
                e,
                error_type="DECOMPOSITION_ERROR",
                additional_context={
                    "repo": f"{issue.get('repo_owner', '')}/{issue.get('repo_name', '')}",
                    "sha_fail": issue.get("sha_fail", ""),
                    "sha_success": issue.get("sha_success", ""),
                    "changed_files": issue.get("changed_files", []),
                },
            )
            decomposed_cache.pop(issue_id, None)

    fwr_trs_dir = Path(args.output_dir)
    fwr_trs_dir.mkdir(parents=True, exist_ok=True)

    _save_decomposed_results(
        list(decomposed_cache.values()), fwr_trs_dir / "decomposed_issues.json"
    )

    print(f"\n{'=' * 80}")
    print("OK Commit-based decomposition complete!")
    print(f"{'=' * 80}")
    print(f"Decomposed cache: {len(decomposed_cache)} issues")
    print("\nOutput saved to:")
    print(f"  - {fwr_trs_dir}/decomposed_issues.json")
    print(f"  - {fwr_trs_dir}/failure_memory.json (L1)")
    print(f"  - {fwr_trs_dir}/repo_memory.json (L2)")
    print(f"  - {fwr_trs_dir}/cross_memory.json (L3)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
