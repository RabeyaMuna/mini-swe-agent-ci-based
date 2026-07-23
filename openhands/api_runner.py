#!/usr/bin/env python3
"""
OpenHands API Runner - Run CI-Bench using OpenHands API server

This replaces interactive_agent.py and uses the real OpenHands server.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openhands.api_client import OpenHandsAPIClient
from openhands.problem_builder import build_problems_from_issue
from data_loader import CIBenchDataLoader
from scripts.model_registry import configure_model_environment

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"


def load_decomposed_issues(path: Optional[str]) -> dict[str, dict[str, Any]]:
    """Load decomposed_issues.json"""
    if not path:
        return {}

    decomp_path = Path(path)
    if not decomp_path.exists():
        print(f"Warning: {decomp_path} not found")
        return {}

    content = decomp_path.read_text()
    if content.startswith("["):
        issues = json.loads(content)
    else:
        issues = [json.loads(line) for line in content.splitlines() if line.strip()]

    # Index by various keys
    index = {}
    for issue in issues:
        for key in ["id", "instance_id", "original_issue_id", "sha_fail"]:
            value = issue.get(key)
            if value:
                index[str(value)] = issue

    return index


def find_decomposed_issue(
    issue_data: dict[str, Any], decomposed_index: dict[str, dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Find decomposed problem for issue"""
    for key in ["id", "instance_id", "sha_fail"]:
        value = issue_data.get(key)
        if value and str(value) in decomposed_index:
            return decomposed_index[str(value)]
    return None


def run_single_issue(
    issue: dict[str, Any],
    log_details: dict[str, Any],
    workflow_cache: dict[str, Any],
    client: OpenHandsAPIClient,
    decomposed_index: dict[str, dict[str, Any]],
    mode: str = "baseline",
) -> dict[str, Any]:
    """
    Run single issue through OpenHands API

    Args:
        issue: Issue from eval_set
        log_details: CI logs cache
        workflow_cache: Workflow validation cache
        client: OpenHands API client
        decomposed_index: Decomposed problems index
        mode: 'baseline' or 'memory'

    Returns:
        Result dict with patch and metadata
    """
    # Load full issue data
    data_loader = CIBenchDataLoader(data_root=str(DATA_ROOT))
    issue_data = data_loader.get_issue_data(issue, log_details, workflow_cache)

    instance_id = issue_data["instance_id"]
    print(f"\nProcessing: {instance_id}")
    print(f"  Repository: {issue_data['repo']}")
    print(f"  Commit: {issue_data['sha_fail'][:8]}")

    # Find decomposed problems if available
    decomposed_issue = find_decomposed_issue(issue_data, decomposed_index)

    # Build structured problems
    problems = build_problems_from_issue(issue_data, decomposed_issue, mode=mode)

    print(f"  Problems: {len(problems)}")
    if decomposed_issue:
        print(f"  Has Decomposition: Yes")

    # Run each problem through OpenHands
    all_patches = []
    all_conversations = []

    for idx, problem in enumerate(problems, 1):
        print(f"\n  Problem {idx}/{len(problems)}: {problem['problem_id']}")
        print(f"    Summary: {problem['summary'][:80]}")

        try:
            # Create conversation
            conv_id = client.create_conversation(problem)
            all_conversations.append(conv_id)

            # Wait for completion
            final_state = client.wait_for_completion(conv_id, timeout=600)

            # Get patch
            patch = client.get_patch(conv_id)
            if patch:
                all_patches.append(patch)
                print(f"    ✓ Got patch ({len(patch)} chars)")
            else:
                print(f"    ✗ No patch generated")

            # Check status
            status = final_state.get("status", "unknown")
            print(f"    Status: {status}")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

    # Combine patches if multiple
    if len(all_patches) > 1:
        combined_patch = "\n\n".join(all_patches)
    elif all_patches:
        combined_patch = all_patches[0]
    else:
        combined_patch = ""

    # Build result
    result = {
        "instance_id": instance_id,
        "model_name_or_path": "openhands-api",
        "model_patch": combined_patch,
        "agent": "openhands-api",
        "mode": mode,
        "problem_count": len(problems),
        "conversation_ids": all_conversations,
        "has_decomposition": bool(decomposed_issue),
        "status": "success" if combined_patch else "no_patch",
    }

    print(f"\n  Final: {result['status']} | Patch: {len(combined_patch)} chars")

    return result


def run_batch(
    eval_issues_path: str,
    mode: str,
    openhands_url: str,
    api_token: Optional[str],
    output_dir: str,
    decomposed_issues_path: Optional[str] = None,
    slice_range: Optional[str] = None,
    hf_dataset: str = "ci-benchmark-user/ci-repair-bench",
    split: str = "train",
):
    """
    Run CI-Bench batch through OpenHands API

    Args:
        eval_issues_path: Path to eval_set.jsonl
        mode: 'baseline' or 'memory'
        openhands_url: OpenHands server URL
        api_token: Optional API token
        output_dir: Output directory
        decomposed_issues_path: Optional decomposed problems
        slice_range: Optional slice (e.g., "0:5")
        hf_dataset: HuggingFace dataset
        split: Dataset split
    """
    print(f"\n{'=' * 80}")
    print("  CI-Bench Evaluation via OpenHands API")
    print(f"{'=' * 80}")
    print(f"Mode: {mode}")
    print(f"OpenHands URL: {openhands_url}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 80}\n")

    # Setup
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_loader = CIBenchDataLoader(data_root=str(DATA_ROOT))
    eval_issues = data_loader.load_eval_issues(
        eval_issues_path, hf_dataset=hf_dataset, split=split
    )

    # Apply slice
    if slice_range:
        start, end = map(int, slice_range.split(":"))
        eval_issues = eval_issues[start:end]
        print(f"Using slice {slice_range}: {len(eval_issues)} issues\n")

    # Load decomposed issues
    decomposed_index = load_decomposed_issues(decomposed_issues_path)
    if decomposed_index:
        print(f"Loaded {len(decomposed_index)} decomposed problem keys\n")

    # Ensure data files
    log_details, workflow_cache = data_loader.ensure_data_files(
        eval_issues, analyzer_llm=None, analyzer_model=None
    )

    # Create API client
    client = OpenHandsAPIClient(base_url=openhands_url, api_token=api_token)

    # Test connection
    try:
        print("Testing OpenHands server connection...")
        # Try to get a non-existent conversation to test API
        try:
            client.get_conversation_state("test-ping")
        except:
            pass  # Expected to fail, just testing connection
        print("✓ OpenHands server is reachable\n")
    except Exception as e:
        print(f"✗ Cannot reach OpenHands server: {e}")
        print(f"  Make sure OpenHands is running at {openhands_url}")
        print(f"  Start with: openhands start --port 3000")
        return 1

    # Run all issues
    results = []
    for issue in tqdm(eval_issues, desc="Processing issues"):
        try:
            result = run_single_issue(
                issue=issue,
                log_details=log_details,
                workflow_cache=workflow_cache,
                client=client,
                decomposed_index=decomposed_index,
                mode=mode,
            )
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error processing {issue.get('instance_id')}: {e}")
            continue

    # Save results
    preds_file = results_dir / "preds.json"
    with open(preds_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f" Completed {len(results)}/{len(eval_issues)} issues")
    print(f" Results: {preds_file}")
    print(f"{'=' * 80}\n")

    # Summary
    successful = sum(1 for r in results if r.get("model_patch"))
    print(f"Success rate: {successful}/{len(results)}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run CI-Bench through OpenHands API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline mode
  python openhands/api_runner.py \\
    --eval-issues data/trs/eval_set.jsonl \\
    --mode baseline \\
    --openhands-url http://localhost:3000 \\
    --output results/openhands-api/baseline

  # Memory mode with decomposition
  python openhands/api_runner.py \\
    --eval-issues data/trs/eval_set.jsonl \\
    --mode memory \\
    --decomposed-issues data/trs/decomposed_issues.json \\
    --openhands-url http://localhost:3000 \\
    --output results/openhands-api/memory

  # Test on first 5 issues
  python openhands/api_runner.py \\
    --eval-issues data/trs/eval_set.jsonl \\
    --mode baseline \\
    --slice 0:5 \\
    --openhands-url http://localhost:3000 \\
    --output results/openhands-api/test
        """,
    )

    parser.add_argument(
        "--eval-issues", required=True, help="Path to eval_set.jsonl"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "memory"],
        help="Mode: baseline (no hints) or memory (with decomposition/hints)",
    )
    parser.add_argument(
        "--openhands-url",
        default="http://localhost:3000",
        help="OpenHands server URL (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="Optional API token for authentication",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--decomposed-issues",
        help="Optional path to decomposed_issues.json",
    )
    parser.add_argument(
        "--slice",
        help="Optional slice range (e.g., '0:5')",
    )
    parser.add_argument(
        "--hf-dataset",
        default="ci-benchmark-user/ci-repair-bench",
        help="HuggingFace dataset",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split",
    )

    args = parser.parse_args()

    return run_batch(
        eval_issues_path=args.eval_issues,
        mode=args.mode,
        openhands_url=args.openhands_url,
        api_token=args.api_token,
        output_dir=args.output,
        decomposed_issues_path=args.decomposed_issues,
        slice_range=args.slice,
        hf_dataset=args.hf_dataset,
        split=args.split,
    )


if __name__ == "__main__":
    sys.exit(main())
