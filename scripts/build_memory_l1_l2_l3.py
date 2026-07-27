#!/usr/bin/env python3
"""
build_memory_l1_l2_l3.py - Build L1, L2, L3 Memory from Decomposed Issues
==========================================================================

Takes decomposed CI issues and builds three-level memory abstraction:
- L1: Concrete problem-level failures
- L2: Repair strategies (sequential, dependency-aware)
- L3: Universal cross-repo fix patterns

Usage:
    # Build from decomposed issues
    python scripts/build_memory_l1_l2_l3.py \
        --decomposed data/decomposed_issues.json \
        --model glm5.2 \
        --output-dir data/memory

    # Build only L1 and L2 (skip L3)
    python scripts/build_memory_l1_l2_l3.py \
        --decomposed data/decomposed_issues.json \
        --model glm5.2 \
        --skip-l3

    # Limit to specific issues
    python scripts/build_memory_l1_l2_l3.py \
        --decomposed data/decomposed_issues.json \
        --model glm5.2 \
        --issue-ids 102,410,500
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from build_memory import build_l2_memory, build_l3_memory
from utilities.model_registry import configure_model_environment


class LitellmModel:
    """Simple LLM wrapper for build_memory compatibility."""

    def __init__(self, model_name: str):
        import litellm

        self.model_name = model_name
        self.litellm = litellm

    def invoke(self, prompt: str) -> Any:
        response = self.litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content


def load_decomposed_issues(path: Path) -> List[Dict[str, Any]]:
    """Load decomposed issues from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "issues" in data:
        return data["issues"]
    else:
        raise ValueError(f"Unexpected format in {path}")


def build_l1_from_decomposed(decomposed_issue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert decomposed issue to L1 memory format.

    Decomposed issue has:
    - issue_id, repo, workflow
    - changed_files
    - decomposed_problems: [{problem_id, verification_cmd, failure_type, problem, root_cause, fix_strategy, files, enabled}]
    """

    issue_id = str(decomposed_issue.get("issue_id", ""))
    repo = decomposed_issue.get("repo", "")
    workflow = decomposed_issue.get("workflow", "")
    changed_files = decomposed_issue.get("changed_files", [])
    problems = decomposed_issue.get("decomposed_problems", [])

    return {
        "issue_id": issue_id,
        "repo": repo,
        "workflow": workflow,
        "changed_files": changed_files,
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build L1, L2, L3 memory from decomposed CI issues"
    )
    parser.add_argument(
        "--decomposed",
        required=True,
        help="Path to decomposed_issues.json",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model for L2/L3 generation (e.g., glm5.2, minimax2.5)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/memory",
        help="Output directory for L1/L2/L3 files",
    )
    parser.add_argument(
        "--skip-l3",
        action="store_true",
        help="Skip L3 generation (only build L1 and L2)",
    )
    parser.add_argument(
        "--issue-ids",
        help="Comma-separated issue IDs to process (optional filter)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of issues to process",
    )

    args = parser.parse_args()

    # Initialize model
    print(f"\n{'=' * 80}")
    args.model = configure_model_environment(args.model) or args.model
    print(f"Initializing LLM: {args.model}")
    print(f"{'=' * 80}\n")

    llm = LitellmModel(model_name=args.model)

    # Load decomposed issues
    decomposed_path = Path(args.decomposed)
    if not decomposed_path.exists():
        print(f"ERROR: Decomposed file not found: {decomposed_path}")
        return 1

    print(f"Loading decomposed issues from: {decomposed_path}")
    decomposed_issues = load_decomposed_issues(decomposed_path)
    print(f"Loaded {len(decomposed_issues)} decomposed issues")

    # Filter by issue IDs if specified
    if args.issue_ids:
        filter_ids = set(args.issue_ids.split(","))
        decomposed_issues = [
            issue
            for issue in decomposed_issues
            if str(issue.get("issue_id", "")) in filter_ids
        ]
        print(f"Filtered to {len(decomposed_issues)} issues: {filter_ids}")

    # Limit if requested
    if args.limit:
        decomposed_issues = decomposed_issues[: args.limit]
        print(f"Limited to first {args.limit} issues")

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    l1_dir = output_dir / "l1"
    l2_dir = output_dir / "l2"
    l3_dir = output_dir / "l3"

    l1_dir.mkdir(exist_ok=True)
    l2_dir.mkdir(exist_ok=True)
    if not args.skip_l3:
        l3_dir.mkdir(exist_ok=True)

    # Process each issue
    l1_records = []
    l2_records = []
    l3_records = []

    for i, decomposed_issue in enumerate(decomposed_issues, 1):
        issue_id = str(decomposed_issue.get("issue_id", ""))
        repo = decomposed_issue.get("repo", "")

        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(decomposed_issues)}] Processing Issue {issue_id} ({repo})")
        print(f"{'=' * 80}")

        try:
            # Build L1
            print("Building L1 (concrete problems)...")
            l1_memory = build_l1_from_decomposed(decomposed_issue)

            # Save L1
            l1_file = l1_dir / f"{issue_id}_l1.json"
            with open(l1_file, "w") as f:
                json.dump(l1_memory, f, indent=2)
            print(f"✓ L1 saved: {l1_file}")
            l1_records.append(l1_memory)

            # Build L2
            print("Building L2 (repair strategies)...")
            l2_memory = build_l2_memory(l1_memory=l1_memory, llm=llm)

            # Save L2
            l2_file = l2_dir / f"{issue_id}_l2.json"
            with open(l2_file, "w") as f:
                json.dump(l2_memory, f, indent=2)
            print(f"✓ L2 saved: {l2_file}")
            l2_records.append(l2_memory)

            # Build L3 (if not skipped)
            if not args.skip_l3:
                print("Building L3 (universal patterns)...")
                l3_memory = build_l3_memory(
                    l1_memory=l1_memory, l2_memory=l2_memory, llm=llm
                )

                # Save L3
                l3_file = l3_dir / f"{issue_id}_l3.json"
                with open(l3_file, "w") as f:
                    json.dump(l3_memory, f, indent=2)
                print(f"✓ L3 saved: {l3_file}")
                l3_records.append(l3_memory)

            print(f"✓ Issue {issue_id} complete")

        except Exception as e:
            print(f"✗ ERROR processing issue {issue_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save aggregated files
    print(f"\n{'=' * 80}")
    print("Saving aggregated memory files...")
    print(f"{'=' * 80}")

    with open(output_dir / "l1_all.json", "w") as f:
        json.dump(l1_records, f, indent=2)
    print(f"✓ L1 aggregated: {output_dir / 'l1_all.json'}")

    with open(output_dir / "l2_all.json", "w") as f:
        json.dump(l2_records, f, indent=2)
    print(f"✓ L2 aggregated: {output_dir / 'l2_all.json'}")

    if not args.skip_l3:
        with open(output_dir / "l3_all.json", "w") as f:
            json.dump(l3_records, f, indent=2)
        print(f"✓ L3 aggregated: {output_dir / 'l3_all.json'}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total issues processed: {len(l1_records)}")
    print(f"L1 records: {len(l1_records)}")
    print(f"L2 records: {len(l2_records)}")
    if not args.skip_l3:
        print(f"L3 records: {len(l3_records)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
