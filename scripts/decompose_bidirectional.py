#!/usr/bin/env python3
"""
decompose_bidirectional.py - Bidirectional Decomposition (Forward + Backward)
==============================================================================

Bidirectional approach: Combines forward (commit) and backward (CI failure) traces
Uses bidirectional_decomposition/ module to reconcile both views.

Pipeline:
1. bidirectional_decomposition/ - Reconcile forward + backward decompositions
2. build_memory/build_l1.py - Build L1 (failure sequences)
3. build_memory/build_l2.py - Build L2 (repair strategies)
4. build_memory/build_l3.py - Build L3 (universal patterns)

Output:
- data/bidirect_trs/decomposed_issues.json
- data/bidirect_trs/failure_memory.json (L1)
- data/bidirect_trs/repo_memory.json (L2)
- data/bidirect_trs/cross_memory.json (L3)

Usage:
    python scripts/decompose_bidirectional.py --batch --use-huggingface --model minimax2.5 --limit 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import bidirectional decomposition
from bidirectional_decomposition.unified_decomposition import (
    build_unified_decomposition,
    save_to_decomposed_issues,
)

# Import memory building functions
from build_memory.build_l1 import generate_l1_from_decomposed_problems
from build_memory.build_l2 import build_l2_memory
from build_memory.build_l3 import build_l3_memory

# Import utilities
from datasets import load_dataset
from utilities.llm_model import LitellmModel
from utilities.model_registry import configure_model_environment


def _issue_id(record: dict) -> str:
    return str(
        record.get("issue_id")
        or record.get("id")
        or record.get("instance_id")
        or record.get("original_issue_id")
        or ""
    )


def _load_existing_memory_files(
    output_dir: Path,
) -> tuple[dict[str, Any], set[str], set[str], set[str], set[str]]:
    """
    Load existing L1/L2/L3 memory files and return per-type IDs.

    Returns:
        - decomposed_cache: Dict of issue_id -> decomposed result
        - l1_ids: Set of issue IDs that have L1
        - l2_ids: Set of issue IDs that have L2
        - l3_ids: Set of issue IDs that have L3
        - processed_ids: Set of issue IDs that have complete L1 AND L2 AND L3
    """
    decomposed_cache = {}
    l1_ids, l2_ids, l3_ids = set(), set(), set()

    # Load decomposed_issues.json
    decomposed_path = output_dir / "decomposed_issues.json"
    if decomposed_path.exists():
        try:
            with open(decomposed_path) as f:
                decomposed_list = json.load(f)
            if isinstance(decomposed_list, list):
                for item in decomposed_list:
                    issue_id = _issue_id(item)
                    if issue_id:
                        decomposed_cache[issue_id] = item
            print(f"Loaded {len(decomposed_cache)} decomposed issues from cache")
        except Exception as e:
            print(f"Warning: Could not load decomposed_issues.json: {e}")

    # Load L1 (failure_memory.json)
    failure_memory_path = output_dir / "failure_memory.json"
    if failure_memory_path.exists():
        try:
            with open(failure_memory_path) as f:
                l1_existing = json.load(f)
            if isinstance(l1_existing, list):
                l1_ids = {
                    str(item["issue_id"])
                    for item in l1_existing
                    if "issue_id" in item
                }
        except Exception as e:
            print(f"Warning: Could not load L1 memory: {e}")

    # Load L2 (repo_memory.json)
    repo_memory_path = output_dir / "repo_memory.json"
    if repo_memory_path.exists():
        try:
            with open(repo_memory_path) as f:
                l2_existing = json.load(f)
            if isinstance(l2_existing, list):
                l2_ids = {
                    str(item["issue_id"])  # L2 uses issue_id, not source_issue_id
                    for item in l2_existing
                    if "issue_id" in item
                }
        except Exception as e:
            print(f"Warning: Could not load L2 memory: {e}")

    # Load L3 (cross_memory.json)
    cross_memory_path = output_dir / "cross_memory.json"
    if cross_memory_path.exists():
        try:
            with open(cross_memory_path) as f:
                l3_existing = json.load(f)
            if isinstance(l3_existing, list):
                l3_ids = {
                    str(item["source_issue_id"])
                    for item in l3_existing
                    if "source_issue_id" in item
                }
        except Exception as e:
            print(f"Warning: Could not load L3 memory: {e}")

    # Issues with ALL THREE (L1 AND L2 AND L3)
    processed_ids = l1_ids & l2_ids & l3_ids

    if processed_ids:
        print(f"Found {len(processed_ids)} complete issues (will skip)")

    return decomposed_cache, l1_ids, l2_ids, l3_ids, processed_ids


def _load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return []


def build_l1_l2_l3_for_bidirectional(
    decomposed_result: dict, llm, output_dir: str = "data/bidirect_trs"
) -> dict:
    """
    Build L1/L2/L3 memory from bidirectional decomposed result.

    Args:
        decomposed_result: Result from unified_decomposition
        llm: LLM instance
        output_dir: Output directory (default: data/bidirect_trs)

    Returns:
        Result dict with memory appended to files
    """
    issue_id = decomposed_result.get("original_issue_id", "unknown")

    # Check if decomposition has problems
    problems = decomposed_result.get("problems", [])
    if not problems:
        print(f"  No problems found for issue {issue_id}, skipping memory build")
        return decomposed_result

    print(f"\n[Memory Building] Issue {issue_id}")

    # Build L1 memory
    print("  [1/3] Building L1 (failure sequences)...")

    # Prepare dependencies in expected format
    deps = decomposed_result.get("dependencies", [])
    if isinstance(deps, list):
        # Wrap list in dict structure expected by build_l1
        dependencies = {
            "dependency_edges": deps,
            "repair_order": decomposed_result.get("repair_sequence", []),
        }
    else:
        # Already a dict
        dependencies = deps if deps else {}

    l1_memory = generate_l1_from_decomposed_problems(
        issue_id=issue_id,
        repo=decomposed_result.get("repo", ""),
        repo_owner=decomposed_result.get("repo_owner", ""),
        workflow_path=decomposed_result.get("workflow_path", ""),
        decomposed_problems=problems,
        dependencies=dependencies,
        ground_truth_files=decomposed_result.get("changed_files", []),
        # Dependencies and repair order are already finalized. Use L1's
        # mechanical path and avoid an unnecessary second reasoning call.
        llm=None,
    )
    # Preserve the canonical workflow display name from the dataset/decomposition
    l1_memory["workflow_path"] = decomposed_result.get("workflow_path", "")
    l1_memory["workflow_name"] = (
        decomposed_result.get("workflow_name") or l1_memory.get("workflow_name", "")
    )
    _synchronize_l1_with_finalized_decomposition(l1_memory, decomposed_result)
    num_problems = len(l1_memory.get("problems", []))
    print(f"  OK L1 generated: {num_problems} problems")

    # CRITICAL: Skip empty decompositions (0 problems) - don't save to memory
    if num_problems == 0:
        print(f"  SKIP Issue {issue_id} has 0 problems - not building L2/L3 or saving to memory")
        return decomposed_result

    # Build L2 memory
    print("  [2/3] Building L2 (repair strategies)...")
    l2_memory = _build_l2_losslessly(l1_memory=l1_memory, llm=llm)
    # Add canonical workflow metadata at the bidirectional boundary without
    # changing the reusable L2 builder.
    l2_memory["workflow_path"] = (
        decomposed_result.get("workflow_path") or l1_memory.get("workflow_path", "")
    )
    l2_memory["workflow_name"] = (
        decomposed_result.get("workflow_name") or l1_memory.get("workflow_name", "")
    )
    print(
        f"  OK L2 generated: {len(l2_memory.get('repair_strategies', []))} strategies"
    )

    # Build L3 memory
    print("  [3/3] Building L3 (universal patterns)...")
    l3_memory = build_l3_memory(l1_memory=l1_memory, l2_memory=l2_memory, llm=llm)
    num_patterns = len(l3_memory.get("universal_patterns", []))
    print(f"  OK L3 generated: {num_patterns} patterns")

    # Append to memory files
    _append_to_bidirect_trs(
        l1_memory=l1_memory,
        l2_memory=l2_memory,
        l3_memory=l3_memory,
        issue_id=issue_id,
        output_dir=output_dir,
    )

    return decomposed_result


def _build_l2_losslessly(l1_memory: dict, llm) -> dict:
    """Build L2 without allowing one malformed response to erase all strategies.

    The reusable L2 builder intentionally degrades to an empty result when its
    model response cannot be parsed. At this pipeline boundary, retry empty
    full-issue results one problem at a time. If an individual response is also
    malformed, preserve that L1 problem as a deterministic strategy instead of
    dropping it.
    """
    l2_memory = build_l2_memory(l1_memory=l1_memory, llm=llm)
    if l2_memory.get("repair_strategies") or not l1_memory.get("problems"):
        return l2_memory

    problems = l1_memory.get("problems", [])
    print(
        "  WARNING: Full L2 response contained no valid strategies; "
        f"retrying {len(problems)} problem(s) independently"
    )
    failure_identify: list[str] = []
    strategies: list[dict] = []

    for problem in problems:
        problem_id = problem.get("problem_id")
        single_l1 = dict(l1_memory)
        single_l1["problems"] = [problem]
        single_l1["total_problems"] = 1
        single_l2 = build_l2_memory(l1_memory=single_l1, llm=llm)

        for failure in single_l2.get("failure_identify", []):
            if failure not in failure_identify:
                failure_identify.append(failure)

        generated = single_l2.get("repair_strategies", [])
        if generated:
            for strategy in generated:
                strategy = dict(strategy)
                strategy["source_problem_ids"] = [problem_id]
                strategies.append(strategy)
        else:
            print(
                f"  WARNING: L2 response for problem {problem_id} was invalid; "
                "preserving it with a deterministic strategy"
            )
            failure = _failure_label(problem)
            if failure not in failure_identify:
                failure_identify.append(failure)
            strategies.append(_strategy_from_l1_problem(problem))

    for step, strategy in enumerate(strategies, start=1):
        strategy["step"] = step

    l2_memory["failure_identify"] = failure_identify
    l2_memory["repair_strategies"] = strategies
    return l2_memory


def _failure_label(problem: dict) -> str:
    failure_type = problem.get("failure_type") or "unspecified_failure"
    verification_cmd = problem.get("verification_cmd") or "unspecified validator"
    files = problem.get("files") or []
    unit = "file" if len(files) == 1 else "files"
    return f"{failure_type} ({verification_cmd}) - {len(files)} {unit}"


def _strategy_from_l1_problem(problem: dict) -> dict:
    """Mechanical, lossless L2 representation used only after parse failures."""
    files = list(problem.get("files") or [])
    failure = _failure_label(problem)
    verification_cmd = problem.get("verification_cmd") or ""
    fix_strategy = problem.get("fix_strategy") or "Apply the recorded L1 repair."
    locations = ", ".join(files) if files else "the affected project area"
    return {
        "step": 0,
        "failure_type": problem.get("failure_type") or "unspecified_failure",
        "validation_cmd": verification_cmd,
        "applies_to_failures": [failure],
        "causal_chain": problem.get("root_cause") or problem.get("problem") or "",
        "summary": problem.get("problem") or "",
        "intent": fix_strategy,
        "reasoning": problem.get("root_cause") or "",
        "rationale": fix_strategy,
        "when_to_apply": problem.get("problem") or "",
        "signals": [problem.get("problem") or failure],
        "key_actions": [fix_strategy]
        + ([f"Verify: {verification_cmd}"] if verification_cmd else []),
        "pitfalls": [f"Do not omit affected files: {locations}"],
        "example_phrasing": fix_strategy,
        "source_problem_ids": [problem.get("problem_id")],
        "fallback_generated": True,
    }


def _synchronize_l1_with_finalized_decomposition(
    l1_memory: dict,
    decomposed_result: dict,
) -> None:
    """Restore finalized IDs/edges after the reusable L1 generation step."""
    final_problems = decomposed_result.get("problems", [])
    final_dependencies = decomposed_result.get("dependencies", [])
    final_sequence = decomposed_result.get("repair_sequence", [])
    enabled_map: dict[int, list[int]] = {}
    for edge in final_dependencies:
        source, target = edge.get("from"), edge.get("to")
        if isinstance(source, int) and isinstance(target, int):
            enabled_map.setdefault(source, []).append(target)

    generated = l1_memory.get("problems", [])
    generated_by_id = {problem.get("problem_id"): problem for problem in generated}
    synchronized = []
    for final_problem in final_problems:
        problem_id = final_problem.get("problem_id")
        memory_problem = generated_by_id.get(problem_id)
        if memory_problem is None:
            # A reusable L1 prompt may omit a record. Recover it losslessly from
            # the already-finalized decomposition rather than rerunning analysis.
            memory_problem = {
                "problem_id": problem_id,
                "verification_cmd": final_problem.get("validation_cmd", ""),
                "failure_type": final_problem.get("failure_type", ""),
                "problem": final_problem.get("problem", ""),
                "root_cause": final_problem.get("root_cause", ""),
                "fix_strategy": final_problem.get("how_fixed", ""),
                "files": final_problem.get("affected_files", []),
            }
        # The decomposition and L1 schemas use different field names. Always
        # copy the finalized values, even when the reusable builder returned a
        # record for this ID: that record can exist while these fields are empty.
        # The finalized decomposition is the authoritative, lossless source.
        memory_problem["verification_cmd"] = final_problem.get(
            "validation_cmd", ""
        )
        memory_problem["failure_type"] = final_problem.get("failure_type", "")
        memory_problem["problem"] = final_problem.get("problem", "")
        memory_problem["root_cause"] = final_problem.get("root_cause", "")
        memory_problem["fix_strategy"] = final_problem.get("how_fixed", "")
        memory_problem["why_fix_works"] = final_problem.get("why_fix_works", "")
        memory_problem["files"] = list(final_problem.get("affected_files", []))
        memory_problem["enabled"] = sorted(set(enabled_map.get(problem_id, [])))
        synchronized.append(memory_problem)

    sequence_rank = {problem_id: index for index, problem_id in enumerate(final_sequence)}
    synchronized.sort(
        key=lambda problem: sequence_rank.get(problem.get("problem_id"), 10_000)
    )
    l1_memory["problems"] = synchronized
    l1_memory["dependencies"] = final_dependencies
    l1_memory["repair_sequence"] = final_sequence


def _append_to_bidirect_trs(
    l1_memory: dict,
    l2_memory: dict,
    l3_memory: dict,
    issue_id: str,
    output_dir: str,
):
    """Append L1/L2/L3 memory to their respective files (with duplicate filtering)."""
    bidirect_dir = Path(output_dir)
    bidirect_dir.mkdir(parents=True, exist_ok=True)

    # Append L1 to failure_memory.json
    if l1_memory:
        failure_memory_path = bidirect_dir / "failure_memory.json"
        existing = []
        if failure_memory_path.exists():
            with open(failure_memory_path) as f:
                existing = json.load(f)

        # Remove duplicates by issue_id
        existing = [e for e in existing if e.get("issue_id") != issue_id]
        existing.append(l1_memory)

        with open(failure_memory_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  OK Appended issue {issue_id} to failure_memory.json")

    # Append L2 to repo_memory.json
    if l2_memory:
        repo_memory_path = bidirect_dir / "repo_memory.json"
        existing = []
        if repo_memory_path.exists():
            with open(repo_memory_path) as f:
                existing = json.load(f)

        # Remove duplicates by issue_id
        existing = [e for e in existing if e.get("issue_id") != issue_id]
        existing.append(l2_memory)

        with open(repo_memory_path, "w") as f:
            json.dump(existing, f, indent=2)
        print("  OK Appended 1 issue to repo_memory.json")

    # Append L3 to cross_memory.json
    if l3_memory:
        cross_memory_path = bidirect_dir / "cross_memory.json"
        existing = []
        if cross_memory_path.exists():
            with open(cross_memory_path) as f:
                existing = json.load(f)

        # Extract patterns and add source tracking
        patterns = l3_memory.get("universal_patterns", [])
        for pattern in patterns:
            if "source_issue_id" not in pattern:
                pattern["source_issue_id"] = issue_id
            if "source_repo" not in pattern:
                pattern["source_repo"] = l3_memory.get("repo", "")

        # Remove duplicate patterns by pattern_id
        existing_pattern_ids = {p.get("pattern_id") for p in existing}
        new_patterns = [
            p for p in patterns if p.get("pattern_id") not in existing_pattern_ids
        ]

        existing.extend(new_patterns)

        with open(cross_memory_path, "w") as f:
            json.dump(existing, f, indent=2)

        num_patterns = len(new_patterns)
        print(f"  OK Appended {num_patterns} patterns to cross_memory.json")


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


def main():
    parser = argparse.ArgumentParser(
        description="Bidirectional Decomposition: Forward + Backward"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process multiple issues from dataset"
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Load dataset from HuggingFace",
    )
    parser.add_argument(
        "--dataset",
        default="data/memory_set.jsonl",
        help="Path to dataset (default: data/memory_set.jsonl)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model for decomposition (e.g., minimax2.5)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/bidirect_trs",
        help="Output directory (default: data/bidirect_trs)",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of issues to process"
    )
    parser.add_argument(
        "--issue-ids",
        help="Comma-separated issue IDs to process (optional filter)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("BIDIRECTIONAL DECOMPOSITION (Forward + Backward)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}/")
    print("=" * 80)
    print()

    # Initialize LLM
    args.model = configure_model_environment(args.model) or args.model
    llm = LitellmModel(model_name=args.model)

    # Load dataset
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if args.batch:
        if args.use_huggingface:
            dataset = load_dataset("Rabee1983/memory_set", split="train")
            records = [dict(item) for item in dataset]
        else:
            with open(args.dataset) as f:
                records = [json.loads(line) for line in f]
    else:
        # Single issue mode
        if args.issue_ids:
            issue_id = args.issue_ids.split(",")[0]
        else:
            print("Error: --issue-ids required when not using --batch")
            return 1

        # Run single issue
        result = build_unified_decomposition(
            issue_id=issue_id,
            llm=llm,
        )

        if result:
            # Result is already saved inside build_unified_decomposition
            # Build L1/L2/L3 from normalized result
            build_l1_l2_l3_for_bidirectional(result, llm, args.output_dir)

        print("\n" + "=" * 80)
        print("OK Bidirectional decomposition complete!")
        print("=" * 80)
        print("Output saved to:")
        print(f"  - {args.output_dir}/decomposed_issues.json")
        print(f"  - {args.output_dir}/failure_memory.json (L1)")
        print(f"  - {args.output_dir}/repo_memory.json (L2)")
        print(f"  - {args.output_dir}/cross_memory.json (L3)")
        print("=" * 80)
        return 0

    # Filter by issue IDs if specified
    if args.issue_ids:
        filter_ids = set(args.issue_ids.split(","))
        records = [r for r in records if _issue_id(r) in filter_ids]

    # Limit number of records
    if args.limit:
        records = records[: args.limit]

    print(f"Processing {len(records)} issues...")
    print()

    # Load existing memory files to check what's already complete
    decomposed_cache, l1_ids, l2_ids, l3_ids, processed_ids = _load_existing_memory_files(
        Path(args.output_dir)
    )
    print()

    # Fix missing memory: For issues with decomposed data, generate any missing L1/L2/L3
    # Only regenerate if the issue has problems to decompose
    missing_memory_ids = set(decomposed_cache.keys()) - processed_ids
    issues_with_problems = {
        issue_id
        for issue_id, data in decomposed_cache.items()
        if data.get("problems") and len(data.get("problems", [])) > 0
    }
    missing_fixable = missing_memory_ids & issues_with_problems

    if missing_fixable:
        print(
            f"Found {len(missing_fixable)} issues with problems but missing L1/L2/L3"
        )
        print(
            f"Generating memory for: {sorted(list(missing_fixable)[:5])}{'...' if len(missing_fixable) > 5 else ''}"
        )
        print()

        for issue_id in sorted(missing_fixable):
            missing_types = []
            if issue_id not in l1_ids:
                missing_types.append("L1")
            if issue_id not in l2_ids:
                missing_types.append("L2")
            if issue_id not in l3_ids:
                missing_types.append("L3")

            print(
                f"[Fix] Issue {issue_id} - Missing {', '.join(missing_types)} - Generating from decomposed data..."
            )
            try:
                decomposed_result = decomposed_cache[issue_id]
                build_l1_l2_l3_for_bidirectional(
                    decomposed_result, llm, args.output_dir
                )
                print(f"  ✓ Complete\n")
            except Exception as e:
                print(f"  ✗ Failed: {e}\n")
                import traceback

                traceback.print_exc()
                continue

        # Reload to update processed_ids with newly created entries
        _, l1_ids, l2_ids, l3_ids, processed_ids = _load_existing_memory_files(
            Path(args.output_dir)
        )
        print(f"After fix: {len(processed_ids)} complete issues")
        print()

    # Process each issue
    results = []
    skipped_count = 0
    for idx, record in enumerate(records, 1):
        issue_id = _issue_id(record)
        print(f"[{idx}/{len(records)}] Issue {issue_id}")
        print("-" * 80)

        try:
            # Skip if already has complete L1/L2/L3
            if issue_id in processed_ids:
                print(
                    f"  ✓ Already has complete L1/L2/L3 - skipping"
                )
                skipped_count += 1
                continue

            # Check if we have decomposed data (can skip decomposition)
            if issue_id in decomposed_cache:
                print(
                    "  ✓ Found in decomposed_issues.json - Building L1/L2/L3 only (no decomposition needed)"
                )
                result = decomposed_cache[issue_id]
            else:
                # Need to decompose from scratch
                print("  → Running full decomposition...")
                result = build_unified_decomposition(
                    issue_id=issue_id,
                    llm=llm,
                )

                if result:
                    # Save to decomposed_issues.json
                    save_to_decomposed_issues(result, Path(args.output_dir))

            # Build L1/L2/L3 from result (result is already normalized)
            if result:
                build_l1_l2_l3_for_bidirectional(
                    result, llm, args.output_dir
                )
                results.append(result)

            print(f"OK Issue {issue_id} complete\n")

        except Exception as e:
            print(f"ERROR Issue {issue_id}: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    # Note: decomposed_issues.json is already saved incrementally in the loop
    # via save_to_decomposed_issues() - no need to save again here

    print("\n" + "=" * 80)
    print("OK Bidirectional decomposition complete!")
    print("=" * 80)
    print(f"Processed: {len(results)} issues")
    print(f"Skipped: {skipped_count} issues (already complete)")
    print(f"Total: {len(records)} issues")
    print("\nOutput saved to:")
    print(f"  - {args.output_dir}/decomposed_issues.json")
    print(f"  - {args.output_dir}/failure_memory.json (L1)")
    print(f"  - {args.output_dir}/repo_memory.json (L2)")
    print(f"  - {args.output_dir}/cross_memory.json (L3)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
