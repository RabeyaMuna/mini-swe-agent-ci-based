#!/usr/bin/env python3
"""
Complete end-to-end pipeline for building memory from raw issues.

Pipeline:
1. Load raw issues (filter for specific repos if needed)
2. Decompose issues (extract atomic problems, validation sequence)
3. Build L1/L2/L3 memory (clean format - transform only, no LLM calls)

Usage:
    # All issues
    python scripts/build_memory_pipeline.py

    # Flower issues only
    python scripts/build_memory_pipeline.py --repo flower

    # Specific issues by ID
    python scripts/build_memory_pipeline.py --issue-ids 121,123,127
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Note: L1/L2/L3 builders now come from integrate_dependencies
# (enhanced with dependency analysis)

# Import simple LLM for decomposition (NOT the mini-swe-agent one with tool parsing)
from decompose_ci_failure import LitellmModel
try:
    # Try to use enhanced decomposition
    from integrate_ci_enhancements import decompose_issue_enhanced as decompose_issue
    USING_ENHANCED = True
except ImportError:
    # Fall back to original
    from decompose_ci_failure import decompose_issue
    USING_ENHANCED = False


def load_and_filter_raw_issues(
    raw_issues_path: Path,
    repo_filter: str = None,
    issue_ids: list = None
) -> list:
    """
    Load raw issues and filter by repo/issue IDs.

    Args:
        raw_issues_path: Path to raw issues JSON
        repo_filter: Filter by repo name (e.g., "flower")
        issue_ids: Filter by specific issue IDs

    Returns:
        List of filtered raw issues
    """
    with open(raw_issues_path) as f:
        all_issues = json.load(f)

    filtered = []
    for issue in all_issues:
        # Check repo filter
        if repo_filter:
            repo_name = issue.get('repo_name', '').lower()
            if repo_filter.lower() not in repo_name:
                continue

        # Check issue ID filter
        if issue_ids:
            issue_id = str(issue.get('id', ''))
            if issue_id not in issue_ids:
                continue

        filtered.append(issue)

    return filtered


def decompose_issues(raw_issues: list, llm, decomposed_output: Path, reuse_decomposed: bool = True) -> list:
    """
    Decompose raw issues into atomic problems with validation sequence.
    """
    print("\n" + "="*80)
    print("STEP 1: DECOMPOSITION")
    print("="*80)

    sha_fails = {issue.get('sha_fail') for issue in raw_issues}

    if reuse_decomposed and decomposed_output.exists():
        print(f"\nFound existing decomposed issues: {decomposed_output}")
        with open(decomposed_output) as f:
            all_decomposed = json.load(f)

        filtered_decomposed = [
            d for d in all_decomposed
            if d.get('sha_fail') in sha_fails
        ]

        if len(filtered_decomposed) == len(raw_issues):
            print(f"Loaded {len(filtered_decomposed)} decomposed issues")
            return filtered_decomposed

        print(
            f"Existing file covers {len(filtered_decomposed)}/{len(raw_issues)} issues; "
            "decomposing missing issues."
        )
    else:
        filtered_decomposed = []

    existing_shas = {row.get("sha_fail") for row in filtered_decomposed}
    results = list(filtered_decomposed)
    errors = []

    # Prepare output path for incremental saving
    decomposed_output.parent.mkdir(parents=True, exist_ok=True)

    for i, issue in enumerate(raw_issues, 1):
        if issue.get("sha_fail") in existing_shas:
            continue
        print(f"\nDecomposing {i}/{len(raw_issues)}: issue {issue.get('id')} ({str(issue.get('sha_fail', ''))[:12]})")
        result = decompose_issue(issue, llm)
        if "error" in result:
            errors.append(result)
        results.append(result)

        # Incremental save after each issue
        try:
            with open(decomposed_output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ Saved progress ({len(results)} issues total)")
        except Exception as e:
            print(f"  WARNING: Could not save progress: {e}")

    # Final save (ensures completion)
    with open(decomposed_output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved decomposed issues: {decomposed_output}")
    print(f"Successful: {len(results) - len(errors)}")
    print(f"Errors: {len(errors)}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: raw issues → decomposition → memory building"
    )
    parser.add_argument(
        "--raw-issues",
        default="data/trs/memory_seed_issues.json",
        help="Path to raw issues JSON"
    )
    parser.add_argument(
        "--repo",
        help="Filter by repo name (e.g., 'flower')"
    )
    parser.add_argument(
        "--issue-ids",
        help="Comma-separated issue IDs (e.g., '121,123,127')"
    )
    parser.add_argument(
        "--output-dir",
        default="data/trs",
        help="Output directory for L1/L2/L3 memory"
    )
    parser.add_argument(
        "--decomposed-output",
        default="data/trs/decomposed_issues.json",
        help="Where to save/reuse decomposed issue analysis"
    )
    parser.add_argument(
        "--no-reuse-decomposed",
        action="store_true",
        help="Force fresh decomposition even if --decomposed-output already exists"
    )
    parser.add_argument(
        "--model",
        default="openrouter/minimax/minimax-m2.5",
        help="LLM model (for decomposition only)"
    )
    # Keep these for backward compatibility but ignore them
    parser.add_argument(
        "--verified-patches",
        help="(Ignored - kept for backward compatibility)"
    )
    parser.add_argument(
        "--no-backward-conditioning",
        action="store_true",
        help="(Ignored - clean build has no LLM calls)"
    )
    args = parser.parse_args()

    print("="*80)
    print("COMPLETE MEMORY BUILDING PIPELINE (CLEAN FORMAT)")
    if USING_ENHANCED:
        print("Using ENHANCED CI-aware decomposition")
    else:
        print("Using standard decomposition")
    print("L1/L2/L3: Transform-only (no LLM calls, <1s build time)")
    print("="*80)

    # Parse issue IDs if provided
    issue_ids = None
    if args.issue_ids:
        issue_ids = [id.strip() for id in args.issue_ids.split(',')]

    # STEP 0: Load and filter raw issues
    print("\n" + "="*80)
    print("STEP 0: LOAD RAW ISSUES")
    print("="*80)

    raw_issues_path = PROJECT_ROOT / args.raw_issues
    if not raw_issues_path.exists():
        print(f"\n ERROR: Raw issues not found: {raw_issues_path}")
        return 1

    raw_issues = load_and_filter_raw_issues(
        raw_issues_path,
        repo_filter=args.repo,
        issue_ids=issue_ids
    )

    print(f"\nLoaded {len(raw_issues)} raw issues")
    if args.repo:
        print(f"  Filtered by repo: {args.repo}")
    if issue_ids:
        print(f"  Filtered by IDs: {issue_ids}")

    print("\nIssues to process:")
    for issue in raw_issues:
        repo_owner = issue.get('repo_owner', '')
        repo_name = issue.get('repo_name', '')
        issue_id = issue.get('id', '')
        sha_fail = issue.get('sha_fail', '')[:12]
        print(f"  - {repo_owner}/{repo_name} #{issue_id} (SHA: {sha_fail})")

    if not raw_issues:
        print("\n No issues to process after filtering!")
        return 1

    # Initialize LLM
    llm = LitellmModel(model_name=args.model)

    # STEP 1: Decompose issues
    decomposed_output = PROJECT_ROOT / args.decomposed_output
    decomposed_issues = decompose_issues(
        raw_issues,
        llm,
        decomposed_output=decomposed_output,
        reuse_decomposed=not args.no_reuse_decomposed,
    )

    if not decomposed_issues:
        print("\n No decomposed issues available!")
        return 1

    # STEP 1.5: Integrate dependency analysis
    print("\n" + "="*80)
    print("STEP 1.5: INTEGRATE DEPENDENCY ANALYSIS")
    print("="*80)

    from integrate_dependencies import (
        phase1_extract_code_changes,
        phase2_analyze_dependencies
    )

    # Phase 1: Extract code changes (deterministic, always run)
    print("Phase 1: Extracting detailed code changes from validation groups...")
    decomposed_issues = phase1_extract_code_changes(decomposed_issues)

    # Phase 2: Analyze cross-problem dependencies (LLM-based)
    print("\nPhase 2: Analyzing cross-problem dependencies...")
    from build_memory_from_decomposed import SimpleLLM
    llm_for_deps = SimpleLLM(args.model)
    decomposed_issues = phase2_analyze_dependencies(decomposed_issues, llm_for_deps)

    # Save enhanced decomposed issues (overwrites with dependency info)
    with open(decomposed_output, 'w') as f:
        json.dump(decomposed_issues, f, indent=2)
    print(f"✓ Saved enhanced decomposed issues: {decomposed_output}")

    # Initialize LLM for L3 analysis
    llm = llm_for_deps  # Reuse same LLM

    # STEP 2: Build ENHANCED L1 memory (with dependency info)
    print("\n" + "="*80)
    print("STEP 2: BUILD L1 (per-file memory with dependencies)")
    print("="*80)
    print("Extracting from enhanced decomposition:")
    print("  - Detailed code changes (line-by-line)")
    print("  - What this file enables/depends on")
    print("  - Repair order")

    from integrate_dependencies import phase3_enhance_l1
    l1_memories = phase3_enhance_l1(decomposed_issues)

    # STEP 3: Build ENHANCED L2 memory (with problem groups and repair sequence)
    print("\n" + "="*80)
    print("STEP 3: BUILD L2 (per-issue memory with problem groups)")
    print("="*80)
    print("Enhanced format:")
    print("  - Atomic problems with code_changes")
    print("  - Problem groups (foundational, dependent)")
    print("  - Dependency graph")
    print("  - Repair sequence")

    from integrate_dependencies import phase4_enhance_l2
    l2_memories = phase4_enhance_l2(decomposed_issues)

    # STEP 4: Build ENHANCED L3 memory (with dependency patterns)
    print("\n" + "="*80)
    print("STEP 4: BUILD L3 (universal patterns with dependencies)")
    print("="*80)
    print("Analyzing:")
    print("  - Universal failure patterns")
    print("  - Cross-issue dependency patterns")
    print("  - Repair strategies")

    from integrate_dependencies import phase5_enhance_l3
    l3_principles = phase5_enhance_l3(decomposed_issues, llm)

    # STEP 5: Save output
    print("\n" + "="*80)
    print("STEP 5: SAVE OUTPUT")
    print("="*80)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save L1
    l1_path = output_dir / "failure_memory.json"
    with open(l1_path, "w") as f:
        json.dump(l1_memories, f, indent=2)
    print(f"\n✓ Saved L1 memory: {l1_path}")
    print(f"  {len(l1_memories)} per-file entries")

    # Save L2
    l2_path = output_dir / "repo_memory.json"
    with open(l2_path, "w") as f:
        json.dump(l2_memories, f, indent=2)
    print(f"\n✓ Saved L2 memory: {l2_path}")
    print(f"  {len(l2_memories)} per-issue entries")
    total_problems = sum(len(l2.get('atomic_problems', [])) for l2 in l2_memories)
    print(f"  Total atomic problems: {total_problems}")

    # Save L3
    l3_path = output_dir / "cross_memory.json"
    with open(l3_path, "w") as f:
        json.dump(l3_principles, f, indent=2)
    print(f"\n✓ Saved L3 memory: {l3_path}")
    print(f"  {len(l3_principles)} universal patterns")

    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE OK")
    print("="*80)
    print(f"\nProcessed: {len(raw_issues)} raw issues → {len(decomposed_issues)} decomposed → {total_problems} atomic problems")
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    print(f"  - decomposed_issues.json (Input: {len(decomposed_issues)} issues)")
    print(f"  - failure_memory.json    (L1: {len(l1_memories)} per-file entries)")
    print(f"  - repo_memory.json       (L2: {len(l2_memories)} issues, {total_problems} problems)")
    print(f"  - cross_memory.json      (L3: {len(l3_principles)} universal patterns)")

    print("\nPerformance Performance:")
    print(f"  - Format: CLEAN (your design)")
    print(f"  - L1/L2/L3 build: <1 second (transform-only, no LLM)")
    print(f"  - Cost: $0 for memory building")

    print("\nInspect Inspect outputs:")
    print(f"  # View decomposed issue")
    print(f"  cat {decomposed_output} | jq '.[0]'")
    print(f"\n  # View L2 clean format")
    print(f"  cat {l2_path} | jq '.[0].atomic_problems[0]'")
    print(f"\n  # List L3 patterns")
    print(f"  cat {l3_path} | jq '.[] | {{name: .pattern_name, examples: .examples_count}}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
