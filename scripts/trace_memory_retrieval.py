#!/usr/bin/env python3
"""
Memory Retrieval Pipeline Tracer

Shows exactly how memory is fetched, filtered, and synthesized for ONE issue.

Usage:
    python scripts/trace_memory_retrieval.py --issue-id 407
    python scripts/trace_memory_retrieval.py --repo camel --index 0
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minisweagent.run.benchmarks.utils.memory_plugin import CIMemoryPlugin
from minisweagent.run.benchmarks.utils.ci_memory_system import CIMemorySystem
import numpy as np


def load_issue(eval_issues_path: Path, issue_id: int = None, repo: str = None, index: int = None):
    """Load a specific issue from eval_issues.json."""
    with open(eval_issues_path) as f:
        issues = json.load(f)

    if issue_id is not None:
        # Find by ID
        for issue in issues:
            if issue.get('id') == issue_id:
                return issue
        raise ValueError(f"Issue {issue_id} not found")

    elif repo is not None and index is not None:
        # Find by repo and index
        repo_issues = [i for i in issues if i.get('repo') == repo or i.get('repo_name') == repo]
        if index >= len(repo_issues):
            raise ValueError(f"Repo {repo} has {len(repo_issues)} issues, index {index} out of range")
        return repo_issues[index]

    else:
        raise ValueError("Must specify either --issue-id or --repo + --index")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def trace_retrieval(
    issue: dict,
    memory_root: Path,
    ablation: str = "L1+L2",
    model_key: str = "MiniMax-M2.5",
):
    """Trace the complete memory retrieval pipeline."""

    print("="*80)
    print("MEMORY RETRIEVAL PIPELINE TRACE")
    print("="*80)

    # Issue info
    print(f"\n{'='*80}")
    print(f"ISSUE INFO")
    print(f"{'='*80}")
    print(f"ID:        {issue.get('id')}")
    print(f"Repo:      {issue.get('repo') or issue.get('repo_name')}")
    print(f"SHA:       {issue.get('sha_fail', 'N/A')[:12]}")
    print(f"Error:     {issue.get('error_type', 'N/A')}")

    # Build query context
    ci_log = issue.get('ci_log', '')
    error_type = issue.get('error_type', '')
    query_text = f"{error_type}\n{ci_log[:500]}"  # First 500 chars

    print(f"\nQuery Text (first 200 chars):")
    print(f"  {query_text[:200]}...")

    # Load memory files
    print(f"\n{'='*80}")
    print(f"STEP 1: LOAD MEMORY FILES")
    print(f"{'='*80}")

    l1_path = memory_root / "failure_memory.json"
    l2_path = memory_root / "repo_memory.json"
    l3_path = memory_root / "cross_memory.json"

    l1_records = json.loads(l1_path.read_text()) if l1_path.exists() else []
    l2_records = json.loads(l2_path.read_text()) if l2_path.exists() else []
    l3_records = json.loads(l3_path.read_text()) if l3_path.exists() else []

    print(f"L1 (failure_memory):  {len(l1_records)} records")
    print(f"L2 (repo_memory):     {len(l2_records)} records")
    print(f"L3 (cross_memory):    {len(l3_records)} records")

    # Filter by repo for L1/L2
    repo_name = issue.get('repo') or issue.get('repo_name')
    l1_repo = [r for r in l1_records if r.get('repo') == repo_name or r.get('repo_name') == repo_name]
    l2_repo = [r for r in l2_records if r.get('repo') == repo_name or r.get('repo_name') == repo_name]

    print(f"\nFiltered for repo '{repo_name}':")
    print(f"L1: {len(l1_repo)} records")
    print(f"L2: {len(l2_repo)} records")
    print(f"L3: {len(l3_records)} records (repo-agnostic)")

    # Check embeddings
    print(f"\n{'='*80}")
    print(f"STEP 2: CHECK EMBEDDINGS")
    print(f"{'='*80}")

    has_l1_embed = sum(1 for r in l1_repo if 'embedding' in r and r['embedding'])
    has_l2_embed = sum(1 for r in l2_repo if 'embedding' in r and r['embedding'])
    has_l3_embed = sum(1 for r in l3_records if 'embedding' in r and r['embedding'])

    print(f"L1 with embeddings: {has_l1_embed}/{len(l1_repo)}")
    print(f"L2 with embeddings: {has_l2_embed}/{len(l2_repo)}")
    print(f"L3 with embeddings: {has_l3_embed}/{len(l3_records)}")

    if has_l1_embed == 0 and has_l2_embed == 0:
        print("\n⚠️  WARNING: No embeddings found! Run precompute_embeddings.py first")
        print("   Without embeddings, cosine similarity cannot be calculated")
        return

    # Initialize memory system
    print(f"\n{'='*80}")
    print(f"STEP 3: COMPUTE QUERY EMBEDDING")
    print(f"{'='*80}")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = model.encode(query_text)
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Query embedding norm: {np.linalg.norm(query_embedding):.4f}")
    except Exception as e:
        print(f"❌ Failed to compute query embedding: {e}")
        return

    # Compute similarities
    print(f"\n{'='*80}")
    print(f"STEP 4: COMPUTE COSINE SIMILARITIES")
    print(f"{'='*80}")

    # L1 similarities
    l1_with_sim = []
    for record in l1_repo:
        if 'embedding' in record and record['embedding']:
            emb = np.array(record['embedding'])
            sim = cosine_similarity(query_embedding, emb)
            l1_with_sim.append({
                'record': record,
                'similarity': sim,
                'file': record.get('file', 'N/A'),
                'failure_pattern': record.get('failure_pattern', 'N/A')
            })

    l1_with_sim.sort(key=lambda x: x['similarity'], reverse=True)

    print(f"\nL1 Top 10 Similarities:")
    for i, item in enumerate(l1_with_sim[:10], 1):
        print(f"  {i:2d}. {item['similarity']:.4f} - {item['file'][:50]}")

    # L2 similarities
    l2_with_sim = []
    for record in l2_repo:
        if 'embedding' in record and record['embedding']:
            emb = np.array(record['embedding'])
            sim = cosine_similarity(query_embedding, emb)
            l2_with_sim.append({
                'record': record,
                'similarity': sim,
                'error_type': record.get('error_type', 'N/A'),
                'failure_pattern': record.get('failure_pattern', 'N/A')
            })

    l2_with_sim.sort(key=lambda x: x['similarity'], reverse=True)

    print(f"\nL2 Top 10 Similarities:")
    for i, item in enumerate(l2_with_sim[:10], 1):
        print(f"  {i:2d}. {item['similarity']:.4f} - {item['failure_pattern'][:50]}")

    # L3 similarities
    l3_with_sim = []
    for record in l3_records:
        if 'embedding' in record and record['embedding']:
            emb = np.array(record['embedding'])
            sim = cosine_similarity(query_embedding, emb)
            l3_with_sim.append({
                'record': record,
                'similarity': sim,
                'principle': record.get('principle', 'N/A')[:50]
            })

    l3_with_sim.sort(key=lambda x: x['similarity'], reverse=True)

    print(f"\nL3 Top 10 Similarities:")
    for i, item in enumerate(l3_with_sim[:10], 1):
        print(f"  {i:2d}. {item['similarity']:.4f} - {item['principle']}")

    # Apply threshold filtering
    print(f"\n{'='*80}")
    print(f"STEP 5: THRESHOLD FILTERING")
    print(f"{'='*80}")

    # Get threshold from ablation config
    ablation_thresholds = {
        "L1": 0.10,
        "L1+L2": 0.10,
        "L1+L2+L3": 0.10
    }
    threshold = ablation_thresholds.get(ablation, 0.10)

    print(f"Ablation: {ablation}")
    print(f"Threshold: {threshold}")

    # For L1+L2, use weighted similarity
    if ablation == "L1+L2":
        print(f"\nWeighted Similarity = 0.556 × L1_score + 0.444 × L2_score")

        # Get top L1 and L2 for this issue
        top_l1 = l1_with_sim[0]['similarity'] if l1_with_sim else 0.0
        top_l2 = l2_with_sim[0]['similarity'] if l2_with_sim else 0.0
        weighted = 0.556 * top_l1 + 0.444 * top_l2

        print(f"Top L1: {top_l1:.4f}")
        print(f"Top L2: {top_l2:.4f}")
        print(f"Weighted: {weighted:.4f}")
        print(f"Passes threshold: {weighted >= threshold}")

    l1_filtered = [x for x in l1_with_sim if x['similarity'] >= threshold]
    l2_filtered = [x for x in l2_with_sim if x['similarity'] >= threshold]
    l3_filtered = [x for x in l3_with_sim if x['similarity'] >= threshold]

    print(f"\nAfter threshold filtering:")
    print(f"L1: {len(l1_filtered)}/{len(l1_with_sim)} passed")
    print(f"L2: {len(l2_filtered)}/{len(l2_with_sim)} passed")
    print(f"L3: {len(l3_filtered)}/{len(l3_with_sim)} passed")

    # Show what passed
    if l1_filtered:
        print(f"\nL1 Passed (top 5):")
        for i, item in enumerate(l1_filtered[:5], 1):
            print(f"  {i}. {item['similarity']:.4f} - {item['file']}")

    if l2_filtered:
        print(f"\nL2 Passed (top 5):")
        for i, item in enumerate(l2_filtered[:5], 1):
            print(f"  {i}. {item['similarity']:.4f} - {item['failure_pattern'][:60]}")

    if l3_filtered:
        print(f"\nL3 Passed (top 5):")
        for i, item in enumerate(l3_filtered[:5], 1):
            print(f"  {i}. {item['similarity']:.4f} - {item['principle']}")

    # LLM filtering (Step 6)
    print(f"\n{'='*80}")
    print(f"STEP 6: LLM1 RELEVANCE FILTERING")
    print(f"{'='*80}")

    if not l1_filtered and not l2_filtered:
        print("⚠️  No candidates passed threshold, LLM1 will not be called")
        print("   System will proceed WITHOUT memory")
        return

    print(f"Would call LLM1 with:")
    print(f"  - Query: {error_type}")
    print(f"  - L1 candidates: {len(l1_filtered)}")
    print(f"  - L2 candidates: {len(l2_filtered)}")
    print(f"  - L3 candidates: {len(l3_filtered)}")

    print(f"\nLLM1 task: Select which memories are ACTUALLY relevant")
    print(f"  Input: CI error + memory summaries")
    print(f"  Output: List of relevant memory IDs")

    # Show example L2 that would be sent to LLM1
    if l2_filtered:
        print(f"\nExample L2 sent to LLM1:")
        top_l2 = l2_filtered[0]['record']
        print(f"  error_type: {top_l2.get('error_type', 'N/A')}")
        print(f"  failure_pattern: {top_l2.get('failure_pattern', 'N/A')}")
        print(f"  overall_failure_reason: {top_l2.get('overall_failure_reason', 'N/A')[:100]}...")
        print(f"  fix_strategy: {top_l2.get('fix_strategy', 'N/A')[:100]}...")
        print(f"  changed_files: {top_l2.get('changed_files', [])}")

    # LLM synthesis (Step 7)
    print(f"\n{'='*80}")
    print(f"STEP 7: LLM2 EXPERIENCE SYNTHESIS")
    print(f"{'='*80}")

    print(f"LLM2 task: Synthesize memories into actionable guidance")
    print(f"  Input: CI error + LLM1-selected memories")
    print(f"  Output: Structured guidance with:")
    print(f"    - diagnosis: What's likely wrong")
    print(f"    - full_scope: Complete problem understanding")
    print(f"    - fix_approach: Step-by-step fix strategy")
    print(f"    - verification: How to verify the fix")

    # Show memory structure quality
    print(f"\n{'='*80}")
    print(f"STEP 8: MEMORY STRUCTURE QUALITY CHECK")
    print(f"{'='*80}")

    if l2_filtered:
        top_l2_record = l2_filtered[0]['record']

        # Check critical fields
        has_failure_reason = bool(top_l2_record.get('overall_failure_reason') and
                                   top_l2_record.get('overall_failure_reason') != 'N/A')
        has_fix_strategy = bool(top_l2_record.get('fix_strategy') and
                               len(top_l2_record.get('fix_strategy', '')) > 50)

        changed_files = top_l2_record.get('changed_files', [])
        has_changed_files = bool(changed_files)

        # Check changed_files structure
        if changed_files:
            is_string_list = isinstance(changed_files[0], str)
            is_object_list = isinstance(changed_files[0], dict) and 'file' in changed_files[0]

            print(f"\nTop L2 Memory Quality:")
            print(f"  ✅ Has overall_failure_reason: {has_failure_reason}")
            print(f"  ✅ Has fix_strategy: {has_fix_strategy}")
            print(f"  ✅ Has changed_files: {has_changed_files}")

            if is_string_list:
                print(f"  ❌ changed_files is STRING LIST: {changed_files}")
                print(f"     Problem: No per-file details (reason, failure_reason, fix_strategy)")
                print(f"     Impact: LLM2 can't generate specific guidance")
            elif is_object_list:
                cf = changed_files[0]
                has_reason = 'reason' in cf
                has_failure_reason = 'failure_reason' in cf
                has_fix_strategy_cf = 'fix_strategy' in cf

                print(f"  ✅ changed_files is OBJECT LIST")
                print(f"     - Has 'reason': {has_reason}")
                print(f"     - Has 'failure_reason': {has_failure_reason}")
                print(f"     - Has 'fix_strategy': {has_fix_strategy_cf}")

                if has_reason and has_failure_reason and has_fix_strategy_cf:
                    print(f"  ✅ STRUCTURE IS CORRECT!")
                else:
                    print(f"  ⚠️  Missing some fields in changed_files objects")
        else:
            print(f"\n  ❌ changed_files is EMPTY!")
            print(f"     Impact: No file-level guidance, agent doesn't know what to modify")

    # Final output
    print(f"\n{'='*80}")
    print(f"STEP 9: FINAL MEMORY CONTEXT (goes to agent)")
    print(f"{'='*80}")

    if not l1_filtered and not l2_filtered:
        print("❌ NO MEMORY USED")
        print("   Agent will receive problem statement WITHOUT memory guidance")
    else:
        print("✅ MEMORY WILL BE USED")
        print(f"   Agent will receive:")
        print(f"   - CI error description")
        print(f"   - Diagnosis from LLM2")
        print(f"   - Full scope analysis")
        print(f"   - Fix approach (step-by-step)")
        print(f"   - Verification steps")

        if l2_filtered and l2_filtered[0]['record'].get('changed_files'):
            cf = l2_filtered[0]['record']['changed_files']
            if isinstance(cf[0], str):
                print(f"\n   ⚠️  Quality: POOR (generic guidance)")
                print(f"       Agent knows 'something' needs fixing but not WHAT")
            else:
                print(f"\n   ✅ Quality: GOOD (specific guidance)")
                print(f"       Agent knows EXACTLY what to fix and HOW")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")

    print(f"\nRetrieval Pipeline:")
    print(f"  1. Load memory:        {len(l1_repo)} L1, {len(l2_repo)} L2, {len(l3_records)} L3")
    print(f"  2. Compute similarity: Top L1={l1_with_sim[0]['similarity']:.4f if l1_with_sim else 0:.4f}, Top L2={l2_with_sim[0]['similarity']:.4f if l2_with_sim else 0:.4f}")
    print(f"  3. Threshold filter:   {len(l1_filtered)} L1, {len(l2_filtered)} L2 passed (threshold={threshold})")
    print(f"  4. LLM1 filter:        Would select most relevant")
    print(f"  5. LLM2 synthesis:     Would generate guidance")
    print(f"  6. Format context:     Pass to agent")

    use_memory = len(l1_filtered) > 0 or len(l2_filtered) > 0
    print(f"\nMemory used: {'✅ YES' if use_memory else '❌ NO'}")

    if use_memory and l2_filtered:
        cf = l2_filtered[0]['record'].get('changed_files', [])
        if cf and isinstance(cf[0], str):
            print(f"Memory quality: ⚠️  POOR (needs rebuild)")
        elif cf and isinstance(cf[0], dict) and 'fix_strategy' in cf[0]:
            print(f"Memory quality: ✅ GOOD")
        else:
            print(f"Memory quality: ⚠️  INCOMPLETE")


def main():
    parser = argparse.ArgumentParser(description="Trace memory retrieval for one issue")
    parser.add_argument("--issue-id", type=int, help="Issue ID to trace")
    parser.add_argument("--repo", type=str, help="Repo name (use with --index)")
    parser.add_argument("--index", type=int, help="Index within repo (use with --repo)")
    parser.add_argument("--eval-issues", type=str, default="data/trs/eval_issues.json",
                       help="Path to eval_issues.json")
    parser.add_argument("--memory-root", type=str, default="data/trs",
                       help="Path to memory directory")
    parser.add_argument("--ablation", type=str, default="L1+L2",
                       choices=["L1", "L1+L2", "L1+L2+L3"],
                       help="Ablation mode")

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    eval_issues_path = project_root / args.eval_issues
    memory_root = project_root / args.memory_root

    # Load issue
    try:
        issue = load_issue(eval_issues_path, args.issue_id, args.repo, args.index)
    except Exception as e:
        print(f"❌ Error loading issue: {e}")
        return 1

    # Trace retrieval
    try:
        trace_retrieval(issue, memory_root, args.ablation)
    except Exception as e:
        print(f"\n❌ Error during tracing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
