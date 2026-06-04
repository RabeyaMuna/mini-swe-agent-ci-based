#!/usr/bin/env python3
"""
check_similarity.py - Memory Similarity Checker
===============================================

Check cosine similarity between issues and their memory retrieval.
Useful for verifying that similar issues can find each other's memories.

Usage:
    # Check similarity between two specific issues
    python scripts/check_similarity.py --issue-1 407 --issue-2 410

    # Check what memories a specific issue retrieves
    python scripts/check_similarity.py --query-issue 410 --top-k 5

    # Batch check all pairs in eval set
    python scripts/check_similarity.py --batch-eval --threshold 0.70
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_embeddings(memory_path: Path) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """Load memory records and their embeddings."""
    records = []
    embeddings = {}

    if memory_path.exists():
        with open(memory_path) as f:
            records = json.load(f)

        for record in records:
            sha = record.get("sha_fail", "")
            emb = record.get("embedding")

            if sha and emb:
                embeddings[sha] = np.array(emb)

    return records, embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_query_embedding(issue: Dict, model) -> np.ndarray:
    """Build query embedding for an issue."""
    error_type = issue.get("error_type", "")
    if isinstance(error_type, list):
        error_type = " ".join(error_type)

    # Build query text
    query_parts = [
        f"Error: {error_type}",
        f"Files: {' '.join(issue.get('changed_files', [])[:5])}",
    ]

    # Add CI log snippet
    logs = issue.get("logs", "")
    if isinstance(logs, list):
        logs = " ".join([str(l) for l in logs[:3]])
    query_parts.append(f"Log: {str(logs)[:500]}")

    query_text = "\n".join(query_parts)

    # Encode
    return model.encode(query_text)


def check_pair_similarity(
    issue1: Dict,
    issue2: Dict,
    l2_records: List[Dict],
    l2_embeddings: Dict[str, np.ndarray],
    model,
) -> Dict:
    """Check similarity between two specific issues."""
    sha1 = issue1.get("sha_fail", "")
    sha2 = issue2.get("sha_fail", "")
    id1 = issue1.get("id", "")
    id2 = issue2.get("id", "")

    print(f"\n{'='*80}")
    print(f"SIMILARITY CHECK: Issue {id1} vs Issue {id2}")
    print(f"{'='*80}")

    # Find their memories
    mem1 = next((r for r in l2_records if r.get("sha_fail", "").startswith(sha1[:12])), None)
    mem2 = next((r for r in l2_records if r.get("sha_fail", "").startswith(sha2[:12])), None)

    if not mem1:
        print(f"❌ No L2 memory found for issue {id1} ({sha1[:12]})")
        return {}

    if not mem2:
        print(f"❌ No L2 memory found for issue {id2} ({sha2[:12]})")
        return {}

    # Get embeddings
    emb1 = l2_embeddings.get(sha1)
    emb2 = l2_embeddings.get(sha2)

    if emb1 is None or emb2 is None:
        print(f"❌ No embeddings found, computing on the fly...")

        # Build query for issue 2
        query_emb = build_query_embedding(issue2, model)

        # Get memory 1 embedding
        if emb1 is None:
            # Build embedding for memory 1
            mem1_text = f"{mem1.get('error_type', '')} {mem1.get('failure_pattern', '')} {mem1.get('failure_reason', '')}"
            emb1 = model.encode(mem1_text)

        similarity = cosine_similarity(query_emb, emb1)
    else:
        similarity = cosine_similarity(emb1, emb2)

    # Display comparison
    print(f"\n### Issue {id1} (SHA: {sha1[:12]})")
    print(f"Error Type: {issue1.get('error_type', 'N/A')}")
    print(f"Changed Files: {issue1.get('changed_files', [])[:3]}...")

    print(f"\n### Issue {id2} (SHA: {sha2[:12]})")
    print(f"Error Type: {issue2.get('error_type', 'N/A')}")
    print(f"Changed Files: {issue2.get('changed_files', [])[:3]}...")

    print(f"\n### Memory Comparison")
    print(f"\nIssue {id1} Memory:")
    print(f"  Error: {mem1.get('error_type', 'N/A')}")
    print(f"  Pattern: {mem1.get('failure_pattern', 'N/A')}")
    print(f"  Changed Files: {mem1.get('changed_files', [])}")

    print(f"\nIssue {id2} Memory:")
    print(f"  Error: {mem2.get('error_type', 'N/A')}")
    print(f"  Pattern: {mem2.get('failure_pattern', 'N/A')}")
    print(f"  Changed Files: {mem2.get('changed_files', [])}")

    print(f"\n### Similarity Score")
    print(f"Cosine Similarity: {similarity:.4f}")

    if similarity >= 0.70:
        print(f"✅ STRONG MATCH (should retrieve)")
    elif similarity >= 0.40:
        print(f"⚠️  MEDIUM MATCH (might retrieve)")
    else:
        print(f"❌ WEAK MATCH (likely won't retrieve)")

    return {
        "issue1_id": id1,
        "issue2_id": id2,
        "similarity": similarity,
        "issue1_mem": mem1,
        "issue2_mem": mem2,
    }


def check_query_retrieval(
    query_issue: Dict,
    all_issues: List[Dict],
    l2_records: List[Dict],
    l2_embeddings: Dict[str, np.ndarray],
    model,
    top_k: int = 5,
) -> List[Dict]:
    """Check what memories a query issue would retrieve."""
    query_id = query_issue.get("id", "")
    query_sha = query_issue.get("sha_fail", "")

    print(f"\n{'='*80}")
    print(f"RETRIEVAL ANALYSIS: Issue {query_id}")
    print(f"{'='*80}")

    print(f"\n### Query Issue")
    print(f"ID: {query_id}")
    print(f"SHA: {query_sha[:12]}")
    print(f"Error: {query_issue.get('error_type', 'N/A')}")
    print(f"Changed Files: {query_issue.get('changed_files', [])[:3]}...")

    # Build query embedding
    query_emb = build_query_embedding(query_issue, model)

    # Compute similarities to all memories
    similarities = []

    for record in l2_records:
        sha = record.get("sha_fail", "")

        # Skip self
        if sha == query_sha:
            continue

        emb = l2_embeddings.get(sha)
        if emb is None:
            continue

        sim = cosine_similarity(query_emb, emb)

        similarities.append({
            "sha": sha,
            "repo": record.get("repo_name", record.get("repo", "")),
            "similarity": sim,
            "error_type": record.get("error_type", ""),
            "failure_pattern": record.get("failure_pattern", ""),
            "changed_files": record.get("changed_files", []),
        })

    # Sort by similarity
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    print(f"\n### Top {top_k} Retrieved Memories")
    print(f"{'Rank':<6} {'Similarity':<12} {'Repo':<15} {'Error Type':<40} {'Files'}")
    print(f"{'-'*100}")

    for i, match in enumerate(similarities[:top_k], 1):
        files_str = str(match['changed_files'])[:40]
        print(f"{i:<6} {match['similarity']:<12.4f} {match['repo']:<15} {str(match['error_type'])[:40]:<40} {files_str}")

    # Analysis
    strong_matches = [m for m in similarities if m["similarity"] >= 0.70]
    medium_matches = [m for m in similarities if 0.40 <= m["similarity"] < 0.70]

    print(f"\n### Summary")
    print(f"Total memories: {len(similarities)}")
    print(f"Strong matches (≥0.70): {len(strong_matches)}")
    print(f"Medium matches (0.40-0.70): {len(medium_matches)}")
    print(f"Weak matches (<0.40): {len(similarities) - len(strong_matches) - len(medium_matches)}")

    return similarities[:top_k]


def batch_eval_similarities(
    eval_issues: List[Dict],
    l2_records: List[Dict],
    l2_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.70,
) -> None:
    """Batch check similarities between all pairs in eval set."""
    print(f"\n{'='*80}")
    print(f"BATCH SIMILARITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Eval issues: {len(eval_issues)}")
    print(f"Similarity threshold: {threshold}")

    # Build SHA → issue lookup
    issue_by_sha = {}
    for issue in eval_issues:
        sha = issue.get("sha_fail", "")
        if sha:
            issue_by_sha[sha] = issue

    # Check all pairs
    results = []

    for i, issue1 in enumerate(eval_issues):
        sha1 = issue1.get("sha_fail", "")
        emb1 = l2_embeddings.get(sha1)

        if not emb1:
            continue

        for issue2 in eval_issues[i+1:]:
            sha2 = issue2.get("sha_fail", "")
            emb2 = l2_embeddings.get(sha2)

            if not emb2:
                continue

            sim = cosine_similarity(emb1, emb2)

            if sim >= threshold:
                results.append({
                    "issue1": issue1.get("id", ""),
                    "issue2": issue2.get("id", ""),
                    "repo1": issue1.get("repo_name", ""),
                    "repo2": issue2.get("repo_name", ""),
                    "similarity": sim,
                })

    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)

    print(f"\n### Strong Matches (≥{threshold})")
    print(f"{'Issue 1':<10} {'Issue 2':<10} {'Repo 1':<15} {'Repo 2':<15} {'Similarity'}")
    print(f"{'-'*70}")

    for match in results:
        print(f"{match['issue1']:<10} {match['issue2']:<10} {match['repo1']:<15} {match['repo2']:<15} {match['similarity']:.4f}")

    print(f"\nTotal strong matches: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Check memory similarity")
    parser.add_argument("--eval-issues", default="data/trs/eval_issues.json", help="Path to eval issues")
    parser.add_argument("--memory-root", default="data/trs", help="Path to memory directory")
    parser.add_argument("--issue-1", help="First issue ID to compare")
    parser.add_argument("--issue-2", help="Second issue ID to compare")
    parser.add_argument("--query-issue", help="Issue ID to query (find similar memories)")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results for query mode")
    parser.add_argument("--batch-eval", action="store_true", help="Batch check all pairs")
    parser.add_argument("--threshold", type=float, default=0.70, help="Similarity threshold for strong match")
    args = parser.parse_args()

    # Load eval issues
    eval_path = Path(args.eval_issues)
    if not eval_path.exists():
        print(f"❌ Eval issues not found: {eval_path}")
        return 1

    with open(eval_path) as f:
        eval_issues = json.load(f)

    # Load memory
    memory_root = Path(args.memory_root)
    l2_records, l2_embeddings = load_embeddings(memory_root / "repo_memory.json")

    if not l2_records:
        print(f"❌ No L2 memory found in {memory_root}")
        return 1

    if not l2_embeddings:
        print(f"⚠️  No embeddings found, will compute on the fly")
        print(f"   Run: python scripts/precompute_embeddings.py --memory-root {memory_root}")

    # Initialize embedding model
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except ImportError:
        print(f"❌ sentence-transformers not installed")
        print(f"   Run: pip install sentence-transformers")
        return 1

    # Mode: pair comparison
    if args.issue_1 and args.issue_2:
        issue1 = next((i for i in eval_issues if str(i.get("id")) == args.issue_1), None)
        issue2 = next((i for i in eval_issues if str(i.get("id")) == args.issue_2), None)

        if not issue1:
            print(f"❌ Issue {args.issue_1} not found")
            return 1
        if not issue2:
            print(f"❌ Issue {args.issue_2} not found")
            return 1

        check_pair_similarity(issue1, issue2, l2_records, l2_embeddings, model)

    # Mode: query retrieval
    elif args.query_issue:
        query_issue = next((i for i in eval_issues if str(i.get("id")) == args.query_issue), None)

        if not query_issue:
            print(f"❌ Issue {args.query_issue} not found")
            return 1

        check_query_retrieval(query_issue, eval_issues, l2_records, l2_embeddings, model, args.top_k)

    # Mode: batch eval
    elif args.batch_eval:
        batch_eval_similarities(eval_issues, l2_records, l2_embeddings, args.threshold)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
