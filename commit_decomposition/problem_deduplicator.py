#!/usr/bin/env python3
"""
problem_deduplicator.py - Cluster and merge similar problems
"""

import json
from typing import List, Dict


def calculate_similarity(problem1: Dict, problem2: Dict) -> float:
    """
    Calculate similarity between two problems (0.0 to 1.0)

    Factors:
    - Same files: +0.4
    - Similar problem text: +0.3
    - Same failure family: +0.2
    - Same validation_cmd: +0.1
    """
    score = 0.0

    # Same files
    files1 = set(problem1.get("files", []))
    files2 = set(problem2.get("files", []))
    if files1 and files2:
        file_overlap = len(files1 & files2) / len(files1 | files2)
        score += file_overlap * 0.4

    # Similar problem description
    prob1_text = problem1.get("problem", "").lower()
    prob2_text = problem2.get("problem", "").lower()
    if prob1_text and prob2_text:
        # Simple word overlap
        words1 = set(prob1_text.split())
        words2 = set(prob2_text.split())
        if words1 and words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
            score += word_overlap * 0.3

    # Same failure family
    if (
        problem1.get("failure_type") == problem2.get("failure_type")
        and problem1.get("issue_type") == problem2.get("issue_type")
    ):
        score += 0.2

    # Same validation command
    if problem1.get("validation_cmd") == problem2.get("validation_cmd"):
        score += 0.1

    return score


def cluster_similar_problems(problems: List[Dict], threshold: float = 0.7) -> List[List[int]]:
    """
    Cluster problems by similarity

    Returns list of clusters, where each cluster is list of problem indices

    Args:
        problems: List of problem dicts
        threshold: Similarity threshold (0.7 = 70% similar)

    Returns:
        [[0, 2], [1], [3, 4]] means problems 0&2 are similar, 1 is alone, 3&4 are similar
    """
    n = len(problems)
    if n == 0:
        return []

    clusters = []
    assigned = set()

    for i in range(n):
        if i in assigned:
            continue

        # Start new cluster
        cluster = [i]
        assigned.add(i)

        # Find similar problems
        for j in range(i + 1, n):
            if j in assigned:
                continue

            similarity = calculate_similarity(problems[i], problems[j])
            if similarity >= threshold:
                cluster.append(j)
                assigned.add(j)

        clusters.append(cluster)

    return clusters


def merge_problems_with_llm(similar_problems: List[Dict], analyzer) -> Dict:
    """
    Use LLM to merge similar problems into one

    Args:
        similar_problems: List of similar problem dicts
        analyzer: CommitAnalyzer with LLM access

    Returns:
        Merged problem dict
    """
    if len(similar_problems) == 1:
        return similar_problems[0]

    problems_json = json.dumps(similar_problems, indent=2)

    prompt = f"""You have {len(similar_problems)} similar problems that need to be merged into ONE problem.

SIMILAR PROBLEMS:
{problems_json}

TASK:
Analyze these similar problems and merge them into a single comprehensive problem.

Consider:
1. Combine all affected files
2. Merge problem descriptions (keep the most complete one)
3. Merge root causes (if different, explain both)
4. Combine changes made across all occurrences
5. Preserve commit/job context from all merged records
6. Keep all validation commands mentioned
7. Treat current_fixed_jobs as jobs that passed in the commit(s), preserving the input job status evidence

OUTPUT JSON (valid JSON only, no markdown):
{{
  "files": ["all", "files", "mentioned"],
  "failure_type": "shared failure family",
  "issue_type": "shared issue family",
  "problem": "Merged comprehensive problem description",
  "root_cause": "Combined root cause explanation",
  "changes_made": "All changes made across occurrences",
  "introduces": "Problems or validation challenges introduced, if any",
  "fixes": "What was fixed and why the changes fix it",
  "current_failed_jobs": [],
  "current_fixed_jobs": [],
  "validation_cmd": "Primary validation command",
  "commit_sha": "commit SHA if known",
  "commit_message": "commit message if known",
  "sha_success": "known passing base SHA if known",
  "sha_fail": "known failing head SHA if known",
  "commit": "comma-separated list if multiple",
  "merge_note": "Note that this was merged from {len(similar_problems)} similar problems"
}}"""

    try:
        response = analyzer._call_llm(prompt)
        result = analyzer._parse_response(response)
        return result
    except Exception as e:
        print(f"    Warning: LLM merge failed: {e}, keeping first problem")
        return similar_problems[0]


def deduplicate_problems(problems: List[Dict], analyzer, similarity_threshold: float = 0.7) -> List[Dict]:
    """
    Main function: Cluster and merge similar problems

    Args:
        problems: List of all problems
        analyzer: CommitAnalyzer for LLM access
        similarity_threshold: How similar to consider merging (0.7 = 70%)

    Returns:
        Deduplicated list of problems
    """
    if len(problems) <= 1:
        return problems

    print(f"    Deduplicating {len(problems)} problems...")

    # Cluster similar problems
    clusters = cluster_similar_problems(problems, similarity_threshold)

    print(f"    Found {len(clusters)} clusters")

    # Merge each cluster
    deduplicated = []
    for i, cluster in enumerate(clusters):
        if len(cluster) == 1:
            # Single problem, keep as is
            deduplicated.append(problems[cluster[0]])
        else:
            # Multiple similar problems, merge with LLM
            print(f"    Merging cluster {i+1}: {len(cluster)} similar problems")
            similar_probs = [problems[idx] for idx in cluster]
            merged = merge_problems_with_llm(similar_probs, analyzer)
            deduplicated.append(merged)

    print(f"    Reduced from {len(problems)} to {len(deduplicated)} problems")

    return deduplicated
