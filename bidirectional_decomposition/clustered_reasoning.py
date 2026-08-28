#!/usr/bin/env python3
"""
Clustered Bidirectional Reasoning
==================================

Smart approach using cosine similarity clustering + batch LLM analysis:

1. Flatten forward and backward with F1, F2, B1, B2 labels
2. Cluster problems by cosine similarity on:
   - Files (affected_files)
   - Problem descriptions
   - Root causes
   - Commands (verification_cmd)
   - Failed jobs
3. Analyze each cluster against failed jobs + CI sequence
4. Merge cluster results
5. Handle unmatched problems (F-only or B-only)
6. Order by dependencies (forward + backward + CI sequence)

Benefits:
- Smaller prompts (process clusters, not all at once)
- Intelligent pre-grouping (similar problems together)
- Still analyzes against CI/failed jobs
- Scalable to large datasets
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from utilities.llm_invoker import invoke_llm_with_retry, get_model_max_output_tokens


def flatten_and_label_problems(
    forward_problems: List[Dict[str, Any]],
    backward_problems: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Flatten forward and backward problems with F1, F2, B1, B2 labels.

    Returns:
        List of all problems with source labels
    """
    flattened = []

    # Forward problems: F1, F2, ...
    for i, problem in enumerate(forward_problems):
        flattened.append({
            **problem,
            "source_id": f"F{i+1}",
            "source": "forward",
            "_original_index": i,
        })

    # Backward problems: B1, B2, ...
    for i, problem in enumerate(backward_problems):
        flattened.append({
            **problem,
            "source_id": f"B{i+1}",
            "source": "backward",
            "_original_index": i,
        })

    return flattened


def compute_text_embedding_simple(text: str) -> np.ndarray:
    """
    Simple text embedding using character n-grams (no ML model needed).

    For production, use sentence-transformers or OpenAI embeddings.
    This is a lightweight fallback.
    """
    if not text:
        return np.zeros(100)

    # Character 3-grams
    text = text.lower()
    ngrams = [text[i:i+3] for i in range(len(text)-2)]

    # Create fixed-size vector
    vector = np.zeros(100)
    for i, ngram in enumerate(ngrams[:100]):
        vector[i % 100] += hash(ngram) % 100

    # Normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector


def compute_problem_embedding(problem: Dict[str, Any]) -> np.ndarray:
    """
    Compute embedding for a problem based on multiple fields.

    Combines:
    - Files (affected_files) - most important for matching
    - Failure type - strong signal
    - Problem description
    - Root cause
    - Verification command
    """
    # Extract fields
    files = " ".join(problem.get("affected_files", []) or problem.get("files", []))
    problem_text = problem.get("problem", "")
    root_cause = problem.get("root_cause", "")
    cmd = problem.get("verification_cmd", "") or problem.get("validation_cmd", "")
    failure_type = problem.get("failure_type", "")
    issue_type = problem.get("issue_type", "")

    # Files are THE MOST important signal (same file = likely same issue)
    # Combine with heavy weighting on files and types
    combined = (
        f"FILES:{files} {files} {files} {files} {files} "  # Files 5x weight
        f"FAILURE:{failure_type} {failure_type} {failure_type} "  # Failure type 3x
        f"ISSUE:{issue_type} {issue_type} "  # Issue type 2x
        f"PROBLEM:{problem_text} "
        f"ROOT:{root_cause} "
        f"CMD:{cmd}"
    )

    return compute_text_embedding_simple(combined)


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def cluster_problems_by_similarity(
    problems: List[Dict[str, Any]],
    similarity_threshold: float = 0.7,
) -> List[List[Dict[str, Any]]]:
    """
    Cluster problems by cosine similarity using file-first matching.

    Strategy:
    1. First group by exact file overlap (strong signal)
    2. Then check cosine similarity within file groups
    3. Problems with no file overlap go to separate clusters

    Args:
        problems: Flattened problems with source_id
        similarity_threshold: Minimum similarity to cluster together (0.7 = 70%)

    Returns:
        List of clusters (each cluster is a list of similar problems)
    """
    if not problems:
        return []

    # Compute embeddings
    embeddings = [compute_problem_embedding(p) for p in problems]

    # Helper: Get files for a problem
    def get_files(problem):
        files = problem.get("affected_files", []) or problem.get("files", [])
        return set(files) if files else set()

    # Helper: Check if files overlap
    def has_file_overlap(p1, p2):
        files1 = get_files(p1)
        files2 = get_files(p2)
        if not files1 or not files2:
            return False
        return len(files1 & files2) > 0

    # Stricter greedy clustering with file-first matching
    clusters = []
    used = set()

    for i, problem in enumerate(problems):
        if i in used:
            continue

        # Start new cluster
        cluster = [problem]
        used.add(i)

        # Find similar problems (must have file overlap OR very high similarity)
        for j, other_problem in enumerate(problems):
            if j in used or i == j:
                continue

            # Compute similarity
            similarity = compute_cosine_similarity(embeddings[i], embeddings[j])
            file_overlap = has_file_overlap(problem, other_problem)

            # Cluster if:
            # 1. Files overlap AND similarity >= threshold, OR
            # 2. Very high similarity (>= 0.85) even without file overlap
            should_cluster = (
                (file_overlap and similarity >= similarity_threshold) or
                (similarity >= 0.85)
            )

            if should_cluster:
                cluster.append(other_problem)
                used.add(j)

        clusters.append(cluster)

    return clusters


def analyze_cluster_with_llm(
    cluster: List[Dict[str, Any]],
    cluster_id: int,
    ci_context: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """
    Analyze one cluster against failed jobs + CI sequence.

    Returns:
        {
            "cluster_id": int,
            "unified_problem": {...},  # Merged from cluster
            "source_ids": ["F1", "B2"],  # Which problems contributed
            "confidence": float,
        }
    """
    print(f"    Analyzing cluster {cluster_id} ({len(cluster)} problems)...")

    # Extract source IDs
    source_ids = [p["source_id"] for p in cluster]
    has_forward = any(p["source"] == "forward" for p in cluster)
    has_backward = any(p["source"] == "backward" for p in cluster)

    prompt = f"""Analyze this cluster of similar problems from bidirectional decomposition.

CLUSTER (similar problems):
{json.dumps(cluster, indent=2)}

FAILED CI JOBS:
{json.dumps(ci_context.get("failed_jobs", []), indent=2)}

CI VALIDATION SEQUENCE:
{json.dumps(ci_context.get("validation_sequence", []), indent=2)}

TASK:
Treat this repository instance independently. Similarity only creates a candidate
group; it is not proof that records describe the same issue. Use only the supplied
records and current CI evidence, without assuming technologies or patterns from
other dataset instances.

Synthesize ONE unified problem by:
1. Merge problem descriptions (take clearest)
2. Merge affected files (union)
3. Merge root cause (take most specific)
4. Choose best verification_cmd
5. Identify which failed jobs this explains

CRITICAL - PACKAGE/DEPENDENCY VERSION SPECIFICITY:

For dependency/config problems, root_cause and how_fixed MUST include:
1. Exact package name
2. Old version → New version (EXACT constraints)
3. Config file changed
4. Technical reason WHY old version failed and WHY new version fixes

Example: "click 8.2.0 broke TyperOption causing TypeError. Changed click from >=8.0.0 to <8.2.0 in framework/pyproject.toml to maintain API compatibility."

SYNTHESIS RULES:
- If cluster has BOTH forward (F*) and backward (B*):
  * affected_files: From forward (actual changes)
  * problem: From backward (clearer failure description)
  * root_cause: From backward (ground truth) + **FOR DEPENDENCY: add exact versions, technical incompatibility**
  * how_fixed: From forward (implementation) + **FOR DEPENDENCY: add exact old→new versions, config file**
  * why_fix_works: From backward (rationale) + **FOR DEPENDENCY: explain technical compatibility restored**
  * verification_cmd: From forward (actual CI command)
  * Source: "bidirectional_matched"
  * Confidence: Calibrate from agreement between the supplied evidence fields

- If cluster has ONLY forward (all F*):
  * Use forward data
  * Infer problem/root_cause from code changes
  * **FOR DEPENDENCY: Extract exact versions from diff, explain technical reason**
  * Source: "forward_only"
  * Confidence: Calibrate from the strength of the supplied change evidence

- If cluster has ONLY backward (all B*):
  * Use backward data
  * Infer affected_files from error locations
  * **FOR DEPENDENCY: Include exact versions if available in error messages/root_cause**
  * Source: "backward_only"
  * Confidence: Calibrate from the strength of the supplied failure evidence

OUTPUT JSON:
{{
  "unified_problem": {{
    "problem_id": {cluster_id},
    "affected_files": ["from forward or inferred"],
    "problem": "Clear problem description",
    "root_cause": "Evidence-backed root cause",
    "how_fixed": "Implementation details",
    "why_fix_works": "Rationale",
    "verification_cmd": "CI command",
    "failure_type": "...",
    "issue_type": "...",

    "_metadata": {{
      "source": "bidirectional_matched|forward_only|backward_only",
      "source_ids": {json.dumps(source_ids)},
      "cluster_id": {cluster_id}
    }},

    "explains_jobs": ["job1", "job2"],
    "confidence": 0.9
  }}
}}
"""

    # Get model-specific max_tokens
    model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', 'unknown')
    max_tokens = get_model_max_output_tokens(model_name)

    result = invoke_llm_with_retry(
        llm=llm,
        prompt=prompt,
        parse_json=True,
        max_tokens=max_tokens,
    )

    return {
        "cluster_id": cluster_id,
        "unified_problem": result.get("unified_problem", {}),
        "source_ids": source_ids,
        "has_forward": has_forward,
        "has_backward": has_backward,
    }


def merge_cluster_results(
    cluster_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge all cluster results into unified problems list.
    """
    unified_problems = []

    for result in cluster_results:
        problem = result["unified_problem"]

        # Ensure problem_id
        if "problem_id" not in problem:
            problem["problem_id"] = result["cluster_id"]

        # Add cluster metadata
        if "_metadata" not in problem:
            problem["_metadata"] = {}

        problem["_metadata"]["cluster_id"] = result["cluster_id"]
        problem["_metadata"]["source_ids"] = result["source_ids"]

        # Determine source type
        if result["has_forward"] and result["has_backward"]:
            problem["_metadata"]["source"] = "bidirectional_matched"
        elif result["has_forward"]:
            problem["_metadata"]["source"] = "forward_only"
        else:
            problem["_metadata"]["source"] = "backward_only"

        unified_problems.append(problem)

    return unified_problems


def compute_dependency_order(
    unified_problems: List[Dict[str, Any]],
    forward_dependencies: List[Dict[str, Any]],
    backward_dependencies: List[Dict[str, Any]],
    ci_validation_sequence: List[Any],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Compute repair order based on:
    1. Forward dependencies
    2. Backward dependencies
    3. CI validation sequence

    Returns:
        (dependencies, repair_sequence)
    """
    # Build dependency graph
    dependencies = []
    dep_graph = defaultdict(list)

    # Map source IDs to unified problem IDs
    source_to_problem = {}
    for problem in unified_problems:
        for source_id in problem.get("_metadata", {}).get("source_ids", []):
            source_to_problem[source_id] = problem["problem_id"]

    # Add forward dependencies
    for dep in forward_dependencies:
        from_source = dep.get("from")
        to_source = dep.get("to")

        if from_source in source_to_problem and to_source in source_to_problem:
            from_id = source_to_problem[from_source]
            to_id = source_to_problem[to_source]

            dependencies.append({
                "from": from_id,
                "to": to_id,
                "type": dep.get("type", "direct"),
                "support": "forward",
            })
            dep_graph[from_id].append(to_id)

    # Add backward dependencies (if not already present)
    for dep in backward_dependencies:
        from_source = dep.get("from")
        to_source = dep.get("to")

        if from_source in source_to_problem and to_source in source_to_problem:
            from_id = source_to_problem[from_source]
            to_id = source_to_problem[to_source]

            # Check if already added from forward
            existing = any(
                d["from"] == from_id and d["to"] == to_id
                for d in dependencies
            )

            if existing:
                # Mark as supported by both
                for d in dependencies:
                    if d["from"] == from_id and d["to"] == to_id:
                        d["support"] = "both"
            else:
                dependencies.append({
                    "from": from_id,
                    "to": to_id,
                    "type": dep.get("type", "direct"),
                    "support": "backward",
                })
                dep_graph[from_id].append(to_id)

    # Topological sort for repair sequence
    repair_sequence = topological_sort(dep_graph, [p["problem_id"] for p in unified_problems])

    return dependencies, repair_sequence


def topological_sort(graph: Dict[int, List[int]], all_nodes: List[int]) -> List[int]:
    """
    Topological sort using DFS.

    Returns nodes in dependency order (dependencies first).
    """
    visited = set()
    result = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)

        # Visit dependencies first
        for neighbor in graph.get(node, []):
            dfs(neighbor)

        result.append(node)

    # Visit all nodes
    for node in all_nodes:
        dfs(node)

    # Reverse to get correct order (dependencies first)
    return list(reversed(result))


def analyze_and_reason_with_ci(
    unified_problems: List[Dict[str, Any]],
    ci_context: Dict[str, Any],
    llm: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Step 4: Analysis & Reasoning based on CI sequence and failed jobs.

    This is the critical step that:
    1. Analyzes each problem against CI validation sequence
    2. Checks which failed jobs each problem explains
    3. Determines CI-relevance (filter out non-CI problems)
    4. Builds overall problems relevant to CI validation
    5. Provides reasoning for why each problem matters

    Returns:
        (ci_relevant_problems, analysis_result)
    """
    print(f"\n[Step 4] Analysis & Reasoning (CI-focused)...")
    print(f"  Analyzing {len(unified_problems)} unified problems against:")
    print(f"    - CI sequence: {len(ci_context.get('validation_sequence', []))} steps")
    print(f"    - Failed jobs: {len(ci_context.get('failed_jobs', []))}")

    prompt = f"""You are analyzing unified problems to determine CI-relevance.

UNIFIED PROBLEMS (from merging within clusters):
{json.dumps(unified_problems, indent=2)}

FAILED CI JOBS:
{json.dumps(ci_context.get("failed_jobs", []), indent=2)}

CI VALIDATION SEQUENCE:
{json.dumps(ci_context.get("validation_sequence", []), indent=2)}

CI WORKFLOW PATH:
{ci_context.get("workflow_path", "")}

TASK - Analysis & Reasoning:

Analyze this repository instance independently. Every conclusion must be grounded
in the unified problems and CI evidence above. Do not assume any language,
framework, validator, job naming scheme, or failure pattern from another instance.

For EACH unified problem, analyze:

1. **CI Sequence Position**: Where in the CI validation sequence does this problem occur?
   - Which input validation step, if one can be identified?
   - What stage in the workflow?

2. **Failed Jobs Explanation**: Which failed jobs does this problem explain?
   - Match problem to specific job failures
   - Provide evidence (error messages, stack traces)

3. **CI Relevance**: Is this problem RELEVANT to CI validation?
   - HIGH: Directly causes CI failure
   - MEDIUM: Affects CI but not primary cause
   - LOW: Not related to CI (scope creep, side-effect)
   - NONE: Not CI-relevant (exclude from final list)

4. **Root Cause in CI Context**: Why did this cause CI to fail?
   - Connect problem → CI failure
   - Explain the causal chain

5. **Validation Impact**: How does this affect CI validation?
   - Which tests/checks fail?
   - Cascading effects?

IMPORTANT FILTERING RULES:

1. **Exclude CI Workflow Problems**: Do NOT include problems that only affect .github/workflows/ files
   UNLESS its failure_type or issue_type explicitly identifies a formatting-only
   change. Apply this rule from the supplied fields, not from inferred repository conventions.

2. **Do NOT Include Source Tracking**: Do NOT add these fields to output:
   - source_forward_ids
   - source_backward_ids
   - alignment
   - alignment_confidence
   - alignment_type
   These are internal tracking fields that should not appear in final output.

OUTPUT JSON:
{{
  "ci_relevant_problems": [
    {{
      "problem_id": 1,
      ...all existing fields (NO source tracking)...,

      "ci_analysis": {{
        "validation_step": "step name from current CI context",
        "step_order": 0,
        "explains_jobs": ["job identifier from current CI context"],
        "failure_evidence": "specific evidence from current CI context",
        "ci_relevance": "high",
        "root_cause_in_ci": "instance-specific causal explanation",
        "validation_impact": "instance-specific validation impact"
      }},

      "confidence": 0.0,
      "include_in_final": true
    }}
  ],

  "excluded_problems": [
    {{
      "problem_id": 5,
      "reason": "Not CI-relevant - code cleanup only",
      "ci_relevance": "none"
    }}
  ],

  "coverage_analysis": {{
    "total_failed_jobs": 3,
    "explained_jobs": ["job1", "job2"],
    "unexplained_jobs": ["job3"],
    "coverage_rate": 0.67,
    "missing_problems": "Possible missing fix for job3"
  }},

  "overall_assessment": {{
    "all_failures_explained": true,
    "ci_validation_complete": true,
    "quality": "high"
  }}
}}
"""

    # Get model-specific max_tokens
    model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', 'unknown')
    max_tokens = get_model_max_output_tokens(model_name)

    result = invoke_llm_with_retry(
        llm=llm,
        prompt=prompt,
        parse_json=True,
        max_tokens=max_tokens,
    )

    # Extract CI-relevant problems only
    ci_relevant = result.get("ci_relevant_problems", unified_problems)
    excluded = result.get("excluded_problems", [])
    coverage = result.get("coverage_analysis", {})
    overall = result.get("overall_assessment", {})

    print(f"  ├─ CI-relevant: {len(ci_relevant)} problems")
    print(f"  ├─ Excluded (non-CI): {len(excluded)} problems")
    print(f"  ├─ Coverage: {coverage.get('coverage_rate', 0):.1%} ({len(coverage.get('explained_jobs', []))}/{coverage.get('total_failed_jobs', 0)} jobs)")
    print(f"  └─ Quality: {overall.get('quality', 'unknown')}")

    analysis_result = {
        "coverage": coverage,
        "overall": overall,
        "excluded": excluded,
    }

    return ci_relevant, analysis_result


def build_with_clustered_reasoning(
    issue_id: str,
    forward_problems: List[Dict[str, Any]],
    backward_problems: List[Dict[str, Any]],
    ci_context: Dict[str, Any],
    forward_dependencies: Optional[List[Dict[str, Any]]] = None,
    backward_dependencies: Optional[List[Dict[str, Any]]] = None,
    llm: Any = None,
    similarity_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Main clustered reasoning pipeline.

    CORRECTED FLOW:
    1. Flatten and label (F1, F2, B1, B2)
    2. Cluster by cosine similarity (file-first matching)
    3. Merge within each cluster → unified problems
    4. Analyze ALL unified problems against CI workflow + failed jobs
    5. Reorder by dependencies (forward + backward + CI sequence)
    """
    print(f"\n{'='*80}")
    print(f"CLUSTERED REASONING: Issue {issue_id}")
    print(f"{'='*80}")
    print(f"Forward: {len(forward_problems)}, Backward: {len(backward_problems)}")
    print(f"Similarity threshold: {similarity_threshold}")

    if not llm:
        raise ValueError("LLM required for clustered reasoning")

    forward_dependencies = forward_dependencies or []
    backward_dependencies = backward_dependencies or []

    # Step 1: Flatten and label
    print(f"\n[Step 1] Flattening and labeling problems...")
    flattened = flatten_and_label_problems(forward_problems, backward_problems)
    print(f"  Total: {len(flattened)} problems (F1-F{len(forward_problems)}, B1-B{len(backward_problems)})")

    # Step 2: Cluster by similarity (file-first matching)
    print(f"\n[Step 2] Clustering by file overlap + cosine similarity...")
    clusters = cluster_problems_by_similarity(flattened, similarity_threshold)
    print(f"  Created {len(clusters)} clusters")
    for i, cluster in enumerate(clusters):
        source_ids = [p["source_id"] for p in cluster]
        files = set()
        for p in cluster:
            files.update(p.get("affected_files", []) or p.get("files", []))
        print(f"    Cluster {i+1}: {source_ids} (files: {list(files)[:3]}...)")

    # Step 3: Merge within each cluster (if similar, otherwise keep distinct)
    print(f"\n[Step 3] Merging within clusters (similar → merge, distinct → separate)...")
    unified_problems = []
    problem_id_counter = 1

    for cluster_idx, cluster in enumerate(clusters):
        # For each cluster, check if problems are truly similar
        # If similar → merge into one
        # If distinct → keep separate

        if len(cluster) == 1:
            # Single problem cluster - just add it
            problem = cluster[0]
            unified = {
                "problem_id": problem_id_counter,
                "affected_files": problem.get("affected_files", []) or problem.get("files", []),
                "problem": problem.get("problem", ""),
                "root_cause": problem.get("root_cause", ""),
                "how_fixed": problem.get("how_fixed", ""),
                "why_fix_works": problem.get("why_fix_works", ""),
                "verification_cmd": problem.get("verification_cmd", "") or problem.get("validation_cmd", ""),
                "failure_type": problem.get("failure_type", ""),
                "issue_type": problem.get("issue_type", ""),
                "_metadata": {
                    "source": f"{problem['source']}_only",
                    "source_ids": [problem["source_id"]],
                    "cluster_id": cluster_idx + 1,
                },
            }
            unified_problems.append(unified)
            problem_id_counter += 1
        else:
            # Multiple problems - DEFAULT to MERGE if in same cluster
            has_forward = any(p["source"] == "forward" for p in cluster)
            has_backward = any(p["source"] == "backward" for p in cluster)

            # Key insight: If clustering put them together (same files),
            # they're likely the SAME issue → MERGE by default
            # Only keep distinct if CLEARLY different

            should_merge = False

            # Rule 1: If cluster has both forward AND backward → MERGE (bidirectional match)
            if has_forward and has_backward:
                should_merge = True

            # Rule 2: If cluster has same source but similar files → MERGE
            elif len(set(p["source"] for p in cluster)) == 1:
                # All from same source (all forward or all backward)
                # Check if they share files
                all_files = set()
                for p in cluster:
                    files = set(p.get("affected_files", []) or p.get("files", []))
                    all_files.update(files)

                if len(all_files) <= 3:  # Few files → likely same issue
                    should_merge = True

            if should_merge:
                # SIMILAR + bidirectional → MERGE into one
                merged = {
                    "problem_id": problem_id_counter,
                    "affected_files": list(set(
                        f for p in cluster
                        for f in (p.get("affected_files", []) or p.get("files", []))
                    )),
                    # Take backward problem description (clearer)
                    "problem": next((p.get("problem") for p in cluster if p["source"] == "backward" and p.get("problem")),
                                   next((p.get("problem") for p in cluster if p.get("problem")), "")),
                    # Take backward root cause (ground truth)
                    "root_cause": next((p.get("root_cause") for p in cluster if p["source"] == "backward" and p.get("root_cause")),
                                      next((p.get("root_cause") for p in cluster if p.get("root_cause")), "")),
                    # Take forward how_fixed (implementation)
                    "how_fixed": next((p.get("how_fixed") for p in cluster if p["source"] == "forward" and p.get("how_fixed")),
                                     next((p.get("how_fixed") for p in cluster if p.get("how_fixed")), "")),
                    # Take backward why_fix_works (rationale)
                    "why_fix_works": next((p.get("why_fix_works") for p in cluster if p["source"] == "backward" and p.get("why_fix_works")),
                                         next((p.get("why_fix_works") for p in cluster if p.get("why_fix_works")), "")),
                    # Take forward verification_cmd (actual CI command)
                    "verification_cmd": next((p.get("verification_cmd") or p.get("validation_cmd") for p in cluster if p["source"] == "forward" and (p.get("verification_cmd") or p.get("validation_cmd"))),
                                            next((p.get("verification_cmd") or p.get("validation_cmd") for p in cluster if p.get("verification_cmd") or p.get("validation_cmd")), "")),
                    "failure_type": next((p.get("failure_type") for p in cluster if p.get("failure_type")), ""),
                    "issue_type": next((p.get("issue_type") for p in cluster if p.get("issue_type")), ""),
                    "_metadata": {
                        "source": "bidirectional_matched",
                        "source_ids": [p["source_id"] for p in cluster],
                        "cluster_id": cluster_idx + 1,
                    },
                }
                unified_problems.append(merged)
                problem_id_counter += 1
            else:
                # NOT SIMILAR or only one source → Keep DISTINCT
                for problem in cluster:
                    unified = {
                        "problem_id": problem_id_counter,
                        "affected_files": problem.get("affected_files", []) or problem.get("files", []),
                        "problem": problem.get("problem", ""),
                        "root_cause": problem.get("root_cause", ""),
                        "how_fixed": problem.get("how_fixed", ""),
                        "why_fix_works": problem.get("why_fix_works", ""),
                        "verification_cmd": problem.get("verification_cmd", "") or problem.get("validation_cmd", ""),
                        "failure_type": problem.get("failure_type", ""),
                        "issue_type": problem.get("issue_type", ""),
                        "_metadata": {
                            "source": f"{problem['source']}_only",
                            "source_ids": [problem["source_id"]],
                            "cluster_id": cluster_idx + 1,
                        },
                    }
                    unified_problems.append(unified)
                    problem_id_counter += 1

    print(f"  Merged: {len(unified_problems)} unified problems from {len(clusters)} clusters")

    # Count by source
    matched = sum(1 for p in unified_problems if p.get("_metadata", {}).get("source") == "bidirectional_matched")
    forward_only = sum(1 for p in unified_problems if p.get("_metadata", {}).get("source") == "forward_only")
    backward_only = sum(1 for p in unified_problems if p.get("_metadata", {}).get("source") == "backward_only")

    print(f"    Matched: {matched}")
    print(f"    Forward-only: {forward_only}")
    print(f"    Backward-only: {backward_only}")

    # Step 4: Analysis & Reasoning (CI-focused)
    # - Analyze against CI sequence and failed jobs
    # - Build overall problems relevant to CI validation
    # - Filter to only CI-relevant problems
    ci_relevant_problems, analysis_result = analyze_and_reason_with_ci(
        unified_problems, ci_context, llm
    )

    # Step 5: Reorder & Sequence
    # - Based on dependencies (forward + backward)
    # - Based on CI validation order
    print(f"\n[Step 5] Reorder & Sequence...")
    dependencies, repair_sequence = compute_dependency_order(
        ci_relevant_problems,
        forward_dependencies,
        backward_dependencies,
        ci_context.get("validation_sequence", []),
    )
    print(f"  ├─ Dependencies: {len(dependencies)}")
    print(f"  └─ Repair sequence: {repair_sequence}")

    print(f"{'='*80}\n")

    # Final summary
    print(f"FINAL SUMMARY:")
    print(f"  Total unified (after merge): {len(unified_problems)}")
    print(f"  CI-relevant: {len(ci_relevant_problems)}")
    print(f"  Excluded (non-CI): {len(analysis_result.get('excluded', []))}")
    print(f"  Final repair sequence: {len(repair_sequence)} problems")
    print(f"{'='*80}\n")

    return {
        "unified_problems": ci_relevant_problems,  # Keep for internal compatibility
        "problems": ci_relevant_problems,  # Add for memory building compatibility
        "dependencies": dependencies,
        "repair_sequence": repair_sequence,
        "coverage_analysis": analysis_result.get("coverage", {}),
        "overall_assessment": analysis_result.get("overall", {}),
        "excluded_problems": analysis_result.get("excluded", []),
        "metadata": {
            "total_unified": len(unified_problems),
            "ci_relevant": len(ci_relevant_problems),
            "excluded_count": len(analysis_result.get("excluded", [])),
            "aligned": matched,
            "forward_only": forward_only,
            "backward_only": backward_only,
            "alignment_rate": matched / len(unified_problems) if unified_problems else 0.0,
            "clusters": len(clusters),
            "coverage_rate": analysis_result.get("coverage", {}).get("coverage_rate", 0.0),
        },
        "method": "clustered_reasoning",
        "reconciliation_quality": analysis_result.get("overall", {}).get("quality", "medium"),
        "memory_build_ready": analysis_result.get("overall", {}).get("ci_validation_complete", True),
    }
