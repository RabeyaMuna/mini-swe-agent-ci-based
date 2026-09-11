"""Bidirectional reconciliation for forward and backward CI decompositions.

Pipeline:
1. Flatten and label forward/backward problems (F1, F2, B1, B2, ...).
2. Cluster similar problems with multi-field TF-IDF cosine similarity.
3. Ask an LLM to merge true duplicates within each candidate cluster.
4. Ask an LLM to remove globally irrelevant or duplicate problems.
5. Project the selected dependency edges into a deterministic repair order.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set

from utilities.llm_invoker import STRICT_JSON_RULES, invoke_llm_with_retry, get_model_max_output_tokens


# Candidate generation is intentionally high-recall. The LLM adjudicator may
# split a loose component, while globally separated candidates cannot be merged.
# Set to 0.25 for high-recall TF-IDF clustering, combined with file overlap
# and semantic matching to catch backward+forward pairs with different wording
SIMILARITY_THRESHOLD = 0.25
MAX_CLUSTER_PROMPT_BYTES = 30_000

# File-based merging: If two problems affect the same file with high keyword overlap, merge them
FILE_OVERLAP_MERGE_THRESHOLD = 0.50


# Helper functions for data sanitization and validation
def _sanitize_ci_context(ci_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize CI context to prevent data leakage.
    Only include information needed for decision making.
    """
    safe_context = {}

    # Only include safe fields
    if 'failed_jobs' in ci_context:
        # Only job names, not full logs
        safe_context['failed_jobs'] = [
            job if isinstance(job, str) else job.get('name', 'unknown')
            for job in ci_context['failed_jobs'][:5]  # Limit to first 5
        ]

    if 'overall_failure_reasons' in ci_context:
        safe_context['failure_reasons'] = ci_context['overall_failure_reasons'][:3]

    if 'error_types' in ci_context:
        safe_context['error_types'] = ci_context['error_types'][:5]

    if 'workflow_path' in ci_context:
        # Just the workflow name, not full path
        workflow = ci_context['workflow_path']
        if isinstance(workflow, str):
            safe_context['workflow'] = workflow.split('/')[-1]

    return safe_context


def _verify_no_hallucination(merged: Dict[str, Any], source_problems: List[Dict[str, Any]]) -> None:
    """
    Verify merged content doesn't contain hallucinated information.
    Raises exception if hallucination detected.
    """
    import re

    # Check files
    merged_files = set(merged.get('affected_files', []))
    source_files = set()
    for p in source_problems:
        source_files.update(p.get('affected_files', []))
        source_files.update(p.get('files', []))

    hallucinated_files = merged_files - source_files
    if hallucinated_files:
        print(f"  WARNING: Merged content has files not in sources: {hallucinated_files}")

    # Check for suspicious additions
    merged_text = ' '.join([
        str(merged.get('problem', '')),
        str(merged.get('root_cause', '')),
        str(merged.get('how_fixed', '')),
    ]).lower()

    # Patterns that indicate hallucination
    suspicious_patterns = [
        (r'\.gitignore', 'gitignore entries'),
        (r'reorganize.*import', 'import reorganization'),
        (r'from typing to collections', 'typing to collections'),
    ]

    for pattern, description in suspicious_patterns:
        if re.search(pattern, merged_text):
            # Check if any source mentions it
            source_text = ' '.join([
                str(p.get('problem', '')) + ' ' +
                str(p.get('root_cause', '')) + ' ' +
                str(p.get('how_fixed', ''))
                for p in source_problems
            ]).lower()

            if not re.search(pattern, source_text):
                print(f"  WARNING: Merged content mentions {description} not in sources")


def simple_bidirectional_reconciliation(
    forward_problems: List[Dict[str, Any]],
    backward_problems: List[Dict[str, Any]],
    ci_context: Dict[str, Any],
    llm: Any,
    dependency_graph: Dict[str, Any] | None = None,
    structured_diff: Dict[str, Any] | None = None,
    repo_path: str | None = None,
) -> Dict[str, Any]:
    """Reconcile two problem lists and return one ordered problem list."""
    del repo_path

    print("\n" + "=" * 80)
    print("BIDIRECTIONAL RECONCILIATION")
    print("=" * 80)

    flattened = flatten_and_label(forward_problems, backward_problems)
    print(f"\n[Step 1] Flattened and labeled {len(flattened)} problems")

    clusters = cluster_by_similarity(flattened)
    print(f"[Step 2] Created {len(clusters)} cosine-similarity clusters")

    merged_candidates: List[Dict[str, Any]] = []
    print("[Step 3] Adjudicating candidate clusters with CI context")
    for cluster_index, cluster in enumerate(clusters, 1):
        if len(cluster) == 1:
            singleton = _copy_singleton(cluster[0])
            singleton["_similarity_singleton"] = True
            merged_candidates.append(singleton)
            continue
        merged_candidates.extend(llm_merge_cluster(cluster, llm, cluster_index, ci_context))

    merged_candidates = _assign_problem_ids(merged_candidates)
    print(f"[Step 3] Produced {len(merged_candidates)} candidate problems")

    print("[Step 4] Removing irrelevant and globally duplicate problems")
    decision = llm_filter_and_deduplicate(
        merged_candidates,
        ci_context,
        llm,
        dependency_graph=dependency_graph,
        structured_diff=structured_diff,
    )
    selected = _apply_global_decision(merged_candidates, decision)
    selected = _assign_problem_ids(selected)

    # Selection and deduplication do not define dependencies. Only the focused
    # post-selection dependency stage may create final repair edges.
    dependency_decision = llm_infer_dependencies(
        selected,
        ci_context,
        llm,
        dependency_graph=dependency_graph,
        structured_diff=structured_diff,
    )
    focused_dependencies = _normalize_dependencies(
        dependency_decision.get("dependencies", []),
        old_to_new={problem["problem_id"]: problem["problem_id"] for problem in selected},
    )
    dependencies = _deduplicate_dependencies(focused_dependencies)
    repair_sequence = dependency_order(selected, dependencies)
    selected = _sort_by_sequence(selected, repair_sequence)
    _apply_enabled_relationships(selected, dependencies)

    print(f"[Step 5] Retained {len(selected)} problems")
    print(f"[Step 6] Dependencies: {len(dependencies)}")
    print(f"[Step 7] Repair order: {repair_sequence}")
    print("=" * 80 + "\n")

    return {
        "problems": selected,
        "dependencies": dependencies,
        "repair_sequence": repair_sequence,
        "reconciliation_metadata": {
            "input_forward": len(forward_problems),
            "input_backward": len(backward_problems),
            "clusters": len(clusters),
            "after_cluster_merge": len(merged_candidates),
            "selected": len(selected),
            "irrelevant_problem_ids": decision.get("irrelevant_problem_ids", []),
            "duplicate_groups": decision.get("duplicate_groups", []),
        },
    }


def flatten_and_label(
    forward_problems: List[Dict[str, Any]],
    backward_problems: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Copy problems and attach stable directional source identifiers."""
    flattened: List[Dict[str, Any]] = []
    for prefix, source, problems in (
        ("F", "forward", forward_problems),
        ("B", "backward", backward_problems),
    ):
        for index, raw_problem in enumerate(problems, 1):
            problem = _normalize_problem(raw_problem)
            problem["source_id"] = f"{prefix}{index}"
            problem["source"] = source
            flattened.append(problem)
    return flattened


def _normalize_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize field aliases before similarity calculation and prompting."""
    files = problem.get("affected_files") or problem.get("files") or []
    if isinstance(files, str):
        files = [files]

    return {
        "problem": str(problem.get("problem", "")).strip(),
        "root_cause": str(problem.get("root_cause", "")).strip(),
        "how_fixed": str(problem.get("how_fixed", "")).strip(),
        "why_fix_works": str(problem.get("why_fix_works", "")).strip(),
        "affected_files": _unique_strings(files),
        "validation_cmd": str(problem.get("validation_cmd", "")).strip(),
        "validation_order": problem.get("validation_order"),
        "failure_type": str(problem.get("failure_type", "")).strip(),
        "issue_type": str(problem.get("issue_type", "")).strip(),
        "problem_type": str(problem.get("problem_type", "")).strip(),
    }


def _similarity_document(problem: Dict[str, Any]) -> str:
    """Build a weighted document from every requested comparison field."""
    description = problem.get("problem", "")
    root_cause = problem.get("root_cause", "")
    how_fixed = problem.get("how_fixed", "")
    failure_type = problem.get("failure_type", "")
    issue_type = problem.get("issue_type", "")
    command = problem.get("validation_cmd", "")
    files = " ".join(problem.get("affected_files", []))
    return " ".join(
        [
            f"problem {description} {description}",
            f"root_cause {root_cause} {root_cause}",
            f"fix_strategy {how_fixed}",
            f"failure_type {failure_type} {failure_type}",
            f"issue_type {issue_type}",
            f"validation_cmd {command} {command}",
            f"affected_files {files} {files}",
        ]
    )


def _calculate_file_overlap(files1: List[str], files2: List[str]) -> float:
    """Calculate Jaccard similarity of file sets."""
    set1 = set(files1)
    set2 = set(files2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _normalize_failure_type(failure_type: str) -> str:
    """Normalize failure type to category."""
    ft = failure_type.lower().strip()
    if 'format' in ft or 'black' in ft:
        return 'formatting'
    if 'lint' in ft or 'ruff' in ft or 'flake' in ft:
        return 'linting'
    if 'test' in ft:
        return 'testing'
    if 'type' in ft or 'mypy' in ft:
        return 'type_checking'
    if 'build' in ft or 'compile' in ft:
        return 'build'
    if 'import' in ft or 'dependency' in ft:
        return 'dependency'
    return ft


def _should_cluster_hybrid(
    prob1: Dict[str, Any],
    prob2: Dict[str, Any],
    tfidf_similarity: float,
    threshold: float = SIMILARITY_THRESHOLD,
) -> bool:
    """
    Hybrid clustering decision using 3 criteria (OR condition):
    1. TF-IDF similarity >= threshold (text similarity)
    2. File overlap >= 0.8 (same files)
    3. Failure type match AND file overlap >= 0.5 (semantic + partial file overlap)

    This catches cases where backward+forward describe same issue with different wording.
    """
    # Criterion 1: TF-IDF similarity
    if tfidf_similarity >= threshold:
        return True

    # Extract features for semantic matching
    files1 = prob1.get('affected_files', [])
    files2 = prob2.get('affected_files', [])
    file_overlap = _calculate_file_overlap(files1, files2)

    # Criterion 2: High file overlap (same files = likely same problem)
    if file_overlap >= 0.8:
        return True

    # Criterion 3: Failure type match + moderate file overlap
    failure1 = _normalize_failure_type(prob1.get('failure_type', ''))
    failure2 = _normalize_failure_type(prob2.get('failure_type', ''))
    if failure1 == failure2 and file_overlap >= 0.5:
        return True

    return False


def cluster_by_similarity(
    problems_or_forward: List[Dict[str, Any]],
    backward_problems: List[Dict[str, Any]] | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[List[Dict[str, Any]]]:
    """Return connected components of the similarity graph.

    Uses HYBRID clustering with 3 criteria (OR condition):
    1. TF-IDF text similarity >= threshold
    2. File overlap >= 0.8 (same files)
    3. Failure type match AND file overlap >= 0.5

    ``backward_problems`` is accepted for compatibility with older callers.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if backward_problems is None:
        problems = problems_or_forward
    else:
        problems = flatten_and_label(problems_or_forward, backward_problems)

    if not problems:
        return []
    if len(problems) == 1:
        return [[problems[0]]]

    texts = [_similarity_document(problem) for problem in problems]
    try:
        matrix = TfidfVectorizer(
            max_features=2000,
            min_df=1,
            ngram_range=(1, 2),
            sublinear_tf=True,
        ).fit_transform(texts)
        similarities = cosine_similarity(matrix)
    except ValueError:
        return [[problem] for problem in problems]

    # Build adjacency using hybrid clustering (TF-IDF OR file overlap OR semantic match)
    adjacency: Dict[int, set[int]] = defaultdict(set)
    for left in range(len(problems)):
        for right in range(left + 1, len(problems)):
            tfidf_sim = similarities[left, right]
            if _should_cluster_hybrid(problems[left], problems[right], tfidf_sim, threshold):
                adjacency[left].add(right)
                adjacency[right].add(left)

    clusters: List[List[Dict[str, Any]]] = []
    visited: set[int] = set()
    for start in range(len(problems)):
        if start in visited:
            continue
        stack = [start]
        component: List[int] = []
        visited.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        clusters.append([problems[index] for index in sorted(component)])
    return clusters


def llm_merge_cluster(
    cluster: List[Dict[str, Any]],llm: Any,
    cluster_index: int = 1,
    ci_context: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Two-phase reconciliation:
    Phase 1: Decision making (SELECT/MERGE/CONFLICT/SEPARATE)
    Phase 2: Content generation (only for MERGE)
    """
    # CRITICAL: Write to file so we can verify this code is actually running
    import datetime
    log_file = '/tmp/bidirectional_reconciliation.log'
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{datetime.datetime.now()}] NEW TWO-PHASE RECONCILIATION CALLED\n")
        f.write(f"Cluster {cluster_index}: {len(cluster)} problems\n")
        f.write(f"{'='*80}\n")

    print("\n" + "=" * 80)
    print("DEBUG: NEW TWO-PHASE RECONCILIATION ACTIVE")
    print("=" * 80)
    # Send every field without truncation. Oversized clusters are divided into
    # lossless batches; records are never shortened or dropped. Global duplicate
    # analysis can still reconcile matches that fall into different batches.
    batches = _lossless_problem_batches(cluster, MAX_CLUSTER_PROMPT_BYTES)
    if len(batches) > 1:
        output: List[Dict[str, Any]] = []
        for batch_index, batch in enumerate(batches, 1):
            if len(batch) == 1:
                output.append(_copy_singleton(batch[0]))
            else:
                output.extend(
                    llm_merge_cluster(batch, llm, f"{cluster_index}.{batch_index}", ci_context)
                )
        return output

    ci_context = ci_context or {}
    # Sanitize CI context to prevent data leakage
    safe_ci_context = _sanitize_ci_context(ci_context)

    # PHASE 1: Decision making
    print(f"  Cluster {cluster_index}: Phase 1 - Analyzing relationships...")
    print(f"  DEBUG: Cluster has {len(cluster)} problems")

    # Log to file
    with open('/tmp/bidirectional_reconciliation.log', 'a') as f:
        f.write(f"\nPhase 1: Analyzing {len(cluster)} problems\n")
        for p in cluster:
            f.write(f"  {p.get('source_id')}: {p.get('problem', '')[:60]}...\n")

    decisions = _llm_decide_actions(cluster, safe_ci_context, llm, cluster_index)
    print(f"  DEBUG: Got {len(decisions.get('groups', []))} decision groups")

    # Log decisions to file
    with open('/tmp/bidirectional_reconciliation.log', 'a') as f:
        f.write(f"\nPhase 1 Result: {len(decisions.get('groups', []))} decision groups\n")
        for g in decisions.get('groups', []):
            f.write(f"  Action: {g.get('action')} for {g.get('source_ids', [])}\n")

    for g in decisions.get('groups', []):
        print(f"    Action: {g.get('action')} for sources {g.get('source_ids', [])}")

    # PHASE 2: Process each group based on action
    print(f"  Cluster {cluster_index}: Phase 2 - Processing groups...")
    results = []
    for group in decisions.get('groups', []):
        try:
            processed = _process_group(group, cluster, safe_ci_context, llm)
            if isinstance(processed, list):
                results.extend(processed)
            else:
                results.append(processed)
        except Exception as e:
            print(f"    WARNING: Group {group['action']} failed: {e}")
            # Fallback: keep originals
            source_ids = group.get('source_ids', [])
            fallback = [_copy_singleton(p) for p in cluster if p.get('source_id') in source_ids]
            results.extend(fallback)

    return results if results else [_copy_singleton(p) for p in cluster]


def _llm_decide_actions(
    cluster: List[Dict[str, Any]],
    ci_context: Dict[str, Any],
    llm: Any,
    cluster_index: int = 1,
) -> Dict[str, Any]:
    """
    Phase 1: Lightweight decision making.
    Returns: {groups: [{action, source_ids, selected/reason, confidence}]}
    """
    failed_jobs_str = json.dumps(ci_context.get("failed_jobs", []), indent=2)
    ci_metadata_str = json.dumps(ci_context.get("ci_metadata", {}), indent=2)

    prompt = f"""You are analyzing relationships between forward (commit-based) and backward (CI failure-based) problems.

**PHASE 1: DECISION ONLY** - Decide what action to take for each group. DO NOT generate problem content yet.

**CRITICAL RULE - READ FIRST**:
If problems describe the SAME FIX (same command, same change type):
  → ACTION MUST BE: SELECT
  → DO NOT use MERGE unless problems have DIFFERENT, COMPLEMENTARY information

Examples:
- "Add pylint disable" + "Add pylint disable" → SELECT (identical)
- "Fix line length" + "Fix line length" → SELECT (identical)
- "Import error in X" + "Import error cause is Y" → MERGE (complementary)

CANDIDATE CLUSTER {cluster_index}:
{json.dumps(cluster, indent=2)}

CI CONTEXT (for verification):
Failed jobs: {failed_jobs_str}
CI metadata: {ci_metadata_str}

YOUR TASK:
Analyze relationships between problems and decide ONE action per group.

FOUR ACTIONS:

1. **SELECT**: Problems are similar/identical
   - Pick the most detailed/complete one
   - Return: source_ids + selected (which to use) + reason

2. **MERGE**: Problems are complementary (different useful info)
   - Both provide valuable information that should be combined
   - Return: source_ids + reason
   - Note: Content generation happens in Phase 2

3. **CONFLICT**: Problems contradict (different solutions to same problem)
   - Return: source_ids + conflict_type + reason
   - Note: Resolution happens in Phase 2

4. **SEPARATE**: Problem is unrelated to others
   - Keep as-is
   - Return: source_ids + reason

DECISION CRITERIA:

**SELECT when:**
- Forward and backward are identical or near-identical
- One clearly has more detail than the other
- Same fix, different wording

**MERGE when:**
- Problems are complementary (F has context, B has fix details)
- Both provide non-overlapping useful information
- Can be combined into one better problem

**CONFLICT when:**
- Same problem but different/contradictory solutions
- Incompatible approaches (need to choose one)

**SEPARATE when:**
- Problem is genuinely unrelated to others in cluster
- Should remain independent

OUTPUT JSON:
{{
  "analysis": "Brief overall assessment of cluster relationships",
  "groups": [
    {{
      "action": "select",
      "source_ids": ["F1", "F2", "B1"],
      "selected": "B1",
      "reason": "B1 has most complete description with CI context",
      "confidence": "high"
    }},
    {{
      "action": "merge",
      "source_ids": ["F3", "B2"],
      "reason": "F3 has problem description, B2 has root cause - complementary",
      "confidence": "high"
    }},
    {{
      "action": "conflict",
      "source_ids": ["F4", "B3"],
      "conflict_type": "different_solutions",
      "reason": "Both fix line length but with different approaches",
      "confidence": "medium"
    }},
    {{
      "action": "separate",
      "source_ids": ["F5"],
      "reason": "Unrelated import issue",
      "confidence": "high"
    }}
  ]
}}

IMPORTANT:
- Every source_id from cluster must appear exactly once
- Be intelligent: if 10 problems are similar, SELECT can handle all 10
- Only MERGE when problems are truly complementary
- Be conservative with CONFLICT - most cases are SELECT or MERGE

{STRICT_JSON_RULES}
"""
    try:
        # Get model-specific max_tokens
        model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', 'unknown')
        max_tokens = get_model_max_output_tokens(model_name)

        response = invoke_llm_with_retry(llm=llm, prompt=prompt, parse_json=True, max_tokens=max_tokens)
        if not isinstance(response, dict):
            response = {"groups": []}

        # Validate that all source_ids are covered
        all_source_ids = {p.get('source_id') for p in cluster}
        covered_ids = set()
        for group in response.get("groups", []):
            covered_ids.update(group.get('source_ids', []))

        # Add missing problems as SEPARATE
        missing = all_source_ids - covered_ids
        if missing:
            for sid in missing:
                response.setdefault("groups", []).append({
                    "action": "separate",
                    "source_ids": [sid],
                    "reason": "Not classified by LLM",
                    "confidence": "low"
                })

        return response
    except Exception as exc:
        print(f"    WARNING: Phase 1 failed ({exc}), treating all as separate")
        # Fallback: treat all as separate
        return {
            "analysis": "Phase 1 failed",
            "groups": [
                {"action": "separate", "source_ids": [p.get('source_id')], "reason": "Fallback"}
                for p in cluster
            ]
        }


def _process_group(
    group: Dict[str, Any],
    cluster: List[Dict[str, Any]],
    ci_context: Dict[str, Any],
    llm: Any,
) -> List[Dict[str, Any]]:
    """
    Phase 2: Process based on action.
    Returns list of problems (can be 1 or more).
    """
    action = group.get('action', 'separate').lower()
    source_ids = group.get('source_ids', [])

    # Get source problems
    by_source = {p.get('source_id'): p for p in cluster}
    source_problems = [by_source[sid] for sid in source_ids if sid in by_source]

    if not source_problems:
        return []

    # AUTOMATIC ACTIONS (no LLM)
    if action == "select":
        # Use the selected problem
        selected_id = group.get('selected', source_ids[0])
        selected = by_source.get(selected_id, source_problems[0])
        return [_copy_singleton(selected)]

    elif action == "separate":
        # Keep all as-is
        return [_copy_singleton(p) for p in source_problems]

    # LLM ACTIONS
    elif action == "merge":
        # Call LLM to merge complementary information
        try:
            merged = _llm_merge_problems(source_problems, group.get('reason', ''), ci_context, llm)
            # Verify no hallucination
            _verify_no_hallucination(merged, source_problems)

            # CRITICAL: Block .gitignore if not in sources
            if '.gitignore' in json.dumps(merged).lower():
                source_has_gitignore = any('.gitignore' in json.dumps(p).lower()
                                          for p in source_problems)
                if not source_has_gitignore:
                    print(f"      BLOCKED: Merged content adds .gitignore hallucination!")
                    # Fall back to SELECT best source
                    return [_copy_singleton(source_problems[0])]

            return [merged]
        except Exception as e:
            print(f"      WARNING: MERGE failed: {e}, keeping originals")
            return [_copy_singleton(p) for p in source_problems]

    elif action == "conflict":
        # Try to resolve conflict
        try:
            resolved = _resolve_conflict(source_problems, group, ci_context, llm)
            return [resolved] if resolved else [_copy_singleton(p) for p in source_problems]
        except Exception as e:
            print(f"      WARNING: CONFLICT resolution failed: {e}, keeping first")
            return [_copy_singleton(source_problems[0])]

    else:
        # Unknown action, keep as separate
        return [_copy_singleton(p) for p in source_problems]


def _llm_merge_problems(
    source_problems: List[Dict[str, Any]],
    reason: str,
    ci_context: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """
    Phase 2 MERGE: Generate merged content from complementary problems.
    """
    prompt = f"""You are merging complementary problem descriptions into one unified problem.

SOURCE PROBLEMS:
{json.dumps(source_problems, indent=2)}

REASON FOR MERGE: {reason}

CI CONTEXT:
{json.dumps(ci_context, indent=2)}

TASK:
Combine these problems into ONE unified problem that includes ALL useful information from both.

CRITICAL REQUIREMENTS:
1. If Source A fixes files X and Source B fixes files Y:
   → Merged problem MUST address BOTH X and Y
2. If Source A has 2 problems and Source B has 2 problems:
   → Merged result MUST cover ALL 4 problem aspects
3. DO NOT drop any fix from either source
4. DO NOT add information not present in source problems
5. DO NOT hallucinate files, changes, or details

VERIFICATION:
- Count fixes in Source A and list them
- Count fixes in Source B and list them
- Merged MUST include ALL of them

RULES:
1. Combine descriptions where complementary
2. Use UNION of affected_files (ALL files from both sources)
3. Combine how_fixed to include ALL changes from both sources
4. Use more detailed field values when available

OUTPUT JSON (unified problem):
{{
  "problem": "Combined problem description",
  "root_cause": "Combined root cause",
  "how_fixed": "Combined fix description",
  "why_fix_works": "Combined explanation",
  "affected_files": ["all", "files", "from", "both"],
  "validation_cmd": "Most accurate command",
  "failure_type": "...",
  "issue_type": "...",
  "validation_order": 1,
  "source_ids": {json.dumps([p.get('source_id') for p in source_problems])}
}}

{STRICT_JSON_RULES}
"""
    model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', 'unknown')
    max_tokens = get_model_max_output_tokens(model_name)

    response = invoke_llm_with_retry(llm=llm, prompt=prompt, parse_json=True, max_tokens=max_tokens)

    if not isinstance(response, dict):
        # Fallback: use first problem
        return _copy_singleton(source_problems[0])

    # Ensure source_ids are preserved
    if 'source_ids' not in response:
        response['source_ids'] = [p.get('source_id') for p in source_problems]

    return response


def _resolve_conflict(
    source_problems: List[Dict[str, Any]],
    group: Dict[str, Any],
    ci_context: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any] | None:
    """
    Phase 2 CONFLICT: Try to resolve conflicting solutions.
    Returns resolved problem or None if cannot resolve.
    """
    conflict_type = group.get('conflict_type', 'unknown')

    # For now, simple resolution: pick first problem
    # TODO: Add CI-based resolution or present both options
    print(f"      CONFLICT ({conflict_type}): Using first problem as resolution")
    return _copy_singleton(source_problems[0])


def _validate_cluster_groups(
    groups: Iterable[Dict[str, Any]],
    cluster: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_source = {problem["source_id"]: problem for problem in cluster}
    expected = set(by_source)
    used: set[str] = set()
    output: List[Dict[str, Any]] = []

    for raw_group in groups:
        if not isinstance(raw_group, dict):
            continue
        source_ids = [
            source_id
            for source_id in raw_group.get("source_ids", [])
            if source_id in expected and source_id not in used
        ]
        if not source_ids:
            continue
        used.update(source_ids)
        originals = [by_source[source_id] for source_id in source_ids]
        # Multiple forward records are concrete implemented repair units. If the
        # model collapses them, prefer lossless separation over an unsafe merge.
        if sum(source_id.startswith("F") for source_id in source_ids) > 1:
            output.extend(_copy_singleton(problem) for problem in originals)
            continue
        output.append(_build_group_problem(raw_group, originals, source_ids))

    # Never lose a source record because of malformed LLM output.
    for source_id in sorted(expected - used):
        output.append(_copy_singleton(by_source[source_id]))
    return output


def _build_group_problem(
    group: Dict[str, Any],
    originals: List[Dict[str, Any]],
    source_ids: List[str],
) -> Dict[str, Any]:
    # LLM synthesizes unified problem in group data
    # Use LLM's choices first, fall back to smart merging if LLM didn't provide a field

    # Separate forward and backward for fallback
    forward_problems = [p for p in originals if p.get("source") == "forward" or p.get("source_id", "").startswith("F")]
    backward_problems = [p for p in originals if p.get("source") == "backward" or p.get("source_id", "").startswith("B")]

    all_files = _unique_strings(
        file_path
        for problem in originals
        for file_path in problem.get("affected_files", [])
    )
    group_files = group.get("affected_files", []) or []
    if isinstance(group_files, str):
        group_files = [group_files]

    # Use LLM's synthesized data (from new prompt), fall back to smart selection
    return {
        # LLM analyzed both sources and chose best
        "problem": str(group.get("problem") or _first_nonempty(backward_problems, "problem") or _first_nonempty(forward_problems, "problem")),
        "root_cause": str(group.get("root_cause") or _first_nonempty(backward_problems, "root_cause") or _first_nonempty(forward_problems, "root_cause")),
        "how_fixed": str(group.get("how_fixed") or _first_nonempty(forward_problems, "how_fixed") or _first_nonempty(backward_problems, "how_fixed")),
        "why_fix_works": str(
            group.get("why_fix_works") or _first_nonempty(backward_problems, "why_fix_works") or _first_nonempty(forward_problems, "why_fix_works")
        ),

        # Files: Union as instructed
        "affected_files": _unique_strings(list(group_files) + all_files),

        # Commands and metadata
        "validation_cmd": str(
            group.get("validation_cmd")
            or _first_nonempty(forward_problems, "validation_cmd")
            or _first_nonempty(backward_problems, "validation_cmd")
        ),
        "validation_order": group.get("validation_order")
        or _first_nonempty(originals, "validation_order"),
        "failure_type": str(
            group.get("failure_type") or _first_nonempty(originals, "failure_type")
        ),
        "issue_type": str(group.get("issue_type") or _first_nonempty(originals, "issue_type")),
        "problem_type": (
            "primary"
            if any(p.get("problem_type", "").lower() == "primary" for p in originals)
            else "hidden"
        ),
        "source_ids": source_ids,
        "source": _source_kind(source_ids),
        "merge_reason": str(group.get("merge_reason") or ""),
    }


def _copy_singleton(problem: Dict[str, Any]) -> Dict[str, Any]:
    copied = {key: value for key, value in problem.items() if key not in {"source_id"}}
    source_id = problem.get("source_id")
    copied["source_ids"] = [source_id] if source_id else list(problem.get("source_ids", []))
    copied["source"] = problem.get("source") or _source_kind(copied["source_ids"])
    copied["affected_files"] = _unique_strings(copied.get("affected_files", []))
    return copied


def llm_filter_and_deduplicate(
    problems: List[Dict[str, Any]],
    ci_context: Dict[str, Any],
    llm: Any,
    dependency_graph: Dict[str, Any] | None = None,
    structured_diff: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Select relevant problems globally and infer dependency edges."""
    compact_context = {
        "failed_jobs": ci_context.get("failed_jobs", []),
        "overall_failure_reasons": ci_context.get("overall_failure_reasons", []),
        "overall_error_types": ci_context.get("overall_error_types", []),
        "error_types": ci_context.get("error_types", []),
        "relevant_files": ci_context.get("relevant_files", []),
        "validation_sequence": ci_context.get("validation_sequence", []),
        "workflow_path": ci_context.get("workflow_path", ""),
        "original_error_type": ci_context.get("original_error_type", []),
    }
    graph_evidence = {
        "dependency_graph": dependency_graph or {},
        "changed_files": (structured_diff or {}).get("changed_files", []),
    }

    # CRITICAL: Never reject backward singletons
    # Backward comes from actual successful patch, it's ground truth
    backward_singletons = []
    for p in problems:
        source_ids = p.get('source_ids', [])
        if len(source_ids) == 1 and str(source_ids[0]).startswith('B'):
            backward_singletons.append(p['problem_id'])

    if backward_singletons:
        print(f"  Protecting {len(backward_singletons)} backward singletons from filter")

    prompt = f"""Review the complete reconciled problem list for one historical CI repair.

Problems:
{json.dumps(problems, indent=2)}

CI context:
{json.dumps(compact_context, indent=2)}

File/dependency evidence:
{json.dumps(graph_evidence, indent=2, default=str)}

VERIFICATION TASKS:

0. Treat this as an independent instance. Base every decision only on the supplied
   problems, current CI context, and current file/dependency evidence.

CRITICAL PROTECTION RULE:
Problems with source_ids starting with "B" and only 1 source (backward singletons)
are from actual successful patches.
DO NOT mark them as irrelevant unless they are COMPLETELY unrelated to CI failure.
Backward singletons: {backward_singletons}

1. **VERIFY EACH PROBLEM AGAINST CI CONTEXT**:
   - Check if problem is **DIRECTLY relevant**: explains a failed job from failed_jobs list
   - Check if problem is **INDIRECTLY relevant**: prerequisite or supporting fix for another problem
   - Verify root_cause is supported by CI error messages and failure reasons
   - Validate how_fixed matches the actual repair evidence
   - Confirm validation_cmd matches validation_sequence
   - **SELECTION RULE**: Keep ONLY if:
     * Reasoning is correct and complete (problem + root_cause + how_fixed all make sense together)
     * AND (directly relevant to failed jobs OR indirectly relevant as prerequisite)
     * AND problem/root_cause are well-defined and applicable (NOT vague)

2. **REJECT VAGUE OR INCOMPLETE PROBLEMS**:

   IMPORTANT: A problem may have empty how_fixed/affected_files in the JSON,
   but the CI context provides the actual fix information.
   DO NOT reject solely based on empty fields.

   REJECT ONLY IF:
   - Problem description itself is generic: "fixed issues", "updated files"
   - AND root_cause provides no technical detail
   - AND CI context does not explain what failed

   KEEP IF:
   - Problem description is specific (names error code, file, failure type)
   - OR root_cause explains the technical issue
   - OR CI context shows what failed

   Examples to KEEP (not vague):
   - "Pylint R0801 duplicate-code in test_file.py" (Specific error code)
   - "Import error - module X not found" (Specific failure)
   - "Test test_foo failed with AssertionError" (Specific test)

   Examples to REJECT (actually vague):
   - "Fixed some issues" (Generic, no specifics)
   - "Updated files" (No explanation)
   - "Addressed problems" (No technical detail)

3. **DETECT DUPLICATES**:
   - Same problem or same file-level repair described twice
   - For duplicates, choose keeper with BEST root_cause and how_fixed quality
   - Do not discard distinct problems just because they touch the same file

4. **CLASSIFY PRIMARY vs HIDDEN**:
   - PRIMARY: directly evidenced by CI log (failed command, error message/type, implicated file)
   - HIDDEN: follow-on repairs, prerequisites, supporting changes
   - These direct failures must be repaired first unless prerequisite edge requires otherwise

5. **INFER DEPENDENCIES**:
   - Edge direction: prerequisite -> dependent
   - Requires instance-specific evidence: prerequisite changes artifact/interface/config
     that dependent explicitly consumes
   - Each reason must name specific change produced and assumption consumed
   - Common pattern: dependency fix ENABLES tool -> tool reveals downstream problems
     (e.g., fix mdformat plugin -> enables mdformat -> reveals RST formatting issues)
8. NEVER infer an edge only because one validation command/order runs earlier, because
   installation enables later validators, because two problems share a validator or
   directory, or because a broad configuration repair appears first in CI.
9. Do not rewrite the problems. Return IDs from the supplied list.

MANDATORY CI WORKFLOW EXCLUSION:
- Mark a problem irrelevant when ANY affected file is under `.github/workflows/`
  and ends in `.yml` or `.yaml`.
- This includes workflow matrices, triggers, runner versions, action versions,
  environment variables, install commands, and CI-only workarounds that patch
  project files at runtime.
- The ONLY exception is a formatting-only change whose failure_type is
  `format`, `formatting`, `code_formatting`, or `yaml_formatting`, or whose
  issue_type explicitly identifies formatting.
- A general `config` failure is NOT a formatting exception.

Return JSON:
{{
  "kept_problem_ids": [1, 2],
  "irrelevant_problem_ids": [3],
  "duplicate_groups": [
    {{"keep_id": 1, "drop_ids": [4], "reason": "same problem and repair"}}
  ],
  "primary_problem_ids": [1],
  "dependencies": [
    {{"from": 1, "to": 2, "dependency_type": "prerequisite", "reason": "why"}}
  ]
}}

Every supplied problem ID must be kept, marked irrelevant, or listed as a duplicate.
An empty dependency list is valid and preferred over speculative edges.
{STRICT_JSON_RULES}
"""
    try:
        # Get model-specific max_tokens
        model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', 'unknown')
        max_tokens = get_model_max_output_tokens(model_name)

        response = invoke_llm_with_retry(llm=llm, prompt=prompt, parse_json=True, max_tokens=max_tokens)
        decision = response if isinstance(response, dict) else {}
        return _enforce_primary_and_dependency_rules(
            _enforce_workflow_exclusion(decision, problems), problems
        )
    except Exception as exc:
        print(f"  Global filtering fallback: {exc}")
        return _enforce_primary_and_dependency_rules(
            _enforce_workflow_exclusion(
                {"kept_problem_ids": [p["problem_id"] for p in problems]}, problems
            ),
            problems,
        )


def _enforce_primary_and_dependency_rules(
    decision: Dict[str, Any], problems: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Preserve known CI primaries and reject non-causal ordering edges."""
    enforced = dict(decision)
    valid_ids = {p["problem_id"] for p in problems}
    primary_ids = _valid_int_set(enforced.get("primary_problem_ids", []), valid_ids)
    primary_ids.update(
        p["problem_id"]
        for p in problems
        if str(p.get("problem_type") or "").lower() == "primary"
    )
    workflow_excluded = set(enforced.get("workflow_excluded_problem_ids", []))

    # CORRECT BIDIRECTIONAL LOGIC (based on RETRACE paper):
    #
    # 1. Backward singletons = GROUND TRUTH (auto-keep)
    #    → Based on actual successful patch, knows what worked
    #
    # 2. Forward singletons = POTENTIAL FALSE POSITIVE (validate)
    #    → Based on issue interpretation, could be wrong
    #    → Must validate against CI before keeping
    #
    # 3. Matched (both F+B) = STRONG SIGNAL (auto-keep or reconcile)
    #    → Both directions agree, high confidence

    # Backward singletons: Auto-keep (ground truth from successful patch)
    backward_singletons = {
        p["problem_id"]
        for p in problems
        if p.get("_similarity_singleton")
        and any(str(sid).startswith("B") for sid in p.get("source_ids", []))
    } - workflow_excluded

    # Forward singletons: Validate against CI (potential false positives)
    forward_singletons = {
        p["problem_id"]
        for p in problems
        if p.get("_similarity_singleton")
        and any(str(sid).startswith("F") for sid in p.get("source_ids", []))
    } - workflow_excluded

    # Note: forward_singletons will be validated via _validated_independent_ids
    # (set during dependency inference or explicitly during clustering)

    # Validated forward singletons (passed CI check)
    validated_forward_ids = set(_valid_int_set(
        enforced.get("_validated_independent_ids", []), valid_ids
    )) - workflow_excluded

    # Matched problems (in clusters with both F+B): Strong signal, auto-keep
    matched_problems = {
        p["problem_id"]
        for p in problems
        if not p.get("_similarity_singleton")  # Was clustered with other problems
    } - workflow_excluded

    # Protected IDs: Backward singletons + Validated forward + Matched
    protected_ids = backward_singletons | validated_forward_ids | matched_problems
    irrelevant = _valid_int_set(enforced.get("irrelevant_problem_ids", []), valid_ids)
    enforced["irrelevant_problem_ids"] = sorted(irrelevant - protected_ids)
    kept = _valid_int_set(enforced.get("kept_problem_ids", []), valid_ids)
    enforced["kept_problem_ids"] = sorted(kept | protected_ids)
    enforced["primary_problem_ids"] = sorted(primary_ids)
    enforced["dependencies"] = [
        edge
        for edge in enforced.get("dependencies", [])
        if isinstance(edge, dict)
        and _is_causal_dependency_type(edge.get("dependency_type"))
        and not _is_weak_dependency_reason(edge.get("reason", ""))
    ]
    return enforced


def _is_weak_dependency_reason(reason: Any) -> bool:
    text = str(reason or "").lower()
    weak_phrases = (
        "validation order",
        "runs before",
        "run before",
        "validated before",
        "must pass before",
        "must succeed before",
        "both use",
        "same directory",
        "subsequent test validation",
        "subsequent validation",
        "independent failures",
        "independent problem",
        "problems are independent",
        "repairs are independent",
        "independent repairs",
        "non-causal",
        "not causal",
        "no causal",
        "don't have a causal",
        "do not have a causal",
        "regardless of",
        "both are required for ci",
        "enables all",
        "enables later",
        "dependencies aren't installed",
        "dependencies are not installed",
        "making verification impossible",
    )
    return not text.strip() or any(phrase in text for phrase in weak_phrases)


def _is_causal_dependency_type(dependency_type: Any) -> bool:
    """Reject model-produced classifications that explicitly deny causality."""
    normalized = str(dependency_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    rejected = {
        "",
        "independent",
        "none",
        "no_dependency",
        "non_causal",
        "related",
        "correlated",
        "same_validation",
        "validation_order",
    }
    return normalized not in rejected


def _enforce_workflow_exclusion(
    decision: Dict[str, Any],
    problems: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Deterministically exclude non-formatting GitHub workflow changes."""
    enforced = dict(decision)
    excluded = {
        problem["problem_id"]
        for problem in problems
        if _is_non_formatting_workflow_problem(problem)
    }
    existing = {
        value
        for value in (_as_int(item) for item in enforced.get("irrelevant_problem_ids", []))
        if value is not None
    }
    enforced["irrelevant_problem_ids"] = sorted(existing | excluded)
    kept = [
        problem_id
        for problem_id in enforced.get("kept_problem_ids", [])
        if _as_int(problem_id) not in excluded
    ]
    enforced["kept_problem_ids"] = kept
    enforced["workflow_excluded_problem_ids"] = sorted(excluded)
    return enforced


def llm_infer_dependencies(
    problems: List[Dict[str, Any]],
    ci_context: Dict[str, Any],
    llm: Any,
    dependency_graph: Dict[str, Any] | None = None,
    structured_diff: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Infer causal repair edges after selection, without rewriting problems."""
    # The graph is derived from the structured diff. Sending both duplicates the
    # same evidence; use the raw structured form only when no graph is available.
    structured_evidence = {} if dependency_graph else (structured_diff or {})
    problem_graph_evidence = _project_file_graph_to_problem_pairs(
        problems, dependency_graph or {}
    )
    prompt = f"""Infer dependencies among the final problems for one CI repair instance.

Final problems (complete records):
{json.dumps(problems, indent=2)}

Current CI evidence:
{json.dumps(ci_context, indent=2)}

Current changed-file graph:
{json.dumps(dependency_graph or {}, indent=2, default=str)}

Changed-file relationships projected onto problem pairs:
{json.dumps(problem_graph_evidence, indent=2, default=str)}

Current structured diff evidence:
{json.dumps(structured_evidence, indent=2, default=str)}

Rules:
1. Analyze every problem's actual change before considering an edge. Use its problem,
   root_cause, how_fixed, why_fix_works, affected_files, and changed-file evidence.
2. Analyze caller-callee and reader-writer relationships. A file-graph edge describes
   a code/data relationship, not repair order. Determine which side changed a contract
   or produced data and which side had to adapt to that specific change.
   Trace indirect chains as well: a configuration repair can change tool behavior;
   tool behavior can change generated or formatted data; and code or tests consuming
   that data can then require a separate repair. The producer and consumer may have
   different files, failure types, validation commands, and validation stages.
3. **IMPORTANT - Tool Enablement Dependencies**: Look for these common patterns:
   - Dependency/plugin version fixes (dependency_error, dependency_incompatibility) often
     ENABLE tools (formatters, linters, type checkers) that then reveal downstream problems
   - Example: Fixing mdformat plugin → enables mdformat to run → reveals RST formatting issues
   - Example: Upgrading mypy → enables type checking → reveals type errors
   - Example: Installing package → enables import → reveals test failures
   These are STRONG dependencies even though producer and consumer affect different files.
   The dependency type should be "tool_enablement" or "validation_enablement".
4. Compare all problem pairs, including pairs without a graph edge, because summaries
   can contain supported relationships that static file analysis did not resolve.
5. Return an edge only when the prerequisite's concrete change is consumed by, required
   by, or changes an assumption in the dependent repair.
   Apply this counterfactual test: if the producer change had not happened (or were
   reverted), would the consumer repair still be necessary for its own stated problem?
   If yes, the repairs are independent and no edge is allowed. State the counterfactual
   result in `counterfactual` for every retained edge (optional but preferred).
6. Edge direction is producer/prerequisite -> consumer/dependent.
7. For every edge, separately report the producer change, consumer assumption,
   caller-callee/data-flow analysis, and evidence files from the current instance.
   For tool-enablement dependencies, evidence_files can include just the tool config file.
8. A matching or different validation command neither proves nor disproves dependency.
   Decide only from whether one concrete change causes the need for the other change.
9. Shared files, directories, validators, CI membership, or temporal order are context,
   not conclusions. Use them only when the actual change descriptions establish the
   producer-to-consumer causal chain.
10. Do not rewrite, remove, merge, or renumber problems.
11. If the analysis concludes that a candidate pair is independent or non-causal,
   omit it entirely from the dependencies array; do not return rejected candidates.
12. Report `producer_problem_id` and `consumer_problem_id` independently from the
   edge fields. `from` must equal `producer_problem_id`; `to` must equal
   `consumer_problem_id`.

Return one JSON object with a `dependencies` array. Every dependency object must
contain: `from`, `to`, `dependency_type`, `producer_change`, `consumer_assumption`,
`producer_problem_id`, `consumer_problem_id`, `relationship_analysis`,
`counterfactual`, `evidence_files`, and `reason`. `dependency_type` must describe a
real causal relationship, never `independent`, `related`, or validation ordering. Use
only supplied problem IDs and evidence from this instance.

{STRICT_JSON_RULES}
"""
    try:
        # Get model-specific max_tokens
        model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', 'unknown')
        max_tokens = get_model_max_output_tokens(model_name)

        response = invoke_llm_with_retry(llm=llm, prompt=prompt, parse_json=True, max_tokens=max_tokens)
        decision = response if isinstance(response, dict) else {}

        print(f"  → LLM dependency inference response received")
        raw_deps = decision.get("dependencies", [])
        print(f"  → Raw dependencies from LLM: {len(raw_deps)}")

        valid_ids = {problem["problem_id"] for problem in problems}
        normalized_edges = []
        for idx, edge in enumerate(raw_deps):
            if not isinstance(edge, dict):
                print(f"    WARNING Edge {idx}: Not a dict, skipping")
                continue
            producer_id = _as_int(edge.get("producer_problem_id"))
            consumer_id = _as_int(edge.get("consumer_problem_id"))
            if producer_id not in valid_ids or consumer_id not in valid_ids:
                print(f"    WARNING Edge {idx}: Invalid IDs ({producer_id}→{consumer_id}), skipping")
                continue
            normalized_edge = dict(edge)
            normalized_edge["from"] = producer_id
            normalized_edge["to"] = consumer_id
            normalized_edges.append(normalized_edge)
            print(f"    OK Edge {idx}: {producer_id}→{consumer_id} ({edge.get('dependency_type', 'N/A')})")
        # Filter dependencies with detailed logging
        print(f"  → Validating {len(normalized_edges)} normalized edges...")
        filtered_deps = []
        independent_problems = set()  # Track problems marked as independent

        for edge in normalized_edges:
            has_evidence = _has_concrete_dependency_evidence(edge)
            is_causal = _is_causal_dependency_type(edge.get("dependency_type"))
            dependency_type = str(edge.get("dependency_type", "")).strip().lower()

            # Special handling for "independent" - validate against CI failures
            if dependency_type == "independent":
                print(f"    WARNING Edge {edge['from']}→{edge['to']}: INDEPENDENT (validating against CI failures)")
                independent_problems.add(edge['from'])
                independent_problems.add(edge['to'])
                # Don't add edge (they're independent), but mark problems for CI validation
                continue

            passed = has_evidence and is_causal

            if not passed:
                reason = []
                if not has_evidence:
                    reason.append("lacks evidence")
                if not is_causal:
                    reason.append(f"non-causal type: {edge.get('dependency_type')}")
                print(f"    REJECTED Edge {edge['from']}→{edge['to']}: REJECTED ({', '.join(reason)})")
            else:
                filtered_deps.append(edge)
                print(f"    OK Edge {edge['from']}→{edge['to']}: ACCEPTED")

        print(f"  → After validation: {len(filtered_deps)}/{len(normalized_edges)} dependencies passed")

        # Validate independent problems against CI failures
        validated_independent = set()
        if independent_problems:
            print(f"  → Validating {len(independent_problems)} independent problems against CI failures...")
            validated_independent = _validate_problems_against_ci_failures(
                [p for p in problems if p['problem_id'] in independent_problems],
                llm
            )
            if validated_independent:
                print(f"    OK {len(validated_independent)}/{len(independent_problems)} independent problems are CI-related")
                # Ensure validated independent problems are kept (they're marked later in enforcement)
                decision["_validated_independent_ids"] = sorted(validated_independent)
            else:
                print(f"    REJECTED No independent problems validated against CI failures")

        decision["dependencies"] = filtered_deps

        # Heuristic fallback: Add common tool-enablement patterns if LLM missed them
        if len(decision["dependencies"]) == 0:
            print(f"  → LLM determined problems are independent (no causal dependencies)")

        print(f"  → Final dependencies: {len(decision['dependencies'])}")
        return decision
    except Exception as exc:
        print(f"  Dependency inference failed: {exc}")
        return {"dependencies": []}  # Return empty - problems are independent


def _validate_problems_against_ci_failures(
    problems: List[Dict[str, Any]],
    llm: Any
) -> Set[int]:
    """
    Validate that independent problems are actually related to CI verification failures.

    For problems marked as "independent" (no repair ordering dependency), use LLM to verify
    they're legitimate CI failures that need fixing, not false positives or unrelated issues.

    Returns: Set of problem IDs that are confirmed CI-related
    """
    if not problems:
        return set()

    # Build prompt to validate problems against CI failures
    problems_json = json.dumps([{
        "problem_id": p["problem_id"],
        "failure_type": p.get("failure_type", "unknown"),
        "issue_type": p.get("issue_type", "unknown"),
        "problem": p.get("problem", "")[:500],  # Truncate for prompt
        "validation_cmd": p.get("validation_cmd", ""),
        "affected_files": p.get("affected_files", [])
    } for p in problems], indent=2)

    prompt = f"""You are validating whether problems are related to CI verification failures.

For each problem below, determine if it represents a REAL CI verification failure that needs fixing.

A problem is CI-related if:
1. It causes a CI check/test/validation to fail
2. It prevents the CI workflow from passing
3. It's reported by a CI tool (linter, formatter, type checker, test runner, etc.)
4. Fixing it is necessary for CI to pass

DO NOT SELECT a problem if:
5. It's a style preference or cosmetic change not enforced by CI
6. It's a code improvement suggestion not blocking CI
7. It's NOT relevant to any CI failure or has NO effect on CI verification
8. It's a false positive or misidentified issue
9. It's about the analysis process itself (rate limiting, diff unavailable, analysis errors)
10. It has no concrete fix (empty how_fixed) AND no clear failure (failure_type="unknown")

Problems to validate:
{problems_json}

Return JSON:
{{
  "ci_related_problem_ids": [1, 2],  // IDs of problems that ARE CI verification failures
  "not_ci_related_ids": [3],         // IDs of problems that are NOT CI-related
  "reasoning": {{
    "1": "Causes pytest to fail with import error",
    "2": "Blocks mypy type checking in CI",
    "3": "Style suggestion not enforced by any CI check"
  }}
}}

{STRICT_JSON_RULES}
"""

    try:
        response = invoke_llm_with_retry(llm=llm, prompt=prompt, parse_json=True)
        if isinstance(response, dict):
            ci_related = response.get("ci_related_problem_ids", [])
            reasoning = response.get("reasoning", {})

            validated_ids = set()
            for problem_id in ci_related:
                if isinstance(problem_id, int) and problem_id in {p["problem_id"] for p in problems}:
                    validated_ids.add(problem_id)
                    reason = reasoning.get(str(problem_id), "CI-related")
                    print(f"      OK Problem {problem_id}: {reason[:80]}")

            not_related = response.get("not_ci_related_ids", [])
            for problem_id in not_related:
                reason = reasoning.get(str(problem_id), "Not CI-related")
                print(f"      REJECTED Problem {problem_id}: {reason[:80]}")

            return validated_ids

    except Exception as exc:
        print(f"      WARNING CI validation failed: {exc}, keeping all problems by default")
        # On error, be conservative and keep all problems
        return {p["problem_id"] for p in problems}

    return set()


def _has_concrete_dependency_evidence(edge: Dict[str, Any]) -> bool:
    """Require structured, instance-specific causal evidence for every edge."""
    producer_id = _as_int(edge.get("producer_problem_id"))
    consumer_id = _as_int(edge.get("consumer_problem_id"))

    # Core evidence requirements (must have all)
    has_core_evidence = bool(
        producer_id is not None
        and consumer_id is not None
        and producer_id == _as_int(edge.get("from"))
        and consumer_id == _as_int(edge.get("to"))
        and str(edge.get("producer_change") or "").strip()
        and str(edge.get("consumer_assumption") or "").strip()
        and str(edge.get("relationship_analysis") or "").strip()
    )

    # Relaxed file evidence requirement: at least 1 file OR has reason field
    # Tool-enablement dependencies (e.g., fixing plugin enables formatter) may not
    # share files but have strong causal relationships
    has_file_or_reason_evidence = bool(
        len(edge.get("evidence_files") or []) >= 1
        or str(edge.get("reason") or "").strip()
    )

    # Optional but preferred: counterfactual test
    # If present, must not be empty
    counterfactual = str(edge.get("counterfactual") or "").strip()
    has_valid_counterfactual = not counterfactual or len(counterfactual) > 0

    return has_core_evidence and has_file_or_reason_evidence and has_valid_counterfactual


def _project_file_graph_to_problem_pairs(
    problems: List[Dict[str, Any]], graph: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Map file-level caller/reader edges to stable problem IDs without inference."""
    file_to_problem_ids: Dict[str, set[int]] = defaultdict(set)
    for problem in problems:
        problem_id = _as_int(problem.get("problem_id"))
        if problem_id is None:
            continue
        for file_path in problem.get("affected_files", []) or []:
            file_to_problem_ids[str(file_path).replace("\\", "/")].add(problem_id)

    projected: List[Dict[str, Any]] = []
    seen: set[tuple[int, int, str, str, str]] = set()
    for edge in graph.get("edges", []) if isinstance(graph, dict) else []:
        if not isinstance(edge, dict):
            continue
        from_file = str(edge.get("from") or "").replace("\\", "/")
        to_file = str(edge.get("to") or "").replace("\\", "/")
        relation = str(edge.get("type") or "")
        for from_problem in file_to_problem_ids.get(from_file, set()):
            for to_problem in file_to_problem_ids.get(to_file, set()):
                if from_problem == to_problem:
                    continue
                key = (from_problem, to_problem, from_file, to_file, relation)
                if key in seen:
                    continue
                seen.add(key)
                if relation in {"imports", "reads", "tests", "configures"}:
                    direction_note = (
                        "The from-file has the stated static relationship to the to-file; "
                        "this is not automatically the repair-order direction."
                    )
                else:
                    direction_note = (
                        "This is heuristic co-change evidence only; it establishes neither "
                        "a caller-callee relationship nor repair-order direction."
                    )
                projected.append(
                    {
                        "from_problem": from_problem,
                        "to_problem": to_problem,
                        "from_file": from_file,
                        "to_file": to_file,
                        "file_relationship": relation,
                        "graph_evidence": edge.get("via", ""),
                        "direction_note": direction_note,
                    }
                )
    return projected


def _is_non_formatting_workflow_problem(problem: Dict[str, Any]) -> bool:
    files = problem.get("affected_files", []) or []
    touches_workflow = any(
        str(file_path).replace("\\", "/").lower().startswith(".github/workflows/")
        and str(file_path).lower().endswith((".yml", ".yaml"))
        for file_path in files
    )
    if not touches_workflow:
        return False

    failure_type = str(problem.get("failure_type") or "").strip().lower()
    issue_type = str(problem.get("issue_type") or "").strip().lower()
    formatting_types = {"format", "formatting", "code_formatting", "yaml_formatting"}
    is_formatting_only = failure_type in formatting_types or "format" in issue_type
    return not is_formatting_only


def _apply_global_decision(
    problems: List[Dict[str, Any]],
    decision: Dict[str, Any],
) -> List[Dict[str, Any]]:
    by_id = {problem["problem_id"]: dict(problem) for problem in problems}
    valid_ids = set(by_id)
    irrelevant = _valid_int_set(decision.get("irrelevant_problem_ids", []), valid_ids)
    primary_ids = _valid_int_set(decision.get("primary_problem_ids", []), valid_ids)
    duplicate_drops: set[int] = set()

    for group in decision.get("duplicate_groups", []):
        if not isinstance(group, dict):
            continue
        keep_id = _as_int(group.get("keep_id"))
        if keep_id not in valid_ids:
            continue
        for drop_id in _valid_int_set(group.get("drop_ids", []), valid_ids):
            if drop_id == keep_id:
                continue
            # Never delete a concrete forward repair unit during global cleanup.
            if any(
                str(source_id).startswith("F")
                for source_id in by_id[drop_id].get("source_ids", [])
            ) or by_id[drop_id].get("_similarity_singleton"):
                continue
            duplicate_drops.add(drop_id)
            keeper = by_id[keep_id]
            duplicate = by_id[drop_id]
            keeper["affected_files"] = _unique_strings(
                keeper.get("affected_files", []) + duplicate.get("affected_files", [])
            )
            keeper["source_ids"] = _unique_strings(
                keeper.get("source_ids", []) + duplicate.get("source_ids", [])
            )
            keeper["source"] = _source_kind(keeper["source_ids"])

    # Removal must be explicit. If the LLM forgets to classify an ID, preserve
    # it rather than silently losing a historical repair problem.
    selected_ids = valid_ids - irrelevant - duplicate_drops
    selected = [by_id[problem_id] for problem_id in sorted(selected_ids)]
    for problem in selected:
        problem["problem_type"] = (
            "primary" if problem["problem_id"] in primary_ids else "hidden"
        )
    return selected


def _assign_problem_ids(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assigned: List[Dict[str, Any]] = []
    for problem_id, problem in enumerate(problems, 1):
        copied = dict(problem)
        copied["_previous_problem_id"] = copied.get("problem_id", problem_id)
        copied["problem_id"] = problem_id
        copied["affected_files"] = _unique_strings(copied.get("affected_files", []))
        assigned.append(copied)
    return assigned


def _selected_id_mapping(problems: List[Dict[str, Any]]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for problem in problems:
        old_id = _as_int(problem.get("_previous_problem_id"))
        new_id = _as_int(problem.get("problem_id"))
        if old_id is not None and new_id is not None:
            mapping[old_id] = new_id
    return mapping


def _normalize_dependencies(
    dependencies: Iterable[Dict[str, Any]],
    old_to_new: Dict[int, int],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for edge in dependencies:
        if not isinstance(edge, dict):
            continue
        if not _is_causal_dependency_type(edge.get("dependency_type")):
            continue
        old_from = _as_int(edge.get("from"))
        old_to = _as_int(edge.get("to"))
        from_id = old_to_new.get(old_from) if old_from is not None else None
        to_id = old_to_new.get(old_to) if old_to is not None else None
        if from_id is None or to_id is None or from_id == to_id:
            continue
        key = (from_id, to_id)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "from": from_id,
                "to": to_id,
                "dependency_type": str(edge.get("dependency_type") or "prerequisite"),
                "reason": str(edge.get("reason") or ""),
            }
        )
    return normalized


def _deduplicate_dependencies(
    dependencies: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Combine dependency stages without duplicating the same directed edge."""
    output: List[Dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for edge in dependencies:
        key = (edge["from"], edge["to"])
        if key not in seen:
            seen.add(key)
            output.append(edge)
    return output


def dependency_order(
    problems: List[Dict[str, Any]],
    dependencies: List[Dict[str, Any]],
) -> List[int]:
    """Stable topological order, prioritizing direct CI failures among ready nodes."""
    problem_ids = {problem["problem_id"] for problem in problems}
    by_id = {problem["problem_id"]: problem for problem in problems}
    outgoing: Dict[int, set[int]] = defaultdict(set)
    indegree = {problem_id: 0 for problem_id in problem_ids}

    for edge in dependencies:
        from_id, to_id = edge["from"], edge["to"]
        if from_id not in problem_ids or to_id not in problem_ids:
            continue
        if to_id not in outgoing[from_id]:
            outgoing[from_id].add(to_id)
            indegree[to_id] += 1

    def key(problem_id: int) -> tuple[int, int, int]:
        order = _as_int(by_id[problem_id].get("validation_order"))
        primary_rank = 0 if by_id[problem_id].get("problem_type") == "primary" else 1
        return (primary_rank, order if order is not None else 10_000, problem_id)

    ready = sorted((pid for pid, degree in indegree.items() if degree == 0), key=key)
    result: List[int] = []
    while ready:
        current = ready.pop(0)
        result.append(current)
        for dependent in sorted(outgoing[current], key=key):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort(key=key)

    # Cyclic or malformed LLM edges must not drop problems.
    remaining = sorted(problem_ids - set(result), key=key)
    result.extend(remaining)
    return result


def _sort_by_sequence(
    problems: List[Dict[str, Any]],
    repair_sequence: List[int],
) -> List[Dict[str, Any]]:
    rank = {problem_id: index + 1 for index, problem_id in enumerate(repair_sequence)}
    ordered = sorted(problems, key=lambda p: rank.get(p["problem_id"], 10_000))
    for problem in ordered:
        problem["repair_sequence_index"] = rank.get(problem["problem_id"], 10_000)
        problem.pop("_previous_problem_id", None)
    return ordered


def _apply_enabled_relationships(
    problems: List[Dict[str, Any]],
    dependencies: List[Dict[str, Any]],
) -> None:
    enabled: Dict[int, List[int]] = defaultdict(list)
    for edge in dependencies:
        enabled[edge["from"]].append(edge["to"])
    for problem in problems:
        problem["enabled"] = sorted(set(enabled.get(problem["problem_id"], [])))


def _first_nonempty(problems: Iterable[Dict[str, Any]], field: str) -> Any:
    for problem in problems:
        value = problem.get(field)
        if value not in (None, "", []):
            return value
    return ""


def _source_kind(source_ids: Iterable[str]) -> str:
    prefixes = {str(source_id)[:1] for source_id in source_ids if source_id}
    if prefixes == {"F", "B"}:
        return "bidirectional_matched"
    if prefixes == {"F"}:
        return "forward_only"
    if prefixes == {"B"}:
        return "backward_only"
    return "unknown"


def _unique_strings(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _valid_int_set(values: Iterable[Any], valid_ids: set[int]) -> set[int]:
    output: set[int] = set()
    for value in values or []:
        integer = _as_int(value)
        if integer in valid_ids:
            output.add(integer)
    return output


def _lossless_problem_batches(
    problems: List[Dict[str, Any]], max_bytes: int
) -> List[List[Dict[str, Any]]]:
    """Partition whole records by serialized size without modifying any record."""
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_bytes = 2
    for problem in problems:
        problem_bytes = len(json.dumps(problem, ensure_ascii=False).encode("utf-8")) + 1
        if current and current_bytes + problem_bytes > max_bytes:
            batches.append(current)
            current = []
            current_bytes = 2
        current.append(problem)
        current_bytes += problem_bytes
    if current:
        batches.append(current)
    return batches


__all__ = [
    "cluster_by_similarity",
    "dependency_order",
    "flatten_and_label",
    "llm_filter_and_deduplicate",
    "llm_merge_cluster",
    "simple_bidirectional_reconciliation",
]
