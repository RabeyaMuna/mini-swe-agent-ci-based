#!/usr/bin/env python3
"""
Semantic Dependency Analysis (works WITHOUT validation_groups)

This approach works with the CURRENT decomposition structure:
1. Group similar problems (same validation_cmd, failure_type)
2. Use LLM to infer dependencies based on semantic understanding
3. Generate enables/enabled_by relationships

Usage:
    python scripts/semantic_dependency_analysis.py \
        --decomposed data/trs/decomposed_issues.json \
        --output data/trs/decomposed_with_semantic_deps.json \
        --model openrouter/anthropic/claude-3.5-sonnet
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.llm_model import LitellmModel


# ============================================================================
# STEP 1: Group Similar Problems
# ============================================================================

def group_similar_problems(issues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group problems by validation_cmd + failure_type.

    Returns:
        Dict mapping group_key -> list of (issue_id, problem_id, problem_data)
    """
    groups = defaultdict(list)

    for issue in issues:
        if "error" in issue:
            continue

        issue_id = issue.get("original_issue_id", "?")
        problems = issue.get("problems", [])

        for problem in problems:
            validation_cmd = problem.get("validation_cmd", "unknown")
            failure_type = problem.get("failure_type", "unknown")

            # Create group key
            group_key = f"{validation_cmd}|{failure_type}"

            # Store with metadata
            groups[group_key].append({
                "issue_id": issue_id,
                "problem_id": problem.get("problem_id"),
                "problem": problem.get("problem", ""),
                "root_cause": problem.get("root_cause", ""),
                "how_fixed": problem.get("how_fixed", ""),
                "why_fix_works": problem.get("why_fix_works", ""),
                "affected_files": problem.get("affected_files", []),
                "problem_type": problem.get("problem_type", ""),
                "validation_order": problem.get("validation_order"),
            })

    return dict(groups)


def merge_duplicate_problems(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Within a group, merge problems that are essentially the same fix.

    Criteria for merging:
    - Similar how_fixed description (>80% similarity)
    - Overlapping affected_files or same file patterns
    """
    from difflib import SequenceMatcher

    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    merged = []
    used = set()

    for i, problem in enumerate(group):
        if i in used:
            continue

        # Find similar problems
        similar = [problem]
        for j in range(i + 1, len(group)):
            if j in used:
                continue

            other = group[j]

            # Check similarity
            fix_sim = similarity(problem["how_fixed"], other["how_fixed"])

            # Check file overlap
            files_a = set(problem["affected_files"])
            files_b = set(other["affected_files"])
            file_overlap = len(files_a & files_b) / max(len(files_a | files_b), 1)

            if fix_sim > 0.8 or file_overlap > 0.5:
                similar.append(other)
                used.add(j)

        # Merge similar problems
        if len(similar) > 1:
            merged_problem = problem.copy()
            merged_problem["merged_from"] = [p["problem_id"] for p in similar]
            merged_problem["affected_files"] = list(set(
                f for p in similar for f in p["affected_files"]
            ))
            merged.append(merged_problem)
        else:
            merged.append(problem)

        used.add(i)

    return merged


# ============================================================================
# STEP 2: LLM-Based Dependency Inference
# ============================================================================

def build_dependency_prompt(group_key: str, problems: List[Dict[str, Any]]) -> str:
    """
    Build prompt for LLM to infer dependencies within a validation group.
    """
    validation_cmd, failure_type = group_key.split("|", 1)

    prompt = f"""You are analyzing CI failures and fixes to determine repair dependencies.

VALIDATION CONTEXT:
- Command: {validation_cmd}
- Failure Type: {failure_type}
- Total Problems: {len(problems)}

PROBLEMS TO ANALYZE:
"""

    for i, problem in enumerate(problems, 1):
        prompt += f"""
Problem {i}:
  ID: Issue {problem['issue_id']}, Problem {problem['problem_id']}
  Problem: {problem['problem']}
  Root Cause: {problem['root_cause']}
  How Fixed: {problem['how_fixed']}
  Why Fix Works: {problem['why_fix_works']}
  Files: {', '.join(problem['affected_files'][:5])}{'...' if len(problem['affected_files']) > 5 else ''}
"""

    prompt += """
TASK:
Analyze the dependencies between these problems. For each problem, determine:

1. **enables**: Which other problems does this fix enable/unblock?
   - Config changes that enable code changes
   - Fixes that must happen before others
   - Root fixes that unblock symptom fixes

2. **enabled_by**: Which problems must be fixed before this one?
   - Prerequisites that must be resolved first
   - Blocking issues

3. **repair_order**: Suggested order (1 = fix first, higher = fix later)

OUTPUT FORMAT (JSON):
{
  "dependencies": [
    {
      "problem_ref": "Issue X, Problem Y",
      "enables": ["Issue A, Problem B", "Issue C, Problem D"],
      "enabled_by": ["Issue E, Problem F"],
      "repair_order": 1,
      "reasoning": "Why this dependency exists"
    }
  ],
  "repair_sequence": [
    {
      "order": 1,
      "problems": ["Issue X, Problem Y"],
      "rationale": "Fix first because..."
    }
  ]
}

IMPORTANT:
- Only specify dependencies that are LOGICALLY necessary
- Empty arrays [] are fine if no dependencies exist
- Focus on causal relationships, not just similar problems
- Consider: config -> code, root cause -> symptom, blocking -> dependent
"""

    return prompt


def infer_dependencies_with_llm(
    group_key: str,
    problems: List[Dict[str, Any]],
    llm: Any
) -> Dict[str, Any]:
    """
    Use LLM to infer dependencies between problems in a group.
    """
    prompt = build_dependency_prompt(group_key, problems)

    try:
        response = llm.generate(prompt, max_tokens=4000, temperature=0.0)

        # Extract JSON from response
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        result = json.loads(response)
        return result

    except Exception as e:
        print(f"  WARNING  LLM failed for {group_key}: {e}")
        return {"dependencies": [], "repair_sequence": []}


# ============================================================================
# STEP 3: Apply Dependencies to Original Issues
# ============================================================================

def apply_dependencies_to_issues(
    issues: List[Dict[str, Any]],
    group_dependencies: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Apply inferred dependencies back to the original decomposed issues.
    """
    # Build lookup: (issue_id, problem_id) -> dependency info
    dep_lookup = {}

    for group_key, dep_info in group_dependencies.items():
        for dep in dep_info.get("dependencies", []):
            problem_ref = dep.get("problem_ref", "")
            if "Issue" in problem_ref and "Problem" in problem_ref:
                parts = problem_ref.split(",")
                issue_id = parts[0].replace("Issue", "").strip()
                problem_id = int(parts[1].replace("Problem", "").strip())

                dep_lookup[(issue_id, problem_id)] = {
                    "enables": dep.get("enables", []),
                    "enabled_by": dep.get("enabled_by", []),
                    "repair_order": dep.get("repair_order", 999),
                    "reasoning": dep.get("reasoning", "")
                }

    # Apply to issues
    for issue in issues:
        if "error" in issue:
            continue

        issue_id = issue.get("original_issue_id")

        for problem in issue.get("problems", []):
            problem_id = problem.get("problem_id")
            key = (issue_id, problem_id)

            if key in dep_lookup:
                dep_info = dep_lookup[key]
                problem["enables"] = dep_info["enables"]
                problem["enabled_by"] = dep_info["enabled_by"]
                problem["repair_order"] = dep_info["repair_order"]
                problem["dependency_reasoning"] = dep_info["reasoning"]

    return issues


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Semantic dependency analysis (works without validation_groups)"
    )
    parser.add_argument("--decomposed", required=True, help="Path to decomposed issues")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--model", default="openrouter/anthropic/claude-3.5-sonnet")
    args = parser.parse_args()

    # Load decomposed issues
    print(f"Loading decomposed issues from: {args.decomposed}")
    with open(args.decomposed) as f:
        issues = json.load(f)
    print(f"Loaded {len(issues)} issues")

    # Initialize LLM
    print(f"\nInitializing LLM: {args.model}")
    llm = LitellmModel(model_name=args.model)

    # STEP 1: Group similar problems
    print("\n" + "="*80)
    print("STEP 1: GROUP SIMILAR PROBLEMS")
    print("="*80)

    groups = group_similar_problems(issues)
    print(f"Found {len(groups)} validation groups:")
    for group_key, problems in groups.items():
        print(f"  {group_key}: {len(problems)} problems")

    # STEP 2: Merge duplicates within groups
    print("\n" + "="*80)
    print("STEP 2: MERGE DUPLICATE PROBLEMS")
    print("="*80)

    merged_groups = {}
    for group_key, problems in groups.items():
        merged = merge_duplicate_problems(problems)
        merged_groups[group_key] = merged
        if len(merged) < len(problems):
            print(f"  {group_key}: {len(problems)} -> {len(merged)} problems (merged duplicates)")

    # STEP 3: LLM dependency inference
    print("\n" + "="*80)
    print("STEP 3: INFER DEPENDENCIES WITH LLM")
    print("="*80)

    group_dependencies = {}
    for group_key, problems in merged_groups.items():
        if len(problems) <= 1:
            print(f"  {group_key}: Only 1 problem, skipping dependency analysis")
            continue

        print(f"\n  Analyzing: {group_key} ({len(problems)} problems)...")
        deps = infer_dependencies_with_llm(group_key, problems, llm)
        group_dependencies[group_key] = deps

        dep_count = len(deps.get("dependencies", []))
        print(f"    -> Found {dep_count} dependency relationships")

    # STEP 4: Apply dependencies to original issues
    print("\n" + "="*80)
    print("STEP 4: APPLY DEPENDENCIES TO ISSUES")
    print("="*80)

    enhanced_issues = apply_dependencies_to_issues(issues, group_dependencies)

    # Save output
    print(f"\nSaving enhanced issues to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(enhanced_issues, f, indent=2)

    print("\nOK Complete! Enhanced decomposition with semantic dependencies saved.")

    # Summary
    total_deps = sum(
        len(p.get("enables", []) + p.get("enabled_by", []))
        for issue in enhanced_issues
        for p in issue.get("problems", [])
    )
    print(f"\nSummary:")
    print(f"  - Total issues: {len(enhanced_issues)}")
    print(f"  - Total dependency relationships: {total_deps}")


if __name__ == "__main__":
    main()
