#!/usr/bin/env python3
"""
build_memory_from_decomposed.py - Build L1/L2/L3 Memory from Decomposed Problems
==================================================================================

Takes decomposed_issues.json (output from decompose_ci_failure.py) and builds:
- L1 (per-file memory with reasoning)
- L2 (per-issue with atomic problems)
- L3 (cross-repo with hierarchical abstraction)

Structure:
  L1: File-level (within-repo)
      - Each file with failure + fix + reasoning
      - Caller/callee dependencies

  L2: Issue-level (within-repo)
      - Multiple atomic problems per issue
      - Each with visibility (visible_in_log vs hidden)
      - Dependencies between problems
      - Repair trajectory inference

  L3: Universal principles (cross-repo)
      - Hierarchical abstraction (3 levels)
      - Evidence from multiple L2 problems

Usage:
    python scripts/build_memory_from_decomposed.py \\
        --decomposed data/trs/decomposed_issues.json \\
        --output-dir data/trs_memory_v2
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minisweagent.models.litellm_model import LiteLLMModel


def build_l1_memory(decomposed_issues: List[Dict]) -> List[Dict]:
    """
    Build L1 (per-file) memory from decomposed atomic problems.

    Each L1 entry:
    - Represents ONE file in ONE atomic problem
    - Has reasoning (WHY failed, HOW fixed, WHY works)
    - Has caller/callee dependencies
    - Links to parent L2 atomic problem
    """

    l1_memories = []

    for issue in decomposed_issues:
        if "error" in issue:
            continue

        issue_id = issue.get("original_issue_id")
        repo = issue.get("repo")
        sha_fail = issue.get("sha_fail")

        for problem in issue.get("problems", []):
            problem_id = problem.get("problem_id")
            full_problem_id = f"{issue_id}_p{problem_id}"

            for file_path in problem.get("affected_files", []):
                l1_entry = {
                    # Identification
                    "file": file_path,
                    "repo": repo,
                    "issue_id": issue_id,
                    "sha_fail": sha_fail,
                    "atomic_problem_id": full_problem_id,

                    # Failure Analysis (WHY)
                    "failure_symptom": problem.get("symptom"),
                    "failure_type": problem.get("problem_type"),
                    "root_cause": problem.get("root_cause"),
                    "why_this_file_failed": f"Part of {problem.get('problem_type')}: {problem.get('why_occurred', '')}",

                    # Fix Analysis (HOW)
                    "fix_description": problem.get("how_fixed"),
                    "why_fix_works": problem.get("why_fix_works"),
                    "diff_evidence": problem.get("diff_evidence", ""),

                    # CI Context
                    "ci_visibility": problem.get("visibility"),
                    "evidence_in_ci_log": problem.get("evidence_in_ci_log"),
                    "ci_workflow_stage": problem.get("ci_workflow_stage"),
                    "ci_command": problem.get("ci_command_that_failed") or problem.get("ci_command_that_would_fail"),

                    # Verification
                    "verification_strategy": problem.get("verification_after_fix"),
                    "next_failure_expected": problem.get("next_failure_after_fix"),

                    # Dependencies (placeholder - would need code analysis)
                    "file_dependencies": {
                        "callers": [],  # Files that import/call this file
                        "callees": []   # Files that this file imports/calls
                    },

                    # Pattern
                    "file_level_pattern": f"{problem.get('problem_type')} in {Path(file_path).suffix} file"
                }

                l1_memories.append(l1_entry)

    print(f"Built {len(l1_memories)} L1 (per-file) memory entries")
    return l1_memories


def build_l2_memory(decomposed_issues: List[Dict]) -> List[Dict]:
    """
    Build L2 (per-issue) memory from decomposed problems.

    Each L2 entry:
    - Represents ONE issue/commit
    - Contains MULTIPLE atomic problems
    - Has repair trajectory (sequence of fixes)
    - Links to L1 files
    """

    l2_memories = []

    for issue in decomposed_issues:
        if "error" in issue:
            continue

        issue_id = issue.get("original_issue_id")
        repo = issue.get("repo")
        sha_fail = issue.get("sha_fail")

        # Get first visible problem
        visible_problem = next(
            (p for p in issue.get("problems", []) if p.get("visibility") == "visible_in_log"),
            issue.get("problems", [{}])[0] if issue.get("problems") else {}
        )

        # Collect all affected files
        all_files = []
        for p in issue.get("problems", []):
            all_files.extend(p.get("affected_files", []))
        all_files = list(set(all_files))

        # Build atomic problems list with full IDs
        atomic_problems = []
        for p in issue.get("problems", []):
            problem_id = f"{issue_id}_p{p.get('problem_id')}"

            atomic_problem = {
                "problem_id": problem_id,
                "problem_type": p.get("problem_type"),
                "visibility": p.get("visibility"),
                "sequence_order": p.get("problem_id"),

                # Symptom Analysis
                "symptom": p.get("symptom"),
                "evidence_in_ci_log": p.get("evidence_in_ci_log"),
                "ci_workflow_stage": p.get("ci_workflow_stage"),
                "ci_command": p.get("ci_command_that_failed") or p.get("ci_command_that_would_fail"),

                # Root Cause Analysis (WHY)
                "root_cause": p.get("root_cause"),
                "why_occurred": p.get("why_occurred"),

                # Fix Analysis (HOW)
                "how_fixed": p.get("how_fixed"),
                "why_fix_works": p.get("why_fix_works"),

                # Affected Files (links to L1)
                "affected_l1_files": [
                    {"file": f, "l1_entry_id": f"{problem_id}_{Path(f).name}"}
                    for f in p.get("affected_files", [])
                ],

                # Verification
                "verification_after_fix": p.get("verification_after_fix"),
                "next_failure_after_fix": p.get("next_failure_after_fix"),

                # Dependencies
                "depends_on": f"{issue_id}_p{p.get('depends_on')}" if p.get("depends_on") else None,
                "dependency_reason": p.get("dependency_reason"),
                "enables": [f"{issue_id}_p{e}" for e in p.get("enables", [])],

                # Repo-specific pattern
                "repo_level_pattern": f"{repo}: {p.get('problem_type')} pattern"
            }

            atomic_problems.append(atomic_problem)

        l2_memory = {
            # Issue-level identification
            "issue_id": issue_id,
            "repo": repo,
            "sha_fail": sha_fail,

            # Visible CI failure
            "visible_ci_failure": visible_problem.get("symptom", ""),
            "ci_stage_failed": visible_problem.get("ci_workflow_stage", ""),

            # Multi-problem decomposition (CRITICAL!)
            "total_atomic_problems": issue.get("total_problems", 0),
            "atomic_problems": atomic_problems,

            # Repair trajectory (inferred)
            "repair_trajectory": issue.get("repair_trajectory_inference", []),

            # Overall analysis
            "overall_reasoning": issue.get("overall_reasoning", ""),
            "total_files_changed": len(all_files),
            "repair_complete": True
        }

        l2_memories.append(l2_memory)

    print(f"Built {len(l2_memories)} L2 (per-issue) memory entries")
    return l2_memories


def abstract_to_hierarchy(similar_problems: List[Dict], llm) -> Dict:
    """
    Create 3-level hierarchical abstraction (STAIR paper approach).

    Levels:
    1. Concrete: Specific but generalized (e.g., "Python CalVer→SemVer")
    2. Pattern: General strategy (e.g., "Package version scheme change")
    3. Universal: Cross-language (e.g., "Version specification evolution")
    """

    # Extract common info
    problem_type = similar_problems[0].get("problem_type")
    symptoms = [p.get("symptom") for p in similar_problems[:3]]
    fixes = [p.get("how_fixed") for p in similar_problems[:3]]

    prompt = f"""
Create 3-level hierarchical abstraction for this repair pattern.

Problem Type: {problem_type}

Example Symptoms:
{chr(10).join(f"- {s}" for s in symptoms[:3])}

Example Fixes:
{chr(10).join(f"- {f}" for f in fixes[:3])}

Create 3 abstraction levels (STAIR paper method):

Level 1 (Concrete): Keep some specifics but generalize slightly
- Example: "Python package CalVer→SemVer migration"
- Include language/ecosystem
- Keep specific patterns

Level 2 (Pattern): General strategy, no specifics
- Example: "Package dependency versioning scheme change"
- Language-agnostic strategy
- Reusable approach

Level 3 (Universal): Cross-language principle
- Example: "Version specification evolution pattern"
- Applies to any ecosystem
- High-level principle

Return JSON:
{{
  "level_1_concrete": {{
    "description": "...",
    "reusable_strategy": "...",
    "when_to_apply": "..."
  }},
  "level_2_pattern": {{
    "description": "...",
    "general_strategy": ["step1", "step2", ...],
    "when_to_apply": "..."
  }},
  "level_3_universal": {{
    "description": "...",
    "universal_principle": "...",
    "applies_to_languages": ["Python", "Node", ...]
  }}
}}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        return json.loads(content)
    except:
        # Fallback
        return {
            "level_1_concrete": {"description": f"Concrete {problem_type} pattern"},
            "level_2_pattern": {"description": f"General {problem_type} strategy"},
            "level_3_universal": {"description": f"Universal {problem_type} principle"}
        }


def build_l3_memory(l2_memories: List[Dict], llm) -> List[Dict]:
    """
    Build L3 (cross-repo universal principles) from L2 atomic problems.

    Each L3 entry:
    - Represents a universal principle
    - Has 3-level hierarchical abstraction (STAIR)
    - Has evidence from multiple L2 problems
    """

    print("Building L3 (cross-repo) memory with hierarchical abstraction...")

    # Group L2 atomic problems by type
    problems_by_type = defaultdict(list)

    for l2 in l2_memories:
        for atomic_problem in l2.get("atomic_problems", []):
            ptype = atomic_problem.get("problem_type")
            problems_by_type[ptype].append({
                **atomic_problem,
                "repo": l2.get("repo"),
                "issue_id": l2.get("issue_id")
            })

    l3_principles = []

    for problem_type, problems in problems_by_type.items():
        if len(problems) < 2:
            continue  # Need at least 2 examples for cross-repo pattern

        print(f"  Abstracting '{problem_type}' pattern ({len(problems)} examples)...")

        # Create hierarchical abstraction
        hierarchy = abstract_to_hierarchy(problems[:5], llm)  # Use first 5 examples

        principle = {
            "principle_id": f"{problem_type}_pattern",
            "principle_name": f"{problem_type.replace('_', ' ').title()} Pattern",

            # Hierarchical abstraction (STAIR)
            "abstraction_hierarchy": hierarchy,

            # Evidence from L2
            "evidence_from_l2": [
                {
                    "repo": p.get("repo"),
                    "issue": p.get("issue_id"),
                    "atomic_problem": p.get("problem_id"),
                    "symptom": p.get("symptom", "")[:80]
                }
                for p in problems[:10]  # Keep first 10 as evidence
            ],

            # Retrieval keywords
            "retrieval_keywords": [
                problem_type,
                *set(p.get("symptom", "").split()[:3] for p in problems[:5])
            ],

            # Stats
            "total_evidence_count": len(problems),
            "repos_covered": len(set(p.get("repo") for p in problems))
        }

        l3_principles.append(principle)

    print(f"Built {len(l3_principles)} L3 (cross-repo) universal principles")
    return l3_principles


def main():
    parser = argparse.ArgumentParser(
        description="Build L1/L2/L3 memory from decomposed issues"
    )
    parser.add_argument(
        "--decomposed",
        default="data/trs/decomposed_issues.json",
        help="Path to decomposed issues"
    )
    parser.add_argument(
        "--output-dir",
        default="data/trs_memory_v2",
        help="Output directory for L1/L2/L3 memory"
    )
    parser.add_argument(
        "--model",
        default="minimax/minimax-m2.5",
        help="LLM model for L3 abstraction"
    )
    args = parser.parse_args()

    # Load decomposed issues
    decomposed_path = Path(args.decomposed)
    if not decomposed_path.exists():
        print(f"✗ Decomposed issues not found: {decomposed_path}")
        print(f"Run decompose_ci_failure.py first!")
        return 1

    print(f"{'='*80}")
    print(f"Building L1/L2/L3 Memory from Decomposed Issues")
    print(f"{'='*80}\n")

    with open(decomposed_path) as f:
        decomposed_issues = json.load(f)

    print(f"Loaded {len(decomposed_issues)} decomposed issues\n")

    # Build L1 (per-file)
    print("Step 1: Building L1 (per-file) memory...")
    l1_memories = build_l1_memory(decomposed_issues)

    # Build L2 (per-issue with atomic problems)
    print("\nStep 2: Building L2 (per-issue) memory...")
    l2_memories = build_l2_memory(decomposed_issues)

    # Build L3 (cross-repo with hierarchical abstraction)
    print("\nStep 3: Building L3 (cross-repo) memory with abstraction...")
    llm = LiteLLMModel(model_name=args.model)
    l3_principles = build_l3_memory(l2_memories, llm)

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save L1 (failure_memory.json for compatibility)
    l1_path = output_dir / "failure_memory.json"
    with open(l1_path, "w") as f:
        json.dump(l1_memories, f, indent=2)
    print(f"\n✓ Saved L1 memory: {l1_path}")

    # Save L2 (repo_memory.json for compatibility)
    l2_path = output_dir / "repo_memory.json"
    with open(l2_path, "w") as f:
        json.dump(l2_memories, f, indent=2)
    print(f"✓ Saved L2 memory: {l2_path}")

    # Save L3 (cross_memory.json for compatibility)
    l3_path = output_dir / "cross_memory.json"
    with open(l3_path, "w") as f:
        json.dump(l3_principles, f, indent=2)
    print(f"✓ Saved L3 memory: {l3_path}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"L1 (per-file): {len(l1_memories)} entries")
    print(f"L2 (per-issue): {len(l2_memories)} entries")
    print(f"  Total atomic problems: {sum(l2['total_atomic_problems'] for l2 in l2_memories)}")
    print(f"L3 (cross-repo): {len(l3_principles)} principles")
    print(f"\nOutput directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
