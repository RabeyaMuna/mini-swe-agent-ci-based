#!/usr/bin/env python3
"""
Integrate Dependency Analysis into Decomposed Issues and L1/L2/L3 Memory

Pipeline:
1. Verify detailed_changes exist (added by decompose_ci_failure.py)
2. Analyze cross-problem dependencies (LLM)
3. Build L1 memory with enables/enabled_by
4. Build L2 memory with problem_groups and repair_sequence
5. Build L3 memory with universal dependency patterns

Usage:
    python scripts/integrate_dependencies.py \
        --decomposed data/trs/decomposed_issues.json \
        --output data/trs/decomposed_issues_with_deps.json \
        --model openrouter/minimax/minimax-m2.5
"""

import argparse
import json
from collections import defaultdict
import sys
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.llm_model import LitellmModel


# ============================================================================
# HELPER: Fetch Detailed Changes from Validation Groups
# ============================================================================

def get_detailed_changes_for_problem(issue: Dict[str, Any], problem: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch detailed changes for a problem from validation_groups.

    NOTE: Returns empty list [] if validation_groups don't exist.
    When validation_groups are missing, L1 memory falls back to high-level
    problem descriptions instead of line-by-line code changes.

    Args:
        issue: The issue containing validation_groups
        problem: The problem to get changes for

    Returns:
        List of detailed changes for this problem's files (or [] if unavailable)
    """
    validation_order = problem.get("validation_order")
    if not validation_order:
        return []

    # Get validation_groups from diff_analysis_context
    diff_context = issue.get("diff_analysis_context", {})
    validation_groups = diff_context.get("validation_groups", {})

    if not validation_groups:
        return []  # Validation groups not available - use semantic analysis fallback

    # Get all_changes for this validation_order
    val_group = validation_groups.get(str(validation_order), {})
    all_changes = val_group.get("all_changes", [])

    if not all_changes:
        return []

    # Filter to only this problem's files
    affected_files = problem.get("affected_files", [])
    if affected_files:
        return [c for c in all_changes if c.get("file") in affected_files]
    else:
        return all_changes


# ============================================================================
# PHASE 0.5: Merge Duplicate Problems
# ============================================================================

def merge_duplicate_problems_in_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge duplicate problems within an issue (DETERMINISTIC).

    Merging Strategy (fully deterministic, no fuzzy matching):
    - Groups problems by: validation_cmd + failure_type + issue_type
    - Combines affected_files from all problems in group
    - Uses most detailed description

    This is deterministic because:
    - Only uses exact string equality (no similarity thresholds)
    - Result is independent of input order
    - Same input always produces same output
    """
    problems = issue.get("problems", [])
    if len(problems) <= 1:
        return issue

    # Group by exact key match
    groups = defaultdict(list)

    for p in problems:
        key = (
            p.get("validation_cmd", ""),
            p.get("failure_type", ""),
            p.get("issue_type", "")
        )
        groups[key].append(p)

    # Build merged problems
    merged_problems = []

    for key, group in groups.items():
        if len(group) == 1:
            # No merge needed
            merged_problems.append(group[0])
        else:
            # Merge group
            base = group[0].copy()

            # Combine affected files (preserve order, remove duplicates)
            all_files = []
            for p in group:
                all_files.extend(p.get("affected_files", []))
            base["affected_files"] = list(dict.fromkeys(all_files))

            # Track merged problem IDs
            base["merged_from_problem_ids"] = [p.get("problem_id") for p in group]
            base["original_problem_count"] = len(group)

            # Use most detailed description
            base["how_fixed"] = max(
                (p.get("how_fixed", "") for p in group),
                key=len
            )
            base["root_cause"] = max(
                (p.get("root_cause", "") for p in group),
                key=len
            )

            merged_problems.append(base)

    # Update issue
    issue["problems"] = merged_problems
    issue["original_problem_count"] = len(problems)
    issue["merged_problem_count"] = len(merged_problems)

    return issue


def phase0_merge_duplicates(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Phase 0: Merge duplicate problems within each issue.
    """
    print("\n" + "="*80)
    print("PHASE 0: MERGE DUPLICATE PROBLEMS")
    print("="*80)

    merged_issues = []
    total_before = 0
    total_after = 0

    for issue in issues:
        if "error" in issue:
            merged_issues.append(issue)
            continue

        issue_id = issue.get("original_issue_id", "?")
        before = len(issue.get("problems", []))

        merged_issue = merge_duplicate_problems_in_issue(issue)
        after = len(merged_issue.get("problems", []))

        total_before += before
        total_after += after

        if before != after:
            print(f"Issue {issue_id}: {before} -> {after} problems (merged {before - after} duplicates)")
        else:
            print(f"Issue {issue_id}: {after} problems (no duplicates found)")

        merged_issues.append(merged_issue)

    print(f"\nOK Phase 0 complete: {total_before} -> {total_after} problems across all issues")
    print(f"  Reduction: {total_before - total_after} duplicate problems merged\n")

    return merged_issues


# ============================================================================
# PHASE 1: Verify Code Changes (Minimal - deprecated, kept for backward compat)
# ============================================================================

def phase1_verify_code_changes(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Phase 1: Verify validation_groups exist for fetching detailed changes.

    Modern decompose_ci_failure.py stores changes in validation_groups (single source of truth).
    This phase verifies the data structure is correct.
    """

    print("\n" + "="*80)
    print("PHASE 1: Verify Validation Groups")
    print("="*80)

    has_validation_groups = False
    issues_with_groups = 0
    issues_without_groups = 0

    for issue in issues:
        if "error" in issue:
            continue

        issue_id = issue.get("original_issue_id", "?")
        diff_context = issue.get("diff_analysis_context", {})
        validation_groups = diff_context.get("validation_groups", {})

        if validation_groups:
            total_changes = sum(len(vg.get("all_changes", [])) for vg in validation_groups.values())
            print(f"Issue {issue_id}: {len(validation_groups)} validation groups, {total_changes} total changes")
            has_validation_groups = True
            issues_with_groups += 1
        else:
            issues_without_groups += 1

    # Summary
    if has_validation_groups:
        print(f"\nOK Phase 1 complete: {issues_with_groups} issues with validation_groups")
    else:
        print(f"\nINFO  No validation_groups found - using semantic analysis fallback")
        print(f"   (Phase 2 will infer dependencies from problem descriptions)")

    print(f"OK Phase 1 complete: Verified {len(issues)} issues\n")
    return issues


# ============================================================================
# PHASE 2: Analyze Cross-Problem Dependencies (LLM)
# ============================================================================

def analyze_cross_problem_dependencies(issue: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    Use LLM to analyze dependencies between problems in same issue.
    Returns dependency graph and repair sequence.
    """

    problems = issue.get("problems", [])
    if len(problems) <= 1:
        return {
            "dependency_graph": {"nodes": [], "edges": []},
            "repair_sequence": []
        }

    # Prepare problem summaries for LLM
    problem_summaries = []
    for p in problems:
        summary = {
            "problem_id": p.get("problem_id"),
            "validation_order": p.get("validation_order"),
            "issue_type": p.get("issue_type"),
            "affected_files": p.get("affected_files", [])[:5],  # First 5 files
            "file_count": len(p.get("affected_files", [])),
            "how_fixed": p.get("how_fixed", "")[:300],  # First 300 chars
            "depends_on": p.get("depends_on", [])
        }
        problem_summaries.append(summary)

    prompt = f"""Analyze cross-problem dependencies in this CI failure fix.

You have {len(problems)} atomic problems. Some problems enable or require other problems.
Identify these dependencies and create a repair sequence.

=== PROBLEMS ===

{json.dumps(problem_summaries, indent=2)}

=== TASK ===

Analyze dependencies and answer:

1. Which problems are FOUNDATIONAL (no dependencies)?
   - Typically: dependency changes, config changes, tool installation

2. Which problems DEPEND ON others?
   - Example: Formatting fixes depend on tool being installed
   - Example: Import cleanup depends on type changes

3. What is the OPTIMAL REPAIR SEQUENCE?
   - Group problems that can be fixed in parallel
   - Order groups by dependency

=== OUTPUT FORMAT ===

{{
  "dependency_graph": {{
    "nodes": [
      {{
        "problem_id": 1,
        "type": "foundational",  // or "dependent"
        "category": "dependency_management"
      }}
    ],
    "edges": [
      {{
        "from": 1,
        "to": 2,
        "type": "enables",
        "mechanism": "Problem 1 installs taplo -> Problem 2 can run taplo validation"
      }}
    ]
  }},

  "repair_sequence": [
    {{
      "order": 1,
      "problem_ids": [1, 3],
      "rationale": "Foundational changes: install dependencies and enable tools",
      "can_parallelize": true
    }},
    {{
      "order": 2,
      "problem_ids": [2, 4],
      "rationale": "Dependent changes: fix validation errors found after tools enabled",
      "can_parallelize": true,
      "depends_on_order": 1
    }}
  ],

  "technical_dependencies": [
    {{
      "primary_problem": 1,
      "dependent_problems": [2],
      "dependency_type": "tool_installation_enablement",
      "technical_flow": [
        "Step 1: Problem 1 fixes config file",
        "Step 2: Tool gets installed",
        "Step 3: Tool validation runs",
        "Step 4: Problem 2 fixes errors found by tool"
      ]
    }}
  ]
}}

=== RULES ===

1. Use actual problem IDs from the input
2. Only create edges for REAL dependencies (not just same validation)
3. Repair sequence should minimize total steps while respecting dependencies
4. If no dependencies exist, return empty edges and single repair step
5. Output valid JSON only
"""

    try:
        result = llm.invoke(prompt)
        content = result.content if hasattr(result, 'content') else str(result)

        # Clean JSON response
        import re

        # Extract from code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Remove comments
        content = re.sub(r'//.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Remove trailing commas
        content = re.sub(r',(\s*[}\]])', r'\1', content)

        # Extract JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        return {
            "dependency_graph": {"nodes": [], "edges": []},
            "repair_sequence": []
        }

    except Exception as e:
        print(f"    Warning: Dependency analysis failed: {e}")
        return {
            "dependency_graph": {"nodes": [], "edges": []},
            "repair_sequence": []
        }


def phase2_analyze_dependencies(issues: List[Dict[str, Any]], llm: Any) -> List[Dict[str, Any]]:
    """
    Phase 2: Analyze cross-problem dependencies using LLM.
    Adds dependency_graph and repair_sequence to each issue.
    """

    print("\n" + "="*80)
    print("PHASE 2: Analyze Dependencies (LLM)")
    print("="*80)

    enhanced_issues = []

    for issue in issues:
        if "error" in issue:
            enhanced_issues.append(issue)
            continue

        issue_id = issue.get("original_issue_id", "?")
        problem_count = len(issue.get("problems", []))

        print(f"\nIssue {issue_id} ({problem_count} problems):")

        if problem_count <= 1:
            print(f"  INFO  Single problem, no dependencies to analyze")
            issue["dependency_graph"] = {"nodes": [], "edges": []}
            issue["repair_sequence"] = []
            enhanced_issues.append(issue)
            continue

        # Analyze dependencies
        dep_analysis = analyze_cross_problem_dependencies(issue, llm)

        # Add to issue
        issue["dependency_graph"] = dep_analysis.get("dependency_graph", {})
        issue["repair_sequence"] = dep_analysis.get("repair_sequence", [])
        issue["technical_dependencies"] = dep_analysis.get("technical_dependencies", [])

        edges_count = len(dep_analysis.get("dependency_graph", {}).get("edges", []))
        seq_count = len(dep_analysis.get("repair_sequence", []))

        print(f"  OK Found {edges_count} dependency edges")
        print(f"  OK Created {seq_count}-step repair sequence")

        # Update problem enables/enabled_by based on graph
        edges = dep_analysis.get("dependency_graph", {}).get("edges", [])
        problem_enables = defaultdict(list)
        problem_enabled_by = defaultdict(list)

        for edge in edges:
            from_id = edge.get("from")
            to_id = edge.get("to")
            problem_enables[from_id].append(to_id)
            problem_enabled_by[to_id].append(from_id)

        # Add to problems
        for problem in issue.get("problems", []):
            pid = problem.get("problem_id")
            problem["enables"] = problem_enables.get(pid, [])
            problem["enabled_by"] = problem_enabled_by.get(pid, [])

        enhanced_issues.append(issue)

    print(f"\nOK Phase 2 complete: Analyzed dependencies for {len(enhanced_issues)} issues")
    return enhanced_issues


# ============================================================================
# PHASE 3: Enhance L1 with Dependency Info
# ============================================================================

def phase3_enhance_l1(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Phase 3: Build enhanced L1 per-file memory with dependency information.
    """

    print("\n" + "="*80)
    print("PHASE 3: Enhance L1 (Per-File Memory)")
    print("="*80)

    l1_entries = []
    file_map = {}

    for issue in issues:
        if "error" in issue:
            continue

        repo = issue.get("repo", "unknown")
        issue_id = issue.get("original_issue_id", "?")

        # Get dependency info
        dep_graph = issue.get("dependency_graph", {})
        edges = dep_graph.get("edges", [])

        for problem in issue.get("problems", []):
            problem_id = problem.get("problem_id")
            affected_files = problem.get("affected_files", [])

            # Fetch detailed changes from validation_groups (avoids duplication in decomposed JSON)
            detailed_changes = get_detailed_changes_for_problem(issue, problem)

            # Get dependency relationships
            enabled_by_problems = problem.get("enabled_by", [])

            # Get repair order from repair_sequence
            repair_order = 1
            for seq in issue.get("repair_sequence", []):
                if problem_id in seq.get("problem_ids", []):
                    repair_order = seq.get("order", 1)
                    break

            # Create L1 entry for each file
            for file_path in affected_files[:20]:  # Limit to 20 files per problem
                key = f"{repo}::{file_path}::{problem.get('failure_type', 'Unknown')}"
                if key in file_map:
                    continue
                file_map[key] = True

                # Find changes for this file from detailed_changes
                file_changes_list = [
                    c for c in detailed_changes
                    if c.get("file") == file_path
                ]

                # Build enables info
                enables_info = []
                for edge in edges:
                    if edge.get("from") == problem_id:
                        # This problem enables others
                        to_problem = next((p for p in issue["problems"] if p["problem_id"] == edge.get("to")), None)
                        if to_problem:
                            enables_info.append({
                                "problem_id": edge.get("to"),
                                "files": to_problem.get("affected_files", [])[:5],
                                "mechanism": edge.get("mechanism", ""),
                                "type": edge.get("type", "")
                            })

                l1_entry = {
                    "file": file_path,
                    "repo": repo,
                    "issue_id": issue_id,
                    "failure_type": problem.get("failure_type", ""),
                    "problem": {
                        "what_wrong": problem.get("problem", ""),
                        "root_cause": problem.get("root_cause", "")
                    },
                    "fixes": [
                        {
                            "line": change.get("line_number") or change.get("line"),
                            "from": change.get("before_snippet") or change.get("before", ""),
                            "to": change.get("after_snippet") or change.get("after", ""),
                            "change_type": change.get("change_type", "")
                        }
                        for change in file_changes_list[:10]  # First 10 changes per file
                    ],
                    "validation_cmd": problem.get("validation_cmd", ""),
                    "repair_order": repair_order,
                    "enables": enables_info,
                    "enabled_by": enabled_by_problems
                }

                l1_entries.append(l1_entry)

    print(f"  OK Created {len(l1_entries)} L1 per-file entries")
    return l1_entries


# ============================================================================
# PHASE 4: Enhance L2 with Problem Groups
# ============================================================================

def phase4_enhance_l2(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Phase 4: Build enhanced L2 per-issue memory with problem grouping.
    """

    print("\n" + "="*80)
    print("PHASE 4: Enhance L2 (Per-Issue Memory)")
    print("="*80)

    l2_entries = []

    for issue in issues:
        if "error" in issue:
            continue

        issue_id = issue.get("original_issue_id", "?")
        problems = issue.get("problems", [])

        # Group problems by pattern
        problem_groups = []
        grouped = set()

        # Find foundational vs dependent problems
        dep_graph = issue.get("dependency_graph", {})
        foundational = []
        dependent = []

        for node in dep_graph.get("nodes", []):
            if node.get("type") == "foundational":
                foundational.append(node.get("problem_id"))
            else:
                dependent.append(node.get("problem_id"))

        # Create foundational group
        if foundational:
            foundational_probs = [p for p in problems if p.get("problem_id") in foundational]
            if foundational_probs:
                group = {
                    "group_id": "foundational_changes",
                    "pattern": "Dependency and configuration changes that enable validations",
                    "problem_ids": foundational,
                    "common_root_cause": "Missing or incorrect dependencies/config",
                    "repair_order": 1,
                    "files_affected": sum(len(p.get("affected_files", [])) for p in foundational_probs)
                }
                problem_groups.append(group)
                grouped.update(foundational)

        # Create dependent group
        if dependent:
            dependent_probs = [p for p in problems if p.get("problem_id") in dependent]
            if dependent_probs:
                group = {
                    "group_id": "dependent_fixes",
                    "pattern": "Validation fixes that depend on foundational changes",
                    "problem_ids": dependent,
                    "enabled_by": ["foundational_changes"] if foundational else [],
                    "repair_order": 2,
                    "files_affected": sum(len(p.get("affected_files", [])) for p in dependent_probs)
                }
                problem_groups.append(group)
                grouped.update(dependent)

        # Any ungrouped problems
        ungrouped = [p.get("problem_id") for p in problems if p.get("problem_id") not in grouped]
        if ungrouped:
            group = {
                "group_id": "independent_fixes",
                "pattern": "Independent fixes with no cross-dependencies",
                "problem_ids": ungrouped,
                "repair_order": 1,
                "files_affected": sum(len(p.get("affected_files", [])) for p in problems if p.get("problem_id") in ungrouped)
            }
            problem_groups.append(group)

        l2_entry = {
            "issue_id": issue_id,
            "repo": issue.get("repo", ""),
            "atomic_problems": problems,
            "problem_groups": problem_groups,
            "dependency_graph": issue.get("dependency_graph", {}),
            "repair_sequence": issue.get("repair_sequence", []),
            "technical_dependencies": issue.get("technical_dependencies", [])
        }

        l2_entries.append(l2_entry)
        print(f"  Issue {issue_id}: {len(problems)} problems, {len(problem_groups)} groups")

    print(f"  OK Created {len(l2_entries)} L2 entries")
    return l2_entries


# ============================================================================
# PHASE 5: Enhance L3 with Universal Dependency Patterns
# ============================================================================

def phase5_enhance_l3(issues: List[Dict[str, Any]], llm: Any) -> List[Dict[str, Any]]:
    """
    Phase 5: Extract universal dependency patterns across all issues.
    """

    print("\n" + "="*80)
    print("PHASE 5: Enhance L3 (Universal Patterns)")
    print("="*80)

    # Collect all technical dependencies
    all_tech_deps = []
    for issue in issues:
        if "error" in issue:
            continue
        for tech_dep in issue.get("technical_dependencies", []):
            tech_dep["issue_id"] = issue.get("original_issue_id")
            all_tech_deps.append(tech_dep)

    if not all_tech_deps:
        print("  INFO  No technical dependencies found")
        return []

    print(f"  Found {len(all_tech_deps)} technical dependencies across issues")

    # Group by dependency type
    by_type = defaultdict(list)
    for dep in all_tech_deps:
        dep_type = dep.get("dependency_type", "unknown")
        by_type[dep_type].append(dep)

    l3_patterns = []

    for dep_type, deps in by_type.items():
        if len(deps) < 2:  # Need at least 2 examples
            continue

        print(f"\n  Analyzing pattern: {dep_type} ({len(deps)} examples)")

        # Analyze with LLM
        prompt = f"""Extract universal dependency pattern from these examples.

Dependency Type: {dep_type}
Examples: {len(deps)}

{json.dumps(deps[:5], indent=2)}  # First 5 examples

Extract:
1. Common mechanism (how does primary change enable dependent changes?)
2. Repair strategy (what's the optimal fix approach?)
3. Detection rules (how to identify this pattern?)

Output JSON:
{{
  "pattern_name": "...",
  "mechanism": "...",
  "repair_strategy": {{
    "order": "...",
    "steps": [...]
  }},
  "detection": {{
    "primary_indicators": [...],
    "dependent_indicators": [...]
  }}
}}
"""

        try:
            result = llm.invoke(prompt)
            content = result.content if hasattr(result, 'content') else str(result)

            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                pattern = json.loads(json_match.group())
                pattern["pattern_id"] = dep_type.lower().replace(" ", "_")
                pattern["occurrences"] = len(deps)
                pattern["examples"] = [
                    {"issue_id": d.get("issue_id"), "primary": d.get("primary_problem")}
                    for d in deps[:3]
                ]
                l3_patterns.append(pattern)
                print(f"    OK Extracted pattern")
        except Exception as e:
            print(f"    Warning: Pattern extraction failed: {e}")

    print(f"\n  OK Created {len(l3_patterns)} L3 universal patterns")
    return l3_patterns


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Integrate dependency analysis into decomposed issues and L1/L2/L3"
    )
    parser.add_argument(
        "--decomposed",
        required=True,
        help="Input decomposed issues JSON"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output enhanced decomposed issues JSON"
    )
    parser.add_argument(
        "--output-dir",
        default="data/trs",
        help="Output directory for L1/L2/L3"
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.7-sonnet",
        help="LLM model for dependency analysis"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM phases (2 and 5)"
    )

    args = parser.parse_args()

    # Load decomposed issues
    print("="*80)
    print("DEPENDENCY INTEGRATION - 5 PHASES")
    print("="*80)

    with open(args.decomposed) as f:
        issues = json.load(f)

    print(f"Loaded {len(issues)} decomposed issues")

    # Initialize LLM
    llm = None
    if not args.skip_llm:
        print(f"Initializing LLM: {args.model}")
        llm = LitellmModel(model_name=args.model)

    # Phase 1: Verify code changes exist (added by decompose script)
    issues = phase1_verify_code_changes(issues)

    # Phase 2: Analyze dependencies (LLM)
    if llm:
        issues = phase2_analyze_dependencies(issues, llm)
    else:
        print("\nWARNING  Skipping Phase 2 (no LLM)")

    # Save enhanced decomposed issues
    with open(args.output, 'w') as f:
        json.dump(issues, f, indent=2)
    print(f"\nOK Saved enhanced decomposed issues: {args.output}")

    # Phase 3: Build enhanced L1
    l1_entries = phase3_enhance_l1(issues)
    l1_path = Path(args.output_dir) / "failure_memory_enhanced.json"
    with open(l1_path, 'w') as f:
        json.dump(l1_entries, f, indent=2)
    print(f"OK Saved enhanced L1: {l1_path}")

    # Phase 4: Build enhanced L2
    l2_entries = phase4_enhance_l2(issues)
    l2_path = Path(args.output_dir) / "repo_memory_enhanced.json"
    with open(l2_path, 'w') as f:
        json.dump(l2_entries, f, indent=2)
    print(f"OK Saved enhanced L2: {l2_path}")

    # Phase 5: Build enhanced L3 (LLM)
    if llm:
        l3_patterns = phase5_enhance_l3(issues, llm)
        l3_path = Path(args.output_dir) / "cross_memory_enhanced.json"
        with open(l3_path, 'w') as f:
            json.dump(l3_patterns, f, indent=2)
        print(f"OK Saved enhanced L3: {l3_path}")
    else:
        print("\nWARNING  Skipping Phase 5 (no LLM)")

    print("\n" + "="*80)
    print("OK ALL PHASES COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Decomposed (enhanced): {args.output}")
    print(f"  L1 (enhanced): {l1_path}")
    print(f"  L2 (enhanced): {l2_path}")
    if llm:
        print(f"  L3 (enhanced): {l3_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
