#!/usr/bin/env python3
"""
decompose_ci_failure.py - Reverse Engineer CI Failures into Atomic Problems
===========================================================================

Based on professor's direction: Given CI failure (FIRST failure only) + ground truth diff,
use LLM to reverse engineer ALL hidden problems.

Key insight: CI stops at FIRST failure, but diff fixes MULTIPLE problems.
We need to infer hidden problems from the diff.

Usage:
    # Decompose single issue
    python scripts/decompose_ci_failure.py --issue-id 410

    # Decompose all eval issues
    python scripts/decompose_ci_failure.py --batch

References:
    - O-CRD: Backward reasoning from ground truth
    - STAIR: Multi-layer hierarchical abstraction
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minisweagent.models.litellm_model import LiteLLMModel


def load_ci_workflow_context(repo: str) -> str:
    """
    Load CI workflow file to understand validation stages.
    If not available, return generic CI stages.
    """
    # Try to load from repo's .github/workflows/
    # For now, return generic stages
    return """
Typical CI Workflow Stages:
1. Install dependencies (pip install, npm install, etc.)
2. Linting (ruff, pylint, eslint, etc.)
3. Type checking (mypy, pyright, tsc, etc.)
4. Formatting (black, prettier, etc.)
5. Unit tests (pytest, jest, etc.)
6. Integration tests
7. Build (if applicable)
"""


def build_decomposition_prompt(issue: Dict) -> str:
    """
    Build prompt to REVERSE ENGINEER atomic problems from:
    - CI failure log (shows FIRST failure only)
    - Ground truth diff (fixes ALL problems)
    - Changed files
    - CI workflow context

    Key: CI stops at first failure, but diff fixes multiple problems.
    We must infer hidden problems!
    """

    repo = issue.get("repo_name", issue.get("repo", "unknown"))
    sha = issue.get("sha_fail", "")[:12]
    error_type = issue.get("error_type", [])
    if isinstance(error_type, list):
        error_type = ", ".join(error_type)

    # Get CI log (FIRST failure only!)
    logs = issue.get("logs", "")
    if isinstance(logs, list):
        ci_log = "\n".join([str(l) for l in logs[:10]])  # More context
    else:
        ci_log = str(logs)
    ci_log = ci_log[:4000]  # Increase limit

    # Get diff (fixes ALL problems)
    diff = issue.get("diff", "")[:5000]  # Increase limit

    # Get changed files
    changed_files = issue.get("changed_files", [])

    # Get CI workflow context
    ci_workflow = load_ci_workflow_context(repo)

    return f"""You are a CI failure analysis expert doing REVERSE ENGINEERING.

CRITICAL CONTEXT:
- The CI log shows ONLY the FIRST failure (CI stops at first error)
- The ground truth diff fixes ALL problems (visible + hidden)
- Your job: Infer HIDDEN problems from the diff

════════════════════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════════════════════

Repository: {repo}
Commit: {sha}
Error Type: {error_type}
Changed Files ({len(changed_files)}): {', '.join(changed_files[:15])}

--- CI FAILURE LOG (FIRST failure only - CI stopped here) ---
{ci_log}

--- GROUND TRUTH DIFF (fixes ALL problems) ---
{diff}

--- CI WORKFLOW CONTEXT ---
{ci_workflow}

════════════════════════════════════════════════════════════════════════════════
YOUR TASK: REVERSE ENGINEER ALL PROBLEMS
════════════════════════════════════════════════════════════════════════════════

The CI log shows Problem 1. But the diff changes {len(changed_files)} files!
This suggests MULTIPLE problems were fixed.

**Step 1: Identify visible problem (from CI log)**
- What failed first?
- Which files in the diff fix THIS problem?
- Which CI stage failed? (install/lint/type-check/test/build)

**Step 2: Identify HIDDEN problems (from diff analysis)**
- Look at OTHER changed files
- What problems do THOSE files fix?
- Which CI stages WOULD have failed after fixing Problem 1?

**Step 3: Infer repair sequence**
- Problem 1: visible in log, must fix first
- Problem 2: hidden, would appear after fixing p1
- Problem 3: hidden, would appear after fixing p2

**Evidence types:**
- Visible problem: Has CI log evidence
- Hidden problem: Inferred from diff, no CI log evidence

**Example reasoning:**
CI log: "ERROR: No matching distribution for fish-audio-sdk>=2024.12.5"
→ Problem 1 (VISIBLE): Dependency constraint
→ Files: pyproject.toml
→ CI stage: pip install

Diff ALSO changes: src/audio/*.py (11 files, type hints changed)
→ Problem 2 (HIDDEN): Type errors after SDK upgrade
→ Would fail at: mypy src/
→ Depends on: Problem 1 (can't check types until SDK installable)

Diff ALSO changes: tests/test_audio.py
→ Problem 3 (HIDDEN): Test expectations outdated
→ Would fail at: pytest tests/
→ Depends on: Problem 2

════════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (VALID JSON ONLY, NO MARKDOWN)
════════════════════════════════════════════════════════════════════════════════

{{
  "total_problems": <number>,
  "problems": [
    {{
      "problem_id": 1,
      "visibility": "visible_in_log",
      "problem_type": "dependency_error | type_checking | formatting | test_failure | configuration | other",
      "symptom": "What failed (from CI log)",
      "evidence_in_ci_log": "Error message from log",
      "ci_workflow_stage": "install | lint | type_check | test | build",
      "ci_command_that_failed": "pip install -e .",

      "root_cause": "WHY it failed (deeper analysis)",
      "why_occurred": "What was wrong in code/config",
      "how_fixed": "Specific changes made",
      "why_fix_works": "Why these changes solve the problem",

      "affected_files": ["file1.py"],
      "diff_evidence": "Specific lines from diff for this problem",

      "depends_on": null,
      "enables": [2],
      "verification_after_fix": "How to verify THIS fix",
      "next_failure_after_fix": "Problem 2 would appear"
    }},
    {{
      "problem_id": 2,
      "visibility": "hidden",
      "problem_type": "type_checking",
      "symptom": "Type errors (INFERRED from diff, not in CI log)",
      "evidence_in_ci_log": null,
      "ci_workflow_stage": "type_check",
      "ci_command_that_would_fail": "mypy src/",

      "root_cause": "SDK API changed",
      "why_occurred": "Type hints outdated",
      "how_fixed": "Updated type hints in 11 files",
      "why_fix_works": "Matches new SDK types",

      "affected_files": ["src/f1.py", "src/f2.py"],
      "diff_evidence": "Type annotations changed",

      "depends_on": 1,
      "dependency_reason": "Can't check types until SDK installable",
      "enables": [3],
      "verification_after_fix": "mypy src/",
      "next_failure_after_fix": "Problem 3 might appear"
    }}
  ],

  "repair_trajectory_inference": [
    {{
      "step": 1,
      "problem_fixed": 1,
      "ci_stage": "install",
      "validation": "pip install --dry-run",
      "result": "success",
      "next_ci_stage": "type_check",
      "expected_next_failure": "Problem 2 (type errors)"
    }},
    {{
      "step": 2,
      "problem_fixed": 2,
      "ci_stage": "type_check",
      "validation": "mypy src/",
      "result": "success",
      "next_ci_stage": "test",
      "expected_next_failure": "Problem 3 or null"
    }}
  ],

  "overall_reasoning": "High-level: Why did this CI failure have multiple problems? How do they relate?"
}}

════════════════════════════════════════════════════════════════════════════════
CRITICAL RULES
════════════════════════════════════════════════════════════════════════════════

1. Problem 1 MUST have visibility="visible_in_log" and evidence_in_ci_log
2. Problems 2, 3, ... MUST have visibility="hidden" and evidence_in_ci_log=null
3. Infer hidden problems from:
   - Changed files not related to visible problem
   - Typical CI workflow order
   - File types (config vs code vs tests)
4. If 10 files have same type of change (type hints), that's ONE problem, not 10
5. Dependencies: hidden problems depend on fixing visible problems first
6. Be SPECIFIC: include line numbers, exact changes, file paths
""".strip()


def decompose_issue(issue: Dict, llm) -> Dict:
    """
    Reverse engineer atomic problems from CI failure + ground truth diff.

    Returns decomposed problems with visibility markers:
    - visible_in_log: Problem 1 (from CI log)
    - hidden: Problems 2+ (inferred from diff)
    """

    issue_id = issue.get('id', '?')
    print(f"\n{'='*80}")
    print(f"Reverse Engineering Issue {issue_id}")
    print(f"  Repo: {issue.get('repo_name', issue.get('repo', '?'))}")
    print(f"  Changed files: {len(issue.get('changed_files', []))}")
    print(f"{'='*80}")

    prompt = build_decomposition_prompt(issue)

    try:
        print(f"  Calling LLM to reverse engineer problems...")
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Remove markdown fences if present
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.startswith("```"):
            content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        content = content.strip()

        result = json.loads(content)

        # Add metadata
        result["original_issue_id"] = issue.get("id")
        result["sha_fail"] = issue.get("sha_fail")
        result["repo"] = issue.get("repo_name", issue.get("repo"))
        result["original_error_type"] = issue.get("error_type")

        # Summary
        total = result.get('total_problems', 0)
        visible = sum(1 for p in result.get('problems', []) if p.get('visibility') == 'visible_in_log')
        hidden = total - visible

        print(f"  ✓ Found {total} problems: {visible} visible + {hidden} hidden")

        # Show problem types
        for p in result.get('problems', []):
            vis_marker = "👁️ " if p.get('visibility') == 'visible_in_log' else "🔍"
            print(f"    {vis_marker} P{p.get('problem_id')}: {p.get('problem_type')} - {p.get('symptom', '')[:60]}")

        return result

    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parsing failed: {e}")
        print(f"  Raw content preview: {content[:500]}")
        return {
            "error": f"JSON parse error: {e}",
            "original_issue_id": issue.get("id"),
            "sha_fail": issue.get("sha_fail"),
            "raw_content": content[:1000]
        }
    except Exception as e:
        print(f"  ✗ Failed to decompose: {e}")
        return {
            "error": str(e),
            "original_issue_id": issue.get("id"),
            "sha_fail": issue.get("sha_fail"),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Reverse engineer CI failures into atomic problems (visible + hidden)"
    )
    parser.add_argument("--issue-id", help="Single issue ID to decompose")
    parser.add_argument("--batch", action="store_true", help="Decompose all eval issues")
    parser.add_argument("--eval-issues", default="data/trs/eval_issues.json", help="Path to eval issues")
    parser.add_argument("--output", default="data/trs/decomposed_issues.json", help="Output file")
    parser.add_argument("--model", default="minimax/minimax-m2.5", help="LLM model")
    parser.add_argument("--limit", type=int, help="Limit number of issues to process")
    args = parser.parse_args()

    # Load eval issues
    eval_path = Path(args.eval_issues)
    if not eval_path.exists():
        print(f"✗ Eval issues not found: {eval_path}")
        return 1

    with open(eval_path) as f:
        issues = json.load(f)

    print(f"Loaded {len(issues)} issues from {eval_path}")

    # Filter if specific issue requested
    if args.issue_id:
        issues = [i for i in issues if str(i.get("id")) == args.issue_id]
        if not issues:
            print(f"✗ Issue {args.issue_id} not found")
            return 1

    # Limit if requested
    if args.limit:
        issues = issues[:args.limit]
        print(f"Limited to first {args.limit} issues")

    # Initialize LLM
    print(f"\n{'='*80}")
    print(f"Initializing LLM: {args.model}")
    print(f"{'='*80}")
    llm = LiteLLMModel(model_name=args.model)

    # Decompose issues
    results = []
    errors = []

    for i, issue in enumerate(issues, 1):
        print(f"\nProgress: {i}/{len(issues)}")
        result = decompose_issue(issue, llm)

        if "error" in result:
            errors.append(result)

        results.append(result)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total issues processed: {len(results)}")
    print(f"Successful: {len(results) - len(errors)}")
    print(f"Errors: {len(errors)}")

    # Count problems
    successful = [r for r in results if "total_problems" in r]
    total_problems = sum(r.get("total_problems", 0) for r in successful)
    visible_problems = sum(
        sum(1 for p in r.get("problems", []) if p.get("visibility") == "visible_in_log")
        for r in successful
    )
    hidden_problems = total_problems - visible_problems

    print(f"\nAtomic problems identified:")
    print(f"  Total: {total_problems}")
    print(f"  Visible (in CI log): {visible_problems}")
    print(f"  Hidden (inferred): {hidden_problems}")

    if successful:
        avg_problems = total_problems / len(successful)
        print(f"  Average per issue: {avg_problems:.1f}")

    # Problem type distribution
    problem_types = {}
    for r in successful:
        for p in r.get("problems", []):
            ptype = p.get("problem_type", "unknown")
            problem_types[ptype] = problem_types.get(ptype, 0) + 1

    if problem_types:
        print(f"\nProblem type distribution:")
        for ptype, count in sorted(problem_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ptype}: {count}")

    print(f"\nOutput saved to: {output_path}")

    if errors:
        print(f"\n⚠️  {len(errors)} issues had errors")
        print(f"Issue IDs with errors: {[e.get('original_issue_id') for e in errors[:5]]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
