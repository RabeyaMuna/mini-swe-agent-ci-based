#!/usr/bin/env python3
"""
track_problem_issues.py - Track which issues had problems during decomposition

This script:
1. Scans decomposed_issues.json for issues with errors or fallback problems
2. Checks debug prompt files for malformed responses
3. Generates a report of problematic issues that need reprocessing
4. Creates a rerun list for fixing

Usage:
    python scripts/track_problem_issues.py --output-dir data/trs
    python scripts/track_problem_issues.py --output-dir data/trs --fix-list problematic_issues.txt
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

def load_decomposed_issues(output_dir: Path):
    """Load decomposed issues from JSON file."""
    decomposed_path = output_dir / "decomposed_issues.json"

    if not decomposed_path.exists():
        print(f"ERROR: {decomposed_path} not found")
        return []

    with open(decomposed_path) as f:
        issues = json.load(f)

    return issues if isinstance(issues, list) else [issues]


def analyze_issue_quality(issue):
    """Analyze if an issue has quality problems."""
    issues_found = []

    issue_id = issue.get("original_issue_id", issue.get("issue_id", "unknown"))

    # Check 1: Has error field
    if "error" in issue:
        issues_found.append({
            "issue_id": issue_id,
            "problem_type": "DECOMPOSITION_ERROR",
            "details": issue.get("error_message", "Unknown error"),
            "severity": "CRITICAL"
        })
        return issues_found

    # Check 2: No problems generated
    problems = issue.get("problems", [])
    if not problems:
        issues_found.append({
            "issue_id": issue_id,
            "problem_type": "NO_PROBLEMS",
            "details": "Decomposition returned 0 atomic problems",
            "severity": "HIGH"
        })
        return issues_found

    # Check 3: Generic fallback problems (missing key fields)
    fallback_count = 0
    for prob in problems:
        is_generic = (
            prob.get("problem", "").startswith("Validation failed:") or
            prob.get("root_cause", "").startswith("Changes needed for") or
            prob.get("how_fixed", "").startswith("Modified") and "files" in prob.get("how_fixed", "")
        )
        if is_generic:
            fallback_count += 1

    if fallback_count > 0:
        issues_found.append({
            "issue_id": issue_id,
            "problem_type": "FALLBACK_PROBLEMS",
            "details": f"{fallback_count}/{len(problems)} problems are generic fallbacks",
            "severity": "MEDIUM"
        })

    # Check 4: Very few problems for large changes
    total_files = issue.get("total_changed_files", 0)
    if total_files > 10 and len(problems) < 2:
        issues_found.append({
            "issue_id": issue_id,
            "problem_type": "UNDERFITTED",
            "details": f"Only {len(problems)} problems for {total_files} changed files",
            "severity": "LOW"
        })

    # Check 5: Missing key fields in problems
    missing_fields_count = 0
    for prob in problems:
        required_fields = ["problem", "root_cause", "how_fixed", "why_fix_works", "affected_files"]
        missing = [f for f in required_fields if not prob.get(f)]
        if missing:
            missing_fields_count += 1

    if missing_fields_count > 0:
        issues_found.append({
            "issue_id": issue_id,
            "problem_type": "INCOMPLETE_PROBLEMS",
            "details": f"{missing_fields_count}/{len(problems)} problems missing required fields",
            "severity": "MEDIUM"
        })

    return issues_found


def find_debug_prompts(output_dir: Path):
    """Find all debug prompt files (indicates LLM failures)."""
    debug_dir = output_dir
    debug_files = list(debug_dir.glob("debug_prompt_*.txt"))

    debug_info = []
    for debug_file in debug_files:
        # Extract validation order and chunk from filename
        # Format: debug_prompt_val{validation_order}_{chunk_label}.txt
        filename = debug_file.stem
        parts = filename.replace("debug_prompt_val", "").split("_")

        debug_info.append({
            "file": str(debug_file),
            "validation": parts[0] if parts else "unknown",
            "chunk": "_".join(parts[1:]) if len(parts) > 1 else "unknown"
        })

    return debug_info


def main():
    parser = argparse.ArgumentParser(description="Track problematic issues during decomposition")
    parser.add_argument("--output-dir", default="data/trs", help="Output directory")
    parser.add_argument("--fix-list", help="Save list of issue IDs to reprocess")
    parser.add_argument("--report", default="problem_report.json", help="Save detailed report")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("="*80)
    print("Issue Quality Analysis")
    print("="*80)
    print(f"Scanning: {output_dir}")
    print()

    # Load decomposed issues
    issues = load_decomposed_issues(output_dir)
    print(f"Loaded {len(issues)} decomposed issues")
    print()

    # Analyze each issue
    all_problems = []
    by_severity = defaultdict(list)
    by_type = defaultdict(list)

    for issue in issues:
        issue_problems = analyze_issue_quality(issue)
        all_problems.extend(issue_problems)

        for prob in issue_problems:
            by_severity[prob["severity"]].append(prob)
            by_type[prob["problem_type"]].append(prob)

    # Find debug prompts
    debug_prompts = find_debug_prompts(output_dir)

    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total issues analyzed: {len(issues)}")
    print(f"Issues with problems: {len(set(p['issue_id'] for p in all_problems))}")
    print(f"Debug prompts found: {len(debug_prompts)}")
    print()

    # By severity
    print("By Severity:")
    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = len(by_severity[severity])
        if count > 0:
            print(f"  {severity}: {count} issues")
    print()

    # By type
    print("By Problem Type:")
    for ptype, probs in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {ptype}: {len(probs)} issues")
    print()

    # List critical/high issues
    critical_and_high = [p for p in all_problems if p["severity"] in ["CRITICAL", "HIGH"]]
    if critical_and_high:
        print("="*80)
        print("CRITICAL & HIGH PRIORITY ISSUES (need reprocessing)")
        print("="*80)
        for prob in critical_and_high:
            print(f"  Issue {prob['issue_id']}: {prob['problem_type']}")
            print(f"    {prob['details']}")
        print()

    # Debug prompts detail
    if debug_prompts:
        print("="*80)
        print("DEBUG PROMPTS (LLM failures)")
        print("="*80)
        for debug in debug_prompts[:10]:
            print(f"  Validation {debug['validation']}, Chunk {debug['chunk']}")
            print(f"    File: {debug['file']}")
        if len(debug_prompts) > 10:
            print(f"  ... and {len(debug_prompts) - 10} more")
        print()

    # Save report
    report = {
        "total_issues": len(issues),
        "problematic_issues": len(set(p['issue_id'] for p in all_problems)),
        "by_severity": {
            severity: [p for p in all_problems if p["severity"] == severity]
            for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        },
        "by_type": {
            ptype: probs
            for ptype, probs in by_type.items()
        },
        "debug_prompts": debug_prompts
    }

    report_path = output_dir / args.report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved detailed report to: {report_path}")
    print()

    # Save fix list
    if args.fix_list:
        fix_list_path = output_dir / args.fix_list
        critical_high_ids = sorted(set(p['issue_id'] for p in critical_and_high))

        with open(fix_list_path, "w") as f:
            for issue_id in critical_high_ids:
                f.write(f"{issue_id}\n")

        print(f"Saved fix list to: {fix_list_path}")
        print(f"  Contains {len(critical_high_ids)} issue IDs that need reprocessing")
        print()
        print("To reprocess these issues:")
        print(f"  cat {fix_list_path} | while read issue_id; do")
        print(f"    python scripts/decompose_ci_failure.py --issue-id $issue_id --output-dir {output_dir}")
        print(f"  done")
        print()

    # Final recommendation
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if len(critical_and_high) > 0:
        print(f"⚠ {len(set(p['issue_id'] for p in critical_and_high))} issues need reprocessing")
        print(f"  Run: python scripts/track_problem_issues.py --fix-list problematic_issues.txt")
        print(f"  Then reprocess using the command above")
    else:
        print("✓ All issues processed successfully!")

    if len(debug_prompts) > 5:
        print(f"⚠ {len(debug_prompts)} debug prompts saved (LLM had trouble)")
        print(f"  Consider:")
        print(f"    - Using a different model (Claude/GPT-4 instead of MiniMax)")
        print(f"    - Reducing chunk size (max_files_per_chunk)")
        print(f"    - Simplifying the prompt")

    print("="*80)


if __name__ == "__main__":
    main()
