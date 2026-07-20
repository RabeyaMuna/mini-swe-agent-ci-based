#!/usr/bin/env python3
"""
failure_type_analysis.py - Analyze Which Failure Types Benefit from Memory
==========================================================================

Task #3: Quick Failure-Type Analysis (TODO #7)
Analyzes which failure types are solved more often with memory vs baseline.

Reads results from:
- results/miniswe-agent/minimax-m2.5/baseline/preds.json
- results/miniswe-agent/minimax-m2.5/L1_L2_L3/preds.json
- data/trs/eval_set.jsonl (for failure types)

Output:
- statistics/failure_type_analysis.json - Raw analysis
- statistics/failure_type_analysis.md - Formatted report
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_predictions(preds_file: Path) -> Dict:
    """Load predictions from preds.json."""
    with open(preds_file) as f:
        preds = json.load(f)
    return preds


def load_eval_set() -> Dict[str, Dict]:
    """Load eval set and create mapping of id -> issue data."""
    eval_path = PROJECT_ROOT / "data" / "trs" / "eval_set.jsonl"
    issues = {}

    with open(eval_path) as f:
        for line in f:
            if line.strip():
                issue = json.loads(line)
                issue_id = str(issue.get("id", ""))
                issues[issue_id] = issue

    return issues


def analyze_failure_types():
    """Analyze which failure types benefit from memory."""

    print("=" * 70)
    print("Failure Type Analysis: Baseline vs Memory")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    eval_issues = load_eval_set()
    print(f"Loaded {len(eval_issues)} eval issues")

    baseline_path = PROJECT_ROOT / "results" / "miniswe-agent" / "minimax-m2.5" / "baseline" / "preds.json"
    memory_path = PROJECT_ROOT / "results" / "miniswe-agent" / "minimax-m2.5" / "L1_L2_L3" / "preds.json"

    if not baseline_path.exists():
        print(f"Error: Baseline predictions not found at {baseline_path}")
        return None

    if not memory_path.exists():
        print(f"Error: Memory predictions not found at {memory_path}")
        return None

    baseline_preds = load_predictions(baseline_path)
    memory_preds = load_predictions(memory_path)

    print(f"Loaded {len(baseline_preds)} baseline predictions")
    print(f"Loaded {len(memory_preds)} memory predictions")

    # Analyze by failure type
    failure_type_stats = defaultdict(lambda: {
        "total": 0,
        "baseline_resolved": 0,
        "memory_resolved": 0,
        "baseline_only": 0,  # Resolved by baseline but not memory
        "memory_only": 0,    # Resolved by memory but not baseline
        "both_resolved": 0,
        "neither_resolved": 0,
        "issues": [],
    })

    all_failure_types = set()

    # Analyze each issue
    for issue_id, issue in eval_issues.items():
        # Get failure types
        failure_types = issue.get("overall_error_types", [])
        if isinstance(failure_types, str):
            failure_types = [failure_types]

        if not failure_types:
            failure_types = ["Unknown"]

        all_failure_types.update(failure_types)

        # Check if resolved
        baseline_resolved = False
        memory_resolved = False

        if issue_id in baseline_preds:
            baseline_resolved = baseline_preds[issue_id].get("resolved", False)

        if issue_id in memory_preds:
            memory_resolved = memory_preds[issue_id].get("resolved", False)

        # Update stats for each failure type
        for ftype in failure_types:
            failure_type_stats[ftype]["total"] += 1

            if baseline_resolved:
                failure_type_stats[ftype]["baseline_resolved"] += 1
            if memory_resolved:
                failure_type_stats[ftype]["memory_resolved"] += 1

            if baseline_resolved and memory_resolved:
                failure_type_stats[ftype]["both_resolved"] += 1
            elif baseline_resolved and not memory_resolved:
                failure_type_stats[ftype]["baseline_only"] += 1
            elif memory_resolved and not baseline_resolved:
                failure_type_stats[ftype]["memory_only"] += 1
            else:
                failure_type_stats[ftype]["neither_resolved"] += 1

            failure_type_stats[ftype]["issues"].append({
                "id": issue_id,
                "baseline_resolved": baseline_resolved,
                "memory_resolved": memory_resolved,
            })

    # Compute improvement metrics
    results = {
        "summary": {
            "total_failure_types": len(failure_type_stats),
            "total_baseline_resolved": sum(1 for _, pred in baseline_preds.items() if pred.get("resolved", False)),
            "total_memory_resolved": sum(1 for _, pred in memory_preds.items() if pred.get("resolved", False)),
        },
        "by_failure_type": {},
    }

    for ftype in sorted(all_failure_types):
        stats = failure_type_stats[ftype]

        baseline_rate = (stats["baseline_resolved"] / stats["total"] * 100) if stats["total"] > 0 else 0
        memory_rate = (stats["memory_resolved"] / stats["total"] * 100) if stats["total"] > 0 else 0
        improvement = memory_rate - baseline_rate

        results["by_failure_type"][ftype] = {
            "total_issues": stats["total"],
            "baseline": {
                "resolved": stats["baseline_resolved"],
                "rate": baseline_rate,
            },
            "memory": {
                "resolved": stats["memory_resolved"],
                "rate": memory_rate,
            },
            "improvement": {
                "absolute": improvement,
                "memory_only": stats["memory_only"],
                "baseline_only": stats["baseline_only"],
                "both": stats["both_resolved"],
                "neither": stats["neither_resolved"],
            },
        }

    return results


def format_analysis_report(results: Dict) -> str:
    """Format analysis results as markdown report."""

    md = "# Failure Type Analysis: Baseline vs Memory (L1+L2+L3)\n\n"

    # Summary
    summary = results["summary"]
    md += "## Overall Summary\n\n"
    md += f"- **Total Failure Types**: {summary['total_failure_types']}\n"
    md += f"- **Baseline Total Resolved**: {summary['total_baseline_resolved']}\n"
    md += f"- **Memory Total Resolved**: {summary['total_memory_resolved']}\n"
    md += f"- **Net Improvement**: +{summary['total_memory_resolved'] - summary['total_baseline_resolved']} issues\n\n"

    # By failure type - sorted by improvement
    md += "## Results by Failure Type\n\n"
    md += "| Failure Type | Total | Baseline | Memory | Improvement | Memory Only | Baseline Only |\n"
    md += "|--------------|-------|----------|--------|-------------|-------------|---------------|\n"

    # Sort by improvement (descending)
    by_type = results["by_failure_type"]
    sorted_types = sorted(
        by_type.items(),
        key=lambda x: x[1]["improvement"]["absolute"],
        reverse=True
    )

    for ftype, stats in sorted_types:
        total = stats["total_issues"]
        baseline_resolved = stats["baseline"]["resolved"]
        baseline_rate = stats["baseline"]["rate"]
        memory_resolved = stats["memory"]["resolved"]
        memory_rate = stats["memory"]["rate"]
        improvement = stats["improvement"]["absolute"]
        memory_only = stats["improvement"]["memory_only"]
        baseline_only = stats["improvement"]["baseline_only"]

        md += f"| {ftype} | {total} | {baseline_resolved} ({baseline_rate:.1f}%) | "
        md += f"{memory_resolved} ({memory_rate:.1f}%) | "
        md += f"**{improvement:+.1f}%** | {memory_only} | {baseline_only} |\n"

    # Key insights
    md += "\n## Key Insights\n\n"

    # Find most improved types
    most_improved = [
        (ftype, stats)
        for ftype, stats in sorted_types
        if stats["improvement"]["absolute"] > 0
    ][:3]

    if most_improved:
        md += "### Failure Types That Benefit Most from Memory:\n\n"
        for ftype, stats in most_improved:
            md += f"- **{ftype}**: +{stats['improvement']['absolute']:.1f}% "
            md += f"({stats['improvement']['memory_only']} issues solved only by memory)\n"

    # Find types where memory hurts
    memory_hurts = [
        (ftype, stats)
        for ftype, stats in sorted_types
        if stats["improvement"]["absolute"] < 0
    ][:3]

    if memory_hurts:
        md += "\n### Failure Types Where Memory Hurts:\n\n"
        for ftype, stats in memory_hurts:
            md += f"- **{ftype}**: {stats['improvement']['absolute']:.1f}% "
            md += f"({stats['improvement']['baseline_only']} issues solved only by baseline)\n"

    # Find hardest types
    hardest = sorted(
        by_type.items(),
        key=lambda x: x[1]["memory"]["rate"]
    )[:3]

    md += "\n### Hardest Failure Types (Low Memory Success Rate):\n\n"
    for ftype, stats in hardest:
        md += f"- **{ftype}**: {stats['memory']['rate']:.1f}% success rate "
        md += f"({stats['memory']['resolved']}/{stats['total_issues']} resolved)\n"

    return md


def main():
    """Main entry point."""
    # Analyze
    results = analyze_failure_types()

    if results is None:
        print("\nFailed to complete analysis.")
        return

    # Save results
    output_dir = PROJECT_ROOT / "statistics"
    output_dir.mkdir(exist_ok=True)

    # Save JSON
    json_file = output_dir / "failure_type_analysis.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved analysis to {json_file}")

    # Generate report
    report = format_analysis_report(results)
    report_file = output_dir / "failure_type_analysis.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"✓ Saved report to {report_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Analysis Summary:")
    print("=" * 70)
    print(f"Total Failure Types: {results['summary']['total_failure_types']}")
    print(f"Baseline Resolved: {results['summary']['total_baseline_resolved']}")
    print(f"Memory Resolved: {results['summary']['total_memory_resolved']}")
    print(f"Net Improvement: +{results['summary']['total_memory_resolved'] - results['summary']['total_baseline_resolved']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
