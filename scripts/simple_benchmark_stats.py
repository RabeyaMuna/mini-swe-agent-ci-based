#!/usr/bin/env python3
"""
Simple Benchmark Statistics

Analyzes core benchmark characteristics (no CI categories):
- Number of commits per pull request
- Number of modified files per pull request
- Number of changed lines of code
- Number of distinct problems per pull request

Usage:
    python scripts/simple_benchmark_stats.py --input data/decomposed_issues.json
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load_decomposed_issues(path: Path) -> list[dict]:
    """Load decomposed issues from JSON or JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text().strip()
    if not text:
        return []

    # Try JSON array first
    if text.startswith("["):
        return json.loads(text)

    # Try JSONL
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def analyze_commits_per_pr(issues: list[dict]) -> dict:
    """Analyze number of commits per pull request."""
    commits_per_pr = []

    for issue in issues:
        # Try repair_trajectory first
        trajectory = issue.get("repair_trajectory", [])
        if trajectory:
            commits_per_pr.append(len(trajectory))
            continue

        # Try git_patches
        patches = issue.get("git_patches", [])
        if patches:
            commits_per_pr.append(len(patches))
            continue

        # Try problems (each problem might have a commit)
        problems = (
            issue.get("problems")
            or issue.get("problem_sequence")
            or issue.get("optimized_problems")
            or []
        )
        if problems:
            # Assume at least 1 commit if we have problems
            commits_per_pr.append(max(1, len(problems)))
            continue

        # Default: 1 commit
        commits_per_pr.append(1)

    return {
        "mean": statistics.mean(commits_per_pr) if commits_per_pr else 0,
        "median": statistics.median(commits_per_pr) if commits_per_pr else 0,
        "std": statistics.stdev(commits_per_pr) if len(commits_per_pr) > 1 else 0,
        "min": min(commits_per_pr) if commits_per_pr else 0,
        "max": max(commits_per_pr) if commits_per_pr else 0,
        "distribution": dict(Counter(commits_per_pr)),
        "data": commits_per_pr,
    }


def analyze_files_per_pr(issues: list[dict]) -> dict:
    """Analyze number of modified files per pull request."""
    files_per_pr = []

    for issue in issues:
        # Collect all unique files from all problems
        all_files = set()

        problems = (
            issue.get("problems")
            or issue.get("problem_sequence")
            or issue.get("optimized_problems")
            or []
        )

        for problem in problems:
            if not isinstance(problem, dict):
                continue

            files = problem.get("affected_files", [])
            if isinstance(files, list):
                all_files.update(str(f) for f in files if f)

        if all_files:
            files_per_pr.append(len(all_files))
        else:
            # Fallback: use total_changed_files if available
            num_files = issue.get("total_changed_files", 1)
            files_per_pr.append(num_files)

    return {
        "mean": statistics.mean(files_per_pr) if files_per_pr else 0,
        "median": statistics.median(files_per_pr) if files_per_pr else 0,
        "std": statistics.stdev(files_per_pr) if len(files_per_pr) > 1 else 0,
        "min": min(files_per_pr) if files_per_pr else 0,
        "max": max(files_per_pr) if files_per_pr else 0,
        "distribution": dict(Counter(files_per_pr)),
        "data": files_per_pr,
    }


def analyze_lines_per_pr(issues: list[dict]) -> dict:
    """Analyze lines of code changed per pull request."""
    lines_per_pr = []

    for issue in issues:
        total_lines = 0

        # Try to count from git patches
        patches = issue.get("git_patches", [])
        for patch in patches:
            if isinstance(patch, str):
                # Count additions and deletions
                for line in patch.split("\n"):
                    if line.startswith("+") and not line.startswith("+++"):
                        total_lines += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        total_lines += 1

        # If no patches, estimate from number of files
        if total_lines == 0:
            num_files = issue.get("total_changed_files", 1)
            # Conservative estimate: 10 lines per file
            total_lines = num_files * 10

        lines_per_pr.append(total_lines)

    return {
        "mean": statistics.mean(lines_per_pr) if lines_per_pr else 0,
        "median": statistics.median(lines_per_pr) if lines_per_pr else 0,
        "std": statistics.stdev(lines_per_pr) if len(lines_per_pr) > 1 else 0,
        "min": min(lines_per_pr) if lines_per_pr else 0,
        "max": max(lines_per_pr) if lines_per_pr else 0,
        "data": lines_per_pr,
    }


def analyze_problems_per_pr(issues: list[dict]) -> dict:
    """Analyze number of distinct problems per pull request."""
    problems_per_pr = []

    for issue in issues:
        problems = (
            issue.get("problems")
            or issue.get("problem_sequence")
            or issue.get("optimized_problems")
            or []
        )

        num_problems = len([p for p in problems if isinstance(p, dict)])

        if num_problems == 0:
            # Fallback: use total_problems field
            num_problems = issue.get("total_problems", 1)

        problems_per_pr.append(num_problems)

    return {
        "mean": statistics.mean(problems_per_pr) if problems_per_pr else 0,
        "median": statistics.median(problems_per_pr) if problems_per_pr else 0,
        "std": statistics.stdev(problems_per_pr) if len(problems_per_pr) > 1 else 0,
        "min": min(problems_per_pr) if problems_per_pr else 0,
        "max": max(problems_per_pr) if problems_per_pr else 0,
        "distribution": dict(Counter(problems_per_pr)),
        "single_problem": sum(1 for p in problems_per_pr if p == 1),
        "multi_problem": sum(1 for p in problems_per_pr if p > 1),
        "data": problems_per_pr,
    }


def print_summary(stats: dict, total_issues: int):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK STATISTICS")
    print("=" * 70)

    print(f"\n📊 Total Issues: {total_issues}")

    print("\n🔧 COMMITS PER PULL REQUEST")
    print(f"  Mean:   {stats['commits']['mean']:.2f}")
    print(f"  Median: {stats['commits']['median']:.1f}")
    print(f"  Std:    {stats['commits']['std']:.2f}")
    print(f"  Range:  {stats['commits']['min']} - {stats['commits']['max']}")

    print("\n📁 FILES PER PULL REQUEST")
    print(f"  Mean:   {stats['files']['mean']:.2f}")
    print(f"  Median: {stats['files']['median']:.1f}")
    print(f"  Std:    {stats['files']['std']:.2f}")
    print(f"  Range:  {stats['files']['min']} - {stats['files']['max']}")

    print("\n📝 LINES CHANGED PER PULL REQUEST")
    print(f"  Mean:   {stats['lines']['mean']:.1f}")
    print(f"  Median: {stats['lines']['median']:.1f}")
    print(f"  Std:    {stats['lines']['std']:.1f}")
    print(f"  Range:  {stats['lines']['min']} - {stats['lines']['max']}")

    print("\n🧩 PROBLEMS PER PULL REQUEST")
    print(f"  Mean:   {stats['problems']['mean']:.2f}")
    print(f"  Median: {stats['problems']['median']:.1f}")
    print(f"  Std:    {stats['problems']['std']:.2f}")
    print(f"  Range:  {stats['problems']['min']} - {stats['problems']['max']}")
    print(
        f"  Single-problem PRs: {stats['problems']['single_problem']} "
        f"({100 * stats['problems']['single_problem'] / max(total_issues, 1):.1f}%)"
    )
    print(
        f"  Multi-problem PRs:  {stats['problems']['multi_problem']} "
        f"({100 * stats['problems']['multi_problem'] / max(total_issues, 1):.1f}%)"
    )

    print("\n" + "=" * 70)


def generate_markdown_table(stats: dict, total_issues: int) -> str:
    """Generate markdown table for slides/README."""
    lines = [
        "| Metric | Mean | Median | Std | Min | Max |",
        "|--------|------|--------|-----|-----|-----|",
        f"| Commits per PR | {stats['commits']['mean']:.2f} | {stats['commits']['median']:.1f} | {stats['commits']['std']:.2f} | {stats['commits']['min']} | {stats['commits']['max']} |",
        f"| Files per PR | {stats['files']['mean']:.2f} | {stats['files']['median']:.1f} | {stats['files']['std']:.2f} | {stats['files']['min']} | {stats['files']['max']} |",
        f"| Lines changed | {stats['lines']['mean']:.1f} | {stats['lines']['median']:.1f} | {stats['lines']['std']:.1f} | {stats['lines']['min']} | {stats['lines']['max']} |",
        f"| Problems per PR | {stats['problems']['mean']:.2f} | {stats['problems']['median']:.1f} | {stats['problems']['std']:.2f} | {stats['problems']['min']} | {stats['problems']['max']} |",
        "",
        f"**Multi-problem PRs:** {stats['problems']['multi_problem']}/{total_issues} ({100 * stats['problems']['multi_problem'] / max(total_issues, 1):.1f}%)",
    ]
    return "\n".join(lines)


def generate_latex_table(stats: dict, total_issues: int) -> str:
    """Generate LaTeX table for papers."""
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Benchmark Statistics}",
        "\\label{tab:benchmark-stats}",
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "\\textbf{Metric} & \\textbf{Mean} & \\textbf{Median} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} \\\\",
        "\\hline",
        f"Commits per PR & {stats['commits']['mean']:.2f} & {stats['commits']['median']:.1f} & {stats['commits']['std']:.2f} & {stats['commits']['min']} & {stats['commits']['max']} \\\\",
        f"Files per PR & {stats['files']['mean']:.2f} & {stats['files']['median']:.1f} & {stats['files']['std']:.2f} & {stats['files']['min']} & {stats['files']['max']} \\\\",
        f"Lines changed & {stats['lines']['mean']:.1f} & {stats['lines']['median']:.1f} & {stats['lines']['std']:.1f} & {stats['lines']['min']} & {stats['lines']['max']} \\\\",
        f"Problems per PR & {stats['problems']['mean']:.2f} & {stats['problems']['median']:.1f} & {stats['problems']['std']:.2f} & {stats['problems']['min']} & {stats['problems']['max']} \\\\",
        "\\hline",
        f"\\multicolumn{{6}}{{l}}{{Multi-problem PRs: {stats['problems']['multi_problem']}/{total_issues} ({100 * stats['problems']['multi_problem'] / max(total_issues, 1):.1f}\\%)}} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def save_distributions(stats: dict, output_dir: Path):
    """Save distribution data for plotting."""
    distributions = {
        "commits_distribution": stats["commits"]["distribution"],
        "files_distribution": stats["files"]["distribution"],
        "problems_distribution": stats["problems"]["distribution"],
    }

    with open(output_dir / "distributions.json", "w") as f:
        json.dump(distributions, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compute simple benchmark statistics")
    parser.add_argument(
        "--input",
        default="data/decomposed_issues.json",
        help="Path to decomposed_issues.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analytics",
        help="Output directory for reports",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    issues = load_decomposed_issues(Path(args.input))
    print(f"Loaded {len(issues)} issues")

    # Analyze
    print("\nAnalyzing...")
    stats = {
        "commits": analyze_commits_per_pr(issues),
        "files": analyze_files_per_pr(issues),
        "lines": analyze_lines_per_pr(issues),
        "problems": analyze_problems_per_pr(issues),
    }

    # Print summary
    print_summary(stats, len(issues))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full report
    report = {
        "total_issues": len(issues),
        "commits_per_pr": stats["commits"],
        "files_per_pr": stats["files"],
        "lines_changed": stats["lines"],
        "problems_per_pr": stats["problems"],
    }

    report_file = output_dir / "benchmark_stats.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Full report: {report_file}")

    # Save markdown table
    md_file = output_dir / "benchmark_stats.md"
    with open(md_file, "w") as f:
        f.write("# Benchmark Statistics\n\n")
        f.write(generate_markdown_table(stats, len(issues)))
    print(f"✅ Markdown table: {md_file}")

    # Save LaTeX table
    tex_file = output_dir / "benchmark_stats.tex"
    with open(tex_file, "w") as f:
        f.write(generate_latex_table(stats, len(issues)))
    print(f"✅ LaTeX table: {tex_file}")

    # Save distributions
    save_distributions(stats, output_dir)
    print(f"✅ Distributions: {output_dir / 'distributions.json'}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
