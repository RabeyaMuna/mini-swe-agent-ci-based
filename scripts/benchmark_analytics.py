#!/usr/bin/env python3
"""
Benchmark Characterization Analytics

Analyzes CI-repair benchmark and compares with SWE-bench variants:
- Number of commits per PR
- Number of modified files per PR
- Lines of code changed
- CI failure categories
- Number of distinct problems per PR

Usage:
    python scripts/benchmark_analytics.py --data-dir data/trs --output results/analytics
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


class BenchmarkAnalyzer:
    """Analyze CI-repair benchmark characteristics."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.decomposed_issues = self._load_json_or_jsonl(
            data_dir / "decomposed_issues.json"
        )
        self.eval_set = self._load_json_or_jsonl(data_dir / "eval_set.jsonl")
        self.log_details = self._load_json(data_dir / "log_details.json")

    @staticmethod
    def _load_json(path: Path) -> dict | list:
        """Load JSON file."""
        if not path.exists():
            print(f"Warning: {path} not found, using empty data")
            return {}
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_json_or_jsonl(path: Path) -> list[dict]:
        """Load JSON array or JSONL file."""
        if not path.exists():
            print(f"Warning: {path} not found, using empty data")
            return []

        text = path.read_text().strip()
        if not text:
            return []

        if text.startswith("["):
            return json.loads(text)

        # JSONL format
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def analyze_commits_per_pr(self) -> dict[str, Any]:
        """Analyze number of commits per pull request."""
        commits_per_pr = []

        for issue in self.decomposed_issues:
            # Get repair trajectory (commit sequence)
            repair_trajectory = issue.get("repair_trajectory", [])
            num_commits = len(repair_trajectory)

            if num_commits == 0:
                # Fallback: count from git_patches if available
                git_patches = issue.get("git_patches", [])
                num_commits = len(git_patches)

            if num_commits == 0:
                # Single commit (sha_fail -> sha_fix)
                num_commits = 1

            commits_per_pr.append(num_commits)

        return {
            "mean": statistics.mean(commits_per_pr) if commits_per_pr else 0,
            "median": statistics.median(commits_per_pr) if commits_per_pr else 0,
            "std": statistics.stdev(commits_per_pr) if len(commits_per_pr) > 1 else 0,
            "min": min(commits_per_pr) if commits_per_pr else 0,
            "max": max(commits_per_pr) if commits_per_pr else 0,
            "distribution": dict(Counter(commits_per_pr)),
            "raw_data": commits_per_pr,
        }

    def analyze_files_per_pr(self) -> dict[str, Any]:
        """Analyze number of modified files per pull request."""
        files_per_pr = []

        for issue in self.decomposed_issues:
            # Collect all unique files across all problems
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

            # Fallback: check total_changed_files
            if not all_files:
                num_files = issue.get("total_changed_files", 1)
                files_per_pr.append(num_files)
            else:
                files_per_pr.append(len(all_files))

        return {
            "mean": statistics.mean(files_per_pr) if files_per_pr else 0,
            "median": statistics.median(files_per_pr) if files_per_pr else 0,
            "std": statistics.stdev(files_per_pr) if len(files_per_pr) > 1 else 0,
            "min": min(files_per_pr) if files_per_pr else 0,
            "max": max(files_per_pr) if files_per_pr else 0,
            "distribution": dict(Counter(files_per_pr)),
            "raw_data": files_per_pr,
        }

    def analyze_lines_changed(self) -> dict[str, Any]:
        """Analyze lines of code changed per pull request."""
        lines_per_pr = []

        for issue in self.decomposed_issues:
            # Try to get from git patches
            git_patches = issue.get("git_patches", [])
            total_lines = 0

            for patch in git_patches:
                if isinstance(patch, str):
                    # Count lines starting with + or - (excluding +++ and ---)
                    additions = len(
                        [
                            line
                            for line in patch.split("\n")
                            if line.startswith("+") and not line.startswith("+++")
                        ]
                    )
                    deletions = len(
                        [
                            line
                            for line in patch.split("\n")
                            if line.startswith("-") and not line.startswith("---")
                        ]
                    )
                    total_lines += additions + deletions

            if total_lines == 0:
                # Fallback: estimate from changed files (rough estimate: 10 lines per file)
                num_files = issue.get("total_changed_files", 1)
                total_lines = num_files * 10  # Conservative estimate

            lines_per_pr.append(total_lines)

        return {
            "mean": statistics.mean(lines_per_pr) if lines_per_pr else 0,
            "median": statistics.median(lines_per_pr) if lines_per_pr else 0,
            "std": statistics.stdev(lines_per_pr) if len(lines_per_pr) > 1 else 0,
            "min": min(lines_per_pr) if lines_per_pr else 0,
            "max": max(lines_per_pr) if lines_per_pr else 0,
            "distribution": self._get_distribution_buckets(lines_per_pr),
            "raw_data": lines_per_pr,
        }

    @staticmethod
    def _get_distribution_buckets(data: list[int]) -> dict[str, int]:
        """Bucket data into ranges for readability."""
        buckets = {
            "1-10": 0,
            "11-50": 0,
            "51-100": 0,
            "101-500": 0,
            "501+": 0,
        }
        for value in data:
            if value <= 10:
                buckets["1-10"] += 1
            elif value <= 50:
                buckets["11-50"] += 1
            elif value <= 100:
                buckets["51-100"] += 1
            elif value <= 500:
                buckets["101-500"] += 1
            else:
                buckets["501+"] += 1
        return buckets

    def analyze_ci_failure_categories(self) -> dict[str, Any]:
        """Analyze CI failure types/categories."""
        all_categories = []
        category_counts = Counter()

        # From decomposed issues
        for issue in self.decomposed_issues:
            error_types = issue.get("original_error_type", [])
            if isinstance(error_types, list):
                all_categories.extend(error_types)
                category_counts.update(error_types)

        # From log details
        if isinstance(self.log_details, dict):
            log_items = self.log_details.values()
        else:
            log_items = self.log_details

        for log_detail in log_items:
            if not isinstance(log_detail, dict):
                continue

            error_types = log_detail.get("error_types", [])
            if isinstance(error_types, list):
                for error in error_types:
                    if isinstance(error, dict):
                        category = error.get("category", "")
                        if category:
                            all_categories.append(category)
                            category_counts[category] += 1

        return {
            "total_categories": len(set(all_categories)),
            "category_distribution": dict(category_counts.most_common()),
            "most_common": category_counts.most_common(10),
            "raw_categories": all_categories,
        }

    def analyze_problems_per_pr(self) -> dict[str, Any]:
        """Analyze number of distinct problems per pull request."""
        problems_per_pr = []

        for issue in self.decomposed_issues:
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
            "multi_problem_prs": sum(1 for p in problems_per_pr if p > 1),
            "single_problem_prs": sum(1 for p in problems_per_pr if p == 1),
            "raw_data": problems_per_pr,
        }

    def analyze_problem_types(self) -> dict[str, Any]:
        """Analyze types of problems (linting, testing, type checking, etc.)."""
        problem_types = Counter()

        for issue in self.decomposed_issues:
            problems = (
                issue.get("problems")
                or issue.get("problem_sequence")
                or issue.get("optimized_problems")
                or []
            )

            for problem in problems:
                if not isinstance(problem, dict):
                    continue

                # Collect problem type indicators
                failure_type = problem.get("failure_type", "")
                issue_type = problem.get("issue_type", "")
                problem_type = problem.get("problem_type", "")

                if failure_type:
                    problem_types[failure_type] += 1
                if issue_type:
                    problem_types[issue_type] += 1
                if problem_type:
                    problem_types[problem_type] += 1

        return {
            "type_distribution": dict(problem_types.most_common()),
            "total_types": len(problem_types),
            "most_common_types": problem_types.most_common(15),
        }

    def generate_full_report(self) -> dict[str, Any]:
        """Generate complete analytics report."""
        print("Analyzing benchmark characteristics...")

        report = {
            "dataset_size": {
                "total_issues": len(self.decomposed_issues),
                "eval_set_size": len(self.eval_set),
            },
            "commits_per_pr": self.analyze_commits_per_pr(),
            "files_per_pr": self.analyze_files_per_pr(),
            "lines_changed": self.analyze_lines_changed(),
            "ci_failure_categories": self.analyze_ci_failure_categories(),
            "problems_per_pr": self.analyze_problems_per_pr(),
            "problem_types": self.analyze_problem_types(),
        }

        return report


class SWEBenchComparator:
    """Compare with SWE-bench variants."""

    def __init__(self, swe_bench_path: Path | None = None):
        self.swe_bench_path = swe_bench_path
        self.swe_bench_data = None
        self.swe_bench_verified = None
        self.swe_bench_lite = None

        if swe_bench_path and swe_bench_path.exists():
            self._load_swe_bench_data()

    def _load_swe_bench_data(self):
        """Load SWE-bench datasets."""
        try:
            # Try to load SWE-bench datasets
            # Adjust paths based on actual SWE-bench data structure
            if (self.swe_bench_path / "swe-bench.json").exists():
                with open(self.swe_bench_path / "swe-bench.json") as f:
                    self.swe_bench_data = json.load(f)

            if (self.swe_bench_path / "swe-bench-verified.json").exists():
                with open(self.swe_bench_path / "swe-bench-verified.json") as f:
                    self.swe_bench_verified = json.load(f)

            if (self.swe_bench_path / "swe-bench-lite.json").exists():
                with open(self.swe_bench_path / "swe-bench-lite.json") as f:
                    self.swe_bench_lite = json.load(f)

            print(f"Loaded SWE-bench data from {self.swe_bench_path}")
        except Exception as e:
            print(f"Warning: Could not load SWE-bench data: {e}")

    def analyze_swe_bench(self, dataset: list[dict]) -> dict[str, Any]:
        """Analyze a SWE-bench dataset."""
        if not dataset:
            return {}

        files_per_pr = []
        lines_per_pr = []

        for item in dataset:
            # SWE-bench has patch field with git diff
            patch = item.get("patch", "")
            if patch:
                # Count modified files
                files = set()
                for line in patch.split("\n"):
                    if line.startswith("diff --git"):
                        # Extract file path
                        parts = line.split()
                        if len(parts) >= 3:
                            files.add(parts[2])

                files_per_pr.append(len(files) if files else 1)

                # Count lines
                additions = len(
                    [
                        l
                        for l in patch.split("\n")
                        if l.startswith("+") and not l.startswith("+++")
                    ]
                )
                deletions = len(
                    [
                        l
                        for l in patch.split("\n")
                        if l.startswith("-") and not l.startswith("---")
                    ]
                )
                lines_per_pr.append(additions + deletions)

        return {
            "dataset_size": len(dataset),
            "files_per_pr": {
                "mean": statistics.mean(files_per_pr) if files_per_pr else 0,
                "median": statistics.median(files_per_pr) if files_per_pr else 0,
                "std": statistics.stdev(files_per_pr) if len(files_per_pr) > 1 else 0,
            },
            "lines_per_pr": {
                "mean": statistics.mean(lines_per_pr) if lines_per_pr else 0,
                "median": statistics.median(lines_per_pr) if lines_per_pr else 0,
                "std": statistics.stdev(lines_per_pr) if len(lines_per_pr) > 1 else 0,
            },
        }

    def generate_comparison_table(
        self, ci_bench_report: dict[str, Any]
    ) -> pd.DataFrame:
        """Generate comparison table with SWE-bench variants."""

        data = {
            "Metric": [
                "Dataset Size",
                "Commits per PR (mean)",
                "Files per PR (mean)",
                "Files per PR (median)",
                "Lines Changed (mean)",
                "Lines Changed (median)",
                "Problems per PR (mean)",
                "Multi-Problem PRs (%)",
            ],
            "CI-Repair Bench": [
                ci_bench_report["dataset_size"]["total_issues"],
                f"{ci_bench_report['commits_per_pr']['mean']:.2f}",
                f"{ci_bench_report['files_per_pr']['mean']:.2f}",
                f"{ci_bench_report['files_per_pr']['median']:.1f}",
                f"{ci_bench_report['lines_changed']['mean']:.1f}",
                f"{ci_bench_report['lines_changed']['median']:.1f}",
                f"{ci_bench_report['problems_per_pr']['mean']:.2f}",
                f"{100 * ci_bench_report['problems_per_pr']['multi_problem_prs'] / max(ci_bench_report['dataset_size']['total_issues'], 1):.1f}%",
            ],
        }

        # Add SWE-bench columns if available
        if self.swe_bench_data:
            swe_stats = self.analyze_swe_bench(self.swe_bench_data)
            data["SWE-bench"] = [
                swe_stats.get("dataset_size", "N/A"),
                "N/A",  # Commits per PR not in SWE-bench
                f"{swe_stats['files_per_pr']['mean']:.2f}"
                if swe_stats.get("files_per_pr")
                else "N/A",
                f"{swe_stats['files_per_pr']['median']:.1f}"
                if swe_stats.get("files_per_pr")
                else "N/A",
                f"{swe_stats['lines_per_pr']['mean']:.1f}"
                if swe_stats.get("lines_per_pr")
                else "N/A",
                f"{swe_stats['lines_per_pr']['median']:.1f}"
                if swe_stats.get("lines_per_pr")
                else "N/A",
                "N/A",  # Problems per PR not in SWE-bench
                "N/A",
            ]

        if self.swe_bench_verified:
            verified_stats = self.analyze_swe_bench(self.swe_bench_verified)
            data["SWE-bench Verified"] = [
                verified_stats.get("dataset_size", "N/A"),
                "N/A",
                f"{verified_stats['files_per_pr']['mean']:.2f}"
                if verified_stats.get("files_per_pr")
                else "N/A",
                f"{verified_stats['files_per_pr']['median']:.1f}"
                if verified_stats.get("files_per_pr")
                else "N/A",
                f"{verified_stats['lines_per_pr']['mean']:.1f}"
                if verified_stats.get("lines_per_pr")
                else "N/A",
                f"{verified_stats['lines_per_pr']['median']:.1f}"
                if verified_stats.get("lines_per_pr")
                else "N/A",
                "N/A",
                "N/A",
            ]

        if self.swe_bench_lite:
            lite_stats = self.analyze_swe_bench(self.swe_bench_lite)
            data["SWE-bench Lite"] = [
                lite_stats.get("dataset_size", "N/A"),
                "N/A",
                f"{lite_stats['files_per_pr']['mean']:.2f}"
                if lite_stats.get("files_per_pr")
                else "N/A",
                f"{lite_stats['files_per_pr']['median']:.1f}"
                if lite_stats.get("files_per_pr")
                else "N/A",
                f"{lite_stats['lines_per_pr']['mean']:.1f}"
                if lite_stats.get("lines_per_pr")
                else "N/A",
                f"{lite_stats['lines_per_pr']['median']:.1f}"
                if lite_stats.get("lines_per_pr")
                else "N/A",
                "N/A",
                "N/A",
            ]

        return pd.DataFrame(data)


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper."""
    return df.to_latex(index=False, escape=False)


def generate_markdown_table(df: pd.DataFrame) -> str:
    """Generate Markdown table for README/slides."""
    return df.to_markdown(index=False)


def print_summary(report: dict[str, Any]):
    """Print human-readable summary."""
    print("\n" + "=" * 80)
    print("CI-REPAIR BENCHMARK CHARACTERIZATION")
    print("=" * 80)

    print(f"\n📊 Dataset Size: {report['dataset_size']['total_issues']} issues")

    print("\n🔧 Commits per PR:")
    commits = report["commits_per_pr"]
    print(f"  Mean: {commits['mean']:.2f}")
    print(f"  Median: {commits['median']:.1f}")
    print(f"  Range: {commits['min']} - {commits['max']}")

    print("\n📁 Files per PR:")
    files = report["files_per_pr"]
    print(f"  Mean: {files['mean']:.2f}")
    print(f"  Median: {files['median']:.1f}")
    print(f"  Range: {files['min']} - {files['max']}")

    print("\n📝 Lines Changed per PR:")
    lines = report["lines_changed"]
    print(f"  Mean: {lines['mean']:.1f}")
    print(f"  Median: {lines['median']:.1f}")
    print(f"  Range: {lines['min']} - {lines['max']}")

    print("\n🔴 CI Failure Categories:")
    categories = report["ci_failure_categories"]
    print(f"  Total unique categories: {categories['total_categories']}")
    print("  Top 10 categories:")
    for cat, count in categories["most_common"][:10]:
        print(f"    {cat}: {count}")

    print("\n🧩 Problems per PR:")
    problems = report["problems_per_pr"]
    print(f"  Mean: {problems['mean']:.2f}")
    print(f"  Median: {problems['median']:.1f}")
    print(
        f"  Multi-problem PRs: {problems['multi_problem_prs']} ({100 * problems['multi_problem_prs'] / max(report['dataset_size']['total_issues'], 1):.1f}%)"
    )
    print(
        f"  Single-problem PRs: {problems['single_problem_prs']} ({100 * problems['single_problem_prs'] / max(report['dataset_size']['total_issues'], 1):.1f}%)"
    )

    print("\n🏷️  Problem Types:")
    types = report["problem_types"]
    print(f"  Total unique types: {types['total_types']}")
    print("  Top 10 types:")
    for ptype, count in types["most_common_types"][:10]:
        print(f"    {ptype}: {count}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CI-repair benchmark characteristics"
    )
    parser.add_argument(
        "--data-dir",
        default="data/trs",
        help="Directory containing decomposed_issues.json and eval_set.jsonl",
    )
    parser.add_argument(
        "--output", default="results/analytics", help="Output directory for reports"
    )
    parser.add_argument(
        "--swe-bench-path",
        help="Path to SWE-bench data for comparison (optional)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze CI-repair benchmark
    analyzer = BenchmarkAnalyzer(Path(args.data_dir))
    report = analyzer.generate_full_report()

    # Print summary
    print_summary(report)

    # Save full report
    report_file = output_dir / "benchmark_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Full report saved to: {report_file}")

    # Generate comparison with SWE-bench
    comparator = SWEBenchComparator(
        Path(args.swe_bench_path) if args.swe_bench_path else None
    )
    comparison_df = comparator.generate_comparison_table(report)

    # Save comparison table
    comparison_csv = output_dir / "benchmark_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"✅ Comparison table saved to: {comparison_csv}")

    # Save LaTeX table
    latex_file = output_dir / "benchmark_comparison.tex"
    with open(latex_file, "w") as f:
        f.write(generate_latex_table(comparison_df))
    print(f"✅ LaTeX table saved to: {latex_file}")

    # Save Markdown table
    markdown_file = output_dir / "benchmark_comparison.md"
    with open(markdown_file, "w") as f:
        f.write(generate_markdown_table(comparison_df))
    print(f"✅ Markdown table saved to: {markdown_file}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print("\n" + comparison_df.to_string(index=False))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
