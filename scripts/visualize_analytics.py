#!/usr/bin/env python3
"""
Visualization for Benchmark Analytics

Generates publication-quality charts for benchmark characterization:
- Distribution plots
- Comparison bar charts
- Problem category pie charts

Usage:
    python scripts/visualize_analytics.py --report results/analytics/benchmark_report.json --output results/analytics/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"


def plot_commits_distribution(report: dict, output_dir: Path):
    """Plot commits per PR distribution."""
    commits = report["commits_per_pr"]
    distribution = commits["distribution"]

    # Sort by number of commits
    sorted_items = sorted(distribution.items(), key=lambda x: int(x[0]))
    x = [int(k) for k, v in sorted_items]
    y = [v for k, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, y, color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Number of Commits per PR", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of Commits per Pull Request", fontweight="bold", pad=20)

    # Add mean line
    mean = commits["mean"]
    ax.axvline(
        mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.2f}"
    )
    ax.legend()

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "commits_distribution.png", bbox_inches="tight")
    plt.savefig(output_dir / "commits_distribution.pdf", bbox_inches="tight")
    plt.close()


def plot_files_distribution(report: dict, output_dir: Path):
    """Plot files per PR distribution."""
    files = report["files_per_pr"]
    distribution = files["distribution"]

    # Group into buckets for readability
    buckets = {
        "1": 0,
        "2-3": 0,
        "4-5": 0,
        "6-10": 0,
        "11+": 0,
    }

    for num_files, count in distribution.items():
        num = int(num_files)
        if num == 1:
            buckets["1"] += count
        elif num <= 3:
            buckets["2-3"] += count
        elif num <= 5:
            buckets["4-5"] += count
        elif num <= 10:
            buckets["6-10"] += count
        else:
            buckets["11+"] += count

    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = list(buckets.keys())
    y = list(buckets.values())

    ax.bar(x_labels, y, color="forestgreen", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Number of Files Modified per PR", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(
        "Distribution of Modified Files per Pull Request", fontweight="bold", pad=20
    )

    # Add mean annotation
    mean = files["mean"]
    ax.text(
        0.95,
        0.95,
        f"Mean: {mean:.2f}\nMedian: {files['median']:.1f}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=12,
    )

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "files_distribution.png", bbox_inches="tight")
    plt.savefig(output_dir / "files_distribution.pdf", bbox_inches="tight")
    plt.close()


def plot_lines_distribution(report: dict, output_dir: Path):
    """Plot lines changed distribution."""
    lines = report["lines_changed"]
    distribution = lines["distribution"]

    labels = list(distribution.keys())
    values = list(distribution.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color="coral", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Lines Changed per PR", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(
        "Distribution of Lines Changed per Pull Request", fontweight="bold", pad=20
    )

    # Add statistics
    mean = lines["mean"]
    median = lines["median"]
    ax.text(
        0.95,
        0.95,
        f"Mean: {mean:.1f}\nMedian: {median:.1f}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=12,
    )

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "lines_distribution.png", bbox_inches="tight")
    plt.savefig(output_dir / "lines_distribution.pdf", bbox_inches="tight")
    plt.close()


def plot_problems_distribution(report: dict, output_dir: Path):
    """Plot problems per PR distribution."""
    problems = report["problems_per_pr"]
    distribution = problems["distribution"]

    sorted_items = sorted(distribution.items(), key=lambda x: int(x[0]))
    x = [int(k) for k, v in sorted_items]
    y = [v for k, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#FF6B6B" if int(k) == 1 else "#4ECDC4" for k, v in sorted_items]
    ax.bar(x, y, color=colors, alpha=0.8, edgecolor="black")
    ax.set_xlabel("Number of Problems per PR", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of Problems per Pull Request", fontweight="bold", pad=20)

    # Add legend for single vs multi-problem
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#FF6B6B", label="Single Problem"),
        Patch(facecolor="#4ECDC4", label="Multi-Problem"),
    ]
    ax.legend(handles=legend_elements)

    # Add statistics
    total_prs = report["dataset_size"]["total_issues"]
    multi_pct = 100 * problems["multi_problem_prs"] / max(total_prs, 1)
    ax.text(
        0.95,
        0.95,
        f"Mean: {problems['mean']:.2f}\nMulti-Problem: {multi_pct:.1f}%",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=12,
    )

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "problems_distribution.png", bbox_inches="tight")
    plt.savefig(output_dir / "problems_distribution.pdf", bbox_inches="tight")
    plt.close()


def plot_ci_categories(report: dict, output_dir: Path):
    """Plot CI failure categories pie chart."""
    categories = report["ci_failure_categories"]
    top_categories = dict(categories["most_common"][:8])

    # Group remaining as "Other"
    total_count = sum(categories["category_distribution"].values())
    top_count = sum(top_categories.values())
    if total_count > top_count:
        top_categories["Other"] = total_count - top_count

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = sns.color_palette("Set3", len(top_categories))
    wedges, texts, autotexts = ax.pie(
        top_categories.values(),
        labels=top_categories.keys(),
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 11, "weight": "bold"},
    )

    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)

    ax.set_title(
        "CI Failure Categories Distribution", fontweight="bold", fontsize=16, pad=20
    )

    plt.tight_layout()
    plt.savefig(output_dir / "ci_categories.png", bbox_inches="tight")
    plt.savefig(output_dir / "ci_categories.pdf", bbox_inches="tight")
    plt.close()


def plot_problem_types(report: dict, output_dir: Path):
    """Plot problem types horizontal bar chart."""
    types = report["problem_types"]
    top_types = dict(types["most_common_types"][:12])

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(top_types))
    values = list(top_types.values())
    labels = list(top_types.keys())

    bars = ax.barh(y_pos, values, color="mediumpurple", alpha=0.8, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Top to bottom
    ax.set_xlabel("Frequency", fontweight="bold")
    ax.set_title("Top Problem Types", fontweight="bold", fontsize=16, pad=20)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            value + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value}",
            va="center",
            fontsize=10,
        )

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "problem_types.png", bbox_inches="tight")
    plt.savefig(output_dir / "problem_types.pdf", bbox_inches="tight")
    plt.close()


def plot_benchmark_comparison(comparison_csv: Path, output_dir: Path):
    """Plot benchmark comparison chart."""
    import pandas as pd

    if not comparison_csv.exists():
        print(f"Warning: {comparison_csv} not found, skipping comparison plot")
        return

    df = pd.read_csv(comparison_csv)

    # Extract numeric metrics for comparison
    metrics_to_plot = [
        "Files per PR (mean)",
        "Lines Changed (mean)",
        "Problems per PR (mean)",
    ]

    # Prepare data
    plot_data = []
    for metric in metrics_to_plot:
        row = df[df["Metric"] == metric]
        if not row.empty:
            for col in df.columns[1:]:  # Skip 'Metric' column
                value_str = row[col].values[0]
                if value_str != "N/A":
                    try:
                        value = float(value_str)
                        plot_data.append(
                            {"Metric": metric, "Benchmark": col, "Value": value}
                        )
                    except ValueError:
                        pass

    if not plot_data:
        print("No data available for comparison plot")
        return

    plot_df = pd.DataFrame(plot_data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    benchmarks = plot_df["Benchmark"].unique()
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    colors = ["#FF6B6B", "#4ECDC4", "#95E1D3", "#F38181"]

    for i, benchmark in enumerate(benchmarks):
        data = plot_df[plot_df["Benchmark"] == benchmark]
        values = [
            data[data["Metric"] == m]["Value"].values[0]
            if len(data[data["Metric"] == m]) > 0
            else 0
            for m in metrics_to_plot
        ]
        offset = width * (i - len(benchmarks) / 2 + 0.5)
        ax.bar(
            x + offset,
            values,
            width,
            label=benchmark,
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    ax.set_xlabel("Metric", fontweight="bold")
    ax.set_ylabel("Value", fontweight="bold")
    ax.set_title("Benchmark Comparison", fontweight="bold", fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace(" (mean)", "") for m in metrics_to_plot], rotation=15, ha="right"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_comparison.png", bbox_inches="tight")
    plt.savefig(output_dir / "benchmark_comparison.pdf", bbox_inches="tight")
    plt.close()


def create_all_visualizations(report_path: Path, output_dir: Path, analytics_dir: Path):
    """Create all visualization plots."""
    # Load report
    with open(report_path) as f:
        report = json.load(f)

    print("Generating visualizations...")

    # Create plots
    plot_commits_distribution(report, output_dir)
    print("  ✓ Commits distribution")

    plot_files_distribution(report, output_dir)
    print("  ✓ Files distribution")

    plot_lines_distribution(report, output_dir)
    print("  ✓ Lines changed distribution")

    plot_problems_distribution(report, output_dir)
    print("  ✓ Problems distribution")

    plot_ci_categories(report, output_dir)
    print("  ✓ CI categories pie chart")

    plot_problem_types(report, output_dir)
    print("  ✓ Problem types bar chart")

    # Comparison plot
    comparison_csv = analytics_dir / "benchmark_comparison.csv"
    if comparison_csv.exists():
        plot_benchmark_comparison(comparison_csv, output_dir)
        print("  ✓ Benchmark comparison chart")

    print(f"\n✅ All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
    parser.add_argument(
        "--report",
        default="results/analytics/benchmark_report.json",
        help="Path to benchmark_report.json",
    )
    parser.add_argument(
        "--output",
        default="results/analytics/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--analytics-dir",
        default="results/analytics",
        help="Directory containing analytics results",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    create_all_visualizations(Path(args.report), output_dir, Path(args.analytics_dir))


if __name__ == "__main__":
    main()
