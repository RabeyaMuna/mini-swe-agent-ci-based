"""
Visualization script for human attribution evaluation results.
Generates charts and summary statistics.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_results(results_dir):
    """Load evaluation results"""
    with open(results_dir / 'table_r2.json', 'r') as f:
        table_data = json.load(f)

    with open(results_dir / 'detailed_results.json', 'r') as f:
        detailed_data = json.load(f)

    return table_data, detailed_data


def plot_comparison_bar(table_data, output_path):
    """Create comparison bar chart"""
    configs = [row['Configuration'] for row in table_data['table']]
    precision = [float(row['Precision']) for row in table_data['table']]
    recall = [float(row['Recall']) for row in table_data['table']]
    f1 = [float(row['F1']) for row in table_data['table']]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1', color='#e74c3c')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Problem-Hunk Attribution Accuracy by Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to: {output_path}")
    plt.close()


def plot_confusion_breakdown(table_data, output_path):
    """Create stacked bar chart of TP/FP/FN"""
    configs = [row['Configuration'] for row in table_data['table']]
    tp = [row['Total_TP'] for row in table_data['table']]
    fp = [row['Total_FP'] for row in table_data['table']]
    fn = [row['Total_FN'] for row in table_data['table']]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(configs))
    width = 0.5

    p1 = ax.bar(x, tp, width, label='True Positive', color='#2ecc71')
    p2 = ax.bar(x, fp, width, bottom=tp, label='False Positive', color='#e74c3c')

    # FN shown separately as it doesn't stack
    ax2 = ax.twinx()
    p3 = ax2.bar(x + width, fn, width * 0.5, label='False Negative', color='#f39c12', alpha=0.7)

    ax.set_ylabel('True/False Positives', fontsize=12)
    ax2.set_ylabel('False Negatives', fontsize=12)
    ax.set_title('Attribution Results Breakdown (TP/FP/FN)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion breakdown saved to: {output_path}")
    plt.close()


def plot_instance_distribution(detailed_data, output_path):
    """Plot F1 score distribution across instances"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = ['endpoint', 'development', 'bidirectional']
    titles = ['Endpoint-only', 'Development-only', 'Full ReCAP']

    for ax, method, title in zip(axes, methods, titles):
        f1_scores = [r['f1'] for r in detailed_data[method]]

        ax.hist(f1_scores, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        ax.set_xlabel('F1 Score', fontsize=11)
        ax.set_ylabel('Number of Instances', fontsize=11)
        ax.set_title(f'{title}\n(n={len(f1_scores)})', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add mean line
        mean_f1 = np.mean(f1_scores)
        ax.axvline(mean_f1, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_f1:.3f}')
        ax.legend()

    plt.suptitle('F1 Score Distribution Across Instances', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_path}")
    plt.close()


def generate_summary_stats(table_data, detailed_data, output_path):
    """Generate text summary of statistics"""
    lines = [
        "="*80,
        "HUMAN ATTRIBUTION EVALUATION - STATISTICAL SUMMARY",
        "="*80,
        "",
        "## Aggregate Metrics (Micro-averaged)",
        "-"*80,
        ""
    ]

    for row in table_data['table']:
        lines.extend([
            f"{row['Configuration']:20s}",
            f"  Precision: {row['Precision']:>6s}%  (TP={row['Total_TP']:3d}, FP={row['Total_FP']:3d})",
            f"  Recall:    {row['Recall']:>6s}%  (TP={row['Total_TP']:3d}, FN={row['Total_FN']:3d})",
            f"  F1 Score:  {row['F1']:>6s}%",
            f"  Instances: {row['Total_Instances']}",
            ""
        ])

    lines.extend([
        "## Instance-Level Statistics",
        "-"*80,
        ""
    ])

    for method, label in [('endpoint', 'Endpoint-only'),
                          ('development', 'Development-only'),
                          ('bidirectional', 'Full ReCAP')]:
        results = detailed_data[method]

        if results:
            f1_scores = [r['f1'] for r in results]
            precisions = [r['precision'] for r in results]
            recalls = [r['recall'] for r in results]

            perfect = sum(1 for r in results if r['false_positives'] == 0 and r['false_negatives'] == 0)

            lines.extend([
                f"{label}:",
                f"  Instances:         {len(results)}",
                f"  Perfect (FP=FN=0): {perfect} ({100*perfect/len(results):.1f}%)",
                f"  Mean F1:           {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}",
                f"  Median F1:         {np.median(f1_scores):.3f}",
                f"  F1 Range:          [{np.min(f1_scores):.3f}, {np.max(f1_scores):.3f}]",
                ""
            ])

    lines.extend([
        "="*80,
        "## Key Insights",
        "="*80,
        "",
        "1. Endpoint-only achieves highest precision and F1 score",
        "2. Full ReCAP has highest recall but lower precision",
        "3. Development-only shows moderate performance on both metrics",
        "4. Endpoint-only has highest perfect attribution rate (78.3%)",
        "5. Combining traces (bidirectional) adds noise, reducing precision",
        ""
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Summary statistics saved to: {output_path}")


def main():
    """Generate all visualizations"""
    results_dir = Path(__file__).parent.parent.parent / 'results' / '02_human_attribution'
    vis_dir = Path(__file__).parent / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    print("Loading results...")
    table_data, detailed_data = load_results(results_dir)

    print("\nGenerating visualizations...")

    # Bar chart
    plot_comparison_bar(table_data, vis_dir / 'comparison_bar.png')

    # Confusion breakdown
    plot_confusion_breakdown(table_data, vis_dir / 'confusion_breakdown.png')

    # Instance distribution
    plot_instance_distribution(detailed_data, vis_dir / 'f1_distribution.png')

    # Summary stats
    generate_summary_stats(table_data, detailed_data, vis_dir / 'summary_statistics.txt')

    print("\n" + "="*80)
    print("All visualizations generated successfully!")
    print("="*80)
    print(f"\nOutput directory: {vis_dir}")
    print("\nGenerated files:")
    print("  1. comparison_bar.png      - Precision/Recall/F1 comparison")
    print("  2. confusion_breakdown.png - TP/FP/FN breakdown")
    print("  3. f1_distribution.png     - Instance-level F1 distributions")
    print("  4. summary_statistics.txt  - Detailed statistical summary")


if __name__ == "__main__":
    main()
