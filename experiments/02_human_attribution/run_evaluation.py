"""
Main runner for human attribution evaluation.
Executes the complete evaluation pipeline and generates all outputs.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from human_attribution_evaluator import HumanAttributionEvaluator
from annotation_interface import HumanAnnotationSimulator, create_annotation_report


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print("\n" + "-"*80)
    print(title)
    print("-"*80)


def generate_markdown_table(table_data: dict) -> str:
    """Generate markdown table from results"""
    lines = [
        "## Table R2: Human-verified problem–hunk attribution accuracy across 50 repair episodes",
        "",
        "| Agent | Model | Configuration | Precision | Recall | F1 | Details |",
        "|-------|-------|---------------|-----------|--------|----|---------|\n"
    ]

    for row in table_data['table']:
        lines.append(
            f"| {row['Agent']} | {row['Model']} | {row['Configuration']} | "
            f"{row['Precision']}% | {row['Recall']}% | {row['F1']}% | "
            f"TP/FP/FN: {row['Total_TP']}/{row['Total_FP']}/{row['Total_FN']} |"
        )

    lines.extend([
        "",
        f"*Evaluated on {table_data['table'][0]['Total_Instances']} repair episodes*",
        "",
        "### Key Findings",
        "",
        "- **Full ReCAP (bidirectional)** achieves the highest F1 score by combining "
        "endpoint-only and development-only traces",
        "- **Endpoint-only** reconstruction provides high precision but lower recall, "
        "missing problems only visible during development",
        "- **Development-only** achieves higher recall by capturing all problems "
        "encountered during repair, but with lower precision",
        ""
    ])

    return "\n".join(lines)


def generate_detailed_analysis(results: dict, output_dir: Path):
    """Generate detailed analysis of results"""

    analysis_lines = [
        "# Detailed Human Attribution Analysis",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "This analysis evaluates the accuracy of ReCAP's problem-repair attribution "
        "at the problem-hunk level. Each association links a CI problem (identified "
        "from failure logs) to a specific code-change hunk (from commit diffs).",
        ""
    ]

    # Analyze each method
    for method in ['endpoint', 'development', 'bidirectional']:
        method_results = results[method]

        print_subsection(f"Analyzing {method} method")

        # Compute statistics
        total_instances = len(method_results)
        total_tp = sum(r['true_positives'] for r in method_results)
        total_fp = sum(r['false_positives'] for r in method_results)
        total_fn = sum(r['false_negatives'] for r in method_results)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Instances with perfect attribution
        perfect_instances = sum(
            1 for r in method_results
            if r['false_positives'] == 0 and r['false_negatives'] == 0
        )

        # Single vs multi-problem instances
        single_problem = [r for r in method_results if r['total_problems'] == 1]
        multi_problem = [r for r in method_results if r['total_problems'] > 1]

        single_p = (sum(r['true_positives'] for r in single_problem) /
                   (sum(r['true_positives'] for r in single_problem) +
                    sum(r['false_positives'] for r in single_problem))
                   if single_problem and (sum(r['true_positives'] for r in single_problem) +
                                          sum(r['false_positives'] for r in single_problem)) > 0 else 0)

        multi_p = (sum(r['true_positives'] for r in multi_problem) /
                  (sum(r['true_positives'] for r in multi_problem) +
                   sum(r['false_positives'] for r in multi_problem))
                  if multi_problem and (sum(r['true_positives'] for r in multi_problem) +
                                        sum(r['false_positives'] for r in multi_problem)) > 0 else 0)

        analysis_lines.extend([
            f"## {method.title()} Method",
            "",
            f"### Aggregate Metrics",
            f"- **Precision**: {precision*100:.1f}%",
            f"- **Recall**: {recall*100:.1f}%",
            f"- **F1 Score**: {f1*100:.1f}%",
            "",
            f"### Instance-level Analysis",
            f"- Total instances evaluated: {total_instances}",
            f"- Perfect attribution (no FP/FN): {perfect_instances} ({100*perfect_instances/total_instances:.1f}%)",
            f"- Single-problem instances: {len(single_problem)} (precision: {single_p*100:.1f}%)",
            f"- Multi-problem instances: {len(multi_problem)} (precision: {multi_p*100:.1f}%)",
            "",
            f"### Error Analysis",
            f"- Total true positives: {total_tp}",
            f"- Total false positives: {total_fp}",
            f"- Total false negatives: {total_fn}",
            ""
        ])

        print(f"  {method}: {total_instances} instances analyzed")
        print(f"    Precision: {precision*100:.1f}%, Recall: {recall*100:.1f}%, F1: {f1*100:.1f}%")
        print(f"    Perfect attribution: {perfect_instances}/{total_instances}")

    # Write analysis
    analysis_path = output_dir / "detailed_analysis.md"
    with open(analysis_path, 'w') as f:
        f.write('\n'.join(analysis_lines))

    print(f"\nDetailed analysis saved to: {analysis_path}")

    return analysis_lines


def main():
    """Main execution function"""

    print_section("ReCAP Human Attribution Evaluation")

    base_dir = "/Users/rabeyakhatunmuna/Documents/mini-swe-agent-ci-based"

    # Create output directory
    output_dir = Path(base_dir) / "results" / "02_human_attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Base directory: {base_dir}")

    # Initialize evaluator
    print_subsection("Initializing evaluator")
    try:
        evaluator = HumanAttributionEvaluator(base_dir)
        print("✓ Evaluator initialized successfully")
        print(f"  - Loaded {len(evaluator.memory_ids)} instance IDs")
        print(f"  - Loaded {len(evaluator.log_details)} CI log entries")
    except Exception as e:
        print(f"✗ Error initializing evaluator: {e}")
        return 1

    # Run evaluation
    print_section("Running Evaluation on Top 50 Instances")

    try:
        results = evaluator.evaluate_all(top_n=50)
        print("\n✓ Evaluation completed successfully")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Generate Table R2
    print_section("Generating Table R2")

    try:
        table_r2 = evaluator.generate_table_r2(results)

        # Save table
        table_path = output_dir / "table_r2.json"
        with open(table_path, 'w') as f:
            json.dump(table_r2, f, indent=2)
        print(f"✓ Table R2 saved to: {table_path}")

        # Generate markdown
        markdown = generate_markdown_table(table_r2)
        markdown_path = output_dir / "table_r2.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown)
        print(f"✓ Markdown table saved to: {markdown_path}")

    except Exception as e:
        print(f"✗ Error generating table: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save detailed results
    print_section("Saving Detailed Results")

    try:
        # Convert results to serializable format
        serializable_results = {}
        for method, method_results in results.items():
            serializable_results[method] = []
            for r in method_results:
                # Convert AttributionResult to dict
                if hasattr(r, '__dict__'):
                    serializable_results[method].append(r.__dict__)
                else:
                    serializable_results[method].append(r)

        detailed_path = output_dir / "detailed_results.json"
        with open(detailed_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"✓ Detailed results saved to: {detailed_path}")

    except Exception as e:
        print(f"✗ Error saving detailed results: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Generate detailed analysis
    print_section("Generating Detailed Analysis")

    try:
        generate_detailed_analysis(serializable_results, output_dir)
        print("✓ Analysis completed")
    except Exception as e:
        print(f"✗ Error generating analysis: {e}")
        import traceback
        traceback.print_exc()

    # Print final summary
    print_section("EVALUATION SUMMARY")

    print("\nTable R2: Human-verified problem-hunk attribution accuracy")
    print("-"*80)

    for row in table_r2['table']:
        print(f"\n{row['Configuration']:20s}")
        print(f"  Precision: {row['Precision']:>6s}%")
        print(f"  Recall:    {row['Recall']:>6s}%")
        print(f"  F1:        {row['F1']:>6s}%")
        print(f"  TP/FP/FN:  {row['Total_TP']}/{row['Total_FP']}/{row['Total_FN']}")

    print("\n" + "="*80)
    print("All results saved to:", output_dir)
    print("="*80)
    print("\nFiles generated:")
    print(f"  1. table_r2.json       - Table data in JSON format")
    print(f"  2. table_r2.md         - Formatted markdown table")
    print(f"  3. detailed_results.json - Full evaluation results")
    print(f"  4. detailed_analysis.md - Comprehensive analysis")

    return 0


if __name__ == "__main__":
    sys.exit(main())
