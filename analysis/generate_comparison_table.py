#!/usr/bin/env python3
"""
Generate detailed comparison table for analyzing direction-specific behavior.

Outputs:
- CSV table with all issues and their status in each direction
- Separate CSV files for each comparison category
- Easy to open in Excel/Google Sheets for analysis

Usage:
    python analysis/generate_comparison_table.py
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_results_full(filepath: Path) -> Dict[str, Dict]:
    """Load results with full details"""
    results = {}
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return {}

    with open(filepath) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                instance_id = item.get("id")
                results[instance_id] = {
                    "id": instance_id,
                    "repo": item.get("repo_name", ""),
                    "workflow": item.get("workflow", ""),
                    "workflow_name": item.get("workflow", "").split("/")[-1],
                    "commit": item.get("commit", "")[:8],
                    "sha_original": item.get("sha_original", "")[:8],
                    "conclusion": item.get("conclusion", ""),
                    "url": item.get("url", ""),
                }
    return results


def generate_master_table(
    forward: Dict, backward: Dict, bidirectional: Dict, output_path: Path
):
    """Generate master comparison table with all issues"""

    # Get all unique IDs
    all_ids = set(forward.keys()) | set(backward.keys()) | set(bidirectional.keys())

    rows = []
    for id in sorted(all_ids):
        fwd_result = forward.get(id, {})
        back_result = backward.get(id, {})
        bidir_result = bidirectional.get(id, {})

        # Determine success pattern
        fwd_success = fwd_result.get("conclusion") == "success"
        back_success = back_result.get("conclusion") == "success"
        bidir_success = bidir_result.get("conclusion") == "success"

        # Categorize
        if fwd_success and back_success and bidir_success:
            category = "All 3 Solved"
        elif fwd_success and back_success and not bidir_success:
            category = "Fwd+Back Only (Bidir MISSED)"
        elif fwd_success and bidir_success and not back_success:
            category = "Fwd+Bidir Only (Back MISSED)"
        elif back_success and bidir_success and not fwd_success:
            category = "Back+Bidir Only (Fwd MISSED)"
        elif fwd_success and not back_success and not bidir_success:
            category = "ONLY Forward"
        elif back_success and not fwd_success and not bidir_success:
            category = "ONLY Backward"
        elif bidir_success and not fwd_success and not back_success:
            category = "ONLY Bidirectional"
        elif not fwd_success and not back_success and not bidir_success:
            category = "All Failed"
        else:
            category = "Other"

        # Get repo name (from any available result)
        repo = (fwd_result.get("repo") or back_result.get("repo")
                or bidir_result.get("repo", ""))
        workflow = (fwd_result.get("workflow_name") or back_result.get("workflow_name")
                   or bidir_result.get("workflow_name", ""))

        row = {
            "ID": id,
            "Repo": repo,
            "Workflow": workflow,
            "Category": category,
            "Forward_Status": fwd_result.get("conclusion", "not_run"),
            "Backward_Status": back_result.get("conclusion", "not_run"),
            "Bidirectional_Status": bidir_result.get("conclusion", "not_run"),
            "Forward_Success": "✓" if fwd_success else "✗",
            "Backward_Success": "✓" if back_success else "✗",
            "Bidirectional_Success": "✓" if bidir_success else "✗",
            "Success_Count": sum([fwd_success, back_success, bidir_success]),
            "Forward_URL": fwd_result.get("url", ""),
            "Backward_URL": back_result.get("url", ""),
            "Bidirectional_URL": bidir_result.get("url", ""),
        }
        rows.append(row)

    # Write master CSV
    fieldnames = [
        "ID", "Repo", "Workflow", "Category",
        "Forward_Success", "Backward_Success", "Bidirectional_Success",
        "Success_Count",
        "Forward_Status", "Backward_Status", "Bidirectional_Status",
        "Forward_URL", "Backward_URL", "Bidirectional_URL"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Master table: {output_path} ({len(rows)} issues)")
    return rows


def generate_category_tables(rows: List[Dict], output_dir: Path):
    """Generate separate CSV for each category"""

    categories = {}
    for row in rows:
        cat = row["Category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(row)

    # Write each category to separate CSV
    for category, cat_rows in categories.items():
        if category == "All Failed":
            continue  # Skip all-failed (too many)

        safe_name = category.replace(" ", "_").replace("+", "and").lower()
        filename = output_dir / f"category_{safe_name}.csv"

        fieldnames = [
            "ID", "Repo", "Workflow", "Category", "Success_Count",
            "Forward_Success", "Backward_Success", "Bidirectional_Success",
            "Forward_Status", "Backward_Status", "Bidirectional_Status",
            "Forward_URL", "Backward_URL", "Bidirectional_URL"
        ]

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cat_rows)

        print(f"  - {category}: {filename.name} ({len(cat_rows)} issues)")


def generate_analysis_ready_table(rows: List[Dict], output_path: Path):
    """Generate analysis-ready table with key columns for investigation"""

    # Filter to only issues where at least one succeeded
    interesting_rows = [r for r in rows if r["Success_Count"] > 0 and r["Success_Count"] < 3]

    analysis_rows = []
    for row in interesting_rows:
        fwd_ok = row["Forward_Success"] == "✓"
        back_ok = row["Backward_Success"] == "✓"
        bidir_ok = row["Bidirectional_Success"] == "✓"

        # Identify what worked and what didn't
        solved_by = []
        failed_in = []

        if fwd_ok:
            solved_by.append("Forward")
        else:
            failed_in.append("Forward")

        if back_ok:
            solved_by.append("Backward")
        else:
            failed_in.append("Backward")

        if bidir_ok:
            solved_by.append("Bidirectional")
        else:
            failed_in.append("Bidirectional")

        analysis_row = {
            "ID": row["ID"],
            "Repo": row["Repo"],
            "Workflow": row["Workflow"],
            "Solved_By": " + ".join(solved_by),
            "Failed_In": " + ".join(failed_in),
            "Category": row["Category"],
            "Investigation_Priority": "",  # Empty for manual notes
            "Root_Cause": "",  # Empty for manual notes
            "Notes": "",  # Empty for manual notes
        }
        analysis_rows.append(analysis_row)

    fieldnames = [
        "ID", "Repo", "Workflow", "Solved_By", "Failed_In", "Category",
        "Investigation_Priority", "Root_Cause", "Notes"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_rows)

    print(f"✓ Analysis table: {output_path} ({len(analysis_rows)} issues)")


def generate_comparison_matrix(rows: List[Dict], output_path: Path):
    """Generate comparison matrix showing patterns"""

    # Count patterns by repo and workflow type
    repo_stats = {}
    workflow_stats = {}

    for row in rows:
        if row["Success_Count"] == 0:
            continue

        repo = row["Repo"]
        workflow = row["Workflow"]
        category = row["Category"]

        if repo not in repo_stats:
            repo_stats[repo] = {
                "total": 0,
                "all_3": 0,
                "only_fwd": 0,
                "only_back": 0,
                "only_bidir": 0,
                "fwd_back_only": 0,
            }

        repo_stats[repo]["total"] += 1

        if category == "All 3 Solved":
            repo_stats[repo]["all_3"] += 1
        elif category == "ONLY Forward":
            repo_stats[repo]["only_fwd"] += 1
        elif category == "ONLY Backward":
            repo_stats[repo]["only_back"] += 1
        elif category == "ONLY Bidirectional":
            repo_stats[repo]["only_bidir"] += 1
        elif category == "Fwd+Back Only (Bidir MISSED)":
            repo_stats[repo]["fwd_back_only"] += 1

    # Write repo stats
    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "Repo", "Total_Success", "All_3", "Only_Forward", "Only_Backward",
            "Only_Bidirectional", "Fwd+Back_Only"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for repo, stats in sorted(repo_stats.items()):
            if stats["total"] < 2:
                continue  # Skip repos with <2 successes

            writer.writerow({
                "Repo": repo,
                "Total_Success": stats["total"],
                "All_3": stats["all_3"],
                "Only_Forward": stats["only_fwd"],
                "Only_Backward": stats["only_back"],
                "Only_Bidirectional": stats["only_bidir"],
                "Fwd+Back_Only": stats["fwd_back_only"],
            })

    print(f"✓ Comparison matrix: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison tables for direction analysis"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/miniswe-agent"),
        help="Base directory for results"
    )
    parser.add_argument(
        "--model",
        default="l1_l2_l3_minimax-m2.5",
        help="Model name"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Output directory for tables"
    )

    args = parser.parse_args()

    # Load results
    forward_path = args.base_dir / "forward" / args.model / "L1_L2_L3" / "jobs_results_diff.jsonl"
    backward_path = args.base_dir / "backward" / args.model / "jobs_results_diff.jsonl"
    bidir_path = args.base_dir / "bidirectional" / args.model / "L1_L2_L3" / "jobs_results_diff.jsonl"

    forward = load_results_full(forward_path)
    backward = load_results_full(backward_path)
    bidirectional = load_results_full(bidir_path)

    if not any([forward, backward, bidirectional]):
        print("Error: No results found!")
        return 1

    print(f"\nLoaded results:")
    print(f"  Forward:       {len(forward)} issues")
    print(f"  Backward:      {len(backward)} issues")
    print(f"  Bidirectional: {len(bidirectional)} issues")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating comparison tables...")
    print("="*80)

    # 1. Master table (all issues)
    master_path = args.output_dir / "comparison_master.csv"
    rows = generate_master_table(forward, backward, bidirectional, master_path)

    # 2. Category-specific tables
    print(f"\n✓ Category tables:")
    generate_category_tables(rows, args.output_dir)

    # 3. Analysis-ready table (for investigation)
    analysis_path = args.output_dir / "analysis_ready.csv"
    print(f"\n")
    generate_analysis_ready_table(rows, analysis_path)

    # 4. Comparison matrix (repo-level patterns)
    matrix_path = args.output_dir / "comparison_matrix_by_repo.csv"
    generate_comparison_matrix(rows, matrix_path)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    categories = {}
    for row in rows:
        cat = row["Category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\nIssue Distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:40} {count:3}")

    print("\n" + "="*80)
    print("GENERATED FILES")
    print("="*80)
    print(f"\n📊 Main Analysis Files:")
    print(f"  1. {master_path.name}")
    print(f"     → Complete table with all issues and their status")
    print(f"\n  2. {analysis_path.name}")
    print(f"     → Ready for investigation (only partial successes)")
    print(f"     → Has empty columns for your notes!")
    print(f"\n  3. {matrix_path.name}")
    print(f"     → Repo-level pattern summary")
    print(f"\n📁 Category-Specific Files:")
    print(f"  → category_*.csv (one per comparison category)")

    print(f"\n💡 Next Steps:")
    print(f"  1. Open {analysis_path.name} in Excel/Google Sheets")
    print(f"  2. Sort by 'Failed_In' to group similar issues")
    print(f"  3. Add notes in Investigation_Priority, Root_Cause, Notes columns")
    print(f"  4. Use category_*.csv files to deep-dive specific patterns")

    return 0


if __name__ == "__main__":
    exit(main())
