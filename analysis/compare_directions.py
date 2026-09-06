#!/usr/bin/env python3
"""
Compare results across different decomposition directions (forward, backward, bidirectional).

Usage:
    python analysis/compare_directions.py --forward <path> --backward <path> --bidirectional <path>

    # Or with defaults:
    python analysis/compare_directions.py  # Uses results/miniswe-agent/*/l1_l2_l3_minimax-m2.5/*/jobs_results_diff.jsonl
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple


def load_results(filepath: Path) -> Dict[str, Dict]:
    """Load JSONL file and return dict of {instance_id: result}"""
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return {}

    results = {}
    with open(filepath) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                instance_id = item.get("id")
                conclusion = item.get("conclusion")
                results[instance_id] = {
                    "conclusion": conclusion,
                    "repo": item.get("repo_name", ""),
                    "workflow": item.get("workflow", ""),
                    "url": item.get("url", ""),
                    "full": item
                }
    return results


def analyze_success_patterns(
    data: Dict[str, Dict[str, Dict]]
) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """Analyze which instances succeeded in which directions."""

    # Get success sets for each direction
    success_by_direction = {
        direction: {id for id, r in results.items() if r["conclusion"] == "success"}
        for direction, results in data.items()
    }

    # Calculate all the Venn diagram regions
    all_three = (
        success_by_direction["forward"]
        & success_by_direction["backward"]
        & success_by_direction["bidirectional"]
    )

    forward_backward = (
        (success_by_direction["forward"] & success_by_direction["backward"])
        - success_by_direction["bidirectional"]
    )
    forward_bidir = (
        (success_by_direction["forward"] & success_by_direction["bidirectional"])
        - success_by_direction["backward"]
    )
    backward_bidir = (
        (success_by_direction["backward"] & success_by_direction["bidirectional"])
        - success_by_direction["forward"]
    )

    only_forward = (
        success_by_direction["forward"]
        - success_by_direction["backward"]
        - success_by_direction["bidirectional"]
    )
    only_backward = (
        success_by_direction["backward"]
        - success_by_direction["forward"]
        - success_by_direction["bidirectional"]
    )
    only_bidir = (
        success_by_direction["bidirectional"]
        - success_by_direction["forward"]
        - success_by_direction["backward"]
    )

    # All instance IDs
    all_ids = set()
    for results in data.values():
        all_ids.update(results.keys())

    none_succeeded = (
        all_ids
        - success_by_direction["forward"]
        - success_by_direction["backward"]
        - success_by_direction["bidirectional"]
    )

    patterns = {
        "all_three": all_three,
        "forward_backward": forward_backward,
        "forward_bidir": forward_bidir,
        "backward_bidir": backward_bidir,
        "only_forward": only_forward,
        "only_backward": only_backward,
        "only_bidir": only_bidir,
        "none": none_succeeded,
    }

    counts = {k: len(v) for k, v in patterns.items()}
    counts["total"] = len(all_ids)

    return patterns, counts


def analyze_repo_performance(data: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    """Analyze success rates by repository and direction."""

    repo_stats = defaultdict(lambda: {
        "forward": {"success": 0, "total": 0},
        "backward": {"success": 0, "total": 0},
        "bidirectional": {"success": 0, "total": 0}
    })

    for direction in ["forward", "backward", "bidirectional"]:
        for id, result in data[direction].items():
            repo = result["repo"]
            repo_stats[repo][direction]["total"] += 1
            if result["conclusion"] == "success":
                repo_stats[repo][direction]["success"] += 1

    # Find repos with significant differences
    repos_with_differences = []
    for repo, stats in sorted(repo_stats.items()):
        f_rate = stats["forward"]["success"] / max(stats["forward"]["total"], 1)
        b_rate = stats["backward"]["success"] / max(stats["backward"]["total"], 1)
        bi_rate = stats["bidirectional"]["success"] / max(stats["bidirectional"]["total"], 1)

        total = (stats["forward"]["total"]
                + stats["backward"]["total"]
                + stats["bidirectional"]["total"])

        if total >= 3:
            diff = max(f_rate, b_rate, bi_rate) - min(f_rate, b_rate, bi_rate)
            if diff > 0.3:  # At least 30% difference
                repos_with_differences.append({
                    "repo": repo,
                    "forward": {"success": stats["forward"]["success"],
                               "total": stats["forward"]["total"],
                               "rate": f_rate},
                    "backward": {"success": stats["backward"]["success"],
                                "total": stats["backward"]["total"],
                                "rate": b_rate},
                    "bidirectional": {"success": stats["bidirectional"]["success"],
                                     "total": stats["bidirectional"]["total"],
                                     "rate": bi_rate},
                    "difference": diff
                })

    return repos_with_differences


def print_report(
    data: Dict[str, Dict[str, Dict]],
    patterns: Dict[str, Set[str]],
    counts: Dict[str, int],
    repo_differences: List[Dict],
    output_json: Path = None
):
    """Print comprehensive comparison report."""

    print("\n" + "="*80)
    print("DIRECTION COMPARISON REPORT")
    print("="*80)

    # Overall stats
    print(f"\nTotal instances: {counts['total']}")
    print("\nResults by direction:")
    for direction in ["forward", "backward", "bidirectional"]:
        success = sum(1 for r in data[direction].values() if r["conclusion"] == "success")
        failure = sum(1 for r in data[direction].values() if r["conclusion"] == "failure")
        total = len(data[direction])
        print(f"  {direction:15} SUCCESS: {success:3} ({success/max(total,1)*100:5.1f}%)  "
              f"FAILURE: {failure:3} ({failure/max(total,1)*100:5.1f}%)")

    # Venn diagram
    print("\n" + "="*80)
    print("VENN DIAGRAM ANALYSIS")
    print("="*80)
    print(f"\n✓ All 3 directions succeeded:     {counts['all_three']:3} ({counts['all_three']/counts['total']*100:5.1f}%)")
    print(f"\n✓ Two directions succeeded:       {counts['forward_backward'] + counts['forward_bidir'] + counts['backward_bidir']:3}")
    print(f"    Forward + Backward only:      {counts['forward_backward']:3}")
    print(f"    Forward + Bidirectional only: {counts['forward_bidir']:3}")
    print(f"    Backward + Bidirectional only:{counts['backward_bidir']:3}")
    print(f"\n✓ Only one direction succeeded:   {counts['only_forward'] + counts['only_backward'] + counts['only_bidir']:3}")
    print(f"    ONLY Forward:                 {counts['only_forward']:3}")
    print(f"    ONLY Backward:                {counts['only_backward']:3}")
    print(f"    ONLY Bidirectional:           {counts['only_bidir']:3}")
    print(f"\n✗ None succeeded:                 {counts['none']:3} ({counts['none']/counts['total']*100:5.1f}%)")

    # Direction-specific examples
    print("\n" + "="*80)
    print("DIRECTION-SPECIFIC SUCCESSES (Examples)")
    print("="*80)

    for direction_name, pattern_key in [
        ("FORWARD", "only_forward"),
        ("BACKWARD", "only_backward"),
        ("BIDIRECTIONAL", "only_bidir")
    ]:
        ids = sorted(patterns[pattern_key])
        if ids:
            print(f"\n--- ONLY {direction_name} SUCCEEDED ({len(ids)} instances) ---")
            for idx, id in enumerate(ids[:5]):
                result = data[direction_name.lower()][id]
                repo = result["repo"]
                workflow = result["workflow"].split("/")[-1]  # Just filename
                print(f"  {idx+1}. [{repo}] {id} - {workflow}")

    # Repository performance
    if repo_differences:
        print("\n" + "="*80)
        print("REPOSITORIES WITH DIRECTION-SPECIFIC PERFORMANCE")
        print("="*80)
        print("\nRepositories with >30% difference between directions (≥3 instances):")

        for repo_data in sorted(repo_differences, key=lambda x: -x["difference"])[:10]:
            print(f"\n{repo_data['repo']} (Δ={repo_data['difference']:.1%}):")
            for direction in ["forward", "backward", "bidirectional"]:
                s = repo_data[direction]
                print(f"  {direction:15} {s['success']:2}/{s['total']:2} = {s['rate']:5.1%}")

    # Save JSON output
    if output_json:
        output_data = {
            "summary": counts,
            "patterns": {k: sorted(list(v)) for k, v in patterns.items()},
            "repo_differences": repo_differences
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n\nDetailed results saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare decomposition direction results"
    )
    parser.add_argument(
        "--forward",
        type=Path,
        help="Path to forward results JSONL"
    )
    parser.add_argument(
        "--backward",
        type=Path,
        help="Path to backward results JSONL"
    )
    parser.add_argument(
        "--bidirectional",
        type=Path,
        help="Path to bidirectional results JSONL"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/miniswe-agent"),
        help="Base directory for default paths"
    )
    parser.add_argument(
        "--model",
        default="l1_l2_l3_minimax-m2.5",
        help="Model name for default paths"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/direction_comparison.json"),
        help="Output JSON file path (ID lists saved to same directory)"
    )
    parser.add_argument(
        "--generate-id-lists",
        action="store_true",
        default=True,
        help="Generate separate ID list files for re-running (default: True)"
    )

    args = parser.parse_args()

    # Use defaults if not specified
    if not args.forward:
        args.forward = (args.base_dir / "forward" / args.model / "L1_L2_L3"
                       / "jobs_results_diff.jsonl")
    if not args.backward:
        args.backward = (args.base_dir / "backward" / args.model
                        / "jobs_results_diff.jsonl")
    if not args.bidirectional:
        args.bidirectional = (args.base_dir / "bidirectional" / args.model / "L1_L2_L3"
                             / "jobs_results_diff.jsonl")

    # Load data
    data = {
        "forward": load_results(args.forward),
        "backward": load_results(args.backward),
        "bidirectional": load_results(args.bidirectional)
    }

    if not any(data.values()):
        print("Error: No valid result files found!")
        return 1

    # Analyze
    patterns, counts = analyze_success_patterns(data)
    repo_differences = analyze_repo_performance(data)

    # Report
    print_report(data, patterns, counts, repo_differences, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
