#!/usr/bin/env python3
"""
Compare results across different decomposition directions and generate ID lists for re-running.

Usage:
    python analysis/compare_directions_v2.py

Generates comprehensive comparison including:
- Common IDs (solved by all)
- Pairwise comparisons (A+B solved, C missed)
- Direction-specific (only one solved)
- Per-direction success/failure lists
- Comparison lists (A solved but B missed)
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


def save_id_lists(patterns: Dict[str, Set[str]], output_dir: Path):
    """Save comprehensive ID lists for re-running experiments."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get success sets for comparisons
    fwd_success = (patterns["all_three"] | patterns["forward_backward"]
                  | patterns["forward_bidir"] | patterns["only_forward"])
    back_success = (patterns["all_three"] | patterns["forward_backward"]
                   | patterns["backward_bidir"] | patterns["only_backward"])
    bidir_success = (patterns["all_three"] | patterns["forward_bidir"]
                    | patterns["backward_bidir"] | patterns["only_bidir"])

    print("\n" + "="*80)
    print("SAVED ID LISTS FOR RE-RUNNING")
    print("="*80)

    # 1. Common - solved by ALL three
    with open(output_dir / "ids_common_all_three.txt", "w") as f:
        for id in sorted(patterns["all_three"]):
            f.write(f"{id}\n")
    print(f"\n✓ COMMON (all 3 solved): {len(patterns['all_three'])} IDs")
    print(f"  → ids_common_all_three.txt")

    # 2. Pairwise common (solved by 2, missed by 1)
    # Forward + Backward solved, Bidirectional missed
    with open(output_dir / "ids_forward_backward_only.txt", "w") as f:
        for id in sorted(patterns["forward_backward"]):
            f.write(f"{id}\n")
    print(f"\n✓ Forward + Backward solved, Bidirectional missed: {len(patterns['forward_backward'])} IDs")
    print(f"  → ids_forward_backward_only.txt")

    # Forward + Bidirectional solved, Backward missed
    with open(output_dir / "ids_forward_bidirectional_only.txt", "w") as f:
        for id in sorted(patterns["forward_bidir"]):
            f.write(f"{id}\n")
    print(f"\n✓ Forward + Bidirectional solved, Backward missed: {len(patterns['forward_bidir'])} IDs")
    print(f"  → ids_forward_bidirectional_only.txt")

    # Backward + Bidirectional solved, Forward missed
    with open(output_dir / "ids_backward_bidirectional_only.txt", "w") as f:
        for id in sorted(patterns["backward_bidir"]):
            f.write(f"{id}\n")
    print(f"\n✓ Backward + Bidirectional solved, Forward missed: {len(patterns['backward_bidir'])} IDs")
    print(f"  → ids_backward_bidirectional_only.txt")

    # 3. Direction-specific (only one solved)
    print(f"\n✓ DIRECTION-SPECIFIC (only 1 solved):")

    with open(output_dir / "ids_only_forward.txt", "w") as f:
        for id in sorted(patterns["only_forward"]):
            f.write(f"{id}\n")
    print(f"  - Only Forward: {len(patterns['only_forward'])} IDs → ids_only_forward.txt")

    with open(output_dir / "ids_only_backward.txt", "w") as f:
        for id in sorted(patterns["only_backward"]):
            f.write(f"{id}\n")
    print(f"  - Only Backward: {len(patterns['only_backward'])} IDs → ids_only_backward.txt")

    with open(output_dir / "ids_only_bidirectional.txt", "w") as f:
        for id in sorted(patterns["only_bidir"]):
            f.write(f"{id}\n")
    print(f"  - Only Bidirectional: {len(patterns['only_bidir'])} IDs → ids_only_bidirectional.txt")

    # 4. Per-direction success/failure lists
    print(f"\n✓ PER-DIRECTION LISTS:")

    # Forward: what it solved vs missed
    fwd_failed = (patterns["only_backward"] | patterns["only_bidir"]
                 | patterns["backward_bidir"] | patterns["none"])
    with open(output_dir / "ids_forward_success.txt", "w") as f:
        for id in sorted(fwd_success):
            f.write(f"{id}\n")
    with open(output_dir / "ids_forward_failure.txt", "w") as f:
        for id in sorted(fwd_failed):
            f.write(f"{id}\n")
    print(f"  - Forward success: {len(fwd_success)} IDs → ids_forward_success.txt")
    print(f"  - Forward failure: {len(fwd_failed)} IDs → ids_forward_failure.txt")

    # Backward: what it solved vs missed
    back_failed = (patterns["only_forward"] | patterns["only_bidir"]
                  | patterns["forward_bidir"] | patterns["none"])
    with open(output_dir / "ids_backward_success.txt", "w") as f:
        for id in sorted(back_success):
            f.write(f"{id}\n")
    with open(output_dir / "ids_backward_failure.txt", "w") as f:
        for id in sorted(back_failed):
            f.write(f"{id}\n")
    print(f"  - Backward success: {len(back_success)} IDs → ids_backward_success.txt")
    print(f"  - Backward failure: {len(back_failed)} IDs → ids_backward_failure.txt")

    # Bidirectional: what it solved vs missed
    bidir_failed = (patterns["only_forward"] | patterns["only_backward"]
                   | patterns["forward_backward"] | patterns["none"])
    with open(output_dir / "ids_bidirectional_success.txt", "w") as f:
        for id in sorted(bidir_success):
            f.write(f"{id}\n")
    with open(output_dir / "ids_bidirectional_failure.txt", "w") as f:
        for id in sorted(bidir_failed):
            f.write(f"{id}\n")
    print(f"  - Bidirectional success: {len(bidir_success)} IDs → ids_bidirectional_success.txt")
    print(f"  - Bidirectional failure: {len(bidir_failed)} IDs → ids_bidirectional_failure.txt")

    # 5. Comparison lists (what one solved that another didn't)
    print(f"\n✓ COMPARISON LISTS (A solved but B missed):")

    # Forward solved, Backward missed
    fwd_not_back = fwd_success - back_success
    with open(output_dir / "ids_forward_beats_backward.txt", "w") as f:
        for id in sorted(fwd_not_back):
            f.write(f"{id}\n")
    print(f"  - Forward solved, Backward missed: {len(fwd_not_back)} IDs → ids_forward_beats_backward.txt")

    # Backward solved, Forward missed
    back_not_fwd = back_success - fwd_success
    with open(output_dir / "ids_backward_beats_forward.txt", "w") as f:
        for id in sorted(back_not_fwd):
            f.write(f"{id}\n")
    print(f"  - Backward solved, Forward missed: {len(back_not_fwd)} IDs → ids_backward_beats_forward.txt")

    # Bidirectional solved, Forward missed
    bidir_not_fwd = bidir_success - fwd_success
    with open(output_dir / "ids_bidirectional_beats_forward.txt", "w") as f:
        for id in sorted(bidir_not_fwd):
            f.write(f"{id}\n")
    print(f"  - Bidirectional solved, Forward missed: {len(bidir_not_fwd)} IDs → ids_bidirectional_beats_forward.txt")

    # Bidirectional solved, Backward missed
    bidir_not_back = bidir_success - back_success
    with open(output_dir / "ids_bidirectional_beats_backward.txt", "w") as f:
        for id in sorted(bidir_not_back):
            f.write(f"{id}\n")
    print(f"  - Bidirectional solved, Backward missed: {len(bidir_not_back)} IDs → ids_bidirectional_beats_backward.txt")

    # Forward + Backward solved, Bidirectional missed
    fwd_back_not_bidir = fwd_success & back_success - bidir_success
    with open(output_dir / "ids_forward_and_backward_beat_bidirectional.txt", "w") as f:
        for id in sorted(fwd_back_not_bidir):
            f.write(f"{id}\n")
    print(f"  - Forward AND Backward solved, Bidirectional missed: {len(fwd_back_not_bidir)} IDs → ids_forward_and_backward_beat_bidirectional.txt")

    # 6. All failures (none succeeded)
    with open(output_dir / "ids_all_failed.txt", "w") as f:
        for id in sorted(patterns["none"]):
            f.write(f"{id}\n")
    print(f"\n✗ All 3 failed: {len(patterns['none'])} IDs → ids_all_failed.txt")

    print(f"\n💡 Use these with --issue-ids-file or --issue-ids $(cat <file>)")
    print(f"   Example: --issue-ids-file analysis/ids_only_forward.txt")


def print_summary(
    data: Dict[str, Dict[str, Dict]],
    patterns: Dict[str, Set[str]],
    counts: Dict[str, int]
):
    """Print brief summary report."""

    print("\n" + "="*80)
    print("DIRECTION COMPARISON SUMMARY")
    print("="*80)

    print(f"\nTotal instances: {counts['total']}")
    print("\nResults by direction:")
    for direction in ["forward", "backward", "bidirectional"]:
        success = sum(1 for r in data[direction].values() if r["conclusion"] == "success")
        total = len(data[direction])
        print(f"  {direction:15} SUCCESS: {success:3} ({success/max(total,1)*100:5.1f}%)")

    print("\n" + "-"*80)
    print(f"✓ All 3 directions succeeded:     {counts['all_three']:3} ({counts['all_three']/counts['total']*100:5.1f}%)")
    print(f"✓ Only Forward succeeded:          {counts['only_forward']:3}")
    print(f"✓ Only Backward succeeded:         {counts['only_backward']:3}")
    print(f"✓ Only Bidirectional succeeded:    {counts['only_bidir']:3}")
    print(f"✗ None succeeded:                  {counts['none']:3} ({counts['none']/counts['total']*100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare decomposition direction results and generate ID lists"
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
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Output directory for ID lists and JSON"
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

    # Print summary
    print_summary(data, patterns, counts)

    # Save ID lists
    save_id_lists(patterns, args.output_dir)

    # Save JSON
    output_json = args.output_dir / "direction_comparison.json"
    output_data = {
        "summary": counts,
        "patterns": {k: sorted(list(v)) for k, v in patterns.items()}
    }
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n📄 JSON saved to: {output_json}")

    return 0


if __name__ == "__main__":
    exit(main())
