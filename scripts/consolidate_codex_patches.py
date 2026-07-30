#!/usr/bin/env python3
"""
Consolidate Codex patches into predictions files.

By default this creates one predictions.json for a single ablation directory.
Use --all to create one output-root predictions.json across every ablation.
"""

import argparse
import json
from pathlib import Path
from typing import Any

def consolidate_patches(results_dir: Path, output_file: Path) -> dict[str, Any]:
    """
    Consolidate all patches from Codex results into a single predictions file.

    Args:
        results_dir: Directory containing Codex results (e.g., results/codex/baseline/)
        output_file: Output predictions.json file

    Returns:
        Statistics about the consolidation
    """

    predictions = []
    stats = {
        "total_issues": 0,
        "patches_generated": 0,
        "empty_patches": 0,
        "total_patch_bytes": 0,
    }

    # Find all result.json files
    for result_file in sorted(results_dir.rglob("result.json")):
        issue_dir = result_file.parent

        # Load result metadata
        with open(result_file) as f:
            result = json.load(f)

        stats["total_issues"] += 1

        # Load patch if it exists
        patch_file = issue_dir / "patch.diff"
        if patch_file.exists():
            patch_content = patch_file.read_text(encoding="utf-8")
        else:
            patch_content = ""

        # Build prediction entry
        prediction = {
            "id": result["id"],
            "sha_fail": result["sha_fail"],
            "repo": result["repo"],
            "diff": patch_content,
            "ablation": result.get("ablation", "unknown"),
            "patch_generated": result["patch_generated"],
            "patch_bytes": result["patch_bytes"],
            "changed_files": result["changed_files"],
            "verification_passed": result.get("verification_passed"),
        }

        predictions.append(prediction)

        if result["patch_generated"]:
            stats["patches_generated"] += 1
            stats["total_patch_bytes"] += result["patch_bytes"]
        else:
            stats["empty_patches"] += 1

    # Write consolidated file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    return stats


def consolidate_all(output_root: Path, output_file: Path) -> dict[str, Any]:
    predictions = []
    stats = {
        "total_issues": 0,
        "patches_generated": 0,
        "empty_patches": 0,
        "total_patch_bytes": 0,
    }

    for result_file in sorted(output_root.glob("*/*/result.json")):
        issue_dir = result_file.parent
        with open(result_file, encoding="utf-8") as f:
            result = json.load(f)

        patch_file = issue_dir / "patch.diff"
        patch_content = patch_file.read_text(encoding="utf-8") if patch_file.exists() else ""

        prediction = {
            "instance_id": result["id"],
            "model_patch": patch_content,
            "model_name_or_path": result.get("model_name_or_path", "codex"),
            "repo": result["repo"],
            "sha_fail": result["sha_fail"],
            "ablation": result.get("ablation", "unknown"),
            "patch_generated": result["patch_generated"],
            "patch_bytes": result["patch_bytes"],
            "changed_files": result["changed_files"],
            "verification_passed": result.get("verification_passed"),
        }
        predictions.append(prediction)

        stats["total_issues"] += 1
        if result["patch_generated"]:
            stats["patches_generated"] += 1
            stats["total_patch_bytes"] += result["patch_bytes"]
        else:
            stats["empty_patches"] += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ablation",
        nargs="?",
        help="Ablation directory name such as baseline, l1, l1_l2, or l1_l2_l3.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/codex"),
        help="Root containing Codex result directories.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Consolidate all ablation directories into <output-root>/predictions.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        output_file = args.output_root / "predictions.json"
        print(f"Consolidating all patches from {args.output_root}...")
        stats = consolidate_all(args.output_root, output_file)
        label = "ALL"
    else:
        if not args.ablation:
            raise SystemExit(
                "Usage: python3 scripts/consolidate_codex_patches.py <ablation>\n"
                "   or: python3 scripts/consolidate_codex_patches.py --all"
            )
        ablation = args.ablation
        results_dir = args.output_root / ablation
        output_file = results_dir / "predictions.json"

        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            raise SystemExit(1)

        print(f"Consolidating patches from {results_dir}...")
        stats = consolidate_patches(results_dir, output_file)
        label = ablation.upper()

    print(f"\n{'='*60}")
    print(f"Consolidation Complete - {label}")
    print(f"{'='*60}")
    print(f"Total Issues:          {stats['total_issues']}")
    print(f"Patches Generated:     {stats['patches_generated']}")
    print(f"Empty Patches:         {stats['empty_patches']}")
    print(f"Total Patch Size:      {stats['total_patch_bytes']:,} bytes")
    print(f"\nOutput saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
