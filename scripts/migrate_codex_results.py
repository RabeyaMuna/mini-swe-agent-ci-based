#!/usr/bin/env python3
"""
Migrate existing Codex results to include model name in directory structure.

Old: results/codex/baseline/43/
New: results/codex/baseline_minimax2_5/43/
"""

import json
import shutil
from pathlib import Path


def migrate_results(
    output_root: Path,
    ablation: str,
    old_dir: str,
    new_dir: str,
) -> None:
    """Migrate results from old structure to new structure."""
    old_path = output_root / old_dir
    new_path = output_root / new_dir

    if not old_path.exists():
        print(f"[SKIP] {old_path} does not exist")
        return

    if new_path.exists():
        print(f"[SKIP] {new_path} already exists")
        return

    print(f"[MIGRATE] {old_path} -> {new_path}")

    # Move the entire directory
    shutil.move(str(old_path), str(new_path))

    print(f"[DONE] Migrated {ablation}")


def main():
    output_root = Path("results/codex")

    # Define migrations
    migrations = [
        ("baseline", "baseline", "baseline_minimax2_5"),
        # Add more migrations if needed
        # ("l1", "l1", "l1_minimax2_5"),
        # ("l1_l2", "l1_l2", "l1_l2_minimax2_5"),
        # ("l1_l2_l3", "l1_l2_l3", "l1_l2_l3_minimax2_5"),
    ]

    print(f"Migrating Codex results in {output_root}")
    print("=" * 80)

    for ablation, old_dir, new_dir in migrations:
        migrate_results(output_root, ablation, old_dir, new_dir)

    print("\n" + "=" * 80)
    print("Migration complete!")

    # Show new structure
    print("\nNew structure:")
    for ablation, _, new_dir in migrations:
        new_path = output_root / new_dir
        if new_path.exists():
            issue_count = len(list(new_path.glob("*/result.json")))
            predictions_file = new_path / "predictions.json"
            has_predictions = "✅" if predictions_file.exists() else "❌"
            print(f"  {new_path}/")
            print(f"    Issues: {issue_count}")
            print(f"    predictions.json: {has_predictions}")


if __name__ == "__main__":
    main()
