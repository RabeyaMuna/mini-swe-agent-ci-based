#!/usr/bin/env python3
"""
Merge two prediction files without deduplication and remove empty diff entries.
"""
import json
from pathlib import Path
import sys

def merge_predictions(file1: str, file2: str, output: str):
    """
    Merge two prediction files without deduplication, remove empty diffs.

    Args:
        file1: First prediction file path
        file2: Second prediction file path
        output: Output file path
    """
    file1_path = Path(file1)
    file2_path = Path(file2)
    output_path = Path(output)

    # Read both files
    all_predictions = []

    for fpath in [file1_path, file2_path]:
        if not fpath.exists():
            print(f"⚠️  {fpath} does not exist, skipping")
            continue

        try:
            with open(fpath) as f:
                data = json.load(f)

            # Handle both list and dict formats
            if isinstance(data, list):
                all_predictions.extend(data)
                print(f"✓ Loaded {len(data)} entries from {fpath.name}")
            elif isinstance(data, dict):
                all_predictions.append(data)
                print(f"✓ Loaded 1 entry from {fpath.name}")
            else:
                print(f"⚠️  Unknown format in {fpath.name}, skipping")

        except Exception as e:
            print(f"✗ Error reading {fpath}: {e}")
            continue

    print(f"\nTotal entries before filtering: {len(all_predictions)}")

    # Remove entries with empty diff
    filtered = []
    removed_count = 0

    for entry in all_predictions:
        # Check if diff field exists and is not empty
        diff_value = entry.get("model_patch") or entry.get("diff") or entry.get("patch") or ""

        # Remove if diff is empty string or None
        if diff_value and str(diff_value).strip():
            filtered.append(entry)
        else:
            removed_count += 1
            # Show which entry was removed
            entry_id = entry.get("instance_id") or entry.get("id") or entry.get("issue_id") or "unknown"
            print(f"  Removed entry {entry_id}: empty diff")

    print(f"\nRemoved {removed_count} entries with empty diff")
    print(f"Final count: {len(filtered)} entries")

    # Create backup if output file exists
    if output_path.exists():
        backup_path = output_path.with_suffix('.json.backup')
        import shutil
        shutil.copy2(output_path, backup_path)
        print(f"\n📦 Backup created: {backup_path}")

    # Save merged file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Merged file saved: {output_path}")
    print(f"   Total entries: {len(filtered)}")

if __name__ == "__main__":
    file1 = "results/codex/predictions.json"
    file2 = "results/codex/baseline_gpt-5_4-mini/predictions.json"
    output = "results/codex/baseline_gpt-5_4-mini/predictions.json"

    merge_predictions(file1, file2, output)
