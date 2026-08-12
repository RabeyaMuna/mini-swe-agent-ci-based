#!/usr/bin/env python3
"""
Merge two prediction files, remove duplicates and empty diffs.
"""
import json
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def merge_predictions(file1_path: str, file2_path: str, output_path: str):
    """
    Merge two prediction files, remove duplicates and empty diffs.

    Args:
        file1_path: First predictions file
        file2_path: Second predictions file
        output_path: Where to save merged result
    """
    file1 = PROJECT_ROOT / file1_path
    file2 = PROJECT_ROOT / file2_path
    output = PROJECT_ROOT / output_path

    # Load both files
    all_predictions = []

    for fpath in [file1, file2]:
        if not fpath.exists():
            print(f"⚠️  {fpath.relative_to(PROJECT_ROOT)} not found, skipping")
            continue

        with open(fpath) as f:
            data = json.load(f)

        if isinstance(data, list):
            all_predictions.extend(data)
            print(f"✓ Loaded {len(data)} entries from {fpath.name}")
        elif isinstance(data, dict):
            all_predictions.append(data)
            print(f"✓ Loaded 1 entry from {fpath.name}")

    print(f"\nTotal before processing: {len(all_predictions)}")

    # Step 1: Remove duplicates (keep first occurrence)
    seen_ids = set()
    unique = []
    duplicates = 0

    for entry in all_predictions:
        entry_id = str(entry.get("id") or entry.get("instance_id") or entry.get("issue_id") or "")
        if entry_id and entry_id in seen_ids:
            duplicates += 1
            continue
        if entry_id:
            seen_ids.add(entry_id)
        unique.append(entry)

    print(f"After deduplication: {len(unique)} entries ({duplicates} duplicates removed)")

    # Step 2: Remove empty diffs
    clean = []
    empty_count = 0

    for entry in unique:
        diff_value = entry.get("model_patch") or entry.get("diff") or entry.get("patch") or ""

        if diff_value and str(diff_value).strip():
            clean.append(entry)
        else:
            empty_count += 1
            entry_id = entry.get("id") or entry.get("instance_id") or "unknown"
            print(f"  Removed {entry_id}: empty diff")

    print(f"After removing empty diffs: {len(clean)} entries ({empty_count} removed)")

    # Create backup if output exists
    if output.exists():
        backup = output.with_suffix(f'.json.backup_{int(time.time())}')
        shutil.copy2(output, backup)
        print(f"\n📦 Backup: {backup.name}")

    # Save merged file
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Merged file saved: {output.relative_to(PROJECT_ROOT)}")
    print(f"   Final count: {len(clean)} entries")

    # Show some stats
    if clean:
        ids = [str(e.get("id", "?")) for e in clean[:10]]
        print(f"   First 10 IDs: {', '.join(ids)}")


if __name__ == "__main__":
    merge_predictions(
        file1_path="results/codex/predictions.json",
        file2_path="results/codex/baseline_gpt-5_4-mini/predictions.json",
        output_path="results/codex/baseline_gpt-5_4-mini/predictions.json"
    )
