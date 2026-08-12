#!/usr/bin/env python3
"""
Append predictions from one file to another WITHOUT removing existing data.
Also remove entries with empty diff and filter by eval IDs.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def append_predictions(
    source_file: str,
    target_file: str,
    eval_ids_file: str,
):
    """
    Append predictions from source to target, keeping ALL existing data.

    Args:
        source_file: File with new predictions to add
        target_file: File to append to (existing data preserved)
        eval_ids_file: File with valid eval IDs for filtering
    """
    source_path = PROJECT_ROOT / source_file
    target_path = PROJECT_ROOT / target_file
    eval_path = PROJECT_ROOT / eval_ids_file

    # Load eval IDs
    with open(eval_path) as f:
        eval_ids = set(json.load(f))
    print(f"✓ Loaded {len(eval_ids)} valid eval IDs")

    # Load existing target data (preserve it!)
    existing_predictions = []
    if target_path.exists():
        with open(target_path) as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_predictions = existing_data
            elif isinstance(existing_data, dict):
                existing_predictions = [existing_data]
        print(f"✓ Loaded {len(existing_predictions)} EXISTING predictions from {target_path.name}")
    else:
        print(f"⚠️  Target file doesn't exist yet, will create new one")

    # Load new source data
    new_predictions = []
    if source_path.exists():
        with open(source_path) as f:
            source_data = json.load(f)
            if isinstance(source_data, list):
                new_predictions = source_data
            elif isinstance(source_data, dict):
                new_predictions = [source_data]
        print(f"✓ Loaded {len(new_predictions)} NEW predictions from {source_path.name}")
    else:
        print(f"⚠️  Source file doesn't exist, nothing to add")
        return

    # Combine: existing + new (NO removal)
    all_predictions = existing_predictions + new_predictions
    print(f"\n📊 Total after combining: {len(all_predictions)} entries")

    # Remove duplicates by ID (keep first occurrence - existing data wins)
    seen_ids = set()
    unique_predictions = []
    duplicate_count = 0

    for entry in all_predictions:
        entry_id = str(entry.get("id") or entry.get("instance_id") or entry.get("issue_id") or "")
        if entry_id and entry_id in seen_ids:
            duplicate_count += 1
            continue
        if entry_id:
            seen_ids.add(entry_id)
        unique_predictions.append(entry)

    if duplicate_count > 0:
        print(f"  Removed {duplicate_count} duplicate IDs (existing data preserved)")
    print(f"  After deduplication: {len(unique_predictions)} entries")

    # Remove entries with empty diff
    filtered = []
    removed_empty = 0

    for entry in unique_predictions:
        diff_value = entry.get("model_patch") or entry.get("diff") or entry.get("patch") or ""
        if diff_value and str(diff_value).strip():
            filtered.append(entry)
        else:
            removed_empty += 1
            entry_id = entry.get("id") or entry.get("instance_id") or "unknown"
            print(f"  Removed {entry_id}: empty diff")

    if removed_empty > 0:
        print(f"  Removed {removed_empty} entries with empty diff")
    print(f"  After removing empty diffs: {len(filtered)} entries")

    # Filter by eval IDs
    final = []
    removed_non_eval = 0

    for entry in filtered:
        entry_id = str(entry.get("id") or entry.get("instance_id") or entry.get("issue_id") or "")
        if entry_id in eval_ids:
            final.append(entry)
        else:
            removed_non_eval += 1

    if removed_non_eval > 0:
        print(f"  Removed {removed_non_eval} entries not in eval set")
    print(f"  Final count: {len(final)} entries")

    # Create backup of existing file
    if target_path.exists():
        import shutil
        import time
        backup_path = target_path.with_suffix(f'.json.backup_{int(time.time())}')
        shutil.copy2(target_path, backup_path)
        print(f"\n📦 Backup created: {backup_path.name}")

    # Save result
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, 'w') as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to: {target_path}")
    print(f"   Total entries: {len(final)}")
    print(f"   Coverage: {len(final)}/{len(eval_ids)} = {len(final)*100//len(eval_ids)}%")


if __name__ == "__main__":
    print("🔄 Appending predictions (preserving existing data)...\n")

    append_predictions(
        source_file="results/codex/predictions.json",
        target_file="results/codex/baseline_gpt-5_4-mini/predictions.json",
        eval_ids_file="data/eval_issue_ids.json",
    )
