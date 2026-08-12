#!/usr/bin/env python3
"""
Filter predictions to only keep entries whose ID is in eval_issue_ids.json
"""
import json
from pathlib import Path

def filter_predictions(predictions_file: str, eval_ids_file: str):
    """
    Filter predictions to only keep entries in eval set.

    Args:
        predictions_file: Path to predictions.json
        eval_ids_file: Path to eval_issue_ids.json
    """
    pred_path = Path(predictions_file)
    eval_path = Path(eval_ids_file)

    # Read eval IDs
    with open(eval_path) as f:
        eval_ids = set(json.load(f))

    print(f"✓ Loaded {len(eval_ids)} valid IDs from {eval_path.name}")

    # Read predictions
    with open(pred_path) as f:
        predictions = json.load(f)

    print(f"✓ Loaded {len(predictions)} predictions from {pred_path.name}")

    # Filter predictions
    filtered = []
    removed = []

    for entry in predictions:
        entry_id = str(entry.get("id") or entry.get("instance_id") or entry.get("issue_id") or "")

        if entry_id in eval_ids:
            filtered.append(entry)
        else:
            removed.append(entry_id)

    print(f"\n📊 Results:")
    print(f"  Kept: {len(filtered)} entries (in eval set)")
    print(f"  Removed: {len(removed)} entries (not in eval set)")

    if removed:
        print(f"\n🗑️  Removed IDs (first 20):")
        for rid in removed[:20]:
            print(f"    - {rid}")
        if len(removed) > 20:
            print(f"    ... and {len(removed) - 20} more")

    # Create backup
    import shutil
    backup_path = pred_path.with_suffix('.json.before_filter')
    shutil.copy2(pred_path, backup_path)
    print(f"\n📦 Backup created: {backup_path}")

    # Save filtered predictions
    with open(pred_path, 'w') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Filtered predictions saved: {pred_path}")
    print(f"   Total entries: {len(filtered)}")

if __name__ == "__main__":
    predictions_file = "results/codex/baseline_gpt-5_4-mini/predictions.json"
    eval_ids_file = "data/eval_issue_ids.json"

    filter_predictions(predictions_file, eval_ids_file)
