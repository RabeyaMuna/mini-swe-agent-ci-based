#!/usr/bin/env python3
"""
Remove entries with empty diff from ALL prediction files in results/
"""
import json
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def clean_predictions_file(file_path: Path):
    """Remove empty diff entries from one predictions file."""
    try:
        with open(file_path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"  ⚠️  {file_path.relative_to(PROJECT_ROOT)}: Not a list, skipping")
            return

        # Filter out entries with empty diff
        cleaned = []
        removed = []

        for entry in data:
            # Check all possible diff field names
            diff_value = (
                entry.get("model_patch") or
                entry.get("diff") or
                entry.get("patch") or
                ""
            )

            # Keep if diff is not empty
            if diff_value and str(diff_value).strip():
                cleaned.append(entry)
            else:
                entry_id = entry.get("id") or entry.get("instance_id") or entry.get("issue_id") or "unknown"
                removed.append(str(entry_id))

        # Only update if something was removed
        if removed:
            # Create backup
            backup_path = file_path.with_suffix(f'.json.backup_{int(time.time())}')
            shutil.copy2(file_path, backup_path)

            # Save cleaned file
            with open(file_path, 'w') as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)

            print(f"  ✓ {file_path.relative_to(PROJECT_ROOT)}")
            print(f"    Before: {len(data)} entries")
            print(f"    After:  {len(cleaned)} entries")
            print(f"    Removed: {len(removed)} ({', '.join(removed[:5])}{'...' if len(removed) > 5 else ''})")
            print(f"    Backup: {backup_path.name}")
        else:
            print(f"  ✓ {file_path.relative_to(PROJECT_ROOT)}: No empty diffs found")

    except Exception as e:
        print(f"  ✗ {file_path.relative_to(PROJECT_ROOT)}: Error: {e}")


def main():
    print("🧹 Removing empty diff entries from all prediction files...\n")

    # Find all predictions files
    prediction_files = []
    for pattern in ["predictions.json", "preds.json"]:
        prediction_files.extend(RESULTS_DIR.rglob(pattern))

    if not prediction_files:
        print("No prediction files found in results/")
        return

    print(f"Found {len(prediction_files)} prediction files:\n")

    total_removed = 0
    for file_path in sorted(prediction_files):
        clean_predictions_file(file_path)

    print(f"\n✅ Cleanup complete!")
    print(f"   Processed {len(prediction_files)} files")


if __name__ == "__main__":
    main()
