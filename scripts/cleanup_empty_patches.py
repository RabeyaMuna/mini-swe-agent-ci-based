#!/usr/bin/env python3
"""
Remove entries with empty patches from prediction JSON files.

Processes preds.json or predictions.json files and removes any entries
where the 'diff' or 'patch' field is empty or contains only whitespace.

Usage:
    python cleanup_empty_patches.py <file_path>
    python cleanup_empty_patches.py <directory>  # processes all matching files recursively
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def has_empty_patch(entry: Dict[str, Any]) -> bool:
    """Check if an entry has an empty patch/diff."""
    # Check common field names for patches
    for field in ['diff', 'patch', 'model_patch']:
        if field in entry:
            value = entry[field]
            # Empty if None, empty string, or only whitespace
            if value is None or (isinstance(value, str) and not value.strip()):
                return True
    return False


def cleanup_predictions_file(file_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Remove entries with empty patches from a predictions file.

    Returns:
        (total_count, removed_count)
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}", file=sys.stderr)
        return 0, 0

    if isinstance(data, dict):
        # Format: {instance_id: {id, sha_fail, diff, ...}, ...}
        original_count = len(data)
        cleaned_data = {
            key: value
            for key, value in data.items()
            if not has_empty_patch(value)
        }
        removed_count = original_count - len(cleaned_data)

    elif isinstance(data, list):
        # Format: [{instance_id, diff, ...}, ...]
        original_count = len(data)
        cleaned_data = [
            entry
            for entry in data
            if not has_empty_patch(entry)
        ]
        removed_count = original_count - len(cleaned_data)

    else:
        print(f"⚠️  Unknown format in {file_path}", file=sys.stderr)
        return 0, 0

    if removed_count > 0:
        if dry_run:
            print(f"🔍 [DRY RUN] Would remove {removed_count}/{original_count} empty patches from {file_path}")
        else:
            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            file_path.rename(backup_path)

            # Write cleaned data
            with open(file_path, 'w') as f:
                json.dump(cleaned_data, f, indent=2)

            print(f"✅ Removed {removed_count}/{original_count} empty patches from {file_path}")
            print(f"   Backup saved to {backup_path}")
    else:
        print(f"✓  No empty patches in {file_path} ({original_count} entries)")

    return original_count, removed_count


def find_prediction_files(path: Path, verbose: bool = False) -> list[Path]:
    """Find all preds.json and predictions.json files in a directory tree."""
    if path.is_file():
        return [path]

    if verbose:
        print(f"🔍 Scanning {path} for prediction files...")

    files = []
    for pattern in ['**/preds.json', '**/predictions.json']:
        if verbose:
            print(f"   Looking for {pattern}...")
        matches = list(path.glob(pattern))
        if verbose and matches:
            print(f"   Found {len(matches)} file(s)")
        files.extend(matches)

    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(
        description='Remove entries with empty patches from prediction files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean a specific file
  python cleanup_empty_patches.py results/baseline_gpt-5.4-mini/preds.json

  # Clean all prediction files in a directory tree
  python cleanup_empty_patches.py results/

  # Dry run to see what would be removed
  python cleanup_empty_patches.py results/ --dry-run
        """
    )
    parser.add_argument(
        'path',
        type=Path,
        help='Path to a prediction file or directory containing prediction files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually modifying files'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files (not recommended)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed progress information'
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"❌ Error: Path does not exist: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Find all prediction files
    files = find_prediction_files(args.path, verbose=args.verbose)

    if not files:
        print(f"⚠️  No prediction files found in {args.path}")
        print("   Looking for: preds.json, predictions.json")
        sys.exit(0)

    print(f"\n📋 Found {len(files)} prediction file(s)")
    print()

    # Process each file
    total_entries = 0
    total_removed = 0

    for idx, file_path in enumerate(files, 1):
        if args.verbose or len(files) > 1:
            print(f"[{idx}/{len(files)}] Processing {file_path.relative_to(args.path if args.path.is_dir() else args.path.parent)}...")
        count, removed = cleanup_predictions_file(file_path, dry_run=args.dry_run)
        total_entries += count
        total_removed += removed
        if args.verbose:
            print()

    # Summary
    print()
    print("=" * 60)
    if args.dry_run:
        print(f"🔍 DRY RUN: Would remove {total_removed}/{total_entries} empty patches")
        print("   Run without --dry-run to actually clean the files")
    else:
        print(f"✅ Cleaned {len(files)} file(s)")
        print(f"   Removed {total_removed}/{total_entries} empty patches")
        if total_removed > 0 and not args.no_backup:
            print(f"   Backups saved with .backup extension")
    print("=" * 60)


if __name__ == '__main__':
    main()
