#!/usr/bin/env python3
"""
Filter out entries with empty diffs from JSONL files.
"""

import json
import sys
from pathlib import Path


def filter_empty_diffs(input_path: Path, output_path: Path = None, backup: bool = True):
    """
    Remove entries with empty diff fields from a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output file (default: overwrite input)
        backup: Create backup before overwriting (default: True)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}", file=sys.stderr)
        return 1

    # Read all entries
    entries = []
    empty_count = 0
    total_count = 0

    print(f"📖 Reading {input_path}...")

    try:
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    total_count += 1

                    # Check if diff exists and is not empty
                    diff_value = entry.get('diff', '')
                    if diff_value and (isinstance(diff_value, str) and diff_value.strip()):
                        entries.append(entry)
                    else:
                        empty_count += 1

                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                    continue

    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        return 1

    # Prepare output path
    if output_path is None:
        output_path = input_path

        # Create backup
        if backup:
            backup_path = input_path.with_suffix(input_path.suffix + '.backup')
            input_path.rename(backup_path)
            print(f"💾 Backup created: {backup_path}")

    # Write filtered entries
    print(f"✍️  Writing filtered entries to {output_path}...")

    try:
        with open(output_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
    except Exception as e:
        print(f"❌ Error writing file: {e}", file=sys.stderr)
        return 1

    # Summary
    kept_count = len(entries)
    print()
    print("=" * 60)
    print(f"✅ Filtering complete!")
    print(f"   Total entries:   {total_count}")
    print(f"   Empty diffs:     {empty_count} ({100*empty_count/total_count:.1f}%)")
    print(f"   Kept entries:    {kept_count} ({100*kept_count/total_count:.1f}%)")
    print("=" * 60)

    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Filter out entries with empty diffs from JSONL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter in place (creates backup)
  python filter_empty_diffs_jsonl.py input.jsonl

  # Filter to new file (no backup)
  python filter_empty_diffs_jsonl.py input.jsonl -o output.jsonl

  # Filter in place without backup (not recommended)
  python filter_empty_diffs_jsonl.py input.jsonl --no-backup
        """
    )

    parser.add_argument('input', type=Path, help='Input JSONL file')
    parser.add_argument('-o', '--output', type=Path, help='Output file (default: overwrite input)')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup when overwriting')

    args = parser.parse_args()

    return filter_empty_diffs(
        args.input,
        args.output,
        backup=not args.no_backup
    )


if __name__ == '__main__':
    sys.exit(main())
