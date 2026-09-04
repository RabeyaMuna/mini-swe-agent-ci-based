#!/usr/bin/env python3
"""
Clean binary patches from predictions JSON files to reduce size
"""

import json
import re
import sys
from pathlib import Path


def is_binary_patch(diff_text: str) -> bool:
    """Check if diff contains binary patch data"""
    if not diff_text:
        return False
    return "GIT binary patch" in diff_text or "Binary files" in diff_text


def get_binary_info(diff_text: str) -> str:
    """Extract binary file information from diff"""
    lines = diff_text.split('\n')
    info = []

    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            info.append(line)
        elif line.startswith('new file mode'):
            info.append(line)
        elif line.startswith('index'):
            info.append(line)
        elif 'Binary files' in line:
            info.append(line)
        elif line.startswith('literal'):
            # Extract size
            match = re.search(r'literal (\d+)', line)
            if match:
                size_bytes = int(match.group(1))
                size_mb = size_bytes / (1024 * 1024)
                info.append(f"[BINARY FILE: {size_mb:.2f}MB]")
            break

    return '\n'.join(info) if info else "[BINARY FILE CHANGE]"


def clean_predictions_file(input_file: Path, output_file: Path = None):
    """
    Clean binary patches from predictions JSON

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (default: input_file with .cleaned.json suffix)
    """
    if output_file is None:
        output_file = input_file.with_suffix('.cleaned.json')

    print(f"Reading {input_file}...")
    print(f"File size: {input_file.stat().st_size / (1024*1024):.2f}MB")

    # Read JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    total_patches = len(data)
    binary_patches = 0
    original_size = 0
    cleaned_size = 0

    print(f"\nProcessing {total_patches} patches...")

    # Process each patch
    for patch_id, patch_data in data.items():
        diff = patch_data.get('diff', '')
        original_size += len(diff)

        if is_binary_patch(diff):
            binary_patches += 1
            # Replace with metadata only
            binary_info = get_binary_info(diff)
            patch_data['diff'] = binary_info
            patch_data['_binary_removed'] = True
            print(f"  Patch {patch_id}: Binary patch removed")

        cleaned_size += len(patch_data.get('diff', ''))

    # Write cleaned file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    output_size = output_file.stat().st_size / (1024*1024)
    input_size = input_file.stat().st_size / (1024*1024)
    reduction = ((input_size - output_size) / input_size) * 100

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total patches: {total_patches}")
    print(f"  Binary patches removed: {binary_patches}")
    print(f"  Original file: {input_size:.2f}MB")
    print(f"  Cleaned file: {output_size:.2f}MB")
    print(f"  Size reduction: {reduction:.1f}%")
    print(f"{'='*60}")

    if output_size < 100:
        print(f"✓ File is now under 100MB and can be pushed to GitHub!")
    else:
        print(f"⚠ File is still {output_size:.2f}MB - may need Git LFS")

    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_binary_patches.py <predictions.json> [output.json]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_file.exists():
        print(f"Error: File {input_file} not found")
        sys.exit(1)

    clean_predictions_file(input_file, output_file)
