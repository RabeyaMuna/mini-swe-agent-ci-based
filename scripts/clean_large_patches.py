#!/usr/bin/env python3
"""
Clean large and unnecessary patches from predictions JSON files
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


def is_large_file(diff_text: str, threshold_kb: int = 1000) -> bool:
    """Check if diff is larger than threshold"""
    return len(diff_text) / 1024 > threshold_kb


def contains_generated_file(diff_text: str) -> bool:
    """Check if diff contains generated/coverage files"""
    patterns = [
        r"coverage.*\.json",
        r"\.coverage",
        r"htmlcov/",
        r"node_modules/",
        r"package-lock\.json",
        r"yarn\.lock",
        r"poetry\.lock",
    ]
    for pattern in patterns:
        if re.search(pattern, diff_text):
            return True
    return False


def get_file_summary(diff_text: str) -> str:
    """Extract a summary of the diff"""
    lines = diff_text.split('\n')
    summary_lines = []

    for line in lines[:20]:  # First 20 lines
        if line.startswith('diff --git'):
            summary_lines.append(line)
        elif line.startswith('index'):
            summary_lines.append(line)
        elif line.startswith('new file') or line.startswith('deleted file'):
            summary_lines.append(line)
        elif line.startswith('---') or line.startswith('+++'):
            summary_lines.append(line)
        elif line.startswith('@@'):
            summary_lines.append(line)
            break

    # Add summary note
    size_kb = len(diff_text) / 1024
    if is_binary_patch(diff_text):
        summary_lines.append(f"[BINARY FILE REMOVED: {size_kb:.2f}KB]")
    else:
        summary_lines.append(f"[LARGE DIFF REMOVED: {size_kb:.2f}KB]")

    return '\n'.join(summary_lines)


def clean_predictions_file(
    input_file: Path,
    output_file: Path = None,
    size_threshold_kb: int = 1000
):
    """
    Clean large patches from predictions JSON

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        size_threshold_kb: Size threshold in KB (default: 1000KB = 1MB)
    """
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}.cleaned{input_file.suffix}"

    print(f"Reading {input_file}...")
    input_size_mb = input_file.stat().st_size / (1024*1024)
    print(f"File size: {input_size_mb:.2f}MB")

    with open(input_file, 'r') as f:
        data = json.load(f)

    total_patches = len(data)
    binary_removed = 0
    large_removed = 0
    generated_removed = 0

    print(f"\nProcessing {total_patches} patches (threshold: {size_threshold_kb}KB)...")

    for patch_id, patch_data in data.items():
        diff = patch_data.get('diff', '')
        size_kb = len(diff) / 1024

        if is_binary_patch(diff):
            binary_removed += 1
            patch_data['diff'] = get_file_summary(diff)
            patch_data['_removed'] = 'binary'
            print(f"  Patch {patch_id}: Binary patch removed ({size_kb:.2f}KB)")

        elif contains_generated_file(diff):
            generated_removed += 1
            patch_data['diff'] = get_file_summary(diff)
            patch_data['_removed'] = 'generated'
            print(f"  Patch {patch_id}: Generated file removed ({size_kb:.2f}KB)")

        elif is_large_file(diff, size_threshold_kb):
            large_removed += 1
            patch_data['diff'] = get_file_summary(diff)
            patch_data['_removed'] = 'large'
            print(f"  Patch {patch_id}: Large diff removed ({size_kb:.2f}KB)")

    # Write cleaned file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    output_size_mb = output_file.stat().st_size / (1024*1024)
    reduction = ((input_size_mb - output_size_mb) / input_size_mb) * 100

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total patches: {total_patches}")
    print(f"  Binary patches removed: {binary_removed}")
    print(f"  Generated files removed: {generated_removed}")
    print(f"  Large diffs removed: {large_removed}")
    print(f"  Total removed: {binary_removed + generated_removed + large_removed}")
    print(f"  Original file: {input_size_mb:.2f}MB")
    print(f"  Cleaned file: {output_size_mb:.2f}MB")
    print(f"  Size reduction: {reduction:.1f}%")
    print(f"{'='*70}")

    if output_size_mb < 100:
        print(f"✓ File is now {output_size_mb:.2f}MB and can be pushed to GitHub!")
    elif output_size_mb < 200:
        print(f"⚠ File is {output_size_mb:.2f}MB - consider Git LFS or split into chunks")
    else:
        print(f"✗ File is still {output_size_mb:.2f}MB - use Git LFS or more aggressive filtering")

    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_large_patches.py <predictions.json> [output.json] [threshold_kb]")
        print("  threshold_kb: Size threshold in KB (default: 1000)")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    if not input_file.exists():
        print(f"Error: File {input_file} not found")
        sys.exit(1)

    clean_predictions_file(input_file, output_file, threshold)
