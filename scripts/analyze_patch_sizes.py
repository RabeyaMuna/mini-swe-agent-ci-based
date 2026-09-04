#!/usr/bin/env python3
"""
Analyze patch sizes in predictions JSON
"""

import json
import sys
from pathlib import Path


def analyze_patch_sizes(json_file: Path, top_n: int = 20):
    """Analyze and report largest patches"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Calculate sizes
    patch_sizes = []
    for patch_id, patch_data in data.items():
        diff = patch_data.get('diff', '')
        size_kb = len(diff) / 1024
        patch_sizes.append((patch_id, size_kb, diff[:200]))

    # Sort by size
    patch_sizes.sort(key=lambda x: x[1], reverse=True)

    print(f"Top {top_n} largest patches:\n")
    print(f"{'ID':<10} {'Size':<15} {'Preview'}")
    print(f"{'-'*80}")

    total_size = 0
    for patch_id, size_kb, preview in patch_sizes[:top_n]:
        total_size += size_kb
        preview_clean = preview.replace('\n', ' ')[:60]
        print(f"{patch_id:<10} {size_kb:>8.2f} KB    {preview_clean}")

    print(f"\n{'-'*80}")
    print(f"Top {top_n} patches total: {total_size/1024:.2f}MB")
    print(f"Total file size: {sum(s[1] for s in patch_sizes)/1024:.2f}MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_patch_sizes.py <predictions.json>")
        sys.exit(1)

    json_file = Path(sys.argv[1])
    analyze_patch_sizes(json_file)
