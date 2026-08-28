#!/usr/bin/env python3
"""
Remove dataset metadata (error_type, failed_jobs) from L1 memory files.

Only LLM-assigned failure_type within problems should remain.
Dataset metadata is removed from top-level of L1 memory entries.
"""

import json
import sys
from pathlib import Path


def clean_memory_file(memory_path: Path) -> int:
    """
    Remove error_type, failed_jobs, total_failed_jobs, total_failed_steps
    from top-level of each L1 memory entry.

    Returns:
        Number of entries cleaned
    """
    if not memory_path.exists():
        print(f"  SKIP {memory_path} (not found)")
        return 0

    # Load existing memory
    with open(memory_path) as f:
        memory_entries = json.load(f)

    print(f"  Loaded {len(memory_entries)} entries from {memory_path.name}")

    cleaned_count = 0
    fields_to_remove = ["error_type", "failed_jobs", "total_failed_jobs", "total_failed_steps"]

    # Clean each entry
    for entry in memory_entries:
        removed_any = False
        for field in fields_to_remove:
            if field in entry:
                del entry[field]
                removed_any = True

        if removed_any:
            cleaned_count += 1

    # Save cleaned memory
    with open(memory_path, 'w') as f:
        json.dump(memory_entries, f, indent=2)

    print(f"  ✓ Cleaned {cleaned_count} entries, saved to {memory_path.name}")
    return cleaned_count


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    dirs_to_clean = [
        ("back_trs", "Backward Traces"),
        ("bidirect_trs", "Bidirectional Traces"),
        ("fwr_trs", "Forward Traces"),
    ]

    print("=" * 80)
    print("REMOVING DATASET METADATA FROM L1 MEMORY FILES")
    print("=" * 80)
    print("Keeping only LLM-assigned failure_type within problems")
    print()

    total_cleaned = 0

    for dir_name, label in dirs_to_clean:
        print(f"{label} ({dir_name}):")
        memory_path = data_dir / dir_name / "failure_memory.json"
        cleaned = clean_memory_file(memory_path)
        total_cleaned += cleaned
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total entries cleaned: {total_cleaned}")
    print()
    print("✓ Done! Dataset metadata removed from all L1 memory files.")
    print("✓ Only LLM-assigned failure_type in problems remains.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
