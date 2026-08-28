#!/usr/bin/env python3
"""
Patch workflow_path into existing memory files from original dataset.

This fixes memory files that were built without workflow_path by looking up
each issue in the original dataset and copying the workflow_path.
"""

import json
from pathlib import Path
import argparse


def load_dataset_as_map(dataset_path: Path) -> dict:
    """Load dataset and create issue_id -> workflow_path mapping."""
    mapping = {}

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            issue = json.loads(line)

            # Get issue identifiers
            issue_id = str(issue.get('id') or issue.get('instance_id', ''))
            sha_fail = str(issue.get('sha_fail', ''))

            # Get workflow_path
            workflow_path = issue.get('workflow_path', '')

            # Map both id and sha to workflow_path
            if issue_id and workflow_path:
                mapping[issue_id] = workflow_path
            if sha_fail and workflow_path:
                mapping[sha_fail] = workflow_path

    print(f"Loaded {len(mapping)} issue -> workflow_path mappings from dataset")
    return mapping


def patch_memory_file(memory_file: Path, dataset_map: dict, dry_run: bool = False):
    """Patch workflow_path into a memory file."""

    if not memory_file.exists():
        print(f"Skip {memory_file.name}: file not found")
        return

    print(f"\nPatching {memory_file.name}...")

    # Load memory file
    with open(memory_file, 'r', encoding='utf-8') as f:
        memory_data = json.load(f)

    if not isinstance(memory_data, list):
        print(f"  Error: Expected list, got {type(memory_data)}")
        return

    # Patch each entry
    patched_count = 0
    already_had = 0
    not_found = 0

    for entry in memory_data:
        # Get issue identifier
        issue_id = str(entry.get('issue_id', ''))

        # Check if already has workflow_path
        current_wp = entry.get('workflow_path')
        if current_wp and current_wp != 'None':
            already_had += 1
            continue

        # Look up in dataset
        if issue_id in dataset_map:
            entry['workflow_path'] = dataset_map[issue_id]
            patched_count += 1
        else:
            not_found += 1

    print(f"  Patched: {patched_count}")
    print(f"  Already had: {already_had}")
    print(f"  Not found in dataset: {not_found}")
    print(f"  Total entries: {len(memory_data)}")

    # Save
    if not dry_run:
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved to {memory_file}")
    else:
        print(f"  (dry run - not saved)")


def patch_decomposed_file(decomposed_file: Path, dataset_map: dict, dry_run: bool = False):
    """Patch workflow_path into decomposed_issues.json."""

    if not decomposed_file.exists():
        print(f"Skip {decomposed_file.name}: file not found")
        return

    print(f"\nPatching {decomposed_file.name}...")

    # Load decomposed file
    with open(decomposed_file, 'r', encoding='utf-8') as f:
        decomposed_data = json.load(f)

    if not isinstance(decomposed_data, list):
        print(f"  Error: Expected list, got {type(decomposed_data)}")
        return

    # Patch each entry
    patched_count = 0
    already_had = 0
    not_found = 0

    for entry in decomposed_data:
        # Get issue identifier
        issue_id = str(entry.get('original_issue_id') or entry.get('issue_id', ''))
        sha_fail = str(entry.get('sha_fail', ''))

        # Check if already has workflow_path
        current_wp = entry.get('workflow_path')
        if current_wp and current_wp not in [None, 'None', '']:
            already_had += 1
            continue

        # Look up in dataset (try both id and sha)
        workflow_path = dataset_map.get(issue_id) or dataset_map.get(sha_fail)

        if workflow_path:
            entry['workflow_path'] = workflow_path
            patched_count += 1
        else:
            not_found += 1

    print(f"  Patched: {patched_count}")
    print(f"  Already had: {already_had}")
    print(f"  Not found in dataset: {not_found}")
    print(f"  Total entries: {len(decomposed_data)}")

    # Save
    if not dry_run:
        with open(decomposed_file, 'w', encoding='utf-8') as f:
            json.dump(decomposed_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved to {decomposed_file}")
    else:
        print(f"  (dry run - not saved)")


def main():
    parser = argparse.ArgumentParser(
        description="Patch workflow_path into existing memory files from dataset"
    )
    parser.add_argument(
        '--dataset',
        default='data/eval_set.jsonl',
        help='Path to dataset (default: data/eval_set.jsonl)'
    )
    parser.add_argument(
        '--memory-dir',
        default='data/back_trs',
        help='Memory directory to patch (default: data/back_trs)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be patched without saving'
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    memory_dir = Path(args.memory_dir)

    print("="*70)
    print("Patching workflow_path from dataset into memory files")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Memory dir: {memory_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Load dataset mapping
    dataset_map = load_dataset_as_map(dataset_path)

    # Patch each memory file
    patch_decomposed_file(memory_dir / "decomposed_issues.json", dataset_map, args.dry_run)
    patch_memory_file(memory_dir / "failure_memory.json", dataset_map, args.dry_run)
    patch_memory_file(memory_dir / "repo_memory.json", dataset_map, args.dry_run)
    patch_memory_file(memory_dir / "cross_memory.json", dataset_map, args.dry_run)

    print("\n" + "="*70)
    if args.dry_run:
        print("Dry run complete. Run without --dry-run to save changes.")
    else:
        print("✓ Patching complete!")
    print("="*70)


if __name__ == "__main__":
    main()
