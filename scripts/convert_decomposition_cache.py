#!/usr/bin/env python3
"""
Convert decomposition_cache.json to new simplified format.

Old format:
{
  "sha_fail": {
    "query": {...},
    "problems": [...],
    "timestamp": "...",
    "model": "..."
  }
}

New format:
{
  "sha_fail": [problems_array]
}
"""

import json
import sys
from pathlib import Path


def convert_cache(cache_file: Path, backup: bool = True):
    """
    Convert decomposition cache to new format.

    Args:
        cache_file: Path to decomposition_cache.json
        backup: Create backup before converting
    """
    print(f"📖 Reading {cache_file}...")

    try:
        with open(cache_file) as f:
            old_cache = json.load(f)
    except Exception as e:
        print(f"❌ Error reading cache: {e}")
        return 1

    print(f"   Found {len(old_cache)} entries")

    # Create backup
    if backup:
        backup_file = cache_file.with_suffix(".json.backup")
        cache_file.rename(backup_file)
        print(f"💾 Backup created: {backup_file}")

    # Convert to new format
    new_cache = {}
    converted = 0
    skipped = 0

    for sha_fail, entry in old_cache.items():
        # Check if already in new format (list)
        if isinstance(entry, list):
            print(f"   ✓ {sha_fail[:12]} - already in new format")
            new_cache[sha_fail] = entry
            converted += 1
            continue

        # Check if old format (dict with "problems" key)
        if isinstance(entry, dict) and "problems" in entry:
            problems = entry["problems"]
            if isinstance(problems, list):
                # Extract workflow fields from top-level query
                workflow_name = ""
                workflow_path = ""
                if "query" in entry and isinstance(entry["query"], dict):
                    workflow_name = entry["query"].get("workflow_name", "")
                    workflow_path = entry["query"].get("workflow_path", "")

                # Populate workflow fields into each problem's l1/l2 queries
                # Also remove validation_sequence (not needed in cache)
                updated_problems = []
                for problem in problems:
                    # Remove validation_sequence if present
                    problem.pop("validation_sequence", None)
                    if "query" in problem and isinstance(problem["query"], dict):
                        # Update L1 query with workflow fields
                        if "l1" in problem["query"] and isinstance(problem["query"]["l1"], dict):
                            if not problem["query"]["l1"].get("workflow_name"):
                                problem["query"]["l1"]["workflow_name"] = workflow_name
                            if not problem["query"]["l1"].get("workflow_path"):
                                problem["query"]["l1"]["workflow_path"] = workflow_path

                        # L2 should have workflow_path but NOT workflow_name (repo-level)
                        if "l2" in problem["query"] and isinstance(problem["query"]["l2"], dict):
                            if not problem["query"]["l2"].get("workflow_path"):
                                problem["query"]["l2"]["workflow_path"] = workflow_path
                            # Ensure L2 doesn't have workflow_name
                            problem["query"]["l2"].pop("workflow_name", None)

                    updated_problems.append(problem)

                new_cache[sha_fail] = updated_problems
                if workflow_name or workflow_path:
                    print(f"   ✓ {sha_fail[:12]} - converted + populated workflow fields")
                else:
                    print(f"   ✓ {sha_fail[:12]} - converted (no workflow fields in source)")
                converted += 1
            else:
                print(f"   ⚠️  {sha_fail[:12]} - problems is not a list, skipping")
                skipped += 1
        else:
            print(f"   ⚠️  {sha_fail[:12]} - unknown format, skipping")
            skipped += 1

    # Save new cache
    print(f"\n✍️  Writing converted cache to {cache_file}...")

    try:
        with open(cache_file, 'w') as f:
            json.dump(new_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Error writing cache: {e}")
        return 1

    # Summary
    print()
    print("=" * 60)
    print(f"✅ Conversion complete!")
    print(f"   Total entries:     {len(old_cache)}")
    print(f"   Converted:         {converted}")
    print(f"   Skipped:           {skipped}")
    print(f"   Final entries:     {len(new_cache)}")
    print("=" * 60)

    # Show example
    if new_cache:
        example_sha = list(new_cache.keys())[0]
        example_problems = new_cache[example_sha]
        print(f"\nExample entry ({example_sha[:12]}):")
        print(f"  - {len(example_problems)} problem(s)")
        if example_problems:
            first_problem = example_problems[0]
            print(f"  - First problem: {first_problem.get('problem', 'N/A')[:60]}...")
            if 'query' in first_problem:
                print(f"  - Has l1/l2/l3 queries: ✓")
                if 'l1' in first_problem['query']:
                    l1 = first_problem['query']['l1']
                    print(f"    - L1 workflow_name: {l1.get('workflow_name', 'N/A')}")
                    print(f"    - L1 workflow_path: {l1.get('workflow_path', 'N/A')}")

    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert decomposition_cache.json to new simplified format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with backup (recommended)
  python convert_decomposition_cache.py data/decomposition_cache.json

  # Convert without backup
  python convert_decomposition_cache.py data/decomposition_cache.json --no-backup
        """
    )

    parser.add_argument(
        'cache_file',
        type=Path,
        help='Path to decomposition_cache.json'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup before converting'
    )

    args = parser.parse_args()

    if not args.cache_file.exists():
        print(f"❌ Error: File not found: {args.cache_file}")
        return 1

    return convert_cache(args.cache_file, backup=not args.no_backup)


if __name__ == '__main__':
    sys.exit(main())
