"""
inspect_memory.py
=================
Inspect what is stored in failure_memory.json (L1), repo_memory.json (L2),
and cross_memory.json (L3) after a run.

Usage:
    python scripts/inspect_memory.py --memory-root data/trs
    python scripts/inspect_memory.py --memory-root results/test_memory_v2

What it shows:
    - How many records per level
    - How many distinct (file, error_type) pairs in L1  ← key check for the fix
    - Whether multi-type files are stored separately
    - Top error_types and failure_patterns
    - Sample records
"""

import argparse
import json
import os
from collections import Counter, defaultdict


def load(path: str):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def inspect_l1(records):
    print(f"\n{'='*60}")
    print(f"L1 — failure_memory.json:  {len(records)} records")
    print(f"{'='*60}")

    if not records:
        print("  (empty)")
        return

    # Key check: how many files have multiple error_types stored?
    file_to_types = defaultdict(set)
    for r in records:
        file_to_types[r.get("file", "")].add(r.get("error_type", ""))

    multi_type_files = {f: types for f, types in file_to_types.items() if len(types) > 1}
    print(f"\n  Distinct files:                  {len(file_to_types)}")
    print(f"  Files with >1 error_type stored: {len(multi_type_files)}  ← should be > 0 with new fix")

    if multi_type_files:
        print("\n  Multi-type files (sample):")
        for f, types in list(multi_type_files.items())[:5]:
            print(f"    {f}  →  {sorted(types)}")

    # error_type distribution
    et_counter = Counter(r.get("error_type", "(none)") for r in records)
    print(f"\n  Top error_types:")
    for et, count in et_counter.most_common(8):
        print(f"    {count:3d}  {et}")

    # failure_pattern distribution
    fp_counter = Counter(r.get("failure_pattern", "(none)") for r in records)
    print(f"\n  Top failure_patterns:")
    for fp, count in fp_counter.most_common(8):
        print(f"    {count:3d}  {fp}")

    # Sample record
    print(f"\n  Sample record:")
    r = records[0]
    for k in ("file", "error_type", "failure_pattern", "issue_type", "failure_reason", "fix_direction"):
        v = str(r.get(k, ""))
        print(f"    {k:20s}: {v[:100]}")


def inspect_l2(records):
    print(f"\n{'='*60}")
    print(f"L2 — repo_memory.json:  {len(records)} records")
    print(f"{'='*60}")

    if not records:
        print("  (empty)")
        return

    repo_counter = Counter(r.get("repo", "(none)") for r in records)
    print(f"\n  Repos covered: {len(repo_counter)}")
    for repo, count in repo_counter.most_common(5):
        print(f"    {count:3d}  {repo}")

    # Check multi-file records
    file_counts = [len(r.get("files", [])) for r in records]
    avg = sum(file_counts) / len(file_counts) if file_counts else 0
    print(f"\n  Avg files per L2 record: {avg:.1f}")


def inspect_l3(records):
    print(f"\n{'='*60}")
    print(f"L3 — cross_memory.json:  {len(records)} records")
    print(f"{'='*60}")

    if not records:
        print("  (empty)")
        return

    et_counter = Counter(r.get("error_type", "(none)") for r in records)
    print(f"\n  error_types covered:")
    for et, count in et_counter.most_common(8):
        print(f"    {count:3d}  {et}")

    # Check accumulation
    multi_repo = [r for r in records if len(r.get("repos", [])) > 1]
    print(f"\n  L3 records seen in >1 repo: {len(multi_repo)}  ← shows cross-repo transfer")


def check_retrieval(memory_root: str, query_error_type: str):
    """Quick sanity check: how many L1 records would match a given error_type?"""
    path = os.path.join(memory_root, "failure_memory.json")
    records = load(path)
    matches = [r for r in records if r.get("error_type", "").lower() == query_error_type.lower()]
    print(f"\n{'='*60}")
    print(f"Retrieval check: error_type='{query_error_type}'")
    print(f"  L1 records matching: {len(matches)} / {len(records)}")
    for r in matches[:3]:
        print(f"    file={r.get('file','')}  pattern={r.get('failure_pattern','')}  fix={r.get('fix_direction','')[:60]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-root", default="data/trs")
    parser.add_argument("--check-type", default="", help="Test retrieval for this error_type")
    args = parser.parse_args()

    root = args.memory_root
    print(f"\nMemory root: {root}")

    l1 = load(os.path.join(root, "failure_memory.json"))
    l2 = load(os.path.join(root, "repo_memory.json"))
    l3 = load(os.path.join(root, "cross_memory.json"))

    inspect_l1(l1)
    inspect_l2(l2)
    inspect_l3(l3)

    if args.check_type:
        check_retrieval(root, args.check_type)

    print()


if __name__ == "__main__":
    main()
