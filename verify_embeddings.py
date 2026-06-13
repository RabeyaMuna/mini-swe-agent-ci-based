#!/usr/bin/env python3
"""Quick verification that embeddings are saved."""

import json
from pathlib import Path

memory_dir = Path("data/trs")

files = {
    "L1": memory_dir / "failure_memory.json",
    "L2": memory_dir / "repo_memory.json",
    "L3": memory_dir / "cross_memory.json"
}

print("=" * 80)
print("EMBEDDING VERIFICATION")
print("=" * 80)

for level, path in files.items():
    if not path.exists():
        print(f"\n{level}: ERROR File not found: {path}")
        continue

    data = json.loads(path.read_text())
    total = len(data)
    embedded = sum(1 for entry in data if "_embedding" in entry)

    print(f"\n{level}: {path.name}")
    print(f"  Total entries: {total}")
    print(f"  With embeddings: {embedded}")
    print(f"  Coverage: {embedded}/{total} ({100*embedded/total if total > 0 else 0:.1f}%)")

    if embedded > 0:
        # Check embedding dimension
        sample = next(e for e in data if "_embedding" in e)
        dim = len(sample["_embedding"])
        print(f"  Embedding dimension: {dim}")

        # Show sample search_document
        if "search_document" in sample:
            doc = sample["search_document"]
            print(f"  Sample search_document length: {len(doc)} chars")
            print(f"  First 150 chars: {doc[:150]}...")

    # Check if repair_trajectory is in search_document (L2 only)
    if level == "L2" and embedded > 0:
        sample = next((e for e in data if "_embedding" in e and "search_document" in e), None)
        if sample:
            search_doc = sample.get("search_document", "")
            has_trajectory = "repair_trajectory:" in search_doc
            print(f"  OK Has repair_trajectory in search_document: {has_trajectory}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

all_embedded = all(
    sum(1 for e in json.loads(files[level].read_text()) if "_embedding" in e) == len(json.loads(files[level].read_text()))
    for level in files if files[level].exists()
)

if all_embedded:
    print("OK SUCCESS: All memory entries have embeddings!")
    print("\n Ready for retrieval testing")
else:
    print("WARNING:  Some entries missing embeddings")
    print("   Run: python scripts/precompute_embeddings.py --memory-root data/trs")
