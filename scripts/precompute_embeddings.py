"""
precompute_embeddings.py
========================
Pre-compute and store embeddings for all records in an existing memory bank.

Run this AFTER building memory (seed_memory.py or build_memory_bank.py) so that
all retrieval queries use stored vectors instead of recomputing from scratch.

Usage:
    # Standard usage
    python scripts/precompute_embeddings.py --memory-root data/trs

    # Verbose mode
    python scripts/precompute_embeddings.py --memory-root data/trs --verbose

What it does:
    1. Loads L1/L2/L3 memory JSON files
    2. Builds search_document for any record that is missing one
    3. Embeds every search_document with sentence-transformers (384-dim)
    4. Stores the vector as "embedding" field in each record
    5. Writes updated JSON files back to disk

After this runs, retrieval will be MUCH faster (no re-embedding on every query).

Compatible with:
    OK Memory from seed_memory.py (runtime generation)
    OK Memory from build_memory_bank.py (CI-REPAIR-BENCH style)
    OK Both old and new field structures
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Precompute embeddings for memory bank"
    )
    parser.add_argument(
        "--memory-root", required=True, help="Path to memory directory (e.g., data/trs)"
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    args = parser.parse_args()

    root = args.memory_root
    paths = {
        "L1": os.path.join(root, "failure_memory.json"),
        "L2": os.path.join(root, "repo_memory.json"),
        "L3": os.path.join(root, "cross_memory.json"),
    }

    # Load embedding provider
    try:
        from memory_plugin.memory_plugin import (
            MemoryPlugin,
            _EmbeddingProvider,
            _load_json_list,
            _write_json_list,
        )
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from the project root with PYTHONPATH set, or use the run script.")
        sys.exit(1)

    provider = _EmbeddingProvider.get()
    if provider._backend == "none":
        print("ERROR: No embedding model available.")
        print("Install: pip install sentence-transformers")
        sys.exit(1)

    # Minimal plugin just to use _build_search_document
    config = {
        "memory_enabled": True,
        "memory_top_k": 3,
        "memory_ablation_levels": "L1+L2+L3",
        "memory_backend": "json",
        "project_result_dir": root,
    }
    plugin = MemoryPlugin(config, root)

    total_embedded = 0
    total_skipped = 0

    for level, path in paths.items():
        if not os.path.exists(path):
            print(f"[{level}] Not found: {path}")
            continue

        records = _load_json_list(path)
        changed = False
        print(f"\n[{level}] {len(records)} records — computing embeddings...")

        for i, record in enumerate(records):
            # Build search_document if missing (seeded records don't have it)
            doc = str(record.get("search_document") or "").strip()
            if not doc:
                doc = plugin._build_search_document(record, level=level)
                if doc:
                    record["search_document"] = doc
                    changed = True

            if not doc:
                total_skipped += 1
                continue

            # Skip if embedding already stored
            if "_embedding" in record:
                total_skipped += 1
                continue

            # Embed
            vec = provider.embed(doc)
            if vec is not None:
                record["_embedding"] = vec.tolist()
                changed = True
                total_embedded += 1

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(records)} done...")

        if changed:
            _write_json_list(path, records)
            print(f"[{level}] Saved {len(records)} records with embeddings -> {path}")
        else:
            print(f"[{level}] No changes needed.")

    print(
        f"\nDone. Embedded: {total_embedded}  Skipped (already done or empty): {total_skipped}"
    )
    print(f"Memory bank at '{root}' is now ready for fast retrieval.")


if __name__ == "__main__":
    main()
