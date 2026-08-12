#!/usr/bin/env python3
"""
Remove error entries from cache files and save them to exception/ folder.
"""
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCEPTION_DIR = PROJECT_ROOT / "exception"
LOG_DETAILS_PATH = PROJECT_ROOT / "data" / "log_details.json"
WORKFLOW_VALIDATION_PATH = PROJECT_ROOT / "data" / "workflow_validation_cache.json"
DECOMPOSITION_CACHE_PATH = PROJECT_ROOT / "data" / "decomposition_cache.json"


def clean_cache_file(cache_path: Path, cache_name: str):
    """Remove error entries from a cache file."""
    if not cache_path.exists():
        print(f"✓ {cache_name}: File doesn't exist (nothing to clean)")
        return

    with open(cache_path) as f:
        cache = json.load(f)

    if isinstance(cache, dict):
        # decomposition_cache.json format
        print(f"\n📋 {cache_name}: Dict format (no error entries expected)")
        return

    if not isinstance(cache, list):
        print(f"⚠️  {cache_name}: Unknown format, skipping")
        return

    # Filter out error entries
    clean_entries = []
    error_entries = []

    for entry in cache:
        if "error" in entry:
            error_entries.append(entry)
        else:
            clean_entries.append(entry)

    if not error_entries:
        print(f"✓ {cache_name}: No error entries found ({len(cache)} valid entries)")
        return

    print(f"\n📋 {cache_name}:")
    print(f"  Total entries: {len(cache)}")
    print(f"  Valid entries: {len(clean_entries)}")
    print(f"  Error entries: {len(error_entries)}")

    # Save error entries to exception folder
    EXCEPTION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    error_log_path = EXCEPTION_DIR / f"{cache_path.stem}_errors_{timestamp}.json"

    with open(error_log_path, "w") as f:
        json.dump(error_entries, f, indent=2)

    print(f"  📝 Saved errors to: {error_log_path.relative_to(PROJECT_ROOT)}")

    # Show error IDs
    error_ids = [
        str(e.get("id") or e.get("issue_id") or e.get("sha_fail", "unknown")[:8])
        for e in error_entries
    ]
    print(f"  Error IDs: {', '.join(error_ids[:10])}")
    if len(error_ids) > 10:
        print(f"    ... and {len(error_ids) - 10} more")

    # Create backup
    backup_path = cache_path.with_suffix(".json.before_clean")
    import shutil
    shutil.copy2(cache_path, backup_path)
    print(f"  📦 Backup: {backup_path.name}")

    # Save cleaned cache
    with open(cache_path, "w") as f:
        json.dump(clean_entries, f, indent=2)

    print(f"  ✅ Cleaned cache saved ({len(clean_entries)} entries)")


if __name__ == "__main__":
    print("🧹 Cleaning error entries from cache files...\n")

    clean_cache_file(LOG_DETAILS_PATH, "log_details.json")
    clean_cache_file(WORKFLOW_VALIDATION_PATH, "workflow_validation_cache.json")
    clean_cache_file(DECOMPOSITION_CACHE_PATH, "decomposition_cache.json")

    print("\n✅ Cleanup complete!")
