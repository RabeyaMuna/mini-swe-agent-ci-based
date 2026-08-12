#!/usr/bin/env python3
"""
Fix all existing patch files to ensure they end with a newline.
Git patches MUST end with a newline or 'git apply' will fail.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fix_patch_file(patch_path: Path) -> bool:
    """
    Ensure patch file ends with newline.

    Returns:
        True if file was modified, False if already correct
    """
    try:
        # Read as bytes to preserve exact content
        with open(patch_path, 'rb') as f:
            content = f.read()

        # Skip empty files
        if not content:
            return False

        # Check if it ends with newline
        if content.endswith(b'\n'):
            return False  # Already correct

        # Add newline
        with open(patch_path, 'wb') as f:
            f.write(content + b'\n')

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("🔧 Fixing patch files to end with newline...\n")

    # Find all patch.diff files in results
    results_dir = PROJECT_ROOT / "results"
    patch_files = list(results_dir.rglob("patch.diff"))

    if not patch_files:
        print("No patch files found in results/")
        return

    print(f"Found {len(patch_files)} patch files\n")

    fixed_count = 0
    skipped_empty = 0
    already_ok = 0

    for patch_path in sorted(patch_files):
        # Get issue ID from path
        issue_id = patch_path.parent.name

        # Skip empty files
        if patch_path.stat().st_size == 0:
            skipped_empty += 1
            continue

        # Fix the file
        if fix_patch_file(patch_path):
            fixed_count += 1
            rel_path = patch_path.relative_to(PROJECT_ROOT)
            print(f"  ✓ Fixed: {rel_path}")
        else:
            already_ok += 1

    print(f"\n✅ Complete!")
    print(f"   Fixed: {fixed_count}")
    print(f"   Already OK: {already_ok}")
    print(f"   Empty (skipped): {skipped_empty}")
    print(f"   Total: {len(patch_files)}")


if __name__ == "__main__":
    main()
