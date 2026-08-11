#!/bin/bash
# Quick wrapper to clean up empty patches from results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🧹 Cleaning up empty patches from prediction files..."

# Add verbose flag if not already present
if [[ ! "$*" =~ "-v" ]] && [[ ! "$*" =~ "--verbose" ]]; then
    python3 "$SCRIPT_DIR/scripts/cleanup_empty_patches.py" --verbose "$@"
else
    python3 "$SCRIPT_DIR/scripts/cleanup_empty_patches.py" "$@"
fi
