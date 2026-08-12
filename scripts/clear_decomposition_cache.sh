#!/bin/bash
# Clear decomposition cache to force regeneration with workflow fields

CACHE_FILE="data/decomposition_cache.json"

if [ -f "$CACHE_FILE" ]; then
    # Create backup
    BACKUP="${CACHE_FILE}.cleared_$(date +%Y%m%d_%H%M%S)"
    mv "$CACHE_FILE" "$BACKUP"
    echo "✅ Cleared cache: $CACHE_FILE"
    echo "💾 Backup saved: $BACKUP"
    echo ""
    echo "Next run will regenerate cache with workflow_name and workflow_path!"
else
    echo "⚠️  Cache file not found: $CACHE_FILE"
fi
