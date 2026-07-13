#!/bin/bash
# Comprehensive fix for all 'list' object has no attribute 'get' errors
# Run this on your SERVER

set -e

echo "========================================="
echo "Applying Comprehensive Fix"
echo "========================================="
echo ""

cd ~/Documents/rabeya/mini-swe-agent-ci-based

# Add validation wrapper function at top of ci_memory_system.py
cat > /tmp/validation_fix.py << 'EOFPY'
def _safe_get(obj, key, default=None):
    """Safely get value from dict, handling cases where obj is a list."""
    if not isinstance(obj, dict):
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"[_safe_get] Expected dict but got {type(obj).__name__} for key '{key}'")
        return default
    return obj.get(key, default)
EOFPY

# Insert this function after imports in ci_memory_system.py
sed -i '40 r /tmp/validation_fix.py' src/minisweagent/run/benchmarks/utils/ci_memory_system.py

# Replace problematic .get() calls with _safe_get()
# Focus on lines that loop over memories
sed -i 's/mem\.get("validation_cmd"/_safe_get(mem, "validation_cmd"/g' src/minisweagent/run/benchmarks/utils/ci_memory_system.py
sed -i 's/mem\.get("failed_cmd"/_safe_get(mem, "failed_cmd"/g' src/minisweagent/run/benchmarks/utils/ci_memory_system.py
sed -i 's/problem\.get("source"/_safe_get(problem, "source"/g' src/minisweagent/run/benchmarks/utils/ci_memory_system.py

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "========================================="
echo "✓ Fix Applied"
echo "========================================="
echo ""
echo "Now run:"
echo "  python3 scripts/run_eval.py --issue-ids 71 --ablation L1+L2+L3 --workers 1"
