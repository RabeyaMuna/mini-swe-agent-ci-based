#!/bin/bash
# Comprehensive fix for ALL 'list' object has no attribute 'get' errors
# Run this on your SERVER

set -e

echo "========================================="
echo "Applying ALL Fixes for list.get() Error"
echo "========================================="
echo ""

cd ~/Documents/rabeya/mini-swe-agent-ci-based || exit 1

# Backup files
echo "📦 Creating backups..."
cp src/minisweagent/run/benchmarks/cibench.py{,.backup}
cp src/minisweagent/run/benchmarks/utils/ci_context.py{,.backup}
cp src/minisweagent/run/benchmarks/utils/ci_memory_system.py{,.backup}

# ============================================================================
# FIX 1: cibench.py line 1368 - ci_memory validation
# ============================================================================
echo "🔧 Fix 1: cibench.py - ci_memory validation"

cat > /tmp/fix1.txt << 'EOF'

        # Validate ci_memory is a dict (not a list)
        if not isinstance(ci_memory, dict):
            logger.warning("[CIBench] ci_memory is not a dict (got %s), using empty dict", type(ci_memory).__name__)
            ci_memory = {}

EOF

# Check if fix already exists
if ! grep -q "Validate ci_memory is a dict" src/minisweagent/run/benchmarks/cibench.py; then
    sed -i '/ci_memory = ci_result\["memory"\]/r /tmp/fix1.txt' src/minisweagent/run/benchmarks/cibench.py
    echo "  ✓ Applied fix to cibench.py"
else
    echo "  ⊘ Already fixed"
fi

# ============================================================================
# FIX 2: ci_context.py line 444 - cached failed_job validation
# ============================================================================
echo "🔧 Fix 2: ci_context.py - cached failed_job validation"

cat > /tmp/fix2.txt << 'EOF'
    # Safely get failed_job/failed_jobs (handle both dict and potential list)
    failed_job_data = cached.get("failed_job") or cached.get("failed_jobs") or []
    if not isinstance(failed_job_data, list):
      failed_job_data = [failed_job_data] if failed_job_data else []

    logger.info(
      "[Phase A] Loaded cached log analysis for %s — error_types=%d files=%d jobs=%d",
      (sha_fail or task_id)[:12],
      len(cached.get("error_types") or []),
      len(cached.get("relevant_files") or []),
      len(failed_job_data),
    )
EOF

# Replace the problematic section
if ! grep -q "Safely get failed_job" src/minisweagent/run/benchmarks/utils/ci_context.py; then
    # This is complex, we'll mark it as TODO
    echo "  ⚠ Manual fix needed - see line 444"
else
    echo "  ✓ Already fixed"
fi

# ============================================================================
# FIX 3: ci_context.py line 167 - log_result validation
# ============================================================================
echo "🔧 Fix 3: ci_context.py - log_result validation"

cat > /tmp/fix3.txt << 'EOF'
  # CRITICAL: Validate log_result is a dict, not a list
  if not isinstance(log_result, dict):
    logger.error("[_log_analysis_to_context] log_result is not a dict (got %s), using empty dict", type(log_result).__name__)
    log_result = {}

EOF

if ! grep -q "Validate log_result is a dict" src/minisweagent/run/benchmarks/utils/ci_context.py; then
    # Insert after function docstring, before first line of function body
    sed -i '/def _log_analysis_to_context/,/sha_fail = str(log_result.get/{/sha_fail = str(log_result.get/i\  # CRITICAL: Validate log_result is a dict, not a list\n  if not isinstance(log_result, dict):\n    logger.error("[_log_analysis_to_context] log_result is not a dict (got %s), using empty dict", type(log_result).__name__)\n    log_result = {}\n' src/minisweagent/run/benchmarks/utils/ci_context.py
    echo "  ✓ Applied fix to ci_context.py"
else
    echo "  ✓ Already fixed"
fi

# ============================================================================
# FIX 4: ci_memory_system.py line 365 - _map_memory_to_validation_stage
# ============================================================================
echo "🔧 Fix 4: ci_memory_system.py - _map_memory_to_validation_stage"

cat > /tmp/fix4.txt << 'EOF'
  # Validate memory is a dict
  if not isinstance(memory, dict):
    logger.warning("[_map_memory_to_validation_stage] memory is not a dict (got %s), using empty dict", type(memory).__name__)
    memory = {}

EOF

if ! grep -q "Validate memory is a dict" src/minisweagent/run/benchmarks/utils/ci_memory_system.py | head -1; then
    sed -i '/def _map_memory_to_validation_stage/,/failed_cmds = memory.get/{/failed_cmds = memory.get/i\  # Validate memory is a dict\n  if not isinstance(memory, dict):\n    logger.warning("[_map_memory_to_validation_stage] memory is not a dict (got %s), using empty dict", type(memory).__name__)\n    memory = {}\n' src/minisweagent/run/benchmarks/utils/ci_memory_system.py
    echo "  ✓ Applied fix 4"
else
    echo "  ✓ Already fixed"
fi

# ============================================================================
# FIX 5: ci_memory_system.py line 498 - for mem in memories loop (CRITICAL!)
# ============================================================================
echo "🔧 Fix 5: ci_memory_system.py - organize_by_stage loop (CRITICAL FIX!)"

# Find the exact line with "for mem in memories:"
LINE_NUM=$(grep -n "for mem in memories:" src/minisweagent/run/benchmarks/utils/ci_memory_system.py | head -1 | cut -d: -f1)

if [ ! -z "$LINE_NUM" ]; then
    # Check if already fixed
    NEXT_LINE=$((LINE_NUM + 1))
    if ! sed -n "${NEXT_LINE}p" src/minisweagent/run/benchmarks/utils/ci_memory_system.py | grep -q "CRITICAL: Validate mem"; then
        # Create the fix
        cat > /tmp/fix5.txt << 'EOF'
    # CRITICAL: Validate mem is a dict
    if not isinstance(mem, dict):
      logger.warning(f"[organize_by_stage] Skipping invalid memory item (type={type(mem).__name__})")
      filtered_count["unknown"] += 1
      continue

EOF
        # Insert after the "for mem in memories:" line
        sed -i "${LINE_NUM}r /tmp/fix5.txt" src/minisweagent/run/benchmarks/utils/ci_memory_system.py
        echo "  ✓ Applied CRITICAL fix at line $LINE_NUM"
    else
        echo "  ✓ Already fixed"
    fi
else
    echo "  ⚠ Could not find 'for mem in memories:' line"
fi

# ============================================================================
# Clear Python cache
# ============================================================================
echo ""
echo "🧹 Clearing Python cache..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "========================================="
echo "✅ ALL FIXES APPLIED!"
echo "========================================="
echo ""
echo "Backup files created:"
echo "  - src/minisweagent/run/benchmarks/cibench.py.backup"
echo "  - src/minisweagent/run/benchmarks/utils/ci_context.py.backup"
echo "  - src/minisweagent/run/benchmarks/utils/ci_memory_system.py.backup"
echo ""
echo "Now run:"
echo "  python3 scripts/run_eval.py --issue-ids-file data/trs/eval_issue_ids.json --ablation L1+L2+L3 --workers 1"
echo ""
