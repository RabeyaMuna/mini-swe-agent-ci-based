#!/bin/bash
# run_all_analyses.sh - Run all priority analyses for professor meeting
# =====================================================================

set -e  # Exit on error

echo "========================================"
echo "Running All Priority Analyses"
echo "========================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check for gitpython
echo ""
echo "Checking dependencies..."
if ! python3 -c "import git" 2>/dev/null; then
    echo "Installing gitpython..."
    pip install gitpython
fi

# Task 1: Benchmark Characterization
echo ""
echo "========================================"
echo "Task 1: Benchmark Characterization"
echo "========================================"
python3 "$SCRIPT_DIR/benchmark_statistics.py"

# Task 2: Commit Decomposition
echo ""
echo "========================================"
echo "Task 2: Commit-Level Decomposition"
echo "========================================"
python3 "$SCRIPT_DIR/commit_decomposition.py"

# Task 3: Failure Type Analysis
echo ""
echo "========================================"
echo "Task 3: Failure Type Analysis"
echo "========================================"
python3 "$SCRIPT_DIR/failure_type_analysis.py"

echo ""
echo "========================================"
echo "All Analyses Complete!"
echo "========================================"
echo ""
echo "Results saved in:"
echo "  - statistics/benchmark_stats.json"
echo "  - statistics/benchmark_comparison.md"
echo "  - statistics/commit_decomposition_summary.json"
echo "  - data/commit_decomposed_issues.json"
echo "  - statistics/failure_type_analysis.json"
echo "  - statistics/failure_type_analysis.md"
echo ""
echo "Next: Review the markdown files for presentation"
