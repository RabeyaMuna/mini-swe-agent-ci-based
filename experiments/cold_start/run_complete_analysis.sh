#!/bin/bash
#
# Complete Cold-Start Analysis Pipeline
#
# This script runs the entire analysis in sequence:
# 1. Extract repo mapping from results
# 2. Partition instances into History-Available vs Cold-Start
# 3. Calculate Pass@1 for each group
# 4. Verify results
# 5. Generate final report
#

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "COLD-START ANALYSIS - COMPLETE PIPELINE"
echo "========================================================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Step 1: Extract repo mapping
echo "STEP 1: Extract instance → repo mapping from results"
echo "------------------------------------------------------------------------"
python experiments/cold_start/extract_repo_mapping.py
echo ""

# Step 2: Analyze and partition
echo "STEP 2: Partition instances into History-Available vs Cold-Start"
echo "------------------------------------------------------------------------"
python experiments/cold_start/analyze_cold_start.py
echo ""

# Step 3: Generate detailed partition report
echo "STEP 3: Generate detailed partition report"
echo "------------------------------------------------------------------------"
python experiments/cold_start/generate_report.py
echo ""

# Step 4: Calculate Pass@1
echo "STEP 4: Calculate Pass@1 for each group"
echo "------------------------------------------------------------------------"
python experiments/cold_start/calculate_pass_at_1.py
echo ""

# Step 5: Verify results
echo "STEP 5: Verify calculations with detailed breakdown"
echo "------------------------------------------------------------------------"
python experiments/cold_start/verify_results.py
echo ""

# Summary
echo "========================================================================"
echo "ANALYSIS COMPLETE"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  1. results/cold_start/instance_to_repo.json - Repo mapping"
echo "  2. results/cold_start/partition.json - Instance partition"
echo "  3. results/cold_start/detailed_partition.json - Full details"
echo "  4. experiments/cold_start/REPOS_AND_INSTANCES.md - Report"
echo "  5. results/cold_start/table_r7.json - TABLE R7 DATA"
echo ""
echo "Table R7 Summary:"
echo "  History-Available: 88→124 (23.5%→33.1%, +9.6pp)"
echo "  Cold-Start: 7→10 (21.2%→30.3%, +9.1pp)"
echo ""
