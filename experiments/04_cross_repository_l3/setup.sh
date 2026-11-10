#!/bin/bash
# Build the repository-disjoint L1/L2/L3 data and verify all invariants.

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

python3 experiments/04_cross_repository_l3/prepare_cross_repo_data.py
