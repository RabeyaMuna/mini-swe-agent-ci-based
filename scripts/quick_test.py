#!/usr/bin/env python3
"""
quick_test.py - Quick test to verify data and results exist
==========================================================
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("=" * 70)
print("Quick Pre-Run Verification")
print("=" * 70)

checks = []

# Check 1: Eval set exists
eval_path = PROJECT_ROOT / "data" / "trs" / "eval_set.jsonl"
if eval_path.exists():
    with open(eval_path) as f:
        num_lines = sum(1 for line in f if line.strip())
    checks.append(("✓", f"eval_set.jsonl exists ({num_lines} issues)"))
else:
    checks.append(("✗", "eval_set.jsonl NOT FOUND"))

# Check 2: Decomposed issues exist
decomposed_path = PROJECT_ROOT / "data" / "trs" / "decomposed_issues.json"
if decomposed_path.exists():
    data = json.load(open(decomposed_path))
    checks.append(("✓", f"decomposed_issues.json exists ({len(data)} issues)"))
else:
    checks.append(("✗", "decomposed_issues.json NOT FOUND"))

# Check 3: Baseline results exist
baseline_path = PROJECT_ROOT / "results" / "miniswe-agent" / "minimax-m2.5" / "baseline" / "preds.json"
if baseline_path.exists():
    data = json.load(open(baseline_path))
    checks.append(("✓", f"Baseline preds.json exists ({len(data)} predictions)"))
else:
    checks.append(("⚠", "Baseline preds.json NOT FOUND - Task 3 will fail"))

# Check 4: Memory results exist
memory_path = PROJECT_ROOT / "results" / "miniswe-agent" / "minimax-m2.5" / "L1_L2_L3" / "preds.json"
if memory_path.exists():
    data = json.load(open(memory_path))
    checks.append(("✓", f"Memory preds.json exists ({len(data)} predictions)"))
else:
    checks.append(("⚠", "Memory preds.json NOT FOUND - Task 3 will fail"))

# Check 5: Git repos exist
repo_dir = PROJECT_ROOT / "repo"
if repo_dir.exists():
    num_repos = len(list(repo_dir.iterdir()))
    checks.append(("✓", f"Repo directory exists ({num_repos} repos)"))
else:
    checks.append(("⚠", "Repo directory NOT FOUND - Task 2 commit analysis will be limited"))

# Check 6: GitPython installed
try:
    import git
    checks.append(("✓", "gitpython installed"))
except ImportError:
    checks.append(("⚠", "gitpython NOT installed - run: pip install gitpython"))

print()
for status, message in checks:
    print(f"{status} {message}")

print("\n" + "=" * 70)
critical_failures = sum(1 for status, _ in checks if status == "✗")
warnings = sum(1 for status, _ in checks if status == "⚠")

if critical_failures > 0:
    print("❌ CRITICAL: Some required files are missing")
    print("   Cannot proceed with analyses")
elif warnings > 0:
    print("⚠️  WARNING: Some optional data missing")
    print("   Task 1 and Task 2 will work")
    print("   Task 3 requires prediction files")
else:
    print("✅ ALL CHECKS PASSED")
    print("   Ready to run: ./scripts/run_all_analyses.sh")

print("=" * 70)
