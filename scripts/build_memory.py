#!/usr/bin/env python3
"""
Build L1/L2/L3 memory from existing decomposed_issues.json
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from decompose_ci_failure import generate_l1_l2_l3_pipeline, LitellmModel, _save_to_memory_files

def main():
    # Load decomposed issues
    decomposed_path = PROJECT_ROOT / "data/trs/decomposed_issues.json"

    if not decomposed_path.exists():
        print(f"ERROR: {decomposed_path} not found!")
        print("Run decomposition first or provide decomposed issues")
        return 1

    print("Loading decomposed issues...")
    with open(decomposed_path) as f:
        decomposed_data = json.load(f)

    print(f"Found {len(decomposed_data)} decomposed issues")

    # Create LLM
    print("\nInitializing LLM...")
    llm = LitellmModel("openrouter/minimax/minimax-m2.5")

    results = []

    # Process each issue
    for i, issue in enumerate(decomposed_data, 1):
        issue_id = issue.get('original_issue_id', '?')
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(decomposed_data)}: Issue {issue_id}")
        print(f"{'='*80}")

        try:
            # Build L1/L2/L3
            result = generate_l1_l2_l3_pipeline(issue, llm)
            results.append(result)

            # Save after each issue
            _save_to_memory_files(results, PROJECT_ROOT / 'data/trs')

        except Exception as e:
            print(f"ERROR processing issue {issue_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"Processed: {len(results)} issues")
    print(f"\nMemory files saved to: {PROJECT_ROOT / 'data/trs'}/")
    print(f"  - failure_memory.json (L1 file-level problems)")
    print(f"  - repo_memory.json (L2 repair sequences)")
    print(f"  - cross_memory.json (L3 analysis)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
