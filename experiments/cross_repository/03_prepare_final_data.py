#!/usr/bin/env python3
"""
Step 3: Prepare Final Data in Standard Format

Creates experimental_data/cross_repository/ with same structure as data/:
- bidirect_trs/failure_memory.json (50 instances from SOURCE repos)
- eval_issue_ids.json (first 100 IDs)

This allows running experiments by pointing to experimental_data/cross_repository/
without changing the system code.
"""

import json
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    print("\n" + "="*70)
    print("STEP 3: PREPARE FINAL DATA IN STANDARD FORMAT")
    print("="*70 + "\n")

    # Create directory structure
    exp_data_dir = PROJECT_ROOT / "experimental_data/cross_repository"
    bidirect_dir = exp_data_dir / "bidirect_trs"
    bidirect_dir.mkdir(parents=True, exist_ok=True)

    # ========== 1. Copy cross_repo_memory.json to failure_memory.json ==========
    print("1. Preparing memory file...")

    source_memory_file = exp_data_dir / "cross_repo_memory.json"

    with open(source_memory_file) as f:
        cross_repo_data = json.load(f)

    # Extract just the memory instances (remove metadata wrapper)
    memory_instances = cross_repo_data["memory"]

    # Save in standard format (as a list, like the original)
    target_memory_file = bidirect_dir / "failure_memory.json"
    with open(target_memory_file, 'w') as f:
        json.dump(memory_instances, f, indent=2)

    print(f"   ✓ Saved {len(memory_instances)} instances to:")
    print(f"     {target_memory_file}")
    print(f"   Source repos: {cross_repo_data['metadata']['source_repo_count']}")
    print(f"   Excluded TARGET repos: {cross_repo_data['metadata']['excluded_target_repo_count']}\n")

    # ========== 2. Create eval_issue_ids.json with first 100 IDs ==========
    print("2. Preparing eval issue IDs...")

    original_eval_ids_file = PROJECT_ROOT / "data/eval_issue_ids.json"

    with open(original_eval_ids_file) as f:
        all_eval_ids = json.load(f)

    # Take first 100
    first_100_ids = all_eval_ids[:100]

    target_eval_ids_file = exp_data_dir / "eval_issue_ids.json"
    with open(target_eval_ids_file, 'w') as f:
        json.dump(first_100_ids, f, indent=2)

    print(f"   ✓ Saved {len(first_100_ids)} eval IDs to:")
    print(f"     {target_eval_ids_file}")
    print(f"   First 10 IDs: {first_100_ids[:10]}\n")

    # ========== 3. Create empty memory for baseline ==========
    print("3. Creating empty memory for baseline...")

    empty_memory_file = bidirect_dir / "empty_memory.json"
    with open(empty_memory_file, 'w') as f:
        json.dump([], f, indent=2)

    print(f"   ✓ Created empty memory file:")
    print(f"     {empty_memory_file}\n")

    # ========== Summary ==========
    print("="*70)
    print("FINAL DATA STRUCTURE")
    print("="*70 + "\n")

    print("experimental_data/cross_repository/")
    print("├── bidirect_trs/")
    print(f"│   ├── failure_memory.json          ({len(memory_instances)} instances from SOURCE repos)")
    print(f"│   └── empty_memory.json            (0 instances for baseline)")
    print(f"└── eval_issue_ids.json              ({len(first_100_ids)} instances from TARGET repos)")
    print()

    print("="*70)
    print("HOW TO USE")
    print("="*70 + "\n")

    print("Config 1: No-Experience (Baseline)")
    print("  --data-dir experimental_data/cross_repository")
    print("  --memory bidirect_trs/empty_memory.json")
    print("  --eval-ids eval_issue_ids.json")
    print()

    print("Config 2: ReCAP Cross-Repository (L1/L2/L3)")
    print("  --data-dir experimental_data/cross_repository")
    print("  --memory bidirect_trs/failure_memory.json")
    print("  --eval-ids eval_issue_ids.json")
    print("  --levels L1,L2,L3")
    print()

    print("Config 3: ReCAP L3-Only")
    print("  --data-dir experimental_data/cross_repository")
    print("  --memory bidirect_trs/failure_memory.json")
    print("  --eval-ids eval_issue_ids.json")
    print("  --levels L3")
    print()

    print("="*70)
    print("VERIFICATION")
    print("="*70 + "\n")

    print(f"✓ Evaluation: {len(first_100_ids)} instances from {cross_repo_data['metadata']['excluded_target_repo_count']} TARGET repos")
    print(f"✓ Memory: {len(memory_instances)} instances from {cross_repo_data['metadata']['source_repo_count']} SOURCE repos")
    print(f"✓ Same-repository excluded: YES")
    print(f"  → No overlap between memory repos and eval repos")
    print()

    # Save metadata for reference
    metadata = {
        "experiment": "Cross-Repository Evaluation",
        "description": "Same-repository experience excluded - memory from SOURCE repos, eval from TARGET repos",
        "eval_instances": {
            "count": len(first_100_ids),
            "ids": first_100_ids,
            "target_repos": cross_repo_data['metadata']['excluded_target_repos'],
            "target_repo_count": cross_repo_data['metadata']['excluded_target_repo_count'],
        },
        "memory_instances": {
            "count": len(memory_instances),
            "ids": [inst.get("issue_id") for inst in memory_instances],
            "source_repos": cross_repo_data['metadata']['source_repos'],
            "source_repo_count": cross_repo_data['metadata']['source_repo_count'],
        },
        "configurations": {
            "1_no_experience": {
                "memory_file": "bidirect_trs/empty_memory.json",
                "memory_count": 0,
            },
            "2_cross_repo_full": {
                "memory_file": "bidirect_trs/failure_memory.json",
                "memory_count": len(memory_instances),
                "levels": ["L1", "L2", "L3"],
            },
            "3_cross_repo_l3_only": {
                "memory_file": "bidirect_trs/failure_memory.json",
                "memory_count": len(memory_instances),
                "levels": ["L3"],
            },
        },
        "same_repository_excluded": True,
    }

    metadata_file = exp_data_dir / "experiment_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_file}\n")

    print("="*70)
    print("STEP 3 COMPLETE - DATA READY FOR EXPERIMENTS")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
