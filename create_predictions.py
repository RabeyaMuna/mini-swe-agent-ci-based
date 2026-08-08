#!/usr/bin/env python3
"""Manually create predictions.json from result.json files."""
import json
from pathlib import Path

def prediction_from_result(result_file: Path):
    issue_dir = result_file.parent

    with open(result_file) as f:
        result = json.load(f)

    patch_file = issue_dir / "patch.diff"
    if patch_file.exists():
        patch_content = patch_file.read_text()
    else:
        patch_content = ""

    return {
        "id": result["id"],
        "sha_fail": result["sha_fail"],
        "repo": result["repo"],
        "diff": patch_content,
        "ablation": result.get("ablation", "unknown"),
        "patch_generated": result["patch_generated"],
        "patch_bytes": result["patch_bytes"],
        "changed_files": result["changed_files"],
        "verification_passed": result.get("verification_passed"),
    }

output_root = Path("results/codex")
ablation = "baseline"
results_dir = output_root / ablation
predictions_file = results_dir / "predictions.json"

predictions = [
    prediction_from_result(result_file)
    for result_file in sorted(results_dir.glob("*/result.json"))
]

if predictions:
    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"✓ Created {predictions_file} with {len(predictions)} predictions")
else:
    print("✗ No result.json files found")
