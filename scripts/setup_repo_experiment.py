#!/usr/bin/env python3
"""
Set up memory seed issues for a repository.

This resolves all issues for a repo from the dataset, then uses
generated_patches_success_only.json to split:
- memory seed issues: repo issues matching generated patch id or sha_fail
- eval issues: remaining repo issues

It does not overwrite data/trs/eval_issues.json.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve memory seed issues for a repo")
    parser.add_argument("--repo", required=True, help="Repo filter, e.g. camel, camel-ai/camel, flower")
    parser.add_argument(
        "--dataset",
        default="ci-benchmark-user/ci-repair-bench",
        help="Full benchmark dataset JSON path or HuggingFace dataset name",
    )
    parser.add_argument(
        "--patches-file",
        default="/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/generated_patches_list/generated_patches_success_only.json",
        help="Generated patches JSON used to select memory seeds by id or sha_fail",
    )
    parser.add_argument(
        "--output",
        default="data/trs/memory_seed_resolved.json",
        help="Output full issue records for memory building",
    )
    parser.add_argument(
        "--eval-output",
        default="",
        help="Output remaining repo issues for evaluation",
    )
    args = parser.parse_args()

    eval_output = args.eval_output
    if not eval_output:
        safe_repo = args.repo.replace("/", "_").replace(" ", "_")
        eval_output = f"data/trs/{safe_repo}_eval_remaining.json"

    cmd = [
        sys.executable,
        "scripts/resolve_memory_seed_issues.py",
        "--dataset",
        args.dataset,
        "--repo",
        args.repo,
        "--patches-file",
        args.patches_file,
        "--output",
        args.output,
        "--eval-output",
        eval_output,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        return result.returncode

    print("\nNext step:")
    print(
        "  ./scripts/build_memory_pipeline_cheap.sh "
        f"--seed-file {args.output} --dataset {args.dataset}"
    )
    print("\nMemory will be appended to the standard data/trs memory files.")
    print(f"\nEval issues saved to:\n  {eval_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
