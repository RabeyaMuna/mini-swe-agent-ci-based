#!/usr/bin/env python3
"""
fetch_dataset.py
================
Download the CI-REPAIR-BENCH dataset from HuggingFace and save it as a
JSONL file ready for use with `mini-swe-agent cibench`.

Usage
-----
    python scripts/fetch_dataset.py                        # saves to data/ci_dataset.jsonl
    python scripts/fetch_dataset.py --split test           # only the test split
    python scripts/fetch_dataset.py --out data/train.jsonl --split train

Requirements
------------
    pip install datasets huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HF_DATASET = "ci-benchmark-user/ci-repair-bench"

# Required fields that every instance must have for cibench to work.
REQUIRED_FIELDS = [
    "instance_id",
    "sha_fail",
    "repo_owner",
    "repo_name",
    "workflow_path",
    "workflow_name",
    "workflow",
    "logs",
]


def fetch(split: str, output_path: Path) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run:  pip install datasets huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"[fetch_dataset] Loading '{HF_DATASET}' split='{split}' from HuggingFace …")
    ds = load_dataset(HF_DATASET, split=split)
    total = len(ds)
    print(f"[fetch_dataset] Downloaded {total} instances.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_any = False
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for row in ds:
            record = dict(row)
            # Warn on first instance with missing fields
            if not missing_any:
                missing = [f for f in REQUIRED_FIELDS if f not in record]
                if missing:
                    print(
                        f"[fetch_dataset] WARNING: first instance is missing fields: {missing}. "
                        "cibench may skip or fail those instances.",
                        file=sys.stderr,
                    )
                    missing_any = True
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[fetch_dataset] Saved {written} instances -> {output_path}")
    print()
    print("Run the benchmark with:")
    print(f"  mini-swe-agent cibench --dataset {output_path} --output results/baseline --no-memory-enabled")
    print(f"  mini-swe-agent cibench --dataset {output_path} --output results/l1_l2_l3 \\")
    print( "      --memory-enabled --memory-root data/trs --memory-ablation L1+L2+L3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CI-REPAIR-BENCH from HuggingFace.")
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download (default: train). Use 'train' or 'all'.",
    )
    parser.add_argument(
        "--out",
        default="data/ci_dataset.jsonl",
        help="Output JSONL file path (default: data/ci_dataset.jsonl).",
    )
    args = parser.parse_args()

    # If split is 'all', fetch every available split into one file
    out_path = Path(args.out)
    if args.split == "all":
        from datasets import load_dataset, get_dataset_split_names  # type: ignore
        splits = get_dataset_split_names(HF_DATASET)
        print(f"[fetch_dataset] Fetching all splits: {splits}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for split in splits:
                ds = load_dataset(HF_DATASET, split=split)
                for row in ds:
                    fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                    written += 1
                print(f"[fetch_dataset]   {split}: {len(ds)} instances")
        print(f"[fetch_dataset] Total saved: {written} -> {out_path}")
    else:
        fetch(args.split, out_path)


if __name__ == "__main__":
    main()
