#!/usr/bin/env python3
"""Evaluate CI-repair ablation preds.json files against eval-issue ground truth.

Metrics:
  Exact Match  predicted changed-file set equals ground-truth file set
  Top-1        first predicted changed file is in ground truth
  Top-3        any of the first 3 predicted changed files is in ground truth
  Precision    total matching predicted files / total predicted files

All metrics are computed from every non-empty generated patch in preds.json.
The optional jobs_success_diff.jsonl is only displayed as an extra count when
explicitly passed with --success-file; it never filters the evaluated patches.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        source_rows = payload if isinstance(payload, list) else list(payload.values())
    else:
        source_rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    for row in source_rows:
        if isinstance(row, dict):
            keys = [
                row.get("instance_id"),
                row.get("id"),
                row.get("sha_fail"),
                row.get("sha_original"),
            ]
            for key in keys:
                if key:
                    rows[str(key)] = row
    return rows


def load_preds(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items() if isinstance(value, dict)}
    if isinstance(payload, list):
        rows = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = row.get("id") or row.get("instance_id") or row.get("sha_fail")
            if key:
                rows[str(key)] = row
        return rows
    raise ValueError(f"Unsupported preds format in {path}")


def normalize_path(path: str) -> str:
    return path.strip().lstrip("/").replace("\\", "/")


def files_from_diff(diff: str) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/.+ b/(.+)$", diff or "", re.MULTILINE):
        path = normalize_path(match.group(1))
        if path and path not in files:
            files.append(path)
    return files


def ground_truth_files(row: dict[str, Any]) -> list[str]:
    raw = row.get("ground_truth_files") or row.get("changed_files") or []
    files: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            path = item.get("file") or item.get("file_path") or item.get("path")
        else:
            path = item
        path = normalize_path(str(path or ""))
        if path and path not in files:
            files.append(path)
    return files


def load_success_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("conclusion") or "").lower() == "success":
                count += 1
    return count


def evaluate_one(
    preds_path: Path,
    dataset: dict[str, dict[str, Any]],
    success_file: Path | None,
) -> dict[str, Any]:
    preds = load_preds(preds_path)
    success_count_from_file = load_success_count(success_file)

    evaluated = 0
    exact = 0
    top1 = 0
    top3 = 0
    precision_hits = 0
    predicted_file_count = 0
    generated_patches = 0

    for pred_id, pred in preds.items():
        pred_files = files_from_diff(str(pred.get("diff") or ""))
        if not pred_files:
            continue
        generated_patches += 1
        predicted_file_count += len(set(pred_files))

        gt_row = dataset.get(pred_id)
        if gt_row is None:
            gt_row = dataset.get(str(pred.get("sha_fail") or ""))
        if gt_row is None:
            gt_row = dataset.get(str(pred.get("id") or ""))
        if gt_row is None:
            continue

        gt_files = ground_truth_files(gt_row)
        if not gt_files:
            continue

        gt_set = set(gt_files)
        pred_set = set(pred_files)

        evaluated += 1

        if pred_set == gt_set:
            exact += 1
        if pred_files[0] in gt_set:
            top1 += 1
        if any(path in gt_set for path in pred_files[:3]):
            top3 += 1
        precision_hits += len(pred_set & gt_set)

    def pct(value: float) -> float:
        return round(100.0 * value, 2)

    precision = precision_hits / predicted_file_count if predicted_file_count else 0.0

    return {
        "ablation": preds_path.parent.name,
        "preds": str(preds_path),
        "success_file": str(success_file) if success_file else "",
        "generated_patches": generated_patches,
        "evaluated_patches": evaluated,
        "exact_match_count": exact,
        "top_1_count": top1,
        "top_3_count": top3,
        "exact_match_accuracy": pct(exact / evaluated) if evaluated else 0.0,
        "top_1_accuracy": pct(top1 / evaluated) if evaluated else 0.0,
        "top_3_accuracy": pct(top3 / evaluated) if evaluated else 0.0,
        "precision_hits": precision_hits,
        "predicted_file_count": predicted_file_count,
        "precision": pct(precision),
        "success_count": success_count_from_file,
    }


def print_markdown(rows: list[dict[str, Any]]) -> None:
    show_success = any(row["success_count"] is not None for row in rows)
    headers = [
        "Ablation",
        "Generated",
        "Evaluated",
        "Exact Match Acc",
        "Top-1 Acc",
        "Top-3 Acc",
        "Precision",
    ]
    if show_success:
        headers.append("Success Count")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = [
            row["ablation"],
            str(row["generated_patches"]),
            str(row["evaluated_patches"]),
            f"{row['exact_match_accuracy']:.2f}%",
            f"{row['top_1_accuracy']:.2f}%",
            f"{row['top_3_accuracy']:.2f}%",
            f"{row['precision']:.2f}%",
        ]
        if show_success:
            values.append("n/a" if row["success_count"] is None else str(row["success_count"]))
        print("| " + " | ".join(values) + " |")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "ablation",
        "preds",
        "success_file",
        "generated_patches",
        "evaluated_patches",
        "exact_match_count",
        "exact_match_accuracy",
        "top_1_count",
        "top_1_accuracy",
        "top_3_count",
        "top_3_accuracy",
        "precision_hits",
        "predicted_file_count",
        "precision",
        "success_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/trs/eval_issues.json",
        type=Path,
        help="Eval issues JSON/JSONL containing ground_truth_files or changed_files.",
    )
    parser.add_argument(
        "preds",
        nargs="+",
        type=Path,
        help="One or more preds.json files, e.g. results/l1/preds.json.",
    )
    parser.add_argument(
        "--success-file",
        type=Path,
        help=(
            "Optional jobs_success_diff.jsonl to display a separate success "
            "count. It is not used to filter preds.json."
        ),
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    parser.add_argument("--json", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    rows = []
    for preds_path in args.preds:
        rows.append(evaluate_one(preds_path, dataset, args.success_file))

    print_markdown(rows)
    if args.csv:
        write_csv(args.csv, rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
