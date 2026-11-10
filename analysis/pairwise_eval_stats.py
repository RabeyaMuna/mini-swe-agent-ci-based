#!/usr/bin/env python3
"""Compute paired evaluation statistics for direction comparisons.

This script reads a CSV of per-ID outcomes and reports:
- overall success counts by method
- available coverage for each method
- paired coverage on the common evaluated IDs
- pairwise McNemar exact test on common IDs

Usage:
    python analysis/pairwise_eval_stats.py --csv analysis/comparison_master.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_success(value: str | None) -> bool | None:
    if value is None:
        return None
    v = str(value).strip()
    if v == "":
        return None
    lower = v.lower()
    if lower in {"1", "true", "success", "successes", "passed", "pass", "yes", "y", "✓"}:
        return True
    if lower in {"0", "false", "failure", "fail", "failed", "no", "n", "✗"}:
        return False
    raise ValueError(f"Cannot parse success flag: {value!r}")


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return list(reader)


def load_subset_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            ids.add(value)
    return ids


def filter_rows_by_ids(rows: List[Dict[str, str]], subset_ids: set[str] | None) -> List[Dict[str, str]]:
    if subset_ids is None:
        return rows
    filtered: List[Dict[str, str]] = []
    for row in rows:
        row_id = row.get("ID") or row.get("id") or row.get("instance_id")
        if str(row_id) in subset_ids:
            filtered.append(row)
    return filtered


def success_counts(rows: Iterable[Dict[str, str]], method_names: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for method in method_names:
        count = 0
        for row in rows:
            parsed = parse_success(row.get(method))
            if parsed is True:
                count += 1
        out[method] = count
    return out


def available_counts(rows: Iterable[Dict[str, str]], method_names: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for method in method_names:
        count = 0
        for row in rows:
            parsed = parse_success(row.get(method))
            if parsed is not None:
                count += 1
        out[method] = count
    return out


def paired_ids_for_pair(rows: List[Dict[str, str]], method_a: str, method_b: str) -> List[str]:
    ids: List[str] = []
    seen = set()
    for row in rows:
        row_id = row.get("ID") or row.get("id") or row.get("instance_id")
        if row_id is None:
            continue
        a = parse_success(row.get(method_a))
        b = parse_success(row.get(method_b))
        if a is None or b is None:
            continue
        if row_id not in seen:
            seen.add(row_id)
            ids.append(str(row_id))
    return ids


def pairwise_vector(rows: List[Dict[str, str]], method_a: str, method_b: str) -> Tuple[List[str], List[bool], List[bool]]:
    ids: List[str] = []
    a_vals: List[bool] = []
    b_vals: List[bool] = []
    for row in rows:
        row_id = row.get("ID") or row.get("id") or row.get("instance_id")
        if row_id is None:
            continue
        a = parse_success(row.get(method_a))
        b = parse_success(row.get(method_b))
        if a is None or b is None:
            continue
        ids.append(str(row_id))
        a_vals.append(bool(a))
        b_vals.append(bool(b))
    return ids, a_vals, b_vals


def exact_mcnemar_p_value(a_vals: List[bool], b_vals: List[bool]) -> Tuple[int, int, float]:
    """Return (n10, n01, p_value) for exact McNemar two-sided test.

    n10 = count where A=1, B=0
    n01 = count where A=0, B=1
    P-value for exact two-sided McNemar is computed from Binomial(n=n10+n01, p=0.5)
    with the more extreme tail probability doubled.
    """
    n10 = sum(1 for a, b in zip(a_vals, b_vals) if a and not b)
    n01 = sum(1 for a, b in zip(a_vals, b_vals) if (not a) and b)
    total = n10 + n01
    if total == 0:
        return n10, n01, 1.0

    # Two-sided exact p-value from binomial( total, 0.5 )
    # Use the probability mass at k = min(n10, n01) and all more extreme values.
    # For a symmetric distribution, the p-value is 2 * P(X <= min(n10, n01)).
    k = min(n10, n01)
    p = 0.0
    for x in range(k + 1):
        p += math.comb(total, x) * (0.5 ** total)
    p_value = 2.0 * p
    if p_value > 1.0:
        p_value = 1.0
    return n10, n01, p_value


def format_pct(numer: int, denom: int) -> str:
    if denom == 0:
        return "0.00%"
    return f"{(numer / denom) * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("analysis/comparison_master.csv"), help="CSV with per-ID method statuses")
    parser.add_argument("--methods", nargs="+", default=["Forward_Success", "Backward_Success", "Bidirectional_Success"], help="Method columns to evaluate")
    parser.add_argument("--subset-ids", type=Path, default=None, help="Optional ID list file to restrict the analysis to a specific evaluated subset (one ID per line)")
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        return 2

    subset_ids = load_subset_ids(args.subset_ids)
    rows = read_rows(csv_path)
    rows = filter_rows_by_ids(rows, subset_ids)
    method_names = args.methods

    print(f"Loaded {len(rows)} rows from {csv_path}")
    if subset_ids is not None:
        print(f"Subset filter applied: {len(subset_ids)} requested IDs, {len(rows)} rows matched")
    print()

    overall_success = success_counts(rows, method_names)
    overall_available = available_counts(rows, method_names)
    print("Counts on the selected evaluation subset:")
    for method in method_names:
        print(f"  {method}: {overall_success[method]} / {overall_available[method]} = {format_pct(overall_success[method], overall_available[method])}")

    print()
    print("Coverage by method:")
    for method in method_names:
        print(f"  {method}: {overall_available[method]} evaluated IDs")

    print()
    print("Pairwise paired comparisons on common evaluated IDs:")
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            a = method_names[i]
            b = method_names[j]
            ids, a_vals, b_vals = pairwise_vector(rows, a, b)
            n10, n01, p = exact_mcnemar_p_value(a_vals, b_vals)
            n11 = sum(1 for aa, bb in zip(a_vals, b_vals) if aa and bb)
            n00 = sum(1 for aa, bb in zip(a_vals, b_vals) if (not aa) and (not bb))
            print(f"  {a} vs {b}: common IDs = {len(ids)}")
            print(f"    table: A success/B success = {n11}, A success/B fail = {n10}, A fail/B success = {n01}, A fail/B fail = {n00}")
            print(f"    exact McNemar p = {p:.6g}")
            if n10 > n01:
                print(f"    direction: {a} favors {a} over {b} on discordant pairs ({n10} vs {n01})")
            elif n01 > n10:
                print(f"    direction: {b} favors {b} over {a} on discordant pairs ({n01} vs {n10})")
            else:
                print(f"    direction: no directional advantage on discordant pairs ({n10} vs {n01})")
            print()

    # Common IDs across all methods.
    common_ids = None
    for row in rows:
        row_id = row.get("ID") or row.get("id") or row.get("instance_id")
        if row_id is None:
            continue
        vals = [parse_success(row.get(method)) for method in method_names]
        if all(v is not None for v in vals):
            if common_ids is None:
                common_ids = {str(row_id)}
            else:
                common_ids.add(str(row_id))
    if common_ids is not None:
        print(f"All-method common evaluated set: {len(common_ids)} IDs")
    else:
        print("All-method common evaluated set: 0 IDs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
