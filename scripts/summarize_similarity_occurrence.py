#!/usr/bin/env python3
"""Summarize TRS similarity/occurrence metadata by repository."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [row for row in payload.values() if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON payload in {path}")


def repo_name(row: dict[str, Any]) -> str:
    return str(row.get("repo_name") or row.get("repo") or "unknown")


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(seed_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eval_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seed_ids = {str(row.get("id")) for row in seed_rows if row.get("id") is not None}

    for row in seed_rows:
        seed_by_repo[repo_name(row)].append(row)
    for row in eval_rows:
        eval_by_repo[repo_name(row)].append(row)

    repos = sorted(set(seed_by_repo) | set(eval_by_repo))
    output: list[dict[str, Any]] = []
    for repo in repos:
        seeded = seed_by_repo.get(repo, [])
        evals = eval_by_repo.get(repo, [])
        all_rows = seeded + evals

        recurring = sum(1 for row in all_rows if int(row.get("recurrence_count") or 0) > 1)
        avg_sim_values = [
            float(row.get("avg_sim"))
            for row in evals
            if row.get("avg_sim") not in (None, "")
        ]

        retrieved_peers = 0
        seeded_peer_hits = 0
        for row in evals:
            for peer in row.get("top_memory_peers") or []:
                if not isinstance(peer, dict):
                    continue
                retrieved_peers += 1
                if str(peer.get("id")) in seed_ids:
                    seeded_peer_hits += 1

        issues = len(all_rows)
        output.append({
            "repo": repo,
            "issues": issues,
            "recurring_percent": round(100.0 * recurring / issues, 2) if issues else 0.0,
            "precision_percent": round(100.0 * seeded_peer_hits / retrieved_peers, 2) if retrieved_peers else 0.0,
            "avg_similarity": round(average(avg_sim_values), 4),
            "seeded_instances": len(seeded),
            "eval_instances": len(evals),
        })
    return output


def add_total_row(rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    seed_ids = {str(row.get("id")) for row in seed_rows if row.get("id") is not None}
    all_rows = seed_rows + eval_rows
    recurring = sum(1 for row in all_rows if int(row.get("recurrence_count") or 0) > 1)
    avg_sim_values = [
        float(row.get("avg_sim"))
        for row in eval_rows
        if row.get("avg_sim") not in (None, "")
    ]
    retrieved_peers = 0
    seeded_peer_hits = 0
    for row in eval_rows:
        for peer in row.get("top_memory_peers") or []:
            if not isinstance(peer, dict):
                continue
            retrieved_peers += 1
            if str(peer.get("id")) in seed_ids:
                seeded_peer_hits += 1

    issues = len(all_rows)
    rows.append({
        "repo": "TOTAL",
        "issues": issues,
        "recurring_percent": round(100.0 * recurring / issues, 2) if issues else 0.0,
        "precision_percent": round(100.0 * seeded_peer_hits / retrieved_peers, 2) if retrieved_peers else 0.0,
        "avg_similarity": round(average(avg_sim_values), 4),
        "seeded_instances": len(seed_rows),
        "eval_instances": len(eval_rows),
    })


def print_markdown(rows: list[dict[str, Any]]) -> None:
    headers = [
        "Repo",
        "Issues",
        "Recurring%",
        "Precision %",
        "Avg Similarity",
        "Seeded Instance",
        "Eval Instances",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join([
                str(row["repo"]),
                str(row["issues"]),
                f"{row['recurring_percent']:.2f}%",
                f"{row['precision_percent']:.2f}%",
                f"{row['avg_similarity']:.4f}",
                str(row["seeded_instances"]),
                str(row["eval_instances"]),
            ])
            + " |"
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "repo",
                "issues",
                "recurring_percent",
                "precision_percent",
                "avg_similarity",
                "seeded_instances",
                "eval_instances",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, default=Path("data/trs/trs_memory_seed_issues.json"))
    parser.add_argument("--eval", type=Path, default=Path("data/trs/eval_issues.json"))
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    parser.add_argument("--no-total", action="store_true", help="Do not append TOTAL row.")
    args = parser.parse_args()

    seed_rows = load_rows(args.seed)
    eval_rows = load_rows(args.eval)
    rows = summarize(seed_rows, eval_rows)
    if not args.no_total:
        add_total_row(rows, seed_rows, eval_rows)

    print_markdown(rows)
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
