#!/usr/bin/env python3
"""Generate reviewer-response Tables R4 and R5.

This analysis intentionally uses CI-Repair-Bench's canonical, paper-facing
``error_type`` labels from ``lca_dataset.parquet``. It does not use the
run-time ``data/log_details.json`` cache, whose categories differ from the
canonical dataset annotations.

Table R4 reports baseline versus bidirectional ReCAP for MiniMax-M2.5 and
DeepSeek-V4-Flash. Table R5 strictly removes an entire evaluation instance if
any of its labels is related to formatting, linting, or code quality.

Missing CI-result records are counted as unsuccessful repairs, so every rate
uses the intended evaluation/category denominator. Diagnostics and coverage
warnings are written to stderr; publication-ready Markdown is written to
stdout.

Usage:
    python verifications/failure_type_analysis.py
    python verifications/failure_type_analysis.py --output verifications/R4_R5.md
    python verifications/failure_type_analysis.py --dataset /path/to/lca_dataset.parquet
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT.parent / "CI-REPAIR-BENCH" / "dataset" / "lca_dataset.parquet"
)
DEFAULT_EVAL_IDS = PROJECT_ROOT / "data" / "eval_issue_ids.json"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "miniswe-agent"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    baseline_path: Path
    recap_path: Path


@dataclass(frozen=True)
class ResultData:
    outcomes: Mapping[str, bool]
    row_count: int
    duplicate_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="CI-Repair-Bench parquet containing canonical error_type labels",
    )
    parser.add_argument(
        "--eval-ids",
        type=Path,
        default=DEFAULT_EVAL_IDS,
        help="JSON list of evaluation instance IDs",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root containing baseline and bidirectional result directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path at which to save the generated Markdown",
    )
    return parser.parse_args()


def normalize_labels(value: object) -> tuple[str, ...]:
    """Convert a parquet label cell into a unique, sorted label tuple."""
    if value is None:
        return ()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped.startswith("["):
            try:
                value = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                value = [stripped]
        else:
            value = [stripped]
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return tuple(sorted({str(label).strip() for label in value if str(label).strip()}))


def load_eval_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        eval_ids = [str(value) for value in json.load(handle)]
    if not eval_ids:
        raise ValueError(f"No evaluation IDs found in {path}")
    if len(eval_ids) != len(set(eval_ids)):
        raise ValueError(f"Duplicate evaluation IDs found in {path}")
    return eval_ids


def load_canonical_labels(
    dataset_path: Path, eval_ids: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Load canonical paper-facing labels for every evaluation instance."""
    dataframe = pd.read_parquet(dataset_path, columns=["id", "error_type"])
    dataframe["id"] = dataframe["id"].astype(str)

    duplicate_ids = dataframe.loc[
        dataframe["id"].duplicated(keep=False), "id"
    ].tolist()
    if duplicate_ids:
        raise ValueError(
            f"Dataset contains duplicate IDs, including: {duplicate_ids[:10]}"
        )

    dataset_labels = {
        str(row.id): normalize_labels(row.error_type)
        for row in dataframe.itertuples(index=False)
    }
    missing_ids = [instance_id for instance_id in eval_ids if instance_id not in dataset_labels]
    if missing_ids:
        raise ValueError(
            f"Dataset is missing {len(missing_ids)} evaluation IDs: {missing_ids[:10]}"
        )

    empty_labels = [instance_id for instance_id in eval_ids if not dataset_labels[instance_id]]
    if empty_labels:
        raise ValueError(
            f"Dataset has empty error_type labels for {len(empty_labels)} evaluation "
            f"instances: {empty_labels[:10]}"
        )

    return {instance_id: dataset_labels[instance_id] for instance_id in eval_ids}


def load_results(path: Path) -> ResultData:
    """Load CI conclusions, permitting only identical duplicate outcomes."""
    outcomes: dict[str, bool] = {}
    duplicate_ids: list[str] = []
    row_count = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row_count += 1
            record = json.loads(line)
            instance_id = str(record.get("id", record.get("instance_id", ""))).strip()
            if not instance_id:
                raise ValueError(f"Missing instance ID in {path}:{line_number}")

            if "conclusion" in record:
                resolved = record["conclusion"] == "success"
            else:
                resolved = bool(record.get("resolved", False))

            if instance_id in outcomes:
                duplicate_ids.append(instance_id)
                if outcomes[instance_id] != resolved:
                    raise ValueError(
                        f"Conflicting duplicate outcomes for ID {instance_id} in {path}"
                    )
            outcomes[instance_id] = resolved

    return ResultData(outcomes, row_count, tuple(sorted(set(duplicate_ids))))


def build_model_specs(results_root: Path) -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            key="minimax",
            display_name="MiniMax-M2.5",
            baseline_path=results_root
            / "baseline_minimax-m2.5"
            / "jobs_results_diff.jsonl",
            recap_path=results_root
            / "bidirectional"
            / "l1_l2_l3_minimax-m2.5"
            / "L1_L2_L3"
            / "jobs_results_diff.jsonl",
        ),
        ModelSpec(
            key="deepseek",
            display_name="DeepSeek-V4-Flash",
            baseline_path=results_root
            / "baseline_deepseek-v4-flash"
            / "jobs_results_diff.jsonl",
            recap_path=results_root
            / "bidirectional"
            / "l1_l2_l3_deepseek-v4-flash"
            / "L1_L2_L3"
            / "jobs_results_diff.jsonl",
        ),
    )


def percentage(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else 0.0


def successes(outcomes: Mapping[str, bool], instance_ids: Iterable[str]) -> int:
    # A missing CI-result record is deliberately counted as unsuccessful.
    return sum(bool(outcomes.get(instance_id, False)) for instance_id in instance_ids)


def is_excluded_quality_label(label: str) -> bool:
    """Identify formatting, linting, and code-quality labels conservatively."""
    normalized = label.casefold()
    return any(token in normalized for token in ("format", "lint", "quality"))


def format_rate(value: float) -> str:
    return f"{value:.1f}%"


def format_gain(value: float) -> str:
    return f"{value:+.1f}"


def render_tables(
    eval_ids: Sequence[str],
    labels: Mapping[str, tuple[str, ...]],
    specs: Sequence[ModelSpec],
    result_sets: Mapping[str, Mapping[str, ResultData]],
) -> str:
    category_counts = Counter(
        category for instance_id in eval_ids for category in labels[instance_id]
    )
    ordered_categories = [
        category
        for category, _ in sorted(
            category_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    lines = [
        "**Table R4. Repair success rates by failure type for baseline and "
        "bidirectional ReCAP.**",
        "",
        "| Failure type | Instances | MiniMax Baseline | MiniMax ReCAP (Bi.) | "
        "Gain (pp) | DeepSeek Baseline | DeepSeek ReCAP (Bi.) | Gain (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for category in ordered_categories:
        category_ids = [
            instance_id
            for instance_id in eval_ids
            if category in labels[instance_id]
        ]
        count = len(category_ids)
        cells = [category, str(count)]
        for spec in specs:
            baseline = result_sets[spec.key]["baseline"].outcomes
            recap = result_sets[spec.key]["recap"].outcomes
            baseline_rate = percentage(successes(baseline, category_ids), count)
            recap_rate = percentage(successes(recap, category_ids), count)
            cells.extend(
                [
                    format_rate(baseline_rate),
                    format_rate(recap_rate),
                    format_gain(recap_rate - baseline_rate),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")

    overall_cells = ["**Overall**", f"**{len(eval_ids)}**"]
    for spec in specs:
        baseline = result_sets[spec.key]["baseline"].outcomes
        recap = result_sets[spec.key]["recap"].outcomes
        baseline_rate = percentage(successes(baseline, eval_ids), len(eval_ids))
        recap_rate = percentage(successes(recap, eval_ids), len(eval_ids))
        overall_cells.extend(
            [
                f"**{format_rate(baseline_rate)}**",
                f"**{format_rate(recap_rate)}**",
                f"**{format_gain(recap_rate - baseline_rate)}**",
            ]
        )
    lines.append("| " + " | ".join(overall_cells) + " |")

    retained_ids = [
        instance_id
        for instance_id in eval_ids
        if not any(is_excluded_quality_label(label) for label in labels[instance_id])
    ]
    excluded_count = len(eval_ids) - len(retained_ids)

    lines.extend(
        [
            "",
            "*Note:* Categories use the canonical `error_type` annotations from "
            "CI-Repair-Bench and are not mutually exclusive. ReCAP (Bi.) denotes "
            "the bidirectional ReCAP configuration. Gains are percentage-point "
            "differences.",
            "",
            "**Table R5. Repair performance excluding linting, formatting, and "
            "code-quality instances.**",
            "",
            "| Model | Instances | Baseline | ReCAP (Bi.) | Gain (pp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for spec in specs:
        baseline = result_sets[spec.key]["baseline"].outcomes
        recap = result_sets[spec.key]["recap"].outcomes
        baseline_resolved = successes(baseline, retained_ids)
        recap_resolved = successes(recap, retained_ids)
        baseline_rate = percentage(baseline_resolved, len(retained_ids))
        recap_rate = percentage(recap_resolved, len(retained_ids))
        lines.append(
            "| "
            + " | ".join(
                [
                    spec.display_name,
                    str(len(retained_ids)),
                    format_rate(baseline_rate),
                    format_rate(recap_rate),
                    format_gain(recap_rate - baseline_rate),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            f"Strict instance-level filtering excludes {excluded_count} of "
            f"{len(eval_ids)} instances and retains {len(retained_ids)}. An "
            "instance is excluded if any canonical label contains `format`, "
            "`lint`, or `quality` (case-insensitive). In the canonical taxonomy, "
            "code-quality failures are represented by `Linting`.",
        ]
    )
    return "\n".join(lines) + "\n"


def print_diagnostics(
    eval_ids: Sequence[str],
    dataset_path: Path,
    specs: Sequence[ModelSpec],
    result_sets: Mapping[str, Mapping[str, ResultData]],
) -> None:
    print(
        f"Canonical labels: {dataset_path} ({len(eval_ids)}/{len(eval_ids)} eval IDs)",
        file=sys.stderr,
    )
    for spec in specs:
        for condition in ("baseline", "recap"):
            result = result_sets[spec.key][condition]
            coverage = sum(instance_id in result.outcomes for instance_id in eval_ids)
            missing = len(eval_ids) - coverage
            duplicate_note = (
                f"; duplicate IDs={list(result.duplicate_ids)}"
                if result.duplicate_ids
                else ""
            )
            print(
                f"{spec.display_name} {condition}: {coverage}/{len(eval_ids)} unique "
                f"eval results; missing={missing}; rows={result.row_count}"
                f"{duplicate_note}",
                file=sys.stderr,
            )


def main() -> None:
    args = parse_args()
    eval_ids = load_eval_ids(args.eval_ids)
    labels = load_canonical_labels(args.dataset, eval_ids)
    specs = build_model_specs(args.results_root)

    result_sets: dict[str, dict[str, ResultData]] = {}
    for spec in specs:
        result_sets[spec.key] = {
            "baseline": load_results(spec.baseline_path),
            "recap": load_results(spec.recap_path),
        }

    print_diagnostics(eval_ids, args.dataset, specs, result_sets)
    markdown = render_tables(eval_ids, labels, specs, result_sets)
    print(markdown, end="")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote Markdown: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
