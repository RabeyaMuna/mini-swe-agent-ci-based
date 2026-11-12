#!/usr/bin/env python3
"""Strictly score human-verified problem-hunk attribution.

This script refuses to produce scores until every canonical problem-hunk pair
has a human label and every reconstructed problem has a human-confirmed
canonical alignment. Direct and shared labels are positive, unrelated is
negative, and uncertain is excluded from the primary analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_LABELS = {"direct", "shared", "unrelated", "uncertain"}
REVIEWED_STATUSES = {"reviewed", "adjudicated"}
METHODS = ("endpoint", "development", "bidirectional")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def episode_id(record: dict[str, Any]) -> str:
    return str(record.get("issue_id") or record.get("original_issue_id") or "")


def normalize_file(value: str) -> str:
    return re.sub(r"^\./", "", str(value or "").replace("\\", "/")).lower()


def files_match(left: str, right: str) -> bool:
    left_norm = normalize_file(left)
    right_norm = normalize_file(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    left_parts = left_norm.split("/")
    right_parts = right_norm.split("/")
    return (
        len(left_parts) >= 2
        and len(right_parts) >= 2
        and left_parts[-2:] == right_parts[-2:]
    )


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def bootstrap(
    rows: list[tuple[int, int, int, int]], seed: int = 20261111
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    values = {"precision": [], "recall": [], "f1": [], "accuracy": []}
    for _ in range(10_000):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        result = metrics(*(sum(row[index] for row in sample) for index in range(4)))
        for key in values:
            values[key].append(float(result[key]))
    return {
        key: [sorted(samples)[249], sorted(samples)[9749]]
        for key, samples in values.items()
    }


def validate_reference(
    rows: list[dict[str, str]], canonical_problems: dict[str, set[str]]
) -> tuple[dict[str, dict[tuple[str, str], str]], set[str]]:
    missing_labels: list[str] = []
    missing_annotators: list[str] = []
    invalid: list[str] = []
    duplicates: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    by_episode: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    annotators: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        current_id = row.get("episode_id", "").strip()
        problem_id = row.get("problem_id", "").strip()
        hunk_id = row.get("hunk_id", "").strip()
        label = row.get("label", "").strip().lower()
        annotator = row.get("annotator", "").strip()
        key = (current_id, problem_id, hunk_id)
        display = f"row {row_number} ({current_id}, {problem_id}, {hunk_id})"
        if key in seen:
            duplicates.append(display)
        seen.add(key)
        if not label:
            missing_labels.append(display)
        elif label not in REFERENCE_LABELS:
            invalid.append(f"{display}: invalid label {label!r}")
        if not annotator:
            missing_annotators.append(display)
        else:
            annotators.add(annotator)
        if problem_id not in canonical_problems.get(current_id, set()):
            invalid.append(f"{display}: unknown canonical problem")
        if label in REFERENCE_LABELS:
            by_episode[current_id][(problem_id, hunk_id)] = label

    errors = []
    if missing_labels:
        errors.append(f"{len(missing_labels)} reference pairs have no label")
    if missing_annotators:
        errors.append(f"{len(missing_annotators)} reference pairs have no annotator")
    if invalid:
        errors.append(f"{len(invalid)} reference rows are invalid")
    if duplicates:
        errors.append(f"{len(duplicates)} reference pairs are duplicated")
    if errors:
        examples = (missing_labels + missing_annotators + invalid + duplicates)[:5]
        raise ValueError("; ".join(errors) + ". Examples: " + "; ".join(examples))
    return by_episode, annotators


def validate_alignments(
    rows: list[dict[str, str]], canonical_problems: dict[str, set[str]]
) -> dict[tuple[str, str, str], str]:
    missing: list[str] = []
    invalid: list[str] = []
    mapping: dict[tuple[str, str, str], str] = {}
    for row_number, row in enumerate(rows, start=2):
        method = row.get("method", "").strip()
        current_id = row.get("episode_id", "").strip()
        predicted_id = row.get("predicted_problem_id", "").strip()
        status = row.get("manual_review_status", "").strip().lower()
        canonical_id = row.get("manual_canonical_problem_id", "").strip()
        display = f"row {row_number} ({method}, {current_id}, {predicted_id})"
        if status not in REVIEWED_STATUSES or not canonical_id:
            missing.append(display)
            continue
        if canonical_id.upper() == "UNMAPPED":
            canonical_id = ""
        elif canonical_id not in canonical_problems.get(current_id, set()):
            invalid.append(f"{display}: unknown canonical ID {canonical_id!r}")
            continue
        key = (method, current_id, predicted_id)
        if key in mapping:
            invalid.append(f"{display}: duplicate alignment")
            continue
        mapping[key] = canonical_id
    if missing or invalid:
        parts = []
        if missing:
            parts.append(f"{len(missing)} prediction alignments are not manually reviewed")
        if invalid:
            parts.append(f"{len(invalid)} prediction alignments are invalid")
        examples = (missing + invalid)[:5]
        raise ValueError("; ".join(parts) + ". Examples: " + "; ".join(examples))
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "results/02_human_attribution_50_repos",
    )
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--alignment-csv", type=Path)
    parser.add_argument(
        "--prediction-mode",
        choices=("native-hunks", "expand-files"),
        default="native-hunks",
        help=(
            "Require predicted supporting_hunk_ids (default), or conservatively "
            "expand each predicted affected file to all of its canonical hunks."
        ),
    )
    args = parser.parse_args()

    reference_path = args.reference_csv or args.bundle / "problem_hunk_pairs.csv"
    alignment_path = args.alignment_csv or args.bundle / "prediction_alignment.csv"
    problems = read_jsonl(args.bundle / "problems.jsonl")
    hunks = read_jsonl(args.bundle / "hunks.jsonl")
    canonical_problems: dict[str, set[str]] = defaultdict(set)
    hunks_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for problem in problems:
        canonical_problems[str(problem["episode_id"])].add(str(problem["problem_id"]))
    for hunk in hunks:
        hunks_by_episode[str(hunk["episode_id"])].append(hunk)

    try:
        reference, annotators = validate_reference(
            read_csv(reference_path), canonical_problems
        )
        alignments = validate_alignments(read_csv(alignment_path), canonical_problems)
    except ValueError as error:
        print(f"Human review incomplete: {error}")
        return 2

    trace_paths = {
        "endpoint": PROJECT_ROOT / "data/back_trs/decomposed_issues.json",
        "development": PROJECT_ROOT / "data/fwr_trs/decomposed_issues.json",
        "bidirectional": PROJECT_ROOT / "data/bidirect_trs/decomposed_issues.json",
    }
    traces = {
        method: {episode_id(item): item for item in read_json(path)}
        for method, path in trace_paths.items()
    }
    selected_ids = set(canonical_problems)
    if args.prediction_mode == "native-hunks":
        missing_hunk_predictions = [
            (method, current_id, str(predicted.get("problem_id")))
            for method in METHODS
            for current_id in selected_ids
            for predicted in traces[method].get(current_id, {}).get("problems", [])
            if not (
                predicted.get("supporting_hunk_ids") or predicted.get("hunk_ids")
            )
        ]
        if missing_hunk_predictions:
            examples = "; ".join(map(str, missing_hunk_predictions[:5]))
            print(
                "Native problem-hunk scoring unavailable: "
                f"{len(missing_hunk_predictions)} reconstructed problems have no "
                f"supporting_hunk_ids. Examples: {examples}. Rerun reconstruction "
                "with hunk IDs, or explicitly use --prediction-mode expand-files "
                "and report the result as file-to-hunk-expanded."
            )
            return 2
    label_counts = {
        label: sum(
            pair_label == label
            for episode_rows in reference.values()
            for pair_label in episode_rows.values()
        )
        for label in sorted(REFERENCE_LABELS)
    }
    summary: dict[str, Any] = {
        "status": "human-verified",
        "episodes": len(selected_ids),
        "annotators": sorted(annotators),
        "label_distribution": label_counts,
        "valid_reference_pairs": label_counts["direct"] + label_counts["shared"],
        "uncertain_pairs_excluded": label_counts["uncertain"],
        "prediction_mode": args.prediction_mode,
        "methods": {},
    }
    evidence: list[dict[str, Any]] = []

    for method in METHODS:
        predicted_by_episode: dict[str, set[tuple[str, str]]] = defaultdict(set)
        provenance: dict[tuple[str, str, str], dict[str, Any]] = {}
        for current_id in selected_ids:
            for predicted in traces[method].get(current_id, {}).get("problems", []):
                predicted_id = str(predicted.get("problem_id"))
                alignment_key = (method, current_id, predicted_id)
                if alignment_key not in alignments:
                    raise ValueError(f"Missing reviewed alignment: {alignment_key}")
                canonical_id = alignments[alignment_key]
                score_problem_id = canonical_id or f"UNMAPPED:{method}:{predicted_id}"
                if args.prediction_mode == "native-hunks":
                    requested_hunk_ids = {
                        str(value)
                        for value in (
                            predicted.get("supporting_hunk_ids")
                            or predicted.get("hunk_ids")
                            or []
                        )
                    }
                    matching_hunks = [
                        hunk
                        for hunk in hunks_by_episode[current_id]
                        if str(hunk["hunk_id"]) in requested_hunk_ids
                    ]
                    found_hunk_ids = {str(hunk["hunk_id"]) for hunk in matching_hunks}
                    unknown_hunk_ids = requested_hunk_ids - found_hunk_ids
                    if unknown_hunk_ids:
                        raise ValueError(
                            f"Unknown hunk IDs for {(method, current_id, predicted_id)}: "
                            f"{sorted(unknown_hunk_ids)}"
                        )
                    for hunk in matching_hunks:
                        pair = (score_problem_id, str(hunk["hunk_id"]))
                        predicted_by_episode[current_id].add(pair)
                        provenance[(current_id, *pair)] = {
                            "method": method,
                            "episode_id": current_id,
                            "predicted_problem_id": predicted_id,
                            "canonical_problem_id": canonical_id,
                            "affected_file": hunk["file_path"],
                            "hunk_id": hunk["hunk_id"],
                            "hunk_file": hunk["file_path"],
                        }
                else:
                    for affected_file in predicted.get("affected_files", []):
                        matching_hunks = [
                            hunk
                            for hunk in hunks_by_episode[current_id]
                            if files_match(str(affected_file), str(hunk["file_path"]))
                        ]
                        if not matching_hunks:
                            matching_hunks = [
                                {
                                    "hunk_id": (
                                        f"NO_HUNK:{normalize_file(str(affected_file))}"
                                    ),
                                    "file_path": str(affected_file),
                                }
                            ]
                        for hunk in matching_hunks:
                            pair = (score_problem_id, str(hunk["hunk_id"]))
                            predicted_by_episode[current_id].add(pair)
                            provenance[(current_id, *pair)] = {
                                "method": method,
                                "episode_id": current_id,
                                "predicted_problem_id": predicted_id,
                                "canonical_problem_id": canonical_id,
                                "affected_file": affected_file,
                                "hunk_id": hunk["hunk_id"],
                                "hunk_file": hunk["file_path"],
                            }

        episode_counts: list[tuple[int, int, int, int]] = []
        for current_id in sorted(selected_ids, key=int):
            labels = reference[current_id]
            gold = {pair for pair, label in labels.items() if label in {"direct", "shared"}}
            uncertain = {pair for pair, label in labels.items() if label == "uncertain"}
            negatives = {pair for pair, label in labels.items() if label == "unrelated"}
            predicted = {pair for pair in predicted_by_episode[current_id] if pair not in uncertain}
            tp_pairs = predicted & gold
            fp_pairs = predicted - gold
            fn_pairs = gold - predicted
            tn_pairs = negatives - predicted
            episode_counts.append(
                (len(tp_pairs), len(fp_pairs), len(fn_pairs), len(tn_pairs))
            )
            for pair in sorted(predicted | gold):
                outcome = "TP" if pair in tp_pairs else "FP" if pair in fp_pairs else "FN"
                evidence.append(
                    {
                        **provenance.get(
                            (current_id, *pair),
                            {
                                "method": method,
                                "episode_id": current_id,
                                "canonical_problem_id": pair[0],
                                "hunk_id": pair[1],
                            },
                        ),
                        "outcome": outcome,
                        "reference_label": labels.get(pair, "not_in_reference"),
                    }
                )
        result = metrics(*(sum(row[index] for row in episode_counts) for index in range(4)))
        result["predicted_pairs"] = sum(row[0] + row[1] for row in episode_counts)
        result["bootstrap_95_ci"] = bootstrap(episode_counts)
        summary["methods"][method] = result

    (args.bundle / "HUMAN_VERIFIED_SCORES.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.bundle / "human_verified_scoring_evidence.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in evidence:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = [
        "# Human-verified problem-hunk attribution results",
        "",
        f"Episodes: {summary['episodes']}; annotators: {', '.join(summary['annotators'])}.",
        f"Prediction mode: {summary['prediction_mode']}.",
        f"Valid reference pairs: {summary['valid_reference_pairs']}; "
        f"uncertain pairs excluded: {summary['uncertain_pairs_excluded']}.",
        "",
        "| Reconstruction | Predicted | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["methods"][method]
        report.append(
            f"| {method} | {row['predicted_pairs']} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {row['tn']} | {100 * row['precision']:.1f}% | "
            f"{100 * row['recall']:.1f}% | {100 * row['f1']:.1f}% | "
            f"{100 * row['accuracy']:.1f}% |"
        )
    report.extend(
        [
            "",
            "Primary outcomes are micro-averaged precision, recall, and F1.",
            "Pairwise accuracy is secondary because unrelated pairs dominate the candidate space.",
        ]
    )
    (args.bundle / "HUMAN_VERIFIED_RESULTS.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
