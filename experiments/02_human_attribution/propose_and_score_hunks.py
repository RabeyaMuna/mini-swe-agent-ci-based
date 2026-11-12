#!/usr/bin/env python3
"""Propose auditable problem-hunk labels and score expanded file predictions.

The labels produced here are model-assisted proposals for subsequent human
verification.  They MUST NOT be described as human annotations before review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_LABELS = {"direct", "shared", "unrelated", "uncertain"}
FILE_WIDE_MARKERS = (
    "black",
    "formatting",
    "formatter",
    "ruff-format",
    "pyupgrade",
    "end-of-file-fixer",
    "trailing whitespace",
    "prettier",
    "isort",
)
REVIEW_OVERRIDES = {
    # Clear coordinated edits missed by the generic location/text rules.
    ("7:P001", "7:H004:af6d09804b42"): ("direct", "Moves use of the extracted fake_isfile_no_nmconn helper into the second affected test."),
    ("7:P001", "7:H005:bd291164c25b"): ("direct", "Removes the duplicated local helper after moving it to class scope."),
    ("58:P001", "58:H001:f3cca11c1a36"): ("direct", "Applies the same categorical-level fix to the related weighted relplot test."),
    ("231:P001", "231:H001:01c8d0513097"): ("direct", "Adds the blank line required by the reported E302 violation."),
    ("231:P001", "231:H002:ad8cce3d6afe"): ("direct", "Corrects the continuation indentation reported as E127."),
    ("244:P001", "244:H003:aa4b9dc96df9"): ("direct", "Handles missing conformity-test classes described by the failure evidence."),
    ("244:P001", "244:H004:801bfd35c035"): ("direct", "Handles missing conformity-test attributes described by the failure evidence."),
    ("397:P001", "397:H001:509bc5819724"): ("direct", "Updates the first caller as part of renaming the incompatible override."),
    ("397:P001", "397:H003:cc706ab281e6"): ("direct", "Updates the second caller as part of renaming the incompatible override."),
    **{
        (f"411:P{index:03d}", hunk_id): ("direct", reason)
        for index in range(1, 6)
        for hunk_id, reason in (
            ("411:H001:09215481f571", "Removes the ChatAgent import from the async toolkit, breaking the reported circular-import chain."),
            ("411:H002:dcc5d57a92d3", "Removes the ChatAgent import from the browser toolkit, breaking the reported circular-import chain."),
        )
    },
    ("425:P002", "425:H001:3adb4cc5619c"): ("direct", "Repairs the import-order violations included in this multi-violation problem."),
    ("482:P001", "482:H003:43a38ece21a2"): ("direct", "Updates the assertion to the one-row local dataset introduced by the repair."),
    ("570:P001", "570:H002:ea9c104dfb49"): ("unrelated", "Formatting cleanup outside the reported I001 import block."),
}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def episode_id(record: dict[str, Any]) -> str:
    return str(record.get("issue_id") or record.get("original_issue_id") or "")


def problem_text(problem: dict[str, Any]) -> str:
    return " ".join(
        str(problem.get(key) or "")
        for key in (
            "problem",
            "root_cause",
            "how_fixed",
            "failure_type",
            "issue_type",
            "validation_cmd",
            "reason",
            "failed_tool",
            "failed_cmd",
        )
    )


def text_similarity(left: str, right: str) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    try:
        matrix = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w.-]+\b",
        ).fit_transform([left, right])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_range(value: str) -> tuple[int, int]:
    start, count = value.split(",", 1)
    return int(start), int(count)


def range_contains(start: int, count: int, line: int) -> bool:
    return start <= line <= start + max(count, 1) - 1


def distance_to_range(start: int, count: int, line: int) -> int:
    end = start + max(count, 1) - 1
    if start <= line <= end:
        return 0
    return min(abs(line - start), abs(line - end))


def identifier_overlap(reason: str, content: str) -> list[str]:
    explicit = re.findall(r"`([^`]{2,80})`", reason)
    symbols = re.findall(r"\b[A-Za-z_][A-Za-z0-9_.]{3,}\b", reason)
    stop = {
        "contains",
        "reported",
        "caused",
        "failed",
        "failure",
        "error",
        "because",
        "during",
        "which",
        "where",
        "directly",
        "file",
        "line",
    }
    candidates = explicit + [
        item
        for item in symbols
        if item.lower() not in stop
        and ("_" in item or "." in item)
    ]
    lowered = content.lower()
    return sorted({item for item in candidates if item.lower() in lowered})


def looks_like_import_fix(row: dict[str, str]) -> bool:
    reason = f"{row['issue_type']} {row['problem_reason']}".lower()
    if "i001" not in reason and "import block" not in reason:
        return False
    content = row["hunk_content"]
    return bool(re.search(r"^[+-](?:from|import)\s", content, flags=re.MULTILINE))


def label_problem_group(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Label every hunk candidate for one canonical problem."""
    same_file = [row for row in rows if files_match(row["problem_file"], row["hunk_file"])]
    reason = " ".join(
        [
            rows[0]["failed_tool"],
            rows[0]["issue_type"],
            rows[0]["problem_reason"],
        ]
    ).lower()
    problem_line = parse_int(rows[0]["problem_line"])

    evidence: dict[str, dict[str, Any]] = {}
    for row in rows:
        is_same_file = files_match(row["problem_file"], row["hunk_file"])
        old_start, old_count = parse_range(row["old_range"])
        new_start, new_count = parse_range(row["new_range"])
        overlap = False
        proximity: int | None = None
        if problem_line is not None and is_same_file:
            overlap = range_contains(old_start, old_count, problem_line) or range_contains(
                new_start, new_count, problem_line
            )
            proximity = min(
                distance_to_range(old_start, old_count, problem_line),
                distance_to_range(new_start, new_count, problem_line),
            )
        identifiers = identifier_overlap(rows[0]["problem_reason"], row["hunk_content"])
        similarity = text_similarity(rows[0]["problem_reason"], row["hunk_content"])
        score = similarity + min(len(identifiers), 3) * 0.12
        if overlap:
            score += 0.8
        elif proximity is not None:
            score += 0.25 * math.exp(-proximity / 30)
        if looks_like_import_fix(row):
            score += 0.5
        if is_same_file:
            score += 0.12
        evidence[row["hunk_id"]] = {
            "same_file": is_same_file,
            "line_overlap": overlap,
            "line_distance": proximity,
            "identifiers": identifiers,
            "text_similarity": similarity,
            "score": score,
        }

    max_score = max((value["score"] for value in evidence.values()), default=0.0)
    max_cross_file_score = max(
        (value["score"] for value in evidence.values() if not value["same_file"]),
        default=0.0,
    )
    file_wide = any(marker in reason for marker in FILE_WIDE_MARKERS)
    import_specific = "i001" in reason or "import block" in reason

    output: list[dict[str, str]] = []
    for original in rows:
        row = dict(original)
        item = evidence[row["hunk_id"]]
        override_key = (row["problem_id"], row["hunk_id"])
        if override_key in REVIEW_OVERRIDES:
            row["label"], override_reason = REVIEW_OVERRIDES[override_key]
            row["confidence"] = "high"
            row["rationale"] = "Evidence review override: " + override_reason
            row["evidence_score"] = f"{item['score']:.4f}"
            output.append(row)
            continue
        direct = False
        confidence = "medium"
        rationale = ""
        if not item["same_file"]:
            is_best_cross_file = item["score"] == max_cross_file_score
            if item["identifiers"] and item["score"] >= max_cross_file_score * 0.75:
                direct = True
                confidence = "medium"
                rationale = "Cross-file repair evidence contains problem-specific identifiers: " + ", ".join(
                    item["identifiers"][:5]
                )
            elif is_best_cross_file and item["text_similarity"] >= 0.18:
                direct = True
                confidence = "medium"
                rationale = "This is the strongest cross-file semantic repair match in the episode."
            elif is_best_cross_file and not same_file and item["score"] >= 0.08:
                row["label"] = "uncertain"
                row["confidence"] = "low"
                row["rationale"] = "Possible cross-file repair, but evidence is insufficient for a direct label."
                row["evidence_score"] = f"{item['score']:.4f}"
                output.append(row)
                continue
            else:
                row["label"] = "unrelated"
                row["confidence"] = "high"
                row["rationale"] = "Different file and no sufficiently specific cross-file repair evidence."
                row["evidence_score"] = f"{item['score']:.4f}"
                output.append(row)
                continue
        elif item["line_overlap"]:
            direct = True
            confidence = "high"
            rationale = "CI problem line overlaps this changed hunk."
        elif import_specific and looks_like_import_fix(row):
            direct = True
            confidence = "high"
            rationale = "The reported import-order problem and this import-changing hunk agree."
        elif file_wide and not import_specific:
            direct = True
            confidence = "medium"
            rationale = "The formatter/pre-commit report names this file and this hunk is part of that file rewrite."
        elif len(same_file) == 1:
            direct = True
            confidence = "medium"
            rationale = "This is the only changed hunk in the file named by the CI problem."
        elif item["identifiers"] and item["score"] >= max_score * 0.55:
            direct = True
            confidence = "medium"
            rationale = "Problem-specific identifiers occur in this hunk: " + ", ".join(
                item["identifiers"][:5]
            )
        elif item["line_distance"] is not None and item["line_distance"] <= 12:
            direct = True
            confidence = "medium"
            rationale = f"This hunk is {item['line_distance']} line(s) from the reported location."
        elif item["score"] == max_score and max_score >= 0.08:
            direct = True
            confidence = "medium"
            rationale = "This is the strongest semantic/location match among hunks in the reported file."

        if direct:
            row["label"] = "direct"
            row["confidence"] = confidence
            row["rationale"] = rationale
        elif item["score"] == max_score and max_score < 0.08:
            row["label"] = "uncertain"
            row["confidence"] = "low"
            row["rationale"] = "Same file, but the available evidence cannot identify the responsible hunk."
        else:
            row["label"] = "unrelated"
            row["confidence"] = "medium"
            row["rationale"] = "Same file, but another hunk has materially stronger problem-specific evidence."
        row["evidence_score"] = f"{item['score']:.4f}"
        output.append(row)
    return output


def mark_shared(rows: list[dict[str, str]]) -> None:
    direct_by_hunk: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row["label"] == "direct":
            direct_by_hunk[(row["episode_id"], row["hunk_id"])].add(row["problem_id"])
    for row in rows:
        if row["label"] == "direct" and len(
            direct_by_hunk[(row["episode_id"], row["hunk_id"])]
        ) > 1:
            row["label"] = "shared"
            row["rationale"] += " The same hunk is associated with multiple CI problems."


def align_predictions(
    method: str,
    trace: dict[str, dict[str, Any]],
    problems_by_episode: dict[str, list[dict[str, Any]]],
    selected_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    records: list[dict[str, Any]] = []
    mapping: dict[tuple[str, str], str] = {}
    for current_id in sorted(selected_ids, key=int):
        references = problems_by_episode[current_id]
        for predicted in trace.get(current_id, {}).get("problems", []):
            predicted_id = str(predicted.get("problem_id"))
            affected_files = list(predicted.get("affected_files", []))
            candidates = []
            for reference in references:
                file_agreement = any(
                    files_match(path, reference.get("file", "")) for path in affected_files
                )
                similarity = text_similarity(problem_text(predicted), problem_text(reference))
                score = similarity + (0.55 if file_agreement else 0.0)
                candidates.append((score, similarity, file_agreement, reference))
            candidates.sort(key=lambda item: item[0], reverse=True)
            best = candidates[0] if candidates else None
            accepted = bool(best and (best[2] or best[1] >= 0.15))
            canonical_id = best[3]["problem_id"] if accepted and best else ""
            mapping[(current_id, predicted_id)] = canonical_id
            records.append(
                {
                    "method": method,
                    "episode_id": current_id,
                    "predicted_problem_id": predicted_id,
                    "predicted_problem": predicted.get("problem", ""),
                    "affected_files": affected_files,
                    "canonical_problem_id": canonical_id,
                    "alignment_score": best[0] if best else 0.0,
                    "text_similarity": best[1] if best else 0.0,
                    "file_agreement": best[2] if best else False,
                    "alignment_status": "mapped" if accepted else "unmapped",
                    "rationale": (
                        "Mapped using matching affected file plus semantic similarity."
                        if accepted and best and best[2]
                        else "Mapped using semantic similarity without a file match."
                        if accepted
                        else "No canonical problem had sufficient file or semantic evidence."
                    ),
                    "manual_canonical_problem_id": "",
                    "manual_review_status": "",
                    "manual_notes": "",
                }
            )
    return records, mapping


def metrics(tp: int, fp: int, fn: int, tn: int = 0) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
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
        result = metrics(
            sum(item[0] for item in sample),
            sum(item[1] for item in sample),
            sum(item[2] for item in sample),
            sum(item[3] for item in sample),
        )
        for key in values:
            values[key].append(float(result[key]))
    return {
        key: [sorted(items)[249], sorted(items)[9749]] for key, items in values.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "results/02_human_attribution_v2",
    )
    args = parser.parse_args()

    pair_path = args.bundle / "problem_hunk_pairs.csv"
    with pair_path.open(newline="", encoding="utf-8") as handle:
        raw_pairs = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_pairs:
        grouped[(row["episode_id"], row["problem_id"])].append(row)
    proposed = [row for rows in grouped.values() for row in label_problem_group(rows)]
    mark_shared(proposed)
    if any(row["label"] not in VALID_LABELS for row in proposed):
        raise RuntimeError("Invalid proposed label")

    proposed_path = args.bundle / "model_proposed_problem_hunk_pairs.csv"
    fields = list(proposed[0])
    with proposed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(proposed)
    priority_rows = [
        {
            **row,
            "model_label": row["label"],
            "manual_label": "",
            "manual_notes": "",
            "review_status": "",
        }
        for row in proposed
        if row["label"] != "unrelated" or float(row["evidence_score"]) >= 0.08
    ]
    priority_path = args.bundle / "manual_check_priority.csv"
    with priority_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(priority_rows[0]))
        writer.writeheader()
        writer.writerows(priority_rows)

    problems = read_jsonl(args.bundle / "problems.jsonl")
    problems_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for problem in problems:
        problems_by_episode[str(problem["episode_id"])].append(problem)
    hunks = read_jsonl(args.bundle / "hunks.jsonl")
    hunks_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hunk in hunks:
        hunks_by_episode[str(hunk["episode_id"])].append(hunk)
    selected_ids = set(problems_by_episode)

    trace_paths = {
        "endpoint": PROJECT_ROOT / "data/back_trs/decomposed_issues.json",
        "development": PROJECT_ROOT / "data/fwr_trs/decomposed_issues.json",
        "bidirectional": PROJECT_ROOT / "data/bidirect_trs/decomposed_issues.json",
    }
    traces = {
        method: {episode_id(item): item for item in read_json(path)}
        for method, path in trace_paths.items()
    }
    label_by_pair = {
        (row["episode_id"], row["problem_id"], row["hunk_id"]): row["label"]
        for row in proposed
    }
    gold_by_episode: dict[str, set[tuple[str, str]]] = defaultdict(set)
    uncertain_by_episode: dict[str, set[tuple[str, str]]] = defaultdict(set)
    negative_by_episode: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in proposed:
        key = (row["problem_id"], row["hunk_id"])
        if row["label"] in {"direct", "shared"}:
            gold_by_episode[row["episode_id"]].add(key)
        elif row["label"] == "uncertain":
            uncertain_by_episode[row["episode_id"]].add(key)
        else:
            negative_by_episode[row["episode_id"]].add(key)

    alignment_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "status": "model-proposed; requires human verification",
        "episodes": len(selected_ids),
        "gold_pairs": sum(len(value) for value in gold_by_episode.values()),
        "uncertain_pairs": sum(len(value) for value in uncertain_by_episode.values()),
        "negative_pairs": sum(len(value) for value in negative_by_episode.values()),
        "evaluable_reference_pairs": sum(
            len(gold_by_episode[current_id]) + len(negative_by_episode[current_id])
            for current_id in selected_ids
        ),
        "label_distribution": dict(
            sorted(
                {
                    label: sum(row["label"] == label for row in proposed)
                    for label in VALID_LABELS
                }.items()
            )
        ),
        "prediction_interpretation": "Each affected file expands to every canonical hunk in that file.",
        "methods": {},
    }

    for method, trace in traces.items():
        current_alignments, mapping = align_predictions(
            method, trace, problems_by_episode, selected_ids
        )
        alignment_rows.extend(current_alignments)
        predicted_by_episode: dict[str, set[tuple[str, str]]] = defaultdict(set)
        provenance: dict[tuple[str, str, str], dict[str, Any]] = {}
        for current_id in selected_ids:
            for predicted in trace.get(current_id, {}).get("problems", []):
                predicted_id = str(predicted.get("problem_id"))
                canonical_id = mapping[(current_id, predicted_id)]
                score_problem_id = canonical_id or f"UNMAPPED:{method}:{predicted_id}"
                for affected_file in predicted.get("affected_files", []):
                    matching_hunks = [
                        hunk
                        for hunk in hunks_by_episode[current_id]
                        if files_match(affected_file, hunk["file_path"])
                    ]
                    if not matching_hunks:
                        matching_hunks = [
                            {
                                "hunk_id": f"NO_HUNK:{normalize_file(affected_file)}",
                                "file_path": affected_file,
                            }
                        ]
                    for hunk in matching_hunks:
                        pair = (score_problem_id, hunk["hunk_id"])
                        predicted_by_episode[current_id].add(pair)
                        provenance[(current_id, *pair)] = {
                            "method": method,
                            "episode_id": current_id,
                            "predicted_problem_id": predicted_id,
                            "canonical_problem_id": canonical_id,
                            "affected_file": affected_file,
                            "hunk_id": hunk["hunk_id"],
                            "hunk_file": hunk["file_path"],
                            "predicted_problem": predicted.get("problem", ""),
                        }

        episode_counts: list[tuple[int, int, int, int]] = []
        for current_id in sorted(selected_ids, key=int):
            gold = gold_by_episode[current_id]
            uncertain = uncertain_by_episode[current_id]
            negatives = negative_by_episode[current_id]
            predicted = {
                pair for pair in predicted_by_episode[current_id] if pair not in uncertain
            }
            tp_pairs = predicted & gold
            fp_pairs = predicted - gold
            fn_pairs = gold - predicted
            tn_pairs = negatives - predicted
            episode_counts.append(
                (len(tp_pairs), len(fp_pairs), len(fn_pairs), len(tn_pairs))
            )
            for pair in sorted(predicted | gold):
                if pair in tp_pairs:
                    outcome = "TP"
                elif pair in fp_pairs:
                    outcome = "FP"
                else:
                    outcome = "FN"
                base = provenance.get(
                    (current_id, *pair),
                    {
                        "method": method,
                        "episode_id": current_id,
                        "canonical_problem_id": pair[0],
                        "hunk_id": pair[1],
                    },
                )
                evidence_rows.append(
                    {
                        **base,
                        "outcome": outcome,
                        "reference_label": label_by_pair.get(
                            (current_id, pair[0], pair[1]), "not_in_reference"
                        ),
                    }
                )
        result = metrics(
            sum(row[0] for row in episode_counts),
            sum(row[1] for row in episode_counts),
            sum(row[2] for row in episode_counts),
            sum(row[3] for row in episode_counts),
        )
        result["bootstrap_95_ci"] = bootstrap(episode_counts)
        result["predicted_pairs"] = sum(row[0] + row[1] for row in episode_counts)
        summary["methods"][method] = result

    alignment_path = args.bundle / "prediction_alignment.csv"
    with alignment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(alignment_rows[0]))
        writer.writeheader()
        writer.writerows(alignment_rows)
    with (args.bundle / "scoring_evidence.jsonl").open("w", encoding="utf-8") as handle:
        for row in evidence_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.bundle / "MODEL_PROPOSED_SCORES.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Model-proposed problem-hunk attribution scores",
        "",
        "> These labels and alignments require human verification.",
        "",
        f"Reference-positive pairs: {summary['gold_pairs']}; uncertain pairs excluded: {summary['uncertain_pairs']}.",
        "",
        "| Reconstruction | Predicted | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("endpoint", "development", "bidirectional"):
        row = summary["methods"][method]
        report.append(
            f"| {method} | {row['predicted_pairs']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['tn']} | {100 * row['precision']:.1f}% | {100 * row['recall']:.1f}% | "
            f"{100 * row['f1']:.1f}% | {100 * row['accuracy']:.1f}% |"
        )
    report.extend(
        [
            "",
            "Interpretation: because current ReCAP records only `affected_files`, every",
            "hunk in each predicted file is conservatively treated as predicted. Human",
            "review should correct the proposed reference labels and prediction alignments.",
            "Accuracy includes all unrelated candidate pairs as negatives and also counts",
            "out-of-reference predictions as false positives; it is inflated by class imbalance.",
        ]
    )
    (args.bundle / "MODEL_PROPOSED_RESULTS.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
