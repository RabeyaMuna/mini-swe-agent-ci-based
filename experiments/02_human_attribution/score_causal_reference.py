#!/usr/bin/env python3
"""Score reconstructions against the independently curated 30-episode audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("endpoint", "development", "bidirectional")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def issue_id(record: dict[str, Any]) -> str:
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


def similarity(left: str, right: str) -> float:
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


def tokens(value: str) -> set[str]:
    stop = {
        "added",
        "changed",
        "fixed",
        "removed",
        "updated",
        "using",
        "with",
        "from",
        "that",
        "this",
        "which",
        "file",
        "code",
        "error",
        "function",
        "method",
    }
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", value or "")
        if token.lower() not in stop
    }


def explanation_support(explanation: str, hunk_text: str) -> tuple[float, list[str]]:
    explanation_tokens = tokens(explanation)
    overlap = sorted(explanation_tokens & tokens(hunk_text))
    score = len(overlap) / len(explanation_tokens) if explanation_tokens else 0.0
    return score, overlap


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
    values = {"precision": [], "recall": [], "f1": []}
    for _ in range(10_000):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        result = metrics(*(sum(row[index] for row in sample) for index in range(4)))
        for key in values:
            values[key].append(float(result[key]))
    return {
        key: [sorted(samples)[249], sorted(samples)[9749]]
        for key, samples in values.items()
    }


def quality_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts = Counter(str(row[field]) for row in rows)
    total = len(rows)
    return {
        "counts": dict(sorted(counts.items())),
        "strict_correct_rate": counts["correct"] / total if total else 0.0,
        "correct_or_partial_rate": (
            (counts["correct"] + counts["partial"]) / total if total else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("causal_reference_30.json"),
    )
    args = parser.parse_args()

    reference = read_json(args.reference)
    selected_ids = set(reference["episode_ids"])
    hunks = read_jsonl(args.bundle / "hunks.jsonl")
    hunk_by_id = {str(hunk["hunk_id"]): hunk for hunk in hunks}
    hunks_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hunk in hunks:
        hunks_by_episode[str(hunk["episode_id"])].append(hunk)

    trace_paths = {
        "endpoint": PROJECT_ROOT / "data/back_trs/decomposed_issues.json",
        "development": PROJECT_ROOT / "data/fwr_trs/decomposed_issues.json",
        "bidirectional": PROJECT_ROOT / "data/bidirect_trs/decomposed_issues.json",
    }
    traces = {
        method: {issue_id(item): item for item in read_json(path)}
        for method, path in trace_paths.items()
    }

    gold_by_episode: dict[str, set[tuple[str, str]]] = defaultdict(set)
    uncertain_by_episode: dict[str, set[tuple[str, str]]] = defaultdict(set)
    universe_by_episode: dict[str, set[tuple[str, str]]] = defaultdict(set)
    canonical_by_episode: dict[str, list[dict[str, Any]]] = {}
    for current_id, problems in reference["episodes"].items():
        canonical_by_episode[current_id] = problems
        episode_hunks = {str(hunk["hunk_id"]) for hunk in hunks_by_episode[current_id]}
        for problem in problems:
            canonical_id = f"{current_id}:{problem['id']}"
            universe_by_episode[current_id].update(
                (canonical_id, hunk_id) for hunk_id in episode_hunks
            )
            gold_by_episode[current_id].update(
                (canonical_id, hunk_id) for hunk_id in problem["valid_hunks"]
            )
            uncertain_by_episode[current_id].update(
                (canonical_id, hunk_id) for hunk_id in problem["uncertain_hunks"]
            )

    reference_rows: list[dict[str, Any]] = []
    for current_id in sorted(selected_ids, key=int):
        hunk_use_counts = Counter(
            hunk_id for _, hunk_id in gold_by_episode[current_id]
        )
        for problem in canonical_by_episode[current_id]:
            canonical_id = f"{current_id}:{problem['id']}"
            for hunk in hunks_by_episode[current_id]:
                pair = (canonical_id, str(hunk["hunk_id"]))
                if pair in uncertain_by_episode[current_id]:
                    proposed_label = "uncertain"
                elif pair in gold_by_episode[current_id]:
                    proposed_label = (
                        "shared" if hunk_use_counts[pair[1]] > 1 else "direct"
                    )
                else:
                    proposed_label = "unrelated"
                reference_rows.append(
                    {
                        "episode_id": current_id,
                        "canonical_problem_id": canonical_id,
                        "canonical_problem": problem["description"],
                        "hunk_id": hunk["hunk_id"],
                        "hunk_file": hunk["file_path"],
                        "hunk_header": hunk["header"],
                        "hunk_content": hunk["content"],
                        "proposed_label": proposed_label,
                        "proposal_note": problem.get("note", ""),
                        "manual_label": "",
                        "manual_notes": "",
                        "manual_review_status": "",
                    }
                )

    with (args.bundle / "causal_reference_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)

    summary: dict[str, Any] = {
        "status": "independent model audit; requires human verification",
        "episodes": len(selected_ids),
        "repositories": len(selected_ids),
        "canonical_problems": sum(len(value) for value in canonical_by_episode.values()),
        "canonical_hunks": len(hunks),
        "valid_reference_pairs": sum(len(value) for value in gold_by_episode.values()),
        "uncertain_pairs_excluded": sum(
            len(value) for value in uncertain_by_episode.values()
        ),
        "prediction_mode": "affected_files expanded to all hunks in each file",
        "methods": {},
    }
    audit_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    for method in METHODS:
        predicted_by_episode: dict[str, set[tuple[str, str]]] = defaultdict(set)
        provenance: dict[tuple[str, str, str], dict[str, Any]] = {}
        for current_id in selected_ids:
            canonical_problems = canonical_by_episode[current_id]
            for predicted in traces[method].get(current_id, {}).get("problems", []):
                predicted_id = str(predicted.get("problem_id"))
                predicted_problem_text = " ".join(
                    str(predicted.get(field) or "")
                    for field in ("problem", "root_cause", "failure_type", "issue_type")
                )
                affected_files = [str(value) for value in predicted.get("affected_files", [])]
                candidates: list[tuple[float, float, bool, dict[str, Any]]] = []
                for canonical in canonical_problems:
                    valid_files = {
                        str(hunk_by_id[hunk_id]["file_path"])
                        for hunk_id in canonical["valid_hunks"]
                    }
                    file_agreement = any(
                        files_match(affected_file, valid_file)
                        for affected_file in affected_files
                        for valid_file in valid_files
                    )
                    semantic = similarity(
                        predicted_problem_text, str(canonical["description"])
                    )
                    candidates.append(
                        (semantic + (0.45 if file_agreement else 0.0), semantic, file_agreement, canonical)
                    )
                candidates.sort(key=lambda item: item[0], reverse=True)
                best = candidates[0]
                mapped = best[2] or best[1] >= 0.08
                canonical = best[3] if mapped else None
                canonical_id = f"{current_id}:{canonical['id']}" if canonical else ""

                matching_hunks = [
                    hunk
                    for hunk in hunks_by_episode[current_id]
                    if any(files_match(path, str(hunk["file_path"])) for path in affected_files)
                ]
                if not matching_hunks:
                    matching_hunks = [
                        {
                            "hunk_id": "NO_HUNK:" + "|".join(map(normalize_file, affected_files)),
                            "file_path": "|".join(affected_files),
                            "content": "",
                        }
                    ]
                score_problem_id = canonical_id or f"UNMAPPED:{method}:{predicted_id}"
                predicted_hunk_ids = {str(hunk["hunk_id"]) for hunk in matching_hunks}
                valid_hunk_ids = set(canonical["valid_hunks"]) if canonical else set()
                supported_hunks = predicted_hunk_ids & valid_hunk_ids
                valid_files = {
                    str(hunk_by_id[hunk_id]["file_path"]) for hunk_id in valid_hunk_ids
                }
                matched_file_count = sum(
                    any(files_match(path, valid_file) for valid_file in valid_files)
                    for path in affected_files
                )

                if not mapped:
                    problem_validity = "incorrect"
                elif best[1] >= 0.15:
                    problem_validity = "correct"
                else:
                    problem_validity = "partial"
                if not affected_files or matched_file_count == 0:
                    file_validity = "incorrect"
                elif matched_file_count == len(affected_files):
                    file_validity = "correct"
                else:
                    file_validity = "partial"
                if not supported_hunks:
                    hunk_validity = "incorrect"
                elif supported_hunks == predicted_hunk_ids:
                    hunk_validity = "correct"
                else:
                    hunk_validity = "partial"

                explanation = " ".join(
                    str(predicted.get(field) or "")
                    for field in ("how_fixed", "why_fix_works")
                )
                supported_text = "\n".join(
                    str(hunk_by_id[hunk_id].get("content", ""))
                    for hunk_id in sorted(supported_hunks)
                )
                explanation_score, explanation_terms = explanation_support(
                    explanation, supported_text
                )
                if not explanation.strip() or not supported_hunks:
                    explanation_validity = "incorrect"
                elif explanation_score >= 0.08 or len(explanation_terms) >= 2:
                    explanation_validity = "correct"
                else:
                    explanation_validity = "partial"

                audit_rows.append(
                    {
                        "method": method,
                        "episode_id": current_id,
                        "predicted_problem_id": predicted_id,
                        "predicted_problem": predicted.get("problem", ""),
                        "predicted_root_cause": predicted.get("root_cause", ""),
                        "predicted_how_fixed": predicted.get("how_fixed", ""),
                        "predicted_why_fix_works": predicted.get("why_fix_works", ""),
                        "affected_files": json.dumps(affected_files),
                        "canonical_problem_id": canonical_id,
                        "canonical_problem": canonical["description"] if canonical else "",
                        "alignment_similarity": best[1],
                        "alignment_file_agreement": best[2],
                        "expanded_hunk_ids": json.dumps(sorted(predicted_hunk_ids)),
                        "supported_hunk_ids": json.dumps(sorted(supported_hunks)),
                        "explanation_support_score": explanation_score,
                        "explanation_support_terms": json.dumps(explanation_terms[:20]),
                        "proposed_problem_validity": problem_validity,
                        "proposed_file_validity": file_validity,
                        "proposed_hunk_validity": hunk_validity,
                        "proposed_repair_explanation_validity": explanation_validity,
                        "manual_problem_validity": "",
                        "manual_file_validity": "",
                        "manual_hunk_validity": "",
                        "manual_repair_explanation_validity": "",
                        "manual_canonical_problem_id": "",
                        "manual_notes": "",
                        "manual_review_status": "",
                    }
                )
                for hunk in matching_hunks:
                    pair = (score_problem_id, str(hunk["hunk_id"]))
                    predicted_by_episode[current_id].add(pair)
                    provenance[(current_id, *pair)] = {
                        "method": method,
                        "episode_id": current_id,
                        "predicted_problem_id": predicted_id,
                        "canonical_problem_id": canonical_id,
                        "hunk_id": hunk["hunk_id"],
                        "hunk_file": hunk["file_path"],
                    }

        episode_counts: list[tuple[int, int, int, int]] = []
        for current_id in sorted(selected_ids, key=int):
            gold = gold_by_episode[current_id]
            uncertain = uncertain_by_episode[current_id]
            universe = universe_by_episode[current_id] - uncertain
            predicted = {
                pair for pair in predicted_by_episode[current_id] if pair not in uncertain
            }
            tp_pairs = predicted & gold
            fp_pairs = predicted - gold
            fn_pairs = gold - predicted
            tn_pairs = (universe - gold) - predicted
            episode_counts.append(
                (len(tp_pairs), len(fp_pairs), len(fn_pairs), len(tn_pairs))
            )
            for pair in sorted(predicted | gold):
                outcome = "TP" if pair in tp_pairs else "FP" if pair in fp_pairs else "FN"
                evidence_rows.append(
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
                    }
                )
        result = metrics(*(sum(row[index] for row in episode_counts) for index in range(4)))
        result["predicted_pairs"] = sum(row[0] + row[1] for row in episode_counts)
        result["bootstrap_95_ci"] = bootstrap(episode_counts)
        method_audits = [row for row in audit_rows if row["method"] == method]
        result["reconstructed_problems"] = len(method_audits)
        for name in (
            "problem_validity",
            "file_validity",
            "hunk_validity",
            "repair_explanation_validity",
        ):
            result[name] = quality_summary(method_audits, f"proposed_{name}")
        summary["methods"][method] = result

    with (args.bundle / "causal_prediction_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    with (args.bundle / "causal_scoring_evidence.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in evidence_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.bundle / "CAUSAL_MODEL_AUDIT_SCORES.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Independent model audit of problem-hunk attribution (30 episodes)",
        "",
        "> These are model-proposed audit results and require human verification.",
        "",
        f"Canonical problems: {summary['canonical_problems']}; canonical hunks: "
        f"{summary['canonical_hunks']}; valid pairs: {summary['valid_reference_pairs']}; "
        f"uncertain pairs excluded: {summary['uncertain_pairs_excluded']}.",
        "",
        "| Reconstruction | Predicted | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["methods"][method]
        ci = row["bootstrap_95_ci"]
        report.append(
            f"| {method} | {row['predicted_pairs']} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {100 * row['precision']:.1f}% "
            f"({100 * ci['precision'][0]:.1f}–{100 * ci['precision'][1]:.1f}) | "
            f"{100 * row['recall']:.1f}% "
            f"({100 * ci['recall'][0]:.1f}–{100 * ci['recall'][1]:.1f}) | "
            f"{100 * row['f1']:.1f}% "
            f"({100 * ci['f1'][0]:.1f}–{100 * ci['f1'][1]:.1f}) |"
        )
    report.extend(
        [
            "",
            "Strict proposed correctness rates by reconstructed problem:",
            "",
            "| Reconstruction | Problem | File | Expanded hunk | Repair explanation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = summary["methods"][method]
        report.append(
            f"| {method} | {100 * row['problem_validity']['strict_correct_rate']:.1f}% | "
            f"{100 * row['file_validity']['strict_correct_rate']:.1f}% | "
            f"{100 * row['hunk_validity']['strict_correct_rate']:.1f}% | "
            f"{100 * row['repair_explanation_validity']['strict_correct_rate']:.1f}% |"
        )
    report.extend(
        [
            "",
            "Because current traces contain affected files rather than native hunk IDs,",
            "the hunk and association results conservatively expand each file to all its hunks.",
            "The four correctness dimensions use evidence-based proposals and are not human",
            "accuracy measurements until the manual columns in the audit CSVs are completed.",
        ]
    )
    (args.bundle / "CAUSAL_MODEL_AUDIT_RESULTS.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
