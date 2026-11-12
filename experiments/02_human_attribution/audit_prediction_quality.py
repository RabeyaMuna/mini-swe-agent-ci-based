#!/usr/bin/env python3
"""Build an auditable four-part quality review for reconstructed problems.

The generated labels are evidence-based proposals, not human annotations. The
CSV keeps proposed and manual fields separate so a reviewer can correct every
decision without recomputing evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def content_tokens(value: str) -> set[str]:
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


def explanation_support(explanation: str, hunk_content: str) -> tuple[float, list[str]]:
    explanation_tokens = content_tokens(explanation)
    hunk_tokens = content_tokens(hunk_content)
    overlap = sorted(explanation_tokens & hunk_tokens)
    score = len(overlap) / len(explanation_tokens) if explanation_tokens else 0.0
    return score, overlap


def validity_counts(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
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
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "results/02_human_attribution_30_repos",
    )
    args = parser.parse_args()

    manifest = read_json(args.bundle / "sample_manifest.json")
    repo_by_episode = {str(row["episode_id"]): row["repo"] for row in manifest}
    problems = read_jsonl(args.bundle / "problems.jsonl")
    problem_by_id = {str(row["problem_id"]): row for row in problems}
    hunks = read_jsonl(args.bundle / "hunks.jsonl")
    hunk_by_id = {str(row["hunk_id"]): row for row in hunks}
    hunks_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hunk in hunks:
        hunks_by_episode[str(hunk["episode_id"])].append(hunk)

    proposed_pairs = read_csv(args.bundle / "model_proposed_problem_hunk_pairs.csv")
    valid_hunks_by_problem: dict[str, set[str]] = defaultdict(set)
    for pair in proposed_pairs:
        if pair["label"] in {"direct", "shared"}:
            valid_hunks_by_problem[pair["problem_id"]].add(pair["hunk_id"])

    alignment_rows = read_csv(args.bundle / "prediction_alignment.csv")
    alignment_by_key = {
        (row["method"], row["episode_id"], row["predicted_problem_id"]): row
        for row in alignment_rows
    }
    trace_paths = {
        "endpoint": PROJECT_ROOT / "data/back_trs/decomposed_issues.json",
        "development": PROJECT_ROOT / "data/fwr_trs/decomposed_issues.json",
        "bidirectional": PROJECT_ROOT / "data/bidirect_trs/decomposed_issues.json",
    }
    traces = {
        method: {episode_id(item): item for item in read_json(path)}
        for method, path in trace_paths.items()
    }

    audit_rows: list[dict[str, Any]] = []
    selected_ids = set(repo_by_episode)
    for method in METHODS:
        for current_id in sorted(selected_ids, key=int):
            for predicted in traces[method].get(current_id, {}).get("problems", []):
                predicted_id = str(predicted.get("problem_id"))
                alignment = alignment_by_key[(method, current_id, predicted_id)]
                canonical_id = alignment["canonical_problem_id"]
                canonical = problem_by_id.get(canonical_id, {})
                affected_files = [str(value) for value in predicted.get("affected_files", [])]
                valid_hunk_ids = valid_hunks_by_problem.get(canonical_id, set())
                valid_hunks = [hunk_by_id[value] for value in sorted(valid_hunk_ids)]
                gold_files = {normalize_file(hunk["file_path"]) for hunk in valid_hunks}
                matched_files = {
                    normalize_file(affected_file)
                    for affected_file in affected_files
                    if any(files_match(affected_file, gold_file) for gold_file in gold_files)
                }
                predicted_hunks = {
                    str(hunk["hunk_id"])
                    for hunk in hunks_by_episode[current_id]
                    if any(files_match(path, str(hunk["file_path"])) for path in affected_files)
                }
                supported_predicted_hunks = predicted_hunks & valid_hunk_ids

                similarity = float(alignment["text_similarity"])
                file_agreement = alignment["file_agreement"] == "True"
                if not canonical_id:
                    problem_validity = "incorrect"
                elif similarity >= 0.15 and file_agreement:
                    problem_validity = "correct"
                else:
                    problem_validity = "partial"

                if not canonical_id or not affected_files or not matched_files:
                    file_validity = "incorrect"
                elif len(matched_files) == len({normalize_file(v) for v in affected_files}):
                    file_validity = "correct"
                else:
                    file_validity = "partial"

                if not supported_predicted_hunks:
                    hunk_support = "incorrect"
                elif supported_predicted_hunks == predicted_hunks:
                    hunk_support = "correct"
                else:
                    hunk_support = "partial"

                explanation = " ".join(
                    str(predicted.get(field) or "")
                    for field in ("how_fixed", "why_fix_works")
                )
                support_text = "\n".join(
                    str(hunk_by_id[hunk_id].get("content", ""))
                    for hunk_id in sorted(supported_predicted_hunks)
                )
                support_score, support_terms = explanation_support(
                    explanation, support_text
                )
                if not explanation.strip() or not supported_predicted_hunks:
                    explanation_validity = "incorrect"
                elif support_score >= 0.08 or len(support_terms) >= 2:
                    explanation_validity = "correct"
                else:
                    explanation_validity = "partial"

                audit_rows.append(
                    {
                        "method": method,
                        "episode_id": current_id,
                        "repo": repo_by_episode[current_id],
                        "predicted_problem_id": predicted_id,
                        "predicted_problem": predicted.get("problem", ""),
                        "predicted_root_cause": predicted.get("root_cause", ""),
                        "predicted_how_fixed": predicted.get("how_fixed", ""),
                        "predicted_why_fix_works": predicted.get("why_fix_works", ""),
                        "affected_files": json.dumps(affected_files),
                        "canonical_problem_id": canonical_id,
                        "canonical_problem_reason": canonical.get("reason", ""),
                        "canonical_problem_file": canonical.get("file_path", ""),
                        "valid_hunk_ids": json.dumps(sorted(valid_hunk_ids)),
                        "valid_hunk_files": json.dumps(
                            sorted({str(hunk["file_path"]) for hunk in valid_hunks})
                        ),
                        "expanded_predicted_hunk_ids": json.dumps(
                            sorted(predicted_hunks)
                        ),
                        "supported_predicted_hunk_ids": json.dumps(
                            sorted(supported_predicted_hunks)
                        ),
                        "text_similarity": similarity,
                        "repair_explanation_support_score": support_score,
                        "repair_explanation_support_terms": json.dumps(
                            support_terms[:20]
                        ),
                        "proposed_problem_validity": problem_validity,
                        "proposed_file_validity": file_validity,
                        "proposed_expanded_hunk_support": hunk_support,
                        "proposed_repair_explanation_validity": explanation_validity,
                        "proposal_warning": (
                            "Model-assisted proposal; verify against raw CI evidence and diff. "
                            "Hunk support expands affected_files because traces lack hunk IDs."
                        ),
                        "manual_problem_validity": "",
                        "manual_file_validity": "",
                        "manual_hunk_validity": "",
                        "manual_repair_explanation_validity": "",
                        "manual_notes": "",
                        "manual_review_status": "",
                    }
                )

    output_path = args.bundle / "prediction_quality_audit.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    summary: dict[str, Any] = {
        "status": "model-proposed; requires human verification",
        "episodes": len(selected_ids),
        "reconstructed_problems": len(audit_rows),
        "warning": (
            "Problem alignment and explanation support are model-assisted proposals. "
            "Hunk support is file-to-hunk-expanded, not native hunk selection."
        ),
        "methods": {},
    }
    fields = {
        "problem_validity": "proposed_problem_validity",
        "file_validity": "proposed_file_validity",
        "expanded_hunk_support": "proposed_expanded_hunk_support",
        "repair_explanation_validity": "proposed_repair_explanation_validity",
    }
    for method in METHODS:
        method_rows = [row for row in audit_rows if row["method"] == method]
        summary["methods"][method] = {
            "reconstructed_problems": len(method_rows),
            **{
                name: validity_counts(method_rows, field)
                for name, field in fields.items()
            },
        }
    (args.bundle / "MODEL_PROPOSED_QUALITY_AUDIT.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
