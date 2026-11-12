#!/usr/bin/env python3
"""Create a reviewer-friendly workbook for manual problem-hunk adjudication."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("endpoint", "development", "bidirectional")
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TP_FILL = PatternFill("solid", fgColor="C6EFCE")
FP_FILL = PatternFill("solid", fgColor="FFC7CE")
UNCERTAIN_FILL = PatternFill("solid", fgColor="FFEB9C")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clipped(value: Any, limit: int = 32_000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 30] + "\n[TRUNCATED IN WORKBOOK]"


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def style_sheet(
    sheet: Any,
    widths: dict[str, int] | None = None,
    freeze: str = "A2",
) -> None:
    sheet.freeze_panes = freeze
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.row_dimensions[1].height = 42
    default_widths = widths or {}
    for column_cells in sheet.columns:
        letter = column_cells[0].column_letter
        header = str(column_cells[0].value or "")
        sheet.column_dimensions[letter].width = default_widths.get(header, 18)
        for cell in column_cells[1:]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def add_dropdown(sheet: Any, header: str, values: str) -> None:
    header_map = {cell.value: cell.column_letter for cell in sheet[1]}
    column = header_map[header]
    validation = DataValidation(type="list", formula1=f'"{values}"', allow_blank=True)
    validation.error = "Choose a value from the list."
    validation.errorTitle = "Invalid review decision"
    sheet.add_data_validation(validation)
    validation.add(f"{column}2:{column}{max(sheet.max_row, 2)}")


def write_rows(sheet: Any, headers: list[str], rows: list[dict[str, Any]]) -> None:
    sheet.append(headers)
    for row in rows:
        sheet.append([clipped(row.get(header, "")) for header in headers])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "results/02_human_attribution_30_causal",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.bundle / "MANUAL_PROBLEM_HUNK_REVIEW_30.xlsx"

    manifest = read_json(args.bundle / "sample_manifest.json")
    repo_by_episode = {str(row["episode_id"]): row["repo"] for row in manifest}
    logs = {str(row["id"]): row for row in read_json(PROJECT_ROOT / "data/log_details.json")}
    hunks = read_jsonl(args.bundle / "hunks.jsonl")
    hunk_by_id = {str(row["hunk_id"]): row for row in hunks}
    reference_rows = read_csv(args.bundle / "causal_reference_pairs.csv")
    reference_by_key = {
        (row["episode_id"], row["canonical_problem_id"], row["hunk_id"]): row
        for row in reference_rows
    }
    audit_rows = read_csv(args.bundle / "causal_prediction_audit.csv")
    audit_by_key = {
        (row["method"], row["episode_id"], row["predicted_problem_id"]): row
        for row in audit_rows
    }
    evidence_rows = read_jsonl(args.bundle / "causal_scoring_evidence.jsonl")

    raw_path = args.bundle / "raw_episodes_30.jsonl.gz"
    raw_by_episode: dict[str, dict[str, Any]] = {}
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                raw_by_episode[str(row["id"])] = row

    trace_paths = {
        "endpoint": PROJECT_ROOT / "data/back_trs/decomposed_issues.json",
        "development": PROJECT_ROOT / "data/fwr_trs/decomposed_issues.json",
        "bidirectional": PROJECT_ROOT / "data/bidirect_trs/decomposed_issues.json",
    }
    trace_problem_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for method, path in trace_paths.items():
        for episode in read_json(path):
            current_id = str(episode.get("issue_id") or episode.get("original_issue_id") or "")
            if current_id not in repo_by_episode:
                continue
            for problem in episode.get("problems", []):
                trace_problem_by_key[(method, current_id, str(problem.get("problem_id")))] = problem

    predicted_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for evidence in evidence_rows:
        method = str(evidence["method"])
        current_id = str(evidence["episode_id"])
        canonical_id = str(evidence.get("canonical_problem_id") or "")
        hunk_id = str(evidence["hunk_id"])
        log = logs[current_id]
        reference = reference_by_key.get((current_id, canonical_id, hunk_id), {})
        hunk = hunk_by_id.get(hunk_id, {})
        ground_context = "\n\n".join(log.get("error_context", []))
        failure_signals = "\n".join(f"• {value}" for value in log.get("failure_signals", []))
        error_types = "\n".join(
            f"• {row.get('category', '')}: {row.get('subcategory', '')}\n  {row.get('evidence', '')}"
            for row in log.get("error_types", [])
        )
        failed_jobs = compact_json(log.get("failed_job", []))
        relevant_files = compact_json(log.get("relevant_files", []))
        raw = raw_by_episode[current_id]
        changed_files = compact_json(raw.get("changed_files", []))

        if evidence["outcome"] == "FN":
            missing_rows.append(
                {
                    "manual_decision": "",
                    "manual_notes": "",
                    "reviewer": "",
                    "review_status": "",
                    "proposed_outcome": "FN",
                    "method": method,
                    "episode_id": current_id,
                    "repo": repo_by_episode[current_id],
                    "sha_fail": raw.get("sha_fail", ""),
                    "sha_success": raw.get("sha_success", ""),
                    "diff_changed_files": changed_files,
                    "canonical_problem_id": canonical_id,
                    "canonical_problem": reference.get("canonical_problem", ""),
                    "hunk_id": hunk_id,
                    "hunk_file": hunk.get("file_path", reference.get("hunk_file", "")),
                    "actual_hunk_diff": hunk.get("content", reference.get("hunk_content", "")),
                    "ci_failure_summary": ground_context,
                    "ci_failure_signals": failure_signals,
                    "ci_error_types": error_types,
                    "ci_log_relevant_files": relevant_files,
                    "failed_workflow_jobs_commands": failed_jobs,
                    "why_proposed_fn": "This human-reference problem–hunk link was not predicted by the method.",
                }
            )
            continue

        predicted_id = str(evidence.get("predicted_problem_id") or "")
        audit = audit_by_key.get((method, current_id, predicted_id), {})
        predicted = trace_problem_by_key.get((method, current_id, predicted_id), {})
        outcome = str(evidence["outcome"])
        if outcome == "TP":
            reason = "Predicted problem maps to the canonical CI problem and this hunk is a valid reference repair."
        elif not canonical_id:
            reason = "The reconstructed problem did not map to a canonical CI problem."
        elif hunk_id.startswith("NO_HUNK:"):
            reason = "The predicted affected file has no canonical hunk in the observed repair diff."
        else:
            reason = "The predicted problem was mapped, but this expanded hunk is not a valid repair for that problem."
        predicted_rows.append(
            {
                "manual_decision": "",
                "manual_problem_correct": "",
                "manual_file_correct": "",
                "manual_hunk_correct": "",
                "manual_repair_explanation_correct": "",
                "manual_notes": "",
                "reviewer": "",
                "review_status": "",
                "proposed_outcome": outcome,
                "proposed_reason": reason,
                "method": method,
                "episode_id": current_id,
                "repo": repo_by_episode[current_id],
                "sha_fail": raw.get("sha_fail", ""),
                "sha_success": raw.get("sha_success", ""),
                "diff_changed_files": changed_files,
                "predicted_problem_id": predicted_id,
                "predicted_failure_type": predicted.get("failure_type", ""),
                "predicted_issue_type": predicted.get("issue_type", ""),
                "predicted_problem": predicted.get("problem", audit.get("predicted_problem", "")),
                "predicted_root_cause": predicted.get("root_cause", audit.get("predicted_root_cause", "")),
                "predicted_repair_how_fixed": predicted.get("how_fixed", audit.get("predicted_how_fixed", "")),
                "predicted_why_fix_works": predicted.get("why_fix_works", audit.get("predicted_why_fix_works", "")),
                "predicted_affected_files": audit.get("affected_files", ""),
                "canonical_problem_id": canonical_id,
                "canonical_problem": audit.get("canonical_problem", reference.get("canonical_problem", "")),
                "reference_pair_label": reference.get("proposed_label", "not_in_reference"),
                "hunk_id": hunk_id,
                "hunk_file": hunk.get("file_path", evidence.get("hunk_file", "")),
                "actual_hunk_header": hunk.get("header", ""),
                "actual_hunk_diff": hunk.get("content", ""),
                "proposed_problem_validity": audit.get("proposed_problem_validity", ""),
                "proposed_file_validity": audit.get("proposed_file_validity", ""),
                "proposed_hunk_validity": audit.get("proposed_hunk_validity", ""),
                "proposed_repair_explanation_validity": audit.get(
                    "proposed_repair_explanation_validity", ""
                ),
                "ci_failure_summary": ground_context,
                "ci_failure_signals": failure_signals,
                "ci_error_types": error_types,
                "ci_log_relevant_files": relevant_files,
                "failed_workflow_jobs_commands": failed_jobs,
            }
        )

    ground_truth_evidence: list[dict[str, Any]] = []
    for current_id in sorted(repo_by_episode, key=int):
        log = logs[current_id]
        raw = raw_by_episode[current_id]
        ground_truth_evidence.append(
            {
                "episode_id": current_id,
                "repo": repo_by_episode[current_id],
                "sha_fail": raw.get("sha_fail", ""),
                "sha_success": raw.get("sha_success", ""),
                "diff_changed_files": compact_json(raw.get("changed_files", [])),
                "cumulative_fail_to_success_diff": raw.get("diff", ""),
                "log_error_context": compact_json(log.get("error_context", [])),
                "log_failure_signals": compact_json(log.get("failure_signals", [])),
                "log_relevant_files": compact_json(log.get("relevant_files", [])),
                "log_error_types": compact_json(log.get("error_types", [])),
                "log_failed_jobs": compact_json(log.get("failed_job", [])),
                "structured_log_details_json": compact_json(log),
                "ci_workflow": raw.get("workflow", ""),
            }
        )

    enriched_reference_rows: list[dict[str, Any]] = []
    for row in reference_rows:
        current_id = row["episode_id"]
        log = logs[current_id]
        raw = raw_by_episode[current_id]
        enriched_reference_rows.append(
            {
                "manual_label": row.get("manual_label", ""),
                "manual_notes": row.get("manual_notes", ""),
                "manual_review_status": row.get("manual_review_status", ""),
                "proposed_label": row["proposed_label"],
                "proposal_note": row.get("proposal_note", ""),
                "episode_id": current_id,
                "repo": repo_by_episode[current_id],
                "sha_fail": raw.get("sha_fail", ""),
                "sha_success": raw.get("sha_success", ""),
                "diff_changed_files": compact_json(raw.get("changed_files", [])),
                "canonical_problem_id": row["canonical_problem_id"],
                "canonical_problem": row["canonical_problem"],
                "hunk_id": row["hunk_id"],
                "hunk_file": row["hunk_file"],
                "hunk_header": row["hunk_header"],
                "hunk_content": row["hunk_content"],
                "log_error_context": compact_json(log.get("error_context", [])),
                "log_failure_signals": compact_json(log.get("failure_signals", [])),
                "log_relevant_files": compact_json(log.get("relevant_files", [])),
                "log_error_types": compact_json(log.get("error_types", [])),
                "log_failed_jobs": compact_json(log.get("failed_job", [])),
            }
        )

    workbook = Workbook()
    instructions = workbook.active
    instructions.title = "Instructions"
    instruction_rows = [
        ["Manual problem–hunk attribution review: 30 episodes"],
        ["Purpose", "Verify whether every reconstructed problem–hunk association is a TP or FP."],
        ["TP rule", "The predicted problem is real, the file/hunk is correct, and the hunk causally repairs that problem."],
        ["FP rule", "The problem is incorrect, the file/hunk is wrong or unrelated, or the claimed repair is unsupported by the diff."],
        ["Uncertain", "Use only when the supplied CI evidence and diff are insufficient; explain why in manual_notes."],
        ["Important", "Do not accept a prediction only because its explanation sounds plausible. Compare it with CI failure evidence, workflow command, and actual hunk."],
        ["Predicted_Associations", "Review all proposed TP/FP rows and fill the eight manual columns on the left."],
        ["Missing_Reference_Links", "Review proposed FN rows to determine whether the method genuinely missed a valid association."],
        ["Reference_Pairs", "Review the independent causal problem–hunk reference labels without relying on ReCAP's prediction."],
        ["Ground_Truth_Evidence", "Contains IDs, repositories, fail/pass SHAs, complete changed-file lists, cumulative diffs, structured log details, and workflows."],
        ["Diff terminology", "The fail-to-success diff is observed repair evidence. It is not automatically a ground-truth association; you must decide which hunk repairs which problem."],
        ["Limitation", "Current traces identify affected files, so every hunk in a predicted file is conservatively expanded as predicted."],
    ]
    for row in instruction_rows:
        instructions.append(row)
    instructions.column_dimensions["A"].width = 28
    instructions.column_dimensions["B"].width = 120
    for row in instructions.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    instructions["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    instructions["A1"].fill = HEADER_FILL
    instructions.merge_cells("A1:B1")

    predicted_sheet = workbook.create_sheet("Predicted_Associations")
    predicted_headers = list(predicted_rows[0])
    write_rows(predicted_sheet, predicted_headers, predicted_rows)
    style_sheet(
        predicted_sheet,
        {
            "manual_decision": 18,
            "manual_notes": 45,
            "proposed_reason": 45,
            "predicted_problem": 55,
            "predicted_root_cause": 55,
            "predicted_repair_how_fixed": 55,
            "predicted_why_fix_works": 55,
            "canonical_problem": 55,
            "actual_hunk_diff": 75,
            "diff_changed_files": 45,
            "ci_failure_summary": 75,
            "ci_failure_signals": 65,
            "ci_error_types": 65,
            "ci_log_relevant_files": 70,
            "failed_workflow_jobs_commands": 55,
        },
        "A2",
    )
    add_dropdown(predicted_sheet, "manual_decision", "TP,FP,UNCERTAIN")
    for header in (
        "manual_problem_correct",
        "manual_file_correct",
        "manual_hunk_correct",
        "manual_repair_explanation_correct",
    ):
        add_dropdown(predicted_sheet, header, "YES,NO,PARTIAL,UNCERTAIN")
    add_dropdown(predicted_sheet, "review_status", "REVIEWED,NEEDS_DISCUSSION")
    predicted_sheet.conditional_formatting.add(
        f"A2:A{predicted_sheet.max_row}", FormulaRule(formula=["A2=\"TP\""], fill=TP_FILL)
    )
    predicted_sheet.conditional_formatting.add(
        f"A2:A{predicted_sheet.max_row}", FormulaRule(formula=["A2=\"FP\""], fill=FP_FILL)
    )
    predicted_sheet.conditional_formatting.add(
        f"A2:A{predicted_sheet.max_row}",
        FormulaRule(formula=["A2=\"UNCERTAIN\""], fill=UNCERTAIN_FILL),
    )

    missing_sheet = workbook.create_sheet("Missing_Reference_Links")
    missing_headers = list(missing_rows[0])
    write_rows(missing_sheet, missing_headers, missing_rows)
    style_sheet(
        missing_sheet,
        {
            "manual_decision": 20,
            "manual_notes": 45,
            "canonical_problem": 55,
            "actual_hunk_diff": 75,
            "diff_changed_files": 45,
            "ci_failure_summary": 75,
            "ci_failure_signals": 65,
            "ci_error_types": 65,
            "ci_log_relevant_files": 70,
            "failed_workflow_jobs_commands": 55,
            "why_proposed_fn": 55,
        },
    )
    add_dropdown(missing_sheet, "manual_decision", "FN,NOT_VALID,UNCERTAIN")
    add_dropdown(missing_sheet, "review_status", "REVIEWED,NEEDS_DISCUSSION")

    reference_sheet = workbook.create_sheet("Reference_Pairs")
    reference_headers = list(enriched_reference_rows[0])
    write_rows(reference_sheet, reference_headers, enriched_reference_rows)
    style_sheet(
        reference_sheet,
        {
            "canonical_problem": 60,
            "hunk_content": 75,
            "proposal_note": 55,
            "manual_notes": 45,
            "diff_changed_files": 45,
            "log_error_context": 75,
            "log_failure_signals": 70,
            "log_relevant_files": 70,
            "log_error_types": 70,
            "log_failed_jobs": 60,
        },
    )
    add_dropdown(reference_sheet, "manual_label", "direct,shared,unrelated,uncertain")
    add_dropdown(reference_sheet, "manual_review_status", "REVIEWED,NEEDS_DISCUSSION")

    instance_sheet = workbook.create_sheet("Ground_Truth_Evidence")
    instance_headers = list(ground_truth_evidence[0])
    write_rows(instance_sheet, instance_headers, ground_truth_evidence)
    style_sheet(
        instance_sheet,
        {
            "repo": 35,
            "diff_changed_files": 50,
            "cumulative_fail_to_success_diff": 100,
            "log_error_context": 80,
            "log_failure_signals": 70,
            "log_relevant_files": 70,
            "log_error_types": 70,
            "log_failed_jobs": 60,
            "structured_log_details_json": 100,
            "ci_workflow": 100,
        },
    )

    summary_sheet = workbook.create_sheet("Manual_Summary")
    summary_sheet.append(
        ["Method", "Manual TP", "Manual FP", "Manual FN", "Precision", "Recall", "F1"]
    )
    for row_index, method in enumerate(METHODS, start=2):
        summary_sheet.cell(row_index, 1, method)
        summary_sheet.cell(
            row_index,
            2,
            f'=COUNTIFS(Predicted_Associations!$K:$K,$A{row_index},Predicted_Associations!$A:$A,"TP")',
        )
        summary_sheet.cell(
            row_index,
            3,
            f'=COUNTIFS(Predicted_Associations!$K:$K,$A{row_index},Predicted_Associations!$A:$A,"FP")',
        )
        summary_sheet.cell(
            row_index,
            4,
            f'=COUNTIFS(Missing_Reference_Links!$F:$F,$A{row_index},Missing_Reference_Links!$A:$A,"FN")',
        )
        summary_sheet.cell(row_index, 5, f'=IFERROR(B{row_index}/(B{row_index}+C{row_index}),0)')
        summary_sheet.cell(row_index, 6, f'=IFERROR(B{row_index}/(B{row_index}+D{row_index}),0)')
        summary_sheet.cell(
            row_index,
            7,
            f'=IFERROR(2*E{row_index}*F{row_index}/(E{row_index}+F{row_index}),0)',
        )
    for row in range(2, 5):
        for column in range(5, 8):
            summary_sheet.cell(row, column).number_format = "0.0%"
    style_sheet(summary_sheet)

    workbook.save(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "predicted_associations": len(predicted_rows),
                "missing_reference_links": len(missing_rows),
                "reference_pairs": len(enriched_reference_rows),
                "instances": len(ground_truth_evidence),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
