#!/usr/bin/env python3
"""Prepare a blinded human-attribution review bundle.

This script deliberately keeps the human reference material separate from the
three reconstruction outputs.  It also computes a *proxy* problem-file score
from the existing ``affected_files`` fields.  The proxy is not presented as a
human-verified problem-hunk result.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Hunk:
    episode_id: str
    hunk_id: str
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    content: str


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def episode_id(record: dict[str, Any]) -> str:
    return str(
        record.get("id")
        or record.get("issue_id")
        or record.get("original_issue_id")
        or ""
    )


def normalize_file(path: str) -> str:
    return re.sub(r"^\./", "", str(path or "").replace("\\", "/")).lower()


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


def parse_range(start_text: str, count_text: str | None) -> tuple[int, int]:
    return int(start_text), int(count_text) if count_text is not None else 1


def parse_hunks(episode: str, diff: str) -> list[Hunk]:
    """Parse ordinary unified-diff hunks and assign stable content IDs."""
    range_pattern = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
    )
    file_pattern = re.compile(r"^diff --git a/(.*?) b/(.*)$")
    parsed: list[Hunk] = []
    current_file = ""
    current_header: str | None = None
    current_ranges: tuple[int, int, int, int] | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_header, current_ranges, current_lines
        if current_header is None or current_ranges is None:
            return
        content = "\n".join(current_lines)
        digest_input = f"{episode}\0{current_file}\0{current_header}\0{content}"
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
        old_start, old_count, new_start, new_count = current_ranges
        parsed.append(
            Hunk(
                episode_id=episode,
                hunk_id=f"{episode}:H{len(parsed) + 1:03d}:{digest}",
                file_path=current_file,
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                header=current_header,
                content=content,
            )
        )
        current_header = None
        current_ranges = None
        current_lines = []

    for line in str(diff or "").splitlines():
        file_match = file_pattern.match(line)
        if file_match:
            flush()
            current_file = file_match.group(2)
            continue
        range_match = range_pattern.match(line)
        if range_match:
            flush()
            old_start, old_count = parse_range(range_match.group(1), range_match.group(2))
            new_start, new_count = parse_range(range_match.group(3), range_match.group(4))
            current_header = line
            current_ranges = (old_start, old_count, new_start, new_count)
            continue
        if current_header is not None:
            current_lines.append(line)
    flush()
    return parsed


def load_trace(path: Path) -> dict[str, dict[str, Any]]:
    return {episode_id(item): item for item in read_json(path)}


def select_stratified_ids(
    memory_ids: list[str],
    raw_by_id: dict[str, dict[str, Any]],
    log_by_id: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, dict[str, Any]]],
    *,
    count: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Balance problem multiplicity and round-robin across repositories."""
    eligible = [
        item
        for item in memory_ids
        if item in raw_by_id
        and item in log_by_id
        and bool(raw_by_id[item].get("diff"))
        and all(item in trace for trace in traces.values())
    ]
    requested = {"single": count // 2, "multi": count - count // 2}
    rng = random.Random(seed)
    selected: list[str] = []
    selected_by_stratum: dict[str, int] = {}

    for stratum in ("single", "multi"):
        pool = [
            item
            for item in eligible
            if (
                "single"
                if len(log_by_id[item].get("relevant_files", [])) == 1
                else "multi"
            )
            == stratum
        ]
        by_repo: dict[str, list[str]] = defaultdict(list)
        for item in pool:
            raw = raw_by_id[item]
            repo = f"{raw.get('repo_owner', '')}/{raw.get('repo_name', '')}"
            by_repo[repo].append(item)
        repo_names = list(by_repo)
        rng.shuffle(repo_names)
        for values in by_repo.values():
            rng.shuffle(values)

        chosen: list[str] = []
        while len(chosen) < requested[stratum]:
            made_progress = False
            for repo in repo_names:
                if by_repo[repo]:
                    chosen.append(by_repo[repo].pop())
                    made_progress = True
                if len(chosen) == requested[stratum]:
                    break
            if not made_progress:
                break
        if len(chosen) != requested[stratum]:
            raise RuntimeError(
                f"Only {len(chosen)} eligible {stratum} episodes; "
                f"need {requested[stratum]}"
            )
        selected.extend(chosen)
        selected_by_stratum[stratum] = len(chosen)

    return selected, {
        "strategy": "balanced problem multiplicity; repository round-robin",
        "seed": seed,
        "eligible_episodes": len(eligible),
        "requested_episodes": count,
        "selected_by_stratum": selected_by_stratum,
    }


def select_unique_project_ids(
    memory_ids: list[str],
    raw_by_id: dict[str, dict[str, Any]],
    log_by_id: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, dict[str, Any]]],
    *,
    count: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Select one episode per project name, maximizing multiplicity balance."""
    eligible = [
        item
        for item in memory_ids
        if item in raw_by_id
        and item in log_by_id
        and bool(raw_by_id[item].get("diff"))
        and all(item in trace for trace in traces.values())
    ]
    by_project: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"single": [], "multi": []}
    )
    for item in eligible:
        project = str(raw_by_id[item].get("repo_name") or "").lower()
        stratum = (
            "single"
            if len(log_by_id[item].get("relevant_files", [])) == 1
            else "multi"
        )
        by_project[project][stratum].append(item)
    if len(by_project) < count:
        raise RuntimeError(
            f"Only {len(by_project)} unique eligible projects; need {count}"
        )

    rng = random.Random(seed)
    for groups in by_project.values():
        rng.shuffle(groups["single"])
        rng.shuffle(groups["multi"])

    single_projects = [
        project for project, groups in by_project.items() if groups["single"]
    ]
    rng.shuffle(single_projects)
    target_single = min(count // 2, len(single_projects))
    chosen_single_projects = single_projects[:target_single]
    selected = [by_project[project]["single"][0] for project in chosen_single_projects]
    used_projects = set(chosen_single_projects)

    multi_projects = [
        project
        for project, groups in by_project.items()
        if project not in used_projects and groups["multi"]
    ]
    rng.shuffle(multi_projects)
    target_multi = count - len(selected)
    if len(multi_projects) < target_multi:
        raise RuntimeError(
            f"Only {len(multi_projects)} remaining multi-problem projects; "
            f"need {target_multi}"
        )
    chosen_multi_projects = multi_projects[:target_multi]
    selected.extend(by_project[project]["multi"][0] for project in chosen_multi_projects)

    return selected, {
        "strategy": "one episode per unique project name; maximize single/multi balance",
        "seed": seed,
        "eligible_episodes": len(eligible),
        "eligible_unique_projects": len(by_project),
        "requested_episodes": count,
        "selected_unique_projects": len(set(chosen_single_projects + chosen_multi_projects)),
        "selected_by_stratum": {
            "single": len(chosen_single_projects),
            "multi": len(chosen_multi_projects),
        },
    }


def metric(rows: list[tuple[int, int, int]]) -> dict[str, float | int]:
    tp = sum(row[0] for row in rows)
    fp = sum(row[1] for row in rows)
    fn = sum(row[2] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def bootstrap_ci(
    rows: list[tuple[int, int, int]], *, seed: int = 20261111, samples: int = 10_000
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    distributions = {"precision": [], "recall": [], "f1": []}
    for _ in range(samples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        result = metric(sample)
        for key in distributions:
            distributions[key].append(float(result[key]))
    result: dict[str, list[float]] = {}
    for key, values in distributions.items():
        values.sort()
        result[key] = [values[249], values[9749]]
    return result


def proxy_rows(
    ids: list[str],
    references: dict[str, list[dict[str, Any]]],
    trace: dict[str, dict[str, Any]],
) -> tuple[list[tuple[int, int, int]], list[dict[str, Any]]]:
    counts: list[tuple[int, int, int]] = []
    prediction_records: list[dict[str, Any]] = []
    for current_id in ids:
        reference_rows = references[current_id]
        predicted_rows: list[tuple[dict[str, Any], str]] = []
        for problem in trace.get(current_id, {}).get("problems", []):
            for affected_file in problem.get("affected_files", []):
                predicted_rows.append((problem, affected_file))
                prediction_records.append(
                    {
                        "episode_id": current_id,
                        "predicted_problem_id": problem.get("problem_id"),
                        "problem": problem.get("problem", ""),
                        "failure_type": problem.get("failure_type", ""),
                        "issue_type": problem.get("issue_type", ""),
                        "affected_file": affected_file,
                    }
                )

        used_reference: set[int] = set()
        matches = 0
        for _, affected_file in predicted_rows:
            for reference_index, reference in enumerate(reference_rows):
                if reference_index in used_reference:
                    continue
                if files_match(affected_file, reference.get("file", "")):
                    used_reference.add(reference_index)
                    matches += 1
                    break
        counts.append(
            (
                matches,
                len(predicted_rows) - matches,
                len(reference_rows) - matches,
            )
        )
    return counts, prediction_records


def make_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Preliminary attribution results",
        "",
        "> These are automated **problem-file proxy** scores, not human-verified",
        "> problem-hunk scores. They use the native `affected_files` output and a",
        "> CI-visible reference derived from `log_details.json`.",
        "",
        "| Reconstruction | Episodes | Predictions | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("endpoint", "development", "bidirectional"):
        row = summary["methods"][method]
        lines.append(
            f"| {row['display_name']} | {summary['episodes']} | {row['predictions']} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | "
            f"{100 * row['precision']:.1f}% | {100 * row['recall']:.1f}% | "
            f"{100 * row['f1']:.1f}% |"
        )
    lines.extend(
        [
            "",
            f"Raw benchmark coverage: {summary['raw_coverage']}/{summary['episodes']} episodes.",
            f"CI-visible changed-file reference pairs: {summary['reference_pairs']} across "
            f"{summary['episodes_with_reference']} episodes.",
            f"Canonical unified-diff hunks: {summary['hunks']}.",
            f"Blinded problem-hunk candidate pairs: {summary['candidate_pairs']}.",
            "",
            "The final reviewer-facing table must be regenerated after a human fills",
            "`problem_hunk_pairs.csv`. Existing outputs do not identify hunk IDs, so a",
            "native problem-hunk comparison additionally requires reconstruction outputs",
            "that contain `supporting_hunk_ids`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--selection",
        choices=("unique-project", "stratified", "first"),
        default="unique-project",
    )
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "02_human_attribution_v2",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    memory_ids = [
        str(item) for item in read_json(PROJECT_ROOT / "data/memory_issue_ids.json")
    ]
    raw_by_id = {episode_id(item): item for item in read_jsonl(args.dataset)}
    log_by_id = {
        episode_id(item): item for item in read_json(PROJECT_ROOT / "data/log_details.json")
    }
    traces = {
        "endpoint": load_trace(PROJECT_ROOT / "data/back_trs/decomposed_issues.json"),
        "development": load_trace(PROJECT_ROOT / "data/fwr_trs/decomposed_issues.json"),
        "bidirectional": load_trace(
            PROJECT_ROOT / "data/bidirect_trs/decomposed_issues.json"
        ),
    }
    if args.selection == "unique-project":
        requested_ids, selection_metadata = select_unique_project_ids(
            memory_ids,
            raw_by_id,
            log_by_id,
            traces,
            count=args.count,
            seed=args.seed,
        )
    elif args.selection == "stratified":
        requested_ids, selection_metadata = select_stratified_ids(
            memory_ids,
            raw_by_id,
            log_by_id,
            traces,
            count=args.count,
            seed=args.seed,
        )
    else:
        requested_ids = memory_ids[: args.count]
        selection_metadata = {
            "strategy": "first N memory IDs (not stratified)",
            "seed": None,
            "eligible_episodes": None,
            "requested_episodes": args.count,
        }
    missing_raw = [item for item in requested_ids if item not in raw_by_id]
    if missing_raw:
        raise RuntimeError(f"Dataset is missing requested memory IDs: {missing_raw}")
    write_json(args.output_dir / "selection_metadata.json", selection_metadata)

    all_hunks: dict[str, list[Hunk]] = {}
    problems: dict[str, list[dict[str, Any]]] = {}
    proxy_references: dict[str, list[dict[str, Any]]] = {}
    manifest: list[dict[str, Any]] = []
    for current_id in requested_ids:
        raw = raw_by_id[current_id]
        current_hunks = parse_hunks(current_id, raw.get("diff", ""))
        all_hunks[current_id] = current_hunks
        relevant_files = list(log_by_id.get(current_id, {}).get("relevant_files", []))
        current_problems = []
        for index, problem in enumerate(relevant_files, 1):
            current_problems.append(
                {
                    "episode_id": current_id,
                    "problem_id": f"{current_id}:P{index:03d}",
                    **problem,
                }
            )
        problems[current_id] = current_problems
        changed_files = list(raw.get("changed_files", []))
        proxy_references[current_id] = [
            problem
            for problem in current_problems
            if any(files_match(problem.get("file", ""), path) for path in changed_files)
        ]
        manifest.append(
            {
                "episode_id": current_id,
                "repo": f"{raw.get('repo_owner', '')}/{raw.get('repo_name', '')}",
                "sha_fail": raw.get("sha_fail", ""),
                "sha_success": raw.get("sha_success", ""),
                "changed_files": len(changed_files),
                "ci_problem_records": len(current_problems),
                "hunks": len(current_hunks),
                "candidate_pairs": len(current_problems) * len(current_hunks),
                "trace_coverage": {
                    method: current_id in trace for method, trace in traces.items()
                },
            }
        )

    write_json(args.output_dir / "sample_manifest.json", manifest)
    write_jsonl(
        args.output_dir / "problems.jsonl",
        (problem for current_id in requested_ids for problem in problems[current_id]),
    )
    write_jsonl(
        args.output_dir / "hunks.jsonl",
        (
            hunk.__dict__
            for current_id in requested_ids
            for hunk in all_hunks[current_id]
        ),
    )

    pair_path = args.output_dir / "problem_hunk_pairs.csv"
    pair_fields = [
        "episode_id",
        "problem_id",
        "problem_file",
        "problem_line",
        "failed_tool",
        "issue_type",
        "problem_reason",
        "hunk_id",
        "hunk_file",
        "old_range",
        "new_range",
        "hunk_header",
        "hunk_content",
        "label",
        "confidence",
        "rationale",
        "annotator",
    ]
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        for current_id in requested_ids:
            for problem in problems[current_id]:
                for hunk in all_hunks[current_id]:
                    writer.writerow(
                        {
                            "episode_id": current_id,
                            "problem_id": problem["problem_id"],
                            "problem_file": problem.get("file", ""),
                            "problem_line": problem.get("line_number", ""),
                            "failed_tool": problem.get("failed_tool", ""),
                            "issue_type": problem.get("issue_type", ""),
                            "problem_reason": problem.get("reason", ""),
                            "hunk_id": hunk.hunk_id,
                            "hunk_file": hunk.file_path,
                            "old_range": f"{hunk.old_start},{hunk.old_count}",
                            "new_range": f"{hunk.new_start},{hunk.new_count}",
                            "hunk_header": hunk.header,
                            "hunk_content": hunk.content,
                            "label": "",
                            "confidence": "",
                            "rationale": "",
                            "annotator": "",
                        }
                    )

    prediction_dir = args.output_dir / "predictions_blinded_from_reference"
    prediction_dir.mkdir(exist_ok=True)
    summary: dict[str, Any] = {
        "evaluation_type": "automated problem-file proxy; not human verified",
        "selection": selection_metadata,
        "episodes": len(requested_ids),
        "raw_coverage": len(requested_ids) - len(missing_raw),
        "reference_pairs": sum(len(value) for value in proxy_references.values()),
        "episodes_with_reference": sum(bool(value) for value in proxy_references.values()),
        "hunks": sum(len(value) for value in all_hunks.values()),
        "candidate_pairs": sum(item["candidate_pairs"] for item in manifest),
        "methods": {},
    }
    display_names = {
        "endpoint": "Endpoint-based",
        "development": "Development-based",
        "bidirectional": "Full ReCAP",
    }
    for method, trace in traces.items():
        counts, prediction_records = proxy_rows(requested_ids, proxy_references, trace)
        write_jsonl(prediction_dir / f"{method}.jsonl", prediction_records)
        result = metric(counts)
        result["display_name"] = display_names[method]
        result["trace_coverage"] = sum(item in trace for item in requested_ids)
        result["predictions"] = sum(row[0] + row[1] for row in counts)
        result["bootstrap_95_ci"] = bootstrap_ci(counts)
        summary["methods"][method] = result

    write_json(args.output_dir / "proxy_problem_file_scores.json", summary)
    (args.output_dir / "PROXY_RESULTS.md").write_text(
        make_markdown_report(summary), encoding="utf-8"
    )
    with gzip.open(args.output_dir / "raw_episodes_50.jsonl.gz", "wt", encoding="utf-8") as handle:
        for current_id in requested_ids:
            handle.write(json.dumps(raw_by_id[current_id], ensure_ascii=False) + "\n")

    print(f"Prepared {len(requested_ids)} episodes in {args.output_dir}")
    print(f"Hunks: {summary['hunks']}")
    print(f"Candidate problem-hunk pairs: {summary['candidate_pairs']}")
    for method, result in summary["methods"].items():
        print(
            f"{method}: P={100 * result['precision']:.1f}% "
            f"R={100 * result['recall']:.1f}% F1={100 * result['f1']:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
