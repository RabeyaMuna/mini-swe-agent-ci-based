#!/usr/bin/env python3
"""Build and validate repository-disjoint L1/L2/L3 experiment data.

The target set is the fixed list in ``experimental_data/eval_issue_ids.json``.
Historical L1 records from any target repository are excluded. L2 records and
L3 patterns are then selected by the retained L1 issue IDs so all levels have
identical source lineage.

Repository comparison deliberately uses case-folded ``repo_name`` rather than
owner/name. This conservatively treats forks or mirrors with the same project
name as the same repository family.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MEMORY_DIR = PROJECT_ROOT / "data" / "bidirect_trs"
OUTPUT_ROOT = PROJECT_ROOT / "experimental_data"
OUTPUT_MEMORY_DIR = OUTPUT_ROOT / "bidirect_trs"
SOURCE_EVAL_IDS = OUTPUT_ROOT / "eval_issue_ids.json"
DEFAULT_EVAL_DATASET = PROJECT_ROOT / "results" / "think_repair" / "instances_with_plans.jsonl"

EXPECTED_EVAL_COUNT = 100
EXPECTED_TARGET_REPO_COUNT = 30
EXPECTED_L1_COUNT = 110
EXPECTED_L2_COUNT = 110
EXPECTED_L3_COUNT = 302
EXPECTED_EXCLUDED_L1_COUNT = 47


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def canonical_repo_name(record: dict[str, Any]) -> str:
    repo_name = str(record.get("repo_name") or "").strip()
    if not repo_name:
        repo_name = str(record.get("repo") or record.get("source_repo") or "").strip()
        repo_name = repo_name.rsplit("/", 1)[-1]
    if not repo_name:
        raise ValueError(f"Record has no repository identity: {record}")
    return repo_name.casefold()


def instance_id(record: dict[str, Any]) -> str:
    value = record.get("instance_id") or record.get("id")
    if value is None:
        raise ValueError(f"Evaluation record has no instance ID: {record}")
    return str(value)


def require_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate IDs")


@dataclass(frozen=True)
class CrossRepoData:
    eval_ids: list[str]
    eval_records: list[dict[str, Any]]
    target_repos: set[str]
    source_repos: set[str]
    retained_ids: list[str]
    excluded_ids: list[str]
    l1: list[dict[str, Any]]
    l2: list[dict[str, Any]]
    l3: list[dict[str, Any]]


def build_cross_repo_data(eval_dataset_path: Path) -> CrossRepoData:
    all_eval_ids = [str(value) for value in load_json(SOURCE_EVAL_IDS)]
    eval_ids = all_eval_ids[:EXPECTED_EVAL_COUNT]
    require_unique(eval_ids, "Evaluation set")
    if len(eval_ids) != EXPECTED_EVAL_COUNT:
        raise ValueError(f"Expected {EXPECTED_EVAL_COUNT} evaluation IDs, found {len(eval_ids)}")

    eval_by_id = {instance_id(record): record for record in load_jsonl(eval_dataset_path)}
    missing_eval_ids = [value for value in eval_ids if value not in eval_by_id]
    if missing_eval_ids:
        raise ValueError(f"Evaluation metadata is missing IDs: {missing_eval_ids}")
    eval_records = [eval_by_id[value] for value in eval_ids]
    target_repos = {canonical_repo_name(record) for record in eval_records}

    full_l1 = load_json(SOURCE_MEMORY_DIR / "failure_memory.json")
    full_l2 = load_json(SOURCE_MEMORY_DIR / "repo_memory.json")
    full_l3 = load_json(SOURCE_MEMORY_DIR / "cross_memory.json")

    l1 = [record for record in full_l1 if canonical_repo_name(record) not in target_repos]
    excluded_l1 = [record for record in full_l1 if canonical_repo_name(record) in target_repos]
    retained_ids = [str(record["issue_id"]) for record in l1]
    excluded_ids = [str(record["issue_id"]) for record in excluded_l1]
    require_unique(retained_ids, "Retained L1 memory")
    retained_id_set = set(retained_ids)

    l2 = [record for record in full_l2 if str(record.get("issue_id")) in retained_id_set]
    l3 = [record for record in full_l3 if str(record.get("source_issue_id")) in retained_id_set]
    source_repos = {canonical_repo_name(record) for record in l1}

    l2_ids = [str(record.get("issue_id")) for record in l2]
    l3_source_ids = {str(record.get("source_issue_id")) for record in l3}
    require_unique(l2_ids, "Retained L2 memory")

    problems: list[str] = []
    if len(target_repos) != EXPECTED_TARGET_REPO_COUNT:
        problems.append(
            f"expected {EXPECTED_TARGET_REPO_COUNT} target repositories, found {len(target_repos)}"
        )
    if len(l1) != EXPECTED_L1_COUNT:
        problems.append(f"expected {EXPECTED_L1_COUNT} L1 records, found {len(l1)}")
    if len(excluded_l1) != EXPECTED_EXCLUDED_L1_COUNT:
        problems.append(
            f"expected {EXPECTED_EXCLUDED_L1_COUNT} excluded L1 records, found {len(excluded_l1)}"
        )
    if len(l2) != EXPECTED_L2_COUNT:
        problems.append(f"expected {EXPECTED_L2_COUNT} L2 records, found {len(l2)}")
    if len(l3) != EXPECTED_L3_COUNT:
        problems.append(f"expected {EXPECTED_L3_COUNT} L3 patterns, found {len(l3)}")
    if set(l2_ids) != retained_id_set:
        problems.append("L2 issue IDs do not exactly match the retained L1 issue IDs")
    if l3_source_ids != retained_id_set:
        problems.append("L3 source issue IDs do not exactly cover the retained L1 issue IDs")
    overlap = source_repos & target_repos
    if overlap:
        problems.append(f"source/target repository overlap is not empty: {sorted(overlap)}")
    if problems:
        raise ValueError("Invalid cross-repository split:\n- " + "\n- ".join(problems))

    return CrossRepoData(
        eval_ids=eval_ids,
        eval_records=eval_records,
        target_repos=target_repos,
        source_repos=source_repos,
        retained_ids=retained_ids,
        excluded_ids=excluded_ids,
        l1=l1,
        l2=l2,
        l3=l3,
    )


def manifest(data: CrossRepoData, eval_dataset_path: Path) -> dict[str, Any]:
    return {
        "method": "exclude every historical record whose case-folded repo_name appears in the fixed evaluation subset",
        "repository_identity": "case-folded repo_name (conservatively groups forks/mirrors with the same project name)",
        "source_eval_dataset": str(eval_dataset_path.relative_to(PROJECT_ROOT)),
        "source_memory_directory": str(SOURCE_MEMORY_DIR.relative_to(PROJECT_ROOT)),
        "counts": {
            "evaluation_instances": len(data.eval_ids),
            "target_repositories": len(data.target_repos),
            "retained_source_repositories": len(data.source_repos),
            "excluded_l1_histories": len(data.excluded_ids),
            "l1_records": len(data.l1),
            "l2_records": len(data.l2),
            "l3_patterns": len(data.l3),
        },
        "target_repositories": sorted(data.target_repos),
        "source_repositories": sorted(data.source_repos),
        "retained_memory_issue_ids": data.retained_ids,
        "excluded_memory_issue_ids": data.excluded_ids,
        "verification": {
            "source_target_repository_overlap": sorted(data.source_repos & data.target_repos),
            "l2_ids_equal_l1_ids": {str(record["issue_id"]) for record in data.l2}
            == set(data.retained_ids),
            "l3_sources_cover_l1_ids": {
                str(record["source_issue_id"]) for record in data.l3
            }
            == set(data.retained_ids),
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def expected_outputs(data: CrossRepoData, eval_dataset_path: Path) -> dict[Path, Any]:
    return {
        OUTPUT_ROOT / "eval_issue_ids.json": data.eval_ids,
        OUTPUT_ROOT / "memory_issue_ids.json": data.retained_ids,
        OUTPUT_ROOT / "split_manifest.json": manifest(data, eval_dataset_path),
        OUTPUT_MEMORY_DIR / "failure_memory.json": data.l1,
        OUTPUT_MEMORY_DIR / "repo_memory.json": data.l2,
        OUTPUT_MEMORY_DIR / "cross_memory.json": data.l3,
    }


def validate_written_outputs(data: CrossRepoData, eval_dataset_path: Path) -> None:
    stale: list[str] = []
    for path, expected in expected_outputs(data, eval_dataset_path).items():
        if not path.exists() or load_json(path) != expected:
            stale.append(str(path.relative_to(PROJECT_ROOT)))

    eval_set_path = OUTPUT_ROOT / "eval_set.jsonl"
    if not eval_set_path.exists() or load_jsonl(eval_set_path) != data.eval_records:
        stale.append(str(eval_set_path.relative_to(PROJECT_ROOT)))
    if stale:
        raise ValueError(
            "Cross-repository outputs are missing or stale; run setup.sh:\n- "
            + "\n- ".join(stale)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dataset",
        type=Path,
        default=DEFAULT_EVAL_DATASET,
        help="JSONL containing repository metadata for the evaluation IDs",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify generated files without rewriting them",
    )
    args = parser.parse_args()
    eval_dataset_path = args.eval_dataset.resolve()
    data = build_cross_repo_data(eval_dataset_path)

    if args.validate_only:
        validate_written_outputs(data, eval_dataset_path)
        action = "Validated"
    else:
        for path, value in expected_outputs(data, eval_dataset_path).items():
            write_json(path, value)
        write_jsonl(OUTPUT_ROOT / "eval_set.jsonl", data.eval_records)
        validate_written_outputs(data, eval_dataset_path)
        action = "Prepared"

    print(
        f"{action} cross-repository data: "
        f"eval={len(data.eval_ids)}, target_repos={len(data.target_repos)}, "
        f"source_repos={len(data.source_repos)}, L1={len(data.l1)}, "
        f"L2={len(data.l2)}, L3={len(data.l3)}, overlap=0"
    )


if __name__ == "__main__":
    main()
