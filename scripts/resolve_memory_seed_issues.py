#!/usr/bin/env python3
"""
Resolve memory seed issues from a benchmark dataset.

Inputs can be:
- a repo filter, e.g. --repo camel or --repo camel-ai/camel
- a generated-patches file containing ids / sha_fail values
- a seed file containing full issue records or id / sha_fail values

Output is always a JSON list of full benchmark issue records suitable for
decompose_ci_failure.py. If --eval-output is provided with --repo and
--patches-file, remaining repo issues are saved for evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

DEFAULT_HF_DATASET = "ci-benchmark-user/ci-repair-bench"


def _load_project_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    load_dotenv(dotenv_path=Path(".env"), override=False)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_dataset_rows(dataset_spec: str) -> List[Dict[str, Any]]:
    path = Path(dataset_spec)
    if path.exists():
        data = _load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"local dataset must be a JSON list: {path}")
        return [row for row in data if isinstance(row, dict)]

    try:
        from datasets import DatasetDict, load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for HuggingFace dataset loading. "
            "Install it or pass a local JSON dataset path."
        ) from exc

    hf_token = (
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    load_kwargs = {"token": hf_token} if hf_token else {}
    loaded = load_dataset(dataset_spec, **load_kwargs)
    rows: List[Dict[str, Any]] = []

    if isinstance(loaded, DatasetDict) or hasattr(loaded, "keys"):
        for split_name in loaded.keys():
            rows.extend(dict(row) for row in loaded[split_name])
    else:
        rows.extend(dict(row) for row in loaded)

    return rows


def _repo_text(issue: Dict[str, Any]) -> str:
    owner = str(issue.get("repo_owner") or "")
    name = str(issue.get("repo_name") or "")
    repo = str(issue.get("repo") or "")
    combined = f"{owner}/{name}".strip("/")
    return " ".join(x for x in [repo, combined, owner, name] if x).lower()


def _matches_repo(issue: Dict[str, Any], repo_filter: str) -> bool:
    query = repo_filter.lower().strip()
    return bool(query) and query in _repo_text(issue)


def _reference_keys(row: Any) -> Set[str]:
    keys: Set[str] = set()
    if isinstance(row, str):
        if row.strip():
            keys.add(row.strip())
        return keys
    if not isinstance(row, dict):
        return keys
    for field in ("id", "issue_id", "instance_id", "sha_fail"):
        value = row.get(field)
        if value not in (None, ""):
            keys.add(str(value))
    return keys


def _is_full_issue(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    required = ("sha_fail", "logs", "diff", "workflow")
    return all(row.get(field) not in (None, "") for field in required)


def _index_dataset(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for key in _reference_keys(row):
            index[key] = row
    return index


def _issue_key(issue: Dict[str, Any]) -> str:
    return str(issue.get("id") or issue.get("sha_fail") or "")


def _issue_matches_keys(issue: Dict[str, Any], keys: Set[str]) -> bool:
    return any(key in keys for key in _reference_keys(issue))


def _load_reference_keys(path: Path) -> Set[str]:
    data = _load_json(path)
    keys: Set[str] = set()
    if isinstance(data, dict):
        keys.update(str(k) for k in data.keys() if str(k).strip())
        for value in data.values():
            keys.update(_reference_keys(value))
    elif isinstance(data, list):
        for row in data:
            keys.update(_reference_keys(row))
    else:
        keys.update(_reference_keys(data))
    return keys


def main() -> int:
    _load_project_env()

    parser = argparse.ArgumentParser(description="Resolve full memory seed issue records")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_HF_DATASET,
        help="Full benchmark dataset JSON path or HuggingFace dataset name",
    )
    parser.add_argument("--repo", help="Repo filter, e.g. camel, agno, flower, camel-ai/camel")
    parser.add_argument(
        "--patches-file",
        help="Generated patches JSON; repo issues matching id or sha_fail become memory seeds",
    )
    parser.add_argument(
        "--seed-file",
        help="Optional JSON file containing full issues or id/sha_fail references",
    )
    parser.add_argument("--output", required=True, help="Output JSON with full issue records")
    parser.add_argument(
        "--eval-output",
        help="Optional output JSON for remaining repo issues not selected as memory seeds",
    )
    args = parser.parse_args()

    try:
        dataset = _load_dataset_rows(args.dataset)
    except Exception as exc:
        print(f"ERROR: failed to load dataset {args.dataset!r}: {exc}")
        return 1

    dataset_index = _index_dataset([row for row in dataset if isinstance(row, dict)])
    selected: List[Dict[str, Any]] = []
    eval_selected: List[Dict[str, Any]] = []

    if args.repo and args.patches_file:
        patches_path = Path(args.patches_file)
        if not patches_path.exists():
            print(f"ERROR: patches file not found: {patches_path}")
            return 1
        patch_keys = _load_reference_keys(patches_path)
        repo_issues = [
            row for row in dataset
            if isinstance(row, dict) and _matches_repo(row, args.repo)
        ]
        selected = [issue for issue in repo_issues if _issue_matches_keys(issue, patch_keys)]
        eval_selected = [issue for issue in repo_issues if not _issue_matches_keys(issue, patch_keys)]
        unresolved_patch_keys = sorted(
            key for key in patch_keys
            if key not in dataset_index
        )
        if unresolved_patch_keys:
            print(f"WARNING: {len(unresolved_patch_keys)} patch keys were not found in dataset")
    elif args.seed_file:
        seed_path = Path(args.seed_file)
        if not seed_path.exists():
            print(f"ERROR: seed file not found: {seed_path}")
            return 1
        seeds = _load_json(seed_path)
        if not isinstance(seeds, list):
            print(f"ERROR: seed file must be a JSON list: {seed_path}")
            return 1

        missing: List[str] = []
        for seed in seeds:
            if _is_full_issue(seed):
                issue = seed
            else:
                keys = _reference_keys(seed)
                issue = next((dataset_index[k] for k in keys if k in dataset_index), None)
                if issue is None:
                    missing.extend(sorted(keys))
                    continue
            if args.repo and not _matches_repo(issue, args.repo):
                continue
            selected.append(issue)

        if missing:
            print(f"WARNING: unresolved seed references: {sorted(set(missing))}")
    elif args.repo:
        selected = [
            row for row in dataset
            if isinstance(row, dict) and _matches_repo(row, args.repo)
        ]
    else:
        selected = [row for row in dataset if isinstance(row, dict)]

    # Deduplicate by id/sha_fail while preserving order.
    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for issue in selected:
        key = _issue_key(issue) or str(len(deduped))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)

    eval_deduped: List[Dict[str, Any]] = []
    eval_seen: Set[str] = set()
    for issue in eval_selected:
        key = _issue_key(issue) or str(len(eval_deduped))
        if key in eval_seen:
            continue
        eval_seen.add(key)
        eval_deduped.append(issue)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(deduped, fh, indent=2)

    print(f"Resolved {len(deduped)} memory seed issues -> {output}")
    if args.repo:
        print(f"Repo filter: {args.repo}")
    print("IDs:", [issue.get("id") for issue in deduped])

    if args.eval_output:
        eval_output = Path(args.eval_output)
        eval_output.parent.mkdir(parents=True, exist_ok=True)
        with eval_output.open("w", encoding="utf-8") as fh:
            json.dump(eval_deduped, fh, indent=2)
        print(f"Resolved {len(eval_deduped)} eval issues -> {eval_output}")
        print("Eval IDs:", [issue.get("id") for issue in eval_deduped])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
