#!/usr/bin/env python3
"""
Run memory-guided CIBench repair on remaining Camel and Flower issues.

This script is a thin ablation harness around the production CIBench runner.
It selects eval issues, writes a temporary JSON dataset, and runs the agent with
different memory levels:

    L1
    L1+L2
    L1+L2+L3

Examples:
    python scripts/test_memory_guided_repair.py
    python scripts/test_memory_guided_repair.py --max-issues 5 --dry-run
    python scripts/test_memory_guided_repair.py --repos camel,flower --ablations L1,L1+L2,L1+L2+L3
    python scripts/test_memory_guided_repair.py --issue-ids 121,145 --redo-existing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_DATASET = PROJECT_ROOT / "data" / "trs" / "memory_seed_issues.json"
DEFAULT_EVAL_DATASET = PROJECT_ROOT / "data" / "trs" / "eval_issues.json"
DEFAULT_LOG_DETAILS = PROJECT_ROOT / "data" / "trs" / "log_details.json"
DEFAULT_DATASET = DEFAULT_SEED_DATASET if DEFAULT_SEED_DATASET.exists() else DEFAULT_EVAL_DATASET
DEFAULT_MEMORY_ROOT = PROJECT_ROOT / "data" / "trs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "memory_guided_ablation"
DEFAULT_SUCCESS_PATCHES = PROJECT_ROOT / "data" / "generated_patches_success_only.json"


@dataclass
class RunResult:
    ablation: str
    command: list[str]
    output_dir: Path
    returncode: int | None
    elapsed_seconds: float
    preds_count: int
    patched_count: int
    failed_count: int
    error: str


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def normalize_repo(value: str) -> str:
    value = str(value or "").strip().lower()
    if "/" in value:
        return value.split("/")[-1]
    return value


def repo_matches(issue: dict[str, Any], repo_filters: set[str]) -> bool:
    if not repo_filters:
        return True
    repo_name = normalize_repo(issue.get("repo_name", ""))
    full_repo = f"{issue.get('repo_owner', '')}/{issue.get('repo_name', '')}".strip("/").lower()
    candidates = {repo_name, full_repo, normalize_repo(full_repo)}
    return bool(candidates & repo_filters)


def issue_id(issue: dict[str, Any]) -> str:
    return str(issue.get("id") or issue.get("instance_id") or "").strip()


def load_eval_issues(dataset_path: Path) -> list[dict[str, Any]]:
    data = read_json(dataset_path, default=[])
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {dataset_path}")
    return [item for item in data if isinstance(item, dict)]


def merge_explicit_id_fallbacks(
    issues: list[dict[str, Any]],
    *,
    explicit_ids: set[str],
    fallback_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if not explicit_ids:
        return issues, {}

    by_id = {issue_id(issue): issue for issue in issues if issue_id(issue)}
    found_sources: dict[str, list[str]] = {iid: ["primary dataset"] for iid in by_id if iid in explicit_ids}

    for path in fallback_paths:
        rows = load_eval_issues(path) if path.exists() else []
        for row in rows:
            iid = issue_id(row)
            if iid not in explicit_ids:
                continue
            found_sources.setdefault(iid, []).append(str(path))
            if iid not in by_id:
                candidate = dict(row)
                candidate.setdefault("_source_dataset", str(path))
                by_id[iid] = candidate

    merged = list(issues)
    existing_ids = {issue_id(issue) for issue in merged}
    for iid in sorted(explicit_ids):
        if iid in by_id and iid not in existing_ids:
            merged.append(by_id[iid])
            existing_ids.add(iid)

    return merged, found_sources


def load_memory_issue_ids(memory_root: Path) -> set[str]:
    ids: set[str] = set()

    for rel_path, keys in [
        ("failure_memory.json", ("issue_id", "original_issue_id", "id", "instance_id")),
        ("repo_memory.json", ("issue_id", "original_issue_id", "id", "instance_id")),
        ("decomposed_issues.json", ("original_issue_id", "issue_id", "id", "instance_id")),
    ]:
        rows = read_json(memory_root / rel_path, default=[])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in keys:
                value = row.get(key)
                if value:
                    ids.add(str(value))
                    break

    return ids


def load_success_patch_issue_ids(path: Path) -> set[str]:
    rows = read_json(path, default=[])
    if not isinstance(rows, list):
        return set()

    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("id", "issue_id", "original_issue_id", "instance_id"):
            value = row.get(key)
            if value:
                ids.add(str(value))
                break
    return ids


def select_issues(
    issues: list[dict[str, Any]],
    *,
    repos: set[str],
    explicit_ids: set[str],
    known_ids: set[str],
    include_known: bool,
    max_issues: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    excluded = []

    for issue in issues:
        iid = issue_id(issue)
        if not iid:
            continue
        if explicit_ids and iid not in explicit_ids:
            continue
        if not explicit_ids and not repo_matches(issue, repos):
            continue
        if not include_known and iid in known_ids:
            excluded.append(issue)
            continue
        selected.append(issue)

    selected.sort(key=lambda item: (normalize_repo(item.get("repo_name", "")), int(issue_id(item)) if issue_id(item).isdigit() else issue_id(item)))
    excluded.sort(key=lambda item: (normalize_repo(item.get("repo_name", "")), int(issue_id(item)) if issue_id(item).isdigit() else issue_id(item)))

    if max_issues is not None:
        selected = selected[:max_issues]
    return selected, excluded


def ablation_output_dir(output_root: Path, ablation: str) -> Path:
    return output_root / ablation.replace("+", "_")


def ablation_levels(ablation: str) -> list[str]:
    return [level.strip().upper() for level in ablation.split("+") if level.strip()]


def load_preds(path: Path) -> dict[str, Any]:
    data = read_json(path, default={})
    return data if isinstance(data, dict) else {}


def summarize_preds(output_dir: Path) -> tuple[int, int, int]:
    preds = load_preds(output_dir / "preds.json")
    patched = 0
    failed = 0
    for row in preds.values():
        if isinstance(row, dict) and str(row.get("diff", "")).strip():
            patched += 1
        else:
            failed += 1
    return len(preds), patched, failed


def build_cibench_command(
    *,
    dataset_path: Path,
    output_root: Path,
    memory_root: Path,
    ablation: str,
    workers: int,
    memory_top_k: int,
    model: str | None,
    config: list[str],
    redo_existing: bool,
    no_save_memory: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "minisweagent.run.benchmarks.cibench",
        "--dataset",
        str(dataset_path),
        "--split",
        "train",
        "--output",
        str(output_root),
        "--workers",
        str(workers),
        "--memory-enabled",
        "--memory-root",
        str(memory_root),
        "--memory-ablation",
        ablation,
        "--memory-top-k",
        str(memory_top_k),
    ]

    for config_spec in config:
        cmd.extend(["--config", config_spec])

    if model:
        cmd.extend(["--model", model])

    if redo_existing:
        cmd.append("--redo-existing")

    if no_save_memory:
        cmd.append("--no-save-memory")

    return cmd


def run_ablation(
    *,
    dataset_path: Path,
    output_root: Path,
    memory_root: Path,
    ablation: str,
    workers: int,
    memory_top_k: int,
    model: str | None,
    config: list[str],
    redo_existing: bool,
    no_save_memory: bool,
    timeout: int,
    dry_run: bool,
) -> RunResult:
    cmd = build_cibench_command(
        dataset_path=dataset_path,
        output_root=output_root,
        memory_root=memory_root,
        ablation=ablation,
        workers=workers,
        memory_top_k=memory_top_k,
        model=model,
        config=config,
        redo_existing=redo_existing,
        no_save_memory=no_save_memory,
    )

    expected_output = ablation_output_dir(output_root, ablation)
    print("\n" + "=" * 80)
    print(f"ABLATION: {ablation}")
    print(f"Active memory levels: {', '.join(ablation_levels(ablation))}")
    print("=" * 80)
    print("Command:")
    print("  " + " ".join(cmd))

    if dry_run:
        return RunResult(
            ablation=ablation,
            command=cmd,
            output_dir=expected_output,
            returncode=None,
            elapsed_seconds=0.0,
            preds_count=0,
            patched_count=0,
            failed_count=0,
            error="dry-run",
        )

    start = time.time()
    error = ""
    returncode = 0

    try:
        process = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            timeout=timeout,
        )
        returncode = process.returncode
        if process.returncode != 0:
            error = f"command exited with {process.returncode}"
    except subprocess.TimeoutExpired:
        returncode = 124
        error = f"timeout after {timeout}s"
    except Exception as exc:
        returncode = 1
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.time() - start
    preds_count, patched_count, failed_count = summarize_preds(expected_output)

    print(f"\nAblation {ablation} finished in {elapsed:.1f}s")
    print(f"  returncode: {returncode}")
    print(f"  preds: {preds_count}")
    print(f"  patched: {patched_count}")
    print(f"  failed/no patch: {failed_count}")
    if error:
        print(f"  error: {error}")

    return RunResult(
        ablation=ablation,
        command=cmd,
        output_dir=expected_output,
        returncode=returncode,
        elapsed_seconds=elapsed,
        preds_count=preds_count,
        patched_count=patched_count,
        failed_count=failed_count,
        error=error,
    )


def print_issue_plan(issues: list[dict[str, Any]], title: str = "Selected issues") -> None:
    by_repo: dict[str, list[str]] = {}
    for issue in issues:
        repo = normalize_repo(issue.get("repo_name", "unknown")) or "unknown"
        by_repo.setdefault(repo, []).append(issue_id(issue))

    print(f"\n{title}:")
    for repo, ids in sorted(by_repo.items()):
        preview = ", ".join(ids[:20])
        suffix = f" ... +{len(ids) - 20} more" if len(ids) > 20 else ""
        print(f"  {repo}: {len(ids)} issue(s): {preview}{suffix}")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run memory-guided repair ablations for remaining Camel and Flower issues."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repos", default="camel,flower")
    parser.add_argument("--issue-ids", default="", help="Comma-separated issue IDs. Overrides repo filtering.")
    parser.add_argument(
        "--ablations",
        "--ablation-levels",
        dest="ablations",
        default="L1,L1+L2,L1+L2+L3",
        help="Comma-separated memory levels to test, for example L1,L1+L2,L1+L2+L3.",
    )
    parser.add_argument("--max-issues", type=int, default=None)
    parser.add_argument("--success-patches", type=Path, default=DEFAULT_SUCCESS_PATCHES)
    parser.add_argument(
        "--include-known",
        "--include-in-memory",
        dest="include_known",
        action="store_true",
        default=False,
        help="Include issues that already exist in memory or generated successful patches. Use only for debugging reruns.",
    )
    parser.add_argument(
        "--exclude-known",
        "--exclude-in-memory",
        dest="include_known",
        action="store_false",
        help="Exclude issues already present in memory or generated successful patches. This is the default evaluation behavior.",
    )
    parser.add_argument("--redo-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--save-memory",
        action="store_true",
        help="Allow CIBench to append new successful repairs into memory. Default is disabled for ablation purity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repos = {normalize_repo(repo) for repo in parse_csv(args.repos)}
    explicit_ids = set(parse_csv(args.issue_ids))
    ablations = parse_csv(args.ablations)

    if not ablations:
        print("ERROR: no ablations requested")
        return 2

    issues = load_eval_issues(args.dataset)
    issues, explicit_sources = merge_explicit_id_fallbacks(
        issues,
        explicit_ids=explicit_ids,
        fallback_paths=[DEFAULT_LOG_DETAILS, args.success_patches],
    )
    memory_issue_ids = load_memory_issue_ids(args.memory_root)
    success_patch_issue_ids = load_success_patch_issue_ids(args.success_patches)
    known_issue_ids = memory_issue_ids | success_patch_issue_ids
    selected, excluded = select_issues(
        issues,
        repos=repos,
        explicit_ids=explicit_ids,
        known_ids=known_issue_ids,
        include_known=args.include_known,
        max_issues=args.max_issues,
    )

    print("=" * 80)
    print("MEMORY-GUIDED REPAIR ABLATION TEST")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Memory root: {args.memory_root}")
    print(f"Success patches: {args.success_patches}")
    print(f"Output root: {args.output_root}")
    print(f"Repos: {', '.join(sorted(repos)) if repos else 'all'}")
    print(f"Ablations: {', '.join(ablations)}")
    print(f"Exclude known issues: {not args.include_known}")
    print(f"Known memory issue IDs: {len(memory_issue_ids)}")
    print(f"Known success patch issue IDs: {len(success_patch_issue_ids)}")
    print(f"Known issue IDs excluded: {len(known_issue_ids) if not args.include_known else 0}")
    print(f"Excluded matching candidates: {len(excluded)}")
    print(f"Selected issues: {len(selected)}")
    if explicit_ids:
        found_ids = {issue_id(issue) for issue in selected + excluded}
        missing_ids = sorted(explicit_ids - found_ids)
        if explicit_sources:
            print("Explicit ID sources:")
            for iid in sorted(explicit_ids):
                sources = explicit_sources.get(iid, [])
                print(f"  {iid}: {', '.join(sources) if sources else 'not found'}")
        if missing_ids:
            print(f"Missing explicit IDs: {', '.join(missing_ids)}")

    if not selected:
        print("\nNo issues selected.")
        if excluded:
            print("All matching candidates are already known from memory or successful generated patches.")
            print_issue_plan(excluded, title="Excluded issues")
            print("\nUse --include-known only for debugging reruns, not for evaluation.")
        return 0

    print_issue_plan(selected)

    run_root = args.output_root / time.strftime("%Y%m%d_%H%M%S")
    dataset_path = run_root / "selected_issues.json"
    write_json(dataset_path, selected)
    print(f"\nWrote selected dataset: {dataset_path}")

    results: list[RunResult] = []
    interrupted = False
    try:
        for ablation in ablations:
            result = run_ablation(
                dataset_path=dataset_path,
                output_root=run_root,
                memory_root=args.memory_root,
                ablation=ablation,
                workers=args.workers,
                memory_top_k=args.memory_top_k,
                model=args.model,
                config=args.config,
                redo_existing=args.redo_existing,
                no_save_memory=not args.save_memory,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
            results.append(result)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Saving partial summary.")

    summary = {
        "dataset": str(args.dataset),
        "selected_dataset": str(dataset_path),
        "memory_root": str(args.memory_root),
        "success_patches": str(args.success_patches),
        "output_root": str(run_root),
        "repos": sorted(repos),
        "issue_ids": [issue_id(issue) for issue in selected],
        "excluded_issue_ids": [issue_id(issue) for issue in excluded],
        "known_memory_issue_ids_count": len(memory_issue_ids),
        "known_success_patch_issue_ids_count": len(success_patch_issue_ids),
        "include_known": args.include_known,
        "interrupted": interrupted,
        "dry_run": args.dry_run,
        "ablations": [
            {
                "ablation": result.ablation,
                "output_dir": str(result.output_dir),
                "returncode": result.returncode,
                "elapsed_seconds": result.elapsed_seconds,
                "preds_count": result.preds_count,
                "patched_count": result.patched_count,
                "failed_count": result.failed_count,
                "error": result.error,
                "command": result.command,
            }
            for result in results
        ],
    }
    summary_path = run_root / "ablation_summary.json"
    write_json(summary_path, summary)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results:
        status = "DRY-RUN" if result.returncode is None else ("OK" if result.returncode == 0 else "FAIL")
        print(
            f"{result.ablation:10} {status:7} "
            f"preds={result.preds_count} patched={result.patched_count} "
            f"failed={result.failed_count} output={result.output_dir}"
        )
    print(f"\nSummary saved to: {summary_path}")

    if interrupted:
        return 130
    return 0 if all(result.returncode in (0, None) for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
