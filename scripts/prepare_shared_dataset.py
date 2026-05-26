#!/usr/bin/env python3
"""
prepare_shared_dataset.py
=========================
Joins the TRS split from CI-REPAIR-BENCH with lca_dataset.parquet and
produces two JSONL files that give every agent the SAME benchmark:

    data/eval_dataset.jsonl    — 189 eval issues (mini-swe-agent cibench input)
    data/memory_seed.jsonl     — 103 memory issues (fed to seed_memory.py)

Using the same eval set + the same seeded memory bank across all agents
(mini-swe-agent, mem-ci-repair-agent, …) guarantees fair, consistent comparison.

Usage
-----
    # Defaults — reads CI-REPAIR-BENCH next to this repo
    python scripts/prepare_shared_dataset.py

    # Explicit paths
    python scripts/prepare_shared_dataset.py \\
        --trs-dir  /path/to/CI-REPAIR-BENCH/results/trs_split \\
        --parquet  /path/to/CI-REPAIR-BENCH/dataset/lca_dataset.parquet \\
        --out-dir  data/

    # Dry-run: print counts but do not write files
    python scripts/prepare_shared_dataset.py --dry-run

Output schema (both files, per line)
-------------------------------------
    instance_id      str   unique key for cibench (= sha_fail)
    sha_fail         str   failing commit SHA
    repo_owner       str
    repo_name        str
    workflow_name    str
    workflow_path    str
    workflow         str   full workflow YAML
    logs             any   raw CI logs (str | list)
    diff             str   ground-truth patch (memory_seed only; "" in eval)
    split_role       str   "memory" | "evaluation"
    overall_error_types  list[str]
    fix_type         str
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# ── Helpers ───────────────────────────────────────────────────────────────────

def _jsonable(val: Any) -> Any:
    """Coerce numpy / non-JSON-serialisable types recursively."""
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, dict):
        return {str(k): _jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_jsonable(v) for v in val]
    try:
        import numpy as np  # type: ignore
        if isinstance(val, np.ndarray):
            return [_jsonable(v) for v in val.tolist()]
        if isinstance(val, np.generic):
            return val.item()
    except Exception:
        pass
    return str(val)


def _load_split(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_record(split_row: Dict, pq_row: Dict, role: str) -> Dict:
    sha = str(split_row.get("sha_fail") or "")

    # logs: keep as-is (str | list) — cibench/ci_log_analyzer handles both
    logs_raw = pq_row.get("logs")
    if logs_raw is None:
        logs_raw = ""
    else:
        logs_raw = _jsonable(logs_raw)

    # workflow YAML
    workflow = str(pq_row.get("workflow") or "")

    # ground-truth diff (only for memory set; empty string for eval)
    diff = str(pq_row.get("diff") or "") if role == "memory" else ""

    return {
        # ── Core identifiers ──────────────────────────────────────────────
        "instance_id":   sha,   # cibench uses this as the unique key
        "id":            str(pq_row.get("id") or split_row.get("id") or sha),
        "sha_fail":      sha,
        # ── Repo ──────────────────────────────────────────────────────────
        "repo_owner":    str(pq_row.get("repo_owner") or split_row.get("repo_owner") or ""),
        "repo_name":     str(pq_row.get("repo_name")  or split_row.get("repo_name")  or ""),
        # ── Workflow ──────────────────────────────────────────────────────
        "workflow_name": str(pq_row.get("workflow_name")  or split_row.get("workflow_name")  or ""),
        "workflow_path": str(pq_row.get("workflow_path")  or split_row.get("workflow_path")  or ""),
        "workflow":      workflow,
        # ── Logs ──────────────────────────────────────────────────────────
        "logs":          logs_raw,
        # ── Ground truth diff (memory only) ───────────────────────────────
        "diff":          diff,
        # ── Split metadata ────────────────────────────────────────────────
        "split_role":        role,
        "overall_error_types": _jsonable(
            split_row.get("overall_error_types")
            or [str(pq_row.get("error_type") or "")]
        ),
        "fix_type":      str(split_row.get("fix_type") or ""),
        "recurrence_count": split_row.get("recurrence_count"),
        "avg_similarity":   split_row.get("avg_similarity"),
    }


# ── Per-repo summary table ─────────────────────────────────────────────────────

def _print_table(records: List[Dict], label: str) -> None:
    from collections import defaultdict, Counter
    by_repo: Dict[str, List] = defaultdict(list)
    for r in records:
        repo = f"{r.get('repo_owner','')}/{r.get('repo_name','')}"
        by_repo[repo].append(r)

    W = 44
    sep = "-" * (W + 50)
    print(f"\n{'=' * len(sep)}")
    print(f"  {label}  ({len(records)} issues)")
    print(f"{'=' * len(sep)}")
    print(f"{'Repo':<{W}}  {'#':>4}  {'Error Types'}")
    print(sep)
    for repo, issues in sorted(by_repo.items(), key=lambda x: -len(x[1])):
        ct: Counter = Counter()
        for iss in issues:
            for et in (iss.get("overall_error_types") or []):
                ct[str(et).strip()] += 1
        et_str = ", ".join(f"{e}({n})" if n > 1 else e for e, n in ct.most_common(3))
        print(f"{repo:<{W}}  {len(issues):>4}  {et_str}")
    print(sep)
    print(f"{'TOTAL':<{W}}  {len(records):>4}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    trs_dir: Path,
    parquet_path: Path,
    out_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    import pandas as pd  # type: ignore

    # ── Load split ─────────────────────────────────────────────────────────────
    memory_split = _load_split(trs_dir / "memory_issues.json")
    eval_split   = _load_split(trs_dir / "eval_issues.json")
    all_shas = {str(r["sha_fail"]) for r in memory_split + eval_split}

    print(f"[prepare] TRS split  → memory: {len(memory_split)}  eval: {len(eval_split)}")

    # ── Load parquet ───────────────────────────────────────────────────────────
    print(f"[prepare] Loading parquet: {parquet_path}")
    df = pd.read_parquet(str(parquet_path))
    df["sha_fail"] = df["sha_fail"].astype(str)
    matched = df[df["sha_fail"].isin(all_shas)]
    pq_map: Dict[str, Any] = {
        row["sha_fail"]: row
        for row in matched.to_dict(orient="records")
    }
    print(f"[prepare] Parquet matched {len(pq_map)} / {len(all_shas)} SHAs")

    missing = all_shas - pq_map.keys()
    if missing:
        print(f"[prepare] WARNING: {len(missing)} SHAs not found in parquet:")
        for sha in sorted(missing)[:10]:
            print(f"           {sha[:20]}…")

    # ── Build records ──────────────────────────────────────────────────────────
    memory_records: List[Dict] = []
    for row in memory_split:
        sha = str(row.get("sha_fail") or "")
        pq  = pq_map.get(sha, {})
        memory_records.append(_build_record(row, pq, "memory"))

    eval_records: List[Dict] = []
    for row in eval_split:
        sha = str(row.get("sha_fail") or "")
        pq  = pq_map.get(sha, {})
        eval_records.append(_build_record(row, pq, "evaluation"))

    # ── Print tables ───────────────────────────────────────────────────────────
    _print_table(memory_records, "MEMORY SEED ISSUES")
    _print_table(eval_records,   "EVAL ISSUES")

    if dry_run:
        print("[prepare] Dry run — no files written.")
        return

    # ── Write outputs ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_path   = out_dir / "eval_dataset.jsonl"
    memory_path = out_dir / "memory_seed.jsonl"

    with eval_path.open("w", encoding="utf-8") as fh:
        for rec in eval_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with memory_path.open("w", encoding="utf-8") as fh:
        for rec in memory_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[prepare] Written:")
    print(f"  {eval_path}    ({len(eval_records)} eval issues)")
    print(f"  {memory_path}  ({len(memory_records)} memory-seed issues)")
    print()
    print("Next steps:")
    print("  1. Seed the shared memory bank (run ONCE, before any agent):")
    print("       python scripts/seed_memory.py")
    print()
    print("  2. Run mini-swe-agent evaluation:")
    print("       scripts/run_cibench_minimax_openrouter.sh")
    print("     (uses data/eval_dataset.jsonl and results/shared_memory/ by default)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _default_trs_dir() -> Path:
    """Best-effort default: sibling CI-REPAIR-BENCH directory."""
    here = Path(__file__).resolve().parents[1]
    candidates = [
        here.parent / "CI-REPAIR-BENCH" / "results" / "trs_split",
        Path.home() / "Documents" / "CI-REPAIR-BENCH" / "results" / "trs_split",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def _default_parquet() -> Path:
    here = Path(__file__).resolve().parents[1]
    candidates = [
        here.parent / "CI-REPAIR-BENCH" / "dataset" / "lca_dataset.parquet",
        Path.home() / "Documents" / "CI-REPAIR-BENCH" / "dataset" / "lca_dataset.parquet",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Join TRS split with parquet to produce shared eval + memory-seed datasets."
    )
    p.add_argument(
        "--trs-dir",
        type=Path,
        default=_default_trs_dir(),
        help="Directory containing memory_issues.json and eval_issues.json (TRS split output)",
    )
    p.add_argument(
        "--parquet",
        type=Path,
        default=_default_parquet(),
        help="Path to lca_dataset.parquet",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tables but do not write any files",
    )
    args = p.parse_args()

    if not args.trs_dir.is_dir():
        print(f"ERROR: TRS split directory not found: {args.trs_dir}", file=sys.stderr)
        print("Run temporal_recurrence_split.py first, or pass --trs-dir explicitly.", file=sys.stderr)
        sys.exit(1)
    if not args.parquet.is_file():
        print(f"ERROR: Parquet not found: {args.parquet}", file=sys.stderr)
        sys.exit(1)

    run(args.trs_dir, args.parquet, args.out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
