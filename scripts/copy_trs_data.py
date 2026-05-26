#!/usr/bin/env python3
"""
copy_trs_data.py
================
Copy the pre-computed TRS data from CI-REPAIR-BENCH into this project.

Produces:
    data/eval_dataset.jsonl          — 189 eval issues (with full logs from parquet)
    results/shared_memory/           — L1/L2/L3 memory bank (direct copy)

Usage:
    python scripts/copy_trs_data.py

    # Explicit paths
    python scripts/copy_trs_data.py \\
        --trs-results /path/to/CI-REPAIR-BENCH/baselines/results/trs \\
        --parquet     /path/to/CI-REPAIR-BENCH/dataset/lca_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List


# ── Helpers ───────────────────────────────────────────────────────────────────

def _jsonable(val: Any) -> Any:
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


def _default_trs_results() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        # Highest priority: data/trs/ inside this project (user-placed)
        project_root / "data" / "trs",
        # Sibling CI-REPAIR-BENCH repo
        project_root.parent / "CI-REPAIR-BENCH" / "baselines" / "results" / "trs",
        Path.home() / "Documents" / "CI-REPAIR-BENCH" / "baselines" / "results" / "trs",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def _default_parquet() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1].parent / "CI-REPAIR-BENCH" / "dataset" / "lca_dataset.parquet",
        Path.home() / "Documents" / "CI-REPAIR-BENCH" / "dataset" / "lca_dataset.parquet",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


# ── Main ───────────────────────────────────────────────────────────────────────

def run(trs_results: Path, parquet_path: Path, project_root: Path) -> None:
    import pandas as pd  # type: ignore

    # ── 1. Load enriched eval issues ───────────────────────────────────────────
    eval_json = trs_results / "trs_eval_issues.json"
    issues: List[Dict] = json.loads(eval_json.read_text(encoding="utf-8"))
    print(f"[copy] Loaded {len(issues)} eval issues from {eval_json}")

    # ── 2. Load parquet — only keep matching SHAs ──────────────────────────────
    print(f"[copy] Loading parquet: {parquet_path}")
    df = pd.read_parquet(str(parquet_path))
    df["sha_fail"] = df["sha_fail"].astype(str)
    sha_set = {str(iss["sha_fail"]) for iss in issues}
    matched = df[df["sha_fail"].isin(sha_set)]
    pq_map: Dict[str, Any] = {
        row["sha_fail"]: row
        for row in matched.to_dict(orient="records")
    }
    print(f"[copy] Parquet matched {len(pq_map)} / {len(sha_set)} SHAs")

    missing = sha_set - pq_map.keys()
    if missing:
        print(f"[copy] WARNING: {len(missing)} SHAs not in parquet — logs will be empty for them")

    # ── 3. Merge: add full logs + ensure instance_id ───────────────────────────
    enriched: List[Dict] = []
    for iss in issues:
        sha = str(iss["sha_fail"])
        pq  = pq_map.get(sha, {})

        # Raw logs from parquet (the full step logs the agent needs)
        logs_raw = pq.get("logs", [])
        logs = _jsonable(logs_raw) if logs_raw is not None else []

        record = {
            # cibench requires instance_id
            "instance_id": sha,
            # keep all pre-computed enriched fields from the TRS pipeline
            **_jsonable(iss),
            # add full logs (overrides any partial logs_summary)
            "logs": logs,
            # add workflow from parquet if missing in iss
            "workflow": str(iss.get("workflow") or pq.get("workflow") or ""),
        }
        enriched.append(record)

    # ── 4. Write eval JSONL ────────────────────────────────────────────────────
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    eval_out = data_dir / "eval_dataset.jsonl"

    with eval_out.open("w", encoding="utf-8") as fh:
        for rec in enriched:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[copy] Written: {eval_out}  ({len(enriched)} issues)")

    # ── 5. Verify a sample ─────────────────────────────────────────────────────
    sample = enriched[0]
    has_logs     = bool(sample.get("logs"))
    has_workflow = bool(sample.get("workflow"))
    has_reasons  = bool(sample.get("overall_failure_reasons"))
    has_files    = bool(sample.get("effected_files"))
    print(f"[copy] Sample check — logs:{has_logs}  workflow:{has_workflow}  "
          f"failure_reasons:{has_reasons}  effected_files:{has_files}")

    # ── 6. Copy memory bank ────────────────────────────────────────────────────
    mem_src = trs_results
    mem_dst = project_root / "results" / "shared_memory"
    mem_dst.mkdir(parents=True, exist_ok=True)

    memory_files = ["failure_memory.json", "repo_memory.json", "cross_memory.json"]
    for fname in memory_files:
        src = mem_src / fname
        dst = mem_dst / fname
        if src.is_file():
            shutil.copy2(str(src), str(dst))
            size = dst.stat().st_size // 1024
            # Count entries
            data = json.loads(dst.read_text(encoding="utf-8"))
            n = len(data) if isinstance(data, list) else len(data)
            print(f"[copy] Copied {fname}: {n} entries  ({size}KB)")
        else:
            print(f"[copy] WARNING: {src} not found — skipping")

    # ── 7. Summary ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Done — shared benchmark data is ready")
    print("=" * 60)
    print(f"  Eval issues  : {eval_out}  ({len(enriched)})")
    print(f"  Memory bank  : {mem_dst}/")
    for f in memory_files:
        p = mem_dst / f
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            print(f"    {f}: {len(d)} entries")
    print()
    print("Run baseline (no memory):")
    print("  scripts/run_cibench_minimax_openrouter.sh --ablation baseline")
    print()
    print("Then L1 / L1+L2 / L1+L2+L3:")
    print("  scripts/run_cibench_minimax_openrouter.sh --ablation L1")
    print("  scripts/run_cibench_minimax_openrouter.sh --ablation L1+L2")
    print("  scripts/run_cibench_minimax_openrouter.sh")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Copy TRS eval issues + memory bank from CI-REPAIR-BENCH into this project."
    )
    p.add_argument("--trs-results", type=Path, default=_default_trs_results(),
                   help="Path to CI-REPAIR-BENCH/baselines/results/trs/")
    p.add_argument("--parquet",     type=Path, default=_default_parquet(),
                   help="Path to lca_dataset.parquet")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if not args.trs_results.is_dir():
        print(f"ERROR: {args.trs_results} not found", file=sys.stderr)
        sys.exit(1)
    if not args.parquet.is_file():
        print(f"ERROR: {args.parquet} not found", file=sys.stderr)
        sys.exit(1)

    run(args.trs_results, args.parquet, project_root)


if __name__ == "__main__":
    main()
