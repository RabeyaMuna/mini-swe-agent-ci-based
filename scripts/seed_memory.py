#!/usr/bin/env python3
"""
seed_memory.py - Runtime Memory Generation
==========================================
Pre-populate the shared L1/L2/L3 memory bank from data/memory_seed.jsonl.

⚠️  WARNING: For high-quality memory suitable for evaluation, use
    build_memory_bank.py instead! This script generates memory at runtime
    with simpler structure (strings instead of 4-field objects).

USE THIS SCRIPT FOR:
    ✅ Quick experiments and prototyping
    ✅ Runtime memory generation during agent runs
    ✅ Fast memory creation (no LLM chain-of-thought)

USE build_memory_bank.py FOR:
    ✅ Evaluation and paper results (proper L2 structure)
    ✅ Two-stage L2 construction (identifies distinct CI failures)
    ✅ 4-field changed_files: [{file, reason, failure_reason, fix_strategy}]
    ✅ Chain-of-thought analysis with cross-failure dependencies

See MEMORY_BUILDING_GUIDE.md for details.

How it works
------------
For each issue in memory_seed.jsonl:
  1. Run CILogAnalyzer on the CI logs           → structured log_analysis_result
  2. Call CIMemorySystem.save(log_analysis_result, diff)
     → appends entries to:
         <memory-root>/failure_memory.json   (L1 per-file)
         <memory-root>/repo_memory.json      (L2 per-repo)
         <memory-root>/cross_memory.json     (L3 cross-repo)

Usage
-----
    # Use default paths + model from env
    python scripts/seed_memory.py

    # Explicit options (generates runtime memory, not recommended for eval)
    python scripts/seed_memory.py \\
        --memory-seed  data/memory_seed.jsonl \\
        --memory-root  results/runtime_memory \\
        --model        minimax/minimax-m2.5 \\
        --workers      4

    # Dry-run: print what would be saved, write nothing
    python scripts/seed_memory.py --dry-run

    # Skip LLM log analysis (fast mode) — build minimal L1/L2/L3 from
    # known error_types and fix_type.  Faster but less rich memory.
    python scripts/seed_memory.py --no-llm-analysis

Environment variables
---------------------
    OPENROUTER_API_KEY   OR   MINIMAX_API_KEY   — API key for the LLM
    MEMCI_LLM_MODEL      — override --model (same name as run script)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_memory")

_LOCK = threading.Lock()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"
if PROJECT_ENV_FILE.is_file():
    load_dotenv(dotenv_path=PROJECT_ENV_FILE, override=False)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_llm(model: str, api_key: str) -> Any:
    """
    Build a LiteLLM-based LLM object for CIMemorySystem / CILogAnalyzer.

    Falls back to a simple callable wrapper that calls litellm.completion().
    """
    try:
        import litellm  # type: ignore

        def _call(prompt: str) -> str:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""

        return _call
    except ImportError:
        pass

    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://openrouter.ai/api/v1")
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0,
        )
    except ImportError:
        pass

    raise RuntimeError(
        "Neither litellm nor langchain_openai is installed. "
        "Run: pip install litellm  OR  pip install langchain-openai"
    )


# ── Minimal log analysis (no LLM) ────────────────────────────────────────────

def _minimal_analysis(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a log_analysis_result without calling any LLM.
    Uses the already-known fields from memory_seed.jsonl.
    """
    sha_fail = str(instance.get("sha_fail") or "")
    task_id  = str(instance.get("instance_id") or instance.get("id") or sha_fail)

    error_types = instance.get("overall_error_types") or []
    # Wrap plain strings into the {category, subcategory, evidence} schema
    error_types_structured = [
        {"category": str(et), "subcategory": "", "evidence": ""}
        for et in error_types
        if str(et).strip()
    ]

    # Construct effected_files from ground-truth diff changed files
    diff = str(instance.get("diff") or "")
    import re
    gt_files = re.findall(r"diff --git a/.+ b/(.+)$", diff, re.MULTILINE)
    relevant_files = [
        {
            "file": f.strip(),
            "line_number": None,
            "reason": f"changed in ground-truth fix ({instance.get('fix_type','unknown_fix')})",
            "issue_type": error_types[0] if error_types else "",
            "failed_cmd": None,
            "failed_tool": None,
        }
        for f in gt_files
        if f.strip()
    ]

    return {
        "sha_fail":       sha_fail,
        "id":             task_id,
        "error_context":  [
            f"fix_type={instance.get('fix_type','')} "
            f"error_types={', '.join(error_types)}"
        ],
        "relevant_files": relevant_files,
        "error_types":    error_types_structured,
        "failed_job":     [],
        "overall_ci_summary": f"memory seed — {instance.get('fix_type','unknown_fix')}",
        "_fallback": True,
    }


# ── Per-issue seeding ─────────────────────────────────────────────────────────

def _seed_one(
    instance: Dict[str, Any],
    memory_system: Any,
    llm: Any,
    *,
    use_llm_analysis: bool,
    dry_run: bool,
) -> str:
    """
    Seed one memory issue.  Returns a status string.
    """
    sha      = str(instance.get("sha_fail") or "")
    task_id  = str(instance.get("instance_id") or instance.get("id") or sha)
    diff     = str(instance.get("diff") or "")

    if not diff.strip():
        return f"SKIP  {sha[:16]}  (no diff)"

    # Step 1: log analysis
    if use_llm_analysis and llm is not None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from minisweagent.run.benchmarks.utils.ci_log_analyzer import CILogAnalyzer
            analyzer = CILogAnalyzer(
                logs=instance.get("logs") or "",
                sha_fail=sha,
                workflow=instance.get("workflow") or "",
                workflow_path=instance.get("workflow_path") or "",
                llm=llm,
                model_name=str(instance.get("model_name") or ""),
                task_id=task_id,
            )
            log_analysis = analyzer.run()
        except Exception as exc:
            logger.warning("[seed] LLM analysis failed for %s (%s) — using minimal", sha[:12], exc)
            log_analysis = _minimal_analysis(instance)
    else:
        log_analysis = _minimal_analysis(instance)

    if dry_run:
        return f"DRY   {sha[:16]}  files={len(log_analysis.get('relevant_files') or [])}"

    # Step 2: save to memory bank
    try:
        memory_system.save(
            task_id=task_id,
            sha_fail=sha,
            repo_name=str(instance.get("repo_name") or ""),
            repo_owner=str(instance.get("repo_owner") or ""),
            workflow_path=str(instance.get("workflow_path") or ""),
            workflow=str(instance.get("workflow") or ""),
            log_analysis_result=log_analysis,
            diff=diff,
            llm=llm,
        )
        return f"OK    {sha[:16]}  repo={instance.get('repo_name','')} files={len(log_analysis.get('relevant_files') or [])}"
    except Exception as exc:
        return f"ERR   {sha[:16]}  {exc}"


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    memory_seed_path: Path,
    memory_root: Path,
    model: str,
    *,
    workers: int = 1,
    dry_run: bool = False,
    use_llm_analysis: bool = True,
) -> None:
    # ── Load issues ────────────────────────────────────────────────────────────
    instances: List[Dict[str, Any]] = []
    with memory_seed_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    instances.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed line: %s", e)

    with_diff = [i for i in instances if str(i.get("diff") or "").strip()]
    logger.info("[seed] Loaded %d issues (%d have a diff)", len(instances), len(with_diff))

    if not with_diff:
        logger.error("[seed] No issues with a diff — nothing to seed. "
                     "Run prepare_shared_dataset.py first.")
        sys.exit(1)

    # ── LLM setup ──────────────────────────────────────────────────────────────
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("MINIMAX_API_KEY")
        or ""
    )

    llm: Any = None
    if use_llm_analysis:
        if not api_key:
            logger.warning("[seed] No API key found (OPENROUTER_API_KEY / MINIMAX_API_KEY). "
                           "Falling back to --no-llm-analysis mode.")
            use_llm_analysis = False
        else:
            try:
                llm = _build_llm(model, api_key)
                logger.info("[seed] LLM ready: %s", model)
            except Exception as exc:
                logger.warning("[seed] Could not build LLM (%s). Using minimal analysis.", exc)
                use_llm_analysis = False

    if not use_llm_analysis:
        logger.info("[seed] Using minimal (no-LLM) log analysis.")

    # ── Memory system ──────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from minisweagent.run.benchmarks.utils.ci_memory_system import CIMemorySystem

    if dry_run:
        logger.info("[seed] DRY RUN — memory bank will not be written.")
        memory_system = None
    else:
        memory_root.mkdir(parents=True, exist_ok=True)
        memory_system = CIMemorySystem.create(
            str(memory_root),
            memory_enabled=True,
            memory_top_k=3,
            memory_ablation_levels="L1+L2+L3",
            llm=llm,
        )
        logger.info("[seed] Memory bank: %s", memory_root)

    # ── Seed ───────────────────────────────────────────────────────────────────
    logger.info("[seed] Seeding %d issues  workers=%d …", len(with_diff), workers)

    results: List[str] = []
    ok_count    = 0
    skip_count  = 0
    err_count   = 0

    def _process(instance: Dict[str, Any]) -> str:
        return _seed_one(
            instance,
            memory_system,
            llm,
            use_llm_analysis=use_llm_analysis,
            dry_run=dry_run,
        )

    if workers == 1:
        for i, inst in enumerate(with_diff, 1):
            status = _process(inst)
            results.append(status)
            if status.startswith("OK"):
                ok_count += 1
            elif status.startswith("SKIP") or status.startswith("DRY"):
                skip_count += 1
            else:
                err_count += 1
            if i % 10 == 0 or i == len(with_diff):
                logger.info("[seed] %d/%d  ok=%d skip=%d err=%d",
                            i, len(with_diff), ok_count, skip_count, err_count)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(_process, inst): inst for inst in with_diff}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                done += 1
                try:
                    status = fut.result()
                except Exception as exc:
                    status = f"ERR   {futures[fut].get('sha_fail','?')[:16]}  {exc}"
                results.append(status)
                if status.startswith("OK"):
                    ok_count += 1
                elif status.startswith("SKIP") or status.startswith("DRY"):
                    skip_count += 1
                else:
                    err_count += 1
                if done % 10 == 0 or done == len(with_diff):
                    logger.info("[seed] %d/%d  ok=%d skip=%d err=%d",
                                done, len(with_diff), ok_count, skip_count, err_count)

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SEED SUMMARY")
    print("=" * 60)
    print(f"  Total issues processed : {len(with_diff)}")
    if dry_run:
        print(f"  Dry-run (no writes)   : {skip_count + ok_count}")
    else:
        print(f"  Saved to memory bank  : {ok_count}")
        print(f"  Skipped (no diff)     : {skip_count}")
        print(f"  Errors                : {err_count}")
        if ok_count:
            print()
            print(f"  Memory bank location  : {memory_root}/")
            print(f"    failure_memory.json   (L1 per-file records)")
            print(f"    repo_memory.json      (L2 per-repo patterns)")
            print(f"    cross_memory.json     (L3 cross-repo principles)")
    print("=" * 60)

    if err_count:
        # Print first few errors for inspection
        print("\nFirst errors:")
        for r in [s for s in results if s.startswith("ERR")][:5]:
            print(" ", r)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Seed the shared L1/L2/L3 memory bank from data/memory_seed.jsonl."
    )
    p.add_argument(
        "--memory-seed",
        type=Path,
        default=Path("data/memory_seed.jsonl"),
        help="Path to memory_seed.jsonl (default: data/memory_seed.jsonl)",
    )
    p.add_argument(
        "--memory-root",
        type=Path,
        default=Path("results/runtime_memory"),
        help="Directory for the shared L1/L2/L3 memory bank (default: results/runtime_memory)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MEMCI_LLM_MODEL", "minimax/minimax-m2.5"),
        help="LLM model for log analysis (default: minimax/minimax-m2.5)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default: 1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be saved without writing any files",
    )
    p.add_argument(
        "--no-llm-analysis",
        dest="use_llm_analysis",
        action="store_false",
        help=(
            "Skip LLM log analysis. Uses fast heuristic analysis "
            "from known error_types and fix_type instead. "
            "Memory entries are less rich but seeding is instant."
        ),
    )
    args = p.parse_args()

    if not args.memory_seed.is_file():
        print(f"ERROR: {args.memory_seed} not found.", file=sys.stderr)
        print("Run: python scripts/prepare_shared_dataset.py  first.", file=sys.stderr)
        sys.exit(1)

    run(
        memory_seed_path=args.memory_seed,
        memory_root=args.memory_root,
        model=args.model,
        workers=args.workers,
        dry_run=args.dry_run,
        use_llm_analysis=args.use_llm_analysis,
    )


if __name__ == "__main__":
    main()
