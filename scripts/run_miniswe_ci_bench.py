#!/usr/bin/env python3
"""
Thin wrapper to run Mini-SWE-Agent's CI bench with our dataset/memory layout.

This avoids relying on a CLI entrypoint in the miniswe-agent package by
importing the Typer command function directly and invoking it with options.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from utilities.model_registry import resolve_model_alias, model_output_name


def _ablation_to_miniswe(ablation: str) -> tuple[bool, str]:
    """Map our ablation string to Mini-SWE's memory flags.

    Returns: (memory_enabled, memory_ablation_levels)
    """
    a = (ablation or "").strip().lower()
    if a == "baseline":
        return False, "L1+L2+L3"  # value unused when disabled
    if a in {"l1", "l1+ l2", "l1+l2", "l1_l2", "l1+l2+l3", "l1_l2_l3"}:
        # normalize with pluses
        a = a.replace("_", "+").replace(" ", "")
        return True, a.upper()
    if a == "l2":
        # Mini-SWE does not support pure L2; use L1+L2 as the closest
        return True, "L1+L2"
    if a == "l3":
        # Mini-SWE does not support pure L3; use full memory
        return True, "L1+L2+L3"
    # Default to full memory
    return True, "L1+L2+L3"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(Path("data/eval_set.jsonl")), help="Path to JSONL dataset")
    p.add_argument("--issue_regex", default="", help="Regex over instance_id to filter (e.g., ^(129|130)$)")
    p.add_argument("--ablation", default="baseline")
    p.add_argument("--direction", default="backward", choices=["backward", "forward"]) 
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output_root", default=str(Path("results/miniswe-agent")))
    p.add_argument("--memory_top_k", type=int, default=5)
    args = p.parse_args()

    # Resolve and slug the model name for stable output paths
    resolved_model = resolve_model_alias(args.model) or args.model
    model_slug = model_output_name(resolved_model)

    memory_enabled, memory_levels = _ablation_to_miniswe(args.ablation)
    memory_root = Path("data/back_trs" if args.direction == "backward" else "data/fwr_trs").resolve()

    # Build output directory: results/miniswe-agent/<ablation>_<model>/
    safe_ablation = args.ablation.replace("+", "_").lower()
    output_dir = Path(args.output_root) / f"{safe_ablation}_{model_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import and call the Typer command function directly
    from minisweagent.run.benchmarks.cibench import main as cibench_main  # type: ignore

    cibench_main(
        dataset=str(Path(args.dataset)),
        split="train",  # not used for local JSONL
        output=str(output_dir),
        workers=args.workers,
        model_name=resolved_model,
        model_class=None,
        config_spec=[],
        filter_spec=args.issue_regex,
        slice_spec="",
        shuffle=False,
        redo_existing=False,
        memory_enabled=memory_enabled,
        memory_root=str(memory_root) if memory_enabled else None,
        memory_top_k=args.memory_top_k,
        memory_ablation_levels=memory_levels,
        memory_plugin_path=None,
        save_memory=True,
        context_model=resolved_model,
    )


if __name__ == "__main__":
    main()

