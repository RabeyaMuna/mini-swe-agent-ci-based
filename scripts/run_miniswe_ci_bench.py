#!/usr/bin/env python3
"""
Thin wrapper to run Mini-SWE-Agent's CI bench with our dataset/memory layout.

This avoids relying on a CLI entrypoint in the miniswe-agent package by
importing the Typer command function directly and invoking it with options.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

from utilities.model_registry import resolve_model_alias, model_output_name

# Suppress Pydantic serialization warnings from LiteLLM/minimax responses
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
    module="pydantic.main"
)


def _ablation_to_miniswe(ablation: str) -> tuple[bool, str]:
    """Map our ablation string to Mini-SWE's memory flags.

    Returns: (memory_enabled, memory_ablation_levels)

    NOTE: For custom memory plugin integration:
    - memory_enabled=True always (plugin handles decomposition for all modes)
    - memory_ablation_levels="" for baseline (decompose_only, no retrieval)
    - memory_ablation_levels="L1", "L1+L2", or "L1+L2+L3" for memory modes
    """
    a = (ablation or "").strip().lower()
    if a == "baseline":
        return True, ""  # Empty levels = baseline mode (decompose only, no retrieval)
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

    # Build output directory: <output_root>/<ablation>_<model>/
    # NOTE: For baseline, output_root should be "results/miniswe-agent" (no direction subdir)
    #       For memory modes, output_root should be "results/miniswe-agent/<direction>"
    safe_ablation = args.ablation.replace("+", "_").lower()
    output_dir = Path(args.output_root) / f"{safe_ablation}_{model_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import and call the Typer command function directly
    from minisweagent.run.benchmarks.cibench import main as cibench_main  # type: ignore

    # Use the cibench.yaml config which includes agent templates
    config_path = Path(__file__).parent.parent / "miniswe-agent/src/minisweagent/config/benchmarks/cibench.yaml"

    cibench_main(
        dataset=str(Path(args.dataset)),
        split="train",  # not used for local JSONL
        output=str(output_dir),
        workers=args.workers,
        model_name=resolved_model,
        model_class=None,
        config_spec=[str(config_path)],  # Load cibench.yaml config
        filter_spec=args.issue_regex,
        slice_spec="",
        shuffle=False,
        redo_existing=False,
        memory_enabled=True,  # Always enable for custom plugin (even baseline uses decompose_only)
        memory_root=str(memory_root),
        memory_top_k=args.memory_top_k,
        memory_ablation_levels=memory_levels,
        memory_plugin_path="memory_plugin.memory_plugin:MemoryPlugin",  # Use custom plugin!
        save_memory=True,
        context_model=resolved_model,
        # NOTE: step_limit, cost_limit, wall_time_limit_seconds removed
        # These parameters are not supported in mini-swe-agent version 2.3.0
        # They were added for handling LimitsExceeded errors but require newer version
    )


if __name__ == "__main__":
    main()

