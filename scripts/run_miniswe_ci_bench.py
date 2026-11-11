#!/usr/bin/env python3
"""
Thin wrapper to run Mini-SWE-Agent's CI bench with our dataset/memory layout.

This avoids relying on a CLI entrypoint in the miniswe-agent package by
importing the Typer command function directly and invoking it with options.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities.model_registry import model_output_name, resolve_model_alias  # noqa: E402

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
    - a single level ("L1", "L2", or "L3") loads only that level
    - cumulative modes use "L1+L2" or "L1+L2+L3"
    """
    a = (ablation or "").strip().lower().replace("_", "+").replace(" ", "")
    if a == "baseline":
        return True, ""  # Empty levels = baseline mode (decompose only, no retrieval)
    if a in {"l1", "l2", "l3", "l1+l2", "l1+l2+l3"}:
        return True, a.upper()
    raise ValueError(
        f"Unsupported ablation {ablation!r}; expected baseline, L1, L2, L3, "
        "L1+L2, or L1+L2+L3"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(Path("data/eval_set.jsonl")), help="Path to JSONL dataset")
    p.add_argument("--issue_regex", default="", help="Regex over instance_id to filter (e.g., ^(129|130)$)")
    p.add_argument("--ablation", default="baseline")
    p.add_argument(
        "--direction",
        default="none",
        choices=[
            "none",
            "backward",
            "forward",
            "bidirectional",
            "cross_repo",
            "whole_history",
        ],
        help=(
            "Memory direction. Use 'none' for baseline; backward, forward, and "
            "bidirectional use the standard memories; cross_repo uses the "
            "repository-disjoint memory; whole_history uses the one-record-per-PR control."
        ),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output_root", default=str(Path("results/miniswe-agent")))
    p.add_argument("--memory_top_k", type=int, default=5)
    p.add_argument("--slice", default="", help="Slice notation for dataset (e.g., '0:150' for first 150 instances)")
    p.add_argument("--reverse", action="store_true", help="Process instances in reverse order (last to first)")
    # Component ablation flags for reviewer experiments
    p.add_argument("--disable-dependency-expansion", action="store_true",
                   help="Skip STAGE 4 dependency extraction (for component ablation)")
    p.add_argument("--disable-recurring-recall", action="store_true",
                   help="Skip STAGE 5 common patterns detection (for component ablation)")
    p.add_argument("--disable-filtering", action="store_true",
                   help="Skip STAGE 2 applicability filtering (for component ablation)")
    p.add_argument("--disable-ordering", action="store_true",
                   help="Skip STAGE 6 dependency-based reordering (for component ablation)")
    p.add_argument("--disable-deduplication", action="store_true",
                   help="Skip STAGE 5.5 semantic clustering and duplicate merging")
    p.add_argument("--disable-memory-optimization", action="store_true",
                   help="Pass all retrieved L1/L2/L3 matches to enrichment without cross-level pruning")
    p.add_argument("--minimal-retrieval", action="store_true",
                   help="Minimal mode: only retrieve and enrich; skip filtering, dependency expansion, recurring recall, deduplication/clustering, ordering, and cross-level memory optimization.")
    args = p.parse_args()

    # If --minimal-retrieval is set, enable all disable flags
    if args.minimal_retrieval:
        args.disable_dependency_expansion = True
        args.disable_recurring_recall = True
        args.disable_filtering = True
        args.disable_ordering = True
        args.disable_deduplication = True
        args.disable_memory_optimization = True

    # Resolve and slug the model name for stable output paths
    resolved_model = resolve_model_alias(args.model) or args.model
    model_slug = model_output_name(resolved_model)

    try:
        memory_enabled, memory_levels = _ablation_to_miniswe(args.ablation)
    except ValueError as exc:
        p.error(str(exc))
    is_baseline = args.ablation.strip().lower() == "baseline"
    direction = "none" if is_baseline else args.direction
    if not is_baseline and direction == "none":
        p.error(
            "memory ablations require backward, forward, bidirectional, or "
            "cross_repo, or whole_history direction"
        )
    memory_roots = {
        "none": "data/back_trs",  # Unused by baseline's decompose-only mode.
        "backward": "data/back_trs",
        "forward": "data/fwr_trs",
        "bidirectional": "data/bidirect_trs",
        "cross_repo": "experimental_data/bidirect_trs",
        "whole_history": "experimental_data/whole_repair_history/whole_pr_trs",
    }
    memory_root = Path(memory_roots[direction]).resolve()

    # Build output directory: <output_root>/<ablation>_<model>/
    # NOTE: For baseline, output_root should be "results/miniswe-agent" (no direction subdir)
    #       For memory modes, output_root should be "results/miniswe-agent/<direction>"
    safe_ablation = args.ablation.replace("+", "_").lower()
    output_dir = Path(args.output_root) / f"{safe_ablation}_{model_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle reverse order if requested
    dataset_path = Path(args.dataset)
    if args.reverse:
        import json
        import tempfile

        print(f"📖 Loading dataset to reverse order: {dataset_path}")
        instances = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    instances.append(json.loads(line))

        print(f"🔄 Reversing {len(instances)} instances (processing last to first)")
        instances.reverse()

        # Write reversed dataset to temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jsonl', prefix='reversed_dataset_')
        with open(temp_fd, 'w') as f:
            for inst in instances:
                f.write(json.dumps(inst) + '\n')

        dataset_path = Path(temp_path)
        print(f"📝 Reversed dataset written to: {dataset_path}")

    # Set component ablation flags as environment variables
    # These will be read by MemoryPlugin to control which stages to skip
    import os
    os.environ["RECAP_ENABLE_DEPENDENCY_EXPANSION"] = "0" if args.disable_dependency_expansion else "1"
    os.environ["RECAP_ENABLE_RECURRING_RECALL"] = "0" if args.disable_recurring_recall else "1"
    os.environ["RECAP_ENABLE_FILTERING"] = "0" if args.disable_filtering else "1"
    os.environ["RECAP_ENABLE_ORDERING"] = "0" if args.disable_ordering else "1"
    os.environ["RECAP_ENABLE_DEDUPLICATION"] = "0" if args.disable_deduplication else "1"
    os.environ["RECAP_ENABLE_MEMORY_OPTIMIZATION"] = "0" if args.disable_memory_optimization else "1"

    # Import and call the Typer command function directly
    from minisweagent.run.benchmarks.cibench import main as cibench_main  # type: ignore

    # Use the cibench.yaml config which includes agent templates
    config_path = Path(__file__).parent.parent / "miniswe-agent/src/minisweagent/config/benchmarks/cibench.yaml"

    cibench_main(
        dataset=str(dataset_path),
        split="train",  # not used for local JSONL
        output=str(output_dir),
        workers=args.workers,
        model_name=resolved_model,
        model_class=None,
        config_spec=[str(config_path)],  # Load cibench.yaml config
        filter_spec=args.issue_regex,
        slice_spec=args.slice,  # Use slice from command line (e.g., "0:150")
        shuffle=False,
        redo_existing=False,
        memory_enabled=True,  # Always enable for custom plugin (even baseline uses decompose_only)
        memory_root=str(memory_root),
        memory_top_k=args.memory_top_k,
        memory_ablation_levels=memory_levels,
        memory_plugin_path="memory_plugin.memory_plugin:MemoryPlugin",  # Use custom plugin!
        save_memory=True,
        context_model=resolved_model,
        direction=direction,
        # NOTE: step_limit, cost_limit, wall_time_limit_seconds removed
        # These parameters are not supported in mini-swe-agent version 2.3.0
        # They were added for handling LimitsExceeded errors but require newer version
    )


if __name__ == "__main__":
    main()
