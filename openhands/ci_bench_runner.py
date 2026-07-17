#!/usr/bin/env python3
"""
CI-Bench Runner for OpenHands
Runs CI-Bench evaluation on OpenHands WITHOUT modifying OpenHands code.
Supports baseline and memory-guided (L1/L2/L3) modes.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse
from tqdm import tqdm

# Import our modules
from data_loader import CIBenchDataLoader
from memory_retriever import MemoryRetriever
from prompt_formatter import PromptFormatter

# Shared paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
REPO_ROOT = PROJECT_ROOT / "repo"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"


class SharedPaths:
    """Paths to shared resources."""
    
    @staticmethod
    def get_memory_root() -> Path:
        return DATA_ROOT / "trs"
    
    @staticmethod
    def get_results_dir(agent_name: str, model_name: str, ablation_level: str) -> Path:
        results_dir = RESULTS_ROOT / agent_name / model_name / ablation_level
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    @staticmethod
    def get_repo_dir(repo_identifier: str) -> Path:
        return REPO_ROOT / repo_identifier


def run_single_issue(
    issue: Dict[str, Any],
    log_details: Dict[str, Any],
    workflow_cache: Dict[str, Any],
    mode: str,
    memory_retriever: Optional[MemoryRetriever],
    model: str,
    results_dir: Path
) -> Dict[str, Any]:
    """
    Run OpenHands on a single issue

    Args:
        issue: Issue data from eval_set.jsonl
        log_details: CI logs cache
        workflow_cache: Workflow cache
        mode: "baseline" or "memory"
        memory_retriever: Memory retriever instance (for memory mode)
        model: Model name
        results_dir: Where to save results

    Returns:
        Result dictionary
    """
    # Get full issue data
    data_loader = CIBenchDataLoader(data_root="../data")
    issue_data = data_loader.get_issue_data(issue, log_details, workflow_cache)

    instance_id = issue_data["instance_id"]
    print(f"\nProcessing: {instance_id}")

    # Format prompt based on mode
    formatter = PromptFormatter()

    if mode == "baseline":
        openhands_task = formatter.format_baseline_task(issue_data)
    else:  # memory mode
        # Retrieve memory
        memory_context = memory_retriever.retrieve(
            instance_id=instance_id,
            problem_statement=issue_data["problem_statement"],
            repo=issue_data["repo"],
            top_k=3
        )

        # Format memory for prompt
        memory_prompt = memory_retriever.format_for_prompt(memory_context)

        # Create task with memory
        openhands_task = formatter.format_memory_task(issue_data, memory_prompt)

    # TODO: Actually run OpenHands here
    # For now, this is a placeholder
    print(f"  Repository: {openhands_task['repository']}")
    print(f"  Branch: {openhands_task['selected_branch']}")
    print(f"  Mode: {mode}")
    print(f"  ⚠️  OpenHands execution not yet implemented")

    # Placeholder result
    result = {
        "instance_id": instance_id,
        "model_name_or_path": model,
        "model_patch": "",  # Would contain the generated patch
        "agent": "openhands",
        "mode": mode,
        "status": "pending_implementation"
    }

    return result


def run_batch(
    eval_issues_path: str,
    mode: str,
    model: str,
    memory_layers: Optional[List[str]],
    output_dir: str,
    slice_range: Optional[str] = None
):
    """
    Run CI-Bench evaluation on OpenHands

    Args:
        eval_issues_path: Path to eval_set.jsonl
        mode: "baseline" or "memory"
        model: Model name (e.g., "glm-4-plus")
        memory_layers: List of memory layers (e.g., ["L1", "L2", "L3"])
        output_dir: Output directory
        slice_range: Optional slice (e.g., "0:5" for first 5 issues)
    """
    print(f"\n{'='*80}")
    print(f"  CI-Bench Evaluation on OpenHands")
    print(f"{'='*80}")
    print(f"Mode: {mode}")
    print(f"Model: {model}")
    print(f"Memory Layers: {memory_layers if mode == 'memory' else 'None (baseline)'}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Setup
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_loader = CIBenchDataLoader(data_root="../data")
    eval_issues = data_loader.load_eval_issues(eval_issues_path)

    # Apply slice if provided
    if slice_range:
        start, end = map(int, slice_range.split(':'))
        eval_issues = eval_issues[start:end]
        print(f"Using slice: {slice_range} ({len(eval_issues)} issues)\n")

    # Ensure data files exist
    log_details, workflow_cache = data_loader.ensure_data_files(eval_issues)

    # Setup memory if needed
    memory_retriever = None
    if mode == "memory":
        memory_retriever = MemoryRetriever(
            memory_root="../data/trs",
            layers=memory_layers or ["L1", "L2", "L3"]
        )

    # Run on all issues
    results = []
    for issue in tqdm(eval_issues, desc="Processing issues"):
        result = run_single_issue(
            issue=issue,
            log_details=log_details,
            workflow_cache=workflow_cache,
            mode=mode,
            memory_retriever=memory_retriever,
            model=model,
            results_dir=results_dir
        )
        results.append(result)

    # Save predictions in mini-swe-agent format
    preds_file = results_dir / "preds.json"
    with open(preds_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Completed {len(results)} issues")
    print(f"✓ Results saved to: {preds_file}")
    print(f"{'='*80}\n")

    # Summary
    print("Next steps:")
    print("1. Implement actual OpenHands agent execution")
    print("2. Replace placeholder results with real patches")
    print(f"3. Evaluate with: python ../scripts/evaluate_ablation_preds.py {preds_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenHands on CI-Bench evaluation set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline mode (no memory)
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode baseline --model glm-4-plus \\
      --output ../results/openhands/glm/baseline

  # Memory mode (L1+L2+L3)
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode memory --memory-layers L1 L2 L3 --model glm-4-plus \\
      --output ../results/openhands/glm/L1_L2_L3

  # Test on first 5 issues
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode baseline --model glm-4-plus --slice 0:5 \\
      --output ../results/openhands/glm/test
        """
    )
    parser.add_argument(
        "--eval-issues",
        required=True,
        help="Path to eval_set.jsonl"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "memory"],
        help="Mode: baseline (no memory) or memory (with L1/L2/L3)"
    )
    parser.add_argument(
        "--model",
        default="glm-4-plus",
        help="Model to use (e.g., glm-4-plus, minimax/minimax-m2.5)"
    )
    parser.add_argument(
        "--memory-layers",
        nargs="+",
        default=["L1", "L2", "L3"],
        choices=["L1", "L2", "L3"],
        help="Memory layers to use (only for memory mode)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--slice",
        help="Optional slice range (e.g., '0:5' for first 5 issues)"
    )

    args = parser.parse_args()

    # Run batch
    run_batch(
        eval_issues_path=args.eval_issues,
        mode=args.mode,
        model=args.model,
        memory_layers=args.memory_layers if args.mode == "memory" else None,
        output_dir=args.output,
        slice_range=args.slice
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
