#!/usr/bin/env python3
"""
CI-Bench Runner for OpenHands
Runs CI-Bench evaluation on OpenHands WITHOUT modifying OpenHands code.
Supports baseline and memory-guided (L1/L2/L3) modes.

Uses SHARED memory plugin for consistency with mini-swe-agent.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

# Add parent directory to path for shared benchmark packages.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from data_loader import CIBenchDataLoader
from prompt_formatter import PromptFormatter
from scripts.model_registry import configure_model_environment, resolve_model_alias

# Shared paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
RESULTS_ROOT = PROJECT_ROOT / 'results'
REPO_ROOT = PROJECT_ROOT / 'repo'
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'


class SharedPaths:
    """Paths to shared resources."""

    @staticmethod
    def get_memory_root() -> Path:
        return DATA_ROOT / 'trs'

    @staticmethod
    def get_results_dir(agent_name: str, model_name: str, ablation_level: str) -> Path:
        results_dir = RESULTS_ROOT / agent_name / model_name / ablation_level
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    @staticmethod
    def get_repo_dir(repo_identifier: str) -> Path:
        return REPO_ROOT / repo_identifier


def _make_analyzer_llm(model: str):
    """Build the callable used by shared CI log/workflow analyzers."""
    try:
        import litellm
    except ImportError:
        return None

    analyzer_model = resolve_model_alias(model) or model
    if (
        analyzer_model.startswith('minimax/')
        and os.getenv('OPENROUTER_API_KEY')
        and (
            'openrouter.ai' in os.getenv('OPENROUTER_BASE_URL', '')
            or 'openrouter.ai' in os.getenv('MINIMAX_BASE_URL', '')
        )
    ):
        analyzer_model = f'openrouter/{analyzer_model}'

    def _call(prompt: str) -> str:
        response = litellm.completion(
            model=analyzer_model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
        )
        return str(response.choices[0].message.content or '').strip()

    return _call


class OpenHandsMemoryAdapter:
    """Small OpenHands-facing adapter over the shared root memory plugin."""

    def __init__(
        self,
        memory_root: str,
        layers: Optional[list[str]] = None,
        llm: Any = None,
    ):
        self.memory_root = memory_root
        self.layers = layers or []
        self.llm = llm
        self._memory_system = None

        if self.layers:
            from memory_plugin.ci_memory_system import CIMemorySystem

            self._memory_system = CIMemorySystem.create(
                memory_root,
                memory_enabled=True,
                memory_ablation_levels='+'.join(self.layers),
                llm=llm,
            )

    def get_problem(self, issue_data: dict[str, Any]) -> dict[str, Any]:
        problem = issue_data.get('problem_statement', '')
        if not self._memory_system:
            return {'problem': problem, 'repair_plan': None}

        from memory_plugin.ci_memory_system import format_memory_context

        repo_owner, _, repo_name = issue_data.get('repo', '').partition('/')
        instance = {
            **issue_data,
            'repo_owner': repo_owner,
            'repo_name': repo_name,
        }
        workflow = issue_data.get('workflow') or {}
        memory = self._memory_system.build_and_retrieve(
            issue_data.get('ci_failure') or {},
            instance=instance,
            llm=self.llm,
            validation_sequence=workflow.get('validation_sequence') or [],
        )
        repair_plan = format_memory_context(memory) or None
        return {'problem': problem, 'repair_plan': repair_plan}


def run_single_issue(
    issue: dict[str, Any],
    log_details: dict[str, Any],
    workflow_cache: dict[str, Any],
    memory_engine: OpenHandsMemoryAdapter,
    model: str,
    results_dir: Path,
) -> dict[str, Any]:
    """
    Run OpenHands on a single issue using SHARED memory plugin

    Args:
        issue: Issue data from eval_set.jsonl
        log_details: CI logs cache
        workflow_cache: Workflow cache
        memory_engine: Shared memory adapter instance
        model: Model name
        results_dir: Where to save results

    Returns:
        Result dictionary
    """
    # Get full issue data
    data_loader = CIBenchDataLoader(data_root=str(DATA_ROOT))
    issue_data = data_loader.get_issue_data(issue, log_details, workflow_cache)

    instance_id = issue_data['instance_id']
    print(f'\nProcessing: {instance_id}')

    # Use shared memory plugin to get problem + repair plan
    memory_result = memory_engine.get_problem(issue_data)

    # Format for OpenHands using shared result
    formatter = PromptFormatter()
    openhands_task = formatter.format_task(
        issue_data,
        memory_context=memory_result[
            'repair_plan'
        ],  # None for baseline, string for memory
    )

    # TODO: Actually run OpenHands here
    # For now, this is a placeholder
    print(f'  Repository: {openhands_task["repository"]}')
    print(f'  Branch: {openhands_task["selected_branch"]}')
    print(f'  Has Repair Plan: {memory_result["repair_plan"] is not None}')
    print('  ⚠️  OpenHands execution not yet implemented')

    # Placeholder result
    result = {
        'instance_id': instance_id,
        'model_name_or_path': model,
        'model_patch': '',  # Would contain the generated patch
        'agent': 'openhands',
        'has_memory': memory_result['repair_plan'] is not None,
        'status': 'pending_implementation',
    }

    return result


def run_batch(
    eval_issues_path: str,
    mode: str,
    model: str,
    memory_layers: Optional[list[str]],
    output_dir: str,
    slice_range: Optional[str] = None,
    hf_dataset: str = 'ci-benchmark-user/ci-repair-bench',
    split: str = 'train',
):
    """
    Run CI-Bench evaluation on OpenHands using SHARED memory plugin

    Args:
        eval_issues_path: Path to eval_set.jsonl
        mode: "baseline" or "memory"
        model: Model name or alias (e.g., "glm5.2" or "minimax2.5")
        memory_layers: List of memory layers (e.g., ["L1", "L2", "L3"])
        output_dir: Output directory
        slice_range: Optional slice (e.g., "0:5" for first 5 issues)
    """
    model = configure_model_environment(model) or model
    print(f'\n{"=" * 80}')
    print('  CI-Bench Evaluation on OpenHands (Shared Memory Plugin)')
    print(f'{"=" * 80}')
    print(f'Mode: {mode}')
    print(f'Model: {model}')
    print(f'Memory Layers: {memory_layers if mode == "memory" else "None (baseline)"}')
    print(f'Output: {output_dir}')
    print(f'{"=" * 80}\n')

    # Setup
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_loader = CIBenchDataLoader(data_root=str(DATA_ROOT))
    eval_issues = data_loader.load_eval_issues(
        eval_issues_path,
        hf_dataset=hf_dataset,
        split=split,
    )

    # Apply slice if provided
    if slice_range:
        start, end = map(int, slice_range.split(':'))
        eval_issues = eval_issues[start:end]
        print(f'Using slice: {slice_range} ({len(eval_issues)} issues)\n')

    # Ensure data files exist. Missing entries are generated by the shared
    # analyzer scripts and then cached for later runs.
    analyzer_llm = _make_analyzer_llm(model)
    log_details, workflow_cache = data_loader.ensure_data_files(
        eval_issues,
        analyzer_llm=analyzer_llm,
        analyzer_model=model,
    )

    # Initialize SHARED memory engine
    if mode == 'memory':
        memory_engine = OpenHandsMemoryAdapter(
            memory_root=str(SharedPaths.get_memory_root()),
            layers=memory_layers or ['L1', 'L2', 'L3'],
            llm=analyzer_llm,
        )
    else:
        # Baseline mode
        memory_engine = OpenHandsMemoryAdapter(
            memory_root=str(SharedPaths.get_memory_root()),
            layers=None,  # No memory layers
        )

    # Run on all issues
    results = []
    for issue in tqdm(eval_issues, desc='Processing issues'):
        result = run_single_issue(
            issue=issue,
            log_details=log_details,
            workflow_cache=workflow_cache,
            memory_engine=memory_engine,  # Same shared plugin for both modes
            model=model,
            results_dir=results_dir,
        )
        results.append(result)

    # Save predictions in mini-swe-agent format
    preds_file = results_dir / 'preds.json'
    with open(preds_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"=" * 80}')
    print(f'✓ Completed {len(results)} issues')
    print(f'✓ Results saved to: {preds_file}')
    print('✓ Used shared memory plugin: consistent with mini-swe-agent')
    print(f'{"=" * 80}\n')

    # Summary
    print('Next steps:')
    print('1. Implement actual OpenHands agent execution')
    print('2. Replace placeholder results with real patches')
    print(f'3. Evaluate with: python scripts/evaluate_ablation_preds.py {preds_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Run OpenHands on CI-Bench evaluation set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline mode (no memory)
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode baseline --model glm5.2 \\
      --output ../results/openhands/glm-5.2/baseline

  # Filter HuggingFace benchmark rows using eval_issues.json
  python ci_bench_runner.py --eval-issues ../data/eval_issues.json \\
      --hf-dataset ci-benchmark-user/ci-repair-bench --split train \\
      --mode baseline --model glm5.2 \\
      --output ../results/openhands/glm-5.2/baseline

  # Memory mode (L1+L2+L3)
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode memory --memory-layers L1 L2 L3 --model glm5.2 \\
      --output ../results/openhands/glm-5.2/L1_L2_L3

  # Test on first 5 issues
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode baseline --model glm5.2 --slice 0:5 \\
      --output ../results/openhands/glm-5.2/test
        """,
    )
    parser.add_argument(
        '--eval-issues',
        required=True,
        help='Path to eval_issues.json / eval_set.jsonl containing issues to run',
    )
    parser.add_argument(
        '--hf-dataset',
        default='ci-benchmark-user/ci-repair-bench',
        help='HuggingFace dataset to load full benchmark rows from',
    )
    parser.add_argument('--split', default='train', help='HuggingFace dataset split')
    parser.add_argument(
        '--mode',
        required=True,
        choices=['baseline', 'memory'],
        help='Mode: baseline (no memory) or memory (with L1/L2/L3)',
    )
    parser.add_argument(
        '--model',
        default='glm5.2',
        help='Model or alias to use (e.g., glm5.2, minimax2.5)',
    )
    parser.add_argument(
        '--memory-layers',
        nargs='+',
        default=['L1', 'L2', 'L3'],
        choices=['L1', 'L2', 'L3'],
        help='Memory layers to use (only for memory mode)',
    )
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument(
        '--slice', help="Optional slice range (e.g., '0:5' for first 5 issues)"
    )

    args = parser.parse_args()

    # Run batch
    run_batch(
        eval_issues_path=args.eval_issues,
        mode=args.mode,
        model=args.model,
        memory_layers=args.memory_layers if args.mode == 'memory' else None,
        output_dir=args.output,
        slice_range=args.slice,
        hf_dataset=args.hf_dataset,
        split=args.split,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
