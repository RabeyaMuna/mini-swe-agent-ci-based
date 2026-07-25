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
from utilities.model_registry import configure_model_environment, resolve_model_alias

from data_loader import CIBenchDataLoader
from interactive_agent import execute_openhands_agent
from prompt_formatter import PromptFormatter

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
            or 'openrouter.ai' in os.getenv('OPENROUTER_BASE_URL', '')
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
        ci_problem = _ci_problem_from_issue(issue_data)
        if not self._memory_system:
            return {'problem': problem, 'repair_plan': None, 'problems': [ci_problem]}

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
        memory_problems = _memory_problems_from_retrieval(memory)
        return {
            'problem': problem,
            'repair_plan': repair_plan,
            'problems': [ci_problem, *memory_problems],
        }


def _ci_problem_from_issue(
    issue_data: dict[str, Any], for_baseline: bool = False
) -> dict[str, Any]:
    """
    Create the CI problem passed to OpenHands.

    Args:
        issue_data: Full issue data
        for_baseline: If True, creates minimal problem for baseline mode

    Returns:
        Problem dict with source, title, description
    """
    workflow = issue_data.get('workflow') or {}

    if for_baseline:
        # BASELINE MODE: Just raw CI failure info, no extra guidance
        # Format the CI failure in baseline mode (errors only, no file list/context)
        from data_loader import CIBenchDataLoader
        from prompt_formatter import PromptFormatter

        log_detail = issue_data.get('ci_failure', {})
        baseline_ci_summary = CIBenchDataLoader._format_processed_ci_failure(
            log_detail, baseline_mode=True
        )
        baseline_description = PromptFormatter.format_baseline_ci_failure(
            baseline_ci_summary
        )

        return {
            'source': 'ci failure',
            'title': 'CI Failure - Fix Required',
            'description': baseline_description,
            'validation_command': issue_data.get('validation_command')
            or workflow.get('validation_command')
            or '',
        }
    else:
        # MEMORY MODE: Can include more structure
        return {
            'source': 'ci failure',
            'title': 'Current CI failure',
            'description': issue_data.get('problem_statement', ''),
            'ci_failure_summary': issue_data.get('ci_failure_summary', ''),
            'workflow_context': workflow,
            'validation_command': issue_data.get('validation_command')
            or workflow.get('validation_command')
            or '',
        }


def _memory_problems_from_retrieval(memory: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract previous-experience follow-up problems from memory retrieval."""
    llm_selection = memory.get('llm_selection') or {}
    raw_problems = (
        llm_selection.get('separate_problems')
        or (llm_selection.get('guidance_document') or {}).get('problems')
        or []
    )
    if not isinstance(raw_problems, list):
        return []

    problems = []
    seen = set()
    for raw in raw_problems:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get('source') or raw.get('memory_level') or '').lower()
        if source in {'ci', 'ci failure'}:
            continue
        description = (
            raw.get('problem_statement')
            or raw.get('description')
            or raw.get('problem')
            or raw.get('error_description')
            or ''
        )
        fix_strategy = raw.get('fix_strategy') or raw.get('how_fixed') or ''
        key = (str(description).strip(), str(fix_strategy).strip())
        if not key[0] or key in seen:
            continue
        seen.add(key)
        problems.append(
            {
                'source': raw.get('source')
                or raw.get('memory_level')
                or 'previous experience',
                'title': f'Previous experience problem {len(problems) + 1}',
                'description': str(description),
                'root_cause': str(raw.get('root_cause') or ''),
                'fix_strategy': str(fix_strategy),
                'validation_command': str(
                    raw.get('verification_cmd') or raw.get('validation_cmd') or ''
                ),
                'files': raw.get('files') or raw.get('affected_files') or [],
            }
        )
    return problems


def _load_decomposed_issues(path: Optional[str]) -> dict[str, dict[str, Any]]:
    """Load decomposed CI problems keyed by id/instance/sha."""
    if not path:
        return {}
    decomp_path = Path(path)
    if not decomp_path.exists():
        raise FileNotFoundError(f'Decomposed issues file not found: {decomp_path}')

    raw_text = decomp_path.read_text(encoding='utf-8').strip()
    if not raw_text:
        return {}
    if raw_text.startswith('['):
        rows = json.loads(raw_text)
    else:
        rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    index = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in (
            'id',
            'issue_id',
            'instance_id',
            'original_issue_id',
            'sha_fail',
        ):
            value = row.get(key)
            if value is not None and str(value):
                index[str(value)] = row
    return index


def _find_decomposed_issue(
    issue: dict[str, Any], decomposed_index: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Find decomposed problem record for an issue."""
    for key in ('id', 'issue_id', 'instance_id', 'original_issue_id', 'sha_fail'):
        value = issue.get(key)
        if value is not None:
            match = decomposed_index.get(str(value))
            if match:
                return match
    return {}


def _decomposed_problem_rows(decomposed_issue: dict[str, Any]) -> list[dict[str, Any]]:
    """Return problem list from either old or commit-based decomposition output."""
    problems = (
        decomposed_issue.get('problem_sequence')
        or decomposed_issue.get('problems')
        or decomposed_issue.get('optimized_problems')
        or []
    )
    return [problem for problem in problems if isinstance(problem, dict)]


def _format_job_records(records: Any) -> str:
    if not isinstance(records, list) or not records:
        return ''
    lines = []
    for record in records[:8]:
        if not isinstance(record, dict):
            continue
        parts = [
            str(record.get('workflow') or ''),
            str(record.get('job') or ''),
            str(record.get('step') or ''),
            str(record.get('validation_cmd') or ''),
        ]
        text = ' | '.join(part for part in parts if part)
        if text:
            lines.append(f'- {text}')
    return '\n'.join(lines)


def _problem_files(problem: dict[str, Any]) -> list[str]:
    files = problem.get('files') or problem.get('affected_files') or []
    if isinstance(files, str):
        return [files]
    return [str(file_path) for file_path in files if file_path]


def _problem_validation_command(problem: dict[str, Any]) -> str:
    return str(
        problem.get('validation_cmd')
        or problem.get('validation_command')
        or problem.get('verification_cmd')
        or ''
    )


def _problem_fix_strategy(problem: dict[str, Any]) -> str:
    return str(
        problem.get('fixes')
        or problem.get('how_fixed')
        or problem.get('fix_strategy')
        or problem.get('changes_made')
        or ''
    )


def _format_decomposed_problem(problem: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert a decomposed CI problem into one agent problem statement."""
    files = _problem_files(problem)
    failed_jobs = _format_job_records(problem.get('current_failed_jobs'))
    fixed_jobs = _format_job_records(problem.get('current_fixed_jobs'))

    description_parts = []
    for key, label in (
        ('problem', 'Problem'),
        ('changes_made', 'Known Repair Change'),
        ('introduces', 'Introduces'),
    ):
        value = problem.get(key)
        if value:
            description_parts.append(f'{label}: {value}')
    if files:
        description_parts.append(f'Affected files: {", ".join(files)}')
    if failed_jobs:
        description_parts.append(f'Current failed CI jobs/steps:\n{failed_jobs}')
    if fixed_jobs:
        description_parts.append(f'Current passed CI jobs:\n{fixed_jobs}')

    failure_type = problem.get('failure_type') or problem.get('problem_type') or 'ci'
    issue_type = problem.get('issue_type') or ''
    title = f'CI problem {problem.get("problem_id") or index}: {failure_type}'
    if issue_type:
        title = f'{title} / {issue_type}'

    return {
        'source': 'ci decomposition',
        'title': title,
        'description': '\n'.join(description_parts).strip()
        or str(problem.get('problem') or ''),
        'root_cause': str(problem.get('root_cause') or ''),
        'fix_strategy': _problem_fix_strategy(problem),
        'validation_command': _problem_validation_command(problem),
        'files': files,
    }


def _ci_problems_for_openhands(
    issue_data: dict[str, Any],
    decomposed_issue: dict[str, Any],
    memory_result: dict[str, Any],
    mode: str = 'baseline',
) -> list[dict[str, Any]]:
    """
    Build the one-at-a-time problem list for the OpenHands agent.

    Args:
        issue_data: Full issue data
        decomposed_issue: Optional decomposed problem data
        memory_result: Memory retrieval result
        mode: 'baseline' or 'memory' - controls whether to use decomposition

    Returns:
        List of problems to solve sequentially
    """
    workflow_context = PromptFormatter._format_workflow_validation(
        issue_data.get('workflow') or {}
    )

    # BASELINE MODE: Always use single CI failure, ignore decomposition
    # Pass for_baseline=True to get minimal, unstructured problem
    if mode == 'baseline':
        problems = [_ci_problem_from_issue(issue_data, for_baseline=True)]
    else:
        # MEMORY MODE: Use decomposition if available, otherwise use memory problems
        decomposed = _decomposed_problem_rows(decomposed_issue)
        if decomposed:
            problems = [
                _format_decomposed_problem(problem, index)
                for index, problem in enumerate(decomposed, 1)
            ]
        else:
            problems = memory_result.get('problems') or [
                _ci_problem_from_issue(issue_data)
            ]

        # Add memory-guided repair plan as additional problem
        if memory_result.get('repair_plan'):
            problems.append(
                {
                    'source': 'previous experience',
                    'title': 'Relevant prior repair guidance',
                    'description': str(memory_result['repair_plan']),
                    'validation_command': issue_data.get('validation_command', ''),
                }
            )

    for problem in problems:
        problem.setdefault('ci_verification_context', workflow_context)
    return problems


def run_single_issue(
    issue: dict[str, Any],
    log_details: dict[str, Any],
    workflow_cache: dict[str, Any],
    memory_engine: OpenHandsMemoryAdapter,
    model: str,
    results_dir: Path,
    decomposed_index: Optional[dict[str, dict[str, Any]]] = None,
    max_steps: int = 15,
    mode: str = 'baseline',
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
    decomposed_issue = _find_decomposed_issue(issue_data, decomposed_index or {})

    # Format for OpenHands using shared result
    formatter = PromptFormatter()
    openhands_task = formatter.format_task(
        issue_data,
        memory_context=memory_result[
            'repair_plan'
        ],  # None for baseline, string for memory
    )

    # Add commit_sha for environment setup
    openhands_task['commit_sha'] = issue_data['sha_fail']
    openhands_task['validation_command'] = issue_data.get('validation_command', '')
    openhands_task['problems'] = _ci_problems_for_openhands(
        issue_data, decomposed_issue, memory_result, mode=mode
    )

    # Execute agent to generate patch
    print(f'  Repository: {openhands_task["repository"]}')
    print(f'  Commit: {openhands_task["commit_sha"][:8]}')
    print(f'  Has Repair Plan: {memory_result["repair_plan"] is not None}')
    print(f'  Has Decomposed Problems: {bool(decomposed_issue)}')
    print(f'  Problems to solve: {len(openhands_task["problems"])}')

    agent_result = execute_openhands_agent(
        openhands_task,
        model=model,
        max_steps=max_steps,
    )

    # Format result
    result = {
        'instance_id': instance_id,
        'model_name_or_path': model,
        'model_patch': agent_result['patch'],
        'agent': 'openhands',
        'has_memory': memory_result['repair_plan'] is not None,
        'has_decomposed_problems': bool(decomposed_issue),
        'problem_count': len(openhands_task['problems']),
        'status': agent_result.get('status')
        or ('success' if agent_result.get('patch') else 'no_patch'),
        'total_cost': agent_result.get('total_cost', 0.0),
        'trajectory': agent_result.get('trajectory', []),
    }

    print(f'  Status: {result["status"]} | Cost: ${result["total_cost"]:.4f}')

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
    decomposed_issues_path: Optional[str] = None,
    max_steps: int = 15,
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
    print(f'Decomposed Issues: {decomposed_issues_path or "None"}')
    print(f'Max Steps Per Problem: {max_steps}')
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

    decomposed_index = _load_decomposed_issues(decomposed_issues_path)
    if decomposed_index:
        print(f' Loaded decomposed CI problems ({len(decomposed_index)} lookup keys)')

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
            decomposed_index=decomposed_index,
            max_steps=max_steps,
            mode=mode,  # Pass mode to ensure baseline doesn't use decomposition
        )
        results.append(result)

    # Save predictions in mini-swe-agent format
    preds_file = results_dir / 'preds.json'
    with open(preds_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"=" * 80}')
    print(f' Completed {len(results)} issues')
    print(f' Results saved to: {preds_file}')
    print(' Used shared memory plugin: consistent with mini-swe-agent')
    print(f'{"=" * 80}\n')

    # Summary
    print('Next steps:')
    print(f'1. Evaluate with: python scripts/evaluate_ablation_preds.py {preds_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Run OpenHands on CI-Bench evaluation set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Baseline: no memory integration
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode baseline --model minimax2.5 \\
      --decomposed-issues ../data/trs/decomposed_issues.json \\
      --output ../results/openhands/minimax-m2.5/baseline

  # 2. L1: failure memory
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode memory --memory-layers L1 --model minimax2.5 \\
      --decomposed-issues ../data/trs/decomposed_issues.json \\
      --output ../results/openhands/minimax-m2.5/L1

  # 3. L1+L2: failure + repository memory
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode memory --memory-layers L1 L2 --model minimax2.5 \\
      --decomposed-issues ../data/trs/decomposed_issues.json \\
      --output ../results/openhands/minimax-m2.5/L1_L2

  # 4. L1+L2+L3: full memory integration
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode memory --memory-layers L1 L2 L3 --model minimax2.5 \\
      --decomposed-issues ../data/trs/decomposed_issues.json \\
      --output ../results/openhands/minimax-m2.5/L1_L2_L3

  # Optional: fetch full rows from Hugging Face using eval IDs
  python ci_bench_runner.py --eval-issues ../data/trs/eval_issue_ids.json \\
      --hf-dataset ci-benchmark-user/ci-repair-bench --split train \\
      --mode baseline --model minimax2.5 \\
      --decomposed-issues ../data/trs/decomposed_issues.json \\
      --output ../results/openhands/minimax-m2.5/baseline

  # Test on first 5 issues
  python ci_bench_runner.py --eval-issues ../data/trs/eval_set.jsonl \\
      --mode baseline --model minimax2.5 --slice 0:5 \\
      --decomposed-issues ../data/trs/decomposed_issues.json \\
      --output ../results/openhands/minimax-m2.5/test
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
    parser.add_argument(
        '--decomposed-issues',
        help=(
            'Optional decomposed_issues.json/JSONL. When provided, each '
            'decomposed CI problem is passed to the agent one at a time in a '
            'single checkout, then one final model_patch is collected.'
        ),
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=15,
        help='Maximum OpenHands agent steps per problem statement',
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
        decomposed_issues_path=args.decomposed_issues,
        max_steps=args.max_steps,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
