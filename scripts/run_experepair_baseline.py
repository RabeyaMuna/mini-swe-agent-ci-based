#!/usr/bin/env python3
"""
run_experepair_baseline.py
==========================
Run ExpeRepair baseline (no memory, no iteration) on CI failure repair instances.

Matches the output format of miniswe-agent for fair comparison.

Usage
-----
  python run_experepair_baseline.py \
    --dataset data/eval_set.jsonl \
    --model minimax/minimax-m2.5 \
    --output results/miniswe-agent/experepair_baseline_minimax-m2.5 \
    --workers 4
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_experepair.wrapper import generate_patch_experepair_baseline


def process_one_instance(
    instance: Dict[str, Any],
    model: str,
    repo_cache_dir: Path
) -> Dict[str, Any]:
    """
    Process a single CI failure instance with ExpeRepair baseline.

    Returns:
        dict with 'id', 'sha_fail', 'diff' (patch), 'cost'
    """
    instance_id = instance.get('instance_id') or instance.get('id') or instance['sha_fail']
    sha_fail = instance['sha_fail']
    repo_owner = instance.get('repo_owner', '')
    repo_name = instance.get('repo_name', '')

    print(f"[{instance_id}] Processing {repo_owner}/{repo_name} @ {sha_fail[:8]}")

    # Clone or use cached repo
    repo_slug = f"{repo_owner}/{repo_name}".replace('/', '_')
    repo_path = repo_cache_dir / repo_slug

    if not repo_path.exists():
        # Clone the repo
        import subprocess
        github_url = f"https://github.com/{repo_owner}/{repo_name}.git"
        print(f"[{instance_id}] Cloning {github_url}")
        subprocess.run(
            ['git', 'clone', '--depth', '50', github_url, str(repo_path)],
            check=True,
            capture_output=True
        )

    # Checkout the failing commit
    import subprocess
    subprocess.run(
        ['git', 'checkout', sha_fail],
        cwd=str(repo_path),
        check=True,
        capture_output=True
    )

    # Extract CI failure context
    failure_description = instance.get('logs', '')
    if isinstance(failure_description, list):
        # logs is list of {step_name, log} dicts
        failure_description = '\n'.join(
            f"[{step.get('step_name', 'unknown')}]\n{step.get('log', '')}"
            for step in failure_description
        )

    # Get changed files (if available in instance)
    changed_files = instance.get('changed_files', [])
    if not changed_files:
        # Extract from diff if available
        diff = instance.get('diff', '')
        if diff:
            import re
            changed_files = re.findall(r'^diff --git a/(.*?) b/', diff, re.MULTILINE)

    # Call ExpeRepair baseline
    try:
        result = generate_patch_experepair_baseline(
            issue_description=failure_description,
            changed_files=changed_files,
            repo_path=str(repo_path),
            model=model,
            diff=instance.get('diff', ''),
            workflow=instance.get('workflow', ''),
            validation_commands=instance.get('validation_commands', '')
        )

        patch = result.get('patch', '')
        cost = result.get('cost', 0.0)

        print(f"[{instance_id}] Generated patch ({len(patch)} chars), cost=${cost:.4f}")

        return {
            'id': instance_id,
            'sha_fail': sha_fail,
            'diff': patch,
            'cost': cost
        }

    except Exception as e:
        print(f"[{instance_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()

        return {
            'id': instance_id,
            'sha_fail': sha_fail,
            'diff': '',
            'cost': 0.0,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Run ExpeRepair baseline on CI failures')
    parser.add_argument('--dataset', required=True, help='Path to eval_set.jsonl')
    parser.add_argument('--model', default='minimax/minimax-m2.5', help='Model to use')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
    parser.add_argument('--issue-ids', help='Comma-separated issue IDs (optional)')
    parser.add_argument('--repo-cache', default='/tmp/experepair_repos', help='Repo cache directory')

    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    instances = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))

    # Filter by issue IDs if provided
    if args.issue_ids:
        issue_id_set = set(args.issue_ids.split(','))
        instances = [
            inst for inst in instances
            if (inst.get('instance_id') or inst.get('id') or inst['sha_fail']) in issue_id_set
        ]

    print(f"Processing {len(instances)} instances with {args.workers} workers")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create repo cache directory
    repo_cache_dir = Path(args.repo_cache)
    repo_cache_dir.mkdir(parents=True, exist_ok=True)

    # Process instances
    results = {}
    total_cost = 0.0

    if args.workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one_instance, inst, args.model, repo_cache_dir): inst
                for inst in instances
            }

            for future in as_completed(futures):
                result = future.result()
                instance_id = result['id']
                results[instance_id] = result
                total_cost += result.get('cost', 0.0)
    else:
        # Sequential processing
        for inst in instances:
            result = process_one_instance(inst, args.model, repo_cache_dir)
            instance_id = result['id']
            results[instance_id] = result
            total_cost += result.get('cost', 0.0)

    # Save results in mini-swe-agent format: preds.json
    preds_file = output_dir / 'preds.json'
    with open(preds_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Completed {len(results)} instances")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Results: {preds_file}")


if __name__ == '__main__':
    main()
