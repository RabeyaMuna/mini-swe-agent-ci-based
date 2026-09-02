#!/usr/bin/env python3
"""
run_experepair_baseline.py
==========================
Run ExpeRepair with memory integration on CI failure repair instances.

Supports:
- baseline: No memory, only decomposed problems
- L1, L1+L2, L1+L2+L3: Memory integration (decomposed + retrieved memories)

Usage
-----
  # Baseline (no memory)
  python run_experepair_baseline.py \
    --dataset data/eval_set.jsonl \
    --model deepseek-v4-flash \
    --output results/miniswe-agent/experepair_baseline_deepseek-v4-flash \
    --workers 1 \
    --ablation baseline

  # L1+L2+L3 with bidirectional memory
  python run_experepair_baseline.py \
    --dataset data/eval_set.jsonl \
    --model deepseek-v4-flash \
    --output results/miniswe-agent/experepair_bidirectional/l1_l2_l3_deepseek-v4-flash \
    --workers 1 \
    --ablation L1+L2+L3 \
    --memory-dir data/bidirect_trs
"""

# Configuration: Clone timeouts
# First attempt uses --no-single-branch to get all branches (useful for checkout)
# If that times out, automatically retries with --single-branch (much faster)
CLONE_TIMEOUT_MULTIBRANCH = 600  # 10 minutes for first attempt
CLONE_TIMEOUT_SINGLE_BRANCH = 300  # 5 minutes for retry

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_experepair.wrapper import generate_patch_experepair_baseline
from memory_plugin.memory_plugin import MemoryPlugin

def process_one_instance(
    instance: Dict[str, Any],
    model: str,
    repo_cache_dir: Path,
    ablation: str = "baseline",
    memory_dir: Optional[Path] = None,
    memory_plugin: Optional[MemoryPlugin] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Process a single CI failure instance with ExpeRepair.

    Args:
        instance: CI failure instance
        model: Model name
        repo_cache_dir: Repository cache directory
        ablation: Memory ablation level (baseline, L1, L1+L2, L1+L2+L3)
        memory_dir: Memory directory path (not used directly, kept for compatibility)
        memory_plugin: MemoryPlugin instance for decomposition + retrieval
        top_k: Top-k memories to retrieve

    Returns:
        dict with 'id', 'sha_fail', 'diff' (patch), 'cost'
    """
    instance_id = instance.get('instance_id') or instance.get('id') or instance['sha_fail']
    sha_fail = instance['sha_fail']
    repo_owner = instance.get('repo_owner', '')
    repo_name = instance.get('repo_name', '')
    repo_slug = f"{repo_owner}/{repo_name}"

    print(f"[{instance_id}] Processing {repo_slug} @ {sha_fail[:8]} (ablation={ablation})")

    # Clone or use cached repo
    repo_slug = f"{repo_owner}/{repo_name}".replace('/', '_')
    repo_path = repo_cache_dir / repo_slug

    import subprocess

    if not repo_path.exists():
        github_url = f"https://github.com/{repo_owner}/{repo_name}.git"

        # Try multi-branch clone first (allows checking out any commit)
        print(f"[{instance_id}] Cloning {github_url} (shallow, multi-branch, {CLONE_TIMEOUT_MULTIBRANCH}s timeout)")
        clone_args = ['git', 'clone', '--depth=1', '--no-single-branch', '--progress', github_url, str(repo_path)]

        try:
            subprocess.run(
                clone_args,
                check=True,
                timeout=CLONE_TIMEOUT_MULTIBRANCH,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                text=True
            )
        except subprocess.TimeoutExpired:
            # If multi-branch timed out, retry with single-branch (much faster)
            print(f"[{instance_id}] ⚠️  Clone timed out, retrying with single-branch ({CLONE_TIMEOUT_SINGLE_BRANCH}s)...")
            try:
                subprocess.run(
                    ['git', 'clone', '--depth=1', '--single-branch', '--progress', github_url, str(repo_path)],
                    check=True,
                    timeout=CLONE_TIMEOUT_SINGLE_BRANCH,
                    stderr=subprocess.STDOUT,
                    stdout=subprocess.PIPE,
                    text=True
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                print(f"[{instance_id}] ❌ Clone retry also timed out/failed, skipping")
                return None
        except subprocess.CalledProcessError as e:
            error_msg = e.stdout if e.stdout else 'unknown error'
            print(f"[{instance_id}] ❌ Clone failed: {error_msg}")
            return None

    # Checkout the failing commit
    try:
        subprocess.run(
            ['git', 'checkout', sha_fail],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
            timeout=60
        )
    except subprocess.CalledProcessError:
        # If checkout fails (shallow clone), fetch the specific commit
        print(f"[{instance_id}] Fetching commit {sha_fail[:8]} (unshallowing)")
        try:
            # Try fetching with depth increase
            subprocess.run(
                ['git', 'fetch', '--depth=100', 'origin', sha_fail],
                cwd=str(repo_path),
                check=True,
                capture_output=True,
                timeout=300
            )
            subprocess.run(
                ['git', 'checkout', sha_fail],
                cwd=str(repo_path),
                check=True,
                capture_output=True
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[{instance_id}] ❌ Failed to fetch/checkout commit: {e}")
            return None
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

    # Use MemoryPlugin to get decomposed problems + memories (like Mini-SWE-Agent)
    memory_context = {}
    if memory_plugin:
        try:
            # Load CI failure analysis from log_details.json
            import json
            ci_failure = {}
            try:
                with open("data/log_details.json") as f:
                    log_cache = json.load(f)
                for entry in log_cache:
                    if entry.get('sha_fail') == sha_fail or entry.get('id') == sha_fail:
                        ci_failure = {
                            "error_context": entry.get("error_context", []),
                            "failure_signals": entry.get("failure_signals", []),
                            "relevant_files": entry.get("relevant_files", []),
                            "error_types": entry.get("error_types", [])
                        }
                        break
            except Exception as e:
                print(f"[{instance_id}] Warning: Could not load log_details: {e}")

            # Load workflow validation (optional)
            verification = None
            try:
                with open("data/workflow_validation_cache.json") as f:
                    val_cache = json.load(f)
                for entry in val_cache:
                    if entry.get('sha_fail') == sha_fail:
                        verification = entry
                        break
            except:
                pass

            # Prepare metadata
            workflow_path = instance.get('workflow_path', instance.get('workflow', ''))
            workflow_name = Path(workflow_path).name if workflow_path else ''

            issue_metadata = {
                "sha_fail": sha_fail,
                "repo": repo_slug,
                "workflow_path": workflow_path,
                "workflow_name": workflow_name
            }

            # MemoryPlugin.retrieve handles:
            # 1. Loading/generating decomposed problems from ci_failure + verification
            # 2. Retrieving L1/L2/L3 memories based on ablation
            memory_result = memory_plugin.retrieve(
                ci_failure=ci_failure,
                verification=verification,
                issue_metadata=issue_metadata
            )

            memory_context = memory_result
            problems_count = len(memory_result.get("problems", []))
            l1_count = len(memory_result.get("L1", []))
            l2_count = len(memory_result.get("L2", []))
            l3_count = len(memory_result.get("L3", []))
            print(f"[{instance_id}] Memory: {problems_count} problems, L1={l1_count}, L2={l2_count}, L3={l3_count}")
        except Exception as e:
            print(f"[{instance_id}] Warning: Memory retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            memory_context = {}

    # Call ExpeRepair
    try:
        result = generate_patch_experepair_baseline(
            issue_description=failure_description,
            changed_files=changed_files,
            repo_path=str(repo_path),
            model=model,
            diff=instance.get('diff', ''),
            workflow=instance.get('workflow', ''),
            validation_commands=instance.get('validation_commands', ''),
            memory_context=memory_context,  # Pass memory context
            sha_fail=sha_fail,  # For cache lookup
            instance_id=instance_id  # For cache lookup
        )

        patch = result.get('patch', '')
        cost = result.get('cost', 0.0)
        applicable = result.get('applicable', False)
        validation_error = result.get('validation_error', '')

        status = "✓ applicable" if applicable else "✗ not applicable"
        print(f"[{instance_id}] Generated patch ({len(patch)} chars), {status}, cost=${cost:.4f}")

        return {
            'id': instance_id,
            'sha_fail': sha_fail,
            'diff': patch,
            'cost': cost,
            'applicable': applicable,
            'validation_error': validation_error
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
            'applicable': False,
            'validation_error': '',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Run ExpeRepair with memory integration')
    parser.add_argument('--dataset', required=True, help='Path to eval_set.jsonl')
    parser.add_argument('--model', default='minimax/minimax-m2.5', help='Model to use')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
    parser.add_argument('--issue-ids', help='Comma-separated issue IDs (optional)')
    parser.add_argument('--repo-cache', default='/tmp/experepair_repos', help='Repo cache directory')
    parser.add_argument('--ablation', default='baseline', help='Memory ablation: baseline|L1|L1+L2|L1+L2+L3')
    parser.add_argument('--memory-dir', help='Memory directory (e.g., data/back_trs)')
    parser.add_argument('--top-k', type=int, default=5, help='Top-k memories to retrieve')
    parser.add_argument('--no-resume', action='store_true', help='Force reprocess all instances (ignore existing results)')

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
    print(f"Ablation: {args.ablation}")
    if args.memory_dir:
        print(f"Memory: {args.memory_dir}")
    print(f"Output: {args.output}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create repo cache directory
    repo_cache_dir = Path(args.repo_cache)
    repo_cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Memory Plugin (like Mini-SWE-Agent does)
    memory_plugin = None
    memory_dir = None
    if args.memory_dir:
        memory_dir = Path(args.memory_dir)
        if memory_dir.exists():
            is_baseline = args.ablation.lower() == "baseline"
            print(f"Initializing MemoryPlugin: ablation={args.ablation}, baseline={is_baseline}")

            # Create LLM client for memory decomposition
            from utilities.llm_provider import get_llm
            llm = get_llm(model_key=args.model)

            memory_plugin = MemoryPlugin(
                memory_root=memory_dir,
                result_dir=str(output_dir),
                ablation=args.ablation,
                top_k=args.top_k,
                llm=llm,  # Pass LLM for decomposition and retrieval
                enabled=not is_baseline  # baseline = decompose only, no retrieval
            )
        else:
            print(f"Warning: Memory directory not found: {memory_dir}")

    # Process instances
    preds_file = output_dir / 'preds.json'

    # Load existing results for resume (unless --no-resume)
    results = {}
    if not args.no_resume and preds_file.exists():
        try:
            with open(preds_file, 'r') as f:
                results = json.load(f)
            print(f"📂 Loaded {len(results)} existing results from {preds_file}")
        except Exception as e:
            print(f"⚠️  Could not load existing results: {e}")
            results = {}

        # Filter out already processed instances
        # Normalize IDs to strings for comparison
        completed_ids = set(str(k) for k in results.keys())
        instances_to_process = [inst for inst in instances if str(inst['id']) not in completed_ids]

        if completed_ids:
            print(f"⏭️  Skipping {len(completed_ids)} already completed instances")
        if not instances_to_process:
            print(f"✅ All {len(instances)} instances already completed!")
            print(f"   Results: {preds_file}")
            return

        print(f"🔄 Processing {len(instances_to_process)} remaining instances")
        instances = instances_to_process
    elif args.no_resume:
        print(f"🔄 --no-resume: Processing all {len(instances)} instances from scratch")
    else:
        print(f"🔄 Processing {len(instances)} instances")

    total_cost = sum(r.get('cost', 0.0) for r in results.values())

    if args.workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_one_instance,
                    inst,
                    args.model,
                    repo_cache_dir,
                    args.ablation,
                    memory_dir,
                    memory_plugin,
                    args.top_k
                ): inst
                for inst in instances
            }

            for future in as_completed(futures):
                result = future.result()
                instance_id = result['id']
                results[instance_id] = result
                total_cost += result.get('cost', 0.0)

                # Save incrementally after each result
                with open(preds_file, 'w') as f:
                    json.dump(results, f, indent=2)
    else:
        # Sequential processing
        for inst in instances:
            result = process_one_instance(
                inst,
                args.model,
                repo_cache_dir,
                args.ablation,
                memory_dir,
                memory_plugin,
                args.top_k
            )
            instance_id = result['id']
            results[instance_id] = result
            total_cost += result.get('cost', 0.0)

            # Save incrementally after each instance
            with open(preds_file, 'w') as f:
                json.dump(results, f, indent=2)

    # Final save (already saved incrementally, but ensure it's complete)
    with open(preds_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Completed {len(results)} instances")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Results: {preds_file}")


if __name__ == '__main__':
    main()
