#!/usr/bin/env python3
"""
Run unified ablation experiments.

This script:
1. Selects 100 instances for evaluation
2. Links existing baseline/full_recap results
3. Runs 5 new ablation configurations
4. Generates comparison tables

REUSES: Existing run_miniswe_ci_bench.py evaluation infrastructure
NO MODIFICATIONS to core code
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation_configs import ABLATION_CONFIGS, get_configs_to_run, get_configs_to_reuse


def select_instances(dataset_path, count=100, seed=42):
    """
    Select instances for ablation study.

    Args:
        dataset_path: Path to eval_set.jsonl
        count: Number of instances to select
        seed: Random seed for reproducibility

    Returns:
        List of instance IDs
    """
    import random
    random.seed(seed)

    # Load dataset
    instances = []
    with open(dataset_path) as f:
        for line in f:
            instances.append(json.loads(line))

    # Select random subset
    selected = random.sample(instances, min(count, len(instances)))
    instance_ids = [inst["instance_id"] for inst in selected]

    print(f"Selected {len(instance_ids)} instances for ablation study")
    return instance_ids


def create_instance_file(instance_ids, output_path):
    """Create instance ID file for evaluation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(instance_ids, f, indent=2)
    print(f"Saved instance IDs to {output_path}")


def link_existing_results(source_dir, target_dir):
    """Create symlink to existing results."""
    source = Path(source_dir)
    target = Path(target_dir)

    if not source.exists():
        print(f"WARNING: Source directory does not exist: {source}")
        return False

    if target.exists():
        print(f"Target already exists: {target}")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())
    print(f"Linked {source} -> {target}")
    return True


def run_ablation_config(config_name, config, instance_ids_file, model, output_root):
    """
    Run one ablation configuration.

    Args:
        config_name: Name of configuration
        config: Configuration dict
        instance_ids_file: Path to instance IDs file
        model: Model name (e.g., deepseek-v4-flash)
        output_root: Root directory for results
    """
    from scripts.run_miniswe_ci_bench import main as run_miniswe

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}\n")

    # Build arguments for run_miniswe_ci_bench
    # We'll call it directly instead of via CLI

    # For now, print the command that should be run
    # TODO: Actually integrate with existing evaluation infrastructure

    cmd = f"""
python scripts/run_miniswe_ci_bench.py \\
    --dataset data/eval_set.jsonl \\
    --model {model} \\
    --ablation {config['ablation']} \\
    --direction bidirectional \\
    --output_root {output_root}/{config_name} \\
    --workers 1
"""

    print(f"Command to run:\n{cmd}")
    print("\nNOTE: Manual integration with custom retrieval classes needed")
    print(f"      Use retrieval class: {config['retrieval_class']}")


def main():
    parser = argparse.ArgumentParser(description="Run unified ablation experiments")
    parser.add_argument("--dataset", default="data/eval_set.jsonl", help="Path to evaluation dataset")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model to use")
    parser.add_argument("--num_instances", type=int, default=100, help="Number of instances")
    parser.add_argument("--output_root", default="results/03_unified_ablations", help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Only show what would be done")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Step 1: Select instances
    print("Step 1: Selecting instances...")
    instance_ids = select_instances(
        PROJECT_ROOT / args.dataset,
        count=args.num_instances,
        seed=42
    )

    # Save instance IDs
    instance_file = output_root / "selected_instances.json"
    create_instance_file(instance_ids, instance_file)

    # Step 2: Link existing results
    print("\nStep 2: Linking existing baseline results...")
    for config_name, config in get_configs_to_reuse().items():
        source_dir = PROJECT_ROOT / config["existing_path"]
        target_dir = output_root / config_name
        link_existing_results(source_dir, target_dir)

    # Step 3: Run new configurations
    print("\nStep 3: Running new ablation configurations...")
    if args.dry_run:
        print("DRY RUN - Not actually running experiments")

    for config_name, config in get_configs_to_run().items():
        if args.dry_run:
            print(f"\n[DRY RUN] Would run: {config_name}")
            print(f"  Description: {config['description']}")
            print(f"  Retrieval class: {config['retrieval_class']}")
        else:
            run_ablation_config(
                config_name,
                config,
                instance_file,
                args.model,
                args.output_root
            )

    print("\n" + "="*60)
    print("Summary:")
    print(f"  Instances: {len(instance_ids)}")
    print(f"  Configs reused: {len(get_configs_to_reuse())}")
    print(f"  Configs to run: {len(get_configs_to_run())}")
    print(f"  Output: {output_root}")
    print("="*60)


if __name__ == "__main__":
    main()
