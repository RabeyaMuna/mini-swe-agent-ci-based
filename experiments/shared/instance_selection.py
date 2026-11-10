"""
Shared utilities for instance selection across experiments.

Ensures consistent instance selection and subset relationships.
"""

import json
import random
from pathlib import Path


def select_instances(dataset_path, count, seed=42, instance_ids=None):
    """
    Select instances for evaluation.

    Args:
        dataset_path: Path to eval_set.jsonl
        count: Number of instances to select
        seed: Random seed for reproducibility
        instance_ids: Optional list of specific IDs to select from

    Returns:
        List of instance dicts
    """
    random.seed(seed)

    # Load dataset
    instances = []
    with open(dataset_path) as f:
        for line in f:
            inst = json.loads(line)
            if instance_ids is None or inst["instance_id"] in instance_ids:
                instances.append(inst)

    # Select random subset
    selected = random.sample(instances, min(count, len(instances)))
    return selected


def save_instance_ids(instances, output_path):
    """Save instance IDs to file."""
    ids = [inst["instance_id"] for inst in instances]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(ids, f, indent=2)
    return ids
