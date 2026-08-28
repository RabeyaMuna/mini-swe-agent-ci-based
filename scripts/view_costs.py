#!/usr/bin/env python3
"""
View API costs from costs.json for all ablation configurations.

Usage:
    python scripts/view_costs.py results/miniswe-agent/baseline_deepseek-v4-flash/costs.json
    python scripts/view_costs.py results/miniswe-agent/  # scans all subdirectories
"""

import json
import sys
from pathlib import Path


def read_costs(costs_path: Path) -> dict:
    """Read costs.json file."""
    if not costs_path.exists():
        return {}
    try:
        return json.loads(costs_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error reading {costs_path}: {e}")
        return {}


def display_costs(costs_data: dict, title: str = ""):
    """Display costs in a formatted table."""
    if not costs_data:
        print("No cost data found.")
        return

    print(f"\n{'='*100}")
    print(f"  {title if title else 'API Cost Summary'}")
    print(f"{'='*100}")

    # Group by ablation
    by_ablation = {}
    for config_key, config_data in costs_data.items():
        ablation = config_data.get("ablation", "unknown")
        if ablation not in by_ablation:
            by_ablation[ablation] = []
        by_ablation[ablation].append((config_key, config_data))

    # Display grouped by ablation
    total_all = 0.0
    for ablation in sorted(by_ablation.keys()):
        configs = by_ablation[ablation]
        print(f"\n{ablation}:")
        print(f"  {'Model':<30} {'Direction':<15} {'Instances':<10} {'Total Cost':<15} {'Avg/Instance'}")
        print(f"  {'-'*30} {'-'*15} {'-'*10} {'-'*15} {'-'*15}")

        for config_key, config_data in sorted(configs, key=lambda x: x[1].get("direction", "")):
            model = config_data.get("model", "unknown")
            direction = config_data.get("direction", "unknown")
            instances = config_data.get("instances", {})

            total_cost = sum(instances.values())
            n_instances = len(instances)
            avg_cost = total_cost / n_instances if n_instances > 0 else 0.0

            print(f"  {model:<30} {direction:<15} {n_instances:<10} ${total_cost:<14.6f} ${avg_cost:.6f}")
            total_all += total_cost

    print(f"\n{'='*100}")
    print(f"  GRAND TOTAL: ${total_all:.6f}")
    print(f"{'='*100}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/view_costs.py <path-to-costs.json or directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        # Single costs.json file
        costs_data = read_costs(path)
        display_costs(costs_data, f"Costs from {path}")

    elif path.is_dir():
        # Scan directory for all costs.json files
        costs_files = list(path.rglob("costs.json"))

        if not costs_files:
            print(f"No costs.json files found in {path}")
            sys.exit(1)

        print(f"\nFound {len(costs_files)} costs.json file(s)\n")

        # Aggregate all costs
        all_costs = {}
        for costs_file in costs_files:
            costs_data = read_costs(costs_file)
            # Merge into all_costs
            for config_key, config_data in costs_data.items():
                if config_key not in all_costs:
                    all_costs[config_key] = config_data
                else:
                    # Merge instances
                    all_costs[config_key]["instances"].update(config_data.get("instances", {}))

        display_costs(all_costs, f"Aggregated Costs from {path}")

    else:
        print(f"Error: {path} is not a file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
