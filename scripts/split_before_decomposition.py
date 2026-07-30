"""
split_before_decomposition.py
==============================
Create chronological split BEFORE decomposition.

This ensures:
1. No temporal data leakage (earliest → memory, latest → eval)
2. Only memory data gets decomposed (saves time and cost)

Usage:
    # All repos
    python scripts/split_before_decomposition.py

    # Specific repos
    python scripts/split_before_decomposition.py --repos agno,flower,camel

    # Custom memory ratio
    python scripts/split_before_decomposition.py --repos agno --memory-ratio 0.2
"""

import sys
from pathlib import Path

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from utilities.chronological_splitter import create_split_from_huggingface

app = typer.Typer()


@app.command()
def main(
    repos: str = typer.Option(
        None,
        help="Comma-separated list of repos to filter (e.g., 'agno,flower,camel')",
    ),
    memory_ratio: float = typer.Option(
        0.3,
        help="Ratio of issues for memory set (default 0.3 = 30%)",
    ),
    output_dir: str = typer.Option(
        "data",
        help="Output directory for split files",
    ),
    dataset_name: str = typer.Option(
        "ci-benchmark-user/ci-repair-bench",
        help="HuggingFace dataset name",
    ),
):
    """
    Create chronological memory/eval split BEFORE decomposition.

    Workflow:
    1. Load dataset from HuggingFace (already sorted by commit_date)
    2. Split chronologically per repository (earliest 30% → memory)
    3. Save memory_set.jsonl and eval_set.jsonl
    4. Decompose ONLY memory_set.jsonl (not all data!)
    """
    # Parse repos
    repo_filter = None
    if repos:
        repo_filter = [r.strip() for r in repos.split(",")]

    # Create split
    result = create_split_from_huggingface(
        dataset_name=dataset_name,
        memory_ratio=memory_ratio,
        output_dir=output_dir,
        repo_filter=repo_filter,
        verbose=True,
    )

    # Print next steps
    print("\n" + "=" * 60)
    print("✓ Split Complete!")
    print("=" * 60)
    print(f"\nMemory: {len(result['memory'])} issues")
    print(f"Eval:   {len(result['eval'])} issues")
    print("\nNext step: Decompose ONLY memory data:")
    print("  python scripts/decompose_ci_failure.py \\")
    print(f"      --input {output_dir}/memory_set.jsonl \\")
    print(f"      --output {output_dir}/memory_decomposed.json")
    print()


if __name__ == "__main__":
    app()
