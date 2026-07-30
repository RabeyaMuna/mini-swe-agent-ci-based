"""
analyze_temporal_leakage.py
============================
Analyze temporal data leakage in the current memory/eval split.

This script:
1. Fetches commit timestamps from GitHub for all issues
2. Analyzes whether future issues are in memory for past eval issues
3. Provides recommendations for chronological splitting

Usage:
    python scripts/analyze_temporal_leakage.py \
        --memory-ids data/trs/memory_issue_ids.json \
        --eval-ids data/trs/eval_issue_ids.json \
        --output data/trs/temporal_analysis.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import requests
import typer
from datasets import load_dataset
from rich.console import Console
from rich.progress import track
from rich.table import Table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer()


def _get_commit_timestamp(owner: str, repo: str, sha: str) -> Optional[str]:
    """
    Fetch commit timestamp from GitHub API.

    Returns:
        ISO timestamp string or None if failed
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"

    try:
        # Add delay to respect rate limits
        time.sleep(0.5)

        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            # Get commit date (author date)
            timestamp = data["commit"]["author"]["date"]
            return timestamp
        elif response.status_code == 404:
            logger.warning(f"Commit not found: {owner}/{repo}@{sha[:8]}")
            return None
        elif response.status_code == 403:
            logger.error(
                "GitHub API rate limit exceeded. Consider using authentication."
            )
            return None
        else:
            logger.warning(
                f"Failed to fetch {owner}/{repo}@{sha[:8]}: {response.status_code}"
            )
            return None

    except Exception as e:
        logger.warning(f"Error fetching commit timestamp: {e}")
        return None


def _fetch_all_timestamps(
    dataset_name: str = "ci-benchmark-user/ci-repair-bench",
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch commit timestamps for all issues in the dataset.

    Returns:
        {issue_id: {timestamp, repo, sha, ...}}
    """
    # Check cache first
    if cache_path and cache_path.exists() and not force_refresh:
        console.print(f"[green]Loading timestamps from cache: {cache_path}[/green]")
        with open(cache_path, "r") as f:
            return json.load(f)

    console.print(f"[cyan]Loading dataset: {dataset_name}[/cyan]")
    ds = load_dataset(dataset_name)
    data = ds["train"]

    timestamps = {}

    console.print(f"[cyan]Fetching commit timestamps for {len(data)} issues...[/cyan]")
    console.print("[yellow]This may take a while due to API rate limits[/yellow]")

    for item in track(data, description="Fetching timestamps"):
        issue_id = str(item["id"])
        owner = item["repo_owner"]
        repo_name = item["repo_name"]
        sha_fail = item["sha_fail"]

        # Fetch timestamp
        timestamp = _get_commit_timestamp(owner, repo_name, sha_fail)

        timestamps[issue_id] = {
            "timestamp": timestamp,
            "repo": f"{owner}/{repo_name}",
            "sha": sha_fail,
            "repo_owner": owner,
            "repo_name": repo_name,
        }

        if not timestamp:
            logger.warning(f"Could not fetch timestamp for issue {issue_id}")

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(timestamps, f, indent=2)
        console.print(f"[green]Saved timestamps to cache: {cache_path}[/green]")

    return timestamps


def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _analyze_temporal_leakage(
    memory_ids: List[str],
    eval_ids: List[str],
    timestamps: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze temporal data leakage.

    Returns analysis dict with:
    - leakage_count: number of eval issues with future memory
    - leakage_examples: list of examples
    - chronological_order: suggested chronological split
    """
    analysis = {
        "total_memory": len(memory_ids),
        "total_eval": len(eval_ids),
        "missing_timestamps": 0,
        "leakage_count": 0,
        "leakage_examples": [],
        "memory_date_range": {},
        "eval_date_range": {},
    }

    # Get timestamps for memory and eval sets
    memory_timestamps = []
    eval_timestamps = []

    for mid in memory_ids:
        ts_data = timestamps.get(mid)
        if ts_data and ts_data.get("timestamp"):
            dt = _parse_timestamp(ts_data["timestamp"])
            if dt:
                memory_timestamps.append((mid, dt, ts_data))

    for eid in eval_ids:
        ts_data = timestamps.get(eid)
        if ts_data and ts_data.get("timestamp"):
            dt = _parse_timestamp(ts_data["timestamp"])
            if dt:
                eval_timestamps.append((eid, dt, ts_data))

    analysis["missing_timestamps"] = (len(memory_ids) - len(memory_timestamps)) + (
        len(eval_ids) - len(eval_timestamps)
    )

    if not memory_timestamps or not eval_timestamps:
        console.print("[red]Not enough timestamp data to analyze leakage[/red]")
        return analysis

    # Date ranges
    memory_dates = [dt for _, dt, _ in memory_timestamps]
    eval_dates = [dt for _, dt, _ in eval_timestamps]

    analysis["memory_date_range"] = {
        "earliest": min(memory_dates).isoformat(),
        "latest": max(memory_dates).isoformat(),
    }
    analysis["eval_date_range"] = {
        "earliest": min(eval_dates).isoformat(),
        "latest": max(eval_dates).isoformat(),
    }

    # Check for temporal leakage
    # For each eval issue, check if any memory issue is from the future
    leakage_cases = []

    for eval_id, eval_dt, eval_data in eval_timestamps:
        future_memory = []

        for mem_id, mem_dt, mem_data in memory_timestamps:
            # Same repo only (cross-repo doesn't matter as much)
            if mem_data["repo"] == eval_data["repo"] and mem_dt > eval_dt:
                days_ahead = (mem_dt - eval_dt).days
                future_memory.append(
                    {
                        "memory_id": mem_id,
                        "memory_date": mem_dt.isoformat(),
                        "days_ahead": days_ahead,
                    }
                )

        if future_memory:
            leakage_cases.append(
                {
                    "eval_id": eval_id,
                    "eval_date": eval_dt.isoformat(),
                    "repo": eval_data["repo"],
                    "future_memory_count": len(future_memory),
                    "future_memory_samples": future_memory[:3],  # Show first 3
                }
            )

    analysis["leakage_count"] = len(leakage_cases)
    analysis["leakage_examples"] = leakage_cases[:10]  # Show first 10

    # Calculate leakage percentage
    if eval_timestamps:
        analysis["leakage_percentage"] = len(leakage_cases) / len(eval_timestamps) * 100

    return analysis


def _suggest_chronological_split(
    timestamps: Dict[str, Dict[str, Any]],
    memory_ratio: float = 0.3,
) -> Dict[str, Any]:
    """
    Suggest a chronological split strategy.

    Returns:
        {
            'strategy': 'description',
            'memory_ids': [...],
            'eval_ids': [...],
            'cutoff_date': '...',
        }
    """
    # Get all issues with timestamps
    issues_with_ts = []
    for issue_id, ts_data in timestamps.items():
        if ts_data.get("timestamp"):
            dt = _parse_timestamp(ts_data["timestamp"])
            if dt:
                issues_with_ts.append((issue_id, dt, ts_data))

    if not issues_with_ts:
        return {}

    # Sort by timestamp (ascending - oldest first)
    issues_with_ts.sort(key=lambda x: x[1])

    # Split by memory_ratio (earliest X% as memory)
    n_memory = max(1, int(len(issues_with_ts) * memory_ratio))

    memory_set = issues_with_ts[:n_memory]
    eval_set = issues_with_ts[n_memory:]

    cutoff_date = memory_set[-1][1] if memory_set else None

    return {
        "strategy": "chronological",
        "description": f"Use earliest {memory_ratio:.1%} as memory, rest for eval",
        "total_issues": len(issues_with_ts),
        "memory_count": len(memory_set),
        "eval_count": len(eval_set),
        "cutoff_date": cutoff_date.isoformat() if cutoff_date else None,
        "memory_ids": [mid for mid, _, _ in memory_set],
        "eval_ids": [eid for eid, _, _ in eval_set],
        "memory_date_range": {
            "earliest": memory_set[0][1].isoformat() if memory_set else None,
            "latest": memory_set[-1][1].isoformat() if memory_set else None,
        },
        "eval_date_range": {
            "earliest": eval_set[0][1].isoformat() if eval_set else None,
            "latest": eval_set[-1][1].isoformat() if eval_set else None,
        },
    }


@app.command()
def main(
    memory_ids: Path = typer.Option(
        "data/trs/memory_issue_ids.json",
        help="Path to memory issue IDs JSON",
    ),
    eval_ids: Path = typer.Option(
        "data/trs/eval_issue_ids.json",
        help="Path to eval issue IDs JSON",
    ),
    output: Path = typer.Option(
        "data/trs/temporal_analysis.json",
        help="Output path for temporal analysis",
    ),
    timestamp_cache: Path = typer.Option(
        "data/trs/commit_timestamps.json",
        help="Cache for commit timestamps",
    ),
    force_refresh: bool = typer.Option(
        False,
        help="Force refresh timestamps from GitHub API",
    ),
    dataset_name: str = typer.Option(
        "ci-benchmark-user/ci-repair-bench",
        help="HuggingFace dataset name",
    ),
):
    """
    Analyze temporal data leakage in memory/eval split.
    """
    console.print("[bold cyan]Temporal Data Leakage Analysis[/bold cyan]\n")

    # Load memory and eval IDs
    with open(memory_ids, "r") as f:
        memory_id_list = json.load(f)

    with open(eval_ids, "r") as f:
        eval_id_list = json.load(f)

    console.print(f"Memory set: {len(memory_id_list)} issues")
    console.print(f"Eval set: {len(eval_id_list)} issues\n")

    # Fetch timestamps
    timestamps = _fetch_all_timestamps(
        dataset_name=dataset_name,
        cache_path=timestamp_cache,
        force_refresh=force_refresh,
    )

    console.print(f"\n[green]Fetched timestamps for {len(timestamps)} issues[/green]\n")

    # Analyze leakage
    console.print("[cyan]Analyzing temporal leakage...[/cyan]")
    leakage_analysis = _analyze_temporal_leakage(
        memory_id_list,
        eval_id_list,
        timestamps,
    )

    # Display results
    console.print("\n[bold]Leakage Analysis Results:[/bold]")
    console.print(f"  Missing timestamps: {leakage_analysis['missing_timestamps']}")
    console.print(
        f"  Memory date range: {leakage_analysis.get('memory_date_range', {})}"
    )
    console.print(f"  Eval date range: {leakage_analysis.get('eval_date_range', {})}")
    console.print(
        f"\n  [bold red]Temporal leakage cases: {leakage_analysis['leakage_count']}[/bold red]"
    )

    if leakage_analysis.get("leakage_percentage"):
        console.print(
            f"  [bold red]Leakage percentage: {leakage_analysis['leakage_percentage']:.1f}%[/bold red]"
        )

    # Show examples
    if leakage_analysis["leakage_examples"]:
        console.print("\n[bold]Example leakage cases:[/bold]")
        table = Table(show_header=True)
        table.add_column("Eval ID")
        table.add_column("Eval Date")
        table.add_column("Repo")
        table.add_column("Future Memory")

        for example in leakage_analysis["leakage_examples"][:5]:
            table.add_row(
                example["eval_id"],
                example["eval_date"][:10],
                example["repo"],
                str(example["future_memory_count"]),
            )

        console.print(table)

    # Suggest chronological split
    console.print("\n[cyan]Computing chronological split suggestion...[/cyan]")
    chronological_split = _suggest_chronological_split(timestamps, memory_ratio=0.3)

    if chronological_split:
        console.print("\n[bold]Chronological Split Suggestion:[/bold]")
        console.print(f"  Strategy: {chronological_split['description']}")
        console.print(
            f"  Cutoff date: {chronological_split.get('cutoff_date', 'N/A')[:10]}"
        )
        console.print(f"  Memory: {chronological_split['memory_count']} issues")
        console.print(f"  Eval: {chronological_split['eval_count']} issues")
        console.print(f"  Memory dates: {chronological_split['memory_date_range']}")
        console.print(f"  Eval dates: {chronological_split['eval_date_range']}")

    # Save full analysis
    full_analysis = {
        "current_split": {
            "memory_count": len(memory_id_list),
            "eval_count": len(eval_id_list),
        },
        "leakage_analysis": leakage_analysis,
        "chronological_split_suggestion": chronological_split,
        "timestamp_cache_path": str(timestamp_cache),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(full_analysis, f, indent=2)

    console.print(f"\n[green]Full analysis saved to: {output}[/green]")

    # Print recommendations
    console.print("\n[bold yellow]Recommendations:[/bold yellow]")
    if leakage_analysis["leakage_count"] > 0:
        console.print("[red] Temporal data leakage detected![/red]")
        console.print("\n  To fix this, use chronological splitting:")
        console.print("  1. Sort all issues by commit timestamp (ascending)")
        console.print("  2. Use earliest 30% as memory set")
        console.print("  3. Use remaining 70% for evaluation")
        console.print("  4. When retrieving memory for an eval issue, only consider")
        console.print("     memory from commits BEFORE the eval issue's timestamp")
    else:
        console.print("[green]✓ No temporal leakage detected[/green]")


if __name__ == "__main__":
    app()
