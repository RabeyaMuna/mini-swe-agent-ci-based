"""
prepare_memory_train_test_split.py
===================================
Analyze entire CI repair dataset, compute similarity, and split into:
- 30% memory set (most representative issues → saved as L2 memory)
- 70% evaluation set (testing issues that will use the 30% as PREVIOUS EXPERIENCE)

Usage:
    python scripts/prepare_memory_train_test_split.py \
        --dataset data/trs/filtered_issues.jsonl \
        --output-dir data/trs \
        --memory-ratio 0.3 \
        --similarity-threshold 0.5
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import typer
from datasets import load_dataset
from rich.console import Console
from rich.progress import track
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer()


def _load_from_huggingface(dataset_name: str = "ci-benchmark-user/ci-repair-bench") -> List[Dict[str, Any]]:
    """Load issues from HuggingFace dataset."""
    console.print(f"[cyan]Loading dataset from HuggingFace: {dataset_name}[/cyan]")

    try:
        # Direct load from HuggingFace
        ds = load_dataset(dataset_name)
        data = ds['train']
        console.print(f"[green]Loaded {len(data)} issues from HuggingFace[/green]")

        # Convert to list of dicts
        issues = [dict(item) for item in data]
        return issues

    except Exception as e:
        console.print(f"[red]Failed to load from HuggingFace: {e}[/red]")
        raise RuntimeError(f"Could not load dataset from HuggingFace: {e}")


def _load_from_jsonl(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from local JSONL file."""
    console.print(f"[cyan]Loading dataset from local file: {dataset_path}[/cyan]")
    issues = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                issues.append(json.loads(line))
    console.print(f"[green]Loaded {len(issues)} issues from file[/green]")
    return issues


def _load_or_create_decomposed_issues(
    issues: List[Dict[str, Any]],
    decomposed_cache_path: Path,
    force_recompute: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Load decomposed issues from cache or create if not exists.

    Returns: {issue_id: decomposed_data}
    """
    # Check cache
    if decomposed_cache_path.exists() and not force_recompute:
        console.print(f"[green]Loading decomposed issues from cache: {decomposed_cache_path}[/green]")
        with open(decomposed_cache_path, "r") as f:
            decomposed_cache = json.load(f)

        # Convert to dict keyed by issue_id
        decomposed_dict = {}
        for item in decomposed_cache if isinstance(decomposed_cache, list) else [decomposed_cache]:
            issue_id = str(item.get("original_issue_id") or item.get("id") or "")
            if issue_id:
                decomposed_dict[issue_id] = item

        console.print(f"[green]Found {len(decomposed_dict)} decomposed issues in cache[/green]")

        # Check if all issues are in cache
        missing_ids = []
        for issue in issues:
            issue_id = str(issue.get("instance_id") or issue.get("id") or "")
            if issue_id and issue_id not in decomposed_dict:
                missing_ids.append(issue_id)

        if missing_ids:
            console.print(f"[yellow]Warning: {len(missing_ids)} issues not in cache:[/yellow]")
            for mid in missing_ids[:5]:
                console.print(f"  - {mid}")
            if len(missing_ids) > 5:
                console.print(f"  ... and {len(missing_ids) - 5} more")

        return decomposed_dict

    # Need to create decomposition
    console.print(f"[yellow]Decomposed cache not found at {decomposed_cache_path}[/yellow]")
    console.print(f"[yellow]You need to run decomposition first using:[/yellow]")
    console.print(f"[cyan]python scripts/decompose_ci_failure.py --batch[/cyan]")
    console.print(f"[yellow]Or provide an existing decomposed_issues.json path[/yellow]")

    return {}


def _extract_diff_patterns(diff: str) -> str:
    """
    Extract patterns from diff (ground truth) for similarity analysis.

    Analyzes:
    - File types changed (.py, .rst, .toml)
    - Change types (additions, deletions, modifications)
    - Change patterns (imports, config, formatting, logic)
    """
    if not diff:
        return ""

    patterns = []

    # Extract changed files and their extensions
    import re
    file_headers = re.findall(r'(?:---|\+\+\+) [ab]/(.*?)(?:\s|$)', diff)
    extensions = set()
    for file_path in file_headers:
        if "." in file_path:
            ext = file_path.split(".")[-1]
            extensions.add(ext)

    if extensions:
        patterns.append(f"Changed: {', '.join(sorted(extensions))}")

    # Detect change patterns
    if re.search(r'^\+import |^\+from .* import', diff, re.MULTILINE):
        patterns.append("Pattern: import changes")
    if re.search(r'^\+\s*".*":|^\+\s*[a-z_]+\s*=', diff, re.MULTILINE):
        patterns.append("Pattern: config/value changes")
    if re.search(r'^\+def |^\+class |^\+async def', diff, re.MULTILINE):
        patterns.append("Pattern: new functions/classes")
    if re.search(r'===+|---+|\*\*\*', diff, re.MULTILINE):
        patterns.append("Pattern: RST/doc formatting")

    # Count additions vs deletions
    additions = len(re.findall(r'^\+[^+]', diff, re.MULTILINE))
    deletions = len(re.findall(r'^-[^-]', diff, re.MULTILINE))

    if additions > deletions * 2:
        patterns.append("Type: mostly additions")
    elif deletions > additions * 2:
        patterns.append("Type: mostly deletions")
    else:
        patterns.append("Type: mixed changes")

    return " | ".join(patterns)


def _build_issue_text_from_decomposed(
    issue: Dict[str, Any],
    decomposed: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build text representation using EXISTING decomposition if available.

    Priority:
    1. Use decomposed data (problems, root causes, fixes) if available
    2. Fall back to raw issue data

    Combines (as user specified):
    - CI log info (error messages, validation outputs)
    - CI validations from workflow (what commands failed)
    - Diff (ground truth - what changes were needed)
    - Decomposed problems (if available)
    """
    parts = []

    # Use decomposed data if available
    if decomposed and decomposed.get("problems"):
        # REUSE existing decomposition!
        problems = decomposed.get("problems", [])

        # Extract from decomposed problems
        validation_cmds = set()
        failure_types = set()
        all_problem_texts = []
        all_root_causes = []
        all_files = set()

        for problem in problems[:5]:  # Use first 5 problems for embedding
            if problem.get("validation_cmd"):
                validation_cmds.add(problem["validation_cmd"])
            if problem.get("failure_type"):
                failure_types.add(problem["failure_type"])
            if problem.get("problem"):
                all_problem_texts.append(problem["problem"][:200])
            if problem.get("root_cause"):
                all_root_causes.append(problem["root_cause"][:200])
            if problem.get("affected_files"):
                all_files.update(problem["affected_files"])

        # Build from decomposed data
        if validation_cmds:
            parts.append(f"Validations: {', '.join(sorted(validation_cmds)[:3])}")
        if failure_types:
            parts.append(f"Failures: {', '.join(sorted(failure_types)[:3])}")
        if all_problem_texts:
            parts.append(f"Problems: {' | '.join(all_problem_texts[:3])}")
        if all_root_causes:
            parts.append(f"RootCauses: {' | '.join(all_root_causes[:3])}")

        # File extensions
        if all_files:
            extensions = set()
            for f in all_files:
                if "." in f:
                    extensions.add(f.split(".")[-1])
            if extensions:
                parts.append(f"Files: {', '.join(sorted(extensions))}")

        # Add diff patterns for additional context
        diff = issue.get("diff", "") or issue.get("patch", "")
        if diff:
            diff_patterns = _extract_diff_patterns(diff)
            if diff_patterns:
                parts.append(f"Diff: {diff_patterns}")

    else:
        # FALLBACK: Use raw issue data (original method)
        # 1. CI VALIDATIONS from workflow
        if issue.get("validation_cmd"):
            parts.append(f"Validation: {issue['validation_cmd']}")

        # Failure type
        if issue.get("failure_type"):
            parts.append(f"Failure: {issue['failure_type']}")

        # 2. CI LOG INFO (error messages)
        logs = issue.get("logs", "")
        if isinstance(logs, str):
            error_tail = logs[-500:] if logs else ""
            if error_tail:
                parts.append(f"Error: {error_tail}")
        elif isinstance(logs, list):
            if logs:
                last_log = logs[-1]
                if isinstance(last_log, dict):
                    error_tail = last_log.get("log", "")[-500:]
                    if error_tail:
                        parts.append(f"Error: {error_tail}")

        # 3. DIFF (ground truth - what was actually changed)
        diff = issue.get("diff", "") or issue.get("patch", "")
        if diff:
            diff_patterns = _extract_diff_patterns(diff)
            if diff_patterns:
                parts.append(f"Diff: {diff_patterns}")

        # Files (extract extensions to find pattern)
        files = issue.get("files", [])
        if files:
            extensions = set()
            for f in files:
                if "." in f:
                    ext = f.split(".")[-1]
                    extensions.add(ext)
            if extensions:
                parts.append(f"Files: {', '.join(sorted(extensions))}")

    return " | ".join(parts)


def _compute_embeddings(
    issues: List[Dict[str, Any]],
    decomposed_dict: Dict[str, Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Compute embeddings for all issues using decomposed data if available.

    Args:
        issues: List of issues
        decomposed_dict: {issue_id: decomposed_data}
        model_name: Sentence transformer model name

    Returns:
        embeddings: (N, embedding_dim) array
        model: SentenceTransformer model
    """
    console.print(f"[cyan]Loading embedding model: {model_name}[/cyan]")
    model = SentenceTransformer(model_name)

    # Build text for each issue using decomposed data if available
    texts = []
    decomposed_count = 0
    raw_count = 0

    for issue in track(issues, description="Building issue texts"):
        issue_id = str(issue.get("instance_id") or issue.get("id") or "")
        decomposed = decomposed_dict.get(issue_id) if issue_id else None

        if decomposed:
            decomposed_count += 1
        else:
            raw_count += 1

        text = _build_issue_text_from_decomposed(issue, decomposed)
        texts.append(text)

    console.print(f"[green]Using decomposed data: {decomposed_count}/{len(issues)} issues[/green]")
    if raw_count > 0:
        console.print(f"[yellow]Using raw data: {raw_count}/{len(issues)} issues[/yellow]")

    # Compute embeddings
    console.print("[cyan]Computing embeddings...[/cyan]")
    embeddings = model.encode(texts, show_progress_bar=True)

    logger.info(f"Computed embeddings: {embeddings.shape}")
    return embeddings, model


def _group_by_repo(issues: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Group issue indices by repository.

    Returns:
        {repo_name: [issue_idx1, issue_idx2, ...]}
    """
    repo_groups = defaultdict(list)
    for idx, issue in enumerate(issues):
        repo_owner = issue.get("repo_owner", "unknown")
        repo_name = issue.get("repo_name", "unknown")
        repo_key = f"{repo_owner}/{repo_name}"
        repo_groups[repo_key].append(idx)

    logger.info(f"Grouped into {len(repo_groups)} repositories")
    for repo, indices in sorted(repo_groups.items()):
        logger.info(f"  {repo}: {len(indices)} issues")

    return repo_groups


def _compute_similarity_within_repo(
    embeddings: np.ndarray,
    repo_indices: List[int]
) -> np.ndarray:
    """
    Compute pairwise cosine similarity for issues within a repo.

    Returns:
        similarity_matrix: (N, N) where N = len(repo_indices)
    """
    repo_embeddings = embeddings[repo_indices]
    similarity = cosine_similarity(repo_embeddings)
    return similarity


def _select_representative_issues(
    similarity_matrix: np.ndarray,
    repo_indices: List[int],
    memory_ratio: float = 0.3
) -> Tuple[List[int], List[int]]:
    """
    Select most representative issues for memory set.

    Strategy:
    1. Compute average similarity for each issue (how similar to others)
    2. Select top memory_ratio% with highest avg similarity
    3. These are "representative" issues that capture common patterns

    Returns:
        memory_indices: Issue indices for memory set
        eval_indices: Issue indices for evaluation set
    """
    n_issues = len(repo_indices)
    n_memory = max(1, int(n_issues * memory_ratio))

    # Compute average similarity for each issue
    avg_similarity = similarity_matrix.mean(axis=1)

    # Sort by average similarity (descending)
    sorted_local_indices = np.argsort(avg_similarity)[::-1]

    # Select top n_memory as memory set
    memory_local = sorted_local_indices[:n_memory]
    eval_local = sorted_local_indices[n_memory:]

    # Convert local indices back to global indices
    memory_indices = [repo_indices[i] for i in memory_local]
    eval_indices = [repo_indices[i] for i in eval_local]

    return memory_indices, eval_indices


def _save_split_datasets(
    issues: List[Dict[str, Any]],
    memory_indices: List[int],
    eval_indices: List[int],
    output_dir: Path
):
    """Save memory and evaluation datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Memory set
    memory_path = output_dir / "memory_set.jsonl"
    with open(memory_path, "w") as f:
        for idx in memory_indices:
            f.write(json.dumps(issues[idx]) + "\n")
    logger.info(f"Saved {len(memory_indices)} issues to {memory_path}")

    # Evaluation set
    eval_path = output_dir / "eval_set.jsonl"
    with open(eval_path, "w") as f:
        for idx in eval_indices:
            f.write(json.dumps(issues[idx]) + "\n")
    logger.info(f"Saved {len(eval_indices)} issues to {eval_path}")

    # Metadata
    metadata = {
        "total_issues": len(issues),
        "memory_set_size": len(memory_indices),
        "eval_set_size": len(eval_indices),
        "memory_ratio": len(memory_indices) / len(issues),
        "split_by_repo": True,
        "selection_strategy": "highest_avg_similarity"
    }
    metadata_path = output_dir / "split_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def _save_similarity_analysis(
    issues: List[Dict[str, Any]],
    embeddings: np.ndarray,
    repo_groups: Dict[str, List[int]],
    memory_indices: List[int],
    eval_indices: List[int],
    output_dir: Path
):
    """Save similarity analysis for inspection."""
    analysis = {
        "repositories": {}
    }

    for repo_key, repo_indices in repo_groups.items():
        similarity_matrix = _compute_similarity_within_repo(embeddings, repo_indices)

        # Compute stats
        avg_sim = similarity_matrix.mean()
        max_sim = similarity_matrix.max()
        min_sim = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].min()

        # Count memory vs eval
        memory_count = sum(1 for idx in repo_indices if idx in memory_indices)
        eval_count = sum(1 for idx in repo_indices if idx in eval_indices)

        analysis["repositories"][repo_key] = {
            "total_issues": len(repo_indices),
            "memory_set": memory_count,
            "eval_set": eval_count,
            "avg_similarity": float(avg_sim),
            "max_similarity": float(max_sim),
            "min_similarity": float(min_sim)
        }

    analysis_path = output_dir / "similarity_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Saved similarity analysis to {analysis_path}")


@app.command()
def main(
    dataset: Optional[Path] = typer.Option(None, help="Path to local dataset JSONL file (optional, uses HuggingFace if not provided)"),
    output_dir: Path = typer.Option("data/trs", help="Output directory for all memory-related data"),
    memory_ratio: float = typer.Option(0.3, help="Ratio of issues to use for memory (0.0-1.0)"),
    similarity_threshold: float = typer.Option(0.5, help="Minimum similarity for grouping (unused for now)"),
    embedding_model: str = typer.Option("all-MiniLM-L6-v2", help="Sentence transformer model"),
    repos: str = typer.Option(None, help="Filter by repo names (comma-separated, e.g., 'flower,agno'). If not specified, processes all repos."),
    hf_dataset: str = typer.Option("ci-benchmark-user/ci-repair-bench", help="HuggingFace dataset name (used if --dataset not provided)"),
):
    """
    Analyze CI repair dataset and split into memory/evaluation sets based on similarity.

    Process:
    1. Load all issues from dataset
    2. Compute embeddings for each issue
    3. Group by repository
    4. For each repo, compute similarity and select most representative issues for memory
    5. Save memory set (30%) and evaluation set (70%)
    """
    console.print("[bold cyan]Starting Memory/Eval Split Analysis[/bold cyan]")
    if dataset:
        console.print(f"Dataset: {dataset} (local file)")
    else:
        console.print(f"Dataset: {hf_dataset} (HuggingFace)")
    console.print(f"Output: {output_dir}")
    console.print(f"Memory ratio: {memory_ratio:.1%}")
    console.print()

    # 1. Load dataset (HuggingFace by default, or local JSONL if provided)
    if dataset:
        all_issues = _load_from_jsonl(dataset)
    else:
        all_issues = _load_from_huggingface(hf_dataset)

    # 1.5. Load decomposed issues (REUSE existing decomposition!)
    decomposed_cache_path = output_dir / "decomposed_issues.json"
    console.print(f"\n[cyan]Checking for decomposed issues...[/cyan]")
    decomposed_dict = _load_or_create_decomposed_issues(
        all_issues,
        decomposed_cache_path,
        force_recompute=False
    )

    # Filter by repos if specified
    if repos:
        repo_filter = [r.strip().lower() for r in repos.split(",")]
        console.print(f"[yellow]Filtering by repos: {', '.join(repo_filter)}[/yellow]")

        issues = []
        for issue in all_issues:
            repo_owner = issue.get("repo_owner", "").lower()
            repo_name = issue.get("repo_name", "").lower()
            repo_key = f"{repo_owner}/{repo_name}"

            # Check if repo_name matches any filter (partial match)
            if any(filter_repo in repo_name or filter_repo in repo_key for filter_repo in repo_filter):
                issues.append(issue)

        console.print(f"[green]Filtered: {len(issues)} issues (from {len(all_issues)} total)[/green]\n")

        if len(issues) == 0:
            console.print("[bold red]No issues found matching the repo filter![/bold red]")
            console.print(f"Available repos in dataset:")
            repos_in_dataset = set()
            for issue in all_issues:
                repo_owner = issue.get("repo_owner", "unknown")
                repo_name = issue.get("repo_name", "unknown")
                repos_in_dataset.add(f"{repo_owner}/{repo_name}")
            for repo in sorted(repos_in_dataset):
                console.print(f"  - {repo}")
            return
    else:
        issues = all_issues

    # 2. Compute embeddings (using decomposed data if available)
    embeddings, model = _compute_embeddings(issues, decomposed_dict, embedding_model)

    # 3. Group by repository
    repo_groups = _group_by_repo(issues)

    # 4. Select memory vs eval for each repo
    all_memory_indices = []
    all_eval_indices = []

    console.print("\n[cyan]Selecting representative issues per repository:[/cyan]")
    for repo_key, repo_indices in sorted(repo_groups.items()):
        # Compute similarity within this repo
        similarity_matrix = _compute_similarity_within_repo(embeddings, repo_indices)

        # Select memory vs eval
        memory_indices, eval_indices = _select_representative_issues(
            similarity_matrix,
            repo_indices,
            memory_ratio
        )

        all_memory_indices.extend(memory_indices)
        all_eval_indices.extend(eval_indices)

        console.print(
            f"  {repo_key}: "
            f"{len(memory_indices)} memory, "
            f"{len(eval_indices)} eval "
            f"(avg_sim={similarity_matrix.mean():.3f})"
        )

    # 5. Save datasets
    console.print("\n[cyan]Saving split datasets...[/cyan]")
    _save_split_datasets(issues, all_memory_indices, all_eval_indices, output_dir)

    # 6. Save similarity analysis
    _save_similarity_analysis(
        issues,
        embeddings,
        repo_groups,
        set(all_memory_indices),
        set(all_eval_indices),
        output_dir
    )

    # Summary
    console.print("\n[bold green]Split Complete![/bold green]")
    console.print(f"Total issues: {len(issues)}")
    console.print(f"Memory set: {len(all_memory_indices)} ({len(all_memory_indices)/len(issues):.1%})")
    console.print(f"Eval set: {len(all_eval_indices)} ({len(all_eval_indices)/len(issues):.1%})")
    console.print(f"\nNext steps:")
    console.print(f"1. Run memory set through agent to generate L2 trajectories")
    console.print(f"2. Save L2 trajectories to memory system")
    console.print(f"3. Evaluate on eval set using memory from step 2")


if __name__ == "__main__":
    app()
