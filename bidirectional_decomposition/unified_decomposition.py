"""
Simple unified bidirectional decomposition.

Clean, simple approach:
1. Load forward + backward decompositions
2. LLM-based reconciliation (clustering, merging, selection)
3. Normalize and save output
"""

from pathlib import Path
from typing import Any, Dict
from functools import lru_cache
import json
import subprocess
import sys

from bidirectional_decomposition.simple_llm_reconciliation import simple_bidirectional_reconciliation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORWARD_CACHE = PROJECT_ROOT / "data/fwr_trs/decomposed_issues.json"
BACKWARD_CACHE = PROJECT_ROOT / "data/back_trs/decomposed_issues.json"


def build_unified_decomposition(
    issue_id: str,
    llm: Any,
    output_dir: str = "data/bidirect_trs"
) -> Dict[str, Any]:
    """
    Main entry point for unified bidirectional decomposition.

    Args:
        issue_id: Issue to decompose
        llm: LLM instance
        output_dir: Output directory

    Returns:
        Unified decomposition result
    """
    print("\n" + "=" * 80)
    print(f"UNIFIED BIDIRECTIONAL DECOMPOSITION: Issue {issue_id}")
    print("=" * 80)

    # Step 1: Load forward decomposition
    print("\nStep 1: Forward decomposition (commit-based)")
    model_name = str(
        getattr(llm, "memci_model_key", "")
        or getattr(llm, "model_name", "")
        or getattr(llm, "model", "")
        or "minimax2.5"
    )
    forward_result = load_forward_decomposition(issue_id, model_name=model_name)
    forward_problems = forward_result.get("problems", [])
    print(f"  → {len(forward_problems)} problems")

    # Step 2: Load backward decomposition
    print("\nStep 2: Backward decomposition (CI-based)")
    backward_result = load_backward_decomposition(issue_id, model_name=model_name)
    backward_problems = backward_result.get("problems", [])
    ci_context = backward_result.get("benchmark_ci_context", {})

    # Enrich ci_context with failed_jobs from dataset for CI verification
    if not ci_context.get("failed_jobs"):
        dataset_row = _dataset_metadata_by_issue().get(str(issue_id), {})
        dataset_failed_jobs = dataset_row.get("failed_jobs", [])
        if dataset_failed_jobs:
            ci_context["failed_jobs"] = dataset_failed_jobs
            print(f"  → Enriched ci_context with {len(dataset_failed_jobs)} failed jobs from dataset")

    print(f"  → {len(backward_problems)} problems")
    print(f"  → {len(ci_context.get('failed_jobs', []))} failed jobs")

    # Step 3: LLM reconciliation with graph-based dependency analysis
    print("\nStep 3: LLM-based reconciliation")

    # Extract graph info for dependency analysis
    dependency_graph = backward_result.get("_dependency_graph") or {}
    structured_diff = backward_result.get("_structured_diff") or {}
    repo_path = ci_context.get("repo_path")

    # Cached decompositions may omit graph artifacts. Historical memory rows
    # retain the successful diff, so reconstruct deterministic file evidence.
    if not dependency_graph:
        dataset_row = _dataset_metadata_by_issue().get(str(issue_id), {})
        raw_diff = dataset_row.get("diff") or ""
        if raw_diff:
            try:
                from utilities.deterministic_diff_parser import parse_diff_to_structured
                from utilities.dependency_evidence import (
                    build_dependency_graph_from_structured_diff,
                )

                structured_diff = parse_diff_to_structured(raw_diff)
                dependency_graph = build_dependency_graph_from_structured_diff(
                    structured_diff, repo_path=repo_path
                )
            except Exception as exc:
                print(f"  ⚠ Could not reconstruct dependency graph: {exc}")

    if not structured_diff:
        structured_diff = {"changed_files": forward_result.get("changed_files", [])}

    if dependency_graph:
        print(
            "  ✓ Using graph-based dependency analysis "
            f"({len(dependency_graph.get('nodes', {}))} files, "
            f"{len(dependency_graph.get('edges', []))} edges)"
        )
    else:
        print("  ℹ No precomputed graph; inferring dependencies from current problem/file evidence")

    result = simple_bidirectional_reconciliation(
        forward_problems=forward_problems,
        backward_problems=backward_problems,
        ci_context=ci_context,
        llm=llm,
        dependency_graph=dependency_graph,
        structured_diff=structured_diff,
        repo_path=repo_path
    )

    # Step 4: Normalize output
    print("\nStep 4: Normalizing output")
    normalized = normalize_result(result, issue_id, backward_result, forward_result)

    # Step 5: Save
    print("\nStep 5: Saving")
    save_to_decomposed_issues(normalized, Path(output_dir))

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)

    return normalized


def load_forward_decomposition(
    issue_id: str, model_name: str = "minimax2.5"
) -> Dict[str, Any]:
    """Load forward data, generating the requested issue when it is missing."""
    result = _cached_issue(FORWARD_CACHE, issue_id)
    if not _complete_decomposition(result):
        reason = "cache file absent" if not FORWARD_CACHE.exists() else "issue absent or incomplete"
        print(f"  Warning: Forward decomposition {reason}; generating issue {issue_id}...")
        run_forward_decomposition(issue_id, model_name=model_name)
        result = _cached_issue(FORWARD_CACHE, issue_id)

    if not _complete_decomposition(result):
        raise RuntimeError(
            f"Forward decomposition for issue {issue_id} is still missing or empty "
            f"after generation; expected it in {FORWARD_CACHE}"
        )
    return result


def load_backward_decomposition(
    issue_id: str, model_name: str = "minimax2.5"
) -> Dict[str, Any]:
    """Load backward data, generating the requested issue when it is missing."""
    result = _cached_issue(BACKWARD_CACHE, issue_id)
    if not _complete_decomposition(result):
        reason = "cache file absent" if not BACKWARD_CACHE.exists() else "issue absent or incomplete"
        print(f"  Warning: Backward decomposition {reason}; generating issue {issue_id}...")
        run_backward_decomposition(issue_id, model_name=model_name)
        result = _cached_issue(BACKWARD_CACHE, issue_id)

    if not _complete_decomposition(result):
        raise RuntimeError(
            f"Backward decomposition for issue {issue_id} is still missing or empty "
            f"after generation; expected it in {BACKWARD_CACHE}"
        )
    return result


def _cached_issue(path: Path, issue_id: str) -> Dict[str, Any]:
    """Return one issue from a JSON cache without treating file presence as a hit."""
    if not path.exists():
        return {}
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  Warning: Could not read decomposition cache {path}: {exc}")
        return {}
    records = data if isinstance(data, list) else [data]
    for item in records:
        if not isinstance(item, dict):
            continue
        item_id = item.get("original_issue_id") or item.get("issue_id")
        if str(item_id) == str(issue_id):
            return item
    return {}


def _complete_decomposition(result: Dict[str, Any]) -> bool:
    """A usable side must exist, be error-free, and contain problem records."""
    return bool(
        isinstance(result, dict)
        and not result.get("error")
        and isinstance(result.get("problems"), list)
        and result.get("problems")
    )


def _dataset_path_for_issue(issue_id: str) -> Path:
    """Select the local split containing the issue so generation is reproducible."""
    for path in (
        PROJECT_ROOT / "data/memory_set.jsonl",
        PROJECT_ROOT / "data/eval_set.jsonl",
    ):
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = row.get("id") or row.get("issue_id")
                if str(row_id) == str(issue_id):
                    return path
    raise RuntimeError(
        f"Issue {issue_id} is not present in data/memory_set.jsonl or data/eval_set.jsonl"
    )


def run_forward_decomposition(issue_id: str, model_name: str = "minimax2.5"):
    """Generate and cache one forward decomposition, failing loudly on errors."""
    dataset_path = _dataset_path_for_issue(issue_id)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/decompose_commits.py"),
            "--batch",
            "--dataset",
            str(dataset_path),
            "--model",
            model_name,
            "--issue-id",
            str(issue_id),
            "--output-dir",
            str(FORWARD_CACHE.parent),
            "--force-recompute",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_backward_decomposition(issue_id: str, model_name: str = "minimax2.5"):
    """Generate and cache one backward decomposition, failing loudly on errors."""
    dataset_path = _dataset_path_for_issue(issue_id)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/decompose_backward.py"),
            "--model",
            model_name,
            "--dataset",
            str(dataset_path),
            "--issue-id",
            str(issue_id),
            "--output-dir",
            str(BACKWARD_CACHE.parent),
            "--skip-memory",
            "--force-recompute",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def normalize_result(
    result: Dict[str, Any],
    issue_id: str,
    backward_result: Dict[str, Any],
    forward_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Normalize to standard format.
    """
    from bidirectional_decomposition.normalize_output import normalize_bidirectional_output

    metadata = resolve_issue_metadata(
        issue_id, forward_result or {}, backward_result
    )

    # Add canonical dataset metadata.
    result["original_issue_id"] = issue_id
    result["issue_id"] = issue_id
    result["repo"] = metadata["repo"]
    result["repo_owner"] = metadata["repo_owner"]
    result["repo_name"] = metadata["repo_name"]
    result["workflow_path"] = metadata["workflow_path"]
    result["workflow_name"] = metadata["workflow_name"]
    result["sha_fail"] = backward_result.get("sha_fail", "")
    result["benchmark_ci_context"] = backward_result.get("benchmark_ci_context", {})
    result["changed_files"] = (forward_result or {}).get("changed_files", [])
    result["total_changed_files"] = len(result["changed_files"])

    # Normalize
    normalized = normalize_bidirectional_output(result)

    return normalized


@lru_cache(maxsize=1)
def _dataset_metadata_by_issue() -> Dict[str, Dict[str, Any]]:
    """Load canonical metadata once for the full decomposition batch."""
    records: Dict[str, Dict[str, Any]] = {}
    for dataset_path in (Path("data/memory_set.jsonl"), Path("data/eval_set.jsonl")):
        if not dataset_path.exists():
            continue
        with dataset_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                issue_id = str(row.get("id") or row.get("issue_id") or "")
                if issue_id:
                    records[issue_id] = row
    return records


def resolve_issue_metadata(
    issue_id: str,
    forward_result: Dict[str, Any],
    backward_result: Dict[str, Any],
) -> Dict[str, str]:
    """Resolve workflow/repository fields, preferring the canonical dataset row."""
    dataset = _dataset_metadata_by_issue().get(str(issue_id), {})
    ci_context = backward_result.get("benchmark_ci_context") or {}

    def first(*values: Any) -> str:
        return next((str(value) for value in values if value not in (None, "")), "")

    repo_owner = first(
        dataset.get("repo_owner"),
        forward_result.get("repo_owner"),
        backward_result.get("repo_owner"),
    )
    repo_name = first(
        dataset.get("repo_name"),
        forward_result.get("repo_name"),
        backward_result.get("repo_name"),
    )
    repo = first(
        dataset.get("repo"),
        f"{repo_owner}/{repo_name}" if repo_owner and repo_name else "",
        forward_result.get("repo"),
        backward_result.get("repo"),
        repo_name,
    )
    workflow_path = first(
        dataset.get("workflow_path"),
        forward_result.get("workflow_path"),
        backward_result.get("workflow_path"),
        ci_context.get("workflow_path"),
    )
    workflow_name = first(
        dataset.get("workflow_name"),
        forward_result.get("workflow_name"),
        backward_result.get("workflow_name"),
        ci_context.get("workflow_name"),
        Path(workflow_path).name if workflow_path else "",
    )
    return {
        "repo": repo,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "workflow_path": workflow_path,
        "workflow_name": workflow_name,
    }


def save_to_decomposed_issues(result: Dict[str, Any], output_dir: Path) -> Path:
    """
    Save to decomposed_issues.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "decomposed_issues.json"

    issue_id = result.get("original_issue_id", "unknown")

    # Load existing
    existing = []
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]

    # Update or add
    found = False
    for i, item in enumerate(existing):
        if item.get("original_issue_id") == issue_id or item.get("issue_id") == issue_id:
            existing[i] = result
            found = True
            break

    if not found:
        existing.append(result)

    # Save
    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Saved to {output_file}")

    return output_file


# For backward compatibility
def reconcile_bidirectional(
    issue_id: str,
    forward_result: Dict[str, Any],
    backward_result: Dict[str, Any],
    llm: Any,
    use_comprehensive: bool = False
) -> Dict[str, Any]:
    """
    Backward compatibility wrapper.
    """
    forward_problems = forward_result.get("problems", [])
    backward_problems = backward_result.get("problems", [])
    ci_context = backward_result.get("benchmark_ci_context", {})

    from bidirectional_decomposition.simple_llm_reconciliation import simple_bidirectional_reconciliation

    result = simple_bidirectional_reconciliation(
        forward_problems=forward_problems,
        backward_problems=backward_problems,
        ci_context=ci_context,
        llm=llm
    )

    # Normalize
    result["original_issue_id"] = issue_id
    result["issue_id"] = issue_id
    result["repo"] = backward_result.get("repo", "")
    result.update(resolve_issue_metadata(issue_id, forward_result, backward_result))

    from bidirectional_decomposition.normalize_output import normalize_bidirectional_output
    normalized = normalize_bidirectional_output(result)

    return normalized
