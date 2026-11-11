#!/usr/bin/env python3
"""Build the Whole-Repair-History control used for reviewer jcJU Q1.

Each historical fail-to-pass PR remains one indivisible episode. This module
does not call ReCAP problem decomposition, problem/hunk attribution,
reconciliation, dependency, or problem-ordering code.

One historical PR produces exactly one L1 issue record (with one aggregate
``problems`` item), one L2 issue record (with one aggregate strategy), and one
flat L3 transferable pattern. Familiar schemas are retained so the existing
retrieval and repair pipeline can consume the control unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from utilities.llm_invoker import (  # noqa: E402
    STRICT_JSON_RULES,
    invoke_llm_with_retry,
)
from utilities.llm_model import LitellmModel  # noqa: E402
from utilities.model_registry import configure_model_environment  # noqa: E402
from utilities.dependency_evidence import (  # noqa: E402
    build_dependency_graph_from_structured_diff,
)
from utilities.deterministic_diff_parser import (  # noqa: E402
    chunk_structured_diff,
    parse_diff_to_structured,
)


DEFAULT_MEMORY_IDS = PROJECT_ROOT / "data" / "memory_issue_ids.json"
DEFAULT_LOG_DETAILS = PROJECT_ROOT / "data" / "log_details.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experimental_data" / "whole_repair_history"
DEFAULT_DATASET = "ci-benchmark-user/ci-repair-bench"


def issue_id(record: dict[str, Any]) -> str:
    """Return the canonical string ID used throughout the repository."""
    return str(
        record.get("id") or record.get("instance_id") or record.get("issue_id") or ""
    ).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _repo_parts(record: dict[str, Any]) -> tuple[str, str, str]:
    owner = str(record.get("repo_owner") or "").strip()
    name = str(record.get("repo_name") or "").strip()
    repo = str(record.get("repo") or "").strip()
    if repo and "/" in repo:
        owner = owner or repo.split("/", 1)[0]
        name = name or repo.split("/", 1)[1]
    repo = repo or "/".join(part for part in (owner, name) if part)
    return repo, owner, name


def load_id_list(path: Path) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"Expected a JSON array in {path}")
    ids = [str(value).strip() for value in values if str(value).strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found in {path}")
    return ids


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def load_benchmark_records(
    *, dataset_path: Path | None, dataset_name: str, revision: str | None
) -> tuple[list[dict[str, Any]], str | None]:
    """Load raw benchmark rows, never existing ReCAP memory artifacts."""
    if dataset_path is not None:
        return load_jsonl(dataset_path), revision

    # Prefer an already materialized Arrow cache. Besides avoiding unnecessary
    # network access, Dataset.from_file is read-only and therefore does not
    # require a writable Hugging Face cache lock.
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / dataset_name.replace("/", "___")
        / "default"
        / "0.0.0"
    )
    cached_candidates: list[Path] = []
    if cache_root.exists():
        for candidate in cache_root.iterdir():
            if not candidate.is_dir() or not (candidate / "dataset_info.json").exists():
                continue
            if revision and candidate.name != revision:
                continue
            if list(candidate.glob("*.arrow")):
                cached_candidates.append(candidate)
    if cached_candidates:
        from datasets import Dataset, concatenate_datasets

        selected = max(
            cached_candidates,
            key=lambda path: (path / "dataset_info.json").stat().st_mtime,
        )
        shards = [
            Dataset.from_file(str(path)) for path in sorted(selected.glob("*.arrow"))
        ]
        dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        print(f"Using cached raw benchmark revision: {selected.name}")
        return [dict(row) for row in dataset], selected.name

    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": "train"}
    if revision:
        kwargs["revision"] = revision
    dataset = load_dataset(dataset_name, **kwargs)
    return [dict(row) for row in dataset], revision


def load_structured_failures(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Index log-only CI analysis by issue ID and failing SHA."""
    if not path.exists():
        return {}, {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
    by_id: dict[str, dict] = {}
    by_sha: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or "").strip()
        sha = str(row.get("sha_fail") or "").strip()
        if row_id:
            by_id[row_id] = row
        if sha:
            by_sha[sha] = row
    return by_id, by_sha


def collect_whole_pr_episode(
    record: dict[str, Any],
    *,
    structured_failure: dict[str, Any] | None,
    allow_incomplete: bool,
) -> dict[str, Any]:
    """Collect one benchmark-only episode without decomposing or attributing it."""
    iid = issue_id(record)
    repo, repo_owner, repo_name = _repo_parts(record)
    sha_fail = str(record.get("sha_fail") or "").strip()
    sha_success = str(record.get("sha_success") or "").strip()
    cumulative_patch = str(record.get("diff") or record.get("patch") or "")
    changed_files = _string_list(record.get("changed_files"))

    workflow_path = str(
        record.get("workflow_path") or record.get("workflow_filename") or ""
    )
    workflow_name = str(
        record.get("workflow_name")
        or record.get("workflow_filename")
        or Path(workflow_path).name
    )
    raw_logs = _as_list(record.get("logs"))
    raw_logs_encoded = json.dumps(raw_logs, ensure_ascii=False, default=str)
    has_structured_failure = bool(structured_failure)
    initial_failure = {
        "sha": sha_fail,
        "ci_evidence_source": (
            "structured_log_analysis" if has_structured_failure else "raw_ci_logs"
        ),
        # Structured log analysis is the same standardized evidence used by
        # the existing decomposition pipeline. Keep raw logs only when that
        # analysis is unavailable; otherwise retain an auditable fingerprint.
        "raw_ci_logs": [] if has_structured_failure else raw_logs,
        "raw_ci_logs_audit": {
            "benchmark_issue_id": iid,
            "log_record_count": len(raw_logs),
            "characters": len(raw_logs_encoded),
            "sha256": hashlib.sha256(raw_logs_encoded.encode("utf-8")).hexdigest(),
            "content_stored_here": not has_structured_failure,
        },
        "structured_ci_failure": structured_failure or {},
        "error_types": _string_list(
            record.get("failure_types") or record.get("error_type")
        ),
        "failed_jobs": _as_list(record.get("failed_jobs")),
        "total_jobs": record.get("total_jobs"),
        "total_failed_jobs": record.get("total_failed_jobs"),
        "total_failed_steps": record.get("total_failed_steps"),
    }
    workflow_metadata = {
        "path": workflow_path,
        "name": workflow_name,
        "definition": str(record.get("workflow") or ""),
        "head_branch": str(record.get("head_branch") or ""),
        "language": str(record.get("language") or ""),
    }
    final_outcome = {
        "sha": sha_success,
        "conclusion": "success",
        "evidence_source": "benchmark fail-to-pass pair",
    }
    repair_transition = {
        "scope": "whole_pr_cumulative_patch",
        "sha_fail": sha_fail,
        "sha_success": sha_success,
        "commit_date": str(record.get("commit_date") or ""),
        "commit_link": str(record.get("commit_link") or ""),
        "head_branch": str(record.get("head_branch") or ""),
    }
    completeness = {
        "initial_ci_evidence": bool(raw_logs or structured_failure),
        "repository_metadata": bool(repo and workflow_path),
        "repair_transition": bool(sha_fail and sha_success),
        "cumulative_patch": bool(cumulative_patch),
        "changed_files": bool(changed_files),
        "successful_outcome": bool(sha_success),
    }
    missing = [name for name, present in completeness.items() if not present]
    if missing and not allow_incomplete:
        raise ValueError(
            f"Issue {iid} is incomplete ({', '.join(missing)}). "
            "Reviewer-grade memory requires all benchmark episode fields; use "
            "--allow-incomplete only for debugging."
        )

    return {
        "evidence_policy": "benchmark_only",
        "episode_schema_version": 2,
        "issue_id": iid,
        "repo": repo,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "workflow": workflow_path,
        "workflow_path": workflow_path,
        "workflow_name": workflow_name,
        "sha_fail": sha_fail,
        "sha_success": sha_success,
        "commit_link": str(record.get("commit_link") or ""),
        "initial_failure": initial_failure,
        "workflow_metadata": workflow_metadata,
        "repair_transition": repair_transition,
        "cumulative_patch": cumulative_patch,
        "all_changed_files": changed_files,
        "final_ci_outcome": final_outcome,
        "completeness": completeness,
    }


def _whole_pr_evidence(episode: dict[str, Any]) -> dict[str, Any]:
    """Return the complete evidence used by the whole-PR abstraction."""
    return {
        "issue_id": episode["issue_id"],
        "repo": episode["repo"],
        "workflow_metadata": episode["workflow_metadata"],
        "initial_failure": episode["initial_failure"],
        "repair_transition": episode["repair_transition"],
        "cumulative_patch": episode["cumulative_patch"],
        "all_changed_files": episode["all_changed_files"],
        "final_ci_outcome": episode["final_ci_outcome"],
    }


def _global_whole_pr_context(episode: dict[str, Any]) -> dict[str, Any]:
    """Small global context repeated across chunk prompts."""
    initial = episode["initial_failure"]
    workflow = episode["workflow_metadata"]
    return {
        "issue_id": episode["issue_id"],
        "repo": episode["repo"],
        "workflow": {
            "path": workflow.get("path"),
            "name": workflow.get("name"),
            "language": workflow.get("language"),
        },
        "failure_types": initial.get("error_types", []),
        "failed_job_count": len(initial.get("failed_jobs") or []),
        "repair_transition": episode["repair_transition"],
        "all_changed_files": episode["all_changed_files"],
        "final_ci_outcome": episode["final_ci_outcome"],
    }


def _split_text_preserving_lines(text: str, max_chars: int) -> list[str]:
    """Split text losslessly at line boundaries, then hard-split huge lines."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    current_size = 0
    for line in text.splitlines(keepends=True):
        if len(line) > max_chars:
            if current:
                parts.append("".join(current))
                current, current_size = [], 0
            parts.extend(
                line[offset : offset + max_chars]
                for offset in range(0, len(line), max_chars)
            )
            continue
        if current and current_size + len(line) > max_chars:
            parts.append("".join(current))
            current, current_size = [], 0
        current.append(line)
        current_size += len(line)
    if current:
        parts.append("".join(current))
    return parts or [""]


def _dependency_descriptor(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep graph topology without repeating before/after payloads."""
    descriptors = []
    for context in contexts:
        caller = context.get("caller") or {}
        callees = context.get("callees") or []
        descriptors.append(
            {
                "dependency_type": context.get("dependency_type", "unknown"),
                "caller": caller.get("file"),
                "callees": [
                    callee.get("file") for callee in callees if callee.get("file")
                ],
            }
        )
    return descriptors


def _dependency_aware_patch_payloads(
    episode: dict[str, Any], *, model_name: str, payload_chars: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse and graph-chunk the entire patch without dropping a file or hunk."""
    structured = parse_diff_to_structured(episode["cumulative_patch"])
    files = structured.get("files") or []
    if not files:
        raise ValueError(
            f"Issue {episode['issue_id']}: cumulative patch could not be parsed"
        )
    graph = build_dependency_graph_from_structured_diff(structured, repo_path=None)
    graph_chunks = chunk_structured_diff(
        structured,
        dependency_graph=graph,
        model_name=model_name,
    )
    if not graph_chunks:
        raise ValueError(
            f"Issue {episode['issue_id']}: dependency-aware chunking returned no chunks"
        )

    expected_files = {str(item.get("path")) for item in files if item.get("path")}
    covered_files = {
        str(item.get("path"))
        for chunk in graph_chunks
        for item in chunk.get("files", [])
        if item.get("path")
    }
    expected_changes = sum(len(item.get("changes") or []) for item in files)
    covered_change_keys = {
        (
            str(item.get("path")),
            json.dumps(change, sort_keys=True, ensure_ascii=False),
        )
        for chunk in graph_chunks
        for item in chunk.get("files", [])
        for change in item.get("changes", [])
    }
    expected_change_keys = {
        (
            str(item.get("path")),
            json.dumps(change, sort_keys=True, ensure_ascii=False),
        )
        for item in files
        for change in item.get("changes", [])
    }
    missing_files = expected_files - covered_files
    missing_change_keys = expected_change_keys - covered_change_keys
    recovery_applied = bool(missing_files or missing_change_keys)
    recovery_change_count = len(missing_change_keys)
    if recovery_applied:
        # The shared graph chunker prioritizes caller continuity when a single
        # dependency component is enormous. A defensive recovery chunk makes
        # whole-PR coverage lossless even if that split omits a callee or hunk.
        recovery_files = []
        for item in files:
            path = str(item.get("path") or "")
            selected_changes = [
                change
                for change in item.get("changes", [])
                if (
                    path,
                    json.dumps(change, sort_keys=True, ensure_ascii=False),
                )
                in missing_change_keys
            ]
            if path in missing_files or selected_changes:
                recovered = dict(item)
                if path not in missing_files:
                    recovered["changes"] = selected_changes
                    recovered["total_changes"] = len(selected_changes)
                recovery_files.append(recovered)
        recovery_paths = {
            str(item.get("path")) for item in recovery_files if item.get("path")
        }
        recovery_contexts = [
            {
                "dependency_type": edge.get("type", "unknown"),
                "caller": {"file": edge.get("from")},
                "callees": [{"file": edge.get("to")}],
            }
            for edge in graph.get("edges", [])
            if edge.get("from") in recovery_paths or edge.get("to") in recovery_paths
        ]
        graph_chunks.append(
            {
                "files": recovery_files,
                "dependency_cluster": sorted(recovery_paths),
                "dependency_contexts": recovery_contexts,
                "chunk_info": "Whole-PR lossless coverage recovery",
                "is_coverage_recovery": True,
            }
        )
        covered_files.update(recovery_paths)
        covered_change_keys.update(missing_change_keys)

    if covered_files != expected_files or not expected_change_keys.issubset(
        covered_change_keys
    ):
        raise ValueError(
            f"Issue {episode['issue_id']}: unrecoverable chunk coverage failure"
        )

    payloads: list[dict[str, Any]] = []
    for graph_index, chunk in enumerate(graph_chunks, 1):
        # Graph clustering happens before capacity fragmentation. If a connected
        # component is still too large, every fragment repeats its graph
        # topology so the final reducer can reconnect related evidence.
        file_payload = json.dumps(chunk.get("files", []), indent=2, ensure_ascii=False)
        fragments = _split_text_preserving_lines(file_payload, payload_chars)
        for fragment_index, fragment in enumerate(fragments, 1):
            payloads.append(
                {
                    "kind": "patch",
                    "graph_chunk": graph_index,
                    "graph_chunk_count": len(graph_chunks),
                    "fragment": fragment_index,
                    "fragment_count": len(fragments),
                    "dependency_cluster": chunk.get("dependency_cluster", []),
                    "dependency_contexts": _dependency_descriptor(
                        chunk.get("dependency_contexts") or []
                    ),
                    "structured_changes_fragment": fragment,
                }
            )

    metadata = {
        "chunking_method": "dependency_graph_then_token_bound",
        "graph_chunk_count": len(graph_chunks),
        "patch_payload_count": len(payloads),
        "dependency_node_count": len(graph.get("nodes") or {}),
        "dependency_edge_count": len(graph.get("edges") or []),
        "dependency_cluster_count": len(graph.get("clusters") or []),
        "coverage_recovery_applied": recovery_applied,
        "coverage_recovery_change_count": recovery_change_count,
        "parsed_file_count": len(expected_files),
        "parsed_change_count": expected_changes,
        "all_files_covered": covered_files == expected_files,
        "all_hunks_covered": expected_change_keys.issubset(covered_change_keys),
    }
    return payloads, metadata


def _context_payloads(
    episode: dict[str, Any], *, payload_chars: int
) -> list[dict[str, Any]]:
    """Losslessly chunk non-patch evidence at logical source boundaries."""
    sources = {
        "initial_failure": episode["initial_failure"],
        "workflow_metadata": episode["workflow_metadata"],
    }
    payloads = []
    for source, value in sources.items():
        encoded = json.dumps(value, indent=2, ensure_ascii=False)
        fragments = _split_text_preserving_lines(encoded, payload_chars)
        for fragment_index, fragment in enumerate(fragments, 1):
            payloads.append(
                {
                    "kind": "context",
                    "source": source,
                    "fragment": fragment_index,
                    "fragment_count": len(fragments),
                    "evidence_fragment": fragment,
                }
            )
    return payloads


def _build_l1_prompt(evidence: dict[str, Any], heading: str) -> str:
    return f"""You are constructing a Whole-Repair-History experimental control.

Treat the complete fail-to-pass PR below as ONE INDIVISIBLE repair episode.
Do not decompose it into separate CI problems. Do not map commits, files, or
patch hunks to individual failures. Do not infer dependencies between problems.

{heading}:
```json
{json.dumps(evidence, indent=2, ensure_ascii=False)}
```

Return exactly one concrete L1 record for the complete PR:
{{
  "failure_type": "aggregate failure category",
  "problem": "combined description of every CI failure in the PR",
  "root_cause": "combined root-cause explanation",
  "fix_strategy": "complete cumulative repair strategy",
  "why_fix_works": "why the complete patch produces a passing outcome",
  "verification_cmd": "overall CI verification command or workflow"
}}

Do not return arrays of problems or per-commit repair records.

{STRICT_JSON_RULES}
"""


def _patch_summary_prompt(
    payload: dict[str, Any], global_context: dict[str, Any]
) -> str:
    return f"""Summarize one dependency-aware portion of a complete PR patch.

This is internal context-size handling, not problem decomposition. Describe
the observed changes and relationships as neutral evidence. Do not create
problem IDs, separate repair records, problem-to-hunk attribution, or repair
ordering. Preserve caller/callee and cross-file relationships supplied by the
dependency graph.

GLOBAL WHOLE-PR CONTEXT:
```json
{json.dumps(global_context, indent=2, ensure_ascii=False)}
```

PATCH CHUNK:
```json
{json.dumps(payload, indent=2, ensure_ascii=False)}
```

Return one object:
{{
  "files": ["every file visible in this chunk"],
  "observed_changes": ["factual change"],
  "dependency_relationships": ["factual cross-file relationship"],
  "repair_contribution": "how these changes contribute to the complete patch",
  "validation_relevance": "relevant CI/workflow effect, without making a separate problem"
}}

{STRICT_JSON_RULES}
"""


def _context_summary_prompt(
    payload: dict[str, Any], global_context: dict[str, Any]
) -> str:
    return f"""Summarize one lossless fragment of whole-PR CI/workflow evidence.

This is internal context-size handling. Do not create separate problems,
repairs, dependencies, or retrieval records. Preserve concrete failure
signals, commands, jobs, workflow constraints, and validation evidence.

GLOBAL WHOLE-PR CONTEXT:
```json
{json.dumps(global_context, indent=2, ensure_ascii=False)}
```

CONTEXT CHUNK:
```json
{json.dumps(payload, indent=2, ensure_ascii=False)}
```

Return one object:
{{
  "evidence_source": "source name",
  "failure_signals": ["concrete signal"],
  "workflow_constraints": ["constraint"],
  "validation_commands": ["command"],
  "summary": "neutral factual summary"
}}

{STRICT_JSON_RULES}
"""


def _summary_batches(
    items: list[dict[str, Any]], max_chars: int
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 2
    for item in items:
        item_size = len(json.dumps(item, ensure_ascii=False)) + 2
        if item_size > max_chars:
            raise ValueError(
                "A chunk summary exceeds the hierarchical reduction budget; "
                "use a larger --max-evidence-chars value"
            )
        if current and current_size + item_size > max_chars:
            batches.append(current)
            current, current_size = [], 2
        current.append(item)
        current_size += item_size
    if current:
        batches.append(current)
    return batches


def _reduction_prompt(batch: list[dict[str, Any]], round_number: int) -> str:
    return f"""REDUCE WHOLE-PR SUMMARIES (round {round_number}).

Combine these internal evidence summaries into one neutral aggregate summary.
Preserve every distinct file, change, failure signal, workflow constraint,
validation command, and cross-file relationship. Do not create separate
problems, problem IDs, attribution, dependency ordering, or memory records.

```json
{json.dumps(batch, indent=2, ensure_ascii=False)}
```

Return one aggregate JSON object with lists for files, observed changes,
failure signals, workflow constraints, validation commands, and cross-file
relationships, plus one overall summary.

{STRICT_JSON_RULES}
"""


def _reduce_summaries_to_fit(
    summaries: list[dict[str, Any]], llm: Any, iid: str, max_chars: int
) -> tuple[list[dict[str, Any]], int]:
    """Hierarchically reduce summaries; never truncate or silently discard one."""
    reduced = summaries
    rounds = 0
    while len(json.dumps(reduced, ensure_ascii=False)) > max_chars:
        rounds += 1
        if rounds > 8:
            raise ValueError(f"Issue {iid}: summary reduction did not converge")
        batches = _summary_batches(reduced, max_chars)
        next_round = []
        for batch_index, batch in enumerate(batches, 1):
            result = _invoke_single_object(
                llm,
                _reduction_prompt(batch, rounds),
                f"Issue {iid} reduction round {rounds} batch {batch_index}",
            )
            next_round.append(
                {
                    "kind": "aggregate_summary",
                    "round": rounds,
                    "batch": batch_index,
                    "summary": result,
                }
            )
        if len(next_round) >= len(reduced) and len(next_round) > 1:
            raise ValueError(
                f"Issue {iid}: summary reduction made no progress; increase "
                "--max-evidence-chars"
            )
        reduced = next_round
    return reduced, rounds


def build_l2_prompt(l1: dict[str, Any]) -> str:
    return f"""Build one repository-level L2 strategy from this ONE whole-PR L1 record.

```json
{json.dumps(l1, indent=2, ensure_ascii=False)}
```

Keep the PR indivisible. Do not split failures, files, commits, or changes into
problem-specific strategies. Return exactly:
{{
  "failure_identify": "one aggregate failure summary",
  "strategy": {{
    "failure_type": "aggregate failure category",
    "summary": "one whole-PR strategy summary",
    "intent": "overall repair intent",
    "reasoning": "reasoning across the complete episode",
    "rationale": "why this whole-episode strategy is appropriate",
    "when_to_apply": "conditions for applying this strategy",
    "signals": ["observable signal"],
    "key_actions": ["ordered whole-episode action"],
    "pitfalls": ["pitfall"],
    "validation_cmd": "overall validation command"
  }}
}}

Do not return a repair_strategies array or multiple strategies.

{STRICT_JSON_RULES}
"""


def build_l3_prompt(l1: dict[str, Any], l2: dict[str, Any]) -> str:
    return f"""Build one transferable L3 pattern from one whole-PR L1 record and
its one whole-PR L2 strategy.

L1:
```json
{json.dumps(l1, indent=2, ensure_ascii=False)}
```

L2:
```json
{json.dumps(l2, indent=2, ensure_ascii=False)}
```

Keep the historical PR indivisible. Return exactly one pattern object:
{{
  "pattern_id": "stable-kebab-case-pattern-id",
  "failure_type": "general failure category",
  "failure_pattern": "generalized whole-episode failure pattern",
  "problem": "generalized aggregate problem",
  "reasoning": "why the generalized whole-episode repair works",
  "when_to_apply": "transfer conditions",
  "signals": ["general signal"],
  "universal_fix": {{
    "approach": "general approach",
    "steps": ["general repair step"],
    "applies_to": ["applicable context"]
  }},
  "examples": [{{"file": "representative file", "before": "before", "after": "after"}}],
  "dependent_changes": []
}}

Do not return a universal_patterns array or multiple patterns.

{STRICT_JSON_RULES}
"""


def _invoke_single_object(llm: Any, prompt: str, context: str) -> dict[str, Any]:
    value = invoke_llm_with_retry(
        llm=llm, prompt=prompt, parse_json=True, max_tokens=16_000
    )
    if not isinstance(value, dict):
        raise ValueError(f"{context}: model did not return one object")
    return value


def _invoke_validated_object(
    llm: Any,
    prompt: str,
    context: str,
    validator: Any,
    *,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Retry schema-incomplete model objects before advancing a pipeline level."""
    attempt_prompt = prompt
    last_error: ValueError | None = None
    for attempt in range(1, max_attempts + 1):
        value = _invoke_single_object(llm, attempt_prompt, context)
        try:
            validator(value)
            return value
        except ValueError as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            print(
                f"      [SCHEMA] {context} incomplete on attempt "
                f"{attempt}/{max_attempts}: {exc}; requesting correction"
            )
            attempt_prompt = f"""{prompt}

Your previous response was incomplete and failed validation.

VALIDATION ERROR:
{exc}

PREVIOUS RESPONSE:
```json
{json.dumps(value, indent=2, ensure_ascii=False)}
```

Return the complete corrected object. Include every required field with a
non-empty value, preserve the one-record whole-PR scope, and return JSON only.
"""
    raise ValueError(
        f"{context}: invalid after {max_attempts} schema attempts: {last_error}"
    )


def _validate_l1_object(value: dict[str, Any], iid: str) -> None:
    for field in (
        "failure_type",
        "problem",
        "root_cause",
        "fix_strategy",
        "why_fix_works",
        "verification_cmd",
    ):
        _required_text(value, field, f"Issue {iid} L1")
    leaked = {"problems", "repair_strategies", "universal_patterns"}.intersection(value)
    if leaked:
        raise ValueError(
            f"Issue {iid}: L1 reintroduced plural records: {sorted(leaked)}"
        )


def _validate_l2_object(value: dict[str, Any], iid: str) -> None:
    _required_text(value, "failure_identify", f"Issue {iid} L2")
    if not isinstance(value.get("strategy"), dict):
        raise ValueError(f"Issue {iid}: L2 strategy must be one object")
    _required_text(value["strategy"], "summary", f"Issue {iid} L2 strategy")
    leaked = {"problems", "repair_strategies", "universal_patterns"}.intersection(value)
    if leaked:
        raise ValueError(
            f"Issue {iid}: L2 reintroduced plural records: {sorted(leaked)}"
        )


def _validate_l3_object(value: dict[str, Any], iid: str) -> None:
    for field in (
        "pattern_id",
        "failure_type",
        "failure_pattern",
        "problem",
        "reasoning",
    ):
        _required_text(value, field, f"Issue {iid} L3")
    if not isinstance(value.get("universal_fix"), dict):
        raise ValueError(f"Issue {iid}: L3 universal_fix must be an object")
    leaked = {"problems", "repair_strategies", "universal_patterns"}.intersection(value)
    if leaked:
        raise ValueError(
            f"Issue {iid}: L3 reintroduced plural records: {sorted(leaked)}"
        )


def generate_whole_pr_abstractions(
    episode: dict[str, Any], llm: Any, max_evidence_chars: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build L1 -> L2 -> L3 sequentially while retaining one record per level."""
    iid = episode["issue_id"]
    complete_evidence = _whole_pr_evidence(episode)
    source_chars = len(json.dumps(complete_evidence, ensure_ascii=False))
    if source_chars <= max_evidence_chars:
        l1_prompt = _build_l1_prompt(complete_evidence, "WHOLE PR EVIDENCE FOR L1")
        processing = {
            "chunking_applied": False,
            "chunking_method": "none",
            "source_evidence_chars": source_chars,
            "graph_chunk_count": 0,
            "patch_payload_count": 0,
            "context_payload_count": 0,
            "reduction_rounds": 0,
            "all_files_covered": True,
            "all_hunks_covered": True,
            "evidence_truncated": False,
            "ci_evidence_source": episode["initial_failure"].get("ci_evidence_source"),
        }
    else:
        model_name = str(getattr(llm, "model_name", None) or "deepseek-v4-flash")
        payload_chars = max(2_000, int(max_evidence_chars * 0.45))
        reduction_chars = max(4_000, int(max_evidence_chars * 0.65))
        global_context = _global_whole_pr_context(episode)
        patch_payloads, processing = _dependency_aware_patch_payloads(
            episode,
            model_name=model_name,
            payload_chars=payload_chars,
        )
        context_payloads = _context_payloads(episode, payload_chars=payload_chars)
        summaries: list[dict[str, Any]] = []
        for payload_index, payload in enumerate(patch_payloads, 1):
            result = _invoke_single_object(
                llm,
                _patch_summary_prompt(payload, global_context),
                f"Issue {iid} patch chunk {payload_index}",
            )
            summaries.append(
                {
                    "kind": "patch_summary",
                    "graph_chunk": payload["graph_chunk"],
                    "fragment": payload["fragment"],
                    "summary": result,
                }
            )
        for payload_index, payload in enumerate(context_payloads, 1):
            result = _invoke_single_object(
                llm,
                _context_summary_prompt(payload, global_context),
                f"Issue {iid} context chunk {payload_index}",
            )
            summaries.append(
                {
                    "kind": "context_summary",
                    "source": payload["source"],
                    "fragment": payload["fragment"],
                    "summary": result,
                }
            )
        reduced_summaries, reduction_rounds = _reduce_summaries_to_fit(
            summaries, llm, iid, reduction_chars
        )
        processing.update(
            {
                "chunking_applied": True,
                "source_evidence_chars": source_chars,
                "context_payload_count": len(context_payloads),
                "summary_count_before_reduction": len(summaries),
                "summary_count_after_reduction": len(reduced_summaries),
                "reduction_rounds": reduction_rounds,
                "evidence_truncated": False,
                "ci_evidence_source": episode["initial_failure"].get(
                    "ci_evidence_source"
                ),
            }
        )
        synthesized_evidence = {
            "global_whole_pr_context": global_context,
            "dependency_chunking": processing,
            "complete_evidence_summaries": reduced_summaries,
        }
        if (
            len(json.dumps(synthesized_evidence, ensure_ascii=False))
            > max_evidence_chars
        ):
            raise ValueError(
                f"Issue {iid}: synthesized evidence still exceeds the context "
                "budget; increase --max-evidence-chars"
            )
        l1_prompt = _build_l1_prompt(
            synthesized_evidence, "WHOLE PR CHUNK SUMMARIES FOR L1"
        )

    processing["processing_schema_version"] = 1
    l1 = _invoke_validated_object(
        llm,
        l1_prompt,
        f"Issue {iid} L1",
        lambda value: _validate_l1_object(value, iid),
    )
    l2 = _invoke_validated_object(
        llm,
        build_l2_prompt(l1),
        f"Issue {iid} L2",
        lambda value: _validate_l2_object(value, iid),
    )
    l3 = _invoke_validated_object(
        llm,
        build_l3_prompt(l1, l2),
        f"Issue {iid} L3",
        lambda value: _validate_l3_object(value, iid),
    )
    value = {"l1": l1, "l2": l2, "l3": l3}
    validate_abstraction(value, iid)
    return value, processing


def _required_text(mapping: dict[str, Any], key: str, context: str) -> str:
    value = str(mapping.get(key) or "").strip()
    if not value:
        raise ValueError(f"{context}: missing non-empty {key}")
    return value


def validate_abstraction(value: dict[str, Any], iid: str) -> None:
    """Reject empty outputs and attempts to reintroduce decomposition."""
    for level in ("l1", "l2", "l3"):
        if not isinstance(value.get(level), dict):
            raise ValueError(f"Issue {iid}: {level} must be one object")
    _validate_l1_object(value["l1"], iid)
    _validate_l2_object(value["l2"], iid)
    _validate_l3_object(value["l3"], iid)


def build_memory_records(
    episode: dict[str, Any],
    abstraction: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Wrap three singular abstractions in existing retrieval schemas."""
    iid = episode["issue_id"]
    l1_abstract = abstraction["l1"]
    problem = {
        "record_scope": "whole_pr",
        "verification_cmd": str(l1_abstract.get("verification_cmd") or ""),
        "failure_type": str(l1_abstract.get("failure_type") or ""),
        "problem": str(l1_abstract.get("problem") or ""),
        "root_cause": str(l1_abstract.get("root_cause") or ""),
        "fix_strategy": str(l1_abstract.get("fix_strategy") or ""),
        "why_fix_works": str(l1_abstract.get("why_fix_works") or ""),
        "files": list(episode.get("all_changed_files", [])),
    }
    l1 = {
        "issue_id": iid,
        "repo": episode["repo"],
        "repo_owner": episode["repo_owner"],
        "repo_name": episode["repo_name"],
        "workflow": episode["workflow"],
        "workflow_path": episode["workflow_path"],
        "workflow_name": episode["workflow_name"],
        "problems": [problem],
        "representation": "whole_repair_history",
    }

    l2_abstract = abstraction["l2"]
    strategy = dict(l2_abstract["strategy"])
    strategy["step"] = 1
    strategy.setdefault("failure_type", problem["failure_type"])
    strategy.setdefault("signals", [])
    strategy.setdefault("key_actions", [])
    strategy.setdefault("pitfalls", [])
    l2 = {
        "issue_id": iid,
        "repo": episode["repo"],
        "workflow": episode["workflow"],
        "workflow_path": episode["workflow_path"],
        "workflow_name": episode["workflow_name"],
        "total_problems": 1,
        "failure_identify": [str(l2_abstract["failure_identify"])],
        "repair_strategies": [strategy],
        "representation": "whole_repair_history",
    }

    l3 = dict(abstraction["l3"])
    raw_pattern_id = str(l3.get("pattern_id") or "whole-repair-history")
    safe_pattern_id = re.sub(r"[^a-z0-9]+", "-", raw_pattern_id.lower()).strip("-")
    l3["pattern_id"] = f"whole-pr-{iid}-{safe_pattern_id}"
    l3["source_issue_id"] = iid
    l3["source_repo"] = episode["repo"]
    l3["representation"] = "whole_repair_history"
    l3.setdefault("dependent_changes", [])
    return l1, l2, l3


def _compact_l1_record(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only retrieval-facing cumulative whole-PR fields in L1 memory."""
    problem_fields = (
        "record_scope",
        "verification_cmd",
        "failure_type",
        "problem",
        "root_cause",
        "fix_strategy",
        "why_fix_works",
        "files",
    )
    problems = []
    for problem in (row.get("problems") or [])[:1]:
        if isinstance(problem, dict):
            problems.append({key: problem.get(key) for key in problem_fields})
    return {
        "issue_id": row.get("issue_id"),
        "repo": row.get("repo"),
        "repo_owner": row.get("repo_owner"),
        "repo_name": row.get("repo_name"),
        "workflow": row.get("workflow"),
        "workflow_path": row.get("workflow_path"),
        "workflow_name": row.get("workflow_name"),
        "problems": problems,
        "representation": "whole_repair_history",
    }


def validate_memory_sets(
    expected_ids: Iterable[str],
    l1_rows: list[dict],
    l2_rows: list[dict],
    l3_rows: list[dict],
) -> None:
    expected = [str(value) for value in expected_ids]
    l1_ids = [str(row.get("issue_id")) for row in l1_rows]
    l2_ids = [str(row.get("issue_id")) for row in l2_rows]
    l3_ids = [str(row.get("source_issue_id")) for row in l3_rows]
    if l1_ids != expected or l2_ids != expected or l3_ids != expected:
        raise ValueError("L1/L2/L3 IDs or ordering differ from requested memory IDs")
    for row in l1_rows:
        leaked_audit_fields = {
            "whole_pr_history",
            "abstraction_input_chunked",
            "abstraction_input_truncated",
            "abstraction_processing",
        }.intersection(row)
        if leaked_audit_fields:
            raise ValueError(
                f"L1 issue {row.get('issue_id')} contains audit-only fields: "
                f"{sorted(leaked_audit_fields)}"
            )
        if len(row.get("problems", [])) != 1:
            raise ValueError(
                f"L1 issue {row.get('issue_id')} does not contain exactly one item"
            )
        forbidden_issue_fields = {"dependencies", "repair_sequence"}.intersection(row)
        forbidden_record_fields = {"problem_id", "enabled"}.intersection(
            row.get("problems", [{}])[0]
        )
        if forbidden_issue_fields or forbidden_record_fields:
            raise ValueError(
                f"L1 issue {row.get('issue_id')} contains attribution fields: "
                f"{sorted(forbidden_issue_fields | forbidden_record_fields)}"
            )
    for row in l2_rows:
        if len(row.get("repair_strategies", [])) != 1:
            raise ValueError(
                f"L2 issue {row.get('issue_id')} does not contain exactly one strategy"
            )
    if len(l3_ids) != len(set(l3_ids)):
        raise ValueError("L3 must contain exactly one pattern per source PR")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_by_id(rows: list[dict], key: str) -> dict[str, dict]:
    return {str(row.get(key)): row for row in rows if row.get(key) is not None}


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [row for row in value if isinstance(row, dict)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-ids", type=Path, default=DEFAULT_MEMORY_IDS)
    parser.add_argument("--dataset", type=Path, help="Raw benchmark JSONL")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-revision", help="Optional pinned Hugging Face revision"
    )
    parser.add_argument("--log-details", type=Path, default=DEFAULT_LOG_DETAILS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--issue-ids", help="Optional comma-separated subset of memory IDs"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect raw episodes without LLM abstraction",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit incomplete debug episodes",
    )
    parser.add_argument("--max-evidence-chars", type=int, default=120_000)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory_ids = load_id_list(args.memory_ids)
    if args.issue_ids:
        requested = {
            value.strip() for value in args.issue_ids.split(",") if value.strip()
        }
        unknown = requested.difference(memory_ids)
        if unknown:
            raise ValueError(
                f"Requested IDs are not in memory split: {sorted(unknown)}"
            )
        memory_ids = [iid for iid in memory_ids if iid in requested]
    if args.limit is not None:
        memory_ids = memory_ids[: args.limit]

    records, effective_dataset_revision = load_benchmark_records(
        dataset_path=args.dataset,
        dataset_name=args.dataset_name,
        revision=args.dataset_revision,
    )
    by_id = {issue_id(row): row for row in records if issue_id(row)}
    missing_rows = [iid for iid in memory_ids if iid not in by_id]
    if missing_rows:
        raise ValueError(f"Raw benchmark is missing memory IDs: {missing_rows[:10]}")

    failures_by_id, failures_by_sha = load_structured_failures(args.log_details)
    output_dir = args.output_dir.resolve()
    # Keep the whole-PR baseline physically distinct from the ordinary
    # bidirectional and cross-repository ReCAP memories.  The files retain the
    # retrieval-compatible names, but the directory describes their true
    # record scope.
    memory_dir = output_dir / "whole_pr_trs"
    raw_path = output_dir / "raw_episodes.json"
    processing_path = output_dir / "abstraction_metadata.json"
    l1_path = memory_dir / "failure_memory.json"
    l2_path = memory_dir / "repo_memory.json"
    l3_path = memory_dir / "cross_memory.json"

    raw_by_id = _rows_by_id(_load_json_list(raw_path), "issue_id")
    original_l1_rows = _load_json_list(l1_path)
    l1_by_id = _rows_by_id(original_l1_rows, "issue_id")
    l2_by_id = _rows_by_id(_load_json_list(l2_path), "issue_id")
    l3_by_id = _rows_by_id(_load_json_list(l3_path), "source_issue_id")
    processing_by_id = _rows_by_id(_load_json_list(processing_path), "issue_id")

    # Migrate earlier whole-PR outputs without another paid LLM call: move
    # construction metadata out of retrieval memory and compact L1 in place.
    migrated_existing = False
    for iid, row in list(l1_by_id.items()):
        embedded_processing = row.get("abstraction_processing")
        if isinstance(embedded_processing, dict) and iid not in processing_by_id:
            processing_by_id[iid] = {"issue_id": iid, **embedded_processing}
            migrated_existing = True
        compact = _compact_l1_record(row)
        if compact != row:
            l1_by_id[iid] = compact
            migrated_existing = True
    if migrated_existing:
        _write_json(l1_path, [l1_by_id[key] for key in memory_ids if key in l1_by_id])
        _write_json(
            processing_path,
            [processing_by_id[key] for key in memory_ids if key in processing_by_id],
        )

    llm = None
    if not args.collect_only:
        model = configure_model_environment(args.model) or args.model
        llm = LitellmModel(model_name=model)

    for index, iid in enumerate(memory_ids, 1):
        existing_processing = processing_by_id.get(iid, {})
        current_l1 = existing_processing.get(
            "processing_schema_version"
        ) == 1 and not existing_processing.get("evidence_truncated", True)
        complete = (
            iid in l1_by_id and iid in l2_by_id and iid in l3_by_id and current_l1
        )
        if complete and not args.rebuild and not args.collect_only:
            print(f"[{index}/{len(memory_ids)}] {iid}: already complete")
            continue

        cached_episode = raw_by_id.get(iid)
        cached_is_usable = bool(
            cached_episode
            and cached_episode.get("evidence_policy") == "benchmark_only"
            and cached_episode.get("episode_schema_version") == 2
            and (
                args.allow_incomplete
                or all((cached_episode.get("completeness") or {}).values())
            )
        )
        if cached_is_usable and not args.rebuild:
            episode = cached_episode
            print(f"[{index}/{len(memory_ids)}] {iid}: using cached whole PR")
        else:
            print(f"[{index}/{len(memory_ids)}] {iid}: collecting whole PR")
            row = by_id[iid]
            structured = failures_by_id.get(iid) or failures_by_sha.get(
                str(row.get("sha_fail") or "")
            )
            episode = collect_whole_pr_episode(
                row,
                structured_failure=structured,
                allow_incomplete=args.allow_incomplete,
            )
            raw_by_id[iid] = episode
            _write_json(
                raw_path, [raw_by_id[key] for key in memory_ids if key in raw_by_id]
            )

        if args.collect_only:
            continue

        print(f"[{index}/{len(memory_ids)}] {iid}: building singular L1/L2/L3")
        abstraction, processing_metadata = generate_whole_pr_abstractions(
            episode, llm, args.max_evidence_chars
        )
        l1, l2, l3 = build_memory_records(
            episode,
            abstraction,
        )
        l1_by_id[iid], l2_by_id[iid], l3_by_id[iid] = l1, l2, l3
        processing_by_id[iid] = {"issue_id": iid, **processing_metadata}
        _write_json(l1_path, [l1_by_id[key] for key in memory_ids if key in l1_by_id])
        _write_json(l2_path, [l2_by_id[key] for key in memory_ids if key in l2_by_id])
        _write_json(l3_path, [l3_by_id[key] for key in memory_ids if key in l3_by_id])
        _write_json(
            processing_path,
            [processing_by_id[key] for key in memory_ids if key in processing_by_id],
        )

    _write_json(output_dir / "memory_issue_ids.json", memory_ids)
    generated_ids = [
        iid
        for iid in memory_ids
        if iid in l1_by_id and iid in l2_by_id and iid in l3_by_id
    ]
    incomplete_ids = [
        iid
        for iid in memory_ids
        if iid in raw_by_id
        and not all((raw_by_id[iid].get("completeness") or {}).values())
    ]
    truncated_ids = [
        iid
        for iid in memory_ids
        if iid in processing_by_id and processing_by_id[iid].get("evidence_truncated")
    ]
    chunked_ids = [
        iid
        for iid in memory_ids
        if iid in processing_by_id and processing_by_id[iid].get("chunking_applied")
    ]
    if not args.collect_only and generated_ids == memory_ids:
        validate_memory_sets(
            memory_ids,
            [l1_by_id[iid] for iid in memory_ids],
            [l2_by_id[iid] for iid in memory_ids],
            [l3_by_id[iid] for iid in memory_ids],
        )

    manifest = {
        "experiment": "whole_repair_history",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset) if args.dataset else args.dataset_name,
        "requested_dataset_revision": args.dataset_revision,
        "effective_dataset_revision": effective_dataset_revision,
        "memory_ids_source": str(args.memory_ids),
        "requested_count": len(memory_ids),
        "raw_episode_count": sum(iid in raw_by_id for iid in memory_ids),
        "complete_memory_count": len(generated_ids),
        "model": None if args.collect_only else args.model,
        "collect_only": args.collect_only,
        "evidence_policy": "benchmark_only",
        "external_api_enrichment": False,
        "allow_incomplete": args.allow_incomplete,
        "incomplete_issue_ids": incomplete_ids,
        "dependency_graph_chunked_issue_ids": chunked_ids,
        "abstraction_input_truncated_issue_ids": truncated_ids,
        "invariants": {
            "problem_decomposition": False,
            "problem_repair_attribution": False,
            "dependency_expansion": False,
            "dependency_graph_used_only_for_chunking": True,
            "evidence_truncation": False,
            "records_per_pr_per_level": 1,
        },
        "files": {},
    }
    for path in (raw_path, processing_path, l1_path, l2_path, l3_path):
        if path.exists():
            manifest["files"][str(path.relative_to(output_dir))] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    _write_json(output_dir / "build_manifest.json", manifest)

    print(f"Raw episodes: {manifest['raw_episode_count']}/{len(memory_ids)}")
    print(f"Complete L1/L2/L3: {len(generated_ids)}/{len(memory_ids)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
