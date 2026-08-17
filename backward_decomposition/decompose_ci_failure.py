#!/usr/bin/env python3
"""
decompose_ci_failure.py - Reverse Engineer CI Failures into Atomic Problems
===========================================================================

Based on professor's direction: Given CI failure (FIRST failure only) + ground truth diff,
use LLM to reverse engineer ALL hidden problems.

Key insight: CI stops at FIRST failure, but diff fixes MULTIPLE problems.
We need to infer hidden problems from the diff.

Usage:
    # Decompose single issue
    python scripts/decompose_ci_failure.py --issue-id 410

    # Decompose all eval issues
    python scripts/decompose_ci_failure.py --batch

References:
    - O-CRD: Backward reasoning from ground truth
    - STAIR: Multi-layer hierarchical abstraction
"""

import argparse
import copy
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# NEW: Import build_memory module for L1/L2/L3 generation
from build_memory import (
    build_l2_memory,  # L2: Generate repair strategies
    build_l3_memory,  # L3: Generate universal patterns
    generate_l1_from_decomposed_problems,  # L1: Pass decomposed problems + dependencies
)

# Import prompt templates
from prompt_template.backward_decomposition import (
    build_atomic_prompt,
    build_classification_prompt_regular,
    build_classification_prompt_with_dependencies,
    build_cluster_merge_prompt,
    build_full_dependency_prompt,
    build_validation_group_dependency_prompt,
)
from utilities.ci_log_analyzer import (
    _log_analysis_to_context,
    _run_log_analysis,
)
from utilities.ci_workflow_aware_retrieval import (
    analyze_workflow_from_benchmark,
)
from utilities.dependency_evidence import (
    build_dependency_graph_from_structured_diff,
)
from utilities.deterministic_diff_parser import (
    chunk_structured_diff,
    format_structured_for_llm,
    parse_diff_to_structured,
)
from utilities.diff_chunker import estimate_tokens
from utilities.llm_invoker import (
    LLMTransientConnectionError,
    invoke_llm_with_retry,
)
from utilities.llm_model import LitellmModel
from utilities.model_registry import (
    configure_model_environment,
)
from utilities.model_token_config import get_model_config


# Load memory issue IDs from workflow_validation_cache.json
def _load_memory_issue_ids() -> list[str]:
    """Load issue IDs from workflow_validation_cache.json."""
    cache_path = PROJECT_ROOT / "data" / "workflow_validation_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
                issue_ids = [str(item["id"]) for item in cache if "id" in item]
                return issue_ids
        except Exception as e:
            print(f"Warning: Could not load workflow_validation_cache.json: {e}")
            return []
    return []


def _issue_id(issue: dict[str, Any]) -> str:
    """Return the benchmark issue identifier across supported dataset schemas."""
    return str(
        issue.get("id")
        or issue.get("instance_id")
        or issue.get("issue_id")
        or issue.get("original_issue_id")
        or ""
    )


def _retrieve_memory_problems(
    issue: dict, memory_retrieval, log_analysis: dict = None
) -> dict:
    """
    Retrieve problems from memory plugin.

    Args:
        issue: Issue data from dataset
        memory_retrieval: STAIRRetrieval instance (or None if disabled)
        log_analysis: Log analysis result (optional, if available)

    Returns:
        Dict with 'problems', 'metadata', 'consecutive_sequences', 'dependencies'
    """
    if not memory_retrieval:
        return {
            "problems": [],
            "metadata": {"mode": "disabled"},
            "consecutive_sequences": [],
            "dependencies": [],
        }

    # Build ci_failure dict
    ci_failure = {
        "repo": issue.get("repo", ""),
        "workflow": issue.get("workflow", ""),
        "problem_statement": issue.get("problem_statement", ""),
        "error_signals": issue.get("error_signals", []),
        "config_signals": issue.get("config_signals", []),
        "failure_type": issue.get("failure_type", "unknown"),
        "stage": issue.get("stage", "unknown"),
    }

    # If log_analysis provided, use it to enrich ci_failure
    if log_analysis:
        ci_failure["problem_statement"] = log_analysis.get(
            "error_summary", ci_failure["problem_statement"]
        )
        ci_failure["error_signals"] = log_analysis.get(
            "error_signals", ci_failure["error_signals"]
        )
        ci_failure["config_signals"] = log_analysis.get(
            "config_signals", ci_failure["config_signals"]
        )
        ci_failure["failure_type"] = log_analysis.get(
            "failure_type", ci_failure["failure_type"]
        )

    # Retrieve from memory
    print(
        f"   Retrieving from memory ({memory_retrieval.enabled_levels if hasattr(memory_retrieval, 'enabled_levels') else 'unknown'})..."
    )
    result = memory_retrieval.retrieve(ci_failure, top_k=5)

    print(f"  OK Found {len(result['problems'])} problems from memory")
    if result["metadata"].get("enabled_levels"):
        print(f"     Levels: {result['metadata']['enabled_levels']}")
        print(
            f"     Retrieved: L1={result['metadata']['retrieved']['l1']}, "
            f"L2={result['metadata']['retrieved']['l2']}, "
            f"L3={result['metadata']['retrieved']['l3']}"
        )

    return result


def _get_model_aware_limits(model_name: str | None = None) -> dict[str, int]:
    """
    Get model-aware chunk limits for diff processing.

    Strategy:
    - diff_chunk: 50% of input capacity (leaves room for prompt + output)
    - findings_inline: 40% of input (more context for inline classification)
    - findings_batch: 30% of input (batch mode processes multiple items)
    - max_files_per_chunk: Based on model capacity (80 for minimax, 150 for GLM)
    - max_changes_per_chunk: Based on model capacity (400 for minimax, 800 for GLM)
    - atomic_analysis_threshold: 50% for complete validation group analysis

    Results (vs old limits):
    - minimax-m2.5: 200k/160k/120k chars, 80 files, 400 changes (6-8x larger)
    - GLM-5.2: 400k/320k/240k chars, 150 files, 800 changes (13-17x larger)

    All limits are SAFE - validated to fit in context with max output.
    """
    try:
        config = get_model_config(model_name)

        return {
            # Character limits for diff/findings chunks
            "diff_chunk_chars": config["input_chunk_chars"] // 2,
            "findings_inline_chars": int(config["input_chunk_chars"] * 0.4),
            "findings_batch_chars": int(config["input_chunk_chars"] * 0.3),
            # File and change counts for structured processing
            "max_files_per_chunk": config["decompose_max_files_per_chunk"],
            "max_changes_per_chunk": config["decompose_max_changes_per_chunk"],
            "output_safe_tokens": config["output_safe_tokens"],
            # Atomic analysis threshold (50% of input capacity)
            "atomic_analysis_threshold_chars": int(config["input_chunk_chars"] * 0.5),
            "max_input_tokens": config.get("max_input_tokens", 200000),
        }
    except Exception as e:
        LOGGER.warning(
            f"Could not get model-aware limits: {e}, using conservative defaults"
        )
        # Fallback to conservative defaults
        return {
            "diff_chunk_chars": 30000,
            "findings_inline_chars": 22000,
            "findings_batch_chars": 14000,
            "max_files_per_chunk": 80,
            "max_changes_per_chunk": 400,
            "output_safe_tokens": 7000,
            "atomic_analysis_threshold_chars": 100000,  # Conservative fallback
            "max_input_tokens": 200000,
        }


def _classification_output_tokens(model_name: str | None = None) -> int:
    """Use a bounded output budget for compact Step 1 classification JSON."""
    model_limits = _get_model_aware_limits(model_name)
    return min(model_limits["output_safe_tokens"], 60_000)


LOGGER = logging.getLogger(__name__)
STRICT_JSON_RULES = """### Output Rules (STRICT) - CRITICAL FOR PARSING
- Output MUST be ONLY valid JSON - nothing else.
- Your FIRST character MUST be { or [ - ABSOLUTELY NO text before.
- Your LAST character MUST be } or ] - ABSOLUTELY NO text after.
- Do NOT wrap in ANY backticks: NO ``` or ```json or ` - NONE AT ALL.
- Do NOT add explanations, comments, markdown, or any text outside the JSON.
- Do NOT start with phrases like "Looking at this", "Here is", "json", "```json", etc.
- Use double quotes for all JSON keys and string values.
- Do not emit trailing commas.
- Your entire response will be passed DIRECTLY to json.loads() - it MUST parse perfectly."""

load_dotenv(PROJECT_ROOT / ".env", override=False)


def _invoke_json(
    llm: Any,
    prompt: str,
    max_tokens: int | None = None,
    retry_count: int = 0,
    rate_limit_retry: int = 0,
) -> Any:
    """
    DEPRECATED: Wrapper around utilities.llm_invoker.invoke_llm_with_retry

    This function is kept for backward compatibility. New code should use
    invoke_llm_with_retry from utilities.llm_invoker directly.

    The centralized retry logic in utilities/llm_invoker.py provides:
    - Rate limit handling with exponential backoff
    - Timeout handling with increased timeout retry
    - Connection error handling (peer closed, network issues)
    - Empty response handling
    - Malformed JSON repair attempts
    """
    # Map old parameters to new centralized function
    # Note: retry_count and rate_limit_retry are used internally by the new function
    return invoke_llm_with_retry(
        llm=llm,
        prompt=prompt,
        max_tokens=max_tokens,
        parse_json=True,
        verbose=True,
        _retry_count=retry_count,
        _rate_limit_retry=rate_limit_retry,
    )


def _repo_checkout_path(issue: dict[str, Any]) -> str | None:
    """Return local checkout path for dependent workflow/config files if present."""
    explicit = issue.get("repo_path") or issue.get("checkout_path")
    if explicit and Path(str(explicit)).exists():
        return str(explicit)

    repo = str(issue.get("repo") or "").strip()
    repo_name = str(issue.get("repo_name") or "").strip()
    repo_owner = str(issue.get("repo_owner") or "").strip()
    candidates: list[Path] = []
    if "/" in repo:
        candidates.append(PROJECT_ROOT / "repo" / repo.replace("/", "__"))
    if repo_owner and repo_name:
        candidates.append(PROJECT_ROOT / "repo" / f"{repo_owner}__{repo_name}")
    if repo_name:
        candidates.append(PROJECT_ROOT / "repo" / repo_name)

    for path in candidates:
        if path.exists():
            return str(path)
    return None


def build_benchmark_ci_context(
    issue: dict, llm: Any, output_dir: str | Path = "data/back_trs"
) -> dict[str, Any]:
    """
    Build structured CI context for decomposition.

    Uses:
      - CILogAnalyzer/precomputed benchmark CI context for failure analysis
      - raw benchmark workflow YAML (`issue["workflow"]`)
      - LLM-extracted ordered validation sequence from that workflow

    This function does not pass raw CI logs into decomposition prompts.
    """
    raw_workflow = str(issue.get("workflow") or "")
    if not raw_workflow.strip():
        raise ValueError(
            f"Issue {_issue_id(issue)} has no workflow YAML in benchmark data"
        )

    workflow_path = str(
        issue.get("workflow_path") or issue.get("workflow_filename") or ""
    )
    repo_path = _repo_checkout_path(issue)
    model_name = str(getattr(llm, "model_name", "") or "")
    issue_id = _issue_id(issue)

    # Check cache first to avoid re-running CILogAnalyzer
    sha_fail = issue.get("sha_fail", "")
    run_dir = Path(output_dir)
    shared_cache_dir = PROJECT_ROOT / "data"
    cache_file = shared_cache_dir / "log_details.json"
    cached_analysis = None

    if cache_file.exists() and sha_fail:
        try:
            with open(cache_file) as f:
                cache = json.load(f)
            # Match by full sha_fail (exact match)
            cached_analysis = next(
                (entry for entry in cache if entry.get("sha_fail") == sha_fail),
                None
            )
            if cached_analysis:
                print(
                    f"  [1/2] Loading cached CI log analysis for {sha_fail[:12]} "
                    f"from {cache_file}..."
                )
        except Exception as e:
            print(f"  WARNING:  Cache load failed: {e}, will re-analyze")

    if cached_analysis:
        log_analysis = cached_analysis
    else:
        print("  [1/2] Building structured CI failure context with CILogAnalyzer...")
        log_analysis = _run_log_analysis(
            issue,
            llm=llm,
            model=model_name,
            output_dir=run_dir / "ci_log_analysis",
        )

        # Save to cache immediately
        if sha_fail:
            try:
                existing_cache = []
                if cache_file.exists():
                    with open(cache_file) as f:
                        existing_cache = json.load(f)

                # Update or append
                updated = False
                for entry in existing_cache:
                    if entry.get("sha_fail") == sha_fail:
                        entry.clear()
                        entry.update(log_analysis)
                        updated = True
                        break

                if not updated:
                    existing_cache.append(log_analysis)

                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(existing_cache, f, indent=2)
                print(f"       Saved CI log analysis to {cache_file}")
            except Exception as e:
                print(f"       WARNING: Failed to save log_details.json cache: {e}")

    context = _log_analysis_to_context(log_analysis, issue, workflow_profile={})

    # Check workflow validation cache
    validation_cache_file = shared_cache_dir / "workflow_validation_cache.json"
    cached_validation = None

    if validation_cache_file.exists() and sha_fail:
        try:
            with open(validation_cache_file) as f:
                validation_cache = json.load(f)
            # Match by full sha_fail (exact match)
            cached_validation = next(
                (entry for entry in validation_cache if entry.get("sha_fail") == sha_fail),
                None,
            )
            if cached_validation:
                print(
                    "  [2/2] Loading cached workflow validation sequence for "
                    f"{sha_fail[:12]} from {validation_cache_file}..."
                )
        except Exception as e:
            print(f"  WARNING:  Validation cache load failed: {e}, will re-analyze")

    if cached_validation:
        validation_sequence = cached_validation.get("validation_sequence", [])
        workflow_validation_context = {
            "id": str(cached_validation.get("id") or issue_id),
            "sha_fail": str(cached_validation.get("sha_fail") or sha_fail),
            "workflow_path": str(
                cached_validation.get("workflow_path") or workflow_path
            ),
            "validation_sequence": validation_sequence,
        }
        print(f"       Found {len(validation_sequence)} validation steps (cached)")
    else:
        print("  [2/2] Analyzing workflow to extract validation sequence...")
        try:
            workflow_validation_context = analyze_workflow_from_benchmark(
                workflow_content=raw_workflow,
                workflow_path=workflow_path,
                repo_path=repo_path,
                llm=llm,
                issue_id=issue_id,
                sha_fail=str(sha_fail or ""),
            )

            validation_sequence = workflow_validation_context.get(
                "validation_sequence", []
            )

            print(f"       Found {len(validation_sequence)} validation steps")
        except Exception as e:
            print(f"       WARNING: Workflow extraction failed: {e}")
            print("       Using fallback: empty validation sequence")
            workflow_validation_context = {
                "workflow_path": str(workflow_path),
                "validation_sequence": [],
            }
            validation_sequence = []

        # Save to global cache
        if sha_fail and validation_sequence:
            try:
                existing_cache = []
                if validation_cache_file.exists():
                    with open(validation_cache_file) as f:
                        existing_cache = json.load(f)

                # Update or append compact workflow-validation context.
                updated = False
                for entry in existing_cache:
                    if entry.get("sha_fail") == sha_fail:
                        entry.clear()
                        entry.update(workflow_validation_context)
                        updated = True
                        break

                if not updated:
                    existing_cache.append(workflow_validation_context)

                validation_cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(validation_cache_file, "w") as f:
                    json.dump(existing_cache, f, indent=2)
                print(f"Saved workflow validation to {validation_cache_file}")
            except Exception as e:
                print(f" Failed to save validation cache: {e}")

    # Allow empty validation_sequence as fallback (decomposition will work in simpler mode)
    if not validation_sequence:
        print(
            "       WARNING: No validation sequence available, using fallback decomposition mode"
        )

    return {
        "context": context,
        "log_analysis": log_analysis,
        "validation_sequence": validation_sequence,
        "workflow_validation_context": workflow_validation_context,
        "workflow_path": workflow_path,
        "workflow_name": str(issue.get("workflow_name") or ""),
        "repo_path": repo_path,
    }


def _has_structured_ci_context(benchmark_context: dict[str, Any]) -> bool:
    """Return True when CILogAnalyzer produced usable structured failure context."""
    context = benchmark_context.get("context") or {}
    log_analysis = benchmark_context.get("log_analysis") or {}
    return bool(
        context.get("overall_failure_reasons")
        or context.get("overall_error_types")
        or context.get("effected_files")
        or context.get("failed_jobs")
        or log_analysis.get("error_context")
        or log_analysis.get("relevant_files")
        or log_analysis.get("error_types")
        or log_analysis.get("failed_job")
        or log_analysis.get("analysis_document")
        or log_analysis.get("overall_ci_summary")
        or log_analysis.get("chunk_summaries")
    )


def validate_required_ci_inputs(benchmark_context: dict[str, Any]) -> bool:
    """Both CI analyzer context and workflow validation sequence are required."""
    has_ci_context = _has_structured_ci_context(benchmark_context)
    has_validation_sequence = bool(benchmark_context.get("validation_sequence"))
    if not has_ci_context:
        print(
            "  ERROR Missing structured CI context from CILogAnalyzer; skipping decomposition"
        )
    if not has_validation_sequence:
        print("  ERROR Missing CI workflow validation sequence; skipping decomposition")
    return has_ci_context and has_validation_sequence


def _compact_context_for_diff_analysis(
    issue: dict,
    benchmark_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Build CI failure context for diff analysis.

    IMPORTANT: Include ALL detailed information from log_analysis so decomposition
    can create SPECIFIC problems, not vague generic ones!
    """
    context = benchmark_context.get("context") or {}
    log_analysis = benchmark_context.get("log_analysis") or {}

    return {
        "issue_id": _issue_id(issue),
        "repo": issue.get("repo_name", issue.get("repo")),
        "workflow_path": benchmark_context.get("workflow_path"),
        # Summary level (quick overview)
        "overall_failure_reasons": context.get("overall_failure_reasons", []),
        "overall_error_types": context.get("overall_error_types", []),
        "failed_jobs": context.get("failed_jobs", []),
        # DETAILED CONTEXT (from CILogAnalyzer) - THIS IS CRITICAL!
        "error_context": log_analysis.get("error_context", []),  # Detailed explanation
        "failure_signals": log_analysis.get("failure_signals", []),  # Observable patterns
        "relevant_files": log_analysis.get("relevant_files", []),  # Files with line numbers
        "error_types": log_analysis.get("error_types", []),  # Detailed with evidence
        "failed_job": log_analysis.get("failed_job", []),  # Job/step/command details
    }


def _is_dependency_file(file_path: str) -> bool:
    """Check if file is a dependency configuration file."""
    dep_files = [
        "pyproject.toml",
        "package.json",
        "requirements.txt",
        "Cargo.toml",
        "pom.xml",
        "build.gradle",
        "setup.py",
    ]
    return any(file_path.endswith(f) for f in dep_files)


def _build_caller_callee_contexts(
    val_group: dict[str, Any],
    filtered_edges: list[dict],
    file_changes_lookup: dict[str, list],
) -> list[dict[str, Any]]:
    """
    Build caller -> callee dependency contexts from graph edges.

    CRITICAL: Only uses edges where BOTH caller AND callee are:
    1. In the changed files (file_changes_lookup)
    2. In this validation group

    This ensures we only analyze real dependencies between files that changed together.

    Args:
        val_group: Validation group with all_files
        filtered_edges: Graph edges filtered to modified files only (caller & callee both modified)
        file_changes_lookup: Map of file -> changes

    Returns:
        List of caller -> callee contexts with changes
    """
    group_files = set(val_group.get("all_files", []))
    contexts = []

    # Group edges by (caller, type) to build caller -> callees structure
    caller_groups = {}
    for edge in filtered_edges:
        caller = edge.get("from")
        callee = edge.get("to")
        dep_type = edge.get("type", "unknown")

        # STRICT FILTER: Both caller AND callee must be in:
        # 1. This validation group (group_files)
        # 2. The changed files (file_changes_lookup)
        if (
            caller in group_files
            and callee in group_files
            and caller in file_changes_lookup
            and callee in file_changes_lookup
        ):
            key = (caller, dep_type)
            if key not in caller_groups:
                caller_groups[key] = {"caller": caller, "type": dep_type, "callees": []}
            if callee not in caller_groups[key]["callees"]:
                caller_groups[key]["callees"].append(callee)

    # Build structured contexts for each caller -> callees relationship
    for (caller, dep_type), group in caller_groups.items():
        callees = group["callees"]

        # Build caller info
        caller_info = {
            "file": caller,
            "changes": _compact_dependency_changes(
                file_changes_lookup.get(caller, []), max_changes=3
            ),
            "role": _classify_file_type_for_role(caller),
        }

        # Build callee infos
        callee_infos = []
        for callee in callees:
            callee_info = {
                "file": callee,
                "changes": _compact_dependency_changes(
                    file_changes_lookup.get(callee, []), max_changes=3
                ),
                "role": _classify_file_type_for_role(callee),
            }
            callee_infos.append(callee_info)

        # Create structured context
        context = {
            "dependency_type": dep_type.upper(),
            "caller": caller_info,
            "callees": callee_infos,
            "cascade_explanation": f"Caller ({caller}) {dep_type.upper()} callees ({len(callees)} files)",
        }

        contexts.append(context)

    return contexts


def _classify_file_type_for_role(file_path: str) -> str:
    """Classify file for role description in caller/callee context."""
    if file_path.endswith("_test.py") or "/tests/" in file_path:
        return "test"
    elif file_path.endswith((".toml", ".yaml", ".yml", ".json", ".ini", ".cfg")):
        return "config"
    elif file_path.endswith((".rst", ".md")):
        return "docs"
    elif file_path.endswith(".py"):
        return "code"
    else:
        return "other"


def _compact_dependency_changes(
    changes: list[dict], max_changes: int = 3
) -> list[dict]:
    """Compact changes to first N with truncated before/after."""
    compacted = []
    for change in changes[:max_changes]:
        compacted.append(
            {
                "line": change.get("line"),
                "before": _compact_text(change.get("before", ""), 200),
                "after": _compact_text(change.get("after", ""), 200),
            }
        )
    return compacted


def merge_chunks_by_validation(
    chunk_findings: list[dict[str, Any]],
    validation_sequence: list[dict[str, Any]],
    structured_chunks: list[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Step 2: Merge chunk findings by validation (deterministic, no LLM).

    Groups all files and changes by validation_order.
    Detects cross-chunk patterns.
    Handles sub-problems (multiple failure types per validation).

    Args:
        chunk_findings: LLM classification output (which files go to which validations)
        validation_sequence: The validation sequence from CI workflow
        structured_chunks: Original parsed diff chunks with actual changes
    """
    validation_groups = {}
    chunk_dependency_lookup = {}
    all_dependency_contexts = []
    if structured_chunks:
        for chunk in structured_chunks:
            dep_cluster = chunk.get("dependency_cluster") or []
            dep_explanation = str(chunk.get("dependency_explanation") or "").strip()
            dep_context = {
                "chunk_index": chunk.get("chunk_index"),
                "dependency_cluster": dep_cluster,
                "dependency_explanation": dep_explanation,
            }
            if (
                dep_cluster
                and dep_explanation
                and dep_explanation != "No dependencies within cluster"
            ):
                all_dependency_contexts.append(dep_context)
                for file_path in dep_cluster:
                    chunk_dependency_lookup.setdefault(file_path, []).append(
                        dep_context
                    )
            for caller_context in chunk.get("dependency_contexts") or []:
                caller = caller_context.get("caller", {})
                callees = caller_context.get("callees", [])
                related_files = [
                    caller.get("file"),
                    *(callee.get("file") for callee in callees),
                ]
                related_files = [file_path for file_path in related_files if file_path]
                if not related_files:
                    continue
                all_dependency_contexts.append(caller_context)
                for file_path in related_files:
                    chunk_dependency_lookup.setdefault(file_path, []).append(
                        caller_context
                    )
    sequence_by_order = {
        str(item.get("order")): item
        for item in validation_sequence
        if item.get("order") is not None
    }

    def _change_scope_summary(entry: dict[str, Any]) -> list[str]:
        value = entry.get("change_scope_summary") or []
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]

    def _extend_change_scope(target: dict[str, Any], entry: dict[str, Any]) -> None:
        target.setdefault("change_scope_summary", [])
        for item in _change_scope_summary(entry):
            if item not in target["change_scope_summary"]:
                target["change_scope_summary"].append(item)

    for chunk in chunk_findings:
        for val_entry in chunk.get("validations_in_this_chunk", []):
            val_order = val_entry.get("validation_order")
            val_cmd = val_entry.get("validation_cmd")
            if (not val_cmd or val_cmd == "unknown") and val_order not in (
                None,
                "unknown",
            ):
                seq_item = sequence_by_order.get(str(val_order), {})
                val_cmd = str(seq_item.get("validation_cmd") or "").strip()
                val_entry["validation_cmd"] = val_cmd
            if val_order in (None, "unknown") or not val_cmd or val_cmd == "unknown":
                continue

            # Check if this validation has multiple failure types
            if val_entry.get("has_multiple_failure_types"):
                # Handle sub-problems
                for sub_problem in val_entry.get("sub_problems", []):
                    # Unique key: validation_order + sub_problem_id
                    sub_id = sub_problem.get("sub_problem_id", "")
                    key = f"{val_order}_{sub_id}" if sub_id else str(val_order)

                    if key not in validation_groups:
                        validation_groups[key] = {
                            "validation_order": val_order,
                            "validation_cmd": val_cmd,
                            "has_sub_problems": True,
                            "sub_problem_id": sub_id,
                            "failure_type": sub_problem.get("failure_type"),
                            "issue_type": sub_problem.get("issue_type")
                            or sub_problem.get("failure_name"),
                            "error_code": sub_problem.get("error_code"),
                            "chunks": [],
                            "all_files": [],
                            "all_changes": [],
                            "has_pattern": False,
                            "total_files": 0,
                            "dependency_contexts": [],
                            "change_scope_summary": [],
                        }

                    validation_groups[key]["chunks"].append(chunk["chunk_index"])
                    validation_groups[key]["all_files"].extend(
                        sub_problem.get("files", [])
                    )
                    # Don't add changes here - they'll be populated from structured_chunks below
                    validation_groups[key]["total_files"] += sub_problem.get(
                        "total_files", len(sub_problem.get("files", []))
                    )
                    _extend_change_scope(
                        validation_groups[key],
                        sub_problem
                        if sub_problem.get("change_scope_summary")
                        else val_entry,
                    )

                    # Cross-chunk pattern detection
                    if sub_problem.get("has_pattern"):
                        validation_groups[key]["has_pattern"] = True
            else:
                # Single failure type - keep distinct issue subtypes separate.
                failure_type = str(val_entry.get("failure_type") or "").strip()
                issue_type = str(
                    val_entry.get("issue_type") or val_entry.get("failure_name") or ""
                ).strip()
                is_cascading = bool(val_entry.get("is_cascading", False))
                dependency_type = str(val_entry.get("dependency_type") or "").strip()
                cascade_explanation = str(
                    val_entry.get("cascade_explanation") or ""
                ).strip()
                key_parts = [str(val_order)]
                if failure_type:
                    key_parts.append(failure_type.lower().replace(" ", "_"))
                if issue_type:
                    key_parts.append(issue_type.lower().replace(" ", "_"))
                key_parts.append("cascading" if is_cascading else "independent")
                if is_cascading and dependency_type:
                    key_parts.append(dependency_type.lower())
                key = "::".join(key_parts)

                if key not in validation_groups:
                    validation_groups[key] = {
                        "validation_order": val_order,
                        "validation_cmd": val_cmd,
                        "failure_type": failure_type,
                        "issue_type": issue_type,
                        "has_sub_problems": False,
                        "chunks": [],
                        "all_files": [],
                        "all_changes": [],
                        "has_pattern": False,
                        "total_files": 0,
                        "dependency_contexts": [],
                        "is_cascading": is_cascading,
                        "dependency_type": dependency_type if is_cascading else "",
                        "cascade_explanation": cascade_explanation
                        if is_cascading
                        else "",
                        "change_type": val_entry.get("change_type", ""),
                        "visibility": val_entry.get("visibility", ""),
                        "change_scope_summary": [],
                    }

                validation_groups[key]["chunks"].append(chunk["chunk_index"])
                validation_groups[key]["all_files"].extend(val_entry.get("files", []))
                # Don't add changes here - they'll be populated from structured_chunks below
                validation_groups[key]["total_files"] += val_entry.get(
                    "total_files", len(val_entry.get("files", []))
                )
                _extend_change_scope(validation_groups[key], val_entry)
                if is_cascading:
                    validation_groups[key]["is_cascading"] = True
                    if dependency_type and not validation_groups[key].get(
                        "dependency_type"
                    ):
                        validation_groups[key]["dependency_type"] = dependency_type
                    if cascade_explanation:
                        existing_explanation = validation_groups[key].get(
                            "cascade_explanation", ""
                        )
                        if cascade_explanation not in existing_explanation:
                            validation_groups[key]["cascade_explanation"] = (
                                f"{existing_explanation}; {cascade_explanation}"
                                if existing_explanation
                                else cascade_explanation
                            )

                # Cross-chunk pattern detection
                if val_entry.get("has_pattern"):
                    validation_groups[key]["has_pattern"] = True

    # CRITICAL FIX: Attach actual file changes to validation groups
    # Build a lookup of file -> changes from the original structured chunks
    file_changes_lookup = {}
    all_changed_files = set()
    dependency_edges = []

    if structured_chunks:
        for chunk in structured_chunks:
            if "files" in chunk:
                for file_info in chunk["files"]:
                    file_path = file_info.get("path", "")
                    if file_path:
                        all_changed_files.add(file_path)
                    if file_path and "changes" in file_info:
                        if file_path not in file_changes_lookup:
                            file_changes_lookup[file_path] = []
                        file_changes_lookup[file_path].extend(file_info["changes"])

            # Extract dependency edges from chunks (for caller -> callee structure)
            if "dependency_graph" in chunk:
                dep_graph = chunk["dependency_graph"]
                if isinstance(dep_graph, dict) and "edges" in dep_graph:
                    dependency_edges.extend(dep_graph["edges"])

    # CRITICAL FILTER: Only include edges where BOTH caller AND callee are in changed files
    # This ensures we only analyze dependencies between files that actually changed together
    filtered_edges = [
        edge
        for edge in dependency_edges
        if edge.get("from") in all_changed_files and edge.get("to") in all_changed_files
    ]

    print(
        f"[DEBUG] Total edges: {len(dependency_edges)}, Filtered (both modified): {len(filtered_edges)}"
    )

    # Now attach changes to each validation group
    for val_group in validation_groups.values():
        val_order = val_group.get("validation_order", "?")
        group_files = val_group.get("all_files", [])
        print(f"[DEBUG] Validation {val_order}: {len(group_files)} files in all_files")
        if group_files:
            print(f"[DEBUG]   Sample files: {group_files[:3]}")

        group_dependency_contexts = []
        for file_path in group_files:
            if file_path in file_changes_lookup:
                # Add all changes for this file
                file_changes = file_changes_lookup[file_path]
                for change in file_changes:
                    # Add file context to each change
                    change_with_file = change.copy()
                    change_with_file["file"] = file_path
                    val_group["all_changes"].append(change_with_file)
            group_dependency_contexts.extend(chunk_dependency_lookup.get(file_path, []))

        print(
            f"[DEBUG] Validation {val_order}: {len(val_group['all_changes'])} changes added"
        )

        # Build caller -> callee dependency contexts from graph edges
        caller_callee_contexts = _build_caller_callee_contexts(
            val_group, filtered_edges, file_changes_lookup
        )

        if caller_callee_contexts:
            val_group["dependency_contexts"] = caller_callee_contexts
        elif group_dependency_contexts:
            # Fallback to old cluster-based approach if no edges
            seen_contexts = set()
            val_group["dependency_contexts"] = []
            for context in group_dependency_contexts:
                if "caller" in context and "callees" in context:
                    caller = context.get("caller", {})
                    callees = context.get("callees", [])
                    key = (
                        context.get("dependency_type", ""),
                        caller.get("file", ""),
                        tuple(callee.get("file") for callee in callees),
                    )
                else:
                    key = (
                        tuple(context.get("dependency_cluster") or []),
                        context.get("dependency_explanation", ""),
                    )
                if key in seen_contexts:
                    continue

                # ENHANCEMENT: Attach actual changes from dependency files
                # This allows LLM to see WHAT CHANGED in config/dependency files
                enriched_context = context.copy()
                dependency_changes = {}

                for dep_file in context.get("dependency_cluster", []):
                    if dep_file in file_changes_lookup and dep_file != file_path:
                        # This is a dependency file (not the current file)
                        # Include its changes so LLM can understand cascading effects
                        dependency_changes[dep_file] = file_changes_lookup[dep_file]

                if dependency_changes:
                    enriched_context["dependency_file_changes"] = dependency_changes

                seen_contexts.add(key)
                val_group["dependency_contexts"].append(enriched_context)

    # Sort by validation order for sequential processing. LLMs should return
    # numeric orders, but keep this deterministic if a string slips through.
    def _validation_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        order = item[1].get("validation_order")
        try:
            return (int(order), str(order))
        except (TypeError, ValueError):
            return (10**9, str(order))

    sorted_groups = dict(sorted(validation_groups.items(), key=_validation_sort_key))

    return {
        "validation_groups": sorted_groups,
        "total_validations": len(
            set(g["validation_order"] for g in sorted_groups.values())
        ),
        "total_groups": len(
            sorted_groups
        ),  # May be > total_validations if sub-problems exist
        "all_changed_files_from_diff": sorted(all_changed_files),
        "all_config_files_from_diff": sorted(
            file_path
            for file_path in all_changed_files
            if _is_dependency_file(file_path)
            or file_path.endswith(
                (".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".lock")
            )
        ),
        "dependency_contexts": all_dependency_contexts,
    }


def _copy_validation_chunk_metadata(
    source: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    """Preserve classification/dependency metadata on deep-analysis chunks."""
    for key in [
        "validation_order",
        "change_type",
        "visibility",
        "is_cascading",
        "dependency_type",
        "cascade_explanation",
    ]:
        if key in source and key not in target:
            target[key] = source.get(key)
    return target


def _chunk_by_dependencies(
    val_group: dict[str, Any],
    dependency_contexts: list[dict],
    max_changes_per_chunk: int,
) -> list[dict[str, Any]]:
    """
    Chunk validation changes by dependency relationships.

    Strategy:
    1. For each dependency context (caller -> callees):
       - Keep caller + callees together in one chunk
       - Include all their changes together
    2. If total changes exceed max, split callees across chunks but keep caller in each

    Args:
        val_group: Validation group
        dependency_contexts: List of caller -> callee contexts
        max_changes_per_chunk: Max changes per chunk

    Returns:
        List of dependency-aware chunks
    """
    chunks = []

    for dep_context in dependency_contexts:
        # Extract caller and callee files
        caller_file = dep_context.get("caller", {}).get("file")
        callee_files = [c.get("file") for c in dep_context.get("callees", [])]
        dep_files = [caller_file] + callee_files if caller_file else callee_files

        # Gather all changes for these files. Keep file groups intact so an
        # individual caller/callee is not cut into unrelated line fragments.
        caller_changes = [
            change
            for change in val_group.get("all_changes", [])
            if caller_file and change.get("file") == caller_file
        ]
        callee_changes = {
            callee_file: [
                change
                for change in val_group.get("all_changes", [])
                if change.get("file") == callee_file
            ]
            for callee_file in callee_files
            if callee_file
        }
        dep_changes = caller_changes + [
            change
            for callee_file in callee_files
            for change in callee_changes.get(callee_file, [])
        ]

        if not dep_changes:
            continue

        # If changes fit in one chunk, create single chunk with dependency context
        if len(dep_changes) <= max_changes_per_chunk:
            chunk = _copy_validation_chunk_metadata(
                val_group,
                {
                    "validation_cmd": val_group.get("validation_cmd", ""),
                    "failure_type": val_group.get("failure_type", ""),
                    "issue_type": val_group.get("issue_type", ""),
                    "all_files": dep_files,
                    "all_changes": dep_changes,
                    "dependency_contexts": [dep_context],  # Attach dependency context
                    "chunk_info": f"Dependency chunk: {caller_file} -> {len(callee_files)} callees",
                },
            )
            chunks.append(chunk)
        else:
            # Split by callee file groups and repeat the complete caller changes
            # in every chunk. A single large file group may exceed the nominal
            # limit because preserving the dependency evidence is more important
            # than slicing related semantic changes apart.
            callee_batches = []
            current_batch = []
            current_count = len(caller_changes)
            effective_limit = max(
                max_changes_per_chunk,
                len(caller_changes) + (1 if callee_files else 0),
            )
            for callee_file in callee_files:
                file_change_count = len(callee_changes.get(callee_file, []))
                if current_batch and current_count + file_change_count > effective_limit:
                    callee_batches.append(current_batch)
                    current_batch = []
                    current_count = len(caller_changes)
                current_batch.append(callee_file)
                current_count += file_change_count
            if current_batch or not callee_files:
                callee_batches.append(current_batch)

            for batch_index, batch_callees in enumerate(callee_batches, 1):
                chunk_changes = list(caller_changes)
                for callee_file in batch_callees:
                    chunk_changes.extend(callee_changes.get(callee_file, []))
                batch_context = copy.deepcopy(dep_context)
                batch_context["callees"] = [
                    callee
                    for callee in dep_context.get("callees", [])
                    if callee.get("file") in batch_callees
                ]
                batch_files = ([caller_file] if caller_file else []) + batch_callees
                chunk = _copy_validation_chunk_metadata(
                    val_group,
                    {
                        "validation_cmd": val_group.get("validation_cmd", ""),
                        "failure_type": val_group.get("failure_type", ""),
                        "issue_type": val_group.get("issue_type", ""),
                        "all_files": batch_files,
                        "all_changes": chunk_changes,
                        "dependency_contexts": [batch_context],
                        "chunk_info": f"Dependency chunk {batch_index}: {caller_file} -> {len(batch_callees)} callees",
                    },
                )
                chunks.append(chunk)

    # Handle remaining changes not covered by dependencies
    all_dep_files = set()
    for dep_context in dependency_contexts:
        caller_file = dep_context.get("caller", {}).get("file")
        callee_files = [c.get("file") for c in dep_context.get("callees", [])]
        if caller_file:
            all_dep_files.add(caller_file)
        all_dep_files.update(callee_files)

    remaining_changes = [
        change
        for change in val_group.get("all_changes", [])
        if change.get("file") not in all_dep_files
    ]

    if remaining_changes:
        # Chunk remaining changes without dependency context
        for start_idx in range(0, len(remaining_changes), max_changes_per_chunk):
            chunk_changes = remaining_changes[
                start_idx : start_idx + max_changes_per_chunk
            ]

            chunk = _copy_validation_chunk_metadata(
                val_group,
                {
                    "validation_cmd": val_group.get("validation_cmd", ""),
                    "failure_type": val_group.get("failure_type", ""),
                    "issue_type": val_group.get("issue_type", ""),
                    "all_files": list(set(ch.get("file") for ch in chunk_changes)),
                    "all_changes": chunk_changes,
                    "dependency_contexts": [],
                    "chunk_info": f"Non-dependency changes {start_idx + 1}-{start_idx + len(chunk_changes)}",
                },
            )
            chunks.append(chunk)

    return (
        chunks if chunks else [val_group]
    )  # Fallback to full group if no chunks created


# ============================================================================
# REMOVED: Old hardcoded limits replaced with model-aware configuration
# Use _get_model_aware_limits() to get current model's actual limits
# ============================================================================


def _compact_text(value: Any, limit: int = 1200) -> str:
    """Compact text fields before placing them in prompts."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _compact_json(value: Any, limit: int = 6000) -> str:
    """Serialize JSON context with a hard character budget."""
    text = json.dumps(value, indent=2, default=str)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0] + "\n  ...\n}"


def _final_verify_config_files(
    validation_groups: dict[str, Any], all_atomic_problems: list[dict[str, Any]]
) -> None:
    """
    Final verification that ALL config files from ground truth are included.

    This runs after all problems are created to catch any config files that
    might have been filtered out during processing.
    """
    config_file_patterns = [
        "pyproject.toml",
        "package.json",
        "requirements.txt",
        "setup.py",
        "Cargo.toml",
        "pom.xml",
        "build.gradle",
        "Gemfile",
        "go.mod",
    ]

    # Prefer original parsed-diff metadata. Falling back to validation_groups
    # alone misses files that were dropped during LLM classification.
    all_config_files = set(validation_groups.get("all_config_files_from_diff") or [])
    groups_data = validation_groups.get("validation_groups", {})

    if not all_config_files:
        for val_order, val_group in groups_data.items():
            for change in val_group.get("all_changes", []):
                file_path = change.get("file", "")
                if any(pattern in file_path for pattern in config_file_patterns):
                    all_config_files.add(file_path)

    if not all_config_files:
        return  # No config files to verify

    # Collect all files from problems
    files_in_problems = set()
    for problem in all_atomic_problems:
        files_in_problems.update(problem.get("affected_files", []))

    # Check for missing config files
    missing_config_files = all_config_files - files_in_problems

    if missing_config_files:
        print("\n CRITICAL ERROR: Config files missing from decomposition!")
        print(f"     Config files in ground truth: {all_config_files}")
        print(f"     Missing from problems: {missing_config_files}")
        print("\n     These files MUST be included in problems (primary or hidden).")
        print(
            "     This is a violation of: NEVER remove config file changes from ground truth"
        )
    else:
        print(
            f"  OK Config file verification: All {len(all_config_files)} config files included in problems"
        )
        # If this happens frequently, we need to strengthen the prompt


def _semantic_cluster_problems(
    problems: list[dict[str, Any]], threshold: float = 0.5
) -> list[list[dict[str, Any]]]:
    """
    Cluster problems by semantic similarity using embeddings.

    Groups problems with cosine similarity > threshold.
    Uses sentence-transformers for fast, quality embeddings.

    Args:
        problems: List of problems to cluster
        threshold: Cosine similarity threshold (0.85 = very similar, 0.75 = moderately similar)

    Returns:
        List of clusters, each cluster is a list of similar problems
    """
    if not problems:
        return []

    if len(problems) == 1:
        return [problems]

    try:
        # Generate text representation for each problem
        problem_texts = []
        for prob in problems:
            # Combine key fields for similarity comparison
            # Note: validation_cmd should be same for all (already grouped by validation)
            text = f"""
Validation: {prob.get("validation_cmd", "")}
Problem: {prob.get("problem", "")}
Root Cause: {prob.get("root_cause", "")}
Fix: {prob.get("how_fixed", "")}
Failure Type: {prob.get("failure_type", "")}
Issue Type: {prob.get("issue_type", "")}
""".strip()
            problem_texts.append(text)

        # Compute embeddings (use lightweight model for speed)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(problem_texts, show_progress_bar=False)

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(embeddings)

        # Cluster using similarity threshold
        clusters = []
        visited = set()

        for i in range(len(problems)):
            if i in visited:
                continue

            # Start new cluster with this problem
            cluster = [problems[i]]
            visited.add(i)

            # Find all problems similar to this one
            for j in range(i + 1, len(problems)):
                if j in visited:
                    continue

                if similarity_matrix[i][j] >= threshold:
                    cluster.append(problems[j])
                    visited.add(j)

            clusters.append(cluster)

        return clusters

    except ImportError:
        # Fallback: no clustering if dependencies not available
        print("    [WARN] sentence-transformers not available, skipping clustering")
        return [[prob] for prob in problems]


def _llm_merge_decision(
    cluster: list[dict[str, Any]], validation_cmd: str, llm: Any
) -> dict[str, Any]:
    """
    LLM analyzes cluster and decides: MERGE, PARTIAL_MERGE, or SEPARATE.

    Decision based on:
    - Are problems describing the SAME underlying issue?
    - Are root causes IDENTICAL?
    - Are fixes following the SAME pattern?

    Returns:
        {
            "action": "merge" | "partial_merge" | "separate",
            "result": List of problems (merged or separate),
            "reasoning": "Why this decision was made"
        }
    """

    if len(cluster) == 1:
        # Single problem, no merge needed
        return {
            "action": "keep_as_is",
            "result": cluster,
            "reasoning": "Single problem in cluster",
        }

    # Build prompt for LLM
    problems_text = []
    for idx, prob in enumerate(cluster, 1):
        files_str = ", ".join(prob.get("affected_files", [])[:5])
        if len(prob.get("affected_files", [])) > 5:
            files_str += f" ... ({len(prob.get('affected_files', []))} total)"

        problems_text.append(
            f"""
Problem {idx}:
  Affected files: {files_str}
  Problem: {prob.get("problem", "N/A")}
  Root cause: {prob.get("root_cause", "N/A")}
  How fixed: {prob.get("how_fixed", "N/A")}
  Failure type: {prob.get("failure_type", "N/A")}
  Issue type: {prob.get("issue_type", "N/A")}
  Cascading: {prob.get("is_cascading", False)}
  Dependency type: {prob.get("dependency_type", "")}
""".strip()
        )

    # Use imported prompt template
    prompt = build_cluster_merge_prompt(
        validation_cmd=validation_cmd,
        cluster_size=len(cluster),
        problems_text=chr(10).join(problems_text),
        strict_json_rules=STRICT_JSON_RULES,
    )

    try:
        # Get model-aware max tokens for merge decision output
        # The issue is NOT token limit - it's the LLM stopping early (finish_reason=stop)
        # So we give it FULL capacity and add better instructions to complete JSON
        model_limits = _get_model_aware_limits(getattr(llm, "model_name", None))
        merge_max_tokens = model_limits["output_safe_tokens"]  # Use FULL capacity!

        print(
            f"        [DEBUG] Merge max_tokens={merge_max_tokens} for {len(cluster)} problems"
        )

        decision = _invoke_json(llm, prompt, max_tokens=merge_max_tokens)

        # NEW STRUCTURE: Process unified response
        # decision = {"problems": [{"merged_problems": [1,2], ...}, ...]}
        output_problems = decision.get("problems", [])

        if not output_problems:
            # Fallback: keep all separate
            print("    [WARN] Empty LLM response, keeping problems separate")
            return {
                "action": "separate",
                "result": cluster,
                "reasoning": "Empty response from LLM",
            }

        # Build result from output_problems
        result = []
        total_merged = 0

        for out_prob in output_problems:
            merged_indices = out_prob.get("merged_problems", [])

            if not merged_indices:
                # Skip empty entries
                continue

            # Get actual problems from cluster (1-based -> 0-based)
            source_problems = [
                cluster[idx - 1] for idx in merged_indices if 0 < idx <= len(cluster)
            ]

            if not source_problems:
                continue

            if len(merged_indices) > 1:
                # MERGED: Multiple problems combined
                all_files = []
                for prob in source_problems:
                    all_files.extend(prob.get("affected_files", []))

                merged = {
                    "affected_files": list(dict.fromkeys(all_files)),  # Deduplicate
                    "problem": out_prob.get(
                        "problem", source_problems[0].get("problem")
                    ),
                    "root_cause": out_prob.get(
                        "root_cause", source_problems[0].get("root_cause")
                    ),
                    "how_fixed": out_prob.get(
                        "how_fixed", source_problems[0].get("how_fixed")
                    ),
                    "why_fix_works": out_prob.get(
                        "why_fix_works", source_problems[0].get("why_fix_works")
                    ),
                    "failure_type": source_problems[0].get("failure_type"),
                    "issue_type": source_problems[0].get("issue_type"),
                    "validation_cmd": source_problems[0].get("validation_cmd"),
                    "validation_order": source_problems[0].get("validation_order"),
                    "problem_type": source_problems[0].get("problem_type"),
                    "is_cascading": source_problems[0].get("is_cascading", False),
                    "dependency_type": source_problems[0].get("dependency_type", ""),
                    "cascade_explanation": source_problems[0].get(
                        "cascade_explanation", ""
                    ),
                    "is_merged": True,
                    "merged_from": [
                        p.get("problem_id", idx)
                        for idx, p in enumerate(source_problems)
                    ],
                    "merge_count": len(source_problems),
                }
                result.append(merged)
                total_merged += len(source_problems)

            else:
                # DISTINCT: Single problem kept as-is
                result.append(source_problems[0])

        # Simple stats: just count how many were actually merged
        merged_count = sum(1 for p in result if p.get("is_merged", False))

        return {
            "result": result,
            "merged_count": merged_count,
            "total_input": len(cluster),
            "total_output": len(result),
        }

    except Exception as e:
        print(f"    [WARN] LLM merge decision failed: {e}, keeping problems separate")
        return {
            "action": "separate",
            "result": cluster,
            "reasoning": f"Error: {e!s}",
        }


def _cluster_and_merge_problems(
    problems: list[dict[str, Any]],
    validation_cmd: str,
    llm: Any,
    similarity_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """
    Cluster similar problems and let LLM decide whether to merge.

    Flow:
    1. Semantic clustering (cosine similarity on embeddings)
    2. LLM analyzes each cluster for merge decision
    3. Returns optimized problem list

    Args:
        problems: List of problems from one validation
        validation_cmd: The validation command for context
        llm: LLM instance
        similarity_threshold: Cosine similarity threshold for clustering

    Returns:
        Optimized list of problems (some merged, some separate)
    """

    if not problems:
        return problems

    if len(problems) == 1:
        return problems

    print(
        f"      Clustering {len(problems)} problems (threshold={similarity_threshold})..."
    )

    # Step 1: Semantic clustering
    clusters = _semantic_cluster_problems(problems, threshold=similarity_threshold)

    multi_problem_clusters = [c for c in clusters if len(c) > 1]
    single_problem_clusters = [c for c in clusters if len(c) == 1]

    print(
        f"        -> {len(clusters)} clusters ({len(multi_problem_clusters)} multi-problem, {len(single_problem_clusters)} single)"
    )

    # Step 2: LLM merge decisions for multi-problem clusters
    optimized = []
    merge_stats = {"merged": 0, "separate": 0}

    for cluster in clusters:
        if len(cluster) == 1:
            # Single problem, keep as-is
            optimized.extend(cluster)
            continue

        # Multi-problem cluster -> LLM decision
        print(f"        Analyzing cluster of {len(cluster)} problems...")
        decision = _llm_merge_decision(cluster, validation_cmd, llm)

        # Stats and logging
        total_input = decision["total_input"]
        total_output = decision["total_output"]
        merged_count = decision["merged_count"]

        if merged_count > 0:
            merge_stats["merged"] += total_input - total_output + merged_count
            print(
                f"          OK MERGED: {total_input} problems -> {total_output} (saved {total_input - total_output})"
            )
        else:
            merge_stats["separate"] += total_input
            print(f"          -> KEPT SEPARATE: {total_input} problems remain distinct")

        optimized.extend(decision["result"])
        time.sleep(0.5)  # Rate limiting

    print(f"      Optimization: {len(problems)} -> {len(optimized)} problems")
    if merge_stats["merged"] > 0:
        print(f"        Merged: {merge_stats['merged']} problems")
    if merge_stats["separate"] > 0:
        print(f"        Kept separate: {merge_stats['separate']}")

    return optimized


def _reorder_by_repair_trajectory(
    problems: list[dict[str, Any]],
    dependency_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Reorder problems using their declared order and analyzed dependencies.

    Before dependency analysis, preserve the order already declared by each
    problem's repair_sequence_index/problem_id, falling back to input order.
    Do not infer priority from filenames, text markers, problem_type, or other
    heuristics.

    After dependency analysis, populate each problem's enabled/enabled_by fields
    and follow dependency_result.repair_sequence exactly. Problem IDs remain
    stable because dependency edges refer to those IDs; only
    repair_sequence_index is updated to reflect the new list position.
    """
    if not problems:
        return problems

    if dependency_result is not None:
        reordered = _populate_dependency_fields(problems, dependency_result)
        for index, problem in enumerate(reordered, 1):
            problem["repair_sequence_index"] = index
    else:
        indexed_problems = list(enumerate(problems))

        def _problem_sort_key(
            indexed_problem: tuple[int, dict[str, Any]],
        ) -> tuple[int, int]:
            input_index, problem = indexed_problem
            declared_order = problem.get(
                "repair_sequence_index", problem.get("problem_id", input_index + 1)
            )
            try:
                declared_order = int(declared_order)
            except (TypeError, ValueError):
                declared_order = input_index + 1
            return (declared_order, input_index)

        reordered = [
            problem
            for _, problem in sorted(indexed_problems, key=_problem_sort_key)
        ]
        for index, problem in enumerate(reordered, 1):
            problem["problem_id"] = index
            problem["repair_sequence_index"] = index

    primary_problems = [
        problem for problem in reordered if problem.get("problem_type") == "primary"
    ]
    hidden_problems = [
        problem for problem in reordered if problem.get("problem_type") != "primary"
    ]
    print(f"    Primary problems: {len(primary_problems)} (files in CI logs)")
    print(f"    Hidden problems: {len(hidden_problems)} (files not in CI logs)")
    print(f"    Total reordered: {len(reordered)}")

    return reordered


def _validation_group_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
    order = item[1].get("validation_order")
    try:
        return (int(order), str(item[0]))
    except (TypeError, ValueError):
        return (10**9, str(item[0]))


def _first_failed_validation_order(groups_data: dict[str, Any]) -> int | None:
    failed_validation_order = None
    for val_group in groups_data.values():
        try:
            val_order = int(val_group.get("validation_order"))
        except (ValueError, TypeError):
            continue
        if failed_validation_order is None or val_order < failed_validation_order:
            failed_validation_order = val_order
    return failed_validation_order


def _validation_info(
    validation_sequence: list[dict[str, Any]], validation_order: Any
) -> dict[str, Any]:
    return next(
        (
            validation
            for validation in validation_sequence
            if str(validation.get("order")) == str(validation_order)
        ),
        {},
    )


def _atomic_changes_data(chunk: dict[str, Any]) -> dict[str, Any]:
    """Keep complete before/after evidence for model-driven atomic analysis.

    Capacity management happens at the validation-change chunk level. Clipping
    individual fields here can silently remove config keys, dependency changes,
    or source adaptations and makes whole-diff coverage impossible.
    """
    changes = chunk.get("all_changes", [])

    return {
        "validation_cmd": chunk.get("validation_cmd", ""),
        "failure_type": chunk.get("failure_type", ""),
        "issue_type": chunk.get("issue_type", ""),
        "classification_change_scope_summary": chunk.get(
            "change_scope_summary", []
        ),
        "files_count": len(
            {change.get("file", "") for change in changes if change.get("file")}
        ),
        "changes_count": len(changes),
        "changes": [
            {
                "file": change.get("file", ""),
                "before": str(change.get("before", "") or ""),
                "after": str(change.get("after", "") or ""),
            }
            for change in changes
        ],
    }


def _dependency_context_for_prompt(chunk: dict[str, Any]) -> str:
    contexts = chunk.get("dependency_contexts") or []
    if not contexts:
        return ""

    # Check if contexts use new caller -> callee structure
    has_caller_callee = any("caller" in ctx and "callees" in ctx for ctx in contexts)

    if has_caller_callee:
        # Use new caller -> callee structure
        return _caller_callee_context_for_prompt(contexts)
    else:
        # Fallback to old cluster-based structure
        return _legacy_cluster_context_for_prompt(contexts)


def _cascading_classification_context(chunk: dict[str, Any]) -> str:
    """
    Format classification results for deep analysis.

    Provides complete classification context including:
    - Files in THIS validation group with their classifications
    - Cross-validation relationships (if cascading)
    - Guidance for root cause analysis
    """
    is_cascading = chunk.get("is_cascading", False)
    dependency_type = chunk.get("dependency_type", "")
    cascade_explanation = chunk.get("cascade_explanation", "")
    issue_type = chunk.get("issue_type", "")
    validation_order = chunk.get("validation_order", "?")
    files = chunk.get("files", [])

    # Build context showing THIS validation group's classification
    context_parts = [
        "## CLASSIFICATION CONTEXT",
        "",
        f"THIS validation group (validation {validation_order}):",
        f"  - Classified issue_type: {issue_type}",
        f"  - Files in this group: {len(files)} file(s)",
        f"  - Cascading: {is_cascading}",
    ]

    if is_cascading and dependency_type and cascade_explanation:
        context_parts.extend(
            [
                f"  - Dependency type: {dependency_type}",
                f"  - Cascade relationship: {cascade_explanation}",
                "",
                "CRITICAL FOR ROOT CAUSE ANALYSIS:",
                "- The cascade_explanation tells you which OTHER validation triggered this",
                "- Analyze BEFORE/AFTER changes to understand the trigger",
                "- Root cause should explain the DEPENDENCY CHANGE that required adaptation",
                "- Different files in THIS group may have DIFFERENT root causes if they",
                "  were triggered by DIFFERENT dependencies!",
                "",
                "ANALYSIS PROCESS:",
                "1. For EACH file in CHANGES, check dependency_context",
                "2. Identify WHICH dependency (if any) triggered that specific file",
                "3. Group files by ROOT CAUSE (which dependency triggered them)",
                "4. Files triggered by DIFFERENT dependencies = SEPARATE problems",
            ]
        )
    else:
        context_parts.extend(
            [
                "",
                "INDEPENDENT validation group:",
                "- No dependency triggers from other validations",
                "- Root cause should focus on DIRECT validation failure",
                "- Analyze BEFORE/AFTER to understand what violated the validation rule",
                "",
                "ANALYSIS PROCESS:",
                "1. For EACH file, identify WHY it failed this validation",
                "2. Group files by ROOT CAUSE",
                "3. Files with DIFFERENT failure reasons = SEPARATE problems",
            ]
        )

    return "\n".join(context_parts)


def _build_cross_validation_context(
    chunk: dict[str, Any], all_validation_groups: list[dict[str, Any]] | None
) -> str:
    """
    Build context showing related validations with their ACTUAL CHANGES.

    Provides COMPLETE context by including:
    - Other validation groups' classifications
    - Their files and BEFORE/AFTER changes
    - How they relate to files in THIS validation
    """
    if not all_validation_groups:
        return ""

    current_validation = chunk.get("validation_order", "?")
    cascade_explanation = chunk.get("cascade_explanation", "").lower()

    # Find validations mentioned in cascade explanation
    related_validations = []
    for group in all_validation_groups:
        val_order = group.get("validation_order")
        if val_order == current_validation:
            continue  # Skip self

        # Check if mentioned in cascade explanation
        if str(val_order) in cascade_explanation:
            related_validations.append(group)

    if not related_validations:
        return ""

    context_parts = [
        "",
        "## RELATED VALIDATION GROUPS (that may have triggered files in THIS validation)",
        "",
    ]

    for group in related_validations[:3]:  # Top 3 most relevant
        val_order = group.get("validation_order", "?")
        issue_type = group.get("issue_type", "unknown")
        files = group.get("files", [])

        context_parts.extend(
            [
                f"VALIDATION {val_order}:",
                f"  Classification: {issue_type}",
                f"  Files ({len(files)}): {', '.join(files[:3])}{'...' if len(files) > 3 else ''}",
            ]
        )

        # Show sample changes from this validation group to understand what triggered
        all_changes = group.get("all_changes", [])
        if all_changes:
            context_parts.append("  Key changes:")
            for change in all_changes[:2]:  # Show 2 sample changes
                file = change.get("file", "")
                before = _compact_text(change.get("before", ""), 60)
                after = _compact_text(change.get("after", ""), 60)
                context_parts.append(f'    {file}: "{before}" -> "{after}"')

        context_parts.append("")

    context_parts.extend(
        [
            "USE THIS TO UNDERSTAND ROOT CAUSES:",
            "- Check dependency_context to see which files in THIS validation were triggered by which validation above",
            "- Files triggered by DIFFERENT validations = DIFFERENT root causes = SEPARATE problems",
            "- Example: If app.py triggered by validation 3, exit_code_test.py triggered by validation 16 -> 2 problems",
        ]
    )

    return "\n".join(context_parts)


def _caller_callee_context_for_prompt(contexts: list[dict[str, Any]]) -> str:
    """Format caller -> callee dependency contexts for prompt."""
    compact_contexts = []

    for context in contexts[:8]:
        caller = context.get("caller", {})
        callees = context.get("callees", [])
        dep_type = context.get("dependency_type", "UNKNOWN")

        if not caller or not callees:
            continue

        context_entry = {
            "dependency_type": dep_type,
            "caller": {
                "file": caller.get("file"),
                "changes": caller.get("changes", [])[:3],  # First 3 changes
                "role": caller.get("role", "unknown"),
            },
            "callees": [
                {
                    "file": callee.get("file"),
                    "changes": callee.get("changes", [])[:3],
                    "role": callee.get("role", "unknown"),
                }
                for callee in callees[:10]  # Limit to 10 callees
            ],
        }
        compact_contexts.append(context_entry)

    if not compact_contexts:
        return ""

    return f"""
DEPENDENCY CONTEXT (Caller -> Callee Structure):

{json.dumps(compact_contexts, indent=2)}

CRITICAL ANALYSIS INSTRUCTIONS:

1. For each dependency:
   - CALLER: The file that triggers changes (config, source code, test)
   - CALLEES: Files that adapt to caller changes
   - RELATIONSHIP: How caller affects callees (CONFIGURES, IMPORTS, TESTS, READS)

2. Check caller changes:
   - What configuration/behavior changed in the caller?
   - Does this change affect the CURRENT validation?

3. Classify the relationship:
   - DEPENDENCY-DRIVEN/CASCADING: The caller change directly required the
     callee adaptation. Name the exact caller, before -> after change, and the
     behavior that forced the callee change.
   - INDEPENDENT: No supplied caller change caused this validation failure.

4. Root-cause wording:
   - For a dependency-driven problem, explain the exact causal chain using only
     names, values, and versions supported by the context.
   - For an independent problem, state that no supplied dependency or
     configuration change triggered it. Do not list or describe unrelated
     callers, packages, versions, files, or changes.
"""


def _legacy_cluster_context_for_prompt(contexts: list[dict[str, Any]]) -> str:
    """Legacy cluster-based dependency context (fallback)."""
    compact_contexts = []
    for context in contexts[:8]:
        explanation = str(context.get("dependency_explanation") or "").strip()
        if not explanation or explanation == "No dependencies within cluster":
            continue

        context_entry = {
            "dependency_cluster": context.get("dependency_cluster", [])[:80],
            "dependency_explanation": _compact_text(explanation, 1800),
        }

        # CRITICAL: Include actual changes from dependency files
        # This allows LLM to determine if changes are truly cascading or independent
        dependency_file_changes = context.get("dependency_file_changes", {})
        if dependency_file_changes:
            # Compact the changes to avoid token bloat
            compact_changes = {}
            for dep_file, changes in dependency_file_changes.items():
                # Show first 3 changes from each dependency file
                compact_changes[dep_file] = [
                    {
                        "line": ch.get("line"),
                        "before": _compact_text(ch.get("before", ""), 200),
                        "after": _compact_text(ch.get("after", ""), 200),
                    }
                    for ch in changes[:3]
                ]
            context_entry["dependency_file_changes"] = compact_changes

        compact_contexts.append(context_entry)

    if not compact_contexts:
        return ""

    return f"""
DEPENDENCY CONTEXT:
{json.dumps(compact_contexts, indent=2)}

CRITICAL ANALYSIS INSTRUCTIONS:
1. Review dependency_file_changes to see WHAT ACTUALLY CHANGED in config/dependency files
2. Determine if the current file's changes are:
   - DEPENDENCY-DRIVEN/CASCADING: Directly required by a supplied dependency or
     configuration change.
   - INDEPENDENT: Not triggered by any supplied dependency or configuration
     change.

3. If DEPENDENCY-DRIVEN/CASCADING:
   - root_cause: Name the exact dependency/configuration before -> after change
     and explain how it triggered the failure.
   - how_fixed: Describe the exact downstream adaptation.
   - For every package addition, removal, replacement, or version change,
     identify the concrete constraint that made the old state fail and explain
     why the complete new constraint or removal satisfies it.
   - Cite only evidence present in dependency_file_changes or the supplied CI
     context. Do not infer deprecation, replacement, incompatibility, or a
     transitive constraint from the edit direction alone. When the evidence is
     insufficient, state that the causal constraint is unspecified.

4. If INDEPENDENT:
   - State that no supplied dependency or configuration change triggered the
     issue.
   - Do not list or describe unrelated dependency files, packages, versions,
     or changes in root_cause/how_fixed.

Use dependency_file_changes to make this determination accurately!
"""


def _full_cascading_context(
    chunk: dict[str, Any],
    all_validation_groups: list[dict[str, Any]] | None,
) -> str:
    """Classification/cascading context plus related-validation context, combined."""
    cascading_context = _cascading_classification_context(chunk)
    cross_validation_context = _build_cross_validation_context(
        chunk, all_validation_groups
    )
    if cross_validation_context:
        cascading_context = cascading_context + "\n" + cross_validation_context
    return cascading_context


def _repair_problem_id_key(problem: dict[str, Any]) -> None:
    """
    Repair the common LLM slip of naming the id key "problem_N" (after the
    item's position in the list) instead of the literal "problem_id" the
    schema asks for. Safe to rename: the value gets overwritten with a
    sequential id downstream regardless (see analyze_validation_groups_with_reasoning).
    """
    if "problem_id" in problem:
        return
    for key in list(problem.keys()):
        if re.fullmatch(r"problem_\d+", key):
            problem["problem_id"] = problem.pop(key)
            return


def _extract_atomic_problems(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        candidates = result
    elif isinstance(result, dict):
        if "problem_id" in result and "atomic_problems" not in result:
            candidates = [result]
        else:
            candidates = result.get("atomic_problems", [])
    else:
        candidates = []

    valid_problems = []
    for problem in candidates if isinstance(candidates, list) else []:
        if isinstance(problem, dict):
            _repair_problem_id_key(problem)
        if isinstance(problem, dict) and "problem_id" in problem:
            valid_problems.append(problem)
        elif isinstance(problem, dict):
            print(
                f"      WARNING: Skipping malformed problem (missing problem_id): {list(problem.keys())[:3]}"
            )
    return valid_problems


def _split_count_for_atomic_chunk(
    prompt_size: int, change_count: int, model_name: str | None = None
) -> int:
    """
    Calculate optimal split count based on model-aware limits.

    Args:
        prompt_size: Current prompt size in chars
        change_count: Number of changes
        model_name: Model name for limit lookup

    Returns:
        Number of splits needed
    """
    model_limits = _get_model_aware_limits(model_name)
    # Target 50% of max chunk size to leave room for CI context overhead
    target_prompt_size = model_limits["diff_chunk_chars"] // 2

    estimated_splits = max(2, (prompt_size // target_prompt_size) + 1)
    max_splits = min(estimated_splits, change_count // 10, 8)
    return max(2, max_splits)


def _split_change_chunk_evenly(
    chunk: dict[str, Any], num_splits: int
) -> list[dict[str, Any]]:
    changes = chunk.get("all_changes", [])
    chunk_size = len(changes) // num_splits
    remainder = len(changes) % num_splits
    sub_chunks = []
    start_idx = 0

    for index in range(num_splits):
        size = chunk_size + (1 if index < remainder else 0)
        end_idx = start_idx + size
        sub_chunks.append({**chunk, "all_changes": changes[start_idx:end_idx]})
        start_idx = end_idx

    return sub_chunks


def _fits_model_capacity(
    prompt: str, chunk: dict[str, Any], model_limits: dict[str, Any]
) -> tuple[bool, str]:
    """
    Single source of truth for whether a chunk's prompt fits model capacity.

    Checks input (50% of input capacity) and output (80% of safe output
    tokens, estimated from the group's file count) budgets. Returns
    (True, "") when it fits, else (False, human-readable reason).
    """
    prompt_size = len(prompt)
    input_tokens = prompt_size / 4  # Rough estimate
    input_threshold = (
        model_limits["atomic_analysis_threshold_chars"] / 4
    )  # 50% of input capacity

    # Derived from all_changes (not all_files): splitting only narrows
    # all_changes, so all_files would stay stuck at the parent group's full
    # count and never shrink as chunks get split smaller.
    changes = chunk.get("all_changes", [])
    num_files = len({c.get("file", "") for c in changes if c.get("file")})
    estimated_output_tokens = (num_files * 1000) + 500
    output_safe_limit = model_limits["output_safe_tokens"] * 0.80  # 80% safety margin

    print(
        f"      Input: {prompt_size:,} chars (~{input_tokens:,.0f} tokens) vs {input_threshold:,.0f} threshold"
    )
    print(
        f"      Output: ~{estimated_output_tokens:,.0f} tokens ({num_files} files) vs {output_safe_limit:,.0f} limit"
    )

    input_ok = input_tokens <= input_threshold
    output_ok = estimated_output_tokens <= output_safe_limit
    if input_ok and output_ok:
        return True, ""

    if not input_ok and not output_ok:
        reason = f"Both input ({input_tokens:,.0f} > {input_threshold:,.0f}) and output ({estimated_output_tokens:,.0f} > {output_safe_limit:,.0f}) exceed limits"
    elif not input_ok:
        reason = f"Input tokens ({input_tokens:,.0f}) exceed 50% threshold ({input_threshold:,.0f})"
    else:
        reason = f"Output tokens ({estimated_output_tokens:,.0f}) exceed safe limit ({output_safe_limit:,.0f})"
    return False, reason


def _split_chunk_for_capacity(
    chunk: dict[str, Any],
    prompt_size: int,
    model_limits: dict[str, Any],
    model_name: str | None,
) -> list[dict[str, Any]]:
    """
    Single splitting strategy, tried in priority order:
    1. Dependency-aware (keeping caller + callees together) when dependency
       contexts exist, including across config/dependency/code types.
    2. By change_type (config/dependency/code) when more than one is present.
    3. Evenly by change count.

    Returns [] when there's only one change left (caller treats as terminal).
    """
    changes = chunk.get("all_changes", [])
    if len(changes) <= 1:
        return []

    num_splits = _split_count_for_atomic_chunk(prompt_size, len(changes), model_name)

    dependency_contexts = chunk.get("dependency_contexts") or []
    if dependency_contexts:
        max_changes_per_chunk = max(1, len(changes) // num_splits)
        sub_chunks = _chunk_by_dependencies(
            chunk, dependency_contexts, max_changes_per_chunk
        )
        if len(sub_chunks) > 1:
            return sub_chunks
        if sub_chunks and sub_chunks[0].get("dependency_contexts"):
            # No safe dependency-preserving split exists. Requeue once as a
            # terminal chunk so the caller sends it intact instead of falling
            # through to type/even splitting.
            sub_chunks[0]["_dependency_preserved_terminal"] = True
            return sub_chunks

    change_type_groups = _group_changes_by_type(changes)
    non_empty_types = [
        t for t in ("config", "dependency", "code") if change_type_groups[t]
    ]

    if len(non_empty_types) > 1:
        print(
            f"      Splitting by type: {len(change_type_groups['config'])} config, "
            f"{len(change_type_groups['dependency'])} dependency, "
            f"{len(change_type_groups['code'])} code"
        )
        return [
            {
                **chunk,
                "all_changes": change_type_groups[change_type],
                "change_type": change_type,
            }
            for change_type in non_empty_types
        ]

    return _split_change_chunk_evenly(chunk, num_splits)


MAX_SPLIT_DEPTH = 5


def _build_atomic_prompt_for(
    chunk: dict[str, Any],
    val_info: dict[str, Any],
    ci_context: dict[str, Any],
    failed_validation_order: Any,
) -> str:
    """Build the atomic-extraction prompt for one chunk (no size logic)."""
    all_validation_groups = chunk.get("_all_validation_groups")
    return build_atomic_prompt(
        ci_context=ci_context,
        failed_validation_order=failed_validation_order,
        val_info=val_info,
        chunk=chunk,
        changes_data=_atomic_changes_data(chunk),
        dependency_context=_dependency_context_for_prompt(chunk),
        cascading_context=_full_cascading_context(chunk, all_validation_groups),
        strict_json_rules=STRICT_JSON_RULES,
        all_validation_groups=all_validation_groups,
    )


def _split_chunk_and_requeue(
    chunk: dict[str, Any],
    prompt_size: int,
    depth: int,
    model_limits: dict[str, Any],
    model_name: str | None,
    queue: list[tuple[dict[str, Any], int]],
) -> None:
    """Split a chunk and push each piece back onto the work queue at depth+1."""
    for sub_chunk in _split_chunk_for_capacity(
        chunk, prompt_size, model_limits, model_name
    ):
        sub_chunk["_all_validation_groups"] = chunk.get("_all_validation_groups")
        sub_chunk.setdefault(
            "dependency_contexts", chunk.get("dependency_contexts", [])
        )
        sub_chunk.setdefault("change_type", chunk.get("change_type", "unknown"))
        queue.append((sub_chunk, depth + 1))


def _build_atomic_problems(
    *,
    chunk: dict[str, Any],
    val_info: dict[str, Any],
    ci_context: dict[str, Any],
    failed_validation_order: Any,
    llm: Any,
    model_limits: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build atomic problems for one validation group.

    Works through a queue of chunks, starting with the whole group. Each
    chunk is measured against model capacity: if it fits, it's sent to the
    LLM; if not (and it still has more than one change, under
    MAX_SPLIT_DEPTH), it's split and its pieces are requeued instead. A
    chunk is also split and requeued if the LLM times out or returns
    nothing for non-empty changes. A chunk that can no longer be split is
    sent to the LLM regardless of size.
    """
    model_name = getattr(llm, "model_name", None)

    dependency_contexts = chunk.get("dependency_contexts", [])
    if dependency_contexts:
        print(f"      Including {len(dependency_contexts)} dependency relationship(s)")

    queue: list[tuple[dict[str, Any], int]] = [(chunk, 0)]
    all_problems: list[dict[str, Any]] = []
    processed = 0

    while queue:
        current, depth = queue.pop(0)
        changes = current.get("all_changes", [])
        if not changes:
            continue

        prompt = _build_atomic_prompt_for(
            current, val_info, ci_context, failed_validation_order
        )
        fits, reason = _fits_model_capacity(prompt, current, model_limits)
        can_split = (
            len(changes) > 1
            and depth < MAX_SPLIT_DEPTH
            and not current.get("_dependency_preserved_terminal")
        )

        if not fits:
            if can_split:
                print(f"      FAIL {reason} - splitting ({len(changes)} changes)")
                _split_chunk_and_requeue(
                    current, len(prompt), depth, model_limits, model_name, queue
                )
                continue
            print(
                f"      FAIL {reason} - cannot split further, using as-is ({len(changes)} change(s))"
            )

        processed += 1
        label = f"chunk_{processed}"
        print(f"      Calling LLM for {label} ({len(changes)} changes)...")

        try:
            result = _invoke_json(
                llm, prompt, max_tokens=model_limits["output_safe_tokens"]
            )

            problems = (
                [] if result == "SPLIT_REQUIRED" else _extract_atomic_problems(result)
            )
        except LLMTransientConnectionError:
            # Network availability says nothing about chunk size. Abort this
            # issue as retryable instead of splitting related changes or
            # returning a partial decomposition.
            print(
                f"      FAIL {label}: connection retries exhausted; "
                "aborting this issue without splitting the chunk"
            )
            raise
        except Exception as exc:
            print(f"      FAIL {label} ERROR: {type(exc).__name__}: {str(exc)[:200]}")
            result, problems = "SPLIT_REQUIRED", []

        needs_retry = result == "SPLIT_REQUIRED" or (changes and not problems)
        if needs_retry and can_split:
            print(f"      {label}: retrying with split")
            _split_chunk_and_requeue(
                current, len(prompt), depth, model_limits, model_name, queue
            )
            continue

        if problems:
            print(f"      OK {label}: {len(problems)} problem(s) extracted")
        all_problems.extend(problems)

        if queue:
            time.sleep(min(2**depth, 8))  # Rate limiting between LLM calls

    return all_problems


def _group_changes_by_type(
    changes: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "config": [],
        "dependency": [],
        "code": [],
    }

    # Debug: Show file paths being classified
    if changes:
        print(f"      [DEBUG] Classifying {len(changes)} changes...")
        sample_files = list(set([c.get("file", "") for c in changes[:5]]))
        print(f"      [DEBUG] Sample files: {sample_files}")

    for change in changes:
        file_path = change.get("file", "")
        before = str(change.get("before", "")).lower()
        after = str(change.get("after", "")).lower()

        if file_path.endswith((".toml", ".yaml", ".yml", ".json", ".ini", ".cfg")):
            groups["config"].append(change)
            print(f"      [DEBUG] Config: {file_path}")
        elif any(term in before or term in after for term in ["import", "package"]):
            groups["dependency"].append(change)
            print(f"      [DEBUG] Dependency: {file_path}")
        else:
            groups["code"].append(change)
            # Only print first few code files to avoid spam
            if len(groups["code"]) <= 3:
                print(f"      [DEBUG] Code: {file_path}")

    return groups


def _filter_dependency_contexts_for_group(
    val_group: dict[str, Any],
) -> dict[str, Any]:
    """
    Filter dependency contexts to ONLY include dependencies involving files in THIS group.

    This ensures we only pass relevant dependency info to atomic analysis,
    keeping token usage focused on what matters for this validation group.
    """
    group_files = set(val_group.get("all_files", []))
    all_dep_contexts = val_group.get("dependency_contexts", [])

    if not group_files or not all_dep_contexts:
        return val_group

    filtered_contexts = []
    for ctx in all_dep_contexts:
        caller = ctx.get("caller", {})
        callees = ctx.get("callees", [])

        caller_file = caller.get("file", "")
        callee_files = {c.get("file", "") for c in callees}

        # Include if caller OR any callee is in THIS group's files
        if caller_file in group_files or group_files.intersection(callee_files):
            filtered_contexts.append(ctx)

    # Create filtered copy
    filtered_group = val_group.copy()
    filtered_group["dependency_contexts"] = filtered_contexts

    if filtered_contexts:
        print(
            f"      Filtered dependency contexts: {len(all_dep_contexts)} -> {len(filtered_contexts)} relevant"
        )

    return filtered_group


def _analyze_one_validation_group(
    *,
    val_order: str,
    val_group: dict[str, Any],
    validation_sequence: list[dict[str, Any]],
    ci_context: dict[str, Any],
    failed_validation_order: Any,
    llm: Any,
) -> list[dict[str, Any]]:
    """
    Analyze one validation group into atomic problems.

    The group is already scoped by validation_cmd + failure_type and carries
    its own changes and dependency context (from merge_chunks_by_validation);
    _build_atomic_problems handles capacity checks and splitting internally.
    """
    print(f"    Validation {val_order}: {val_group.get('validation_cmd', '')}")

    validation_order = val_group.get("validation_order", val_order)
    val_info = _validation_info(validation_sequence, validation_order)
    filtered_group = _filter_dependency_contexts_for_group(val_group)
    model_limits = _get_model_aware_limits(getattr(llm, "model_name", None))

    all_problems = _build_atomic_problems(
        chunk=filtered_group,
        val_info=val_info,
        ci_context=ci_context,
        failed_validation_order=failed_validation_order,
        llm=llm,
        model_limits=model_limits,
    )

    print(f"      OK Total: {len(all_problems)} problem(s)")
    return all_problems


def analyze_validation_groups_with_reasoning(
    validation_groups: dict[str, Any],
    validation_sequence: list[dict[str, Any]],
    ci_context: dict[str, Any],
    llm: Any,
    output_dir: str | Path = "data/back_trs",
) -> dict[str, Any]:
    """
    Analyze validation groups into atomic problems.

    Flow:
    1. Grouped input is already scoped by validation_cmd + failure_type.
    2. Analyze each validation group, chunking only when needed.
    3. Merge chunk-level results back within the same validation so variants of
       one failure family become one atomic problem.
    """
    groups_data = validation_groups.get("validation_groups", {})
    print(f"  Processing {len(groups_data)} validation groups...")

    failed_validation_order = _first_failed_validation_order(groups_data)
    print(f"  First failed validation: {failed_validation_order}")

    # Build summary of all validation groups for context
    # Include classification info AND changes so atomic analysis can see
    # what triggered files in this validation from other validations
    all_validation_groups_summary = []
    for val_order, val_group in sorted(
        groups_data.items(), key=_validation_group_sort_key
    ):
        all_validation_groups_summary.append(
            {
                "validation_order": val_group.get("validation_order", val_order),
                "validation_cmd": val_group.get("validation_cmd", ""),
                "failure_type": val_group.get("failure_type", ""),
                "issue_type": val_group.get("issue_type", ""),
                "files": val_group.get("all_files", []),
                "all_changes": val_group.get(
                    "all_changes", []
                ),  # Include actual changes!
                "is_cascading": val_group.get("is_cascading", False),
                "dependency_type": val_group.get("dependency_type", ""),
                "cascade_explanation": val_group.get("cascade_explanation", ""),
            }
        )

    all_problems = []
    next_id = 1
    for val_order, val_group in sorted(
        groups_data.items(), key=_validation_group_sort_key
    ):
        # SIMPLER: Attach all_validation_groups to val_group once
        # Then it flows down naturally with the chunk
        val_group["_all_validation_groups"] = all_validation_groups_summary
        val_group["_output_dir"] = str(output_dir)

        validation_problems = _analyze_one_validation_group(
            val_order=val_order,
            val_group=val_group,
            validation_sequence=validation_sequence,
            ci_context=ci_context,
            failed_validation_order=failed_validation_order,
            llm=llm,
        )
        for problem in validation_problems:
            problem["problem_id"] = next_id
            next_id += 1
            all_problems.append(problem)

    _final_verify_config_files(validation_groups, all_problems)
    print(f"  OK Total: {len(all_problems)} problems created")

    print("  Reordering problems by repair trajectory...")
    return {"atomic_problems": _reorder_by_repair_trajectory(all_problems)}


def _effective_validation_cmd(validation: dict[str, Any]) -> str:
    return str(
        validation.get("validation_cmd")
        or validation.get("installation_cmd")
        or validation.get("validates")
        or ""
    ).strip()


def _format_validation_sequence(
    validation_sequence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "order": item.get("order"),
            "validates": item.get("validates", ""),
            "validation_cmd": item.get("validation_cmd", ""),
            "effective_cmd": _effective_validation_cmd(item),
            "source": item.get("source", ""),
            "evidence": item.get("evidence", ""),
        }
        for item in validation_sequence
    ]


def _extract_validation_list(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        for key in ["validations", "groups", "result", "data", "validation_groups"]:
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _normalize_classification_validations(
    validations: list[dict[str, Any]],
    validation_sequence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sequence_by_order = {
        str(item.get("order")): item
        for item in validation_sequence
        if item.get("order") is not None
    }

    for validation in validations:
        if not validation.get("validation_order"):
            val_cmd = str(validation.get("validation_cmd", "")).strip()
            for seq_item in validation_sequence:
                effective_cmd = _effective_validation_cmd(seq_item)
                if (
                    seq_item.get("validation_cmd") == val_cmd
                    or effective_cmd == val_cmd
                ):
                    validation["validation_order"] = seq_item.get("order")
                    break

        seq_item = sequence_by_order.get(str(validation.get("validation_order")), {})
        if seq_item and not str(validation.get("validation_cmd") or "").strip():
            validation["validation_cmd"] = _effective_validation_cmd(seq_item)

    return [
        validation for validation in validations if validation.get("validation_order")
    ]


def _chunk_file_paths(chunk: dict[str, Any]) -> list[str]:
    chunk_files = chunk.get("files", [])
    if isinstance(chunk_files, dict):
        return [str(path) for path in chunk_files.keys() if path]
    return [
        str(file_info.get("path") or "")
        for file_info in chunk_files
        if isinstance(file_info, dict) and file_info.get("path")
    ]


def _fallback_validation_for_missing_file(
    validation_sequence: list[dict[str, Any]],
) -> dict[str, Any]:
    if not validation_sequence:
        return {}
    return validation_sequence[0]


def _add_missing_file_fallbacks(
    *,
    valid: list[dict[str, Any]],
    actual_files: list[str],
    validation_sequence: list[dict[str, Any]],
    chunk_index: int,
) -> list[dict[str, Any]]:
    classified_files = {
        str(file_path)
        for entry in valid
        for file_path in (entry.get("files") or [])
        if file_path
    }
    missing_files = [
        file_path for file_path in actual_files if file_path not in classified_files
    ]

    if missing_files:
        print(
            f"    WARNING Chunk {chunk_index}: classifier missed {len(missing_files)} changed file(s); "
            "adding deterministic fallback groups"
        )

    fallback_validation = validation_sequence[0] if validation_sequence else {}
    if not fallback_validation:
        return valid

    for file_path in missing_files:
        is_config = _is_dependency_file(file_path) or file_path.endswith(
            (".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".lock")
        )
        fallback_validation = _fallback_validation_for_missing_file(validation_sequence)
        effective_cmd = _effective_validation_cmd(fallback_validation)
        valid.append(
            {
                "validation_order": fallback_validation.get("order"),
                "validation_cmd": effective_cmd,
                "failure_type": "Configuration/Dependency"
                if is_config
                else "Unclassified Changed File",
                "issue_type": "Diff-backed file omitted by classifier",
                "change_type": "config" if is_config else "code",
                "visibility": "hidden",
                "files": [file_path],
                "total_files": 1,
                "_fallback_reason": "LLM omitted changed file from validation classification",
            }
        )
    return valid


def _estimate_chunk_tokens(chunk_files: Any) -> int:
    """
    Estimate token count for a chunk based on file content.
    Uses the same estimation as utilities/diff_chunker.py
    """

    files = chunk_files.values() if isinstance(chunk_files, dict) else chunk_files
    total_tokens = 0

    for file_info in files:
        # Estimate tokens from changes
        changes = file_info.get("changes", [])
        for change in changes:
            before_text = change.get("before", "")
            after_text = change.get("after", "")
            total_tokens += estimate_tokens(before_text + after_text)

        # Add overhead for file metadata
        total_tokens += 50  # File name, structure, etc.

    return total_tokens


def _split_structured_chunk_by_size(
    chunk: dict[str, Any],
    target_max_tokens: int = 50000,
) -> list[dict[str, Any]]:
    """
    Split structured chunk into smaller chunks based on token size.

    Strategy:
    1. Estimate tokens for each file
    2. Bin-packing algorithm to group files under target size
    3. Ensure balanced splits

    Args:
        chunk: Structured chunk with files and metadata
        target_max_tokens: Maximum tokens per chunk (default: 50k)

    Returns:
        List of sub-chunks, each under target_max_tokens
    """

    files = chunk.get("files", [])
    files_list = list(files.items()) if isinstance(files, dict) else list(files)

    if not files_list:
        return []

    # Calculate token size for each file
    file_sizes = []
    for file_key, file_info in files_list:
        file_tokens = 0
        changes = file_info.get("changes", [])
        for change in changes:
            before_text = change.get("before", "")
            after_text = change.get("after", "")
            file_tokens += estimate_tokens(before_text + after_text)
        file_tokens += 50  # Metadata overhead
        file_sizes.append((file_key, file_info, file_tokens))

    # Sort by size (largest first) for better bin packing
    file_sizes.sort(key=lambda x: x[2], reverse=True)

    # Bin packing: Group files into chunks under target size
    sub_chunks = []
    current_chunk_files = []
    current_chunk_tokens = 0

    for file_key, file_info, file_tokens in file_sizes:
        # If single file exceeds target, give it its own chunk
        if file_tokens > target_max_tokens:
            if current_chunk_files:
                sub_chunks.append(current_chunk_files)
                current_chunk_files = []
                current_chunk_tokens = 0

            sub_chunks.append([(file_key, file_info)])
            continue

        # If adding this file would exceed target, start new chunk
        if (
            current_chunk_tokens + file_tokens > target_max_tokens
            and current_chunk_files
        ):
            sub_chunks.append(current_chunk_files)
            current_chunk_files = []
            current_chunk_tokens = 0

        current_chunk_files.append((file_key, file_info))
        current_chunk_tokens += file_tokens

    # Add remaining files
    if current_chunk_files:
        sub_chunks.append(current_chunk_files)

    # If only one chunk, fall back to simple half-split
    if len(sub_chunks) <= 1:
        half = len(files_list) // 2
        if half == 0:
            half = 1
        sub_chunks = [files_list[:half], files_list[half:]]

    # Convert to structured chunks
    def change_count(file_records: Any) -> int:
        return sum(len(f[1].get("changes", [])) for f in file_records)

    def with_metadata(chunk_files: list) -> dict[str, Any]:
        if isinstance(files, dict):
            chunk_files_dict = dict(chunk_files)
        else:
            chunk_files_dict = [f[1] for f in chunk_files]

        return {
            "dependency_cluster": chunk.get("dependency_cluster"),
            "dependency_explanation": chunk.get("dependency_explanation"),
            "is_partial_cluster": chunk.get("is_partial_cluster", True),
            "files": chunk_files_dict,
            "total_files": len(chunk_files),
            "total_changes": change_count(chunk_files),
            "estimated_tokens": _estimate_chunk_tokens(chunk_files_dict),
        }

    result_chunks = [with_metadata(sc) for sc in sub_chunks if sc]

    # Log split details
    print(f"      Smart split: {len(files_list)} files -> {len(result_chunks)} chunks")
    for i, rc in enumerate(result_chunks, 1):
        print(
            f"        Chunk {i}: {rc.get('total_files', 0)} files, ~{rc.get('estimated_tokens', 0):,} tokens"
        )

    return result_chunks


def _is_token_limit_error(error: Exception) -> bool:
    error_msg = str(error).lower()
    return any(
        keyword in error_msg
        for keyword in ["token", "length", "limit", "too long", "maximum"]
    )


def classify_chunk_with_fallback(
    chunk: dict,
    chunk_index: int,
    total_chunks: int,
    visible_failure_context: dict,
    validation_sequence: list,
    llm: Any,
) -> list[dict]:
    """
    Classify a chunk with automatic fallback to smaller chunks if token limit hit.

    NEW: Routes to specialized classification based on dependency presence:
    - With dependencies: Use dependency-focused analysis
    - Without dependencies: Use regular file-by-file classification

    Strategy:
    - Try with current chunk size
    - If token limit -> split in half and retry recursively
    - Works down to 1 file if needed
    """

    files_in_chunk = chunk.get("total_files", 0)

    if files_in_chunk == 0:
        return []

    # Check for caller -> callee dependency contexts
    dependency_contexts = chunk.get("dependency_contexts", [])
    has_caller_callee = any(
        "caller" in ctx and "callees" in ctx for ctx in dependency_contexts
    )

    # ROUTE TO SPECIALIZED CLASSIFICATION
    if has_caller_callee and dependency_contexts:
        # PATH 1: Dependency-aware classification
        # Chunk already contains caller + callees + all changes
        # Use specialized dependency analysis instructions
        return _classify_chunk_with_dependencies(
            chunk=chunk,
            dependency_contexts=dependency_contexts,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            visible_failure_context=visible_failure_context,
            validation_sequence=validation_sequence,
            llm=llm,
        )
    else:
        # PATH 2: Regular classification
        # Chunk contains independent files
        # Use regular file-by-file classification
        return _classify_chunk_regular(
            chunk=chunk,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            visible_failure_context=visible_failure_context,
            validation_sequence=validation_sequence,
            llm=llm,
        )


def _classify_chunk_with_dependencies(
    chunk: dict,
    dependency_contexts: list[dict],
    chunk_index: int,
    total_chunks: int,
    visible_failure_context: dict,
    validation_sequence: list,
    llm: Any,
) -> list[dict]:
    """
    Specialized classification for chunks WITH caller -> callee dependencies.

    Analyzes dependency relationships FIRST, then classifies as cascading or independent.
    """
    files_in_chunk = chunk.get("total_files", 0)
    ci_visible_files = [
        rf.get("file", "") for rf in visible_failure_context.get("relevant_files", [])
    ]
    formatted_validations = _format_validation_sequence(validation_sequence)

    # Format compact dependency context
    dependency_info = _format_caller_callee_for_dependency_classification(
        dependency_contexts
    )

    # Use imported prompt template
    prompt = build_classification_prompt_with_dependencies(
        ci_failure_context=visible_failure_context,
        ci_visible_files=ci_visible_files,
        formatted_validations=formatted_validations,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        files_in_chunk=files_in_chunk,
        formatted_chunk=format_structured_for_llm(
            chunk,
            max_changes_per_file=None,
            max_chars_per_value=None,
        ),
        dependency_info=dependency_info,
        strict_json_rules=STRICT_JSON_RULES,
    )

    try:
        output_safe_tokens = _classification_output_tokens(
            getattr(llm, "model_name", None)
        )

        result = _invoke_json(llm, prompt, max_tokens=output_safe_tokens)
        valid = _normalize_classification_validations(
            _extract_validation_list(result),
            validation_sequence,
        )

        print(
            f"    OK Chunk {chunk_index} DEPENDENCY-AWARE ({files_in_chunk} files): {len(valid)} validation groups"
        )
        return valid

    except Exception as e:
        if _is_token_limit_error(e):
            print(
                f"    WARNING Token limit hit with {files_in_chunk} files, falling back to regular classification..."
            )
            # Fallback to regular classification if dependency analysis is too large
            return _classify_chunk_regular(
                chunk,
                chunk_index,
                total_chunks,
                visible_failure_context,
                validation_sequence,
                llm,
            )
        else:
            print(
                f"    FAIL Chunk {chunk_index} dependency classification failed: {str(e)[:100]}"
            )
            return []


def _format_caller_callee_for_dependency_classification(
    dependency_contexts: list[dict],
) -> str:
    """
    Format caller -> callee contexts for semantic dependency analysis.

    Provides BEFORE/AFTER context for both caller and callees to enable
    semantic understanding of cascading relationships.

    Shows:
    - All significant caller changes (up to 5)
    - Pattern analysis across callees (common changes)
    - BEFORE/AFTER context for impact analysis
    """
    formatted = []

    for idx, ctx in enumerate(dependency_contexts, 1):
        caller = ctx.get("caller", {})
        callees = ctx.get("callees", [])
        dep_type = ctx.get("dependency_type", "UNKNOWN")

        if not caller or not callees:
            continue

        caller_file = caller.get("file", "unknown")
        caller_changes = caller.get("changes", [])
        caller_role = caller.get("role", "unknown")

        # Show up to 5 significant caller changes (for semantic context)
        caller_change_details = []
        for ch in caller_changes[:5]:
            line_num = ch.get("line", "?")
            before = _compact_text(ch.get("before", ""), 100)
            after = _compact_text(ch.get("after", ""), 100)
            caller_change_details.append(
                f'    Line {line_num}: "{before}" -> "{after}"'
            )

        if len(caller_changes) > 5:
            caller_change_details.append(
                f"    ... and {len(caller_changes) - 5} more changes"
            )

        caller_changes_str = (
            "\n".join(caller_change_details)
            if caller_change_details
            else "    No changes shown"
        )

        # Analyze PATTERN across callees (first 5 detailed, then summary)
        callee_pattern_analysis = []

        # Show first 3 callees in detail
        for callee in callees[:3]:
            callee_file = callee.get("file", "unknown")
            callee_role = callee.get("role", "unknown")
            callee_changes = callee.get("changes", [])

            if callee_changes:
                # Show first 2 changes from this callee
                for ch in callee_changes[:2]:
                    before = _compact_text(ch.get("before", ""), 80)
                    after = _compact_text(ch.get("after", ""), 80)
                    callee_pattern_analysis.append(
                        f'    {callee_file} ({callee_role}) Line {ch.get("line", "?")}: "{before}" -> "{after}"'
                    )

        # Detect common patterns across ALL callees
        if len(callees) > 3:
            callee_pattern_analysis.append(
                f"\n    PATTERN ACROSS {len(callees)} CALLEES:"
            )

            # Analyze if there's a common before/after pattern
            common_befores = []
            common_afters = []
            for callee in callees:
                for ch in callee.get("changes", [])[:1]:  # First change from each
                    common_befores.append(_compact_text(ch.get("before", ""), 40))
                    common_afters.append(_compact_text(ch.get("after", ""), 40))

            # Show pattern if multiple callees exist
            if common_befores and common_afters:
                callee_pattern_analysis.append(
                    f'    Common BEFORE pattern: "{common_befores[0]}" (example from first file)'
                )
                callee_pattern_analysis.append(
                    f'    Common AFTER pattern: "{common_afters[0]}" (example from first file)'
                )
                callee_pattern_analysis.append(
                    f"    ... similar pattern across remaining {len(callees) - 1} files"
                )

        callee_pattern_str = (
            "\n".join(callee_pattern_analysis)
            if callee_pattern_analysis
            else "    No pattern detected"
        )

        formatted.append(f"""
DEPENDENCY {idx}: {dep_type}

CALLER: {caller_file} ({caller_role})
  BEFORE/AFTER Changes:
{caller_changes_str}

CALLEES: {len(callees)} files total
  BEFORE/AFTER Pattern:
{callee_pattern_str}

RELATIONSHIP: Caller {dep_type.lower()}s callees

ANALYSIS TASK:
1. What semantic change happened in CALLER?
2. What semantic pattern changed across CALLEES?
3. Is CALLER change adapting to CALLEE changes (cascading)?
4. Or are CALLEES adapting to CALLER change (configures/triggers)?
5. What is the ROOT CAUSE - which change happened first conceptually?
6. What is the actual PROBLEM being fixed (not just validation name)?
""")

    return "\n".join(formatted) if formatted else "No dependency contexts available"


def _classify_chunk_regular(
    chunk: dict,
    chunk_index: int,
    total_chunks: int,
    visible_failure_context: dict,
    validation_sequence: list,
    llm: Any,
) -> list[dict]:
    """
    Regular classification for chunks WITHOUT dependencies.

    Analyzes files independently based on their changes.
    """
    files_in_chunk = chunk.get("total_files", 0)
    ci_visible_files = [
        rf.get("file", "") for rf in visible_failure_context.get("relevant_files", [])
    ]
    formatted_validations = _format_validation_sequence(validation_sequence)

    # Use imported prompt template
    prompt = build_classification_prompt_regular(
        ci_failure_context=visible_failure_context,
        ci_visible_files=ci_visible_files,
        formatted_validations=formatted_validations,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        files_in_chunk=files_in_chunk,
        formatted_chunk=format_structured_for_llm(
            chunk,
            max_changes_per_file=None,
            max_chars_per_value=None,
        ),
        strict_json_rules=STRICT_JSON_RULES,
    )

    try:
        output_safe_tokens = _classification_output_tokens(
            getattr(llm, "model_name", None)
        )

        # Try classification. The output token budget uses the selected model's
        # configured safe output limit; file count is controlled earlier by
        # chunk_structured_diff().
        result = _invoke_json(
            llm,
            prompt,
            max_tokens=output_safe_tokens,
        )

        valid = _normalize_classification_validations(
            _extract_validation_list(result),
            validation_sequence,
        )

        # Enforce ground-truth diff coverage. The prompt asks the model to
        # include every file, but setup/config files are too important to trust
        # to prompt compliance alone.
        valid = _add_missing_file_fallbacks(
            valid=valid,
            actual_files=_chunk_file_paths(chunk),
            validation_sequence=validation_sequence,
            chunk_index=chunk_index,
        )

        print(
            f"    OK Chunk {chunk_index} ({files_in_chunk} files): {len(valid)} validation groups"
        )
        return valid

    except Exception as e:
        if _is_token_limit_error(e):
            print(f"    WARNING Token limit hit with {files_in_chunk} files")

            # Can't split further
            if files_in_chunk <= 1:
                print("    FAIL Cannot split 1 file further, skipping")
                return []

            # Use smart size-aware splitting
            print("    -> Using smart size-aware splitting...")

            # Calculate target based on model's context window
            model_name = getattr(llm, "memci_model_key", None) or getattr(
                llm, "model_name", None
            )
            config = get_model_config(model_name)
            target_tokens = (
                config.get("max_input_tokens", 100000) // 2
            )  # Conservative split

            sub_chunks = _split_structured_chunk_by_size(
                chunk, target_max_tokens=target_tokens
            )

            if not sub_chunks:
                print("    FAIL Could not split chunk")
                return []

            # Recursively process all sub-chunks
            all_results = []
            for i, sub_chunk in enumerate(sub_chunks, 1):
                print(f"    -> Processing sub-chunk {i}/{len(sub_chunks)}...")
                sub_result = classify_chunk_with_fallback(
                    sub_chunk,
                    chunk_index,
                    total_chunks,
                    visible_failure_context,
                    validation_sequence,
                    llm,
                )
                all_results.extend(sub_result)

            return all_results
        else:
            # Other error - log and return empty
            print(f"    FAIL Chunk {chunk_index} failed: {str(e)[:100]}")
            return []


def analyze_diff_chunks(
    issue: dict,
    benchmark_context: dict[str, Any],
    llm: Any,
    output_dir: str | Path = "data/back_trs",
) -> dict[str, Any]:
    """
    Three-step diff analysis with deterministic pre-processing:
    0. Parse diff into structured format (deterministic, no LLM)
    1. Chunk and classify by validation (per chunk, LLM with auto-fallback)
    2. Merge by validation (deterministic)
    3. Deep reasoning with full context (LLM)
    """
    diff = str(issue.get("diff") or "")
    if not diff.strip():
        raise ValueError(f"Issue {_issue_id(issue)} has no ground-truth diff")

    # Step 0: Deterministic diff parsing (NEW!)
    print("  Step 0: Parsing diff into structured format...")

    structured_diff = parse_diff_to_structured(diff)
    total_files = structured_diff["total_files"]
    total_changes = structured_diff["total_changes"]
    print(f"    Parsed {total_files} files with {total_changes} changes")

    # Step 0.5: Build dependency graph (NEW!)
    print("  Step 0.5: Building file dependency graph...")

    dependency_graph = build_dependency_graph_from_structured_diff(
        structured_diff, repo_path=benchmark_context.get("repo_path")
    )
    total_clusters = len(dependency_graph.get("clusters", []))
    total_edges = len(dependency_graph.get("edges", []))
    print(f"    Found {total_clusters} dependency clusters with {total_edges} edges")

    # Chunk by file count (not char count) - cleaner and more predictable
    # Now model-aware: minimax=80 files, GLM=150 files
    # NOW TOKEN + DEPENDENCY-AWARE: keeps related files together while respecting token limits
    model_name = llm.model_name if hasattr(llm, "model_name") else "glm-5.2"
    model_limits = _get_model_aware_limits(model_name)
    max_files = model_limits["max_files_per_chunk"]

    chunks = chunk_structured_diff(
        structured_diff,
        max_files_per_chunk=max_files,  # Fallback for non-dependency chunks
        dependency_graph=dependency_graph,
        model_name=model_name,  # NEW: Token-aware chunking
    )
    print(f"    Using token + dependency-aware chunking (model: {model_name})")
    if not chunks:
        raise ValueError(
            f"Issue {_issue_id(issue)} ground-truth diff could not be chunked"
        )

    visible_failure_context = _compact_context_for_diff_analysis(
        issue, benchmark_context
    )
    validation_sequence = benchmark_context.get("validation_sequence") or []
    chunk_findings: list[dict[str, Any]] = []

    print(
        f"  Step 1: Classifying patch changes by repaired validation ({total_files} files in {len(chunks)} chunk(s))..."
    )

    for index, chunk in enumerate(chunks, start=1):
        # Use fallback function with automatic splitting
        # Now model-aware: minimax=80 files, GLM=150 files
        validations = classify_chunk_with_fallback(
            chunk=chunk,
            chunk_index=index,
            total_chunks=len(chunks),
            visible_failure_context=visible_failure_context,
            validation_sequence=validation_sequence,
            llm=llm,
        )

        if validations:
            chunk_findings.append(
                {"chunk_index": index, "validations_in_this_chunk": validations}
            )

    # Step 2: Merge by validation (deterministic)
    print("  Step 2: Merging chunks by validation...")
    validation_groups = merge_chunks_by_validation(
        chunk_findings, validation_sequence, chunks
    )
    print(
        f"    Found {validation_groups['total_groups']} groups from {validation_groups['total_validations']} validations"
    )

    # Step 2.5: CI-Diff Correlation (deterministic)
    print("  Step 2.5: Analyzing CI-Diff correlation (layered structure)...")
    ci_context = _compact_context_for_diff_analysis(issue, benchmark_context)

    # Step 3: Deep reasoning with full context + correlation
    print("  Step 3: Deep reasoning with correlation context...")
    reasoning_result = analyze_validation_groups_with_reasoning(
        validation_groups,
        validation_sequence,
        ci_context,
        llm,
        output_dir=output_dir,
    )
    # Log results
    atomic_problems = reasoning_result.get("atomic_problems", [])
    if atomic_problems:
        print(f"  OK Identified {len(atomic_problems)} atomic problems")
    else:
        print("  WARNING: No atomic problems identified")
    return {
        "mode": "structured_diff_3step",
        "total_files": total_files,
        "total_changes": total_changes,
        "chunk_count": len(chunks),
        "chunk_findings": chunk_findings,
        "validation_groups": validation_groups,
        "atomic_problems": atomic_problems,
        "sequential_workflow_metadata": reasoning_result.get(
            "sequential_workflow_metadata", {}
        ),
    }


def decompose_issue(
    issue: dict, llm, output_dir: str | Path = "data/back_trs"
) -> dict:
    """
    Three-step reverse engineering from CI failure + ground truth diff:

    1. Classify chunks by validation (per chunk)
    2. Merge by validation (deterministic)
    3. Deep reasoning with full context (ConRAD/STAIR style)

    Returns specific, actionable atomic problems for mini-swe-agent.
    """

    issue_id = _issue_id(issue) or "?"
    print(f"\n{'=' * 80}")
    print(f"Reverse Engineering Issue {issue_id}")
    print(f"  Repo: {issue.get('repo_name', issue.get('repo', '?'))}")
    print(f"  Changed files: {len(issue.get('changed_files', []))}")
    print(f"{'=' * 80}")

    try:
        print("  Fetching benchmark CI context and validation sequence...")
        benchmark_context = build_benchmark_ci_context(
            issue,
            llm=llm,
            output_dir=output_dir,
        )
        if not validate_required_ci_inputs(benchmark_context):
            return {}

        # Three-step analysis
        diff_context = analyze_diff_chunks(
            issue, benchmark_context, llm, output_dir=output_dir
        )

        atomic_problems = diff_context.get("atomic_problems", [])
        diff_context.get("sequential_workflow_metadata", {})

        if not atomic_problems:
            print("  WARNING: No atomic problems identified")
            return {}

        print(
            f"  OK Identified {len(atomic_problems)} atomic problems (before merging)"
        )

        # NEW: Auto-cluster and merge similar problems
        print("  Step 4: Clustering and merging similar problems...")
        validation_groups_for_merge = defaultdict(list)
        for prob in atomic_problems:
            validation_cmd = prob.get("validation_cmd", "unknown")
            validation_groups_for_merge[validation_cmd].append(prob)

        print(
            f"    Grouped into {len(validation_groups_for_merge)} validation commands"
        )

        optimized_problems = []
        for validation_cmd, val_problems in validation_groups_for_merge.items():
            if len(val_problems) > 1:
                print(f"    {validation_cmd}: {len(val_problems)} problems")
                # Apply clustering + LLM merge
                optimized = _cluster_and_merge_problems(
                    val_problems,
                    validation_cmd=validation_cmd,
                    llm=llm,
                    similarity_threshold=0.85,  # High threshold for similar problems
                )
                print(f"      -> Optimized to {len(optimized)} problems")
                optimized_problems.extend(optimized)
            else:
                # Single problem, keep as-is
                optimized_problems.extend(val_problems)

        # Reorder after merging
        print("  Step 5: Reordering by repair trajectory...")
        final_problems = _reorder_by_repair_trajectory(optimized_problems)

        print(
            f"  OK Final: {len(final_problems)} problems (merged {len(atomic_problems) - len(final_problems)} duplicates)"
        )

        # Build final result
        context = benchmark_context.get("context", {})
        log_analysis = benchmark_context.get("log_analysis", {})

        result = {
            "original_issue_id": issue_id,
            "sha_fail": issue.get("sha_fail"),
            "repo": issue.get("repo_name", issue.get("repo")),
            "original_error_type": issue.get("error_type"),
            # Final problems after merging and reordering
            "problems": final_problems,
            "total_problems": len(final_problems),
            "raw_atomic_problems_count": len(atomic_problems),
            "total_changed_files": len(issue.get("changed_files", [])),
            # Benchmark CI context (cleaned - no redundancy)
            "benchmark_ci_context": {
                "workflow_path": benchmark_context.get("workflow_path"),
                "workflow_name": benchmark_context.get("workflow_name"),
                "validation_sequence": benchmark_context.get("validation_sequence", []),
                # Summary level (for quick access)
                "overall_failure_reasons": context.get("overall_failure_reasons", []),
                "overall_error_types": context.get("overall_error_types", []),
                # Detailed analysis (structured)
                "error_types": log_analysis.get(
                    "error_types", []
                ),  # Detailed with subcategory + evidence
                "relevant_files": log_analysis.get(
                    "relevant_files", []
                ),  # Files with line numbers
                "failed_jobs": log_analysis.get(
                    "failed_job", []
                ),  # Job/step/command info
            },
            # Structured diff analysis metadata
            "diff_analysis_context": {
                "mode": diff_context.get("mode"),
                "total_files": diff_context.get("total_files", 0),
                "total_changes": diff_context.get("total_changes", 0),
                "chunk_count": diff_context.get("chunk_count", 0),
                "validation_groups_count": diff_context.get(
                    "validation_groups", {}
                ).get("total_groups", 0),
                "validation_groups": diff_context.get("validation_groups", {}).get(
                    "validation_groups", {}
                ),
            },
        }

        return result

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"  ERROR Failed to decompose: {e}")
        print("\n--- FULL ERROR TRACE ---")
        print(error_trace)
        print("--- END TRACE ---\n")
        return {
            "error": "DECOMPOSITION_ERROR",
            "error_message": str(e),
            "error_trace": error_trace,
            "error_type": type(e).__name__,
            "original_issue_id": _issue_id(issue),
            "sha_fail": issue.get("sha_fail"),
        }


def generate_l1_l2_l3_pipeline(
    decomposed_result: dict, llm, output_dir: str = "data/back_trs"
) -> dict:
    """
    Full L1/L2/L3 generation pipeline.

    Stage 1: Deduplicate similar problems (mechanical)
    Stage 2: Detect dependencies (LLM)
    Stage 3: Generate L1 file-level (mechanical - reuse data)
    Stage 4: Generate L2 repair sequence (LLM)
    Stage 5: Generate L3 analysis (LLM)

    Args:
        decomposed_result: Output from decompose_issue()
        llm: LLM instance for prompts

    Returns:
        Dictionary with l1, l2, l3 sections
    """

    issue_id = decomposed_result.get("original_issue_id", "?")
    if "error" in decomposed_result or not decomposed_result.get("problems"):
        issue_id = str(
            decomposed_result.get("original_issue_id")
            or decomposed_result.get("issue_id")
            or "unknown"
        )
        repo = decomposed_result.get("repo", "unknown")
        workflow_path = decomposed_result.get("benchmark_ci_context", {}).get(
            "workflow_path",
            decomposed_result.get("workflow_path", "unknown"),
        )
        workflow_name = workflow_path.split("/")[-1] if workflow_path else ""
        note = (
            decomposed_result.get("error") or "Backward decomposition found no problems"
        )
        result = {
            "issue_id": issue_id,
            "repo": repo,
            "workflow_path": workflow_path,
            "l1_memory": {
                "issue_id": issue_id,
                "repo": repo,
                "repo_owner": repo.split("/")[0] if "/" in repo else "unknown",
                "repo_name": repo.split("/")[1] if "/" in repo else repo,
                "workflow": workflow_path,
                "workflow_name": workflow_name,
                "changed_files": decomposed_result.get("changed_files", []),
                "problems": [],
                "note": note,
            },
            "l2_memory": {
                "issue_id": issue_id,
                "repo": repo,
                "repo_name": repo.split("/")[1] if "/" in repo else repo,
                "workflow": workflow_path,
                "total_problems": 0,
                "failure_identify": [],
                "repair_strategies": [],
                "note": note,
            },
            "l3_memory": {
                "issue_id": issue_id,
                "repo": repo,
                "repo_name": repo.split("/")[1] if "/" in repo else repo,
                "workflow": workflow_path,
                "universal_patterns": [
                    {
                        "pattern_id": f"no-backward-problems-{issue_id}",
                        "failure_type": "no_backward_problems",
                        "failure_pattern": "",
                        "problem": "",
                        "reasoning": note,
                        "when_to_apply": "",
                        "signals": [],
                        "universal_fix": {"approach": "", "steps": []},
                        "examples": [],
                        "no_decomposed_problems": True,
                    }
                ],
            },
            "metadata": {
                "original_problems_count": 0,
                "deduplicated_count": 0,
                "l1_problems": 0,
                "l2_strategies": 0,
                "l3_patterns": 1,
            },
        }
        _append_to_memory_files(result, output_dir=output_dir)
        return result

    print(f"\n{'=' * 80}")
    print(f"L1/L2/L3 Pipeline for Issue {issue_id}")
    print(f"{'=' * 80}")

    # Stage 1: Deduplicate (mechanical)
    print("\n[Stage 1/5] Clustering and optimizing problems...")
    original_problems = decomposed_result.get("problems", [])
    print(f"  Input: {len(original_problems)} problems")

    # Group by validation_cmd first

    validation_groups = defaultdict(list)
    for prob in original_problems:
        validation_cmd = prob.get("validation_cmd", "unknown")
        validation_groups[validation_cmd].append(prob)

    print(f"  Grouped into {len(validation_groups)} validations")

    # Cluster and merge within each validation
    optimized_problems = []
    for validation_cmd, val_problems in validation_groups.items():
        print(f"    {validation_cmd}: {len(val_problems)} problems")

        if len(val_problems) > 1:
            # Apply clustering + LLM merge
            optimized = _cluster_and_merge_problems(
                val_problems,
                validation_cmd=validation_cmd,
                llm=llm,
                similarity_threshold=0.5,
            )
            print(f"      -> Optimized to {len(optimized)} problems")
            optimized_problems.extend(optimized)
        else:
            # Single problem, keep as-is
            optimized_problems.extend(val_problems)

    # Reorder: primary -> hidden, sorted by validation_order
    deduplicated = _reorder_by_repair_trajectory(optimized_problems)

    print(
        f"  Output: {len(deduplicated)} optimized problems (after clustering & merge)"
    )
    print(f"  Reduction: {len(original_problems) - len(deduplicated)} problems merged")

    # Stage 2: Detect dependencies + Generate repair sequence (UNIFIED LLM call)
    print("\n[Stage 2/5] Analyzing dependencies and repair sequence...")
    dep_result = _stage2_analyze_dependencies_and_sequence(deduplicated, llm)
    print(f"  Dependencies found: {len(dep_result.get('dependencies', []))}")
    print(f"  Repair sequence: {dep_result.get('repair_sequence', [])}")

    # Populate dependency fields and apply the analyzed repair sequence while
    # preserving the stable problem IDs referenced by dependency edges.
    print("  Populating dependency fields and applying repair sequence...")
    deduplicated = _reorder_by_repair_trajectory(
        deduplicated, dependency_result=dep_result
    )

    # Convert to old format for backward compatibility with build_memory
    dependencies = {
        "dependency_edges": [
            {"from": d["from"], "to": d["to"]} for d in dep_result.get("dependencies", [])
        ],
        "repair_order": dep_result.get("repair_sequence", [])
    }

    # Stage 3: Generate L1 (with NEW build_memory module)
    print("\n[Stage 3/5] Generating L1 (problem-level concrete failures)...")
    repo = decomposed_result.get("repo", "unknown")
    workflow_path = decomposed_result.get("benchmark_ci_context", {}).get(
        "workflow_path", "unknown"
    )
    issue_id_for_l1 = decomposed_result.get("original_issue_id", issue_id)
    changed_files = decomposed_result.get("changed_files", [])

    # Extract repo_owner and repo_name from repo (format: "owner/repo_name")
    repo_owner = repo.split("/")[0] if "/" in repo else "unknown"
    repo_name = repo.split("/")[1] if "/" in repo else repo

    # Use build_memory to generate L1 with dependencies
    l1_memory = generate_l1_from_decomposed_problems(
        issue_id=str(issue_id_for_l1),
        repo=repo,
        repo_owner=repo_owner,
        repo_name=repo_name,
        workflow_path=workflow_path,
        decomposed_problems=deduplicated,
        dependencies=dependencies,
        ground_truth_files=changed_files,
        llm=llm,
    )
    print(f"  L1 file-level problems: {len(l1_memory.get('problems', []))}")

    # Stage 4: Generate L2 (repair strategies with NEW build_memory module)
    print("\n[Stage 4/5] Generating L2 (repair strategies)...")
    l2_memory = build_l2_memory(l1_memory=l1_memory, llm=llm)
    print(f"  L2 repair strategies: {len(l2_memory.get('repair_strategies', []))}")

    # Stage 5: Generate L3 (universal patterns with NEW build_memory module)
    print("\n[Stage 5/5] Generating L3 (universal patterns)...")
    l3_memory = build_l3_memory(
        l1_memory=l1_memory,
        l2_memory=l2_memory,
        llm=llm,
    )
    num_patterns = len(l3_memory.get("universal_patterns", []))
    print(f"  L3 universal patterns: {num_patterns}")

    # Build final result (NEW format)
    result = {
        "issue_id": str(issue_id),
        "repo": decomposed_result.get("repo", "unknown"),
        "workflow_path": workflow_path,
        "l1_memory": l1_memory,  # NEW: Complete L1 structure
        "l2_memory": l2_memory,  # NEW: Repair strategies
        "l3_memory": l3_memory,  # NEW: Universal patterns
        "metadata": {
            "original_problems_count": len(decomposed_result.get("problems", [])),
            "deduplicated_count": len(deduplicated),
            "l1_problems": len(l1_memory.get("problems", [])),
            "l2_strategies": len(l2_memory.get("repair_strategies", [])),
            "l3_patterns": num_patterns,
        },
    }

    # Save immediately to back_trs/ (append mode)
    print("\n[Stage 6/5] Saving to back_trs/ (APPEND mode)...")
    _append_to_memory_files(result, output_dir=output_dir)

    print(f"\n{'=' * 80}")
    print("Pipeline Complete!")
    print(f"{'=' * 80}")

    return result


def _stage2_analyze_dependencies_and_sequence(problems: list[dict], llm) -> dict:
    """
    NEW UNIFIED APPROACH:
    Detect dependencies AND generate repair sequence in ONE LLM call.

    Returns lightweight format:
    {
      "dependencies": [{"from": 2, "to": 3, "reason": "..."}],
      "repair_sequence": [1, 2, 3]
    }
    """
    # Build graph info from problems
    graph_info = _build_graph_info_from_problems(problems)

    # Preserve complete problem evidence for cross-chunk dependency recovery.
    # These problems may have been generated from separate capacity chunks, so
    # truncating their causal fields here can hide the only shared API/config/
    # dependency signal that connects them.
    problems_summary = []
    for idx, prob in enumerate(problems, 1):
        problems_summary.append({
            "problem_id": prob.get("problem_id", idx),
            "validation_order": prob.get("validation_order", "unknown"),
            "validation_cmd": prob.get("validation_cmd", "unknown"),
            "problem_type": prob.get("problem_type", "unknown"),
            "failure_type": prob.get("failure_type", "unknown"),
            "issue_type": prob.get("issue_type", "unknown"),
            "problem": prob.get("problem", ""),
            "root_cause": prob.get("root_cause", ""),
            "how_fixed": prob.get("how_fixed", ""),
            "why_fix_works": prob.get("why_fix_works", ""),
            "affected_files": prob.get("affected_files", []),
            "is_cascading": prob.get("is_cascading", False),
            "dependency_type": prob.get("dependency_type", ""),
            "cascade_explanation": prob.get("cascade_explanation", ""),
        })

    # Use NEW unified prompt
    prompt = build_full_dependency_prompt(
        problems=problems_summary,
        graph_info=graph_info,
        strict_json_rules=STRICT_JSON_RULES,
    )

    try:
        time.sleep(3)  # Rate limiting
        response = _invoke_json(llm, prompt)

        if isinstance(response, dict) and "dependencies" in response and "repair_sequence" in response:
            return response
        else:
            print("  WARNING: LLM returned invalid format, using fallback")
            return _fallback_dependencies_and_sequence(problems)
    except Exception as e:
        print(f"  ERROR in unified dependency analysis: {e}")
        return _fallback_dependencies_and_sequence(problems)


def _build_graph_info_from_problems(problems: list[dict]) -> dict:
    """Build graph info structure from problems for LLM context."""
    # Group by validation
    validation_groups = {}
    for prob in problems:
        val_order = prob.get("validation_order", 0)
        val_cmd = prob.get("validation_cmd", "unknown")
        if val_order not in validation_groups:
            validation_groups[val_order] = {
                "order": val_order,
                "cmd": val_cmd,
                "problem_ids": []
            }
        validation_groups[val_order]["problem_ids"].append(prob.get("problem_id", 0))

    # File relationships
    file_to_problems = {}
    for prob in problems:
        for file in prob.get("affected_files", []):
            if file not in file_to_problems:
                file_to_problems[file] = []
            file_to_problems[file].append(prob.get("problem_id", 0))

    return {
        "validation_sequence": sorted(validation_groups.values(), key=lambda x: x["order"]),
        "file_relationships": file_to_problems
    }


def _populate_dependency_fields(problems: list[dict], dep_result: dict) -> list[dict]:
    """
    Programmatically populate enabled/enabled_by/enabled_reason fields.
    This is FAST - no LLM call needed.
    """
    # Initialize fields for all problems
    for p in problems:
        p["enabled"] = []
        p["enabled_reason"] = []
        p["enabled_by"] = []

    # Populate from dependencies
    dependencies = dep_result.get("dependencies", [])
    for dep in dependencies:
        from_id = dep["from"]
        to_id = dep["to"]
        reason = dep.get("reason", "")

        # Find problems by problem_id
        from_problem = None
        to_problem = None
        for p in problems:
            if p.get("problem_id") == from_id:
                from_problem = p
            if p.get("problem_id") == to_id:
                to_problem = p

        if from_problem and to_problem:
            # Add to enabled
            from_problem["enabled"].append(to_id)
            from_problem["enabled_reason"].append({
                "problem_id": to_id,
                "reason": reason
            })

            # Add to enabled_by (reverse)
            to_problem["enabled_by"].append(from_id)

    # Reorder problems by repair_sequence
    repair_sequence = dep_result.get("repair_sequence", [])
    if repair_sequence:
        ordered_problems = []
        for problem_id in repair_sequence:
            for p in problems:
                if p.get("problem_id") == problem_id:
                    ordered_problems.append(p)
                    break
        # Add any missing problems (safety)
        for p in problems:
            if p not in ordered_problems:
                ordered_problems.append(p)
        return ordered_problems

    return problems


def _fallback_dependencies_and_sequence(problems: list[dict]) -> dict:
    """Fallback when LLM fails - use validation_order."""
    # No dependencies, just order by validation_order
    sorted_problems = sorted(problems, key=lambda p: p.get("validation_order", 999))
    repair_sequence = [p.get("problem_id", i) for i, p in enumerate(sorted_problems, 1)]

    return {
        "dependencies": [],
        "repair_sequence": repair_sequence
    }


def _stage2_detect_dependencies_llm(problems: list[dict], llm) -> dict:
    """
    Stage 2: Detect dependencies between problems using LLM.

    Three-tier approach:
    - Tier 1: Try full LLM analysis (small data)
    - Tier 2: Grouped LLM per validation (large data)
    - Tier 3: Mechanical heuristics (LLM fails)
    """

    # Check data size
    problems_json_size = len(
        json.dumps(
            [
                {
                    "id": idx,
                    "validation_order": prob.get("validation_order"),
                    "validation_cmd": prob.get("validation_cmd"),
                    "problem_type": prob.get("problem_type"),
                    "problem": prob.get("problem", "")[:300],
                    "root_cause": prob.get("root_cause", "")[:300],
                    "affected_files": prob.get("affected_files", [])[:10],
                }
                for idx, prob in enumerate(problems, 1)
            ],
            indent=2,
        )
    )

    if problems_json_size < 50000 and len(problems) <= 20:
        # Small data - try full LLM
        print(
            f"  Dependencies: Full LLM approach ({problems_json_size} chars, {len(problems)} problems)"
        )
        return _stage2_dependencies_full_llm(problems, llm)
    else:
        # Large data - use grouped approach
        print(
            f"  Dependencies: Data too large ({problems_json_size} chars, {len(problems)} problems)"
        )
        print("  Using grouped LLM approach...")
        return _stage2_dependencies_grouped_llm(problems, llm)


def _stage2_dependencies_full_llm(problems: list[dict], llm) -> dict:
    """Full LLM dependency analysis for small/normal data."""

    # Prepare FULL context for LLM (not just summary)
    problems_summary = []
    for idx, prob in enumerate(problems, 1):
        problems_summary.append(
            {
                "id": idx,
                "validation_order": prob.get("validation_order", "unknown"),
                "validation_cmd": prob.get("validation_cmd", "unknown"),
                "problem_type": prob.get("problem_type", "unknown"),
                "failure_type": prob.get("failure_type", "unknown"),
                "problem": prob.get("problem", "")[:300],
                "root_cause": prob.get("root_cause", "")[:300],
                "how_fixed": prob.get("how_fixed", "")[:300],
                "why_fix_works": prob.get("why_fix_works", "")[:300],
                "is_cascading": prob.get("is_cascading", False),
                "dependency_type": prob.get("dependency_type", ""),
                "cascade_explanation": prob.get("cascade_explanation", "")[:300],
                "affected_files": prob.get("affected_files", [])[:10],
                "files_count": len(prob.get("affected_files", [])),
            }
        )

    # Use imported prompt template
    prompt = build_full_dependency_prompt(
        problems=problems_summary,
        strict_json_rules=STRICT_JSON_RULES,
    )

    try:
        time.sleep(3)  # Rate limiting
        response = _invoke_json(llm, prompt)

        if isinstance(response, dict):
            return response
        else:
            print("  WARNING: LLM returned non-dict, using grouped fallback")
            return _stage2_dependencies_grouped_llm(problems, llm)
    except Exception as e:
        print(f"  ERROR in full LLM dependency detection: {e}")
        return _stage2_dependencies_grouped_llm(problems, llm)


def _stage2_dependencies_grouped_llm(problems: list[dict], llm) -> dict:
    """
    Grouped LLM dependency detection for large data.

    Strategy:
    1. Group problems by validation_cmd
    2. LLM analyzes dependencies within each validation
    3. Combine all dependencies
    4. Generate repair order
    """
    print("  Using validation-grouped dependency detection...")

    # Step 1: Group by validation_cmd
    validation_groups = {}
    for idx, prob in enumerate(problems, 1):
        cmd = prob.get("validation_cmd", "unknown")
        if cmd not in validation_groups:
            validation_groups[cmd] = []
        validation_groups[cmd].append({"idx": idx, "problem": prob})

    print(f"  Grouped into {len(validation_groups)} validation groups")

    # Step 2: Analyze dependencies within each validation group
    all_edges = []

    for validation_cmd, group in validation_groups.items():
        print(f"  Analyzing: {validation_cmd} ({len(group)} problems)")

        if len(group) == 1:
            # Single problem - no dependencies
            print("    -> Single problem, no dependencies")
            continue

        try:
            # LLM analyzes this validation group
            group_edges = _analyze_validation_dependencies_llm(
                validation_cmd=validation_cmd, group=group, llm=llm
            )
            all_edges.extend(group_edges)
            print(f"    -> Found {len(group_edges)} dependencies")

        except Exception as e:
            print(f"    -> LLM failed: {e}, using mechanical heuristics")
            # Fallback to mechanical for this group
            group_edges = _detect_within_validation_dependencies_mechanical(group)
            all_edges.extend(group_edges)
            print(f"    -> Mechanical: {len(group_edges)} dependencies")

    # Step 3: Generate repair order
    repair_order = _compute_repair_order_from_edges(
        problems=problems, dependency_edges=all_edges
    )

    return {
        "dependency_edges": all_edges,
        "repair_order": repair_order,
        "reasoning": "Grouped LLM per validation + topological sort",
    }


def _analyze_validation_dependencies_llm(
    validation_cmd: str, group: list[dict], llm
) -> list[dict]:
    """
    Use LLM to analyze dependencies within a validation group.

    LLM analyzes:
    1. File-based: How file changes link to each other
    2. Problem-based: How one problem links to another
    3. Context-aware: Within this validation's context
    """

    # Prepare problem summaries for LLM
    problems_data = []
    for item in group:
        prob = item["problem"]
        problems_data.append(
            {
                "id": item["idx"],
                "validation_cmd": validation_cmd,
                "validation_order": prob.get("validation_order"),
                "problem_type": prob.get("problem_type", "unknown"),
                "problem": prob.get("problem", "")[:250],
                "root_cause": prob.get("root_cause", "")[:250],
                "how_fixed": prob.get("how_fixed", "")[:250],
                "is_cascading": prob.get("is_cascading", False),
                "dependency_type": prob.get("dependency_type", ""),
                "cascade_explanation": prob.get("cascade_explanation", "")[:250],
                "affected_files": prob.get("affected_files", []),
                "file_count": len(prob.get("affected_files", [])),
            }
        )

    # Use imported prompt template
    prompt = build_validation_group_dependency_prompt(
        validation_cmd=validation_cmd,
        problems_data=problems_data,
        strict_json_rules=STRICT_JSON_RULES,
    )

    time.sleep(2)  # Rate limiting
    response = _invoke_json(llm, prompt)

    # Check if response is valid dict
    if not isinstance(response, dict):
        raise ValueError(f"LLM returned invalid response type: {type(response)}")

    dependencies = response.get("dependencies", [])

    # Convert to edge format
    edges = []
    for dep in dependencies:
        edges.append(
            {
                "from": dep.get("from"),
                "to": dep.get("to"),
                "type": dep.get("type", "affects"),
                "reason": dep.get("reason", ""),
                "file_link": dep.get("file_link", ""),
                "strength": dep.get("strength", "medium"),
            }
        )

    return edges


def _detect_within_validation_dependencies_mechanical(group: list[dict]) -> list[dict]:
    """
    Mechanical dependency detection within a validation group (no LLM).

    Heuristics:
    1. Config files affect code files
    2. File overlap >50% = dependency
    3. Fewer files before more files
    4. Same file = strong dependency
    """
    edges = []

    for i, item_a in enumerate(group):
        prob_a = item_a["problem"]
        idx_a = item_a["idx"]
        files_a = set(prob_a.get("affected_files", []))

        if not files_a:
            continue

        for j, item_b in enumerate(group):
            if i >= j:  # Skip self and already compared
                continue

            prob_b = item_b["problem"]
            idx_b = item_b["idx"]
            files_b = set(prob_b.get("affected_files", []))

            if not files_b:
                continue

            # Rule 1: Config files affect others
            if _is_config_problem(prob_a) and not _is_config_problem(prob_b):
                edges.append(
                    {
                        "from": idx_a,
                        "to": idx_b,
                        "type": "enables",
                        "reason": "Config change may affect validation behavior",
                        "strength": "medium",
                    }
                )
                continue

            # Rule 2: File overlap
            overlap = files_a & files_b
            if overlap:
                overlap_ratio = len(overlap) / min(len(files_a), len(files_b))

                if overlap_ratio > 0.5:
                    # Significant overlap
                    edges.append(
                        {
                            "from": idx_a,
                            "to": idx_b,
                            "type": "affects",
                            "reason": f"Shares {len(overlap)} files ({int(overlap_ratio * 100)}% overlap)",
                            "strength": "strong" if overlap_ratio > 0.8 else "medium",
                        }
                    )
                elif overlap_ratio > 0.2:
                    # Some overlap
                    edges.append(
                        {
                            "from": idx_a,
                            "to": idx_b,
                            "type": "affects",
                            "reason": f"Shares {len(overlap)} files",
                            "strength": "weak",
                        }
                    )

            # Rule 3: Fewer files before more files (if no other relationship)
            elif len(files_a) < len(files_b) and len(files_a) <= 3:
                edges.append(
                    {
                        "from": idx_a,
                        "to": idx_b,
                        "type": "affects",
                        "reason": "Simpler fix (fewer files) before complex",
                        "strength": "weak",
                    }
                )

    return edges


def _is_config_problem(problem: dict) -> bool:
    """Check if problem involves config file changes."""
    files = problem.get("affected_files", [])

    config_patterns = [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        ".yml",
        ".yaml",
        "Makefile",
        ".ini",
        ".cfg",
        "tox.ini",
    ]

    for f in files:
        for pattern in config_patterns:
            if pattern in f:
                return True

    return False


def _compute_repair_order_from_edges(
    problems: list[dict], dependency_edges: list[dict]
) -> list[int]:
    """
    Compute repair order from dependency edges.

    Strategy:
    1. Separate by problem_type (primary vs hidden)
    2. Topological sort within each group
    3. Use validation_order as tie-breaker
    4. Final: primary + hidden
    """

    # Separate by problem_type
    primary_indices = []
    hidden_indices = []

    for idx, prob in enumerate(problems, 1):
        ptype = prob.get("problem_type", "primary")
        if ptype == "primary":
            primary_indices.append(idx)
        else:
            hidden_indices.append(idx)

    # Topological sort each group
    primary_order = _topological_sort_with_validation(
        indices=primary_indices, problems=problems, edges=dependency_edges
    )

    hidden_order = _topological_sort_with_validation(
        indices=hidden_indices, problems=problems, edges=dependency_edges
    )

    return primary_order + hidden_order


def _topological_sort_with_validation(
    indices: list[int], problems: list[dict], edges: list[dict]
) -> list[int]:
    """
    Topological sort respecting dependencies + validation_order tie-breaker.
    """
    if not indices:
        return []

    # Build adjacency list for these indices only
    graph = {idx: [] for idx in indices}
    in_degree = {idx: 0 for idx in indices}

    for edge in edges:
        from_idx = edge.get("from")
        to_idx = edge.get("to")

        if from_idx in indices and to_idx in indices:
            graph[from_idx].append(to_idx)
            in_degree[to_idx] += 1

    # Topological sort with validation_order as tie-breaker
    result = []
    queue = []

    # Start with nodes that have no dependencies
    for idx in indices:
        if in_degree[idx] == 0:
            queue.append(idx)

    # Sort queue by validation_order
    queue.sort(key=lambda idx: problems[idx - 1].get("validation_order", 999))

    while queue:
        # Take node with smallest validation_order
        current = queue.pop(0)
        result.append(current)

        # Process neighbors
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

        # Keep queue sorted by validation_order
        queue.sort(key=lambda idx: problems[idx - 1].get("validation_order", 999))

    # If there are remaining nodes (cycle detected), add them sorted by validation_order
    if len(result) < len(indices):
        remaining = [idx for idx in indices if idx not in result]
        remaining.sort(key=lambda idx: problems[idx - 1].get("validation_order", 999))
        result.extend(remaining)

    return result


def _append_to_memory_files(result: dict, output_dir: str = "data/back_trs"):
    """
    Append a SINGLE result to memory files IMMEDIATELY.

    This ensures we don't lose data if the script crashes.

    Args:
        result: Single result dict with l1_memory, l2_memory, l3_memory
        output_dir: Output directory (already includes back_trs path)
    """

    back_trs_dir = Path(output_dir)
    back_trs_dir.mkdir(parents=True, exist_ok=True)

    issue_id = result.get("issue_id", "unknown")

    # Extract L1 as complete issue record (NOT flattened)
    l1_memory = result.get("l1_memory", {})

    # Extract L2 record
    l2_memory = result.get("l2_memory", {})

    # Extract L3 patterns
    l3_patterns = []
    l3_memory = result.get("l3_memory", {})
    if l3_memory and "universal_patterns" in l3_memory:
        for pattern in l3_memory["universal_patterns"]:
            pattern_with_meta = dict(pattern)
            pattern_with_meta["source_issue_id"] = issue_id
            pattern_with_meta["source_repo"] = l3_memory.get("repo", "")
            l3_patterns.append(pattern_with_meta)

    # APPEND to each file (using atomic writes to prevent corruption)
    if l1_memory:
        failure_memory_path = back_trs_dir / "failure_memory.json"
        existing = []
        if failure_memory_path.exists():
            with open(failure_memory_path, "r") as f:
                existing = json.load(f)
        existing.append(l1_memory)  # Append complete issue record

        # Atomic write: write to temp file, then rename
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(dir=back_trs_dir, suffix='.tmp')
        try:
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(existing, f, indent=2)
            # Atomic rename (safe even if interrupted)
            os.replace(temp_path, failure_memory_path)
        except:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

        num_problems = len(l1_memory.get("problems", []))
        print(
            f"  OK Appended issue {issue_id} with {num_problems} problems to failure_memory.json"
        )

    if l2_memory:
        repo_memory_path = back_trs_dir / "repo_memory.json"
        existing = []
        if repo_memory_path.exists():
            with open(repo_memory_path, "r") as f:
                existing = json.load(f)
        existing.append(l2_memory)

        # Atomic write: write to temp file, then rename
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(dir=back_trs_dir, suffix='.tmp')
        try:
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(existing, f, indent=2)
            os.replace(temp_path, repo_memory_path)
        except:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

        print("  OK Appended 1 issue to repo_memory.json")

    if l3_patterns:
        cross_memory_path = back_trs_dir / "cross_memory.json"
        existing = []
        if cross_memory_path.exists():
            with open(cross_memory_path, "r") as f:
                existing = json.load(f)
        existing.extend(l3_patterns)

        # Atomic write: write to temp file, then rename
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(dir=back_trs_dir, suffix='.tmp')
        try:
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(existing, f, indent=2)
            os.replace(temp_path, cross_memory_path)
        except:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

        print(f"  OK Appended {len(l3_patterns)} patterns to cross_memory.json")


def _save_to_memory_files(results: list[dict], output_dir: str):
    """
    Save results to 3 memory files (APPEND mode - preserves existing data):
    1. failure_memory.json - L1 file-level problems (flat array)
    2. repo_memory.json - L2 repair sequences per issue
    3. cross_memory.json - L3 analysis per issue
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter successful results (skip errors) - handle BOTH old and new formats
    successful = [
        r
        for r in results
        if "l1_file_level" in r
        or "l2_repair_sequence" in r
        or "l1_memory" in r
        or "l2_memory" in r
    ]

    # Build issue_id set from new results to avoid duplicates
    new_issue_ids = {
        r.get("original_issue_id") or r.get("issue_id", "unknown") for r in successful
    }

    # 1. failure_memory.json - Handle both old format (l1_file_level) and new format (l1_memory)
    all_l1_problems = []
    for result in successful:
        # New format: l1_memory is a complete issue record
        if "l1_memory" in result:
            l1_memory = result.get("l1_memory", {})
            if l1_memory:
                all_l1_problems.append(l1_memory)  # Append complete record
        # Old format: l1_file_level is a flat array
        elif "l1_file_level" in result:
            l1_problems = result.get("l1_file_level", [])
            all_l1_problems.extend(l1_problems)

    # 2. repo_memory.json - Handle both old and new formats
    l2_sequences = []
    for result in successful:
        # New format: l2_memory is already in the correct format
        if "l2_memory" in result:
            l2_data = result.get("l2_memory", {})
            if l2_data:
                l2_sequences.append(l2_data)
        # Old format: l2_repair_sequence needs formatting
        elif "l2_repair_sequence" in result:
            l2_data = result.get("l2_repair_sequence", {})
            # Fix failure_type underscores in problems
            l2_problems = []
            for prob in l2_data.get("problems", []):
                prob["failure_type"] = prob.get("failure_type", "").replace("_", " ")
                l2_problems.append(prob)

            # Get workflow path and issue ID
            workflow_path = result.get("workflow_path", "unknown")
            issue_id = result.get("original_issue_id") or result.get(
                "issue_id", "unknown"
            )

            l2_sequences.append(
                {
                    "issue_id": issue_id,
                    "repo": result.get("repo"),
                    "workflow": workflow_path,
                    "problems": l2_problems,
                }
            )

    # 3. L3 analysis - Handle both old and new formats
    all_l3_patterns = []
    for result in successful:
        # New format: l3_memory.universal_patterns
        if "l3_memory" in result:
            l3_memory = result.get("l3_memory", {})
            l3_patterns = l3_memory.get("universal_patterns", [])
            issue_id = result.get("original_issue_id") or result.get(
                "issue_id", "unknown"
            )
            for pattern in l3_patterns:
                pattern_with_meta = dict(pattern)
                pattern_with_meta["source_issue_id"] = issue_id
                pattern_with_meta["source_repo"] = result.get("repo", "")
                all_l3_patterns.append(pattern_with_meta)
        # Old format: l3_analysis is a flat array
        elif "l3_analysis" in result:
            l3_patterns = result.get("l3_analysis", [])
            if isinstance(l3_patterns, list):
                all_l3_patterns.extend(l3_patterns)

    # Load existing data and merge (APPEND mode)
    existing_l1 = []
    existing_l2 = []
    existing_l3 = []

    # Load existing L1
    l1_path = output_dir / "failure_memory.json"
    if l1_path.exists():
        try:
            with open(l1_path) as f:
                existing_l1 = json.load(f)
                if not isinstance(existing_l1, list):
                    existing_l1 = []
                existing_l1 = [
                    item
                    for item in existing_l1
                    if str(item.get("issue_id", "")) not in new_issue_ids
                ]
        except Exception as e:
            print(f"Warning: Could not load existing L1: {e}")

    # Load existing L2 and filter out issues we're updating
    l2_path = output_dir / "repo_memory.json"
    if l2_path.exists():
        try:
            with open(l2_path) as f:
                existing_l2 = json.load(f)
                if not isinstance(existing_l2, list):
                    existing_l2 = []
                # Keep only issues NOT in new results (avoid duplicates)
                existing_l2 = [
                    item
                    for item in existing_l2
                    if item.get("issue_id") not in new_issue_ids
                ]
        except Exception as e:
            print(f"Warning: Could not load existing L2: {e}")

    # Load existing L3
    l3_path = output_dir / "cross_memory.json"
    if l3_path.exists():
        try:
            with open(l3_path) as f:
                existing_l3 = json.load(f)
                if not isinstance(existing_l3, list):
                    existing_l3 = []
                existing_l3 = [
                    item
                    for item in existing_l3
                    if str(item.get("source_issue_id", "")) not in new_issue_ids
                ]
        except Exception as e:
            print(f"Warning: Could not load existing L3: {e}")

    # Merge: existing + new
    merged_l1 = existing_l1 + all_l1_problems
    merged_l2 = existing_l2 + l2_sequences
    merged_l3 = existing_l3 + all_l3_patterns

    # Save merged data
    with open(l1_path, "w") as f:
        json.dump(merged_l1, f, indent=2)

    with open(l2_path, "w") as f:
        json.dump(merged_l2, f, indent=2)

    with open(l3_path, "w") as f:
        json.dump(merged_l3, f, indent=2)

    print(f"  -> Saved to {output_dir}/")
    print(
        f"     - failure_memory.json ({len(all_l1_problems)} new + {len(existing_l1)} existing = {len(merged_l1)} total)"
    )
    print(
        f"     - repo_memory.json ({len(l2_sequences)} new + {len(existing_l2)} existing = {len(merged_l2)} total)"
    )
    print(
        f"     - cross_memory.json ({len(all_l3_patterns)} new + {len(existing_l3)} existing = {len(merged_l3)} total)"
    )


def load_issues_from_huggingface(issue_ids: list[str] = None) -> list[dict[str, Any]]:
    """
    Load issues directly from HuggingFace dataset.

    Args:
        issue_ids: Optional list of issue IDs to filter. If None, loads all issues.

    Returns:
        List of issues matching the provided IDs (or all if no IDs specified)
    """
    print("Loading dataset from HuggingFace: ci-benchmark-user/ci-repair-bench")

    # Load with verification disabled to bypass feature compatibility issues
    try:
        # Delete cached dataset info to force reload

        cache_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "datasets"
            / "ci-benchmark-user___ci-repair-bench"
        )
        if cache_dir.exists():
            info_file = cache_dir / "default" / "0.0.0" / "dataset_info.json"
            if info_file.exists():
                print("Removing cached dataset_info.json to force reload...")
                info_file.unlink()

        # Load without verification
        ds = load_dataset(
            "ci-benchmark-user/ci-repair-bench",
            verification_mode="no_checks",
            download_mode="reuse_cache_if_exists",
        )
        data = ds["train"]
        print(f"Loaded {len(data)} issues from HuggingFace")

    except Exception as e:
        print(f"Dataset loading failed: {e}")
        raise RuntimeError(f"Could not load dataset from HuggingFace: {e}")

    if issue_ids:
        # Convert to set for faster lookup
        issue_ids_set = set(str(id) for id in issue_ids)

        # Filter to only requested IDs
        issues = []
        for item in data:
            if str(item.get("id")) in issue_ids_set:
                issues.append(dict(item))

        print(f"Filtered to {len(issues)} issues matching provided IDs")

        # Warn about missing IDs
        found_ids = set(str(i.get("id")) for i in issues)
        missing_ids = issue_ids_set - found_ids
        if missing_ids:
            print(
                f"WARNING: {len(missing_ids)} IDs not found in dataset: {sorted(missing_ids)}"
            )
    else:
        # Load all issues
        issues = [dict(item) for item in data]
        print(f"Loaded all {len(issues)} issues from dataset")

    return issues


def _load_jsonl_issues(dataset_path: Path) -> list[dict[str, Any]]:
    issues = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                issues.append(json.loads(line))
    return issues


def _load_issues_for_args(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if args.dataset:
        print(f"\n{'=' * 80}")
        print(f"Loading issues from JSONL file: {args.dataset}")
        print(f"{'=' * 80}")

        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"ERROR Dataset file not found: {dataset_path}")
            return None

        issues = _load_jsonl_issues(dataset_path)
        print(f"Loaded {len(issues)} issues from {dataset_path}")
        return issues

    if args.batch or args.use_huggingface or args.issue_id:
        print(f"\n{'=' * 80}")
        print("Loading issues from HuggingFace dataset")
        print(f"{'=' * 80}")

        if args.batch:
            memory_issue_ids = _load_memory_issue_ids()
            if memory_issue_ids:
                print(f"Using memory issue IDs: {memory_issue_ids}")
                return load_issues_from_huggingface(memory_issue_ids)
            print("No cached memory issue IDs found; loading all HuggingFace issues")
            return load_issues_from_huggingface(None)
        if args.issue_id:
            issues = load_issues_from_huggingface([args.issue_id])
            if not issues:
                print(f"ERROR Issue {args.issue_id} not found in HuggingFace dataset")
                return None
            return issues
        return load_issues_from_huggingface(None)

    print(f"\n{'=' * 80}")
    print(f"Loading issues from local file: {args.eval_issues}")
    print(f"{'=' * 80}")

    eval_path = Path(args.eval_issues)
    if not eval_path.exists():
        print(f"ERROR Eval issues not found: {eval_path}")
        print("TIP: Use --use-huggingface to load from HuggingFace instead")
        return None

    with open(eval_path) as f:
        issues = json.load(f)

    print(f"Loaded {len(issues)} issues from {eval_path}")
    return issues


def _load_decomposed_cache(decomposed_issues_path: Path) -> dict[str, dict[str, Any]]:
    decomposed_cache: dict[str, dict[str, Any]] = {}
    if not decomposed_issues_path.exists():
        return decomposed_cache

    try:
        with open(decomposed_issues_path) as f:
            existing_decomposed = json.load(f)
        if isinstance(existing_decomposed, list):
            for item in existing_decomposed:
                issue_id = _issue_id(item)
                if issue_id:
                    decomposed_cache[issue_id] = item
            print(
                f"Found {len(decomposed_cache)} decomposed issues (can reuse for L1/L2/L3)"
            )
    except Exception as e:
        print(f"Warning: Could not load decomposed issues: {e}")

    return decomposed_cache


def _load_existing_l2_results(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Load existing complete L1/L2/L3 issue IDs to avoid reprocessing."""
    failure_memory_path = output_dir / "failure_memory.json"
    repo_memory_path = output_dir / "repo_memory.json"
    cross_memory_path = output_dir / "cross_memory.json"

    if not repo_memory_path.exists():
        return [], set()

    try:
        with open(repo_memory_path) as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            return [], set()

        l2_ids = {
            str(result["issue_id"]) for result in existing if "issue_id" in result
        }

        l1_ids: set[str] = set()
        if failure_memory_path.exists():
            with open(failure_memory_path) as f:
                l1_existing = json.load(f)
            if isinstance(l1_existing, list):
                l1_ids = {
                    str(result["issue_id"])
                    for result in l1_existing
                    if "issue_id" in result
                }

        l3_ids: set[str] = set()
        if cross_memory_path.exists():
            with open(cross_memory_path) as f:
                l3_existing = json.load(f)
            if isinstance(l3_existing, list):
                l3_ids = {
                    str(result["source_issue_id"])
                    for result in l3_existing
                    if "source_issue_id" in result
                }

        processed_ids = l1_ids & l2_ids & l3_ids
        partial_ids = (l1_ids | l2_ids | l3_ids) - processed_ids
        print(f"Loaded {len(processed_ids)} complete L1/L2/L3 results (will skip)")
        if partial_ids:
            print(
                f"Found {len(partial_ids)} partial memory issues (will rebuild missing/replace existing)"
            )
        return existing, processed_ids
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def _save_decomposed_cache(
    decomposed_cache: dict[str, dict[str, Any]],
    decomposed_issues_path: Path,
) -> None:
    # Create parent directory if it doesn't exist
    decomposed_issues_path.parent.mkdir(parents=True, exist_ok=True)

    decomposed_list = list(decomposed_cache.values())
    with open(decomposed_issues_path, "w") as f:
        json.dump(decomposed_list, f, indent=2)
    print(f"  OK Saved to decomposed_issues.json ({len(decomposed_list)} issues)")


def _print_summary(
    *,
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total issues processed: {len(results)}")
    print(f"Successful: {len(results) - len(errors)}")
    print(f"Errors: {len(errors)}")

    successful = [result for result in results if "total_problems" in result]
    total_problems = sum(result.get("total_problems", 0) for result in successful)
    visible_problems = sum(
        sum(
            1
            for problem in result.get("problems", [])
            if problem.get("visibility") == "visible_in_log"
        )
        for result in successful
    )
    hidden_problems = total_problems - visible_problems

    print("\nAtomic problems identified:")
    print(f"  Total: {total_problems}")
    print(f"  Visible (in structured CI context): {visible_problems}")
    print(f"  Hidden (inferred): {hidden_problems}")

    if successful:
        avg_problems = total_problems / len(successful)
        print(f"  Average per issue: {avg_problems:.1f}")

    problem_types: dict[str, int] = {}
    for result in successful:
        for problem in result.get("problems", []):
            ptype = problem.get("problem_type", "unknown")
            problem_types[ptype] = problem_types.get(ptype, 0) + 1

    if problem_types:
        print("\nProblem type distribution:")
        for ptype, count in sorted(
            problem_types.items(), key=lambda item: item[1], reverse=True
        ):
            print(f"  {ptype}: {count}")

    print(f"\nOutput saved to: {output_dir}/")

    if errors:
        print(f"\nWARNING:  {len(errors)} issues had errors")
        print(
            f"Issue IDs with errors: {[error.get('original_issue_id') for error in errors[:5]]}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Reverse engineer CI failures into atomic problems (visible + hidden)"
    )
    parser.add_argument("--issue-id", help="Single issue ID to decompose")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Decompose all memory issues from HuggingFace",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Load from HuggingFace instead of local JSON",
    )
    parser.add_argument(
        "--dataset", help="Path to JSONL dataset file (filtered issues)"
    )
    parser.add_argument(
        "--eval-issues",
        default="data/trs/eval_issues.json",
        help="Path to eval issues (legacy mode)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/back_trs",
        help="Directory for all run outputs (default: data/back_trs)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model or alias. Use minimax2.5 or glm5.2.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of issues to process")
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Skip building memory files (L1/L2/L3) - only save decomposed_issues.json. Use this when you plan to do similarity-based split later.",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        help="Automatic 3-phase workflow: (1) Decompose all, (2) Cosine similarity split per repo (30%% memory, 70%% eval), (3) Build L1/L2/L3 only for memory set. Recommended for full pipeline.",
    )
    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=0.3,
        help="Memory set ratio for auto-split (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--use-memory-retrieval",
        action="store_true",
        help="Use memory plugin for problem retrieval (STAIR-based)",
    )
    parser.add_argument(
        "--memory-mode",
        choices=["baseline", "l1", "l1+l2", "l1+l2+l3"],
        default="l1+l2+l3",
        help="Memory retrieval mode: baseline (no memory), l1 (repo+workflow), l1+l2 (+ patterns), l1+l2+l3 (full STAIR). Only used with --use-memory-retrieval.",
    )
    parser.add_argument(
        "--memory-dir",
        default="data/fwr_trs",
        help="Memory directory for retrieval (default: data/fwr_trs)",
    )
    args = parser.parse_args()

    issues = _load_issues_for_args(args)
    if issues is None:
        return 1

    # Limit if requested
    if args.limit:
        issues = issues[: args.limit]
        print(f"Limited to first {args.limit} issues")

    # Initialize LLM
    print(f"\n{'=' * 80}")
    args.model = configure_model_environment(args.model) or args.model
    print(f"Initializing LLM: {args.model}")
    print(f"{'=' * 80}")
    llm = LitellmModel(model_name=args.model)

    # Initialize Memory Plugin (if requested)
    memory_retrieval = None
    if args.use_memory_retrieval:
        print(f"\n{'=' * 80}")
        print(f"Initializing Memory Plugin: {args.memory_mode}")
        print(f"Memory Directory: {args.memory_dir}")
        print(f"{'=' * 80}")

        from memory_plugin import STAIRRetrieval

        if args.memory_mode == "baseline":
            memory_retrieval = STAIRRetrieval(
                memory_dir=args.memory_dir, llm_client=llm, baseline_mode=True
            )
            print("Mode: BASELINE (no memory)")
        else:
            memory_retrieval = STAIRRetrieval(
                memory_dir=args.memory_dir,
                llm_client=llm,
                memory_levels=args.memory_mode,
            )
            print(f"Mode: {args.memory_mode.upper()} memory retrieval")
    else:
        print("\nMemory plugin: DISABLED (use --use-memory-retrieval to enable)")
    _ = memory_retrieval

    # Prepare output path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    print(f"Run output directory: {output_dir}")

    results = []
    errors = []
    processed_ids = set()

    # Save/load decomposed_issues.json from output_dir (backward traces)
    decomposed_issues_path = output_dir / "decomposed_issues.json"

    decomposed_cache = _load_decomposed_cache(decomposed_issues_path)
    results, processed_ids = _load_existing_l2_results(output_dir)

    # AUTO-SPLIT MODE: Three-phase workflow
    if args.auto_split:
        print(f"\n{'=' * 80}")
        print("AUTO-SPLIT MODE: 3-Phase Workflow")
        print(f"{'=' * 80}")
        print("Phase 1: Decompose all issues")
        print("Phase 2: Cosine similarity split per repo")
        print("Phase 3: Build L1/L2/L3 only for memory set")
        print(f"{'=' * 80}\n")

        # PHASE 1: Decompose all issues (no L1/L2/L3)
        print(f"\n{'=' * 80}")
        print(f"PHASE 1: Decomposing {len(issues)} issues")
        print(f"{'=' * 80}\n")

        for i, issue in enumerate(issues, 1):
            issue_id = _issue_id(issue)
            if not issue_id:
                print(f"\nProgress: {i}/{len(issues)} - missing issue id, skipping")
                continue

            print(f"\nProgress: {i}/{len(issues)}")

            # Check cache
            if issue_id in decomposed_cache:
                print("  OK Found in cache - skipping decomposition")
                continue
            else:
                # Decompose
                print("   Decomposing...")
                decomposed_result = decompose_issue(
                    issue, llm, output_dir=args.output_dir
                )

                if "error" not in decomposed_result:
                    decomposed_cache[issue_id] = decomposed_result
                    _save_decomposed_cache(decomposed_cache, decomposed_issues_path)
                    print("  OK Saved to cache")

        # PHASE 2: Cosine similarity split
        print(f"\n{'=' * 80}")
        print("PHASE 2: Cosine Similarity Split (per repo)")
        print(f"{'=' * 80}\n")

        print("  Running prepare_memory_train_test_split.py...")
        print("  This will compute embeddings and split by similarity...\n")

        # Run the split script

        split_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_memory_train_test_split.py"),
            "--dataset",
            str(args.dataset) if args.dataset else "data/trs/filtered_issues.jsonl",
            "--output-dir",
            str(output_dir),
            "--memory-ratio",
            str(args.memory_ratio),
        ]

        try:
            result = subprocess.run(split_cmd, check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            print(f"  FAIL ERROR: Similarity split failed: {e}")
            print("  You can run it manually:")
            print("    python scripts/prepare_memory_train_test_split.py \\")
            print(f"      --dataset {args.dataset} \\")
            print(f"      --output-dir {output_dir} \\")
            print(f"      --memory-ratio {args.memory_ratio}")
            return 1

        # Load the split results
        memory_issues_path = output_dir / "memory_issues.jsonl"
        eval_issues_path = output_dir / "eval_issues.jsonl"

        if not memory_issues_path.exists():
            print(f"  FAIL ERROR: {memory_issues_path} not found")
            print("  The similarity split may have failed.")
            return 1

        # Load memory issue IDs
        memory_ids = []
        with open(memory_issues_path) as f:
            for line in f:
                if line.strip():
                    issue = json.loads(line)
                    memory_ids.append(_issue_id(issue))

        eval_count = 0
        if eval_issues_path.exists():
            with open(eval_issues_path) as f:
                for line in f:
                    if line.strip():
                        eval_count += 1

        print("\n  OK Similarity split complete")
        print(f"  Total issues: {len(decomposed_cache)}")
        print(
            f"  Memory set: {len(memory_ids)} issues ({len(memory_ids) / len(decomposed_cache) * 100:.1f}%)"
        )
        print(
            f"  Eval set: {eval_count} issues ({eval_count / len(decomposed_cache) * 100:.1f}%)"
        )

        # PHASE 3: Build L1/L2/L3 only for memory set
        print(f"\n{'=' * 80}")
        print(f"PHASE 3: Building L1/L2/L3 for Memory Set ({len(memory_ids)} issues)")
        print(f"{'=' * 80}\n")

        for i, issue_id in enumerate(memory_ids, 1):
            print(f"\nMemory Issue {i}/{len(memory_ids)}: {issue_id}")

            decomposed_result = decomposed_cache.get(issue_id)
            if not decomposed_result or "error" in decomposed_result:
                print("  WARNING  Skipping - decomposition error or missing")
                continue

            # Deep copy to prevent cache pollution

            decomposed_copy = copy.deepcopy(decomposed_result)
            result = generate_l1_l2_l3_pipeline(
                decomposed_copy, llm, output_dir=args.output_dir
            )
            results.append(result)

            # Incremental save
            try:
                _save_to_memory_files(results, args.output_dir)
                print(f"  OK Saved ({len(results)} memory issues total)")
            except Exception as e:
                print(f"  WARNING  Could not save: {e}")

        # Final save
        _save_to_memory_files(results, args.output_dir)

        print(f"\n{'=' * 80}")
        print("AUTO-SPLIT COMPLETE")
        print(f"{'=' * 80}")
        print(f"OK Decomposed: {len(decomposed_cache)} issues")
        print(f"OK Memory set: {len(memory_ids)} issues with L1/L2/L3")
        print(f"OK Eval set: {eval_count} issues (decomposition only)")
        print("\nOutput files:")
        print(f"  - decomposed_issues.json ({len(decomposed_cache)} issues)")
        print(f"  - memory_issues.jsonl ({len(memory_ids)} issues)")
        print(f"  - eval_issues.jsonl ({eval_count} issues)")
        print(f"  - l1_file_level.json ({len(results)} issues)")
        print(f"  - l2_repair_sequences.json ({len(results)} issues)")
        print(f"  - l3_analysis.json ({len(results)} issues)")
        print("  - similarity_analysis.json (cosine similarity data)")
        print(f"{'=' * 80}\n")

        return 0

    # REGULAR MODE: Decompose issues with incremental saving
    for i, issue in enumerate(issues, 1):
        issue_id = _issue_id(issue)
        if not issue_id:
            print(f"\nProgress: {i}/{len(issues)} - missing issue id, skipping")
            errors.append(
                {
                    "error": "MISSING_ISSUE_ID",
                    "error_message": "Issue has no id, instance_id, issue_id, or original_issue_id",
                }
            )
            continue

        # Skip if already in L1/L2/L3 format
        if issue_id in processed_ids:
            print(
                f"\nProgress: {i}/{len(issues)} - Issue {issue_id} already has L1/L2/L3, skipping"
            )
            continue

        print(f"\nProgress: {i}/{len(issues)}")

        # Check if we have decomposed data (can skip decomposition)
        if issue_id in decomposed_cache:
            print(
                "  Found in decomposed_issues.json - Building L1/L2/L3 directly (no decomposition needed)"
            )
            decomposed_result = decomposed_cache[issue_id]
        else:
            # Need to decompose from scratch
            print("  Not found in cache - Running full decomposition...")
            decomposed_result = decompose_issue(
                issue, llm, output_dir=args.output_dir
            )

        # Save failed issues under this run's output directory, not in the cache.
        if "error" in decomposed_result:
            errors.append(decomposed_result)

            # Keep error artifacts scoped to the selected run directory.
            try:
                error_dir = output_dir / "errors"
                error_dir.mkdir(parents=True, exist_ok=True)
                error_file = error_dir / f"{issue_id}.json"
                with open(error_file, "w") as f:
                    json.dump(decomposed_result, f, indent=2)
                print(f"  ERROR saved to: {error_file}")
            except Exception as save_error:
                print(
                    f"  WARNING: Could not save error to {output_dir / 'errors'}: {save_error}"
                )
            # Do not add errors to decomposed_cache or successful results.
        else:
            # Success - save to decomposed_issues.json
            decomposed_cache[issue_id] = decomposed_result
            _save_decomposed_cache(decomposed_cache, decomposed_issues_path)

            # Run full L1/L2/L3 pipeline (unless --skip-memory)
            if not args.skip_memory:
                decomposed_copy = copy.deepcopy(decomposed_result)
                result = generate_l1_l2_l3_pipeline(
                    decomposed_copy, llm, output_dir=args.output_dir
                )
                results.append(result)
            else:
                # Just save decomposed result without L1/L2/L3
                results.append(decomposed_result)

        # Incremental save after each issue - save to 3 memory files (unless --skip-memory)
        if not args.skip_memory:
            try:
                _save_to_memory_files(results, args.output_dir)
                print(f"  OK Saved progress ({len(results)} issues total)")
            except Exception as e:
                print(f"  WARNING: Could not save progress: {e}")

    # Final save - save to 3 memory files (unless --skip-memory)
    if not args.skip_memory:
        _save_to_memory_files(results, args.output_dir)
    else:
        print(f"\n{'=' * 80}")
        print("Skipping memory file generation (--skip-memory flag)")
        print(f"{'=' * 80}")
        print("Only decomposed_issues.json was saved.")
        print(
            "Use this file with prepare_memory_train_test_split.py for similarity-based split."
        )

    _print_summary(results=results, errors=errors, output_dir=output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
