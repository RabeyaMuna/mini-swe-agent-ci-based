#!/usr/bin/env python3
"""
Run Codex CLI over CI-repair benchmark issues using prepared CI documents.

This is intentionally an external runner. It does not change Codex core
behavior; it prepares the same task document that a user would paste into
Codex, then invokes `codex exec` from a checked-out failing repository.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Load environment variables from .env file at project root
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # dotenv not installed, rely on environment variables

from memory_plugin import MemoryPlugin
from utilities.ci_log_analyzer import _run_log_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval_set.jsonl"
DEFAULT_ISSUE_IDS_FILE = PROJECT_ROOT / "data" / "eval_issue_ids.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "codex"
DEFAULT_MEMORY_ROOT = PROJECT_ROOT / "data" / "back_trs"
DEFAULT_LOG_CACHE = PROJECT_ROOT / "data" / "log_details.json"
DEFAULT_WORKFLOW_CACHE = PROJECT_ROOT / "data" / "workflow_validation_cache.json"

sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def issue_id(issue: dict[str, Any]) -> str:
    return str(issue.get("instance_id") or issue.get("id") or issue.get("sha_fail"))


def repo_slug(issue: dict[str, Any]) -> str:
    owner = str(issue.get("repo_owner") or "").strip()
    name = str(issue.get("repo_name") or "").strip()
    repo = str(issue.get("repo") or "").strip()
    if owner and name:
        return f"{owner}/{name}"
    return repo


def load_issues(dataset: Path) -> list[dict[str, Any]]:
    if dataset.suffix == ".jsonl":
        rows = []
        with dataset.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    data = load_json(dataset, [])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset shape: {dataset}")


def load_issue(dataset: Path, wanted_id: str) -> dict[str, Any]:
    for row in load_issues(dataset):
        if issue_id(row) == str(wanted_id):
            return row
    raise KeyError(f"Issue id not found in {dataset}: {wanted_id}")


def load_issue_ids(path: Path) -> list[str]:
    data = load_json(path, [])
    if not isinstance(data, list):
        raise ValueError(f"Issue ids file must be a JSON list: {path}")
    return [str(item) for item in data if str(item).strip()]


def load_huggingface_issues(dataset_name: str) -> list[dict[str, Any]]:
    try:
        from utilities.dataset_fetcher import HF_DATASET, fetch_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Fetching issue data from Hugging Face requires the shared "
            "utilities.dataset_fetcher module and its dependencies. "
            "Install them or provide data/eval_set.jsonl."
        ) from exc

    if dataset_name != HF_DATASET:
        raise ValueError(
            f"The shared dataset fetcher is configured for {HF_DATASET!r}; "
            f"got {dataset_name!r}."
        )
    return fetch_dataset(split="train", verbose=True)


def load_issue_index(dataset: Path, hf_dataset: str | None) -> dict[str, dict[str, Any]]:
    if dataset.exists():
        rows = load_issues(dataset)
    elif hf_dataset:
        rows = load_huggingface_issues(hf_dataset)
    else:
        raise FileNotFoundError(
            f"Dataset not found: {dataset}. Provide --dataset or --hf-dataset."
        )
    return {issue_id(row): row for row in rows}


def cache_index(path: Path) -> dict[str, dict[str, Any]]:
    records = load_json(path, [])
    if isinstance(records, dict):
        records = list(records.values())

    index: dict[str, dict[str, Any]] = {}
    for row in records or []:
        if not isinstance(row, dict):
            continue
        for key in (row.get("id"), row.get("instance_id"), row.get("sha_fail")):
            if key:
                index[str(key)] = row
    return index


def prepare_repo_checkout(
    issue: dict[str, Any],
    checkout: Path,
    refresh: bool,
    dry_run: bool,
) -> str:
    """
    Prepare a repository checkout at the failing commit.

    Simple approach: Clone directly from GitHub to the checkout directory.

    Returns:
        The SHA of the checkout commit (for later diff comparison)
    """
    sha = str(issue.get("sha_fail") or "").strip()
    slug = repo_slug(issue)
    if not slug:
        raise ValueError(f"Issue {issue_id(issue)} has no repo owner/name")
    if not sha:
        raise ValueError(f"Issue {issue_id(issue)} has no sha_fail")

    # Remove old checkout if refresh requested
    if checkout.exists() and refresh:
        shutil.rmtree(checkout)

    if dry_run and not (checkout / ".git").exists():
        checkout.mkdir(parents=True, exist_ok=True)
        return sha  # Return the SHA even in dry run

    # Clone directly from GitHub
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", f"https://github.com/{slug}.git", str(checkout)],
            check=True,
        )

    # Try to checkout the commit - if it fails, fetch it first
    result = subprocess.run(
        ["git", "checkout", "--force", sha],
        cwd=checkout,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Commit not in default branch - fetch it
        print(f"[codex] Commit {sha[:8]} not in default branch, fetching...")
        fetch_result = subprocess.run(
            ["git", "fetch", "origin", sha],
            cwd=checkout,
            capture_output=True,
            text=True,
        )

        if fetch_result.returncode != 0:
            # Try fetching all refs as fallback
            subprocess.run(
                ["git", "fetch", "--all", "--tags"],
                cwd=checkout,
                capture_output=True,
            )

        # Try checkout again
        subprocess.run(
            ["git", "checkout", "--force", sha],
            cwd=checkout,
            check=True,
        )

    subprocess.run(["git", "clean", "-fdx"], cwd=checkout, check=True)

    # Return the SHA we checked out (for later diff comparison)
    return sha


def make_context_llm(model: str | None) -> Any:
    if not model:
        return None

    try:
        from utilities.llm_model import LitellmModel
        from utilities.model_registry import configure_model_environment, resolve_model_alias
    except Exception as exc:
        raise RuntimeError(f"Could not import context LLM dependencies: {exc}") from exc

    model_name = str(resolve_model_alias(model))
    configure_model_environment(model_name)

    # Return proper LLM object with .invoke() method
    return LitellmModel(model_name=model_name)


def is_placeholder_analysis(data: dict[str, Any]) -> bool:
    source = str(data.get("source") or "")
    return source in {
        "fallback_log_scan",
        "missing_workflow_validation_cache",
    }


def is_usable_ci_failure(data: dict[str, Any]) -> bool:
    if not data or is_placeholder_analysis(data):
        return False
    if data.get("error") and not any(
        data.get(key) for key in ("error_context", "failure_signals", "relevant_files")
    ):
        return False
    return any(
        data.get(key)
        for key in ("error_context", "failure_signals", "relevant_files", "error_types")
    )


def is_usable_verification(data: dict[str, Any]) -> bool:
    if not data or is_placeholder_analysis(data):
        return False
    return bool(data.get("validation_sequence"))


def load_or_generate_ci_failure(
    issue: dict[str, Any],
    result_dir: Path,
    log_cache: Path,
    context_llm: Any,
    context_model: str | None,
    generate_missing: bool,
) -> dict[str, Any]:
    out_json = result_dir / "ci_failure.json"
    out_md = result_dir / "ci_failure.md"
    if out_json.exists():
        cached = load_json(out_json, {})
        if is_usable_ci_failure(cached):
            return cached

    index = cache_index(log_cache)
    analysis = index.get(issue_id(issue)) or index.get(str(issue.get("sha_fail"))) or {}
    if not is_usable_ci_failure(analysis):
        if not generate_missing:
            raise RuntimeError(
                f"No usable CI failure cache entry for issue {issue_id(issue)} "
                f"or sha {issue.get('sha_fail')}. Enable generation or precompute "
                f"{log_cache}."
            )
        if context_llm is None:
            raise RuntimeError(
                "No CI failure cache entry was found, and generation requires "
                "--context-model because utilities.ci_log_analyzer is LLM-backed."
            )

        try:

            analysis = _run_log_analysis(issue, context_llm, context_model or "")
            analysis["source"] = "ci_log_analyzer"

            # Save to shared cache for future runs
            print(f"[codex] Saving CI failure analysis to cache: {log_cache}")
            _append_to_cache(log_cache, analysis)

        except Exception as exc:
            raise RuntimeError(
                f"CI failure generation failed for issue {issue_id(issue)}: {exc}"
            ) from exc
        else:
            analysis["source"] = analysis.get("source") or "ci_failure_cache"

    write_json(out_json, analysis)
    write_text(out_md, format_ci_failure_markdown(analysis))
    return analysis


def _append_to_cache(cache_path: Path, new_entry: dict[str, Any]) -> None:
    """Append a new entry to the cache file, avoiding duplicates."""
    cache_data = load_json(cache_path, [])

    # Remove any existing entry with same ID
    entry_id = new_entry.get("id") or new_entry.get("instance_id")
    if entry_id:
        cache_data = [x for x in cache_data if x.get("id") != entry_id and x.get("instance_id") != entry_id]

    # Append new entry
    cache_data.append(new_entry)

    # Save back
    write_json(cache_path, cache_data)


def load_or_generate_verification(
    issue: dict[str, Any],
    result_dir: Path,
    workflow_cache: Path,
    checkout: Path,
    context_llm: Any,
    generate_missing: bool,
) -> dict[str, Any]:
    out_json = result_dir / "ci_verification.json"
    if out_json.exists():
        cached = load_json(out_json, {})
        if is_usable_verification(cached):
            return cached

    index = cache_index(workflow_cache)
    verification = index.get(issue_id(issue)) or index.get(str(issue.get("sha_fail"))) or {}
    if not is_usable_verification(verification):
        if not generate_missing:
            raise RuntimeError(
                f"No usable workflow verification cache entry for issue {issue_id(issue)} "
                f"or sha {issue.get('sha_fail')}. Enable generation or precompute "
                f"{workflow_cache}."
            )
        if context_llm is None:
            raise RuntimeError(
                "No workflow verification cache entry was found, and generation "
                "requires --context-model because "
                "utilities.ci_workflow_aware_retrieval is LLM-backed."
            )

        try:
            from utilities.ci_workflow_aware_retrieval import (
                analyze_workflow_from_benchmark,
            )

            verification = analyze_workflow_from_benchmark(
                workflow_content=str(issue.get("workflow") or ""),
                workflow_path=str(issue.get("workflow_path") or ""),
                repo_path=str(checkout),
                llm=context_llm,
                issue_id=issue_id(issue),
                sha_fail=str(issue.get("sha_fail") or ""),
            )
            verification["source"] = "ci_workflow_aware_retrieval"

            # Save to shared cache for future runs
            print(f"[codex] Saving workflow verification to cache: {workflow_cache}")
            _append_to_cache(workflow_cache, verification)

        except Exception as exc:
            raise RuntimeError(
                f"Workflow verification generation failed for issue "
                f"{issue_id(issue)}: {exc}"
            ) from exc
        else:
            verification["source"] = verification.get("source") or "workflow_cache"

    write_json(out_json, verification)
    return verification


def load_memory_context(
    issue: dict[str, Any],
    ci_failure: dict[str, Any],
    verification: dict[str, Any],
    result_dir: Path,
    memory_root: Path,
    ablation: str,
    top_k: int,
    context_llm: Any = None,
) -> tuple[str, dict]:
    """
    Load memory context by passing raw CI failure data to memory plugin.

    Memory plugin handles all query building and retrieval logic.
    This function just passes data and formats the result for Codex.

    Returns:
        (formatted_context, retrieval_dict): Formatted markdown and raw retrieval with problems list
    """
    if ablation.lower() == "baseline":
        return "", {"problems": []}

    try:
        # Initialize memory plugin with LLM (required for dynamic stages)
        plugin = MemoryPlugin(
            memory_root=memory_root,
            result_dir=str(result_dir),
            ablation=ablation,
            top_k=top_k,
            llm=context_llm,  # Pass the LLM!
            enabled=True
        )

        # Pass RAW ci_failure and verification to memory plugin
        # Memory plugin will build query internally from this data
        retrieval = plugin.retrieve(
            ci_failure=ci_failure,
            verification=verification,
            issue_metadata={
                "task_id": issue_id(issue),
                "sha_fail": issue.get("sha_fail"),
                "repo": repo_slug(issue),
            }
        )

        # Save full retrieval for analysis
        write_json(result_dir / "memory_retrieval.json", retrieval)

        # Format for Codex prompt (agent-specific formatting)
        formatted = plugin.format_for_prompt(retrieval)

        # Add detailed JSON for debugging (simplified structure)
        problems = retrieval.get("problems", [])
        details = json.dumps(
            {
                "total_problems": len(problems),
                "problem_types": [p.get("problem_type", "unknown") for p in problems],
                "failure_types": [p.get("failure_type", "unknown") for p in problems],
            },
            indent=2,
            ensure_ascii=False,
        )

        # Combine formatted text and details
        memory_md = f"{formatted}\n\n### Debug Details\n\n```json\n{details[:20000]}\n```"
        write_text(result_dir / "memory_context.md", memory_md)
        return memory_md, retrieval

    except Exception as exc:
        import traceback
        error_msg = f"Memory retrieval failed: {exc}\n\n{traceback.format_exc()}"
        print(f"[WARNING] Memory retrieval failed: {exc}")
        print(f"[WARNING] Falling back to baseline mode (no memory)")

        # Check if it's a missing dependency error
        if "No module named 'sentence_transformers'" in str(exc):
            error_msg += "\n\n⚠️  MISSING DEPENDENCY: sentence-transformers\n"
            error_msg += "Install with: pip install sentence-transformers>=2.7.0\n"
            error_msg += "Falling back to baseline mode (no memory).\n"

        write_text(result_dir / "memory_context.md", error_msg)

        # Return empty problems - will trigger baseline fallback in build_unified_problem_list
        return error_msg, {"problems": []}


def format_ci_failure_markdown(analysis: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CI Failure Analysis",
            "",
            "## Failure Context",
            bullet_lines(analysis.get("error_context")),
            "",
            "## Failure Signals",
            bullet_lines(analysis.get("failure_signals")),
            "",
            "## Relevant Files",
            bullet_lines(analysis.get("relevant_files")),
            "",
            "## Failed Jobs",
            bullet_lines(analysis.get("failed_job") or analysis.get("failed_jobs")),
            "",
            "## Error Types",
            bullet_lines(analysis.get("error_types")),
            "",
        ]
    )


def bullet_lines(value: Any) -> str:
    if not value:
        return "- Not available"
    if not isinstance(value, list):
        value = [value]
    lines = []
    for item in value:
        if isinstance(item, (dict, list)):
            item = json.dumps(item, ensure_ascii=False)
        lines.append(f"- {str(item).strip()}")
    return "\n".join(lines)


def validation_markdown(verification: dict[str, Any]) -> str:
    steps = verification.get("validation_sequence") or []
    if not steps:
        return "No workflow-aware verification sequence was available."

    lines = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        lines.extend(
            [
                f"### Step {step.get('order', '?')}: {step.get('validates', '')}",
                f"- Install: {step.get('installation_cmd') or 'n/a'}",
                f"- Validate: {step.get('validation_cmd') or 'n/a'}",
                f"- Source: {step.get('source') or 'n/a'}",
                f"- Evidence: {step.get('evidence') or 'n/a'}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def build_unified_problem_list(
    ci_failure: dict[str, Any],
    ci_problems: list[dict[str, Any]],
    memory_problems: list[dict[str, Any]],
    ablation: str,
) -> list[dict[str, Any]]:
    """
    Build unified problem list.

    Baseline mode:
      - Use CI decomposition only
      - No repair strategies (agent must figure it out)

    Memory modes (l1_l2_l3):
      - Memory plugin already processed CI failure
      - Memory plugin returns organized list:
        1. CI failure problems (WITH repair strategies if found in memory)
        2. Consecutive problems (appear after this type of fix)
        3. Common problems (general patterns)
      - Just use memory plugin's output directly!
    """

    # Baseline: Use CI decomposition only (no memory)
    if ablation.lower() == "baseline":
        print(f"[DEBUG] Baseline mode: Building {len(ci_problems)} CI problems (no memory)")
        unified = []
        for ci_prob in ci_problems:
            ci_signals = extract_error_signals_from_ci(ci_failure, ci_prob)
            unified.append({
                "problem": ci_prob.get("reason") or f"Fix {ci_prob.get('file', 'CI failure')}",
                "root_cause": "Analyze from CI failure context below",
                "files": [ci_prob.get("file")] if ci_prob.get("file") else [],
                "failure_signals": ci_signals,
                "repair_strategy": None,  # No memory guidance
                "failure_type": infer_failure_type(ci_prob),
                "verification_cmd": ci_prob.get("failed_cmd", ""),
                "source": "ci_failure",
            })

        # Add sequential numbers
        for idx, prob in enumerate(unified, 1):
            prob["number"] = idx

        print(f"[DEBUG] Total problems: {len(unified)}")
        return unified

    # Memory modes: Memory plugin already did ALL the work!
    # It processed CI failure, found repair strategies, consecutive problems, common patterns
    # Just use its organized output directly
    else:
        # Fallback: If memory plugin failed or returned empty, use CI problems
        if not memory_problems:
            print(f"[DEBUG] Memory mode ({ablation}): Memory plugin returned 0 problems")
            print(f"[DEBUG] Falling back to CI decomposition: {len(ci_problems)} problems")

            # Use same structure as baseline mode
            unified = []
            for ci_prob in ci_problems:
                ci_signals = extract_error_signals_from_ci(ci_failure, ci_prob)
                unified.append({
                    "problem": ci_prob.get("reason") or f"Fix {ci_prob.get('file', 'CI failure')}",
                    "root_cause": "Analyze from CI failure context below",
                    "files": [ci_prob.get("file")] if ci_prob.get("file") else [],
                    "failure_signals": ci_signals,
                    "repair_strategy": None,  # No memory guidance
                    "failure_type": infer_failure_type(ci_prob),
                    "verification_cmd": ci_prob.get("failed_cmd", ""),
                    "source": "ci_failure",  # Mark as CI-derived
                })

            # Add sequential numbers
            for idx, prob in enumerate(unified, 1):
                prob["number"] = idx

            print(f"[DEBUG] Fallback: Using {len(unified)} CI problems")
            return unified

        # Normal memory mode: use memory plugin's output
        print(f"[DEBUG] Memory mode ({ablation}): Using {len(memory_problems)} problems from memory plugin")

        # Add sequential numbers
        for idx, prob in enumerate(memory_problems, 1):
            prob["number"] = idx

        # Debug: show problem types
        for prob in memory_problems:
            ptype = prob.get("problem_type", "unknown")
            has_repair = "✓" if prob.get("repair_strategy") else "✗"
            print(f"[DEBUG]   Problem {prob['number']}: [{ptype}] [repair:{has_repair}] {prob.get('problem', '')[:80]}")

        return memory_problems



def calculate_match_score(ci_file: str, ci_signals: list[str], mem_prob: dict[str, Any]) -> float:
    """
    Calculate match score between CI problem and memory problem.

    Returns score 0.0-1.0 based on:
    - File name similarity (0.5 weight)
    - Error signal overlap (0.5 weight)
    """
    score = 0.0

    # File similarity
    mem_files = mem_prob.get("files", [])
    if ci_file and mem_files:
        for mem_file in mem_files:
            if ci_file in mem_file or mem_file in ci_file:
                score += 0.5
                break
            # Partial match on filename
            ci_basename = ci_file.split("/")[-1]
            mem_basename = mem_file.split("/")[-1]
            if ci_basename == mem_basename:
                score += 0.3
                break

    # Error signal similarity
    mem_signals = mem_prob.get("failure_signals", [])
    if ci_signals and mem_signals:
        # Count overlapping keywords
        ci_keywords = set()
        for sig in ci_signals:
            ci_keywords.update(sig.lower().split())

        mem_keywords = set()
        for sig in mem_signals:
            mem_keywords.update(sig.lower().split())

        if ci_keywords and mem_keywords:
            overlap = len(ci_keywords & mem_keywords)
            total = len(ci_keywords | mem_keywords)
            if total > 0:
                score += 0.5 * (overlap / total)

    return min(score, 1.0)  # Cap at 1.0


def extract_error_signals_from_ci(ci_failure: dict[str, Any], ci_problem: dict[str, Any]) -> list[str]:
    """Extract error signals relevant to a specific CI problem (filtered by file)."""
    signals = []
    problem_file = ci_problem.get("file", "")

    # Get error signals from CI failure analysis
    # Filter to only signals mentioning this specific file
    for signal in ci_failure.get("failure_signals", []):
        if isinstance(signal, str):
            # If we have a specific file, only include signals mentioning it
            if problem_file and problem_file in signal:
                signals.append(signal)
            # If no specific file, include all signals (fallback)
            elif not problem_file:
                signals.append(signal)
                if len(signals) >= 5:
                    break

    # If no file-specific signals found, include first few general signals as context
    if not signals and problem_file:
        for signal in ci_failure.get("failure_signals", [])[:2]:
            if isinstance(signal, str):
                signals.append(signal)

    return signals[:5]  # Limit to 5 signals max


def extract_all_error_signals_from_ci(ci_failure: dict[str, Any]) -> list[str]:
    """Extract all error signals from CI failure."""
    signals = []
    for signal in ci_failure.get("failure_signals", [])[:10]:
        if isinstance(signal, str):
            signals.append(signal)
    return signals


def infer_failure_type(ci_problem: dict[str, Any]) -> str:
    """Infer failure type from CI problem."""
    cmd = ci_problem.get("failed_cmd", "")
    reason = ci_problem.get("reason", "").lower()

    if "mypy" in cmd or "type" in reason:
        return "type_checking"
    elif "pylint" in cmd or "lint" in reason:
        return "linting"
    elif "test" in cmd or "pytest" in cmd:
        return "test_failure"
    elif "black" in cmd or "format" in reason:
        return "formatting"
    else:
        return "unknown"


def extract_problem_list(ci_failure: dict[str, Any]) -> list[dict[str, Any]]:
    files = ci_failure.get("relevant_files") or []
    problems = []
    for idx, item in enumerate(files, start=1):
        if isinstance(item, dict):
            file_name = item.get("file") or item.get("path") or ""
            reason = item.get("reason") or item.get("issue_type") or ""
            failed_cmd = item.get("failed_cmd") or ""
        else:
            file_name = str(item)
            reason = ""
            failed_cmd = ""
        if file_name or reason:
            problems.append(
                {
                    "number": idx,
                    "title": file_name or f"Problem {idx}",
                    "file": file_name,
                    "reason": reason,
                    "failed_cmd": failed_cmd,
                }
            )

    if not problems:
        context = "\n".join(str(x) for x in ci_failure.get("error_context", [])[:10])
        problems.append(
            {
                "number": 1,
                "title": "Primary CI failure",
                "file": "",
                "reason": context or "Repair the CI failure described in the logs.",
                "failed_cmd": "",
            }
        )

    return problems


def compose_baseline_prompt(
    issue: dict[str, Any],
    ci_failure: dict[str, Any],
    verification: dict[str, Any],
    problem: dict[str, Any],
    problem_count: int,
) -> str:
    """
    Baseline mode prompt: No memory plugin.

    Agent gets full CI failure analysis and must figure out the fix from scratch.
    """
    problem_desc = problem.get("problem", "See CI failure analysis")
    root_cause = problem.get("root_cause", "")
    files = problem.get("files", [])
    signals = problem.get("failure_signals", [])

    return f"""# CI Repair Task

You are fixing a CI failure in this repository.

## Problem To Fix

**[Problem {problem.get('number', '')} of {problem_count}]**

**Problem Description:**
{problem_desc}

**Root Cause:**
{root_cause if root_cause else "Analyze from CI failure context below"}

**Affected Files:**
{chr(10).join(f'  - {f}' for f in files) if files else "  (Identify from CI context)"}

**Error Signals:**
{chr(10).join(f'  - {s}' for s in signals[:5]) if signals else "  (See CI failure output below)"}

**Repository Context:**
- Repository: {repo_slug(issue)}
- Failing commit: {issue.get("sha_fail")}
- Workflow: {issue.get("workflow_path") or "unknown"}

{format_ci_failure_markdown(ci_failure)}

## CI Verification Details
{validation_markdown(verification)}

## Instructions

**Your Task:**
- Analyze the CI failure context above
- Identify the root cause from error signals and relevant files
- Determine the minimal fix needed
- Implement and validate the fix

**For automated tool failures (formatters, linters, type checkers):**
- Prefer running the tool with auto-fix flags: `black .`, `ruff --fix .`, `mypy --install-types`, etc.
- Let the tool fix all affected files at once
- Only manually edit if the tool cannot auto-fix

**General workflow:**
1. Inspect the repository and understand the problem from CI context
2. Make the minimal correct change to fix the issue
3. Do not modify unrelated files
4. Commit your changes with a descriptive message
5. OPTIONAL: Run the validation command ONLY on files you changed (not the whole repo)
6. The fix should be committed in git (not as uncommitted changes)

**Scope:**
- Fix this problem only
- Preserve existing behavior unless proven wrong by CI
- Do not remove tests or weaken checks
- Do not update dependencies unless required by the fix
- Do NOT run validation on the entire repository (may have unrelated failures)
- If you verify, run validation ONLY on the specific files you changed

**Important:**
- COMMIT your changes (don't leave as uncommitted diff)
- Use descriptive commit messages
- Don't worry if repo-wide validation fails due to other issues
- Your fix should address the specific problem described above

**Final report:**
- Root cause identified
- Files changed
- Whether validation passed on YOUR changes (not whole repo)
- Any remaining risks
"""


def compose_memory_prompt(
    issue: dict[str, Any],
    verification: dict[str, Any],
    problem: dict[str, Any],
    problem_count: int,
) -> str:
    """
    Memory mode prompt: Memory plugin active.

    Agent gets a self-contained problem with repair strategy from similar past fixes.
    NO full CI failure analysis (would confuse with errors from other problems).
    """
    problem_desc = problem.get("problem", "")
    root_cause = problem.get("root_cause", "")
    files = problem.get("files", [])
    signals = problem.get("failure_signals", [])
    repair_strategy = problem.get("repair_strategy") or {}

    # Build repair plan section if available
    repair_plan = ""
    if repair_strategy and isinstance(repair_strategy, dict):
        summary = repair_strategy.get("summary", "")
        actions = repair_strategy.get("actions", [])
        validation_cmd = repair_strategy.get("validation_cmd", "")
        pitfalls = repair_strategy.get("pitfalls", [])

        if summary or actions:
            repair_plan = f"""

**Repair Plan (from similar past fixes):**

Approach: {summary if summary else "Follow the actions below"}

Action Steps:
{chr(10).join(f'{i}. {action}' for i, action in enumerate(actions, 1)) if actions else "1. Analyze problem and determine fix"}

Validation Command:
  {validation_cmd if validation_cmd else "./ci-validation-command (see CI Verification section)"}

Pitfalls to Avoid:
{chr(10).join(f'  - {p}' for p in pitfalls) if pitfalls else "  (Use best practices)"}

**IMPORTANT:** The repair plan above is based on similar past fixes that worked. Follow the action steps as your primary strategy."""

    return f"""# CI Repair Task

You are fixing a CI failure in this repository using guidance from similar past fixes.

## Problem To Fix

**[Problem {problem.get('number', '')} of {problem_count}]**

**Problem Description:**
{problem_desc}

**Root Cause:**
{root_cause}

**Affected Files:**
{chr(10).join(f'  - {f}' for f in files) if files else "  (See repair plan)"}

**Error Signals:**
{chr(10).join(f'  - {s}' for s in signals[:5]) if signals else "  (See repair plan)"}
{repair_plan}

**Repository Context:**
- Repository: {repo_slug(issue)}
- Failing commit: {issue.get("sha_fail")}
- Workflow: {issue.get("workflow_path") or "unknown"}

## CI Verification Details
{validation_markdown(verification)}

## Instructions

**If repair plan is provided above:**
- Follow the action steps as your primary repair strategy
- The steps are based on similar past fixes that worked
- Pay attention to the pitfalls listed
- Run the validation command specified

**If repair plan is missing or incomplete:**
- Use the problem description, root cause, and error signals above
- Identify the fix from the affected files
- Make the minimal correct change

**For automated tool failures (formatters, linters, type checkers):**
- Prefer running the tool with auto-fix flags: `black .`, `ruff --fix .`, `mypy --install-types`, etc.
- Let the tool fix all affected files at once
- Only manually edit if the tool cannot auto-fix

**General workflow:**
1. Inspect the repository and understand the problem
2. Make the minimal correct change to fix the issue
3. Do not modify unrelated files
4. Commit your changes with a descriptive message
5. OPTIONAL: Run validation ONLY on files you changed (not the whole repo)
6. The fix should be committed in git (not as uncommitted changes)

**Scope:**
- Fix this problem only (do not fix unrelated issues)
- Preserve existing behavior unless proven wrong by CI
- Do not remove tests or weaken checks
- Do not update dependencies unless required by the fix
- Do NOT run validation on the entire repository (may have unrelated failures)
- If you verify, run validation ONLY on the specific files you changed

**Important:**
- COMMIT your changes (don't leave as uncommitted diff)
- Use descriptive commit messages
- Don't worry if repo-wide validation fails due to other issues
- Your fix should address the specific problem described above

**Final report:**
- Root cause identified
- Files changed
- Whether validation passed on YOUR changes (not whole repo)
- Any remaining risks
"""


def compose_issue_document(
    issue: dict[str, Any],
    ci_failure: dict[str, Any],
    verification: dict[str, Any],
    problem: dict[str, Any],
    problem_count: int,
    ablation: str,
) -> str:
    """
    Route to appropriate prompt template based on mode.

    Baseline mode: Full CI failure analysis, agent analyzes from scratch
    Memory mode: Self-contained problem with repair strategy
    """
    is_baseline = ablation.lower() == "baseline"

    if is_baseline:
        return compose_baseline_prompt(
            issue, ci_failure, verification, problem, problem_count
        )
    else:
        return compose_memory_prompt(
            issue, verification, problem, problem_count
        )


def write_issue_document(result_dir: Path, problem: dict[str, Any], document: str) -> Path:
    path = result_dir / f"issue_document_problem_{problem['number']}.md"
    write_text(path, document)
    return path


def run_codex(
    checkout: Path,
    prompt_path: Path,
    transcript_path: Path,
    command: str,
    timeout: int,
    dry_run: bool,
) -> dict[str, Any]:
    """
    Run codex agent and stream output in real-time to both terminal and file.

    This allows you to see what the agent is doing while it runs.
    """
    cmd = shlex.split(command)
    prompt = prompt_path.read_text(encoding="utf-8")
    started = time.time()

    if dry_run:
        write_text(transcript_path, f"DRY RUN: would execute {cmd} in {checkout}\n")
        return {"returncode": 0, "elapsed_seconds": 0.0, "dry_run": True}

    print(f"\n{'='*80}")
    print(f"[AGENT] Starting: {' '.join(cmd)}")
    print(f"[AGENT] Working directory: {checkout}")
    print(f"[AGENT] Output will be saved to: {transcript_path}")
    print(f"{'='*80}\n")

    # Stream output to both terminal and file in real-time
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass environment variables to subprocess
    # CRITICAL: Filter out OpenRouter key to prevent API key conflicts
    env = os.environ.copy()

    # Remove OpenRouter variables if using OpenAI/Anthropic directly
    if 'OPENAI_API_KEY' in env and env.get('OPENAI_API_KEY', '').startswith('sk-proj-'):
        # Using OpenAI directly - remove OpenRouter
        env.pop('OPENROUTER_API_KEY', None)
        env.pop('OPENAI_BASE_URL', None)
    elif 'ANTHROPIC_API_KEY' in env:
        # Using Anthropic directly - remove OpenAI/OpenRouter
        env.pop('OPENROUTER_API_KEY', None)
        env.pop('OPENAI_API_KEY', None)
        env.pop('OPENAI_BASE_URL', None)

    with open(transcript_path, 'w', encoding='utf-8') as f:
        proc = subprocess.Popen(
            cmd,
            cwd=checkout,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,  # Pass environment to subprocess
        )

        # Send prompt to stdin
        if proc.stdin:
            proc.stdin.write(prompt)
            proc.stdin.close()

        # Stream output line by line
        if proc.stdout:
            for line in proc.stdout:
                # Print to terminal (real-time visibility)
                print(line, end='', flush=True)
                # Save to file
                f.write(line)
                f.flush()

        proc.wait(timeout=timeout)

    elapsed = time.time() - started

    print(f"\n{'='*80}")
    print(f"[AGENT] Completed in {elapsed:.1f}s with exit code {proc.returncode}")
    print(f"{'='*80}\n")

    return {"returncode": proc.returncode, "elapsed_seconds": elapsed, "dry_run": False}


def git_diff(checkout: Path, original_commit: str = None) -> str:
    """
    Get diff of all changes made by agent (committed + uncommitted).

    Codex agent commits its changes, so we need to diff against the original commit
    before the agent ran, not just uncommitted working directory changes.
    """
    # Try to get uncommitted changes first
    proc = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    uncommitted = proc.stdout.strip()

    # If original_commit provided, diff from it to HEAD
    if original_commit:
        proc = subprocess.run(
            ["git", "diff", "--binary", original_commit, "HEAD"],
            cwd=checkout,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        committed_diff = proc.stdout.strip()
        # Return committed changes if any, otherwise uncommitted
        return committed_diff if committed_diff else uncommitted

    # Fallback: try to get diff from last 5 commits (in case agent made commits)
    # This covers the case where agent committed changes
    proc = subprocess.run(
        ["git", "diff", "--binary", "HEAD~5..HEAD"],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    recent_commits = proc.stdout.strip() if proc.returncode == 0 else ""

    # Return whichever has content
    return recent_commits if recent_commits else uncommitted


def changed_files(checkout: Path, original_commit: str = None) -> list[str]:
    """
    Get list of files changed by agent (committed + uncommitted).

    Args:
        checkout: Path to git repository
        original_commit: Original commit SHA to diff from (if agent made commits)

    Returns:
        List of changed file paths
    """
    # Try uncommitted changes first
    proc = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    uncommitted_files = [x.strip() for x in proc.stdout.splitlines() if x.strip()]

    # If original_commit provided, get files from commits
    if original_commit:
        proc = subprocess.run(
            ["git", "diff", "--name-only", original_commit, "HEAD"],
            cwd=checkout,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        committed_files = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
        # Return committed files if any, otherwise uncommitted
        return committed_files if committed_files else uncommitted_files

    # Fallback: try recent commits
    proc = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~5..HEAD"],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    recent_files = [x.strip() for x in proc.stdout.splitlines() if x.strip()] if proc.returncode == 0 else []

    # Return whichever has content
    return recent_files if recent_files else uncommitted_files


def candidate_validation_commands(verification: dict[str, Any]) -> list[str]:
    commands = []
    for step in verification.get("validation_sequence") or []:
        if not isinstance(step, dict):
            continue
        cmd = str(step.get("validation_cmd") or "").strip()
        if cmd and cmd.lower() != "n/a":
            commands.append(cmd)
    return commands


# External verification removed - agent handles verification internally for each problem


def save_patch_and_result(
    result_dir: Path,
    checkout: Path,
    issue: dict[str, Any],
    ablation: str,
    model_name: str,
    problem_results: list[dict[str, Any]],
    verification: dict[str, Any],
    verification_result: dict[str, Any] | None,
    original_sha: str = None,
) -> None:
    print(f"[save_patch_and_result] Getting git diff from {checkout}")
    print(f"[save_patch_and_result] Original SHA: {original_sha}")
    diff = git_diff(checkout, original_sha)
    files = changed_files(checkout, original_sha)

    print(f"[save_patch_and_result] Diff size: {len(diff)} bytes")
    print(f"[save_patch_and_result] Changed files: {len(files)} files")
    print(f"[save_patch_and_result] Writing to {result_dir / 'patch.diff'}")

    write_text(result_dir / "patch.diff", diff)
    write_json(
        result_dir / "result.json",
        {
            "id": issue_id(issue),
            "repo": repo_slug(issue),
            "sha_fail": issue.get("sha_fail"),
            "ablation": ablation,
            "model_name_or_path": model_name,
            "patch_generated": bool(diff.strip()),
            "patch_bytes": len(diff.encode("utf-8")),
            "changed_files": files,
            "candidate_validation_commands": candidate_validation_commands(verification),
            "verification": verification_result,
            "verification_passed": (
                None
                if verification_result is None
                else verification_result.get("returncode") == 0
            ),
            "problem_results": problem_results,
        },
    )


def run_issue(args: argparse.Namespace, ablation: str, issue: dict[str, Any]) -> None:
    safe_ablation = ablation.replace("+", "_").lower()

    # Include model name in output directory (e.g., baseline_minimax2.5)
    if args.context_model:
        model_suffix = f"_{args.context_model.replace('/', '_').replace('.', '_')}"
        ablation_dir = f"{safe_ablation}{model_suffix}"
    else:
        ablation_dir = safe_ablation

    result_dir = args.output_root / ablation_dir / issue_id(issue)
    checkout = result_dir / "checkout"

    original_sha = prepare_repo_checkout(
        issue,
        checkout,
        refresh=args.refresh_checkout,
        dry_run=args.dry_run,
    )
    context_llm = make_context_llm(args.context_model) if args.generate_missing_analysis else None
    ci_failure = load_or_generate_ci_failure(
        issue,
        result_dir,
        args.log_cache,
        context_llm,
        args.context_model,
        args.generate_missing_analysis,
    )
    verification = load_or_generate_verification(
        issue,
        result_dir,
        args.workflow_cache,
        checkout,
        context_llm,
        args.generate_missing_analysis,
    )
    # For baseline: extract problems from CI failure (simple extraction)
    # For memory modes: let memory plugin decompose (uses LLM, smarter)
    if ablation.lower() == "baseline":
        # Baseline: Simple CI extraction, no memory
        ci_problems = extract_problem_list(ci_failure)
        memory_problems = []
    else:
        # Memory modes: Let memory plugin decompose CI and search
        _, memory_retrieval = load_memory_context(
            issue,
            ci_failure,
            verification,
            result_dir,
            args.memory_root,
            ablation,
            args.memory_top_k,
            context_llm,  # Pass LLM for dynamic stages!
        )

        memory_problems = memory_retrieval.get("problems", [])

        # Only extract CI problems as fallback if memory returned empty
        # (Memory plugin's Stage 0 decomposition is better, but if it crashed, use this)
        ci_problems = extract_problem_list(ci_failure) if not memory_problems else []

    print(f"\n[DEBUG] Building unified problem list:")
    print(f"[DEBUG]   CI failure problems: {len(ci_problems)}")
    print(f"[DEBUG]   Memory problems: {len(memory_problems)}")

    # Build unified problem list by matching CI with Memory
    problems = build_unified_problem_list(
        ci_failure=ci_failure,
        ci_problems=ci_problems,
        memory_problems=memory_problems,
        ablation=ablation
    )

    # Count by source
    ci_count = sum(1 for p in problems if p.get('source') == 'ci_failure')
    mem_count = sum(1 for p in problems if p.get('source') == 'memory')

    print(f"\n[DEBUG] Unified problem list:")
    print(f"[DEBUG]   Total problems: {len(problems)}")
    print(f"[DEBUG]   From CI: {ci_count} (no repair strategies)")
    print(f"[DEBUG]   From Memory: {mem_count} (HAS repair strategies)")

    # If no problems, save empty result and return early
    if not problems:
        print(f"\n[ERROR] No problems to fix! This should never happen.")
        print(f"[ERROR]   CI problems: {len(ci_problems)}")
        print(f"[ERROR]   Memory problems: {len(memory_problems)}")
        print(f"[ERROR] Saving empty result and returning...")

        save_patch_and_result(
            result_dir,
            checkout,
            issue,
            ablation,
            prediction_model_name(args),
            [],  # No problem results
            verification,
            None,
            original_sha,
        )
        return

    print(f"\n[DEBUG] Problems to fix (one by one):")
    for idx, p in enumerate(problems, 1):
        source = p.get('source', 'unknown')
        has_repair = '✓' if p.get('repair_strategy') else '✗'
        print(f"[DEBUG]   {idx}. [{source}] [repair:{has_repair}] {p.get('problem', 'N/A')[:70]}...")
    print()
    problem_results = []
    for problem in problems:
        document = compose_issue_document(
            issue,
            ci_failure,
            verification,
            problem,
            len(problems),
            ablation,
        )
        prompt_path = write_issue_document(result_dir, problem, document)
        transcript_path = result_dir / f"codex_transcript_problem_{problem['number']}.txt"
        run_result = run_codex(
            checkout,
            prompt_path,
            transcript_path,
            args.codex_command,
            args.timeout,
            args.dry_run,
        )
        problem_results.append({"problem": problem, **run_result})
        if run_result["returncode"] != 0:
            break

    # Agent handles verification internally for each problem
    # No external verification needed
    print(f"\n[DEBUG] Saving patch and result for issue {issue_id(issue)}...")
    print(f"[DEBUG]   Result dir: {result_dir}")
    print(f"[DEBUG]   Checkout: {checkout}")
    print(f"[DEBUG]   Problem results: {len(problem_results)} problems processed")

    save_patch_and_result(
        result_dir,
        checkout,
        issue,
        ablation,
        prediction_model_name(args),
        problem_results,
        verification,
        None,  # No external verification
        original_sha,
    )

    print(f"[DEBUG] ✓ Saved patch.diff and result.json")


def prediction_model_name(args: argparse.Namespace) -> str:
    return str(args.model_name or args.context_model or "codex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--issue-ids", help="Comma-separated issue ids")
    parser.add_argument(
        "--issue-ids-file",
        type=Path,
        default=DEFAULT_ISSUE_IDS_FILE,
        help="JSON list of issue ids to run when --issue-ids is omitted",
    )
    parser.add_argument(
        "--hf-dataset",
        default="ci-benchmark-user/ci-repair-bench",
        help="Hugging Face dataset used only if --dataset does not exist",
    )
    parser.add_argument(
        "--ablations",
        default="baseline,L1,L1+L2,L1+L2+L3",
        help="Comma-separated ablations to run",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--log-cache", type=Path, default=DEFAULT_LOG_CACHE)
    parser.add_argument("--workflow-cache", type=Path, default=DEFAULT_WORKFLOW_CACHE)
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument(
        "--codex-command",
        default=os.environ.get("CODEX_REPAIR_COMMAND", "codex exec --full-auto"),
        help="Command used to run Codex non-interactively",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    # Note: Verification is handled by agent internally, no external verification needed
    parser.add_argument(
        "--generate-missing-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate missing CI/log/workflow analysis with --context-model when "
            "cache entries are missing. Enabled by default; use "
            "--no-generate-missing-analysis to require cached data only."
        ),
    )
    parser.add_argument(
        "--context-model",
        default=os.environ.get("CODEX_CONTEXT_MODEL"),
        help="Model used by missing CI/log workflow analyzers",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("CODEX_MODEL_NAME"),
        help=(
            "Model name recorded in predictions.json. Defaults to --context-model "
            "when set, otherwise 'codex'."
        ),
    )
    parser.add_argument("--refresh-checkout", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prediction_from_result(result_file: Path) -> dict[str, Any]:
    issue_dir = result_file.parent

    with open(result_file, encoding="utf-8") as f:
        result = json.load(f)

    patch_file = issue_dir / "patch.diff"
    if patch_file.exists():
        patch_content = patch_file.read_text(encoding="utf-8")
    else:
        patch_content = ""

    return {
        "id": result["id"],
        "sha_fail": result["sha_fail"],
        "repo": result["repo"],
        "diff": patch_content,
        "ablation": result.get("ablation", "unknown"),
        "patch_generated": result["patch_generated"],
        "patch_bytes": result["patch_bytes"],
        "changed_files": result["changed_files"],
        "verification_passed": result.get("verification_passed"),
    }


def consolidate_predictions(
    output_root: Path, ablation: str, context_model: str | None = None
) -> list[dict[str, Any]]:
    """
    Consolidate one ablation's patches into a predictions.json file.

    This creates a predictions file compatible with evaluation tools,
    similar to Mini-SWE-Agent's format.
    """
    safe_ablation = ablation.replace("+", "_").lower()

    # Include model name in directory (e.g., baseline_minimax2.5)
    if context_model:
        model_suffix = f"_{context_model.replace('/', '_').replace('.', '_')}"
        ablation_dir = f"{safe_ablation}{model_suffix}"
    else:
        ablation_dir = safe_ablation

    results_dir = output_root / ablation_dir
    predictions_file = results_dir / "predictions.json"

    predictions = [
        prediction_from_result(result_file)
        for result_file in sorted(results_dir.glob("*/result.json"))
    ]

    # Write consolidated file
    if predictions:
        write_json(predictions_file, predictions)
        print(f"\n[codex-ci-repair] Consolidated {len(predictions)} predictions -> {predictions_file}")

    return predictions


def consolidate_all_predictions(
    output_root: Path, ablations: list[str], context_model: str | None = None
) -> None:
    """Write one predictions.json containing every selected ablation level."""
    predictions: list[dict[str, Any]] = []
    for ablation in ablations:
        safe_ablation = ablation.replace("+", "_").lower()

        # Include model name in directory (e.g., baseline_minimax2.5)
        if context_model:
            model_suffix = f"_{context_model.replace('/', '_').replace('.', '_')}"
            ablation_dir = f"{safe_ablation}{model_suffix}"
        else:
            ablation_dir = safe_ablation

        results_dir = output_root / ablation_dir
        predictions.extend(
            prediction_from_result(result_file)
            for result_file in sorted(results_dir.glob("*/result.json"))
        )

    if predictions:
        predictions_file = output_root / "predictions.json"
        write_json(predictions_file, predictions)
        print(
            f"\n[codex-ci-repair] Consolidated {len(predictions)} total predictions "
            f"-> {predictions_file}"
        )


def main() -> int:
    args = parse_args()
    if args.issue_ids:
        issue_ids = [x.strip() for x in args.issue_ids.split(",") if x.strip()]
    else:
        issue_ids = load_issue_ids(args.issue_ids_file)

    ablations = [x.strip() for x in args.ablations.split(",") if x.strip()]
    issue_index = load_issue_index(args.dataset, args.hf_dataset)

    for ablation in ablations:
        print(f"\n[codex-ci-repair] Starting ablation: {ablation}")
        print(f"[codex-ci-repair] Issues to process: {len(issue_ids)}")

        failed_issues = []

        for wanted_id in issue_ids:
            issue = issue_index.get(str(wanted_id))
            if issue is None:
                print(f"[codex-ci-repair] WARNING: Issue {wanted_id} not found in dataset, skipping")
                failed_issues.append((wanted_id, "not found in dataset"))
                continue

            print(f"[codex-ci-repair] issue={wanted_id} ablation={ablation}")

            try:
                run_issue(args, ablation, issue)
            except Exception as exc:
                print(f"[codex-ci-repair] ERROR processing issue {wanted_id}: {exc}")
                failed_issues.append((wanted_id, str(exc)))
                # Continue with next issue instead of crashing
                continue

        # Auto-consolidate predictions after processing all issues
        consolidate_predictions(args.output_root, ablation, args.context_model)
        print(f"[codex-ci-repair] Completed ablation: {ablation}")

        if failed_issues:
            print(f"\n[codex-ci-repair] WARNING: {len(failed_issues)} issues failed:")
            for issue_id, error in failed_issues[:10]:  # Show first 10
                print(f"  - {issue_id}: {error[:100]}")
            if len(failed_issues) > 10:
                print(f"  ... and {len(failed_issues) - 10} more")

    # NOTE: For ablation studies, each ablation+model gets its own predictions.json
    # Top-level consolidation disabled to avoid mixing different runs
    # consolidate_all_predictions(args.output_root, ablations, args.context_model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
