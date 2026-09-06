#!/usr/bin/env python3
"""
Run Codex CLI over CI-repair benchmark issues using prepared CI documents.

This is intentionally an external runner. It does not change Codex core
behavior; it prepares the same task document that a user would paste into
Codex, then invokes `codex exec` from a checked-out failing repository.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import requests
import signal
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
from codex.scripts.ci_repair_prompts import (
    prompt_cache_info,
    prompt_mode,
    render_ci_repair_prompt,
)
from utilities.ci_log_analyzer import _run_log_analysis
from utilities.git_checkout import prepare_repo_checkout as prepare_git_checkout
from utilities.git_patch import PatchValidationError
from utilities.run_metrics import (
    RunMetricsRecorder,
    completed_instance_ids,
    estimate_cost_usd,
    metric_instance_ids,
    safe_metrics_call,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval_set.jsonl"
DEFAULT_ISSUE_IDS_FILE = PROJECT_ROOT / "data" / "eval_issue_ids.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "codex"
DEFAULT_MEMORY_ROOT = PROJECT_ROOT / "data" / "back_trs"
DEFAULT_LOG_CACHE = PROJECT_ROOT / "data" / "log_details.json"
DEFAULT_WORKFLOW_CACHE = PROJECT_ROOT / "data" / "workflow_validation_cache.json"

sys.path.insert(0, str(PROJECT_ROOT))


def _print_auth_banner(provider: str, endpoint: str, code_home: Path) -> None:
    print("\n==========================================")
    print("Auth mode: API key")
    print(f"Provider: {provider}")
    print(f"Endpoint: {endpoint}")
    print(f"CODEX_HOME: {code_home}")


def _headers(token: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if extra:
        h.update(extra)
    return h


# Canonical model aliases so users can pass simple names
ALIASES: dict[str, str] = {
    # MiniMax aliases
    "minimax": "minimax/minimax-m2.5",
    "minimax2.5": "minimax/minimax-m2.5",
    "minimax-m2.5": "minimax/minimax-m2.5",
}


def canonical_model(name: str | None) -> str | None:
    if not name:
        return None
    n = str(name).strip()
    return ALIASES.get(n, n)


def preflight_model(
    model: str, metrics_recorder: RunMetricsRecorder | None = None
) -> None:
    """Verify model with minimal API call and print banner.

    Handles GPT-5.* (max_completion_tokens / max_output_tokens) and OpenRouter
    (max_tokens).
    """
    if not model:
        return

    model = canonical_model(model) or model
    configured_provider = os.environ.get("CODEX_PROVIDER", "").strip().lower()
    is_openai_model = model.startswith(("gpt-", "chatgpt-", "codex-")) or (
        len(model) > 1 and model[0] == "o" and model[1].isdigit()
    )
    provider = (
        configured_provider
        if configured_provider in {"openai", "openrouter"}
        else ("openai" if is_openai_model else "openrouter")
    )
    if provider == "openai":
        endpoint = os.environ.get(
            "CODEX_API_BASE",
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        key_env = "OPENAI_API_KEY"
        token = os.environ.get("OPENAI_API_KEY", "").strip()
    else:
        endpoint = os.environ.get(
            "CODEX_API_BASE",
            os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        key_env = "OPENROUTER_API_KEY"
        token = os.environ.get("OPENROUTER_API_KEY", "").strip()

    # Ensure Codex sees OpenAI-compatible env regardless of provider
    if provider == "openai":
        os.environ["OPENAI_BASE_URL"] = endpoint
    else:
        os.environ["OPENAI_BASE_URL"] = endpoint
        if token:
            os.environ["OPENAI_API_KEY"] = token

    configured_code_home = os.environ.get("CODEX_HOME", "").strip()
    code_home = (
        Path(configured_code_home).expanduser()
        if configured_code_home
        else PROJECT_ROOT / "codex" / ".codex-config"
    )
    code_home.mkdir(parents=True, exist_ok=True)
    _print_auth_banner(provider, endpoint, code_home)

    if not token:
        raise SystemExit(f"Missing {key_env} for provider {provider}")

    # Try chat/completions
    url = endpoint.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": "ok"}]}
    if provider == "openai":
        payload["max_completion_tokens"] = 16
    else:
        payload["max_tokens"] = 16

    print(
        f"[codex-ci-repair] Preflight: verifying {model} through {provider} "
        "(request timeout: 20s)",
        flush=True,
    )
    safe_metrics_call(
        metrics_recorder,
        "begin_api_call",
        phase="model_preflight",
        model=model,
    )
    call_started = time.time()
    r = requests.post(url, headers=_headers(token), json=payload, timeout=20)
    if r.status_code == 200:
        if metrics_recorder is not None:
            safe_metrics_call(
                metrics_recorder,
                "record_response",
                response=r.json(),
                phase="model_preflight",
                model=model,
                duration_seconds=time.time() - call_started,
            )
        print("\n✓ Model verified: chat/completions")
        return

    safe_metrics_call(
        metrics_recorder,
        "record_api_call",
        phase="model_preflight",
        model=model,
        duration_seconds=time.time() - call_started,
        status="failed",
        error=f"chat/completions returned HTTP {r.status_code}",
    )

    if provider == "openai":
        url2 = endpoint.rstrip("/") + "/responses"
        payload2 = {"model": model, "input": "ok", "max_output_tokens": 16}
        safe_metrics_call(
            metrics_recorder,
            "begin_api_call",
            phase="model_preflight",
            model=model,
        )
        call_started = time.time()
        r2 = requests.post(url2, headers=_headers(token), json=payload2, timeout=20)
        if r2.status_code == 200:
            if metrics_recorder is not None:
                safe_metrics_call(
                    metrics_recorder,
                    "record_response",
                    response=r2.json(),
                    phase="model_preflight",
                    model=model,
                    duration_seconds=time.time() - call_started,
                )
            print("\n✓ Model verified: responses")
            return
        print("\n✗ FATAL: Model test failed!")
        print("  Model: ", model)
        print("  Error: ", f"chat={r.status_code} {str(r.text)[:200]} | responses={r2.status_code} {str(r2.text)[:200]}")
        raise SystemExit(1)

    print("\n✗ FATAL: Model test failed!")
    print("  Model: ", model)
    print("  Error: ", f"chat={r.status_code} {str(r.text)[:200]}")
    raise SystemExit(1)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_json_atomic(path: Path, data: Any) -> None:
    """Atomically replace a JSON file after writing it in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


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

    Delegates to utilities.git_checkout.prepare_repo_checkout for the actual work.

    Returns:
        The SHA of the checkout commit (for later diff comparison)
    """
    return prepare_git_checkout(issue, checkout, refresh, dry_run)


def make_context_llm(model: str | None, metrics_recorder: Any = None) -> Any:
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
    return LitellmModel(model_name=model_name, metrics_recorder=metrics_recorder)


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
            # Use shared cache utility - checks cache first, generates if missing
            from utilities.ci_cache import load_validation_sequence

            verification = load_validation_sequence(
                issue_id=issue_id(issue),
                sha_fail=str(issue.get("sha_fail") or ""),
                workflow_content=str(issue.get("workflow") or ""),
                workflow_path=str(issue.get("workflow_path") or ""),
                repo_path=str(checkout),
                llm=context_llm,
                save=True,  # Auto-save to cache if generated
            )
            verification["source"] = verification.get("source") or "workflow_cache"

        except Exception as exc:
            raise RuntimeError(
                f"Workflow verification generation failed for issue "
                f"{issue_id(issue)}: {exc}"
            ) from exc

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
                "workflow_name": issue.get("workflow_name", ""),
                "workflow_path": issue.get("workflow_path", ""),
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


def _ci_problem_to_unified(
    ci_failure: dict[str, Any], ci_problem: dict[str, Any]
) -> dict[str, Any]:
    """Preserve either the current Stage 0 schema or the legacy CI schema."""
    files = ci_problem.get("files") or []
    if not files and ci_problem.get("file"):
        files = [ci_problem["file"]]
    elif not isinstance(files, list):
        files = [files]

    failure_signals = ci_problem.get("failure_signals") or []
    if not failure_signals:
        failure_signals = extract_error_signals_from_ci(ci_failure, ci_problem)

    return {
        **ci_problem,
        "problem": ci_problem.get("problem")
        or ci_problem.get("reason")
        or f"Fix {ci_problem.get('file', 'CI failure')}",
        "root_cause": ci_problem.get("root_cause")
        or "Analyze from CI failure context below",
        "files": files,
        "failure_signals": failure_signals,
        "repair_strategy": None,
        "failure_type": ci_problem.get("failure_type")
        or infer_failure_type(ci_problem),
        "verification_cmd": ci_problem.get("verification_cmd")
        or ci_problem.get("validation_cmd")
        or ci_problem.get("failed_cmd", ""),
        "source": "ci_failure",
    }


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
        unified = [
            _ci_problem_to_unified(ci_failure, ci_prob)
            for ci_prob in ci_problems
        ]

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
            unified = [
                _ci_problem_to_unified(ci_failure, ci_prob)
                for ci_prob in ci_problems
            ]

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
    """
    Simple fallback extraction of problems from CI failure.

    This is only used when Stage 0 LLM decomposition fails.
    Normally, baseline mode will use MemoryPlugin.decompose_only() instead.
    """
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

    dynamic_context = f"""## Problem To Fix

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
"""
    return render_ci_repair_prompt("baseline", dynamic_context)


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

    dynamic_context = f"""## Problem To Fix

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
"""
    return render_ci_repair_prompt("memory", dynamic_context)


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
    mode = prompt_mode(ablation)

    if mode == "baseline":
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


def _validate_and_locate_problem_files(
    problem: dict[str, Any], checkout: Path
) -> tuple[bool, str]:
    """
    Validate problem before attempting to fix it.
    Try to locate files even if paths don't match exactly.

    Returns:
        (is_valid, skip_reason) - True if valid, False with reason if should skip
    """
    # Check for impossible stdlib module import errors
    error_signals = problem.get("error_signals", []) or problem.get("failure_signals", [])
    if not isinstance(error_signals, list):
        error_signals = [error_signals] if error_signals else []

    # Python stdlib modules that cannot actually be missing
    STDLIB_MODULES = [
        'itertools', 'os', 'sys', 'json', 're', 'time', 'datetime',
        'collections', 'functools', 'typing', 'pathlib', 'subprocess',
        'math', 'random', 'string', 'copy', 'io', 'traceback'
    ]

    for signal in error_signals:
        signal_str = str(signal).lower()
        for mod in STDLIB_MODULES:
            if f"no module named '{mod}'" in signal_str or f'no module named "{mod}"' in signal_str:
                return False, f"Invalid error: '{mod}' is Python stdlib, cannot be missing"

    # Check and try to locate affected files
    files = problem.get("files", [])
    if not files:
        # No files specified - let agent try (might be environment/config issue)
        return True, ""

    if not isinstance(files, list):
        files = [files]

    located_files = []
    missing_files = []

    for f in files:
        # Extract path from various formats
        if isinstance(f, dict):
            file_path_str = f.get("path") or f.get("file") or ""
        else:
            file_path_str = str(f)

        if not file_path_str:
            continue

        file_path = checkout / file_path_str

        # 1. Check exact path
        if file_path.exists():
            located_files.append(file_path_str)
            continue

        # 2. Try to find by filename using find command
        filename = Path(file_path_str).name
        try:
            result = subprocess.run(
                ["find", str(checkout), "-type", "f", "-name", filename, "-not", "-path", "*/.git/*"],
                capture_output=True,
                text=True,
                timeout=5
            )

            found_paths = [p.strip() for p in result.stdout.split('\n') if p.strip()]

            if found_paths:
                # Found at least one match - use first one
                try:
                    relative_path = Path(found_paths[0]).relative_to(checkout)
                    located_files.append(str(relative_path))
                    print(f"  ✓ Located {filename} at: {relative_path}")
                    # Update problem with correct path
                    if isinstance(f, dict):
                        f["path"] = str(relative_path)
                        f["file"] = str(relative_path)
                    continue
                except ValueError:
                    pass
        except (subprocess.TimeoutExpired, Exception):
            pass

        # 3. Search in common directories
        for common_dir in ['src', 'lib', 'tests', 'test', '.', 'app']:
            search_path = checkout / common_dir
            if not search_path.exists():
                continue
            # Use find in subdirectory
            try:
                result = subprocess.run(
                    ["find", str(search_path), "-type", "f", "-name", filename, "-not", "-path", "*/.git/*"],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                found = [p.strip() for p in result.stdout.split('\n') if p.strip()]
                if found:
                    try:
                        relative_path = Path(found[0]).relative_to(checkout)
                        located_files.append(str(relative_path))
                        print(f"  ✓ Located {filename} in {common_dir}/: {relative_path}")
                        if isinstance(f, dict):
                            f["path"] = str(relative_path)
                            f["file"] = str(relative_path)
                        break
                    except ValueError:
                        pass
            except (subprocess.TimeoutExpired, Exception):
                continue
        else:
            # Not found anywhere
            missing_files.append(file_path_str)

    # Decision: Skip only if ALL files are missing
    if files and not located_files and missing_files:
        return False, f"Cannot locate any affected files: {', '.join(missing_files[:3])}"

    if missing_files:
        print(f"  ⚠️  Could not locate {len(missing_files)} file(s): {', '.join(missing_files[:2])}")
        print(f"  ✓ But found {len(located_files)} file(s) - proceeding with those")

    return True, ""


def run_codex(
    checkout: Path,
    prompt_path: Path,
    transcript_path: Path,
    command: str,
    timeout: int,
    dry_run: bool,
    metrics_recorder: RunMetricsRecorder | None = None,
    metrics_phase: str = "repair_agent",
) -> dict[str, Any]:
    """
    Run codex agent and stream output in real-time to both terminal and file.

    This allows you to see what the agent is doing while it runs.
    """
    cmd = shlex.split(command)
    json_mode = "--json" in cmd
    if not json_mode:
        cmd.append("--json")
        json_mode = True
    prompt = prompt_path.read_text(encoding="utf-8")
    started = time.time()
    usage_totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    total_cost_usd = 0.0
    cost_sources: set[str] = set()
    usage_event_recorded = False
    expected_model = None
    for i, arg in enumerate(cmd):
        if arg == "--model" and i + 1 < len(cmd):
            expected_model = cmd[i + 1]
            break

    if dry_run:
        write_text(transcript_path, f"DRY RUN: would execute {cmd} in {checkout}\n")
        return {
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "dry_run": True,
            "usage": usage_totals,
            "cost_usd": 0.0,
        }

    print(f"\n{'='*80}")
    print(f"[AGENT] Starting: {' '.join(cmd)}")
    print(f"[AGENT] Working directory: {checkout}")
    print(f"[AGENT] Output will be saved to: {transcript_path}")
    print(f"{'='*80}\n")

    # Stream output to both terminal and file in real-time
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass environment variables to subprocess
    # Keep subprocess credentials aligned with the provider selected by the
    # launcher. Do not infer the provider from the shape of an API key.
    env = os.environ.copy()

    selected_provider = env.get("CODEX_PROVIDER", "").strip().lower()
    if selected_provider == "openai":
        env.pop('OPENROUTER_API_KEY', None)
        env.pop('OPENROUTER_BASE_URL', None)
        env.pop('MINIMAX_API_KEY', None)
        env.pop('ANTHROPIC_API_KEY', None)
        env['OPENAI_BASE_URL'] = env.get('CODEX_API_BASE', 'https://api.openai.com/v1')
    elif selected_provider == "openrouter":
        env.pop('ANTHROPIC_API_KEY', None)
        env.pop('MINIMAX_API_KEY', None)
        env['OPENAI_BASE_URL'] = env.get('CODEX_API_BASE', 'https://openrouter.ai/api/v1')
    elif 'ANTHROPIC_API_KEY' in env:
        # Backward-compatible cleanup for callers that do not use the launcher.
        env.pop('OPENROUTER_API_KEY', None)
        env.pop('OPENAI_API_KEY', None)
        env.pop('OPENAI_BASE_URL', None)

    proc: subprocess.Popen[str] | None = None
    watchdog_stop = threading.Event()
    timed_out = False

    def kill_process_group():
        """Watchdog thread: enforce hard deadline and kill process group."""
        nonlocal timed_out
        if watchdog_stop.wait(timeout=timeout if timeout else 3600):
            return  # Process finished normally

        # Timeout reached - kill the entire process group
        timed_out = True
        print(f"\n{'='*80}")
        print(f"⏱️  WATCHDOG TIMEOUT: Killing process after {timeout}s")
        print(f"{'='*80}\n")

        if proc and proc.poll() is None:
            try:
                # Kill entire process group (includes child processes)
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(10)
                if proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                # Process already dead or permission issue - try direct kill
                try:
                    proc.terminate()
                    time.sleep(5)
                    if proc.poll() is None:
                        proc.kill()
                except:
                    pass

    safe_metrics_call(
        metrics_recorder,
        "begin_api_call",
        phase=metrics_phase,
        model=expected_model,
    )
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            # Start process in its own process group for clean termination
            proc = subprocess.Popen(
                cmd,
                cwd=checkout,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,  # Create new process group
            )

            # Start watchdog thread
            watchdog = threading.Thread(target=kill_process_group, daemon=True)
            watchdog.start()

            # Send prompt to stdin
            if proc.stdin:
                proc.stdin.write(prompt)
                proc.stdin.close()

            # Stream output line by line and checkpoint usage events.
            inactivity_timeout = 300  # 5 minutes of silence = timeout
            last_output_time = time.time()

            if proc.stdout:
                model_verified = json_mode
                detected_model = None
                line_count = 0

                for line in proc.stdout:
                    # Check if watchdog already killed the process
                    if timed_out:
                        break

                    # Check inactivity timeout
                    now = time.time()
                    if now - last_output_time > inactivity_timeout:
                        print(f"\n{'='*80}")
                        print(f"⏱️  INACTIVITY TIMEOUT: No output for {inactivity_timeout}s")
                        print(f"{'='*80}\n")
                        watchdog_stop.set()  # Stop watchdog
                        try:
                            pgid = os.getpgid(proc.pid)
                            os.killpg(pgid, signal.SIGTERM)
                            time.sleep(5)
                            if proc.poll() is None:
                                os.killpg(pgid, signal.SIGKILL)
                        except:
                            proc.terminate()
                            time.sleep(5)
                            if proc.poll() is None:
                                proc.kill()
                        raise subprocess.TimeoutExpired(cmd=command, timeout=inactivity_timeout)

                    last_output_time = now

                    event = None
                    if json_mode:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            event = None
                    if isinstance(event, dict) and event.get("type") == "turn.completed":
                        usage_event_recorded = True
                        event_usage = event.get("usage") or {}
                        normalized_usage = {
                            field: int(event_usage.get(field) or 0)
                            for field in usage_totals
                        }
                        for field, value in normalized_usage.items():
                            usage_totals[field] += value
                        call_cost, cost_source = estimate_cost_usd(
                            expected_model or "", normalized_usage
                        )
                        total_cost_usd += call_cost
                        cost_sources.add(cost_source)
                        if metrics_recorder is not None:
                            safe_metrics_call(
                                metrics_recorder,
                                "record_api_call",
                                phase=metrics_phase,
                                model=expected_model,
                                duration_seconds=time.time() - started,
                                usage=normalized_usage,
                                cost_usd=call_cost,
                                cost_source=cost_source,
                            )

                    # Human output identifies the model near the beginning.
                    if not json_mode and line_count < 20:
                        if line.startswith("model:"):
                            detected_model = line.split("model:")[1].strip()
                            if expected_model and detected_model != expected_model:
                                print(f"\n{'='*80}")
                                print(f"❌ MODEL MISMATCH ERROR!")
                                print(f"{'='*80}")
                                print(f"Expected model: {expected_model}")
                                print(f"Detected model: {detected_model}")
                                print(f"\nCodex is using a DIFFERENT model than requested!")
                                print(f"Stopping execution to prevent incorrect results.")
                                print(f"{'='*80}\n")
                                proc.kill()
                                raise RuntimeError(
                                    f"Model mismatch: expected '{expected_model}' but got '{detected_model}'. "
                                    f"Check your Codex configuration."
                                )
                            elif expected_model and detected_model == expected_model:
                                model_verified = True
                                print(f"✓ Model verified: {detected_model}\n")

                    line_count += 1

                    # Print to terminal and save the raw JSONL transcript.
                    print(line, end="", flush=True)
                    f.write(line)
                    f.flush()

                # JSON mode relies on the explicit command and API preflight.
                if expected_model and not model_verified:
                    print(f"\n{'='*80}")
                    print(f"⚠️  WARNING: Could not verify model '{expected_model}'")
                    print(f"Codex output did not include model identification.")
                    print(f"{'='*80}\n")

            # Stop watchdog - process finished normally
            watchdog_stop.set()

            # Wait for process to complete
            proc.wait(timeout=10)

            # Check if watchdog killed the process
            if timed_out:
                raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    except subprocess.TimeoutExpired as exc:
        watchdog_stop.set()  # Stop watchdog
        # Handle timeout gracefully - return timed_out result instead of crashing
        elapsed = time.time() - started
        if not usage_event_recorded and metrics_recorder is not None:
            safe_metrics_call(
                metrics_recorder,
                "record_api_call",
                phase=metrics_phase,
                model=expected_model,
                duration_seconds=elapsed,
                status="timed_out",
                error=f"Timeout after {elapsed:.1f}s",
            )
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        print(f"\n{'='*80}")
        print(f"⏱️  TIMED OUT after {elapsed:.1f}s")
        print(f"{'='*80}\n")

        # Return partial result with timed_out flag
        return {
            "returncode": -1,
            "elapsed_seconds": elapsed,
            "dry_run": False,
            "timed_out": True,
            "usage": usage_totals,
            "cost_usd": round(total_cost_usd, 10),
            "cost_source": ",".join(sorted(cost_sources)) or "unavailable",
        }
    except BaseException as exc:
        watchdog_stop.set()  # Stop watchdog
        if not usage_event_recorded and metrics_recorder is not None:
            safe_metrics_call(
                metrics_recorder,
                "record_api_call",
                phase=metrics_phase,
                model=expected_model,
                duration_seconds=time.time() - started,
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise

    elapsed = time.time() - started
    if not usage_event_recorded and metrics_recorder is not None:
        safe_metrics_call(
            metrics_recorder,
            "record_api_call",
            phase=metrics_phase,
            model=expected_model,
            duration_seconds=elapsed,
            status="completed" if proc.returncode == 0 else "failed",
            error=(
                None
                if proc.returncode == 0
                else f"codex exec exited with status {proc.returncode}"
            ),
        )

    print(f"\n{'='*80}")
    print(f"[AGENT] Completed in {elapsed:.1f}s with exit code {proc.returncode}")
    print(f"{'='*80}\n")

    return {
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "dry_run": False,
        "usage": usage_totals,
        "cost_usd": round(total_cost_usd, 10),
        "cost_source": ",".join(sorted(cost_sources)) or "unavailable",
    }


def _run_git_diff(
    checkout: Path, revision_args: list[str], *, force_binary: bool = False
) -> subprocess.CompletedProcess[bytes]:
    """Run git diff as bytes so non-UTF-8 repository content is safe."""
    command = ["git"]
    attributes_file = None
    try:
        if force_binary:
            attributes_file = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8"
            )
            attributes_file.write("* binary\n")
            attributes_file.flush()
            command.extend(["-c", f"core.attributesFile={attributes_file.name}"])

        command.extend(["diff", "--binary", *revision_args])
        return subprocess.run(
            command,
            cwd=checkout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        if attributes_file is not None:
            attributes_file.close()


def _decode_git_diff(checkout: Path, revision_args: list[str]) -> tuple[str, int]:
    """Return a UTF-8-safe patch, falling back to Git binary patch records."""
    proc = _run_git_diff(checkout, revision_args)
    try:
        return proc.stdout.decode("utf-8"), proc.returncode
    except UnicodeDecodeError as exc:
        print(
            "[save_patch_and_result] Non-UTF-8 bytes in git diff "
            f"at offset {exc.start}; regenerating affected content as a binary patch"
        )

    binary_proc = _run_git_diff(checkout, revision_args, force_binary=True)
    try:
        # Preserve the terminating newline: Git binary patch records are
        # invalid if callers strip their final line ending.
        patch = binary_proc.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "Git binary diff unexpectedly contained non-UTF-8 bytes"
        ) from exc
    return patch, binary_proc.returncode


def git_diff(checkout: Path, original_commit: str = None) -> str:
    """Capture all agent changes against the failed commit as a unified diff."""
    if not original_commit:
        raise PatchValidationError("The failed commit is required to generate a complete repair patch")

    result = subprocess.run(
        ["git", "diff", original_commit],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


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

    # If original_commit provided, get files changed in working tree vs base
    if original_commit:
        proc = subprocess.run(
            ["git", "diff", "--name-only", original_commit],
            cwd=checkout,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        unified_files = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
        return unified_files if unified_files else uncommitted_files

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
    diff = git_diff(checkout, original_sha or issue.get("sha_fail"))
    stats = subprocess.run(
        ["git", "apply", "--numstat", "-z"], input=diff.encode("utf-8"),
        cwd=checkout, capture_output=True, check=True,
    ).stdout if diff else b""
    files = [os.fsdecode(record.split(b"\t", 2)[2]) for record in stats.split(b"\0") if record]

    print(f"[save_patch_and_result] Diff size: {len(diff)} bytes")
    print(f"[save_patch_and_result] Changed files: {len(files)} files")
    print(f"[save_patch_and_result] Writing to {result_dir / 'patch.diff'}")

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "patch.diff").write_bytes(diff.encode("utf-8"))
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


def _run_issue(
    args: argparse.Namespace,
    ablation: str,
    issue: dict[str, Any],
    metrics: RunMetricsRecorder,
) -> bool:
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
    context_llm = (
        make_context_llm(args.context_model, metrics_recorder=metrics)
        if args.generate_missing_analysis
        else None
    )
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
    # Both baseline and memory modes use Stage 0 LLM decomposition
    # Baseline: Stage 0 only (no memory retrieval)
    # Memory modes: Stage 0 + full memory pipeline (Stages 1-9)
    if ablation.lower() == "baseline":
        # Baseline: Use Stage 0 decomposition only (no memory retrieval)
        print(f"[DEBUG] Baseline mode: Using Stage 0 LLM decomposition (no memory)")

        if not context_llm:
            # Fallback: If no LLM, use simple extraction
            print(f"[DEBUG]   No LLM available, falling back to simple extraction")
            ci_problems = extract_problem_list(ci_failure)
        else:
            # Use Stage 0 LLM decomposition (same as memory modes, but stop there)
            try:
                from memory_plugin import MemoryPlugin
                temp_plugin = MemoryPlugin(
                    memory_root=args.memory_root or Path("data/back_trs"),
                    result_dir=str(result_dir),
                    ablation="baseline",
                    llm=context_llm,
                    enabled=False  # Don't access memory, just use decompose_only
                )
                ci_problems = temp_plugin.decompose_only(
                    ci_failure=ci_failure,
                    verification=verification,
                    issue_metadata={
                        "task_id": issue_id(issue),
                        "sha_fail": issue.get("sha_fail"),
                        "repo": repo_slug(issue),
                        "workflow_name": issue.get("workflow_name", ""),
                        "workflow_path": issue.get("workflow_path", ""),
                    }
                )
                # Add source tag
                for prob in ci_problems:
                    prob["source"] = "ci_failure"
                    prob["repair_strategy"] = None  # No repair strategies in baseline
                print(f"[DEBUG]   Stage 0 decomposed into {len(ci_problems)} problems")
            except Exception as e:
                print(f"[DEBUG]   Stage 0 decomposition failed: {e}")
                print(f"[DEBUG]   Falling back to simple extraction")
                ci_problems = extract_problem_list(ci_failure)

        memory_problems = []
    else:
        # Memory modes: Let memory plugin decompose CI and search memory (full pipeline)
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
        return False

    print(f"\n[DEBUG] Problems to fix (one by one):")
    for idx, p in enumerate(problems, 1):
        source = p.get('source', 'unknown')
        has_repair = '✓' if p.get('repair_strategy') else '✗'
        print(f"[DEBUG]   {idx}. [{source}] [repair:{has_repair}] {p.get('problem', 'N/A')[:70]}...")
    print()
    problem_results = []
    for problem in problems:
        # ═══════════════════════════════════════════════════════════════
        # VALIDATE PROBLEM: Skip if files can't be located or problem is invalid
        # ═══════════════════════════════════════════════════════════════
        is_valid, skip_reason = _validate_and_locate_problem_files(problem, checkout)
        if not is_valid:
            print(f"\n{'='*80}")
            print(f"⚠️  SKIPPING Problem {problem.get('number', '?')}: {skip_reason}")
            print(f"{'='*80}\n")
            problem_results.append({
                "problem": problem,
                "returncode": -1,
                "elapsed_seconds": 0,
                "skipped": True,
                "skip_reason": skip_reason,
                "usage": {},
                "cost_usd": 0,
            })
            continue  # Skip to next problem

        document = compose_issue_document(
            issue,
            ci_failure,
            verification,
            problem,
            len(problems),
            ablation,
        )
        cache_info = prompt_cache_info(document)
        print(
            "[Prompt cache] "
            f"layout={cache_info['layout']} "
            f"template={cache_info['template_fingerprint']} "
            f"stable_chars={cache_info['stable_prefix_chars']} "
            f"dynamic_chars={cache_info['dynamic_context_chars']}"
        )
        prompt_path = write_issue_document(result_dir, problem, document)
        transcript_path = result_dir / f"codex_transcript_problem_{problem['number']}.txt"
        # Ensure Codex command has a model flag; if missing, inject canonical model
        codex_cmd = args.codex_command
        if " --model " not in f" {codex_cmd} ":
            cm = canonical_model(args.context_model)
            if cm:
                codex_cmd = f"{codex_cmd} --model {cm}"

        # Run Codex with timeout - catch timeout exception to continue to next problem
        try:
            run_result = run_codex(
                checkout,
                prompt_path,
                transcript_path,
                codex_cmd,
                args.timeout,
                args.dry_run,
                metrics_recorder=metrics,
                metrics_phase=f"repair_agent.problem_{problem['number']}",
            )
        except subprocess.TimeoutExpired:
            print(f"\n{'='*80}")
            print(f"⏱️  TIMEOUT: Problem {problem.get('number', '?')} exceeded {args.timeout}s limit")
            print(f"{'='*80}\n")
            run_result = {
                "returncode": -2,
                "elapsed_seconds": args.timeout,
                "timeout": True,
                "usage": {},
                "cost_usd": 0,
            }
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ ERROR: Problem {problem.get('number', '?')} failed: {e}")
            print(f"{'='*80}\n")
            run_result = {
                "returncode": -3,
                "elapsed_seconds": 0,
                "error": str(e),
                "usage": {},
                "cost_usd": 0,
            }

        problem_results.append(
            {"problem": problem, "prompt_cache": cache_info, **run_result}
        )

        # Save intermediate results on failure too, so we capture partial fixes
        failed = run_result["returncode"] != 0

        # Optional mid-issue checkpoint to avoid losing fixes if run stops
        if getattr(args, "save_after_each_problem", True):
            try:
                save_patch_and_result(
                    result_dir,
                    checkout,
                    issue,
                    ablation,
                    prediction_model_name(args),
                    problem_results,
                    verification,
                    None,  # no external verification
                    original_sha,
                )
            except Exception as exc:
                print(f"[codex-ci-repair] WARNING: mid-issue save failed: {exc}")

        # Stop processing further problems if this one failed
        if failed:
            break

    # Agent handles verification internally for each problem
    # No external verification needed
    print(f"\n[DEBUG] Saving patch and result for issue {issue_id(issue)}...")
    print(f"[DEBUG]   Result dir: {result_dir}")
    print(f"[DEBUG]   Checkout: {checkout}")
    print(f"[DEBUG]   Problem results: {len(problem_results)} problems processed")

    # Simple patch collection - no reconciliation
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
    completed = bool(
        len(problem_results) == len(problems)
        and all(result.get("returncode") == 0 for result in problem_results)
        and (result_dir / "patch.diff").exists()
        and (result_dir / "patch.diff").read_text(encoding="utf-8").strip()
    )
    if completed and getattr(args, "incremental_predictions", True):
        append_prediction_for_issue(
            args.output_root, ablation, issue_id(issue), args.context_model
        )
    return completed


def run_issue(args: argparse.Namespace, ablation: str, issue: dict[str, Any]) -> None:
    """Run one issue while preserving every interrupted or retried attempt."""
    result_dir = (
        args.output_root
        / _ablation_dir_name(ablation, args.context_model)
        / issue_id(issue)
    )
    metrics = RunMetricsRecorder(
        result_dir.parent,
        issue_id(issue),
        agent="codex",
        model=prediction_model_name(args),
        direction=args.direction,
        ablation=ablation,
    )
    try:
        completed = _run_issue(args, ablation, issue, metrics)
    except KeyboardInterrupt as exc:
        safe_metrics_call(
            metrics,
            "finish",
            status="interrupted",
            error=str(exc) or "KeyboardInterrupt",
        )
        raise
    except BaseException as exc:
        safe_metrics_call(
            metrics,
            "finish",
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        safe_metrics_call(
            metrics,
            "finish",
            status="completed" if completed else "failed",
            metadata={"patch_generated": completed},
        )


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
    parser.add_argument(
        "--incremental-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append/update predictions.json after each issue completes. "
            "Enabled by default to avoid data loss if a run stops early."
        ),
    )
    parser.add_argument(
        "--save-after-each-problem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write patch.diff/result.json after each problem within an issue. "
            "Provides mid-issue checkpoints so partial work is preserved."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip issue ids already present in this model/ablation's "
            "predictions.json. Enabled by default; use --no-resume to rerun them."
        ),
    )
    parser.add_argument("--refresh-checkout", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of issues to process in parallel (per ablation). "
            "Use with care; heavy on network and disk. Default: 1."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--direction",
        choices=["backward", "forward"],
        default="backward",
        help="Select backward (default) or forward memory root when memory is enabled",
    )
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

    # Aggregate cost and time from problem results
    total_cost_usd = sum(
        pr.get("cost_usd", 0.0) for pr in result.get("problem_results", [])
    )
    total_elapsed_seconds = sum(
        pr.get("elapsed_seconds", 0.0) for pr in result.get("problem_results", [])
    )

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
        "total_cost_usd": total_cost_usd,
        "total_elapsed_seconds": total_elapsed_seconds,
    }


def prediction_has_patch(prediction: Any) -> bool:
    """Return whether a prediction is complete enough to skip on resume."""
    if not isinstance(prediction, dict):
        return False
    if prediction.get("patch_generated") is False:
        return False
    return bool(str(prediction.get("diff") or "").strip())


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

    local_predictions = [
        prediction_from_result(result_file)
        for result_file in sorted(results_dir.glob("*/result.json"))
    ]

    # Consolidation is intentionally merge-only. Existing uploaded predictions
    # are the resume ledger and must never disappear merely because their
    # per-issue result directories are not present on this machine.
    import fcntl

    lock_file = predictions_file.with_suffix(".lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("w") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            predictions = load_json(predictions_file, [])
            if not isinstance(predictions, list):
                raise ValueError(f"Expected a JSON list in {predictions_file}")

            existing_indexes = {
                str(prediction.get("id")): index
                for index, prediction in enumerate(predictions)
                if isinstance(prediction, dict) and prediction.get("id") is not None
            }
            additions = 0
            replacements = 0
            for prediction in local_predictions:
                prediction_id = str(prediction.get("id"))
                existing_index = existing_indexes.get(prediction_id)
                if existing_index is None:
                    predictions.append(prediction)
                    existing_indexes[prediction_id] = len(predictions) - 1
                    additions += 1
                elif (
                    not prediction_has_patch(predictions[existing_index])
                    and prediction_has_patch(prediction)
                ):
                    predictions[existing_index] = prediction
                    replacements += 1

            if additions or replacements:
                write_json_atomic(predictions_file, predictions)
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)

    if additions or replacements:
        print(
            f"\n[codex-ci-repair] Added {additions} missing predictions and "
            f"completed {replacements} previously empty predictions "
            f"-> {predictions_file}"
        )

    return predictions


def _ablation_dir_name(ablation: str, context_model: str | None) -> str:
    safe_ablation = ablation.replace("+", "_").lower()
    if context_model:
        model_suffix = f"_{context_model.replace('/', '_').replace('.', '_')}"
        return f"{safe_ablation}{model_suffix}"
    return safe_ablation


def existing_prediction_ids(
    output_root: Path, ablation: str, context_model: str | None = None
) -> set[str]:
    """Return IDs with non-empty patches for one model/ablation run."""
    results_dir = output_root / _ablation_dir_name(ablation, context_model)
    predictions_file = results_dir / "predictions.json"
    predictions = load_json(predictions_file, [])
    if not isinstance(predictions, list):
        return set()
    metrics_completed = completed_instance_ids(results_dir)
    metrics_known = metric_instance_ids(results_dir)
    return {
        str(prediction.get("id"))
        for prediction in predictions
        if isinstance(prediction, dict)
        and prediction.get("id") is not None
        and prediction_has_patch(prediction)
        and (
            str(prediction.get("id")) not in metrics_known
            or str(prediction.get("id")) in metrics_completed
        )
    }


def append_prediction_for_issue(
    output_root: Path, ablation: str, issue_id: str, context_model: str | None = None
) -> bool:
    """Append new prediction to predictions.json - SKIP if ID already exists.

    IMPORTANT: Only appends NEW IDs, never overwrites existing entries.
    This prevents data loss when multiple processes or manual updates occur.
    """
    import fcntl

    ablation_dir = _ablation_dir_name(ablation, context_model)
    results_dir = output_root / ablation_dir
    result_file = results_dir / str(issue_id) / "result.json"
    if not result_file.exists():
        return False  # nothing to append yet

    prediction = prediction_from_result(result_file)
    new_id = str(prediction.get("id"))
    if not prediction_has_patch(prediction):
        raise RuntimeError(
            f"Issue {new_id} completed without generating a patch; "
            "leaving it pending for the next resume run"
        )

    predictions_file = results_dir / "predictions.json"
    lock_file = predictions_file.with_suffix(".lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_file, "w") as lock_fd:
        # Acquire exclusive lock
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            # Read current file state
            existing = load_json(predictions_file, [])
            if not isinstance(existing, list):
                raise ValueError(f"Expected a JSON list in {predictions_file}")

            # Check if ID already exists - SKIP if present
            existing_index = next(
                (
                    index
                    for index, item in enumerate(existing)
                    if isinstance(item, dict) and str(item.get("id")) == new_id
                ),
                None,
            )
            if existing_index is not None:
                if prediction_has_patch(existing[existing_index]):
                    print(
                        f"  [codex-ci-repair] Skipping {new_id}: "
                        "a completed prediction already exists"
                    )
                    return False

                existing[existing_index] = prediction
                write_json_atomic(predictions_file, existing)
                print(
                    f"  [codex-ci-repair] Completed previously empty prediction "
                    f"{new_id} in predictions.json"
                )
                return True

            # Only append if NEW
            existing.append(prediction)
            write_json_atomic(predictions_file, existing)
            print(f"  [codex-ci-repair] Appended {new_id} to predictions.json")
            return True
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)


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
    # Adjust memory root from direction if memory is enabled and default is in use
    if args.direction == "forward" and str(args.memory_root).endswith("back_trs"):
        args.memory_root = PROJECT_ROOT / "data" / "fwr_trs"

    if args.issue_ids:
        issue_ids = [x.strip() for x in args.issue_ids.split(",") if x.strip()]
    else:
        issue_ids = load_issue_ids(args.issue_ids_file)

    ablations = [x.strip() for x in args.ablations.split(",") if x.strip()]

    needs_processing = True
    if args.resume:
        needs_processing = any(
            any(
                str(wanted_id)
                not in existing_prediction_ids(
                    args.output_root, ablation, args.context_model
                )
                for wanted_id in issue_ids
            )
            for ablation in ablations
        )

    # Verify the model only when at least one selected issue still needs work.
    if args.context_model and needs_processing:
        model_slug = str(args.context_model).replace("/", "_").replace(".", "_")
        preflight_metrics = RunMetricsRecorder(
            args.output_root / f"_run_overhead_{model_slug}",
            f"preflight-{time.time_ns()}",
            agent="codex",
            model=args.context_model,
            direction=args.direction,
            ablation="run_overhead",
        )
        try:
            preflight_model(args.context_model, metrics_recorder=preflight_metrics)
        except BaseException as exc:
            safe_metrics_call(
                preflight_metrics,
                "finish",
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            safe_metrics_call(preflight_metrics, "finish", status="completed")
    elif args.context_model:
        print(
            "[codex-ci-repair] Resume: all selected issues already exist; "
            "skipping model preflight"
        )

    issue_index = load_issue_index(args.dataset, args.hf_dataset)

    total_failed_issues = 0
    for ablation in ablations:
        pending_issue_ids = issue_ids
        if args.resume:
            completed_ids = existing_prediction_ids(
                args.output_root, ablation, args.context_model
            )
            pending_issue_ids = [
                wanted_id
                for wanted_id in issue_ids
                if str(wanted_id) not in completed_ids
            ]
            skipped_count = len(issue_ids) - len(pending_issue_ids)
            if skipped_count:
                print(
                    f"[codex-ci-repair] Resume: skipping {skipped_count} issue(s) "
                    "already present in predictions.json"
                )

        print(f"\n[codex-ci-repair] Starting ablation: {ablation}")
        print(f"[codex-ci-repair] Issues to process: {len(pending_issue_ids)}")

        failed_issues = []
        if args.workers and args.workers > 1:
            print(f"[codex-ci-repair] Running with workers={args.workers}")
            tasks = {}
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for wanted_id in pending_issue_ids:
                    issue = issue_index.get(str(wanted_id))
                    if issue is None:
                        print(f"[codex-ci-repair] WARNING: Issue {wanted_id} not found in dataset, skipping")
                        failed_issues.append((wanted_id, "not found in dataset"))
                        continue
                    print(f"[codex-ci-repair] issue={wanted_id} ablation={ablation}")
                    tasks[ex.submit(run_issue, args, ablation, issue)] = str(wanted_id)

                for fut in as_completed(tasks):
                    wid = tasks[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        print(f"[codex-ci-repair] ERROR processing issue {wid}: {exc}")
                        failed_issues.append((wid, str(exc)))
        else:
            for wanted_id in pending_issue_ids:
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
        print(f"[codex-ci-repair] Completed ablation: {ablation}")

        if failed_issues:
            total_failed_issues += len(failed_issues)
            print(f"\n[codex-ci-repair] WARNING: {len(failed_issues)} issues failed:")
            for issue_id, error in failed_issues[:10]:  # Show first 10
                print(f"  - {issue_id}: {error[:100]}")
            if len(failed_issues) > 10:
                print(f"  ... and {len(failed_issues) - 10} more")

    # NOTE: For ablation studies, each ablation+model gets its own predictions.json
    # Top-level consolidation disabled to avoid mixing different runs
    # consolidate_all_predictions(args.output_root, ablations, args.context_model)

    return 1 if total_failed_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
