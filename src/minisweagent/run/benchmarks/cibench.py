"""
cibench.py
==========
Run mini-SWE-agent on CI-failure repair instances in batch mode.

Each instance is processed entirely locally:
  • The repository is cloned from GitHub at run time (no Docker images needed).
  • The failing commit (sha_fail) is checked out inside the clone.
  • The agent runs in a LocalEnvironment against that checkout.

Input dataset  (JSONL or HuggingFace)
--------------------------------------
Each instance must contain:
  instance_id   str   unique identifier
  sha_fail      str   the commit SHA where CI failed
  repo_owner    str
  repo_name     str
  workflow_path str   e.g. ".github/workflows/test.yml"
  workflow_name str   e.g. "test"
  workflow      str   full workflow YAML content
  logs          any   raw CI logs  (str | list[{step_name,log}])

Output predictions  (preds.json)
---------------------------------
{
  "<instance_id>": {
    "id":       "<instance_id>",
    "sha_fail": "<sha>",
    "diff":     "<git diff output — the patch>"
  },
  ...
}

Each instance also gets a trajectory saved to:
  <output_dir>/<sha_fail>/<sha_fail>.traj.json

Usage examples
--------------
  # Run on all instances in a JSONL file
  mini-swe-agent cibench --dataset ci_data.jsonl --output results/

  # With memory enabled
  mini-swe-agent cibench --dataset ci_data.jsonl --output results/ \\
    --memory-root /tmp/ci_memory --memory-enabled

  # Without memory (baseline / ablation)
  mini-swe-agent cibench --dataset ci_data.jsonl --output results_no_mem/ \\
    --no-memory-enabled

  # Custom model
  mini-swe-agent cibench --dataset ci_data.jsonl --output results/ \\
    -m anthropic/claude-opus-4

  # Filter / slice
  mini-swe-agent cibench --dataset ci_data.jsonl --output results/ \\
    --filter "^owner-repo" --slice 0:10
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from minisweagent.run.benchmarks.utils.ci_context import build_ci_context, save_memory_after_patch
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.project_env import load_project_env
from minisweagent.utils.serialize import UNSET, recursive_merge

# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "cibench.yaml"
_OUTPUT_FILE_LOCK   = threading.Lock()
app = typer.Typer(rich_markup_mode="rich", add_completion=False)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPO_CACHE_ROOT = Path(os.getenv("MSWEA_REPO_CACHE_ROOT") or (PROJECT_ROOT / "repo")).resolve()

def _make_context_llm(config: Dict[str, Any]) -> Any:
    """
    Build a plain callable (prompt: str) -> str for Phase A and Phase C
    using the SAME model already configured for the repair agent.

    Reuses OpenRouterModel directly — no LiteLLM, no separate credentials.
    The model reads OPENROUTER_API_KEY / OPENROUTER_BASE_URL from env,
    exactly as the repair agent does.
    """
    try:
        model = get_model(config=config.get("model", {}))

        def _call(prompt: str) -> str:
            try:
                # query() expects a messages list; we pass a single user turn.
                # We bypass the bash-tool parsing and just read raw content.
                import json as _json
                api_key  = os.getenv("OPENROUTER_API_KEY", "")
                api_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                model_name = config.get("model", {}).get("model_name", "minimax/minimax-m2.5")

                import requests as _req
                payload = {
                    "model":    model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                }
                resp = _req.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type":  "application/json",
                    },
                    data=_json.dumps(payload),
                    timeout=120,
                )
                resp.raise_for_status()
                return (resp.json()["choices"][0]["message"]["content"] or "").strip()
            except Exception as exc:
                logger.warning("[CIBench] context LLM call failed: %s", exc)
                return ""

        return _call

    except Exception as exc:
        logger.warning("[CIBench] Could not build context LLM: %s", exc)
        return None


_HELP_TEXT = """
Run mini-SWE-agent on CI-failure repair instances.

Each instance is run locally: the repo is cloned from GitHub, the failing
commit is checked out, and the agent edits the code in place.

Input: a JSONL dataset (one JSON object per line) or a HuggingFace dataset path.
Output: per-instance patch in preds.json, format {id, sha_fail, diff}.

[bold green]With memory:[/bold green]  --memory-enabled --memory-root /path/to/store
[bold yellow]Without memory:[/bold yellow] --no-memory-enabled  (baseline)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

_seed_log_index: Optional[Dict[str, Dict[str, Any]]] = None


def _load_seed_log_index() -> Dict[str, Dict[str, Any]]:
    """
    Load data/trs/seed_log_details.json and index by sha_fail and id.
    Contains pre-computed CILogAnalyzer output (error_context, error_types,
    relevant_files, failed_job) nested under an 'analysis' key.
    Cached after first load.
    """
    global _seed_log_index
    if _seed_log_index is not None:
        return _seed_log_index

    seed_path = PROJECT_ROOT / "data" / "trs" / "seed_log_details.json"
    if not seed_path.exists():
        _seed_log_index = {}
        return {}

    try:
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else list(data.values())
        index: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            sha = str(rec.get("sha_fail") or "")
            iid = str(rec.get("id") or rec.get("instance_id") or "")
            if sha:
                index[sha] = rec
            if iid:
                index[iid] = rec
        _seed_log_index = index
        logger.info("[CIBench] Loaded seed_log_details index: %d records", len(index))
    except Exception as exc:
        logger.warning("[CIBench] Could not load seed_log_details.json: %s", exc)
        _seed_log_index = {}

    return _seed_log_index


def _inject_precomputed_analysis(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Look up the instance in seed_log_details.json and inject the pre-computed
    CILogAnalyzer output as top-level fields so Phase A is skipped entirely.

    seed_log_details stores analysis under a nested 'analysis' key:
      {
        "sha_fail": "...",
        "analysis": {
          "error_context":  [str]   → overall_failure_reasons
          "error_types":    [dict]  → overall_error_types (already set)
          "relevant_files": [dict]  → effected_files
          "failed_job":     [dict]  → failed_jobs
        }
      }

    Mapping to top-level fields that _has_precomputed_analysis checks for:
        analysis.error_context  → overall_failure_reasons
        analysis.relevant_files → effected_files
        analysis.failed_job     → failed_jobs
        analysis.error_types    → error_types (for Phase A query building)
    """
    index = _load_seed_log_index()
    if not index:
        return inst

    sha = str(inst.get("sha_fail") or "")
    iid = str(inst.get("instance_id") or inst.get("id") or "")
    seed_rec = index.get(sha) or index.get(iid)

    if not seed_rec:
        return inst

    analysis = seed_rec.get("analysis") or {}
    if not analysis:
        return inst

    inst = dict(inst)

    # Map nested analysis fields to the top-level fields the pipeline expects
    if not inst.get("overall_failure_reasons") and analysis.get("error_context"):
        inst["overall_failure_reasons"] = [
            str(x) for x in analysis["error_context"] if str(x).strip()
        ]

    if not inst.get("effected_files") and analysis.get("relevant_files"):
        inst["effected_files"] = analysis["relevant_files"]

    if not inst.get("failed_jobs") and analysis.get("failed_job"):
        inst["failed_jobs"] = analysis["failed_job"]

    if not inst.get("error_types") and analysis.get("error_types"):
        inst["error_types"] = analysis["error_types"]

    logger.debug(
        "[CIBench] Injected pre-computed analysis for sha=%s: "
        "reasons=%d  files=%d  jobs=%d",
        sha[:12],
        len(inst.get("overall_failure_reasons") or []),
        len(inst.get("effected_files") or []),
        len(inst.get("failed_jobs") or []),
    )
    return inst


def _has_local_precomputed_analysis(inst: Dict[str, Any]) -> bool:
    return bool(
        inst.get("overall_failure_reasons")
        and inst.get("effected_files")
        and inst.get("failed_jobs")
    )


def _prepare_local_instance(inst: Dict[str, Any], *, allow_hf_enrichment: bool) -> Dict[str, Any]:
    normalized = _normalize_instance(inst)
    injected = _inject_precomputed_analysis(normalized)
    if _has_local_precomputed_analysis(injected) or not allow_hf_enrichment:
        return injected
    enriched = _enrich_from_hf(injected)
    return _inject_precomputed_analysis(enriched)


def _normalize_instance(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize field names so both eval_dataset.jsonl and eval_issues.json
    work without any changes to the rest of the pipeline.

    Field aliases handled:
        id               → instance_id        (eval_issues.json uses "id")
        workflow_filename → workflow_path      (if workflow_path is missing)
        error_type list  → overall_error_types (eval_issues has a list, not a string)
    """
    if not isinstance(inst, dict):
        return inst
    inst = dict(inst)

    # Prefer "id" over "instance_id" for benchmark consistency
    # (Benchmarks use numeric/string IDs like "102", "121", not repo@sha format)
    if "id" in inst:
        inst["instance_id"] = str(inst["id"])
    elif "instance_id" not in inst:
        # Fallback: generate from repo@sha if neither exists
        repo = f"{inst.get('repo_owner', '')}/{inst.get('repo_name', '')}".strip('/')
        sha = str(inst.get('sha_fail', ''))[:12]
        inst["instance_id"] = f"{repo}@{sha}" if repo and sha else "unknown"

    # workflow_filename → workflow_path
    if "workflow_path" not in inst and "workflow_filename" in inst:
        wf = str(inst["workflow_filename"])
        inst["workflow_path"] = f".github/workflows/{wf}" if not wf.startswith(".github") else wf

    # error_type as list → overall_error_types
    # eval_issues.json stores ["Dependency Issues", "Syntax Error", ...]
    # The pipeline uses overall_error_types (list) and error_type (str)
    et = inst.get("error_type")
    if isinstance(et, list):
        inst["overall_error_types"] = et
        inst["error_type"] = et[0] if et else ""
    elif isinstance(et, str) and et:
        inst.setdefault("overall_error_types", [et])

    return inst


_HF_DATASET_NAME = "ci-benchmark-user/ci-repair-bench"
_hf_index: Optional[Dict[str, Dict[str, Any]]] = None   # module-level cache


def _load_hf_index() -> Dict[str, Dict[str, Any]]:
    """
    Load the full HuggingFace dataset once and index it by sha_fail and id.
    Cached after first load so subsequent calls are instant.
    """
    global _hf_index
    if _hf_index is not None:
        return _hf_index

    try:
        from datasets import load_dataset  # type: ignore
        logger.info("[CIBench] Loading HuggingFace dataset %s for instance enrichment...", _HF_DATASET_NAME)
        # Suppress noisy HuggingFace "Repo card metadata block was not found" warnings
        import logging as _logging
        _hf_logger = _logging.getLogger("datasets")
        _prev_level = _hf_logger.level
        _hf_logger.setLevel(_logging.ERROR)

        # Try common split names — dataset uses 'train' as the only split
        try:
            for split_name in ("test", "train", "validation", "all"):
                try:
                    ds = load_dataset(_HF_DATASET_NAME, split=split_name)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"No usable split found in {_HF_DATASET_NAME}")
        finally:
            _hf_logger.setLevel(_prev_level)
        index: Dict[str, Dict[str, Any]] = {}
        for row in ds:
            row = dict(row)
            sha = str(row.get("sha_fail") or "")
            iid = str(row.get("instance_id") or row.get("id") or "")
            if sha:
                index[sha] = row       # match by sha_fail
            if iid:
                index[iid] = row       # match by id / instance_id
            # Also index by numeric string of id (eval_issues stores id as string)
            raw_id = row.get("id")
            if raw_id is not None:
                index[str(int(raw_id))] = row
        _hf_index = index
        logger.info("[CIBench] HuggingFace index built: %d records", len(index))
        return index
    except Exception as exc:
        logger.warning("[CIBench] Could not load HuggingFace dataset (%s) — using local data only", exc)
        _hf_index = {}
        return {}


def _enrich_from_hf(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Look up the instance in the HuggingFace dataset by sha_fail or id.
    If found, merge HF record (base) with local record (overlay) so that
    any pre-computed analysis fields from HF are used, but local overrides
    (like error_type list) are preserved.
    Returns the enriched instance.
    """
    index = _load_hf_index()
    if not index:
        return inst

    sha = str(inst.get("sha_fail") or "")
    iid = str(inst.get("instance_id") or inst.get("id") or "")

    hf_row = index.get(sha) or index.get(iid)
    if not hf_row:
        logger.debug("[CIBench] No HF match for sha=%s id=%s — using local data", sha[:12], iid)
        return inst

    # Merge: HF record is the base (has pre-computed fields like overall_failure_reasons),
    # local inst overlays its own fields (logs, error_type list, etc.)
    merged = {**hf_row, **inst}
    return _normalize_instance(merged)


def load_ci_instances(dataset_path: str, split: str = "test") -> List[Dict[str, Any]]:
    """
    Load CI benchmark instances.

    Supports three sources:
        .json   — JSON array (eval_issues.json) — IDs are looked up in HuggingFace
                  to enrich with pre-computed analysis fields
        .jsonl  — one JSON object per line (eval_dataset.jsonl)
        str     — HuggingFace dataset name (loaded directly)

    For .json files: each instance is enriched by fetching the matching
    full record from HuggingFace by sha_fail/id, so the pipeline always
    gets complete data regardless of what the local file contains.
    """
    p = Path(dataset_path)

    if p.exists():
        raw_text = p.read_text(encoding="utf-8").strip()

        # JSON array — enrich each instance from HuggingFace
        if raw_text.startswith("["):
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse JSON array in {dataset_path}: {e}") from e
            instances = []
            for x in data:
                if not isinstance(x, dict):
                    continue
                instances.append(_prepare_local_instance(x, allow_hf_enrichment=True))
            logger.info("Loaded %d instances from %s (enriched from HuggingFace)", len(instances), dataset_path)
            return instances

        # JSONL — one object per line.  Filtered eval files may contain only
        # the compact local issue rows, so normalize, enrich, and inject the
        # same precomputed analysis used for JSON arrays.
        instances_: List[Dict[str, Any]] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    for x in obj:
                        if not isinstance(x, dict):
                            continue
                        instances_.append(_prepare_local_instance(x, allow_hf_enrichment=True))
                elif isinstance(obj, dict):
                    instances_.append(_prepare_local_instance(obj, allow_hf_enrichment=True))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line in %s: %s", dataset_path, e)
        logger.info("Loaded %d instances from %s (enriched from HuggingFace/local cache)", len(instances_), dataset_path)
        return instances_

    # HuggingFace dataset name passed directly
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(dataset_path, split=split)
        instances_ = [_normalize_instance(dict(x)) for x in ds]  # type: ignore
        logger.info("Loaded %d instances from HuggingFace dataset %s", len(instances_), dataset_path)
        return instances_
    except Exception as e:
        raise ValueError(
            f"Could not load dataset from '{dataset_path}'. "
            f"Expected a .json / .jsonl file path or a HuggingFace dataset name. Error: {e}"
        ) from e


def filter_instances(
    instances: List[Dict[str, Any]],
    *,
    filter_spec: str = "",
    slice_spec: str  = "",
    shuffle:    bool = False,
) -> List[Dict[str, Any]]:
    import random

    if shuffle:
        instances = sorted(instances, key=lambda x: str(x.get("instance_id", "")))
        random.seed(42)
        random.shuffle(instances)

    if filter_spec:
        before = len(instances)
        instances = [
            inst for inst in instances
            if re.match(filter_spec, str(inst.get("instance_id", "")))
        ]
        logger.info("Filter '%s': %d → %d instances", filter_spec, before, len(instances))

    if slice_spec:
        before = len(instances)
        parts  = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*parts)]
        logger.info("Slice '%s': %d → %d instances", slice_spec, before, len(instances))

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Predictions file helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_preds(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def update_preds_file(
    output_path: Path,
    instance_id: str,
    sha_fail: str,
    diff: str,
) -> None:
    """Write / update the prediction for one instance (thread-safe)."""
    with _OUTPUT_FILE_LOCK:
        data = _read_preds(output_path)
        data[instance_id] = {
            "id":       instance_id,
            "sha_fail": sha_fail,
            "diff":     diff or "",
        }
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def remove_from_preds_file(output_path: Path, instance_id: str) -> None:
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        data = _read_preds(output_path)
        if instance_id in data:
            del data[instance_id]
            output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Local environment setup  (clone → checkout → LocalEnvironment)
# ─────────────────────────────────────────────────────────────────────────────

def setup_local_environment(
    config: Dict[str, Any],
    instance: Dict[str, Any],
    instance_dir: Path,
) -> Tuple[LocalEnvironment, Path]:
    """
    Prepare a local repository checkout and return *(env, testbed_path)*.

    • Keeps a shared network clone cache at ``<project_root>/repo/``.
    • Creates ``<instance_dir>/testbed/`` from that cache when needed.
    • Checks out the failing commit (``sha_fail``).
    • Wraps the directory in a ``LocalEnvironment`` using settings from the
      ``environment`` section of the config (timeout, interpreter, env vars).

    If the testbed directory already contains a ``.git`` folder (re-run),
    the local working copy is reused so previous work is preserved.
    """
    repo_owner = instance.get("repo_owner", "")
    repo_name  = instance.get("repo_name", "")
    sha_fail   = str(instance.get("sha_fail") or "")

    testbed_path = instance_dir / "testbed"
    cache_dir_name = f"{repo_owner}__{repo_name}"
    repo_cache_path = REPO_CACHE_ROOT / cache_dir_name
    clone_url = f"https://github.com/{repo_owner}/{repo_name}.git"

    def _run_git(args: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _ensure_commit_available(repo_path: Path, commit: str, remote: str = "origin") -> None:
        if not commit:
            return
        verify = _run_git(["rev-parse", "--verify", f"{commit}^{{commit}}"], repo_path)
        if verify.returncode == 0:
            return
        fetch = _run_git(["fetch", "--quiet", remote, commit], repo_path)
        if fetch.returncode != 0:
            raise RuntimeError(
                f"git fetch failed for commit {commit} in {repo_path}:\n"
                f"{fetch.stderr[:800]}"
            )
        verify = _run_git(["rev-parse", "--verify", f"{commit}^{{commit}}"], repo_path)
        if verify.returncode != 0:
            raise RuntimeError(
                f"commit {commit} still unavailable in {repo_path} after fetch:\n"
                f"{verify.stderr[:800]}"
            )

    def _must_run_git(args: list[str], cwd: Path, error_label: str, timeout: int = 300) -> None:
        result = _run_git(args, cwd, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"{error_label} in {cwd}:\n{result.stderr[:800] or result.stdout[:800]}")

    def _prepare_worktree(repo_path: Path, commit: str) -> None:
        if not commit:
            raise RuntimeError(f"Missing sha_fail for worktree preparation in {repo_path}")
        _must_run_git(
            ["config", "user.email", "ci-repair@mini-swe-agent"],
            repo_path,
            "git config user.email failed",
        )
        _must_run_git(
            ["config", "user.name", "mini-swe-agent"],
            repo_path,
            "git config user.name failed",
        )
        _must_run_git(
            ["checkout", "--detach", "--force", commit],
            repo_path,
            f"git checkout {commit} failed",
        )
        _must_run_git(
            ["reset", "--hard", commit],
            repo_path,
            f"git reset --hard {commit} failed",
        )
        _must_run_git(
            ["clean", "-fdx"],
            repo_path,
            "git clean -fdx failed",
        )

    # ── Shared repo cache ─────────────────────────────────────────────────────
    REPO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not (repo_cache_path / ".git").exists():
        logger.info("[CIBench] Cloning %s → shared cache %s", clone_url, repo_cache_path)
        result = subprocess.run(
            ["git", "clone", "--quiet", clone_url, str(repo_cache_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed for {repo_owner}/{repo_name}:\n"
                f"{result.stderr[:800]}"
            )
    else:
        logger.info("[CIBench] Reusing shared cache at %s", repo_cache_path)
        fetch_result = subprocess.run(
            ["git", "-C", str(repo_cache_path), "fetch", "--all", "--tags", "--prune", "--quiet"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if fetch_result.returncode != 0:
            logger.warning(
                "[CIBench] git fetch failed for shared cache %s:\n%s",
                repo_cache_path,
                fetch_result.stderr[:800],
            )
    _ensure_commit_available(repo_cache_path, sha_fail, remote="origin")

    # ── Per-instance working copy ─────────────────────────────────────────────
    if not (testbed_path / ".git").exists():
        testbed_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("[CIBench] Creating local working copy %s → %s", repo_cache_path, testbed_path)
        result = subprocess.run(
            ["git", "clone", "--quiet", "--shared", str(repo_cache_path), str(testbed_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"local git clone failed for {repo_owner}/{repo_name} from cache:\n"
                f"{result.stderr[:800]}"
            )
    else:
        logger.info("[CIBench] Reusing existing working copy at %s", testbed_path)
    _ensure_commit_available(testbed_path, sha_fail, remote="origin")
    _prepare_worktree(testbed_path, sha_fail)

    # ── Build LocalEnvironment ────────────────────────────────────────────────
    # Strip environment_class from the env config dict (LocalEnvironment doesn't accept it).
    env_cfg: Dict[str, Any] = {
        k: v for k, v in config.get("environment", {}).items()
        if k != "environment_class"
    }
    env_cfg["cwd"] = str(testbed_path)
    env_cfg.setdefault("timeout", 120)

    env = LocalEnvironment(**env_cfg)

    # ── Git setup + checkout sha_fail ─────────────────────────────────────────
    startup_tpl = config.get("run", {}).get("startup_command", "")
    if startup_tpl:
        try:
            rendered = Template(startup_tpl, undefined=StrictUndefined).render(**instance)
        except Exception:
            # Fallback: manual substitution if instance has unexpected keys
            rendered = startup_tpl.replace("{{sha_fail}}", sha_fail)
        out = env.execute({"command": rendered})
        if out.get("returncode", 0) != 0:
            raise RuntimeError(
                f"startup command failed for {instance.get('instance_id')}:\n"
                f"{out.get('output', '')[:800]}"
            )

    return env, testbed_path


# ─────────────────────────────────────────────────────────────────────────────
# Per-instance processing
# ─────────────────────────────────────────────────────────────────────────────

def process_instance(
    instance:         Dict[str, Any],
    output_dir:       Path,
    config:           Dict[str, Any],
    progress_manager: RunBatchProgressManager,
    *,
    memory_root:             Optional[str],
    memory_enabled:          bool,
    memory_top_k:            int,
    memory_ablation_levels:  str,
    memory_plugin_path:      Optional[str],
    context_model:           str,
    save_memory:             bool,
) -> None:
    """
    Full pipeline for one CI instance:
      pre-process logs → clone repo → checkout sha_fail → run agent → save diff
    """
    instance_id = str(instance.get("instance_id") or instance.get("id") or "unknown")
    sha_fail    = str(instance.get("sha_fail") or "")
    dir_name    = sha_fail or instance_id   # prefer sha_fail as the directory key
    instance_dir = output_dir / dir_name
    instance_dir.mkdir(parents=True, exist_ok=True)

    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{dir_name}.traj.json").unlink(missing_ok=True)

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pre-processing CI logs")

    agent        = None
    exit_status: Optional[str] = None
    diff         = ""
    ci_ctx       = {}
    ci_memory: Dict[str, Any] = {}   # full memory retrieval result — saved to trajectory
    extra_info: Dict[str, Any] = {}

    # ── Phase 1: build enriched CI problem statement ──────────────────────────
    try:
        context_llm = _make_context_llm(config)
        ci_result = build_ci_context(
            instance,
            memory_root=memory_root,
            memory_enabled=memory_enabled,
            memory_top_k=memory_top_k,
            memory_ablation_levels=memory_ablation_levels,
            memory_plugin_path=memory_plugin_path,
            model=context_model,
            llm=context_llm,
        )
        ci_ctx    = ci_result["context"]
        task      = ci_result["problem_statement"]
        ci_memory = ci_result["memory"]
        extra_info["memory_summary"] = {
            "enabled":             memory_enabled,
            "weighted_similarity": ci_memory.get("weighted_similarity", 0.0),
            "levels_retrieved":    ci_memory.get("selected_memory_levels", []),
        }
        # Write per-instance retrieval diagnostics for manual inspection
        _save_retrieval_diagnostic(output_dir / "memory_retrieval_debug.jsonl", instance_id, ci_ctx, ci_memory)
    except Exception as exc:
        logger.error("[CIBench] CI pre-processing failed for %s: %s", instance_id, exc)
        raw_log = instance.get("logs") or instance.get("log") or ""
        if isinstance(raw_log, list):
            raw_log = "\n".join(str(x) for x in raw_log)
        task = (
            f"# CI Failure\n\n"
            f"Repo: {instance.get('repo_owner','')}/{instance.get('repo_name','')}\n"
            f"sha_fail: {sha_fail}\n\n"
            f"## Logs\n{str(raw_log)[:3000]}"
        )
        extra_info["ci_preprocess_error"] = str(exc)

    # ── Phase 2: clone repo + checkout sha_fail ───────────────────────────────
    progress_manager.update_instance_status(instance_id, "Cloning repository")

    try:
        env, testbed_path = setup_local_environment(config, instance, instance_dir)

        model_ = get_model(config=config.get("model", {}))
        agent = ProgressTrackingAgent(
            model_,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        # Inject the real checkout path so {{testbed_path}} resolves in templates
        agent.extra_template_vars["testbed_path"] = str(testbed_path)

        # ── Phase 3: run agent ────────────────────────────────────────────────
        progress_manager.update_instance_status(instance_id, "Running agent")
        info = agent.run(task)
        exit_status = info.get("exit_status")

        # Agent submits via:
        #   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat <testbed_path>/patch.txt
        raw_submission = info.get("submission") or ""
        diff = _extract_diff(raw_submission)

        # ── Phase 4: save memory record ───────────────────────────────────────
        if save_memory and diff and ci_ctx and memory_root:
            try:
                save_memory_after_patch(
                    instance,
                    ci_ctx,
                    diff,
                    memory_root=memory_root,
                    memory_plugin_path=memory_plugin_path,
                )
            except Exception as exc:
                logger.warning("[CIBench] save_memory_after_patch failed for %s: %s", instance_id, exc)

    except Exception as exc:
        logger.error("[CIBench] Agent run failed for %s: %s", instance_id, exc, exc_info=True)
        exit_status = type(exc).__name__
        diff = ""
        extra_info.update({"traceback": traceback.format_exc(), "exception_str": str(exc)})

    finally:
        if agent is not None:
            traj_path = instance_dir / f"{dir_name}.traj.json"
            llm_sel   = ci_memory.get("llm_selection") or {}
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission":  diff,
                        "sha_fail":    sha_fail,
                        **extra_info,
                    },
                    "instance_id": instance_id,

                    # ── Phase A + B: Structured CI failure context ────────
                    "ci_context": {
                        "overall_failure_reasons": ci_ctx.get("overall_failure_reasons", []),
                        "overall_error_types":     ci_ctx.get("overall_error_types", []),
                        "effected_files":          ci_ctx.get("effected_files", []),
                        "failed_jobs":             ci_ctx.get("failed_jobs", []),
                        "workflow_profile":        ci_ctx.get("workflow_profile", {}),
                    } if ci_ctx else {},

                    # ── Phase C: Full memory retrieval + two-LLM analysis ─
                    "memory_retrieval": _build_memory_traj(ci_memory, task),
                },
            )
            logger.info("[CIBench] Saved trajectory to '%s'", traj_path)

        update_preds_file(output_dir / "preds.json", instance_id, sha_fail, diff)
        progress_manager.on_instance_end(instance_id, exit_status)


def _build_memory_traj(ci_memory: Dict[str, Any], task: str) -> Dict[str, Any]:
    """
    Build the memory_retrieval block saved to every trajectory.

    Captures the full two-LLM pipeline result so every run can be
    analyzed without re-running:

      cosine_search   — what L1/L2/L3 returned (scores, files, patterns)
      llm1_filter     — which candidates LLM 1 selected as relevant + why
      llm2_guidance   — the full repair document LLM 2 synthesized
      injected        — exact text injected into the agent's problem statement
    """
    if not ci_memory:
        return {}

    llm_sel   = ci_memory.get("llm_selection") or {}
    threshold = float((ci_memory.get("thresholds") or {}).get("similarity_threshold") or 0.0)
    weighted  = float(ci_memory.get("weighted_similarity") or 0.0)

    return {
        # ── Retrieval metadata ────────────────────────────────────────────
        "enabled":            bool(ci_memory.get("enabled", False)),
        "weighted_similarity": round(weighted, 4),
        "level_scores":       {
            k: round(float(v), 4)
            for k, v in (ci_memory.get("level_scores") or {}).items()
        },
        "threshold":          round(threshold, 4),
        "above_threshold":    weighted >= threshold,
        "counts": {
            "L1": len(ci_memory.get("l1_matches") or []),
            "L2": len(ci_memory.get("l2_matches") or []),
            "L3": len(ci_memory.get("l3_matches") or []),
        },

        # ── Step 3-4: Raw cosine search results ───────────────────────────
        # All matches stored in full — no truncation — for manual analysis
        "cosine_search": {
            "L1": _slim_matches(ci_memory.get("l1_matches") or []),
            "L2": _slim_matches(ci_memory.get("l2_matches") or []),
            "L3": _slim_matches(ci_memory.get("l3_matches") or []),
        },

        # ── Step 5a: LLM 1 — Relevance Filter output ─────────────────────
        # Which candidates did LLM 1 select as relevant and why
        "llm1_filter": {
            "use_memory":          bool(llm_sel.get("use_memory", False)),
            "n_relevant":          len(llm_sel.get("relevant_candidates") or []),
            "relevant_candidates": llm_sel.get("relevant_candidates") or [],
            # easy to check: did LLM 1 fire? how many candidates passed?
        },

        # ── Step 5b: LLM 2 — Experience Synthesizer output ───────────────
        # The full repair guidance document produced from relevant candidates
        "llm2_guidance": llm_sel.get("guidance_document") or {},

        # ── What went into the agent's context ────────────────────────────
        # The exact ## Memory Context section injected into problem_statement
        "injected_memory_block": _extract_memory_block(task),

        # ── Quick-read summary ────────────────────────────────────────────
        "analysis_summary": str(llm_sel.get("analysis_summary") or ""),
    }


def _slim_matches(matches: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    """Keep only the fields needed for analysis — avoids bloating the trajectory."""
    out = []
    for row in matches[:n]:
        out.append({
            "score":           round(float(row.get("similarity_score") or 0.0), 4),
            "file":            row.get("file", ""),
            "error_type":      row.get("error_type", ""),
            "failure_pattern": row.get("failure_pattern") or row.get("issue_type", ""),
            "failure_reason":  str(row.get("failure_reason") or row.get("overall_failure_reason") or "")[:300],
            "fix_direction":   str(row.get("fix_direction") or row.get("fix_strategy") or "")[:300],
            "matched_on":      row.get("matched_on") or {},
            "repo":            row.get("repo", ""),
            "sha_fail":        str(row.get("sha_fail") or "")[:12],
        })
    return out


def _extract_memory_block(problem_statement: str) -> str:
    """Extract just the ## Memory Context section from the assembled problem statement."""
    if not problem_statement:
        return ""
    marker = "## Memory Context"
    idx = problem_statement.find(marker)
    if idx == -1:
        return ""
    return problem_statement[idx:].strip()


def _save_retrieval_diagnostic(
    log_path: Path,
    instance_id: str,
    context: Dict[str, Any],
    memory: Dict[str, Any],
) -> None:
    """
    Append one line to memory_retrieval_debug.jsonl with everything needed
    to manually verify whether memory retrieval is working correctly.

    Fields saved:
      instance_id, repo, sha_fail, error_type, failure_pattern
      memory_enabled, weighted_similarity, level_scores
      counts per level (how many matches found)
      above_threshold (was weighted_sim >= threshold?)
      llm_used_memory, llm_analysis_summary
      selected_items  (what the LLM gate kept — key_insight + fix_direction)
      top_matches     (raw top-3 records per level for manual similarity check)
    """
    try:
        llm_sel   = memory.get("llm_selection") or {}
        query     = memory.get("query") or {}
        scores    = memory.get("level_scores") or {"L1": 0.0, "L2": 0.0, "L3": 0.0}
        threshold = (memory.get("thresholds") or {}).get("similarity_threshold", 0.0)
        weighted  = float(memory.get("weighted_similarity") or 0.0)

        # Counts per level
        counts = {
            "L1": len(memory.get("l1_matches") or []),
            "L2": len(memory.get("l2_matches") or []),
            "L3": len(memory.get("l3_matches") or []),
        }

        # Top-3 raw matches per level — enough to manually judge similarity
        def _top_matches(level_key: str, n: int = 3) -> List[Dict[str, Any]]:
            out = []
            for row in (memory.get(level_key) or [])[:n]:
                out.append({
                    "level":           row.get("memory_level", level_key.split("_")[0].upper()),
                    "score":           round(float(row.get("similarity_score") or 0.0), 4),
                    "file":            row.get("file", ""),
                    "error_type":      row.get("error_type", ""),
                    "failure_pattern": row.get("failure_pattern") or row.get("issue_type", ""),
                    "failure_reason":  str(row.get("failure_reason") or row.get("overall_failure_reason") or "")[:200],
                    "fix_direction":   str(row.get("fix_direction") or row.get("fix_strategy") or "")[:200],
                    "matched_on":      row.get("matched_on") or {},
                })
            return out

        top_matches = (
            _top_matches("l1_matches")
            + _top_matches("l2_matches")
            + _top_matches("l3_matches")
        )

        # LLM 1 — relevance filter result
        relevant_candidates = llm_sel.get("relevant_candidates") or []

        # LLM 2 — guidance document (summarised for the debug log)
        guidance = llm_sel.get("guidance_document") or {}

        record = {
            # ── Instance identity ─────────────────────────────────────────
            "instance_id":     instance_id,
            "repo":            context.get("repo", ""),
            "sha_fail":        str(context.get("sha_fail", ""))[:12],
            # ── Current failure ───────────────────────────────────────────
            "error_type":      (
                query.get("error_type")
                or (context.get("overall_error_types") or [""])[0]
            ),
            "failure_pattern": query.get("failure_pattern", ""),
            "failure_reason":  str(query.get("overall_failure_reason") or ""),
            # ── Cosine retrieval outcome ──────────────────────────────────
            "memory_enabled":  bool(memory.get("enabled", False)),
            "level_scores":    {k: round(float(v), 4) for k, v in scores.items()},
            "weighted_sim":    round(weighted, 4),
            "threshold":       round(float(threshold), 4),
            "above_threshold": weighted >= threshold if threshold > 0 else False,
            "counts":          counts,
            "top_matches":     top_matches,           # raw candidates for manual check
            # ── LLM 1: Relevance Filter ───────────────────────────────────
            "llm1": {
                "used_memory":         bool(llm_sel.get("use_memory", False)),
                "n_relevant":          len(relevant_candidates),
                "relevant_candidates": relevant_candidates,
                # each item: {index, memory_level, similarity_score, relevance, why_relevant}
            },
            # ── LLM 2: Experience Synthesizer ─────────────────────────────
            "llm2": {
                "produced_guidance": bool(guidance),
                "confidence":        guidance.get("confidence", ""),
                "confidence_reason": guidance.get("confidence_reason", ""),
                "diagnosis":         guidance.get("diagnosis", ""),
                "full_scope":        guidance.get("full_scope", {}),
                "linked_issues":     guidance.get("linked_issues", []),
                "fix_approach":      guidance.get("fix_approach", []),
                "post_fix_patterns": guidance.get("post_fix_patterns", []),
                "verification":      guidance.get("verification", {}),
                "summary":           guidance.get("summary", ""),
                # full document also stored for completeness
                "_full_guidance_document": guidance,
            },
            # ── What was injected into agent context ──────────────────────
            "analysis_summary": str(llm_sel.get("analysis_summary") or ""),
        }

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _OUTPUT_FILE_LOCK:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as exc:
        logger.warning("[CIBench] Failed to write retrieval diagnostic for %s: %s", instance_id, exc)


def _extract_diff(submission: str) -> str:
    """
    Pull the git diff out of the agent's final submission string.

    The agent echoes ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` then cats the
    patch file.  We take everything after that sentinel (or the full string
    if the sentinel is absent).

    NOTE: We deliberately avoid str.strip() here because it removes trailing
    blank context lines (" \\n") from the diff, causing git to report
    "corrupt patch" (hunk header claims N lines but body only has N-1).
    Instead we lstrip leading whitespace and remove only truly empty trailing
    lines so that blank context lines like " " are preserved.
    """
    sentinel = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    if sentinel in submission:
        diff = submission[submission.index(sentinel) + len(sentinel):].lstrip()
    else:
        diff = submission.lstrip()
    # Strip only completely empty trailing lines, not blank context lines (" ")
    parts = diff.split("\n")
    while parts and parts[-1] == "":
        parts.pop()
    return ("\n".join(parts) + "\n") if parts else ""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    dataset: str = typer.Option(
        ..., "--dataset", "-d",
        help="Path to JSONL dataset or HuggingFace dataset name",
        rich_help_panel="Data selection",
    ),
    split: str = typer.Option(
        "test", "--split",
        help="Dataset split (for HuggingFace datasets)",
        rich_help_panel="Data selection",
    ),
    output: str = typer.Option(
        "", "-o", "--output",
        help="Output directory for predictions and trajectories",
        rich_help_panel="Basic",
    ),
    workers: int = typer.Option(
        1, "-w", "--workers",
        help="Parallel worker threads (each clones to its own instance_dir)",
        rich_help_panel="Basic",
    ),
    model_name: Optional[str] = typer.Option(
        None, "-m", "--model",
        help="LLM model to use for the repair agent",
        rich_help_panel="Basic",
    ),
    model_class: Optional[str] = typer.Option(
        None, "--model-class",
        help="Model class (e.g. 'anthropic' or full import path)",
        rich_help_panel="Advanced",
    ),
    config_spec: List[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)], "-c", "--config",
        help="Config file(s) or key=value overrides (merged left to right)",
        rich_help_panel="Basic",
    ),
    filter_spec: str = typer.Option(
        "", "--filter",
        help="Regex filter on instance_id",
        rich_help_panel="Data selection",
    ),
    slice_spec: str = typer.Option(
        "", "--slice",
        help="Slice (e.g. '0:10' for first 10 instances)",
        rich_help_panel="Data selection",
    ),
    shuffle: bool = typer.Option(
        False, "--shuffle",
        help="Shuffle instances before slicing/filtering",
        rich_help_panel="Data selection",
    ),
    redo_existing: bool = typer.Option(
        False, "--redo-existing",
        help="Re-run instances already present in preds.json",
        rich_help_panel="Data selection",
    ),
    # ── Memory options ────────────────────────────────────────────────────────
    memory_enabled: bool = typer.Option(
        False, "--memory-enabled/--no-memory-enabled",
        help=(
            "Enable hierarchical memory (L1/L2/L3) retrieval. "
            "Requires --memory-root."
        ),
        rich_help_panel="Memory",
    ),
    memory_root: Optional[str] = typer.Option(
        None, "--memory-root",
        help="Directory to persist L1/L2/L3 memory JSON files",
        rich_help_panel="Memory",
    ),
    memory_top_k: int = typer.Option(
        3, "--memory-top-k",
        help="Top-k records to retrieve per memory level",
        rich_help_panel="Memory",
    ),
    memory_ablation_levels: str = typer.Option(
        "L1+L2+L3", "--memory-ablation",
        help="Which memory levels to use: 'L1', 'L1+L2', or 'L1+L2+L3'",
        rich_help_panel="Memory",
    ),
    memory_plugin_path: Optional[str] = typer.Option(
        None, "--memory-plugin-path",
        help="Explicit path to memory_plugin.py if not on PYTHONPATH",
        rich_help_panel="Memory",
    ),
    save_memory: bool = typer.Option(
        True, "--save-memory/--no-save-memory",
        help="Save memory entries after successful patches",
        rich_help_panel="Memory",
    ),
    # ── Context / log analysis options ───────────────────────────────────────
    context_model: str = typer.Option(
        "gpt-4o-mini", "--context-model",
        help="LLM model for CILogAnalyzer (log parsing + workflow analysis)",
        rich_help_panel="Advanced",
    ),
) -> None:
    # fmt: on
    """Run mini-SWE-agent on CI failure instances (batch mode, local environment)."""
    project_env_path = load_project_env()
    if project_env_path:
        logger.info("[CIBench] Loaded project env from %s", project_env_path)

    # ── Output directory ──────────────────────────────────────────────────────
    if not output:
        output = f"ci_repair_results_{int(time.time())}"
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("[CIBench] Results will be saved to %s", output_path)
    add_file_handler(output_path / "cibench.log")

    # ── Memory root ───────────────────────────────────────────────────────────
    if memory_enabled and not memory_root:
        memory_root = str(output_path / "ci_memory")
        logger.info("[CIBench] memory_root not set; using %s", memory_root)
    if memory_root:
        Path(memory_root).mkdir(parents=True, exist_ok=True)
        logger.info("[CIBench] Memory storage: %s  levels=%s", memory_root, memory_ablation_levels)

    # ── Load dataset ──────────────────────────────────────────────────────────
    instances = load_ci_instances(dataset, split=split)
    instances = filter_instances(
        instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle
    )

    if not redo_existing and (output_path / "preds.json").exists():
        existing = set(_read_preds(output_path / "preds.json").keys())
        before   = len(instances)
        instances = [i for i in instances if str(i.get("instance_id", "")) not in existing]
        logger.info("[CIBench] Skipping %d already-completed instances", before - len(instances))

    logger.info("[CIBench] Running on %d instances with %d worker(s)", len(instances), workers)

    # ── Config ────────────────────────────────────────────────────────────────
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "model": {
            "model_name":  model_name  or UNSET,
            "model_class": model_class or UNSET,
        },
    })
    config = recursive_merge(*configs)

    # ── Progress ──────────────────────────────────────────────────────────────
    progress_manager = RunBatchProgressManager(
        len(instances),
        output_path / f"exit_statuses_{int(time.time())}.yaml",
    )

    def _process(instance: Dict[str, Any]) -> None:
        process_instance(
            instance,
            output_path,
            config,
            progress_manager,
            memory_root=memory_root,
            memory_enabled=memory_enabled,
            memory_top_k=memory_top_k,
            memory_ablation_levels=memory_ablation_levels,
            memory_plugin_path=memory_plugin_path,
            context_model=context_model,
            save_memory=save_memory and memory_enabled,
        )

    def _process_futures(futures: "dict[concurrent.futures.Future, str]") -> None:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as exc:
                iid = futures[future]
                logger.error("[CIBench] Uncaught exception for %s: %s", iid, exc, exc_info=True)
                progress_manager.on_uncaught_exception(iid, exc)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process, instance): str(instance.get("instance_id", ""))
                for instance in instances
            }
            try:
                _process_futures(futures)
            except KeyboardInterrupt:
                logger.info("[CIBench] Cancelling pending jobs. Press ^C again to exit immediately.")
                for f in futures:
                    if not f.running() and not f.done():
                        f.cancel()
                _process_futures(futures)

    # ── Final summary ─────────────────────────────────────────────────────────
    preds = _read_preds(output_path / "preds.json")
    n_patched = sum(1 for v in preds.values() if v.get("diff", "").strip())
    logger.info(
        "[CIBench] Done. %d/%d instances produced a non-empty patch. See %s",
        n_patched, len(preds), output_path / "preds.json",
    )


if __name__ == "__main__":
    app()
