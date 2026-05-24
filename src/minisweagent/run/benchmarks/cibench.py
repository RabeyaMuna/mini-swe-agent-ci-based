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
  <output_dir>/<instance_id>/<instance_id>.traj.json

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
from minisweagent.utils.serialize import UNSET, recursive_merge

# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "cibench.yaml"
_OUTPUT_FILE_LOCK   = threading.Lock()
app = typer.Typer(rich_markup_mode="rich", add_completion=False)

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

def load_ci_instances(dataset_path: str, split: str = "test") -> List[Dict[str, Any]]:
    """
    Load CI benchmark instances from a JSONL file or a HuggingFace dataset.

    Each returned dict must have at minimum:
        instance_id, sha_fail, repo_owner, repo_name,
        workflow, workflow_path, workflow_name, logs
    """
    p = Path(dataset_path)

    # Local JSONL file
    if p.exists() and p.suffix in (".jsonl", ".json"):
        instances: List[Dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, list):
                        instances.extend(obj)
                    else:
                        instances.append(obj)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed line in %s: %s", dataset_path, e)
        logger.info("Loaded %d instances from %s", len(instances), dataset_path)
        return instances

    # Local JSON array file
    if p.exists() and p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        instances = data if isinstance(data, list) else [data]
        logger.info("Loaded %d instances from %s", len(instances), dataset_path)
        return instances

    # HuggingFace dataset
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(dataset_path, split=split)
        instances = list(ds)  # type: ignore
        logger.info("Loaded %d instances from HuggingFace dataset %s", len(instances), dataset_path)
        return instances
    except Exception as e:
        raise ValueError(
            f"Could not load dataset from '{dataset_path}'. "
            f"Expected a .jsonl / .json file path or a HuggingFace dataset name. Error: {e}"
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
    Clone the repository from GitHub and return *(env, testbed_path)*.

    • Clones ``https://github.com/<repo_owner>/<repo_name>`` to
      ``<instance_dir>/testbed/``.
    • Checks out the failing commit (``sha_fail``).
    • Wraps the directory in a ``LocalEnvironment`` using settings from the
      ``environment`` section of the config (timeout, interpreter, env vars).

    If the testbed directory already contains a ``.git`` folder (re-run),
    the clone step is skipped so previous work is preserved.
    """
    repo_owner = instance.get("repo_owner", "")
    repo_name  = instance.get("repo_name", "")
    sha_fail   = str(instance.get("sha_fail") or "")

    testbed_path = instance_dir / "testbed"

    # ── Clone ─────────────────────────────────────────────────────────────────
    if not (testbed_path / ".git").exists():
        testbed_path.parent.mkdir(parents=True, exist_ok=True)
        clone_url = f"https://github.com/{repo_owner}/{repo_name}.git"
        logger.info("[CIBench] Cloning %s → %s", clone_url, testbed_path)
        result = subprocess.run(
            ["git", "clone", "--quiet", clone_url, str(testbed_path)],
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
        logger.info("[CIBench] Reusing existing clone at %s", testbed_path)

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
            logger.warning(
                "[CIBench] Startup command returned non-zero for %s:\n%s",
                instance.get("instance_id"),
                out.get("output", "")[:500],
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
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pre-processing CI logs")

    agent      = None
    exit_status: Optional[str] = None
    diff       = ""
    ci_ctx     = {}
    extra_info: Dict[str, Any] = {}

    # ── Phase 1: build enriched CI problem statement ──────────────────────────
    try:
        ci_result = build_ci_context(
            instance,
            memory_root=memory_root,
            memory_enabled=memory_enabled,
            memory_top_k=memory_top_k,
            memory_ablation_levels=memory_ablation_levels,
            memory_plugin_path=memory_plugin_path,
            model=context_model,
        )
        ci_ctx = ci_result["context"]
        task   = ci_result["problem_statement"]
        extra_info["memory_summary"] = {
            "enabled":             memory_enabled,
            "weighted_similarity": ci_result["memory"].get("weighted_similarity", 0.0),
            "levels_retrieved":    ci_result["memory"].get("selected_memory_levels", []),
        }
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
            traj_path = instance_dir / f"{instance_id}.traj.json"
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
                    "ci_context": {
                        "overall_failure_reasons": ci_ctx.get("overall_failure_reasons", []),
                        "overall_error_types":     ci_ctx.get("overall_error_types", []),
                        "effected_files":          ci_ctx.get("effected_files", []),
                        "failed_jobs":             ci_ctx.get("failed_jobs", []),
                    } if ci_ctx else {},
                },
            )
            logger.info("[CIBench] Saved trajectory to '%s'", traj_path)

        update_preds_file(output_dir / "preds.json", instance_id, sha_fail, diff)
        progress_manager.on_instance_end(instance_id, exit_status)


def _extract_diff(submission: str) -> str:
    """
    Pull the git diff out of the agent's final submission string.

    The agent echoes ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` then cats the
    patch file.  We take everything after that sentinel (or the full string
    if the sentinel is absent).
    """
    sentinel = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    if sentinel in submission:
        return submission[submission.index(sentinel) + len(sentinel):].strip()
    if submission.strip().startswith("diff --git") or submission.strip().startswith("---"):
        return submission.strip()
    return submission.strip()


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
