"""
ci_context.py
=============
Pre-processing pipeline that enriches a CI failure instance before passing
it to mini-swe-agent.

══════════════════════════════════════════════════════════════════════
PIPELINE OVERVIEW
══════════════════════════════════════════════════════════════════════

Phase A — Log Summarisation  (CILogAnalyzer — 3 LLM calls)
  Normalize raw CI logs -> per-step chunks -> LLM evidence extraction
  -> LLM merge/summarise into:
    error_context, relevant_files, error_types, failed_job

  CILogAnalyzer is self-contained and works with any LLM callable.
  It is ported from CI-REPAIR-BENCH's CILogAnalyzerLLM and produces
  the same output schema so memory entries are interoperable.

Phase B — Workflow Profile Extraction (ErrorContextAgent Phase 3 — optional)
  Analyse workflow YAML with 3 LLM calls:
    LLM-1 prompt_workflow_analysis()
        -> installation_cmd (list of install commands)
        -> validation_cmd  (list of commands CI runs to validate)
    LLM-2 prompt_config_prioritization()
        -> pick the most relevant config files
    LLM-3 prompt_project_validation_checks()
        -> refine installation_cmd / validation_cmd using config files
  Falls back to basic YAML parsing when mem-ci-repair-agent is not installed.
  ┌─────────────────────────────────────────────────────────────────┐
  │ OUTPUT (the only workflow info passed to the agent):      │
  │  installation_cmd — how to install the project        │
  │  validation_cmd  — exact commands to run to test a fix   │
  │  critical_steps  — steps the CI considers mandatory     │
  └─────────────────────────────────────────────────────────────────┘

Phase C — Memory Retrieval (CIMemorySystem — 5 explicit steps)
  Step 1 Build a structured query document from Phase A/B output
      (error_type, failure_pattern, failure_reason, relevant_files,
       failed_tool, workflow_text)

  Step 2 Embedding Generation
      sentence-transformers all-MiniLM-L6-v2 (primary)
      OR fastembed BAAI/bge-base-en-v1.5 (secondary)
      -> dense unit-norm vectors per memory record

  Step 3 Cosine Similarity Search (per-level weighted scoring)
      L1 per-file failure records
        0.45×file_exact + 0.50×semantic + 0.05×tool_cmd
      L2 repo-scoped recurring patterns
        0.90×semantic  + 0.10×tool_cmd
      L3 cross-repo generalised principles
        0.90×semantic  + 0.10×tool_cmd

  Step 4 Ranking — merge L1+L2+L3, sort similarity high -> low

  Step 5 LLM Relevance Gate
      ALL ranked candidates -> LLM prompt that:
       • Compares each candidate against the CURRENT failure context
       • Selects ONLY genuinely relevant items
       • Returns selected items + key_insight + fix_direction per item
       • Produces an analysis_summary (2–3 sentences)

Phase D — CI Document Assembly
  Build the final problem_statement from:
    Phase A (error reasons, failed jobs, affected files)
    Phase B (validation_cmd, installation_cmd)
    Phase C (LLM-selected memory context only)

══════════════════════════════════════════════════════════════════════
OUTPUT of build_ci_context()
══════════════════════════════════════════════════════════════════════
{
 "context": {
  "id":           str,
  "sha_fail":        str,
  "repo":          str,
  "repo_name":        str,
  "repo_owner":       str,
  "workflow_name":      str,
  "workflow_path":      str,
  "overall_failure_reasons": [str],
  "overall_error_types":   [str],
  "effected_files": [{file, reason, issue_type, failed_cmd, failed_tool, line_number}],
  "failed_jobs":   [{job, step, command}],
  "workflow_profile": {
   "installation_cmd": [str],  # Phase B output
   "validation_cmd":  [str],  # Phase B output
   "critical_steps":  [str],
  },
  "_log_analysis":  dict,    # raw Phase A output for memory save
 },
 "memory": {
  "enabled":       bool,
  "raw_retrieval":    dict,  # full MemoryPlugin.retrieve() result
  "llm_selection":    dict,  # LLM gate result (Step 5)
  "all_ranked_candidates": [...],
  "weighted_similarity": float,
  "selected_memory_levels": [str],
 },
 "problem_statement": str,     # ready for agent.run()
}
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Local utilities
from minisweagent.run.benchmarks.utils.ci_log_analyzer import CILogAnalyzer
from minisweagent.run.benchmarks.utils.ci_memory_system import (
  CIMemorySystem,
  format_memory_context,
)
from minisweagent.run.benchmarks.utils.ci_workflow_aware_retrieval import (
  analyze_workflow_from_benchmark,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
WORKFLOW_VALIDATION_CACHE = PROJECT_ROOT / "data" / "trs" / "workflow_validation_cache.json"
LOG_ANALYSIS_CACHE = PROJECT_ROOT / "data" / "trs" / "log_details.json"


# ══════════════════════════════════════════════════════════════════════════════
# Optional ErrorContextAgent import (Phase B — workflow profile)
# ══════════════════════════════════════════════════════════════════════════════

def _try_import_error_context_agent():
  """Try to import ErrorContextAgent from mem-ci-repair-agent."""
  try:
    from memci.agents.error_context_based_agent import ErrorContextAgent # type: ignore
    return ErrorContextAgent
  except ImportError:
    logger.debug("memci not installed — ErrorContextAgent unavailable; using YAML fallback.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Schema mapping: CILogAnalyzer output -> context dict
# ══════════════════════════════════════════════════════════════════════════════

def _log_analysis_to_context(
  log_result: Dict[str, Any],
  instance: Dict[str, Any],
  workflow_profile: Dict[str, Any],
) -> Dict[str, Any]:
  """
  Map CILogAnalyzer.run() output (CI-REPAIR-BENCH schema) to the context
  dict that build_problem_statement() uses.

  CILogAnalyzer output keys:
    sha_fail, id, error_context, relevant_files, error_types, failed_job

  Context keys consumed by build_problem_statement:
    id, sha_fail, repo, overall_failure_reasons, overall_error_types,
    effected_files, failed_jobs, workflow_profile
  """
  sha_fail = str(log_result.get("sha_fail") or instance.get("sha_fail") or "")
  task_id  = str(log_result.get("id") or instance.get("instance_id") or instance.get("id") or sha_fail)
  repo_name = str(instance.get("repo_name") or "")
  repo_owner = str(instance.get("repo_owner") or "")
  repo   = str(
    instance.get("repo")
    or f"{repo_owner}/{repo_name}".strip("/")
  )
  workflow_path = str(instance.get("workflow_path") or "")
  workflow_name = (
    str(instance.get("workflow_name") or "")
    or os.path.splitext(os.path.basename(workflow_path))[0]
  )

  # overall_failure_reasons ← error_context (list of strings)
  error_context = log_result.get("error_context") or []
  if isinstance(error_context, list):
    overall_failure_reasons = [str(x) for x in error_context if str(x).strip()]
  else:
    overall_failure_reasons = [str(error_context)] if error_context else []

  # overall_error_types ← error_types[*].category
  raw_error_types = log_result.get("error_types") or []
  overall_error_types: List[str] = []
  for et in raw_error_types:
    if isinstance(et, dict):
      cat = str(et.get("category") or et.get("subcategory") or "").strip()
      if cat:
        overall_error_types.append(cat)
    else:
      s = str(et).strip()
      if s:
        overall_error_types.append(s)

  # effected_files ← relevant_files
  raw_files = log_result.get("relevant_files") or []
  effected_files: List[Dict[str, Any]] = []
  for f in raw_files:
    if isinstance(f, dict):
      effected_files.append({
        "file":     str(f.get("file") or "").strip(),
        "reason":    str(f.get("reason") or "").strip(),
        "issue_type":  str(f.get("issue_type") or "").strip(),
        "failed_cmd":  f.get("failed_cmd"),
        "failed_tool": f.get("failed_tool"),
        "failed_command": str(f.get("failed_cmd") or ""), # alias for compat
        "line_number": f.get("line_number"),
        "error_type":  str(f.get("issue_type") or "").strip(),
      })
    elif isinstance(f, str) and f.strip():
      effected_files.append({"file": f.strip(), "reason": "", "issue_type": ""})

  # failed_jobs ← failed_job
  raw_jobs = log_result.get("failed_job") or log_result.get("failed_jobs") or []
  failed_jobs: List[Dict[str, Any]] = []
  for j in (raw_jobs if isinstance(raw_jobs, list) else []):
    if isinstance(j, dict):
      failed_jobs.append({
        "job":      str(j.get("job") or ""),
        "step":      str(j.get("step") or ""),
        "command":    str(j.get("command") or ""),
        "failed_command": str(j.get("command") or ""), # alias for compat
      })
    else:
      s = str(j).strip()
      if s:
        failed_jobs.append({"step": s, "command": "", "failed_command": ""})

  return {
    "id":           task_id,
    "sha_fail":        sha_fail,
    "repo":          repo,
    "repo_name":        repo_name,
    "repo_owner":       repo_owner,
    "workflow_name":      workflow_name,
    "workflow_path":      workflow_path,
    "overall_failure_reasons": overall_failure_reasons,
    "overall_error_types":   overall_error_types,
    "effected_files":     effected_files,
    "failed_jobs":       failed_jobs,
    "workflow_profile":    workflow_profile,
    "_log_analysis":      log_result, # kept for memory save
  }


# ══════════════════════════════════════════════════════════════════════════════
# Phase A — CILogAnalyzer (log summarisation)
# ══════════════════════════════════════════════════════════════════════════════

def _has_precomputed_analysis(instance: Dict[str, Any]) -> bool:
  """
  Return True when the instance already carries TRS-pipeline analysis fields
  so Phase A (CILogAnalyzer) can be skipped entirely — saving 3+ LLM calls.

  Required: at least one of the three primary analysis outputs must be present.
  """
  return bool(
    instance.get("overall_failure_reasons")
    or instance.get("effected_files")
    or instance.get("failed_jobs")
  )


def _build_from_precomputed(instance: Dict[str, Any]) -> Dict[str, Any]:
  """
  Build a log_analysis_result directly from TRS-pipeline pre-computed fields.

  Field mapping (instance -> CI-REPAIR-BENCH log_analysis schema):
    overall_failure_reasons -> error_context
    effected_files     -> relevant_files  (with key aliases)
    error_types / overall_error_types -> error_types (wrapped into {category, ...})
    failed_jobs       -> failed_job
  """
  sha_fail = str(instance.get("sha_fail") or "")
  task_id = str(instance.get("instance_id") or instance.get("id") or sha_fail)

  # ── error_context ─────────────────────────────────────────────────────────
  error_context: List[str] = [
    str(r).strip()
    for r in (instance.get("overall_failure_reasons") or [])
    if str(r).strip()
  ]

  # ── relevant_files — normalise key names ──────────────────────────────────
  raw_files = instance.get("effected_files") or []
  relevant_files: List[Dict[str, Any]] = []
  for f in raw_files:
    if not isinstance(f, dict):
      continue
    relevant_files.append({
      "file":     str(f.get("file")     or "").strip(),
      "line_number": f.get("line_number"),
      "reason":    str(f.get("reason")    or "").strip(),
      "issue_type":  str(f.get("issue_type")  or f.get("error_type") or "").strip(),
      "failed_cmd":  f.get("failed_cmd")  or f.get("failed_command"),
      "failed_tool": f.get("failed_tool"),
      "failed_tools": f.get("failed_tools") or [],
      "failed_command": str(f.get("failed_cmd") or f.get("failed_command") or ""),
      "error_type":  str(f.get("issue_type")  or f.get("error_type") or "").strip(),
      "error_subtype": str(f.get("error_subtype") or "").strip(),
    })

  # ── error_types — wrap plain strings into {category, subcategory, evidence} ─
  raw_et = instance.get("error_types") or instance.get("overall_error_types") or []
  error_types: List[Dict[str, Any]] = []
  for et in raw_et:
    if isinstance(et, dict):
      error_types.append(et)
    elif str(et).strip():
      error_types.append({
        "category":  str(et).strip(),
        "subcategory": "",
        "evidence":  "",
      })

  # ── failed_job ─────────────────────────────────────────────────────────────
  raw_jobs = instance.get("failed_jobs") or []
  failed_job: List[Dict[str, Any]] = []
  for j in raw_jobs:
    if isinstance(j, dict):
      failed_job.append({
        "job":   str(j.get("job")       or ""),
        "step":  str(j.get("step")       or ""),
        "command": str(j.get("command")     or j.get("failed_command") or ""),
        "failed_command": str(j.get("failed_command") or j.get("command") or ""),
      })
    elif str(j).strip():
      failed_job.append({"step": str(j), "command": "", "failed_command": ""})

  # ── overall_ci_summary ─────────────────────────────────────────────────────
  overall_ci_summary = " | ".join(error_context[:3]) if error_context else ""

  return {
    "sha_fail":      sha_fail,
    "id":         task_id,
    "error_context":   error_context,
    "relevant_files":   relevant_files,
    "error_types":    error_types,
    "failed_job":     failed_job,
    "overall_ci_summary": overall_ci_summary,
    "chunk_summaries":  [],
    "analysis_document": overall_ci_summary,
    "_precomputed":    True,  # flag: skipped CILogAnalyzer
  }


def _load_log_analysis_cache(*, sha_fail: str, task_id: str) -> Dict[str, Any]:
  if not LOG_ANALYSIS_CACHE.exists():
    return {}
  try:
    payload = json.loads(LOG_ANALYSIS_CACHE.read_text(encoding="utf-8"))
  except Exception as exc:
    logger.warning("[Phase A] Could not read log analysis cache: %s", exc)
    return {}

  rows = payload if isinstance(payload, list) else list(payload.values()) if isinstance(payload, dict) else []
  for row in rows:
    if not isinstance(row, dict):
      continue
    if sha_fail and str(row.get("sha_fail") or "") == sha_fail:
      return row
    if task_id and str(row.get("id") or row.get("instance_id") or "") == task_id:
      return row
  return {}


def _save_log_analysis_cache(result: Dict[str, Any]) -> None:
  sha_fail = str(result.get("sha_fail") or "")
  task_id = str(result.get("id") or result.get("instance_id") or "")
  if not sha_fail and not task_id:
    return
  if result.get("error"):
    return

  try:
    rows: List[Dict[str, Any]] = []
    if LOG_ANALYSIS_CACHE.exists():
      payload = json.loads(LOG_ANALYSIS_CACHE.read_text(encoding="utf-8"))
      if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
      elif isinstance(payload, dict):
        rows = [row for row in payload.values() if isinstance(row, dict)]

    updated = False
    for index, row in enumerate(rows):
      if (sha_fail and str(row.get("sha_fail") or "") == sha_fail) or (
        task_id and str(row.get("id") or row.get("instance_id") or "") == task_id
      ):
        rows[index] = result
        updated = True
        break
    if not updated:
      rows.append(result)

    LOG_ANALYSIS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LOG_ANALYSIS_CACHE.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[Phase A] Saved log analysis to %s", LOG_ANALYSIS_CACHE)
  except Exception as exc:
    logger.warning("[Phase A] Could not save log analysis cache: %s", exc)


def _run_log_analysis(
  instance: Dict[str, Any],
  llm: Any,
  model: str,
) -> Dict[str, Any]:
  """
  Phase A — build a structured log_analysis_result.

  Fast path (no LLM cost):
    If the instance already carries pre-computed TRS-pipeline fields
    (overall_failure_reasons, effected_files, failed_jobs), use them
    directly. This avoids 3+ LLM calls per instance and typically
    produces *better* results because the TRS pipeline ran the full
    CILogAnalyzerLLM with access to complete logs.

  Slow path (LLM required):
    Run CILogAnalyzer on the raw logs. Falls back to the minimal
    heuristic analysis if the LLM is unavailable or the analyzer fails.
  """
  sha_fail   = str(instance.get("sha_fail") or "")
  task_id    = str(instance.get("instance_id") or instance.get("id") or sha_fail)

  if _has_precomputed_analysis(instance):
    result = _build_from_precomputed(instance)
    logger.info(
      "[Phase A] Using pre-computed fields (skipped CILogAnalyzer) — "
      "error_types=%d files=%d jobs=%d",
      len(result["error_types"]),
      len(result["relevant_files"]),
      len(result["failed_job"]),
    )
    _save_log_analysis_cache(result)
    return result

  cached = _load_log_analysis_cache(sha_fail=sha_fail, task_id=task_id)
  if cached:
    logger.info(
      "[Phase A] Loaded cached log analysis for %s — error_types=%d files=%d jobs=%d",
      (sha_fail or task_id)[:12],
      len(cached.get("error_types") or []),
      len(cached.get("relevant_files") or []),
      len((cached.get("failed_job") or cached.get("failed_jobs")) or []),
    )
    return cached

  workflow   = instance.get("workflow") or ""
  workflow_path = str(instance.get("workflow_path") or "")
  logs     = instance.get("logs") or instance.get("log") or ""

  try:
    analyzer = CILogAnalyzer(
      logs=logs,
      sha_fail=sha_fail,
      workflow=workflow,
      workflow_path=workflow_path,
      llm=llm,
      model_name=model,
      task_id=task_id,
    )
    result = analyzer.run()
    if result.get("error"):
      logger.warning("[Phase A] CILogAnalyzer returned error: %s", result["error"])
      return _minimal_log_analysis(instance)
    _save_log_analysis_cache(result)
    return result
  except Exception as exc:
    logger.warning("[Phase A] CILogAnalyzer failed (%s); using raw log fallback.", exc)
    return _minimal_log_analysis(instance)


def _minimal_log_analysis(instance: Dict[str, Any]) -> Dict[str, Any]:
  """Minimal structured analysis used when CILogAnalyzer fails or has no LLM."""
  sha_fail = str(instance.get("sha_fail") or "")
  task_id = str(instance.get("instance_id") or instance.get("id") or sha_fail)
  logs   = instance.get("logs") or instance.get("log") or ""

  chunk_summaries: List[Dict[str, Any]] = []
  overall_ci_summary_parts: List[str] = []

  if isinstance(logs, list):
    for index, item in enumerate(logs, start=1):
      if isinstance(item, dict):
        step_name = str(item.get("step_name") or f"step_{index}")
        log_text = str(item.get("log") or item.get("text") or "")
      else:
        step_name = f"step_{index}"
        log_text = str(item)
      token_keywords = _extract_structured_keywords(log_text)
      code_context = _extract_structured_code_context(log_text)
      summary = _summarize_unstructured_log_text(log_text, token_keywords=token_keywords, code_context=code_context)
      chunk_summaries.append({
        "step_name": step_name,
        "chunk_index": 1,
        "chunk_total": 1,
        "chunk_token_count": None,
        "token_keywords": token_keywords,
        "summary": summary,
        "relevant_failures": [],
        "relevant_files": [],
        "code_context": code_context,
        "chunk_document": f"step={step_name} ; keywords={', '.join(token_keywords[:8])}" if token_keywords else f"step={step_name}",
      })
      if summary:
        overall_ci_summary_parts.append(f"{step_name}: {summary}")
  else:
    log_text = str(logs or "")
    token_keywords = _extract_structured_keywords(log_text)
    code_context = _extract_structured_code_context(log_text)
    summary = _summarize_unstructured_log_text(log_text, token_keywords=token_keywords, code_context=code_context)
    chunk_summaries.append({
      "step_name": "ci_log",
      "chunk_index": 1,
      "chunk_total": 1,
      "chunk_token_count": None,
      "token_keywords": token_keywords,
      "summary": summary,
      "relevant_failures": [],
      "relevant_files": [],
      "code_context": code_context,
      "chunk_document": f"step=ci_log ; keywords={', '.join(token_keywords[:8])}" if token_keywords else "step=ci_log",
    })
    if summary:
      overall_ci_summary_parts.append(summary)

  overall_ci_summary = " | ".join(part for part in overall_ci_summary_parts if part)
  analysis_lines = []
  if overall_ci_summary:
    analysis_lines.append(f"overall_ci_summary: {overall_ci_summary}")
  for chunk in chunk_summaries:
    summary = str(chunk.get("summary") or "").strip()
    if summary:
      analysis_lines.append(f"{chunk['step_name']}: {summary}")
    code_context = chunk.get("code_context") or []
    if code_context:
      analysis_lines.append("code_context: " + " | ".join(str(x).strip() for x in code_context[:3] if str(x).strip()))

  return {
    "sha_fail":   sha_fail,
    "id":      task_id,
    "error_context": [overall_ci_summary] if overall_ci_summary else [],
    "relevant_files": [],
    "error_types":  [],
    "failed_job":  [],
    "chunk_summaries": chunk_summaries,
    "overall_ci_summary": overall_ci_summary,
    "analysis_document": "\n".join(analysis_lines).strip(),
    "_fallback":   True,
  }


# ══════════════════════════════════════════════════════════════════════════════
# Phase B — Workflow profile (ErrorContextAgent / YAML fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_workflow_profile(
  instance: Dict[str, Any],
  *,
  memory_root: str,
  memory_enabled: bool,
  model: str,
  llm: Any = None,
) -> Dict[str, Any]:
  """
  Phase B — extract the ordered validation sequence from benchmark workflow YAML.

  Priority:
   1. data/trs/workflow_validation_cache.json by sha_fail/id.
   2. ci_workflow_aware_retrieval.analyze_workflow_from_benchmark().
   3. Legacy failed-job/YAML fallback for compatibility.
  """
  workflow_profile = _workflow_profile_from_cache_or_analyzer(instance, llm=llm)
  if workflow_profile.get("validation_sequence"):
    return workflow_profile

  # Compatibility fallback 1: extract validation_cmd from pre-computed failed_jobs
  # (the failing command IS the test/lint command the agent needs to re-run)
  precomputed = _extract_validation_from_failed_jobs(instance)
  if precomputed.get("validation_cmd"):
    logger.info("[Phase B] Using pre-computed validation_cmd from failed_jobs")
    yaml_profile = _parse_workflow_yaml_commands(str(instance.get("workflow") or ""))
    return {
      "installation_cmd": _coerce_str_list(yaml_profile.get("installation_cmd")),
      "validation_cmd": _coerce_str_list(precomputed.get("validation_cmd")),
      "critical_steps": _coerce_str_list(yaml_profile.get("critical_steps")),
      "validation_sequence": [],
      "dependent_files": [],
    }

  ErrorContextAgent = _try_import_error_context_agent()

  if ErrorContextAgent is not None:
    profile = _run_error_context_agent_phase_b(
      instance,
      ErrorContextAgent=ErrorContextAgent,
      memory_root=memory_root,
      memory_enabled=memory_enabled,
      model=model,
    )
    # Only return if we got meaningful results
    if profile.get("installation_cmd") or profile.get("validation_cmd"):
      return profile

  # Fallback 2: basic YAML parsing
  return _parse_workflow_yaml_commands(str(instance.get("workflow") or ""))


def _workflow_profile_from_cache_or_analyzer(instance: Dict[str, Any], *, llm: Any = None) -> Dict[str, Any]:
  sha_fail = str(instance.get("sha_fail") or "")
  issue_id = str(instance.get("id") or instance.get("instance_id") or "")
  workflow_path = str(instance.get("workflow_path") or instance.get("workflow_filename") or "")

  cached = _load_workflow_validation_cache(sha_fail=sha_fail, issue_id=issue_id)
  if cached:
    logger.info("[Phase B] Loaded cached workflow validation sequence for %s", (sha_fail or issue_id)[:12])
    return _workflow_validation_context_to_profile(cached)

  workflow_content = str(instance.get("workflow") or "")
  if not workflow_content.strip() or llm is None:
    return {"installation_cmd": [], "validation_cmd": [], "critical_steps": [], "validation_sequence": [], "dependent_files": []}

  try:
    logger.info("[Phase B] Extracting workflow validation sequence with ci_workflow_aware_retrieval")
    context = analyze_workflow_from_benchmark(
      workflow_content=workflow_content,
      workflow_path=workflow_path,
      repo_path=str(instance.get("repo_path") or instance.get("checkout_path") or "") or None,
      llm=llm,
      issue_id=issue_id,
      sha_fail=sha_fail,
    )
  except Exception as exc:
    logger.warning("[Phase B] Workflow validation extraction failed (%s); using legacy fallback.", exc)
    return {"installation_cmd": [], "validation_cmd": [], "critical_steps": [], "validation_sequence": [], "dependent_files": []}

  if context.get("validation_sequence"):
    _save_workflow_validation_cache(context)
    return _workflow_validation_context_to_profile(context)

  return {"installation_cmd": [], "validation_cmd": [], "critical_steps": [], "validation_sequence": [], "dependent_files": []}


def _load_workflow_validation_cache(*, sha_fail: str, issue_id: str) -> Dict[str, Any]:
  if not WORKFLOW_VALIDATION_CACHE.exists():
    return {}
  try:
    payload = json.loads(WORKFLOW_VALIDATION_CACHE.read_text(encoding="utf-8"))
  except Exception as exc:
    logger.warning("[Phase B] Could not read workflow validation cache: %s", exc)
    return {}

  rows = payload if isinstance(payload, list) else list(payload.values()) if isinstance(payload, dict) else []
  for row in rows:
    if not isinstance(row, dict):
      continue
    if sha_fail and str(row.get("sha_fail") or "") == sha_fail:
      return row
    if issue_id and str(row.get("id") or row.get("instance_id") or "") == issue_id:
      return row
  return {}


def _save_workflow_validation_cache(context: Dict[str, Any]) -> None:
  sha_fail = str(context.get("sha_fail") or "")
  issue_id = str(context.get("id") or context.get("instance_id") or "")
  if not sha_fail and not issue_id:
    return

  try:
    rows: List[Dict[str, Any]] = []
    if WORKFLOW_VALIDATION_CACHE.exists():
      payload = json.loads(WORKFLOW_VALIDATION_CACHE.read_text(encoding="utf-8"))
      if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
      elif isinstance(payload, dict):
        rows = [row for row in payload.values() if isinstance(row, dict)]

    compact = {
      "id": issue_id,
      "sha_fail": sha_fail,
      "workflow_path": str(context.get("workflow_path") or ""),
      "dependent_files": context.get("dependent_files", []) or [],
      "validation_sequence": context.get("validation_sequence", []) or [],
    }

    updated = False
    for index, row in enumerate(rows):
      if (sha_fail and str(row.get("sha_fail") or "") == sha_fail) or (
        issue_id and str(row.get("id") or row.get("instance_id") or "") == issue_id
      ):
        rows[index] = compact
        updated = True
        break
    if not updated:
      rows.append(compact)

    WORKFLOW_VALIDATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    WORKFLOW_VALIDATION_CACHE.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[Phase B] Saved workflow validation sequence to %s", WORKFLOW_VALIDATION_CACHE)
  except Exception as exc:
    logger.warning("[Phase B] Could not save workflow validation cache: %s", exc)


def _workflow_validation_context_to_profile(context: Dict[str, Any]) -> Dict[str, Any]:
  sequence = context.get("validation_sequence") or []
  installation_cmd: List[str] = []
  validation_cmd: List[str] = []
  critical_steps: List[str] = []

  for step in sequence if isinstance(sequence, list) else []:
    if not isinstance(step, dict):
      continue
    install = str(step.get("installation_cmd") or "").strip()
    validate = str(step.get("validation_cmd") or "").strip()
    label = str(step.get("validates") or validate or install or "").strip()
    if install and install not in installation_cmd:
      installation_cmd.append(install)
    if validate and validate not in validation_cmd:
      validation_cmd.append(validate)
    if label and label not in critical_steps:
      critical_steps.append(label)

  logger.info(
    "[Phase B] Workflow validation sequence: steps=%d install_cmds=%d validation_cmds=%d",
    len(sequence) if isinstance(sequence, list) else 0,
    len(installation_cmd),
    len(validation_cmd),
  )
  return {
    "installation_cmd": installation_cmd,
    "validation_cmd": validation_cmd,
    "critical_steps": critical_steps,
    "validation_sequence": sequence if isinstance(sequence, list) else [],
    "dependent_files": context.get("dependent_files", []) or [],
  }


def _run_error_context_agent_phase_b(
  instance: Dict[str, Any],
  *,
  ErrorContextAgent: Any,
  memory_root: str,
  memory_enabled: bool,
  model: str,
) -> Dict[str, Any]:
  """
  Run ErrorContextAgent in a temp repo dir to extract the workflow profile.
  Returns {installation_cmd, validation_cmd, critical_steps}.
  """
  sha_fail   = str(instance.get("sha_fail") or "")
  repo_owner  = str(instance.get("repo_owner") or "")
  repo_name   = str(instance.get("repo_name") or "")
  workflow   = str(instance.get("workflow") or "")
  workflow_path = str(instance.get("workflow_path") or "")
  workflow_name = (
    str(instance.get("workflow_name") or "")
    or os.path.splitext(os.path.basename(workflow_path))[0]
  )
  logs  = instance.get("logs") or instance.get("log") or ""
  task_id = str(instance.get("instance_id") or instance.get("id") or sha_fail)

  with tempfile.TemporaryDirectory(prefix="mini_swe_ci_phaseB_") as tmp_repo:
    try:
      agent = ErrorContextAgent(
        repo_path=tmp_repo,
        logs=logs,
        sha_fail=sha_fail,
        workflow=workflow,
        workflow_path=workflow_path,
        workflow_name=workflow_name,
        repo_owner=repo_owner,
        repo_name=repo_name,
        model=model,
        memory_root=memory_root,
        memory_enabled=memory_enabled,
        task_id=task_id,
      )
      result = agent.run()

      # Pull workflow profile from agent's memory store
      installation_cmd: List[str] = []
      validation_cmd:  List[str] = []
      critical_steps:  List[str] = []

      if hasattr(agent, "memory"):
        pid = str(result.get("workflow_profile_id") or "")
        if pid:
          profile = agent.memory.get_workflow_profile(pid)
          if profile:
            vp = profile.get("validation_profile") or {}
            installation_cmd = _coerce_str_list(vp.get("installation_cmd"))
            validation_cmd  = _coerce_str_list(vp.get("validation_cmd"))
            critical_steps  = _coerce_str_list(vp.get("critical_steps"))

      logger.info(
        "[Phase B] ErrorContextAgent: install_cmds=%d validation_cmds=%d",
        len(installation_cmd), len(validation_cmd),
      )
      return {
        "installation_cmd": installation_cmd,
        "validation_cmd":  validation_cmd,
        "critical_steps":  critical_steps,
      }

    except Exception as exc:
      logger.warning("[Phase B] ErrorContextAgent phase B failed (%s); using YAML fallback.", exc)
      return {"installation_cmd": [], "validation_cmd": [], "critical_steps": []}


def _extract_validation_from_failed_jobs(instance: Dict[str, Any]) -> Dict[str, Any]:
  """
  Extract validation_cmd directly from pre-computed failed_jobs field.

  The command that failed in CI *is* the validation command the agent needs
  to run to reproduce and then verify their fix. This is more accurate
  than YAML regex parsing because it comes from the actual CI execution.
  """
  failed_jobs = instance.get("failed_jobs") or []
  validation_cmd: List[str] = []
  seen: set = set()

  for j in failed_jobs:
    if not isinstance(j, dict):
      continue
    cmd = str(j.get("failed_command") or j.get("command") or "").strip()
    if cmd and cmd not in seen:
      validation_cmd.append(cmd)
      seen.add(cmd)

  # Also check effected_files for per-file failed commands
  for ef in (instance.get("effected_files") or []):
    if not isinstance(ef, dict):
      continue
    cmd = str(ef.get("failed_cmd") or ef.get("failed_command") or "").strip()
    if cmd and cmd not in seen:
      validation_cmd.append(cmd)
      seen.add(cmd)

  # Deduplicate: prefer pytest/mypy/ruff invocations, drop install commands
  _install_pat = re.compile(
    r"(pip install|poetry install|npm install|yarn install|apt-get)", re.I
  )
  validation_cmd = [c for c in validation_cmd if not _install_pat.search(c)]

  return {
    "validation_cmd":  validation_cmd[:6],
    "installation_cmd": [],
    "critical_steps":  [],
  }


def _parse_workflow_yaml_commands(workflow_yaml: str) -> Dict[str, Any]:
  """
  Basic YAML regex parser to extract install / test / lint commands.
  Used as Phase B fallback when ErrorContextAgent is unavailable.

  Heuristics:
   • Lines with 'pip install', 'npm install', 'poetry install', etc. -> installation_cmd
   • Lines with 'pytest', 'python -m pytest', 'npm test', etc. -> validation_cmd
  """
  installation_cmd: List[str] = []
  validation_cmd:  List[str] = []
  critical_steps:  List[str] = []

  if not workflow_yaml:
    return {"installation_cmd": [], "validation_cmd": [], "critical_steps": []}

  # Extract all 'run:' blocks from YAML
  run_blocks: List[str] = re.findall(r'run:\s*\|?\s*(.*?)(?=\n\s*\w+:|$)', workflow_yaml, re.DOTALL)
  # Also extract inline run: "command"
  run_lines: List[str] = re.findall(r'run:\s*["\']?([^\n"\']+)["\']?', workflow_yaml)

  all_cmds: List[str] = []
  for block in run_blocks:
    for line in block.strip().splitlines():
      cmd = line.strip().lstrip("- ").strip()
      if cmd and not cmd.startswith("#"):
        all_cmds.append(cmd)
  for line in run_lines:
    cmd = line.strip()
    if cmd and cmd not in all_cmds:
      all_cmds.append(cmd)

  _install_pats = re.compile(
    r'(pip install|pip3 install|poetry install|pipenv install|'
    r'conda install|npm install|yarn install|gem install|'
    r'bundle install|cargo install|go get|apt-get install|'
    r'apt install|brew install|setup\.py|python setup|'
    r'pip install -e|pip install -r)',
    re.IGNORECASE,
  )
  _validate_pats = re.compile(
    r'(pytest|python -m pytest|py\.test|unittest|'
    r'npm test|yarn test|jest|mocha|'
    r'mypy|flake8|pylint|ruff|black --check|isort --check|'
    r'cargo test|go test|mvn test|gradle test|'
    r'make test|make check|tox)',
    re.IGNORECASE,
  )

  seen_install:  set = set()
  seen_validate: set = set()

  for cmd in all_cmds:
    if _install_pats.search(cmd) and cmd not in seen_install:
      installation_cmd.append(cmd)
      seen_install.add(cmd)
    elif _validate_pats.search(cmd) and cmd not in seen_validate:
      validation_cmd.append(cmd)
      seen_validate.add(cmd)

  logger.debug(
    "[Phase B] YAML fallback: install_cmds=%d validation_cmds=%d",
    len(installation_cmd), len(validation_cmd),
  )
  return {
    "installation_cmd": installation_cmd[:6],
    "validation_cmd":  validation_cmd[:6],
    "critical_steps":  critical_steps,
  }


def _coerce_str_list(value: Any) -> List[str]:
  """Coerce any value to a deduplicated list of non-empty strings."""
  if isinstance(value, list):
    seen: set = set()
    out: List[str] = []
    for x in value:
      s = str(x).strip()
      if s and s not in seen:
        seen.add(s)
        out.append(s)
    return out
  s = str(value or "").strip()
  return [s] if s else []


# ══════════════════════════════════════════════════════════════════════════════
# Phase D — CI document assembly helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clip(text: str, limit: int = 400) -> str:
  text = (text or "").strip()
  return text if len(text) <= limit else text[:limit - 3] + "..."


def _extract_structured_keywords(text: str, limit: int = 12) -> List[str]:
  candidates: List[str] = []
  for match in re.finditer(r"[A-Za-z0-9_./:-]{3,}", text or ""):
    token = match.group(0).strip()
    lower = token.lower()
    if lower.startswith(("http://", "https://")) or token.isdigit():
      continue
    if "/" in token or "." in token or "_" in token or "::" in token:
      candidates.append(token)
      continue
    if any(ch.isupper() for ch in token[1:]):
      candidates.append(token)
      continue
    if lower in {"error", "errors", "failed", "failure", "traceback", "exception", "warning"}:
      candidates.append(token)
  seen = set()
  ordered: List[str] = []
  for token in candidates:
    if token not in seen:
      seen.add(token)
      ordered.append(token)
    if len(ordered) >= limit:
      break
  return ordered


def _looks_like_structured_code(line: str) -> bool:
  code_markers = (
    "Traceback",
    "File \"",
    " line ",
    "Error:",
    "Exception",
    "AssertionError",
    "FAILED",
    "E  ",
    "def ",
    "class ",
    "import ",
    "from ",
    "return ",
    "raise ",
    "pytest",
    "mypy",
    "ruff",
    "flake8",
  )
  if any(marker in line for marker in code_markers):
    return True
  return bool(re.search(r"[A-Za-z0-9_./-]+\.(py|js|ts|tsx|jsx|java|go|rb|php|yml|yaml|json|toml|ini|cfg)(:\d+)?", line))


def _extract_structured_code_context(text: str, limit: int = 5) -> List[str]:
  lines: List[str] = []
  for raw_line in (text or "").splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if len(line) > 160:
      line = line[:157] + "..."
    if _looks_like_structured_code(line):
      lines.append(line)
    if len(lines) >= limit:
      break
  return lines


def _summarize_unstructured_log_text(
  text: str,
  *,
  token_keywords: List[str],
  code_context: List[str],
) -> str:
  line_count = len([line for line in (text or "").splitlines() if line.strip()])
  parts: List[str] = []
  if line_count:
    parts.append(f"log captured across {line_count} non-empty lines")
  if token_keywords:
    parts.append("keywords: " + ", ".join(token_keywords[:8]))
  if code_context:
    parts.append("code signals: " + " | ".join(code_context[:2]))
  return "; ".join(parts)


def _fmt_list(items: List[str], limit: int = 500) -> str:
  lines = [f" - {_clip(str(x), limit)}" for x in (items or []) if str(x).strip()]
  return "\n".join(lines) if lines else " (none)"


def _fmt_failed_jobs(jobs: List[Dict[str, Any]]) -> str:
  lines = []
  for j in (jobs or []):
    step = str(j.get("step") or "").strip()
    cmd = str(j.get("command") or j.get("failed_command") or "").strip()
    job_ = str(j.get("job") or "").strip()
    prefix = f"[{job_}] " if job_ else ""
    if step and cmd:
      lines.append(f" - {prefix}step `{step}` -> `{cmd}`")
    elif cmd:
      lines.append(f" - {prefix}`{cmd}`")
    elif step:
      lines.append(f" - {prefix}step `{step}`")
  return "\n".join(lines) if lines else " (none)"


def _fmt_effected_files(files: List[Dict[str, Any]]) -> str:
  lines = []
  for f in (files or []):
    path  = str(f.get("file") or "").strip()
    if not path:
      continue
    reason = str(f.get("reason") or "").strip()
    etype = str(f.get("issue_type") or f.get("error_type") or "").strip()
    cmd  = str(f.get("failed_cmd") or f.get("failed_command") or "").strip()
    detail = " — " + _clip(": ".join(x for x in [etype, reason] if x), 200) if (etype or reason) else ""
    cmd_p = f" (cmd: `{_clip(cmd, 80)}`)" if cmd else ""
    lines.append(f" - `{path}`{detail}{cmd_p}")
  return "\n".join(lines) if lines else " (none)"


def _fmt_validation_cmds(profile: Dict[str, Any]) -> str:
  cmds = _coerce_str_list(profile.get("validation_cmd"))
  if not cmds:
    return " (not detected — inspect workflow YAML manually)"
  parts = []
  for c in cmds[:8]:
    parts.append(f" ```bash\n {c}\n ```")
  return "\n".join(parts)


def _fmt_install_cmds(profile: Dict[str, Any]) -> str:
  cmds = _coerce_str_list(profile.get("installation_cmd"))
  if not cmds:
    return ""
  return "\n".join(f" {_clip(c, 200)}" for c in cmds[:6])


def _fmt_chunk_summaries(log_analysis: Dict[str, Any]) -> str:
  chunks = log_analysis.get("chunk_summaries") or []
  lines = []
  for chunk in chunks[:8]:
    if not isinstance(chunk, dict):
      continue
    step_name = str(chunk.get("step_name") or "").strip()
    chunk_idx = chunk.get("chunk_index")
    chunk_total = chunk.get("chunk_total")
    summary = str(chunk.get("summary") or "").strip()
    token_count = chunk.get("chunk_token_count")
    keywords = [str(x).strip() for x in (chunk.get("token_keywords") or []) if str(x).strip()]
    code_context = [str(x).strip() for x in (chunk.get("code_context") or []) if str(x).strip()]
    if not any([step_name, summary, keywords, code_context]):
      continue
    prefix = f"{step_name}" if step_name else "chunk"
    if chunk_idx and chunk_total:
      prefix += f" [{chunk_idx}/{chunk_total}]"
    detail = summary or "No summary"
    if token_count:
      detail += f" (tokens: {token_count})"
    if keywords:
      detail += f" | keywords: {', '.join(keywords[:6])}"
    if code_context:
      detail += f" | code: {' | '.join(_clip(line, 100) for line in code_context[:2])}"
    lines.append(f" - {detail}" if prefix == "chunk" else f" - {prefix}: {detail}")
  return "\n".join(lines) if lines else " (none)"


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Strip embeddings from memory to save tokens
# ══════════════════════════════════════════════════════════════════════════════

def _strip_embeddings_from_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
  """
  Remove _embedding fields from memory retrieval results to avoid wasting tokens.

  Embeddings are 384-512 dimensional float arrays only needed for cosine similarity
  search. Once we've retrieved the top matches, we don't need to send embeddings
  to the LLM for analysis or repair planning.

  This can save 20-50K tokens per request depending on how many memories were retrieved.
  """
  import copy

  def strip_embedding(obj):
    """Recursively remove _embedding keys from dicts and lists."""
    if isinstance(obj, dict):
      return {k: strip_embedding(v) for k, v in obj.items() if k != "_embedding"}
    elif isinstance(obj, list):
      return [strip_embedding(item) for item in obj]
    else:
      return obj

  return strip_embedding(copy.deepcopy(memory))


# ══════════════════════════════════════════════════════════════════════════════
# Public: build_problem_statement
# ══════════════════════════════════════════════════════════════════════════════

def build_problem_statement(
  context: Dict[str, Any],
  memory: Dict[str, Any],
) -> str:
  """
  Phase D — Assemble the full CI document from Phase A/B/C output.

  Sections (in order):
   1. Header: repo, sha_fail, workflow
   2. Why the CI Failed (Phase A: overall_failure_reasons)
   3. Failed Jobs / Commands (Phase A: failed_jobs)
   4. Affected Files (Phase A: effected_files)
   5. Validation Hints (Phase B: validation_cmd)
   6. Installation Hints (Phase B: installation_cmd)
   7. Memory Context (Phase C: LLM-selected only)
  """
  profile  = context.get("workflow_profile") or {}
  mem_block = format_memory_context(memory)
  is_fallback = context.get("_fallback", False) or (
    context.get("_log_analysis") or {}
  ).get("_fallback", False)

  lines = [
    "# CI Failure Report",
    "",
    f"**Repository**: `{context.get('repo', '')}`",
    f"**Failing Commit (sha_fail)**: `{context.get('sha_fail', '')}`",
    f"**Workflow**: `{context.get('workflow_name') or context.get('workflow_path', '')}`",
  ]

  error_types = context.get("overall_error_types") or []
  if error_types:
    lines += ["", f"**Error categories**: {', '.join(error_types[:6])}"]

  overall_ci_summary = str((context.get("_log_analysis") or {}).get("overall_ci_summary") or "").strip()
  if overall_ci_summary:
    lines += ["", "## Overall CI Summary", f" - {_clip(overall_ci_summary, 2000)}"]

  lines += [
    "",
    "## Why the CI Failed",
    _fmt_list(context.get("overall_failure_reasons") or []),
    "",
    "## Failed Jobs / Commands",
    _fmt_failed_jobs(context.get("failed_jobs") or []),
    "",
    "## Affected Files",
    _fmt_effected_files(context.get("effected_files") or []),
    "",
    "## Validation Hints",
    "These are candidate commands the agent may run, adapt, or ignore depending on local feasibility.",
    _fmt_validation_cmds(profile),
  ]

  install_block = _fmt_install_cmds(profile)
  if install_block:
    lines += [
      "",
      "## Installation Hints",
      "These are candidate setup commands the agent may extend with extra local preparation if needed.",
      install_block,
    ]

  chunk_block = _fmt_chunk_summaries(context.get("_log_analysis") or {})
  if chunk_block and chunk_block != " (none)":
    lines += ["", "## CI Chunk Evidence", chunk_block]

  if mem_block:
    lines += ["", mem_block]

  # Add repair plan if available
  repair_plan = memory.get("repair_plan")
  if repair_plan and not repair_plan.get("error"):
    plan_lines = [
      "",
      "## Suggested Repair Plan",
      "",
      f"**Problem Statement**: {repair_plan.get('problem_statement', '')}",
      "",
    ]

    root_causes = repair_plan.get("root_causes", [])
    if root_causes:
      plan_lines += ["### Root Causes"]
      for rc in root_causes:
        plan_lines.append(f"- **Cause**: {rc.get('cause', '')}")
        plan_lines.append(f" - Evidence: {rc.get('evidence', '')}")
        plan_lines.append(f" - Validation Stage: {rc.get('validation_stage', '?')}")
      plan_lines.append("")

    repair_steps = repair_plan.get("repair_steps", [])
    if repair_steps:
      plan_lines += ["### Repair Steps (in order)"]
      for step in repair_steps:
        step_id = step.get("step_id", "?")
        plan_lines.append(f"#### Step {step_id}")
        plan_lines.append(f"- **What to Fix**: {step.get('what_to_fix', '')}")
        plan_lines.append(f"- **Where to Fix**: {step.get('where_to_fix', '')}")
        plan_lines.append(f"- **How to Fix**: {step.get('how_to_fix', '')}")
        plan_lines.append(f"- **Why This Fixes It**: {step.get('why_this_fixes', '')}")
        plan_lines.append(f"- **Verify By**: {step.get('verify_by', '')}")
        if step.get("depends_on"):
          plan_lines.append(f"- **Depends On**: Steps {', '.join(map(str, step['depends_on']))}")
        plan_lines.append("")

    verification_order = repair_plan.get("verification_order", [])
    if verification_order:
      plan_lines.append(f"**Verification Order**: {' -> '.join(map(str, verification_order))}")
      plan_lines.append("")

    complexity = repair_plan.get("estimated_complexity", "unknown")
    plan_lines.append(f"**Estimated Complexity**: {complexity}")

    lines += plan_lines

  if is_fallback:
    lines += [
      "",
      "> **Note**: Structured log analysis was unavailable.",
      "> The failure text above is a raw excerpt from CI logs.",
    ]

  return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_ci_context(
  instance: Dict[str, Any],
  *,
  memory_root: Optional[str] = None,
  memory_enabled: bool = True,
  memory_top_k: int = 3,
  memory_ablation_levels: str = "L1+L2+L3",
  memory_plugin_path: Optional[str] = None,
  model: str = "gpt-4o-mini",
  llm: Any = None,
) -> Dict[str, Any]:
  """
  Full CI pre-processing pipeline (Phases A -> B -> C -> D).

  Parameters
  ----------
  instance
    CI benchmark instance dict with keys:
      instance_id / id, sha_fail, repo_owner, repo_name,
      workflow (YAML text), workflow_path, workflow_name,
      logs (str | list[{step_name, log}] | {steps: [...]})

  memory_root
    Directory for shared MemoryPlugin JSON files
    (failure_memory.json, repo_memory.json, cross_memory.json).
    Created automatically if it does not exist.
    Required when memory_enabled=True; uses a temp dir otherwise.

  memory_enabled
    Whether to run Phase C (memory retrieval + LLM selection).

  memory_top_k
    Top-k records to retrieve per memory level in Steps 2–3.

  memory_ablation_levels
    Active memory levels: "L1", "L1+L2", or "L1+L2+L3".

  memory_plugin_path
    Explicit path to memory_plugin.py if not importable normally
    (i.e. CI-REPAIR-BENCH is not on sys.path).

  model
    LLM model name — used for token estimation and Phase B (ErrorContextAgent).

  llm
    LLM client passed to:
     • Phase A (CILogAnalyzer) — for log summarisation
     • Phase C (CIMemorySystem) — for Step-5 LLM relevance gate
     Supports: LangChain ChatModel, memci LLMClient, Any callable(prompt)->str

  Returns
  -------
  {
    "context":      dict,  # Phase A + B output (mapped schema)
    "memory":      dict,  # Phase C output (raw + LLM-selected)
    "problem_statement": str,  # Phase D output (ready for agent.run())
  }
  """
  # Resolve memory root
  if memory_enabled and memory_root is None:
    logger.warning(
      "[CIContext] memory_enabled=True but memory_root=None; "
      "using a temporary directory (memory will not persist)."
    )
    memory_root = tempfile.mkdtemp(prefix="mini_swe_mem_")

  effective_memory_root = memory_root or tempfile.mkdtemp(prefix="mini_swe_mem_")
  os.makedirs(effective_memory_root, exist_ok=True)

  repo = str(
    instance.get("repo")
    or f"{instance.get('repo_owner','')}/{instance.get('repo_name','')}".strip("/")
  )
  logger.info(
    "[CIContext] START sha=%s repo=%s memory=%s levels=%s llm=%s",
    str(instance.get("sha_fail") or "")[:12],
    repo,
    memory_enabled,
    memory_ablation_levels,
    type(llm).__name__ if llm else "None",
  )

  # ── Phase A — Log Summarisation (CILogAnalyzer) ───────────────────────────
  log_analysis = _run_log_analysis(instance, llm=llm, model=model)
  logger.info(
    "[Phase A] done — error_types=%d files=%d jobs=%d",
    len(log_analysis.get("error_types") or []),
    len(log_analysis.get("relevant_files") or []),
    len((log_analysis.get("failed_job") or log_analysis.get("failed_jobs")) or []),
  )

  # ── Phase B — Workflow Profile Extraction ─────────────────────────────────
  workflow_profile = _extract_workflow_profile(
    instance,
    memory_root=effective_memory_root,
    memory_enabled=memory_enabled,
    model=model,
    llm=llm,
  )
  logger.info(
    "[Phase B] done — install_cmds=%d validation_cmds=%d",
    len(workflow_profile.get("installation_cmd") or []),
    len(workflow_profile.get("validation_cmd") or []),
  )

  # ── Build context dict (Phase A output + Phase B workflow_profile) ────────
  context = _log_analysis_to_context(log_analysis, instance, workflow_profile)

  # ── Phase C — Memory Retrieval (CIMemorySystem — Steps 1–5) ──────────────
  memory_system = CIMemorySystem.create(
    effective_memory_root if memory_enabled else None,
    memory_enabled=memory_enabled,
    memory_top_k=memory_top_k,
    memory_ablation_levels=memory_ablation_levels,
    memory_plugin_path=memory_plugin_path,
  )

  memory = memory_system.build_and_retrieve(
    log_analysis_result=log_analysis,
    instance=instance,
    llm=llm,
    validation_sequence=workflow_profile.get("validation_sequence", []),
  )

  logger.info(
    "[Phase C] done — weighted_sim=%.3f selected=%d use_memory=%s",
    memory.get("weighted_similarity", 0.0),
    len((memory.get("llm_selection") or {}).get("selected_items") or []),
    (memory.get("llm_selection") or {}).get("use_memory", False),
  )

  # ── Phase D — Document Assembly ──────────────────────────────────────────
  # Single production direction:
  #   Phase C two-prompt memory synthesis -> agent_problem_statement -> repair agent.
  # If memory/LLM synthesis is unavailable, fall back to the standard CI report.
  agent_problem_statement = format_memory_context(memory).strip()
  problem_statement = agent_problem_statement or build_problem_statement(context, memory)

  logger.info(
    "[CIContext] DONE sha=%s problem_statement=%d chars",
    str(context.get("sha_fail") or "")[:12],
    len(problem_statement),
  )

  return {
    "context":      context,
    "memory":      memory,
    "problem_statement": problem_statement,
  }


# ══════════════════════════════════════════════════════════════════════════════
# Post-patch memory persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_memory_after_patch(
  instance: Dict[str, Any],
  context: Dict[str, Any],
  diff: str,
  *,
  memory_root: str,
  memory_plugin_path: Optional[str] = None,
  llm: Any = None,
) -> None:
  """
  Persist L1/L2/L3 memory entries after a successful patch.

  Called by cibench.py after the agent submits a non-empty diff.

  L1: per-file record (file, error_type, failure_pattern, fix_direction,
             dependent_files, failure_reason)
  L2: repo-scoped pattern (aggregated from L1 files of this issue)
  L3: cross-repo principle (merged by error_type + issue_type)
  """
  if not diff or not diff.strip():
    return

  memory_system = CIMemorySystem.create(
    memory_root,
    memory_enabled=True,
    memory_plugin_path=memory_plugin_path,
  )

  # Use the raw log_analysis_result stored in context for save_memory_entry
  log_analysis_result = context.get("_log_analysis") or {
    "error_types": [
      {"category": et, "subcategory": "", "evidence": ""}
      for et in (context.get("overall_error_types") or [])
    ],
    "error_context": context.get("overall_failure_reasons") or [],
    "failed_job":  context.get("failed_jobs") or [],
    "relevant_files": [
      {"file": f["file"]}
      for f in (context.get("effected_files") or [])
      if f.get("file")
    ],
  }

  memory_system.save(
    task_id=str(context.get("id") or ""),
    sha_fail=str(context.get("sha_fail") or ""),
    repo_name=str(context.get("repo_name") or context.get("repo") or ""),
    repo_owner=str(context.get("repo_owner") or ""),
    workflow_path=str(context.get("workflow_path") or ""),
    workflow=str(instance.get("workflow") or ""),
    log_analysis_result=log_analysis_result,
    diff=diff,
    llm=llm,
  )
