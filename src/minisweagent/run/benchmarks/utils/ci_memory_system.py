"""
ci_memory_system.py
===================
Wires MemoryPlugin into the ci_context.py pre-processing pipeline.

Phase C:
 1. Build structured query from CILogAnalyzer output
 2. Embed query (sentence-transformers or fastembed)
 3. Cosine similarity search across L1 / L2 / L3 memory stores
 4. Rank candidates highest -> lowest
 5. Prompt 1 groups retrieved memory into validation-stage atomic problems
 6. Prompt 2 turns those stage findings into the ordered agent problem statement.

Design principles:
 - Keep retrieval deterministic and transparent
 - Keep LLM reasoning schema-based and validation-stage aware
 - Fall back to deterministic guidance if the synthesis call fails
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Universal LLM caller
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(llm: Any, prompt: str) -> str:
  if llm is None:
    return ""
  try:
    try:
      from langchain_core.messages import HumanMessage # type: ignore
      result = llm.invoke([HumanMessage(content=prompt)])
      return (getattr(result, "content", None) or "").strip()
    except (ImportError, AttributeError):
      pass
    result = llm.invoke(prompt)
    if hasattr(result, "content"):
      return (result.content or "").strip()
    return str(result).strip()
  except Exception:
    pass
  try:
    return str(llm(prompt)).strip()
  except Exception:
    return ""


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
  """Parse LLM JSON with comprehensive cleaning for malformed output."""
  import re

  raw = raw.strip()

  # Remove markdown fences
  raw = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', raw, flags=re.DOTALL)

  # Fix trailing commas
  raw = re.sub(r',(\s*[}\]])', r'\1', raw)

  # Fix missing commas between string values
  raw = re.sub(r'"\s+"', '", "', raw)

  # Fix double commas
  raw = re.sub(r',\s*,', ',', raw)

  # Fix missing commas between objects
  raw = re.sub(r'}\s*{', '}, {', raw)
  raw = re.sub(r'}\s*\[', '}, [', raw)
  raw = re.sub(r']\s*{', '], {', raw)

  # Extract JSON from surrounding text
  raw = raw.strip()
  start_brace = raw.find('{')
  start_bracket = raw.find('[')

  if start_brace == -1 and start_bracket == -1:
    return None

  # Find first { or [
  if start_brace == -1:
    start = start_bracket
    end_char = ']'
  elif start_bracket == -1:
    start = start_brace
    end_char = '}'
  else:
    start = min(start_brace, start_bracket)
    end_char = '}' if start == start_brace else ']'

  # Find matching closing
  end = raw.rfind(end_char)
  if end > start:
    raw = raw[start:end+1]

  # Try parsing
  try:
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else None
  except json.JSONDecodeError:
    # Log the error for debugging
    logger.debug(f"[_parse_json] Failed to parse cleaned JSON. First 200 chars: {raw[:200]}")
    return None
  except Exception as e:
    logger.debug(f"[_parse_json] Unexpected error: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Plugin import
# ─────────────────────────────────────────────────────────────────────────────

def _import_memory_plugin():
  from minisweagent.run.benchmarks.utils.memory_plugin import MemoryPlugin # type: ignore
  return MemoryPlugin


# ─────────────────────────────────────────────────────────────────────────────
# CIMemorySystem
# ─────────────────────────────────────────────────────────────────────────────

class CIMemorySystem:
  """
  Three-level memory store (L1 / L2 / L3) for CI repair.

  L1: Per-file failure records (repo + workflow scoped)
  L2: Repo-scoped recurring patterns (repo + workflow scoped)
  L3: Cross-repo generalized principles (no filtering)
  """

  def __init__(self, plugin: Any, *, memory_enabled: bool) -> None:
    self._plugin = plugin
    self._enabled = memory_enabled

  @classmethod
  def create(
    cls,
    memory_root: Optional[str],
    *,
    memory_enabled: bool = True,
    memory_top_k: int = 3,
    memory_ablation_levels: str = "L1+L2+L3",
    memory_plugin_path: Optional[str] = None,
    llm: Any = None,
  ) -> "CIMemorySystem":
    if not memory_enabled or not memory_root:
      return cls(None, memory_enabled=False)

    import os
    os.makedirs(memory_root, exist_ok=True)

    MemoryPlugin = _import_memory_plugin()
    config = {
      "memory_enabled":     True,
      "memory_top_k":      memory_top_k,
      "memory_ablation_levels": memory_ablation_levels,
      "memory_backend":     "json",
      "project_result_dir":   memory_root,
    }
    plugin = MemoryPlugin(config, memory_root, llm=llm)
    return cls(plugin, memory_enabled=True)

  @property
  def plugin(self):
    """Expose the underlying MemoryPlugin for direct access."""
    return self._plugin

  def is_enabled(self) -> bool:
    return (
      self._enabled
      and self._plugin is not None
      and self._plugin.is_enabled()
    )

  def build_and_retrieve(
    self,
    log_analysis_result: Dict[str, Any],
    *,
    instance: Optional[Dict[str, Any]] = None,
    llm: Any = None,
    validation_sequence: List[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Run Steps 1–5 of Phase C with validation-aware synthesis."""
    _empty = {
      "enabled":        False,
      "weighted_similarity":  0.0,
      "selected_memory_levels": [],
      "llm_selection": _empty_selection(),
    }

    if not self.is_enabled():
      return _empty

    instance  = instance or {}
    repo_owner = str(instance.get("repo_owner") or "")
    repo_name = str(instance.get("repo_name") or "")
    repo_full = f"{repo_owner}/{repo_name}".strip("/")

    try:
      query = self._plugin.build_query(
        repo_name=repo_full,
        workflow_path=str(instance.get("workflow_path") or ""),
        log_analysis_result=log_analysis_result,
        changed_files_info=None,
      )
    except Exception as exc:
      logger.error("[CIMemorySystem] build_query failed: %s", exc)
      return _empty

    try:
      raw = self._plugin.retrieve(query)

      # CLEAN EMBEDDINGS ONCE - right after retrieval, before any downstream processing
      # This ensures all synthesis functions (organize, reason, prompts) get clean data
      if raw.get("matches"):
        logger.info(f"[CIMemorySystem] Cleaning embeddings from {len(raw['matches'])} retrieved memories")
        raw["matches"] = [_clean_memory_for_llm(m) for m in raw["matches"]]
        raw["l1_matches"] = [_clean_memory_for_llm(m) for m in raw.get("l1_matches", [])]
        raw["l2_matches"] = [_clean_memory_for_llm(m) for m in raw.get("l2_matches", [])]
        raw["l3_matches"] = [_clean_memory_for_llm(m) for m in raw.get("l3_matches", [])]

    except Exception as exc:
      logger.error("[CIMemorySystem] retrieve failed: %s", exc)
      return _empty

    effective_llm = llm or self._plugin.llm
    llm_selection = _run_two_llm_gate(raw, effective_llm, validation_sequence)

    return {**raw, "llm_selection": llm_selection}

  def save(
    self,
    *,
    task_id: str,
    sha_fail: str,
    repo_name: str,
    repo_owner: str,
    workflow_path: str,
    workflow: str,
    log_analysis_result: Dict[str, Any],
    diff: str,
    fault_localizer: Optional[Dict[str, Any]] = None,
    llm: Any = None,
  ) -> None:
    if not self.is_enabled() or not diff:
      return
    try:
      self._plugin.save_memory_entry(
        task_id=task_id,
        sha_fail=sha_fail,
        repo_name=repo_name,
        repo_owner=repo_owner,
        workflow_path=workflow_path,
        workflow=workflow,
        log_analysis_result=log_analysis_result,
        changed_files_info=None,
        fault_localizer=fault_localizer,
        patch_generator={"diff": diff},
      )
    except Exception as exc:
      logger.error("[CIMemorySystem] save_memory_entry failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _identify_failed_stage(failed_cmd: List[str], validation_sequence: List[Dict[str, Any]]) -> str:
  """Identify which validation stage failed based on failed command."""
  if not failed_cmd or not validation_sequence:
    return "unknown"

  failed_cmd_str = " ".join(failed_cmd) if isinstance(failed_cmd, list) else str(failed_cmd)

  for stage in validation_sequence:
    val_cmd = stage.get("validation_cmd", "")
    if val_cmd and val_cmd in failed_cmd_str:
      return stage.get("validates", "unknown")

  return "unknown"


def _get_current_failure_stage(failed_cmd: List[str], validation_sequence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
  """
  Get full validation stage info for current failure.
  Returns: {order, validates, validation_cmd} or None
  """
  if not failed_cmd or not validation_sequence:
    return None

  cmd_str = " ".join(str(c) for c in failed_cmd).lower()

  for val in validation_sequence:
    val_cmd = (val.get("validation_cmd") or "").lower()
    if val_cmd and val_cmd in cmd_str:
      return {
        "order": val.get("order", 999),
        "validates": val.get("validates", ""),
        "validation_cmd": val.get("validation_cmd", "")
      }

  # Fallback: keyword-based inference
  keyword_map = [
    (["ruff", "flake8", "pylint"], "lint"),
    (["mypy", "pyright"], "type"),
    (["pytest", "unittest"], "test"),
    (["black", "isort"], "format")
  ]

  for keywords, category in keyword_map:
    if any(kw in cmd_str for kw in keywords):
      for val in validation_sequence:
        if category in val.get("validates", "").lower():
          return {
            "order": val.get("order", 999),
            "validates": val.get("validates", ""),
            "validation_cmd": val.get("validation_cmd", "")
          }

  return None


def _get_consecutive_stages(current_order: int, validation_sequence: List[Dict[str, Any]], max_ahead: int = 10) -> List[Dict[str, Any]]:
  """
  Get validation stages that come AFTER current stage.
  These will run after current failure is fixed.
  """
  consecutive = []
  for val in validation_sequence:
    order = val.get("order", 999)
    if current_order < order <= current_order + max_ahead:
      consecutive.append({
        "order": order,
        "validates": val.get("validates", ""),
        "validation_cmd": val.get("validation_cmd", "")
      })

  return sorted(consecutive, key=lambda x: x["order"])


def _get_validation_cmd(stage_name: str, validation_sequence: List[Dict[str, Any]]) -> str:
  """Get validation_cmd for a given stage name."""
  for val in validation_sequence:
    if val.get("validates", "") == stage_name:
      return val.get("validation_cmd", "")
  return ""


def _map_memory_to_validation_stage(memory: Dict[str, Any], validation_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
  """
  Map memory's failed_cmd to validation stage from sequence.
  Returns: {validates, order, validation_cmd}
  """
  failed_cmds = memory.get("failed_cmd", [])
  if isinstance(failed_cmds, str):
    failed_cmds = [failed_cmds]

  # Try exact matching first
  for cmd in failed_cmds:
    cmd_normalized = str(cmd).lower().strip()
    for val in validation_sequence:
      val_cmd = (val.get("validation_cmd") or "").lower().strip()
      if val_cmd and val_cmd in cmd_normalized:
        return {
          "validates": val.get("validates", ""),
          "order": val.get("order", 999),
          "validation_cmd": val.get("validation_cmd", "")
        }

  # Fallback: keyword-based inference
  cmd_str = " ".join(str(c) for c in failed_cmds).lower()

  keyword_map = {
    "ruff": "lint", "flake8": "lint", "pylint": "lint",
    "mypy": "type", "pyright": "type",
    "pytest": "test", "unittest": "test",
    "black": "format", "isort": "format"
  }

  for keyword, category in keyword_map.items():
    if keyword in cmd_str:
      for val in validation_sequence:
        validates = val.get("validates", "").lower()
        if category in validates:
          return {
            "validates": val.get("validates", ""),
            "order": val.get("order", 999),
            "validation_cmd": val.get("validation_cmd", "")
          }

  return {"validates": "Unknown", "order": 999, "validation_cmd": ""}


def _extract_files_from_memory(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
  """Extract file information from memory in standardized format."""
  files = []

  file_data = memory.get("files") or memory.get("modified_files") or memory.get("file") or []

  if isinstance(file_data, str):
    files.append({"path": file_data})
  elif isinstance(file_data, list):
    for f in file_data:
      if isinstance(f, dict):
        files.append({
          "path": f.get("file") or f.get("path") or "",
          "reason": f.get("reason") or "",
          "fix": f.get("fix") or ""
        })
      elif isinstance(f, str):
        files.append({"path": f})
  elif isinstance(file_data, dict):
    files.append({
      "path": file_data.get("file") or file_data.get("path") or "",
      "reason": file_data.get("reason") or "",
      "fix": file_data.get("fix") or ""
    })

  return files


def _extract_identification_criteria(memory: Dict[str, Any]) -> List[str]:
  """Extract or generate identification criteria from memory."""
  criteria = []

  existing = memory.get("identification_criteria") or memory.get("grep_pattern") or []
  if existing:
    criteria.extend(existing if isinstance(existing, list) else [existing])

  error_type = memory.get("error_type") or memory.get("issue_type") or ""
  if error_type:
    criteria.append(f"Error type: {error_type}")

  problem = memory.get("problem") or memory.get("reason") or ""
  if problem and len(problem) < 200:
    criteria.append(f"Look for: {problem}")

  return criteria[:5]


def _organize_memories_by_validation_stage(
  memories: List[Dict[str, Any]],
  validation_sequence: List[Dict[str, Any]],
  current_failed_order: int = 0
) -> Dict[str, Any]:
  """
  Group memories by validation stage, ordered by sequence.

  CRITICAL FILTERING:
  - EXCLUDES stages < current_failed_order (earlier stages passed)
  - EXCLUDES stages == current_failed_order (current CI output already shows ALL errors)
  - INCLUDES stages > current_failed_order (ONLY next stages for predictions)

  Why EXCLUDE current stage?
  - CI output already shows ALL errors for the failed validation
  - Example: If ruff fails, CI shows F632, F401, F841 - we have them all
  - We don't need memory to find more errors in the same stage
  - Memory is ONLY for predicting problems in NEXT stages

  Args:
    current_failed_order: Order of current failed stage (e.g., 11)
                         Only organize memories from NEXT stages (order > 11)

  Returns: {
    "organized_by_stage": [
      {order, validates, validation_cmd, problems: [...]},
      ...
    ]
  }
  """
  by_stage = {}

  # Note: memories are already cleaned in build_and_retrieve() before reaching here
  for mem in memories[:30]:
    stage_info = _map_memory_to_validation_stage(mem, validation_sequence)
    stage_name = stage_info["validates"]
    stage_order = stage_info["order"]

    # FILTER: ONLY include NEXT stages (after current failed stage)
    # - Skip earlier stages (they passed!)
    # - Skip current stage (CI already shows ALL errors)
    # - Only keep next stages (for Problem #N+ predictions)
    if stage_order <= current_failed_order:
      continue

    if stage_name not in by_stage:
      by_stage[stage_name] = {
        "order": stage_order,
        "validates": stage_name,
        "validation_cmd": stage_info["validation_cmd"],
        "problems": []
      }

    # Extract repair_trajectory for pattern analysis (already cleaned upstream)
    repair_trajectory = mem.get("repair_trajectory", [])

    problem = {
      "pattern": mem.get("error_type") or mem.get("issue_type") or mem.get("failure_pattern"),
      "error_description": mem.get("problem") or mem.get("reason") or "",
      "fix_strategy": mem.get("fix_strategy") or mem.get("fix_approach") or "",
      "files": _extract_files_from_memory(mem),
      "from_memory": f"{mem.get('memory_level', 'L?')}_similarity_{mem.get('similarity_score', 0):.3f}",
      "identification_criteria": _extract_identification_criteria(mem),
      "repair_trajectory": repair_trajectory  # Include for pattern analysis
    }

    by_stage[stage_name]["problems"].append(problem)

  organized = sorted(by_stage.values(), key=lambda x: x["order"])

  total_problems = sum(len(s["problems"]) for s in organized)
  return {
    "organized_by_stage": organized,
    "summary": f"Organized {total_problems} problems across {len(organized)} validation stages (ONLY next stages, current stage errors already from CI)"
  }


def _extract_current_failure_as_problem_1(memory_result: Dict[str, Any], validation_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
  """
  Extract current CI failure as Problem #1 (COMPLETE context, all errors together).

  IMPORTANT:
  - Problem #1 = the ENTIRE current CI failure (one validation, multiple errors)
  - All errors from the failed validation are grouped together
  - Example: Ruff failed with F632, F401, F841 → ONE problem with 3 file-level errors
  - Memory is ONLY used for predicting Problem #2+ (NEXT validation stages)

  Returns: Single problem dict with all CI context
  """
  q = memory_result.get("query", {})

  current_stage = _get_current_failure_stage(q.get("failed_cmd", []), validation_sequence)

  if not current_stage:
    current_stage = {"order": 0, "validates": "Unknown", "validation_cmd": ""}

  # Extract ALL file-level errors from query_file_details
  query_file_details = memory_result.get("query_file_details", [])

  files = []
  error_types = []

  if query_file_details:
    # Collect ALL errors from the failed validation with FULL details
    for detail in query_file_details:
      issue_type = detail.get("issue_type", "")
      reason = detail.get("reason", "")

      # Extract code context if available
      code_snippet = detail.get("code_snippet", "") or detail.get("code_context", "")

      # Build actionable fix description
      fix_desc = ""
      if issue_type == "F632":
        fix_desc = "Replace `is` with `==` when comparing to literal values"
      elif issue_type == "F401":
        fix_desc = "Remove the unused import statement"
      elif issue_type == "F841":
        fix_desc = "Remove the unused variable or use it"
      else:
        # Generic fix based on reason
        fix_desc = f"Fix: {reason}" if reason else "Apply the fix described in the error"

      files.append({
        "path": detail.get("file", ""),
        "line": str(detail.get("line_number", "")),
        "issue_type": issue_type,
        "reason": reason,  # Full error message from CI
        "current_code": code_snippet,
        "required_fix": fix_desc
      })

      if issue_type:
        error_types.append(issue_type)
  else:
    # No file details - use relevant files from query
    for f in q.get("relevant_files", []):
      files.append({
        "path": f,
        "issue_type": q.get("error_type", ""),
        "reason": q.get("failure_pattern", ""),
        "current_code": "",
        "required_fix": "See CI logs for specific fix"
      })

  # Build error summary (unique error types)
  unique_errors = list(dict.fromkeys(error_types))  # Preserve order, remove duplicates
  error_summary = ", ".join(unique_errors) if unique_errors else q.get("error_type", "")

  # Build detailed error description
  error_description = q.get("failure_pattern", "")
  if query_file_details and len(query_file_details) > 1:
    error_description = f"Multiple errors detected: {len(query_file_details)} issues found. " + error_description

  # Build comprehensive fix strategy
  fix_strategy = f"Fix ALL {len(files)} error(s) in this validation stage. "
  if len(files) > 1:
    fix_strategy += "Each file has specific issues listed below. Address them one by one. "
  fix_strategy += f"After fixing, run: `{current_stage['validation_cmd']}` to verify all issues are resolved."

  return {
    "problem_number": 1,
    "status": "confirmed",
    "validation_order": current_stage["order"],
    "validation_stage": current_stage["validates"],
    "validation_cmd": current_stage["validation_cmd"],
    "error_type": error_summary,  # Unique error types (e.g., "F632, F401, F841")
    "error_description": error_description,  # Detailed description with count
    "root_cause": q.get("overall_failure_reason", ""),
    "files": files,  # All affected files with full details
    "fix_strategy": fix_strategy,
    "verification_cmd": current_stage["validation_cmd"],
    "check_first": False,
    "reasoning": f"CONFIRMED from CI logs: {current_stage['validates']} validation failed with {len(files)} error(s). This is the current blocking issue - must fix ALL errors in this stage before CI will proceed to next validation stage.",
    "identification_criteria": [],
    "interdependency": f"Current validation failure at stage {current_stage['order']} ({current_stage['validates']}). All {len(files)} error(s) must be fixed before CI runs the next validation stage. This is blocking further progress.",
    "from_memory": "",
    "confidence": "confirmed",
    "total_errors": len(files),  # How many individual errors in this validation
    "error_details": [
      {
        "file": f.get("path", ""),
        "line": f.get("line", ""),
        "error": f.get("issue_type", ""),
        "message": f.get("reason", ""),
        "fix": f.get("required_fix", "")
      }
      for f in files
    ]  # Structured list for easy parsing
  }


def _analyze_interdependency(problem_1: Dict[str, Any], next_stage: Dict[str, Any]) -> str:
  """Analyze interdependency between current problem and next stage."""
  current_stage = problem_1.get("validation_stage", "").lower()
  next_stage_name = next_stage.get("validates", "").lower()

  if "lint" in current_stage and "type" in next_stage_name:
    return f"Problem #1 ({problem_1['validation_stage']}) blocks {next_stage['validates']} from running. Must fix #1 first."

  if "type" in current_stage and "test" in next_stage_name:
    return f"Type errors in Problem #1 prevent tests from importing modules. Must fix #1 first."

  if "format" in current_stage and "lint" in next_stage_name:
    return f"Format errors in Problem #1 may hide lint issues. Must fix #1 first."

  return f"Problem #1 must be fixed before {next_stage['validates']} runs (sequential validation order)."


def _analyze_repair_trajectory_patterns(problem_1: Dict[str, Any], organized_by_stage: List[Dict[str, Any]]) -> Dict[str, Any]:
  """
  Analyze repair_trajectory from memories to find NEXT-STAGE problem patterns.

  IMPORTANT:
  - We ONLY analyze what came NEXT in different validation stages
  - We do NOT analyze same-stage patterns (CI already shows all errors)
  - Example: When F632 was fixed, what failed in mypy? In pytest?

  Returns: {
    "next_stage_problems": [
      {"validation_stage": "Type checking", "error_type": "Union", "frequency": 3},
      {"validation_stage": "Unit tests", "error_type": "ImportError", "frequency": 2}
    ],
    "pattern_reasoning": "F632 commonly followed by Union in Type checking (3 cases)"
  }
  """
  current_error = problem_1.get("error_type", "").lower()
  current_stage = problem_1.get("validation_stage", "").lower()

  # Count: what came AFTER current problem in NEXT stages?
  next_stage_count = {}

  for stage_group in organized_by_stage:
    for prob in stage_group.get("problems", []):
      trajectory = prob.get("repair_trajectory", [])

      if not trajectory:
        continue

      # Find if current problem appears in this trajectory
      for i, step in enumerate(trajectory):
        step_error = str(step.get("error", "")).lower()
        step_stage = str(step.get("validation", "")).lower()

        # Match: similar error type in same stage as Problem #1
        if current_error and current_error in step_error and current_stage in step_stage:

          # What came NEXT in trajectory (DIFFERENT stage)?
          if i + 1 < len(trajectory):
            next_step = trajectory[i + 1]
            next_stage = next_step.get("validation", "")
            next_error = next_step.get("error", "")

            # Only count if it's a DIFFERENT validation stage
            if next_stage.lower() != current_stage:
              key = f"{next_stage}:{next_error}"
              if key not in next_stage_count:
                next_stage_count[key] = {
                  "validation_stage": next_stage,
                  "error_type": next_error,
                  "frequency": 0
                }
              next_stage_count[key]["frequency"] += 1

  # Sort by frequency
  next_stage_sorted = sorted(next_stage_count.values(), key=lambda x: -x["frequency"])

  reasoning = ""
  if next_stage_sorted:
    reasoning = f"{current_error or 'This error'} commonly followed by {next_stage_sorted[0]['error_type']} in {next_stage_sorted[0]['validation_stage']} ({next_stage_sorted[0]['frequency']} cases)"

  return {
    "next_stage_problems": next_stage_sorted[:5],  # Top 5 next-stage
    "pattern_reasoning": reasoning
  }


def _predict_consecutive_problems(problem_1: Dict[str, Any], organized_by_stage: List[Dict[str, Any]], validation_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """
  Predict problems that will appear in NEXT validation stages (AFTER current stage passes).

  IMPORTANT:
  - We DO NOT predict same-stage problems (CI already shows ALL errors for current stage)
  - We ONLY predict problems for NEXT validation stages
  - Example: If ruff fails, CI shows ALL ruff errors (F632, F401, F841)
  - We predict what will fail in mypy, pytest, etc. (next stages)

  Uses TWO sources:
  1. Validation sequence order (what stages run next)
  2. Repair trajectory patterns (what actually happened in past repairs)
  """
  current_order = problem_1.get("validation_order", 0)
  consecutive_stages = _get_consecutive_stages(current_order, validation_sequence, max_ahead=10)

  # Analyze repair trajectory patterns (ONLY for next stages)
  trajectory_analysis = _analyze_repair_trajectory_patterns(problem_1, organized_by_stage)
  next_stage_patterns = trajectory_analysis["next_stage_problems"]
  pattern_reasoning = trajectory_analysis["pattern_reasoning"]

  predicted = []
  problem_num = 2  # Start from Problem #2 (Problem #1 is current CI failure)

  # Build predictions for NEXT validation stages
  prioritized_stages = []

  # Prioritize stages from repair trajectory patterns (high confidence)
  for next_prob in next_stage_patterns:
    stage_name = next_prob["validation_stage"]
    if stage_name and stage_name not in [s["validates"] for s in prioritized_stages]:
      # Find this stage in consecutive_stages
      for stage in consecutive_stages:
        if stage["validates"].lower() == stage_name.lower():
          prioritized_stages.append({
            **stage,
            "from_trajectory": True,
            "trajectory_frequency": next_prob["frequency"],
            "trajectory_error": next_prob["error_type"]
          })
          break

  # Add remaining consecutive stages (medium confidence)
  for stage in consecutive_stages:
    if stage["validates"] not in [s["validates"] for s in prioritized_stages]:
      prioritized_stages.append({**stage, "from_trajectory": False})

  # Build next-stage predicted problems
  for stage in prioritized_stages[:5]:  # Max 5 next-stage problems
    stage_name = stage["validates"]

    stage_problems = None
    for org_stage in organized_by_stage:
      if org_stage["validates"] == stage_name:
        stage_problems = org_stage.get("problems", [])
        break

    if not stage_problems:
      continue

    interdependency = _analyze_interdependency(problem_1, stage)

    # If from trajectory: use trajectory error as primary
    if stage.get("from_trajectory"):
      # Find problem matching trajectory error
      trajectory_error = stage.get("trajectory_error", "").lower()
      matching_prob = None
      for prob in stage_problems:
        if trajectory_error in str(prob.get("pattern", "")).lower():
          matching_prob = prob
          break

      prob = matching_prob or stage_problems[0]

      reasoning = (
        f"NEXT-STAGE PATTERN: After {problem_1['validation_stage']} (order {current_order}) passes, "
        f"{stage_name} (order {stage['order']}) commonly shows this error "
        f"(appeared in {stage.get('trajectory_frequency', 0)} past repair sequences)."
      )
    else:
      # From validation sequence only
      prob = stage_problems[0]
      reasoning = (
        f"NEXT-STAGE PREDICTION: After {problem_1['validation_stage']} (order {current_order}) passes, "
        f"{stage_name} (order {stage['order']}) will run. "
        f"Memory shows this problem may appear in {stage_name}."
      )

    predicted.append({
      "problem_number": problem_num,
      "status": "probable",
      "validation_order": stage["order"],
      "validation_stage": stage_name,
      "validation_cmd": stage["validation_cmd"],
      "error_type": prob.get("pattern", ""),
      "error_description": prob.get("error_description", ""),
      "fix_strategy": prob.get("fix_strategy", ""),
      "files": prob.get("files", []),
      "identification_criteria": prob.get("identification_criteria", []),
      "verification_cmd": stage["validation_cmd"],
      "check_first": True,
      "reasoning": reasoning,
      "interdependency": interdependency,
      "from_memory": prob.get("from_memory", ""),
      "confidence": "high" if stage.get("from_trajectory") else "medium"
    })
    problem_num += 1

  return predicted


# ─────────────────────────────────────────────────────────────────────────────
# Simple helpers
# ─────────────────────────────────────────────────────────────────────────────

def _empty_selection() -> Dict[str, Any]:
  return {
    "use_memory":      False,
    "relevant_candidates":  [],
    "selected_items":    [],
    "guidance_document":   {},
    "analysis_summary":   "",
  }


def _safe_list(value: Any) -> List[Any]:
  if value is None:
    return []
  return value if isinstance(value, list) else [value]


def _normalize_path(path: Any) -> str:
  return str(path or "").strip().lstrip("/").replace("\\", "/")


def _clean_memory_for_llm(memory: Dict[str, Any]) -> Dict[str, Any]:
  """Remove irrelevant metadata (embeddings, scores, etc.) before sending to LLM."""
  # Fields to exclude - these waste tokens and aren't useful for LLM analysis
  exclude_fields = {
    "_embedding"
  }

  cleaned = {}
  for key, value in memory.items():
    if key in exclude_fields:
      continue

    # Recursively clean nested dicts
    if isinstance(value, dict):
      cleaned[key] = _clean_memory_for_llm(value)
    # Recursively clean lists of dicts
    elif isinstance(value, list):
      cleaned[key] = [
        _clean_memory_for_llm(item) if isinstance(item, dict) else item
        for item in value
      ]
    else:
      cleaned[key] = value

  return cleaned


def _build_simple_fallback(memory_result: Dict[str, Any], validation_sequence: List[Dict[str, Any]] = None) -> Dict[str, Any]:
  """
  Deterministic fallback when LLM unavailable.
  Still creates structured repair plan using helper functions!
  """
  matches = memory_result.get("matches", [])[:5]

  # Extract Problem #1 (complete current CI failure)
  problem_1 = _extract_current_failure_as_problem_1(memory_result, validation_sequence or [])

  # Start with Problem #1
  problems = [problem_1]

  # If we have memories, predict Problem #2+ for NEXT validation stages
  if matches and validation_sequence:
    current_failed_order = problem_1.get("validation_order", 0)

    # Organize memories by validation stage (ONLY NEXT stages, not current)
    organized = _organize_memories_by_validation_stage(
      matches,
      validation_sequence or [],
      current_failed_order
    )

    # Predict problems in NEXT stages based on memory patterns
    consecutive_problems = _predict_consecutive_problems(
      problem_1,
      organized.get("organized_by_stage", []),
      validation_sequence or []
    )

    # Renumber consecutive problems starting from #2
    for idx, prob in enumerate(consecutive_problems[:5], start=2):  # Max 5 next-stage problems
      prob["problem_number"] = idx

    problems.extend(consecutive_problems[:5])

  # Build markdown
  agent_statement = _format_problems_as_markdown(problems, len(problems))

  return {
    "total_problems": len(problems),
    "problems": problems,
    "agent_problem_statement": agent_statement,
    "summary": f"Deterministic plan: {len(problems)} problems (no LLM synthesis)",
    "relevant_candidates": matches
  }


def _build_organization_prompt(organized: Dict[str, Any], validation_sequence: List[Dict[str, Any]] = None) -> Optional[str]:
  """
  PROMPT 1: ORGANIZE (OPTIONAL REFINEMENT)
  Pre-organized memories already grouped by validation stage.
  Ask LLM to refine with detailed identification criteria.

  Returns None if organized is empty (skip Prompt 1).
  """
  organized_stages = organized.get("organized_by_stage", [])

  if not organized_stages:
    return None  # Skip Prompt 1 if no organized problems

  return f"""PROMPT 1: REFINE organization of problems by validation stage.

VALIDATION SEQUENCE (CI runs these checks IN ORDER):
{json.dumps(validation_sequence or [], indent=2, ensure_ascii=False)}

PRE-ORGANIZED MEMORIES (already grouped by validation stage):
{json.dumps(organized, indent=2, ensure_ascii=False)}

TASK:
Refine the organization above. For EACH problem:
1. Add DETAILED identification_criteria (grep patterns, error messages, code patterns)
2. Extract specific file details (line numbers if available from memory)
3. Ensure fix_strategy is actionable
4. Combine duplicate/similar problems within same stage

CRITICAL REQUIREMENTS:
1. Make problems IDENTIFIABLE - include grep patterns, error messages, code patterns
2. Include SPECIFIC details - line numbers, code context, exact error text
3. One validation_stage can have MULTIPLE distinct problems (different patterns)
4. Keep the same validation stage grouping and order

OUTPUT STRICT JSON:
{{
  "organized_by_stage": [
    {{
      "validation_cmd": "python -m ruff check py --no-respect-gitignore",
      "validates": "lint",
      "problems": [
        {{
          "pattern": "F632",
          "error_description": "F632: use `==` to compare constant literals (strings, bytes, None), not `is`",
          "identification_criteria": [
            "Look for comparisons like: 'is b\\\"\\\"', 'is \\\"\\\"', 'is None' with literal values",
            "Grep pattern: \\\\sis\\\\s+(b\\\\\\\"\\\\\\\"|''|None|True|False|[0-9])",
            "CI error message contains: 'use `==` to compare constant literals'"
          ],
          "files": [
            {{
              "path": "py/flwr/supernode/start_client_internal.py",
              "line": "389",
              "code_context": "while (content := object_store.get(tree.object_id)) is b\\\"\\\":",
              "specific_error": "comparing bytes literal b\\\"\\\" using `is` instead of `==`",
              "specific_fix": "Change `is b\\\"\\\"` to `== b\\\"\\\"`"
            }}
          ],
          "fix_strategy": "Replace `is` operator with `==` when comparing to literal values (strings, bytes, numbers, None)",
          "verification_pattern": "python -m ruff check <file> --select=F632",
          "from_memory": "L2_similarity_0.85"
        }}
      ]
    }}
  ],
  "summary": "Organized N problems across M validation stages from memory"
}}

RULES:
- Array structure, ordered by validation_sequence
- Include validation_cmd and validates (stage name) per stage
- DETAILED identification_criteria (grep patterns, error messages, code patterns)
- Multiple problems per stage if different patterns exist
- Extract from atomic_problems in L2 memories
- No markdown fences, just clean JSON"""


def _build_reasoning_prompt(problem_1: Dict[str, Any], organized_problems: Dict[str, Any], validation_sequence: List[Dict[str, Any]]) -> str:
  """
  PROMPT 2: REASON & BUILD REPAIR PLAN
  Build sequential repair plan with interdependency analysis.
  """
  consecutive_stages = _get_consecutive_stages(
    problem_1.get("validation_order", 0),
    validation_sequence,
    max_ahead=10
  )

  return f"""PROMPT 2: Build sequential repair plan with interdependency analysis.

═══════════════════════════════════════════════════════════════════
PROBLEM #1 (CONFIRMED - Current CI Failure):
═══════════════════════════════════════════════════════════════════
{json.dumps(problem_1, indent=2, ensure_ascii=False)}

This MUST be first in your output. It is ALREADY FAILING NOW.
You MUST copy this exactly as Problem #1 in your response.

═══════════════════════════════════════════════════════════════════
VALIDATION SEQUENCE - What Runs After Problem #1:
═══════════════════════════════════════════════════════════════════
Current failure is at: {problem_1.get('validation_stage', '?')} (order {problem_1.get('validation_order', '?')})

Consecutive stages that will run AFTER #1 is fixed:
{json.dumps(consecutive_stages, indent=2, ensure_ascii=False)}

═══════════════════════════════════════════════════════════════════
ORGANIZED PROBLEMS FROM MEMORY (grouped by validation stage):
═══════════════════════════════════════════════════════════════════
{json.dumps(organized_problems.get("organized_by_stage", []), indent=2, ensure_ascii=False)}

═══════════════════════════════════════════════════════════════════
TASK: Predict ALL problems that will fail in THIS CI run
═══════════════════════════════════════════════════════════════════

Build a sequential repair plan:

1. **Problem #1** = Current failure (given above)
   - Status: "confirmed"
   - check_first: false (already failing)
   - COPY EXACTLY from above

2. **Problem #2+** = What will fail AFTER #1 is fixed
   - Look at consecutive_stages (stages that run after current)
   - Check if organized_problems has problems for those stages
   - Analyze interdependencies
   - Filter by relevance to THIS repo
   - Status: "probable"
   - check_first: true (need to verify it exists)

REASONING PROCESS:

Step 1: Identify consecutive stages
- Current: {problem_1.get('validation_stage', '?')} (order {problem_1.get('validation_order', '?')})
- Next stages: {', '.join(s['validates'] for s in consecutive_stages[:3]) if consecutive_stages else 'None'}

Step 2: Check organized_problems for those stages
- Does memory have problems for consecutive stages?
- What patterns/errors commonly appear?

Step 3: Analyze interdependencies
- How does Problem #1 relate to Problem #2?
- Examples:
  * "Lint blocks type-check → fixing lint exposes type errors"
  * "Type errors prevent test imports → fixing types allows tests to run"

Step 4: Filter by relevance
- Does THIS repo likely have this problem?
- Is the error pattern applicable to THIS codebase?

INTERDEPENDENCY EXAMPLES:
- "Problem #1 is ruff lint (order 11). After fixing, mypy runs (order 12). Memory shows Union errors → Problem #2 = mypy Union errors."
- "Problem #1 is mypy type error (order 12). After fixing, pytest runs (order 14). Type errors prevent imports → Problem #2 = pytest ImportError."

OUTPUT STRICT JSON - DYNAMIC TEMPLATE:

Each problem MUST include these fields (adapt to ANY problem type):

{{
  "total_problems": <number>,
  "problems": [
    {{
      "problem_number": 1,
      "status": "confirmed",  // Always "confirmed" for Problem #1
      "validation_stage": "<exact validation stage name>",
      "validation_cmd": "<exact command that verifies this problem is fixed>",
      "validation_order": <number from validation sequence>,

      "problem_statement": "<DETAILED description - see requirements below>",
      "root_cause": "<WHY this problem occurs - deep analysis>",
      "fix_strategy": "<HOW to fix it - specific actionable steps>",

      "error_type": "<error category: type-error, lint-error, import-error, dependency-error, config-error, test-failure, build-error, etc>",
      "error_description": "<exact error message from CI logs>",

      "files": [
        {{
          "path": "<relative file path>",
          "line": "<line number or range>",
          "current_code": "<code that causes the error>",
          "required_fix": "<what to change it to>",
          "context": "<why this file is affected>"
        }}
      ],

      "check_first": false,  // Always false for Problem #1
      "reasoning": "<why this is Problem #1 - reference CI logs>"
    }},
    {{
      "problem_number": 2,
      "status": "probable",  // Always "probable" for Problem #2+
      "validation_stage": "<exact validation stage name>",
      "validation_cmd": "<exact verification command>",
      "validation_order": <number from validation sequence>,

      "problem_statement": "<DETAILED description - see requirements below>",
      "root_cause": "<WHY this problem occurs>",
      "fix_strategy": "<HOW to fix it>",

      "error_type": "<error category>",
      "error_description": "<predicted error based on memories>",

      "files": [
        {{
          "path": "<predicted file path from memories>",
          "line": "<predicted line or section>",
          "current_code": "<code pattern that likely fails>",
          "required_fix": "<fix pattern from memories>",
          "context": "<why this file will be affected>"
        }}
      ],

      "check_first": true,  // Always true for Problem #2+
      "reasoning": "<why this problem is predicted - reference memories + interdependencies>",
      "interdependency": "<how this relates to Problem #1 or previous problems>"
    }}
  ],
  "agent_problem_statement": "<markdown formatted repair plan for agent - see formatting requirements below>",
  "summary": "<brief summary of total problems found>"
}}

═══════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS FOR "problem_statement" FIELD:
═══════════════════════════════════════════════════════════════════

The "problem_statement" field MUST be DETAILED and SPECIFIC. It should include:

1. **Exact validation command** that fails
   Example: "Validation 'python -m mypy py' fails..."

2. **Exact file path** where error occurs
   Example: "...on file 'src/py/flwr/common/ndarrays_arithmetic.py'..."

3. **Exact line number or range**
   Example: "...at line 42..." or "...at import section (lines 15-20)..."

4. **Exact error message** from CI logs or predicted from memories
   Example: "...Error: 'Module \\"numpy.typing\\" has no attribute \\"DTypeLike\\" [attr-defined]'..."

5. **What caused the error** (immediate trigger)
   Example: "...The import statement uses DTypeLike from numpy.typing which is a private module..."

6. **Impact explanation** (why it breaks the build)
   Example: "...Mypy cannot access private modules, causing type checking to fail..."

EXAMPLE DETAILED PROBLEM STATEMENTS (for different error types):

Type Error:
"Validation 'python -m mypy py' fails on file 'src/py/flwr/common/ndarrays_arithmetic.py' at line 42. Error: 'Module \\"numpy.typing\\" has no attribute \\"DTypeLike\\" [attr-defined]'. The import statement uses DTypeLike from numpy.typing which is a private module (_typing). Mypy cannot access private modules, causing type checking to fail."

Dependency Error:
"Validation 'taplo fmt --check' fails on file 'framework/pyproject.toml' at line 77. Error: 'TOML formatting error - missing required dependency entry'. The pyproject.toml is missing a 'click' dependency entry that is required for typer/click compatibility, causing pytest import failures with 'TypeError: Secondary flag is not valid for non-boolean flag'."

Import Error:
"Validation 'pytest tests/' fails on file 'tests/test_client.py' at line 5. Error: 'ModuleNotFoundError: No module named \\"flwr.common\\"'. The test imports from flwr.common but the package is not installed in the test environment, causing test collection to fail."

Lint Error:
"Validation 'python -m ruff check src/' fails on file 'src/utils/helpers.py' at line 23. Error: 'F401 [*] `typing.Union` imported but unused'. After fixing type annotations to use | operator instead of Union, the Union import at line 23 is no longer used, triggering ruff's unused import check."

Config Error:
"Validation 'pre-commit run --all-files' fails on config file '.pre-commit-config.yaml' at hook 'mypy'. Error: 'additional_dependencies list is missing types-requests stub package'. The mypy hook configuration is missing required type stubs for the requests library, causing mypy to fail on files that import requests."

═══════════════════════════════════════════════════════════════════
REQUIREMENTS FOR "root_cause" FIELD:
═══════════════════════════════════════════════════════════════════

Explain WHY the problem occurs at a deeper level:
- What is the underlying issue? (API change, missing config, wrong assumption, etc.)
- What triggered it? (code change, dependency update, environment difference, etc.)
- Why does it break? (incompatibility, missing requirement, incorrect usage, etc.)

═══════════════════════════════════════════════════════════════════
REQUIREMENTS FOR "fix_strategy" FIELD:
═══════════════════════════════════════════════════════════════════

Explain HOW to fix it with specific actionable steps:
- What exact changes to make? (add line, remove line, replace text, update config, etc.)
- Where to make them? (specific files and lines)
- What values to use? (specific imports, specific versions, specific syntax, etc.)
- How to verify? (what command to run, what output to expect)

═══════════════════════════════════════════════════════════════════
REQUIREMENTS FOR "agent_problem_statement" FIELD:
═══════════════════════════════════════════════════════════════════

This field contains the markdown-formatted repair plan that the agent will receive.
It should be structured as:

# CI Repair Plan - Sequential Multi-Problem Fix

**IMPORTANT**: This is a MULTI-PROBLEM repair plan with N problems. You must fix them in sequence.

## Problem #1 (CONFIRMED - Currently Failing)

**Validation**: <exact command>
**File**: <path:line>
**Error**: <exact error message>

<full problem_statement text>

**Root Cause**: <root_cause text>

**Fix Strategy**: <fix_strategy text>

**Verification**: Run `<validation_cmd>` → must return exit code 0

**Status**: This problem is CONFIRMED (already failing in CI). Fix it immediately.

---

## Problem #2 (PROBABLE - Check First)

**Validation**: <exact command>
**Interdependency**: <how it relates to Problem #1>

<full problem_statement text>

**Root Cause**: <root_cause text>

**Fix Strategy**: <fix_strategy text>

**Verification**: Run `<validation_cmd>` → must return exit code 0

**Status**: This problem is PREDICTED. Check if it exists first by running the verification command.

---

## Execution Instructions

1. **Fix Problem #1**: Apply fix → run verification
2. **Check Problem #2**: Run verification to see if it exists
3. **If exists**: Apply fix → run verification
4. **Repeat for all N problems**
5. **Final verification**: Run full CI

RULES:
- Problem #1 MUST be the current CI failure (status="confirmed", check_first=false)
- Problem #2+ from organized_problems filtered by relevance (status="probable", check_first=true)
- Include specific file paths, line numbers, error messages
- Include verification commands for each problem
- agent_problem_statement is detailed markdown for the repair agent
- Include interdependency chains: how problems relate
- No markdown fences around JSON"""


def _run_single_llm_synthesis(memory_result: Dict[str, Any], llm: Any, validation_sequence: List[Dict[str, Any]] = None) -> Dict[str, Any]:
  """
  TWO-STEP LLM synthesis with proper validation-stage organization:
  1. ORGANIZE: Group memories by validation stage (deterministic + optional LLM refinement)
  2. REASON: Build sequential repair plan (current failure + consecutive problems)
  """

  # If no LLM, use simple fallback
  if llm is None:
    return _build_simple_fallback(memory_result, validation_sequence)

  # EXTRACT Problem #1 (complete current CI failure)
  problem_1 = _extract_current_failure_as_problem_1(memory_result, validation_sequence)

  if not problem_1:
    # No current failure? Return empty
    return {
      "total_problems": 0,
      "problems": [],
      "agent_problem_statement": "No current failure detected.",
      "summary": "No problems found"
    }

  matches = memory_result.get("matches", [])

  if not matches:
    # No memories: Just return Problem #1 from CI
    logger.info(f"[CIMemorySystem] No memories retrieved, returning Problem #1 from CI")

    return {
      "total_problems": 1,
      "problems": [problem_1],
      "agent_problem_statement": _format_problems_as_markdown([problem_1], 1),
      "summary": "Problem #1 from CI (no memory predictions for next stages)"
    }

  # STEP 1: ORGANIZE memories by validation stage
  logger.info("[CIMemorySystem] Step 1: Organizing memories by validation stage...")

  # Pre-organize deterministically (ONLY NEXT stages, not current - we already have all errors from CI)
  current_failed_order = problem_1.get("validation_order", 0)
  organized = _organize_memories_by_validation_stage(
    matches,
    validation_sequence,
    current_failed_order
  )

  logger.info(
    f"[CIMemorySystem] Filtering memories: only stages >= order {current_failed_order} "
    f"(current stage + next stages for predictions)"
  )

  # Optional: Ask LLM to refine (adds identification criteria, combines duplicates)
  org_prompt = _build_organization_prompt(organized, validation_sequence)
  if org_prompt:
    org_raw = _call_llm(llm, org_prompt)
    llm_organized = _parse_json(org_raw)

    if llm_organized and llm_organized.get("organized_by_stage"):
      organized = llm_organized
      logger.info(f"[CIMemorySystem] Step 1 complete: LLM refined organization")
    else:
      logger.warning("[CIMemorySystem] Step 1 LLM refinement failed, using pre-organized memories")
  else:
    logger.info("[CIMemorySystem] Step 1 complete: Using pre-organized memories (no refinement)")

  total_problems_organized = sum(len(s.get("problems", [])) for s in organized.get("organized_by_stage", []))
  logger.info(f"[CIMemorySystem] Organized {total_problems_organized} problems across {len(organized.get('organized_by_stage', []))} stages")

  # STEP 2: Build sequential repair plan
  logger.info("[CIMemorySystem] Step 2: Building sequential repair plan...")
  reason_prompt = _build_reasoning_prompt(problem_1, organized, validation_sequence)
  reason_raw = _call_llm(llm, reason_prompt)
  repair_plan = _parse_json(reason_raw)

  if not repair_plan or not repair_plan.get("problems"):
    logger.warning("[CIMemorySystem] Step 2 failed, building deterministic plan")
    # Deterministic fallback: predict consecutive problems
    consecutive_problems = _predict_consecutive_problems(
      problem_1,
      organized.get("organized_by_stage", []),
      validation_sequence
    )
    all_problems = [problem_1] + consecutive_problems

    repair_plan = {
      "total_problems": len(all_problems),
      "problems": all_problems,
      "agent_problem_statement": _format_problems_as_markdown(all_problems, len(all_problems)),
      "summary": f"Deterministic plan: {len(all_problems)} problems (LLM failed)",
      "organized_problems": organized
    }
  else:
    # Validate Problem #1 is first and confirmed
    if repair_plan["problems"] and repair_plan["problems"][0].get("status") != "confirmed":
      logger.warning("[CIMemorySystem] LLM violated rule: Problem #1 not confirmed! Fixing...")
      repair_plan["problems"][0] = problem_1

    logger.info(f"[CIMemorySystem] Step 2 complete: {repair_plan.get('total_problems', 0)} problems in repair plan")

  # Add metadata
  repair_plan.setdefault("organized_problems", organized)
  repair_plan.setdefault("relevant_candidates", [
    {
      "index": i,
      "level": m.get("memory_level", ""),
      "score": m.get("similarity_score", 0.0)
    }
    for i, m in enumerate(matches[:30])
  ])

  return repair_plan



# ─────────────────────────────────────────────────────────────────────────────
# Single synthesis entry point
# ─────────────────────────────────────────────────────────────────────────────

def _run_two_llm_gate(
  memory_result: Dict[str, Any],
  llm: Any,
  validation_sequence: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """
  Build the memory guidance used by the repair agent.

  ALWAYS synthesizes repair plan (even with 0 memories).
  Generates detailed repair plan with problems mapped to validation sequence.
  """
  result = _empty_selection()

  # ALWAYS synthesize (even if matches=0)
  # With 0 memories: still predict consecutive problems from validation sequence
  if llm is None:
    # No LLM: deterministic fallback
    guidance_document = _build_simple_fallback(memory_result, validation_sequence)
  else:
    # With LLM: full synthesis (works with or without memories)
    guidance_document = _run_single_llm_synthesis(memory_result, llm, validation_sequence)

  relevant_candidates = guidance_document.get("relevant_candidates", [])

  result["use_memory"] = True  # Mark as usable (even if 0 memories, we have synthesis)
  result["relevant_candidates"] = relevant_candidates
  result["selected_items"] = relevant_candidates
  result["guidance_document"] = guidance_document
  result["analysis_summary"] = guidance_document.get("summary", "")
  return result


# ─────────────────────────────────────────────────────────────────────────────
# Format memory context for the problem statement (Phase D)
# ─────────────────────────────────────────────────────────────────────────────

def format_memory_context(memory: Dict[str, Any]) -> str:
  """
  Convert Phase C output into a Markdown block for the agent's problem statement.

  If LLM synthesis produced structured problems, format them into sequential repair instructions.
  Otherwise return the agent_problem_statement markdown as-is.
  """
  if not memory or not memory.get("enabled", False):
    return ""

  llm_sel = memory.get("llm_selection") or {}
  if not llm_sel.get("use_memory", False):
    return ""

  doc: Dict[str, Any] = llm_sel.get("guidance_document") or {}
  if not doc:
    return ""

  # Option 1: LLM provided agent_problem_statement markdown
  agent_statement = str(doc.get("agent_problem_statement") or "").strip()
  if agent_statement:
    return agent_statement

  # Option 2: Build from structured problems array (if LLM returned it)
  problems = doc.get("problems", [])
  if problems:
    return _format_problems_as_markdown(problems, doc.get("total_problems", len(problems)))

  # Fallback: empty
  return ""


def _format_problems_as_markdown(problems: List[Dict[str, Any]], total: int) -> str:
  """
  Convert structured problems array into sequential repair markdown for the agent.

  Format:
  - Problem #1 (CONFIRMED): Current CI failure - fix immediately
  - Problem #2+ (PROBABLE): Check first, then fix if exists
  - Clear verification commands for each step
  """
  lines = [
    "# CI Repair Plan - Sequential Multi-Problem Fix",
    "",
    f"**IMPORTANT**: This repair plan contains {total} problems. You must fix them in sequence:",
    "1. Fix each problem in order",
    "2. For PROBABLE problems: check if they exist first (run verification command)",
    "3. Verify each fix before proceeding to the next",
    "4. Run full CI at the end",
    "",
    "---",
    "",
  ]

  for prob in problems:
    num = prob.get("problem_number", "?")
    status = prob.get("status", "unknown")
    stage = prob.get("validation_stage", "")
    error_type = prob.get("error_type", "")
    error_desc = prob.get("error_description", "")
    files = prob.get("files", [])
    fix_strategy = prob.get("fix_strategy", "")
    verify_cmd = prob.get("verification_cmd", "")
    check_first = prob.get("check_first", False)
    reasoning = prob.get("reasoning", "")
    interdep = prob.get("interdependency", "")

    status_label = "CONFIRMED - Currently Failing" if status == "confirmed" else "PROBABLE - May Appear After Previous Fixes"

    lines.extend([
      f"## Problem #{num} ({status_label})",
      "",
      f"**Validation Stage**: {stage}",
      f"**Error Type**: {error_type}",
      f"**Description**: {error_desc}",
      "",
    ])

    if files:
      lines.append("**Errors to Fix**:")
      for idx, f in enumerate(files, start=1):
        path = f.get("path", "")
        line_num = f.get("line", "")
        issue_type = f.get("issue_type", "")
        reason = f.get("reason", "")
        current = f.get("current_code", "")
        required = f.get("required_fix", "")

        # Build comprehensive error entry
        if len(files) > 1:
          lines.append(f"{idx}. **File**: `{path}`" + (f" (line {line_num})" if line_num else ""))
        else:
          lines.append(f"**File**: `{path}`" + (f" (line {line_num})" if line_num else ""))

        if issue_type:
          lines.append(f"   - **Error Code**: {issue_type}")
        if reason:
          lines.append(f"   - **Issue**: {reason}")
        if current:
          lines.append(f"   - **Current Code**: `{current}`")
        if required:
          lines.append(f"   - **Required Fix**: {required}")

        lines.append("")  # Blank line between errors

    # Add identification criteria for PROBABLE problems
    identification = prob.get("identification_criteria", [])
    if identification and check_first:
      lines.append("**How to Identify This Problem**:")
      for criterion in identification:
        lines.append(f"- {criterion}")
      lines.append("")

    if fix_strategy:
      lines.extend([
        f"**Fix Strategy**: {fix_strategy}",
        "",
      ])

    if check_first:
      lines.extend([
        f"**Action Steps**:",
        f"1. **Check if exists**: Run `{verify_cmd}`",
        f"2. **If verification fails**: Problem exists - apply the fix described above",
        f"3. **Re-verify**: Run `{verify_cmd}` again to confirm fix",
        f"4. **If verification passes**: Problem doesn't exist or is fixed - proceed to next problem",
        "",
      ])
    else:
      lines.extend([
        f"**Action Steps**:",
        f"1. **Apply fixes**: Address ALL {prob.get('total_errors', len(files))} error(s) listed above",
        f"2. **Verify**: Run `{verify_cmd}` to confirm all issues resolved",
        f"3. **Proceed**: Move to next problem after verification passes",
        "",
      ])

    if reasoning:
      lines.extend([
        f"**Why This Problem**: {reasoning}",
        "",
      ])

    if interdep:
      lines.extend([
        f"**Dependencies**: {interdep}",
        "",
      ])

    lines.append("---")
    lines.append("")

  lines.extend([
    "## Final Verification",
    "",
    "After fixing all problems:",
    "1. Run the full CI workflow command",
    "2. Verify all validation stages pass",
    "3. Submit the patch",
  ])

  return "\n".join(lines)
