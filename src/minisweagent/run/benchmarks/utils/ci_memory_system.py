"""
ci_memory_system.py
===================
Wires MemoryPlugin into the ci_context.py pre-processing pipeline.

Phase C (Steps 1–5):
  1. Build structured query from CILogAnalyzer output
  2. Embed query (sentence-transformers or fastembed)
  3. Cosine similarity search across L1 / L2 / L3 memory stores
  4. Rank candidates highest → lowest
  5. One synthesis prompt turns retrieved memory into previous-experience
     repair guidance for the agent problem statement.

Design principles:
  - Keep retrieval deterministic and transparent
  - Use one LLM synthesis step for reasoning over visible + hidden failures
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
            from langchain_core.messages import HumanMessage  # type: ignore
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
    from minisweagent.run.benchmarks.utils.memory_plugin import MemoryPlugin  # type: ignore
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
        self._plugin  = plugin
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
            "memory_enabled":         True,
            "memory_top_k":           memory_top_k,
            "memory_ablation_levels": memory_ablation_levels,
            "memory_backend":         "json",
            "project_result_dir":     memory_root,
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
    ) -> Dict[str, Any]:
        """Run Steps 1–5 of Phase C."""
        _empty = {
            "enabled":                False,
            "weighted_similarity":    0.0,
            "selected_memory_levels": [],
            "llm_selection": _empty_selection(),
        }

        if not self.is_enabled():
            return _empty

        instance   = instance or {}
        repo_owner = str(instance.get("repo_owner") or "")
        repo_name  = str(instance.get("repo_name") or "")
        repo_full  = f"{repo_owner}/{repo_name}".strip("/")

        try:
            query = self._plugin.build_query(
                task_id=str(instance.get("instance_id") or instance.get("id") or ""),
                sha_fail=str(instance.get("sha_fail") or ""),
                repo_name=repo_full,
                workflow_path=str(instance.get("workflow_path") or ""),
                workflow=str(instance.get("workflow") or ""),
                log_analysis_result=log_analysis_result,
                changed_files_info=None,
            )
        except Exception as exc:
            logger.error("[CIMemorySystem] build_query failed: %s", exc)
            return _empty

        try:
            raw = self._plugin.retrieve(query)
        except Exception as exc:
            logger.error("[CIMemorySystem] retrieve failed: %s", exc)
            return _empty
        effective_llm = llm or self._plugin.llm
        llm_selection = _run_two_llm_gate(raw, effective_llm)

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

def _empty_selection() -> Dict[str, Any]:
    return {
        "use_memory":            False,
        "relevant_candidates":   [],
        "selected_items":        [],
        "guidance_document":     {},
        "analysis_summary":      "",
    }


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _normalize_path(path: Any) -> str:
    return str(path or "").strip().lstrip("/").replace("\\", "/")


def _add_file(files: List[Dict[str, str]], seen: set[str], file_path: Any, reason: str = "", fix: str = "") -> None:
    path = _normalize_path(file_path)
    if not path or path in seen:
        return
    seen.add(path)
    row = {"file": path, "reason": reason}
    if fix:
        row["fix"] = fix
    files.append(row)


def _build_deterministic_guidance(memory_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a production-safe previous-experience document directly from ranked
    L1/L2/L3 memories.

    This is intentionally simple: no extra LLM calls, no hidden filtering, and
    no separate trajectory prompt. The repair agent receives the current CI
    context plus the most relevant prior repair evidence and can reason from it.
    """
    q = memory_result.get("query") or {}
    matches: List[Dict[str, Any]] = memory_result.get("matches") or []
    if not matches:
        return {}

    files_in_log = [_normalize_path(f) for f in _safe_list(q.get("relevant_files")) if _normalize_path(f)]
    changed_files = [_normalize_path(f) for f in _safe_list(q.get("changed_files")) if _normalize_path(f)]
    primary_seen: set[str] = set()
    additional_seen: set[str] = set(files_in_log)
    primary_files: List[Dict[str, str]] = []
    additional_files: List[Dict[str, str]] = []
    linked_issues: List[Dict[str, Any]] = []
    fix_steps: List[str] = []
    post_fix_patterns: List[Dict[str, str]] = []
    verification_commands: List[str] = []
    relevant_candidates: List[Dict[str, Any]] = []

    for index, row in enumerate(matches):
        level = row.get("memory_level", "")
        score = float(row.get("similarity_score") or 0.0)
        reason = str(
            row.get("overall_failure_reason")
            or row.get("failure_reason")
            or row.get("reason")
            or row.get("principle")
            or ""
        ).strip()
        fix = str(
            row.get("fix_strategy")
            or row.get("fix_approach")
            or row.get("fix_direction")
            or row.get("principle")
            or ""
        ).strip()
        pattern = str(row.get("failure_pattern") or row.get("issue_type") or row.get("pattern_name") or "").strip()

        relevant_candidates.append({
            "index": index,
            "memory_level": level,
            "similarity_score": round(score, 4),
            "relevance": "high" if score >= 0.65 else "medium" if score >= 0.35 else "low",
            "why_relevant": reason or fix or pattern,
        })

        row_file = _normalize_path(row.get("file"))
        if row_file:
            target = primary_files if row_file in files_in_log or level == "L1" else additional_files
            seen = primary_seen if target is primary_files else additional_seen
            _add_file(target, seen, row_file, reason or f"Relevant {level} memory match", fix)

        for file_row in _safe_list(row.get("files") or row.get("modified_files") or row.get("example_files")):
            if isinstance(file_row, dict):
                file_reason = str(file_row.get("reason") or file_row.get("failure_reason") or reason or "").strip()
                file_fix = str(file_row.get("fix_direction") or file_row.get("fix_strategy") or fix or "").strip()
                _add_file(additional_files, additional_seen, file_row.get("file"), file_reason, file_fix)
            else:
                _add_file(additional_files, additional_seen, file_row, reason, fix)

        for dep in _safe_list(row.get("dependent_files")):
            if isinstance(dep, dict):
                _add_file(additional_files, additional_seen, dep.get("file"), dep.get("reason", "Dependent file from prior repair"), fix)
            else:
                _add_file(additional_files, additional_seen, dep, "Dependent file from prior repair", fix)

        if fix and fix not in fix_steps:
            fix_steps.append(fix)
        for command in _safe_list(row.get("verification") or row.get("verification_after_fix") or row.get("ci_command")):
            text = str(command).strip()
            if text and text not in verification_commands:
                verification_commands.append(text)

        atomic = row.get("atomic_problems") or row.get("all_problems") or []
        repair_trajectory = row.get("repair_trajectory") or []
        if atomic or repair_trajectory:
            linked_issues.append({
                "root_cause": reason or pattern,
                "affected_files": [
                    item["file"] for item in additional_files if item.get("file")
                ],
                "fix_pattern": fix,
                "missing_from_log": "Prior repair includes linked atomic problems or trajectory steps that may appear after the first CI failure is fixed.",
                "atomic_problems": atomic,
                "repair_trajectory": repair_trajectory,
            })
            post_fix_patterns.append({
                "pattern": "Downstream CI stages may reveal hidden failures after the visible failure is fixed.",
                "likelihood": "medium",
                "how_to_fix": "Follow the linked repair trajectory and run the workflow validations in order.",
            })

    for file_path in files_in_log:
        _add_file(primary_files, primary_seen, file_path, "Mentioned by current CI failure/log analysis", "")
    for file_path in changed_files:
        _add_file(additional_files, additional_seen, file_path, "Changed file from current issue context", "")

    best_score = max((float(row.get("similarity_score") or 0.0) for row in matches), default=0.0)
    confidence = "high" if best_score >= 0.65 else "medium" if best_score >= 0.35 else "low"
    diagnosis_parts = [
        str(q.get("overall_failure_reason") or "").strip(),
        next((str(row.get("overall_failure_reason") or row.get("failure_reason") or row.get("principle") or "").strip() for row in matches if row), ""),
    ]
    diagnosis = " ".join(part for part in diagnosis_parts if part).strip()

    command = verification_commands[0] if verification_commands else ""
    return {
        "diagnosis": diagnosis,
        "primary_files": primary_files,
        "full_scope": {
            "files_in_log": files_in_log,
            "primary_files": primary_files,
            "additional_files": additional_files,
            "l1_followup_queries": [
                {
                    "file": item["file"],
                    "reason": "Fetch file-level L1 memory/details because this file is linked by prior repair evidence.",
                }
                for item in additional_files
            ],
        },
        "linked_issues": linked_issues,
        "fix_approach": fix_steps or ["Inspect the primary files, then apply the matching prior repair pattern to linked files."],
        "post_fix_patterns": post_fix_patterns,
        "verification": {
            "command": command,
            "expected_output": "The workflow stage that failed should pass; then run downstream validation stages from the CI workflow.",
            "files_to_check": [item["file"] for item in primary_files + additional_files if item.get("file")],
        },
        "confidence": confidence,
        "confidence_reason": f"Best memory similarity score is {best_score:.2f}; {len(matches)} memory candidates were retrieved.",
        "summary": (
            "Use the current CI failure and workflow context as the primary signal. "
            "The memory bank provides prior repair evidence for similar failures, including files to inspect, likely linked changes, "
            "and validation steps to run after the visible failure is fixed."
        ),
        "relevant_candidates": relevant_candidates,
    }


def _compact_candidate(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    files: List[str] = []
    for value in _safe_list(row.get("file")):
        path = _normalize_path(value)
        if path and path not in files:
            files.append(path)
    for field in ("files", "modified_files", "example_files", "dependent_files", "changed_files", "affected_files"):
        for item in _safe_list(row.get(field)):
            if isinstance(item, dict):
                path = _normalize_path(item.get("file") or item.get("path"))
            else:
                path = _normalize_path(item)
            if path and path not in files:
                files.append(path)

    return {
        "index": index,
        "memory_level": row.get("memory_level", ""),
        "similarity_score": row.get("similarity_score", 0.0),
        "repo": row.get("repo") or row.get("repo_name") or "",
        "issue_id": row.get("issue_id") or row.get("sha_fail") or "",
        "error_type": row.get("error_type", ""),
        "failure_pattern": row.get("failure_pattern") or row.get("issue_type") or row.get("pattern_name") or "",
        "symptom_or_reason": row.get("symptom") or row.get("overall_failure_reason") or row.get("failure_reason") or row.get("reason") or row.get("principle") or "",
        "root_cause": row.get("root_cause") or row.get("why_occurred") or "",
        "fix": row.get("how_fixed") or row.get("fix_strategy") or row.get("fix_approach") or row.get("fix_direction") or row.get("principle") or "",
        "files": files,
        "atomic_problems": row.get("atomic_problems") or row.get("all_problems") or [],
        "repair_trajectory": row.get("repair_trajectory") or [],
        "repair_trajectory_summary": row.get("repair_trajectory_summary") or "",  # ← ADDED: Pass trajectory reasoning
        "verification_sequence": row.get("verification_sequence") or [],  # ← ADDED: Pass verification steps
        "verification": row.get("verification_after_fix") or row.get("verification") or row.get("ci_command") or "",
    }


def _build_single_synthesis_prompt(memory_result: Dict[str, Any], deterministic_doc: Dict[str, Any]) -> str:
    q = memory_result.get("query") or {}
    compact_candidates = [
        _compact_candidate(row, index)
        for index, row in enumerate(memory_result.get("matches") or [])
    ]
    current_context = {
        "task_id": q.get("task_id"),
        "sha_fail": q.get("sha_fail"),
        "repo": q.get("repo"),
        "workflow_path": q.get("workflow_path"),
        "error_type": q.get("error_type"),
        "failure_pattern": q.get("failure_pattern"),
        "overall_failure_reason": q.get("overall_failure_reason"),
        "relevant_files_from_log": q.get("relevant_files") or [],
        "changed_files": q.get("changed_files") or [],
        "failed_cmd": q.get("failed_cmd") or [],
        "failed_tool": q.get("failed_tool") or [],
        "level_scores": memory_result.get("level_scores") or {},
        "weighted_similarity": memory_result.get("weighted_similarity"),
    }

    return f"""You are synthesizing CI repair memory into a previous-experience note for a repair agent.

Use the current CI context plus retrieved L1/L2/L3 memories. The CI log may show only the first failure. Your job is to reason from prior repairs and identify:
- primary files to inspect first
- hidden/downstream CI problems likely to appear after the visible failure is fixed
- extra file-level L1 information that should be consulted
- concrete fixes and validation commands
- a concise previous-experience summary that can be merged into the problem statement

CRITICAL GUIDANCE FOR MULTI-PROBLEM CI FAILURES:
1. Check "repair_trajectory_summary" in L2 memories - this explains:
   - Which problems to fix FIRST (dependency roots)
   - Which problems are HIDDEN (will fail after visible ones are fixed)
   - WHY that order (based on CI validation sequence)
   - Step-by-step verification commands

2. Check "verification_sequence" in L2 memories - this shows:
   - The CI validation order (install → lint → type → test)
   - Hidden validations that haven't run yet
   - What will fail NEXT after current fix

3. For multi-file failures, look for:
   - File dependencies (file A blocks file B)
   - Validation cascades (mypy must pass before pytest runs)
   - Pattern-based problems (same fix across many files)

4. In "additional_files", include ALL files that:
   - Share the same root cause (even if not in current CI log)
   - Will fail in later CI validations (based on repair_trajectory_summary)
   - Are mentioned in "atomic_problems" but not visible yet

Do not invent facts. Use only the current context and retrieved memories. If evidence is weak, mark confidence low.

CURRENT CI CONTEXT
{json.dumps(current_context, indent=2, ensure_ascii=False)}

RETRIEVED MEMORY CANDIDATES
{json.dumps(compact_candidates, indent=2, ensure_ascii=False)}

DETERMINISTIC FALLBACK SUMMARY
{json.dumps(deterministic_doc, indent=2, ensure_ascii=False)}

Return STRICT JSON only:
{{
  "diagnosis": "reasoned root cause and hidden CI picture",
  "primary_files": [
    {{"file": "path", "reason": "why primary", "fix": "specific fix pattern if known"}}
  ],
  "full_scope": {{
    "files_in_log": ["paths from current log"],
    "primary_files": [
      {{"file": "path", "reason": "why primary", "fix": "specific fix pattern if known"}}
    ],
    "additional_files": [
      {{"file": "path", "reason": "why likely needed", "fix": "specific prior fix pattern"}}
    ],
    "l1_followup_queries": [
      {{"file": "path", "reason": "what file-level detail should be fetched/checked"}}
    ]
  }},
  "linked_issues": [
    {{
      "root_cause": "shared cause",
      "affected_files": ["path"],
      "fix_pattern": "common fix",
      "missing_from_log": "hidden/downstream failure inferred from prior repair",
      "workflow_stage": "install/lint/test/build/etc"
    }}
  ],
  "fix_approach": [
    "actionable step"
  ],
  "post_fix_patterns": [
    {{"pattern": "what may fail next", "likelihood": "high|medium|low", "how_to_fix": "what to do"}}
  ],
  "verification": {{
    "command": "best command from workflow or memory",
    "expected_output": "what passing looks like",
    "files_to_check": ["path"]
  }},
  "confidence": "high|medium|low",
  "confidence_reason": "why",
  "summary": "short previous-experience note for the repair agent",
  "relevant_candidates": [
    {{"index": 0, "memory_level": "L1|L2|L3", "similarity_score": 0.0, "relevance": "high|medium|low", "why_relevant": "why selected"}}
  ]
}}"""


def _run_single_llm_synthesis(memory_result: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    deterministic_doc = _build_deterministic_guidance(memory_result)
    if llm is None:
        return deterministic_doc

    prompt = _build_single_synthesis_prompt(memory_result, deterministic_doc)
    raw = _call_llm(llm, prompt)
    parsed = _parse_json(raw)
    if not parsed:
        logger.warning("[CIMemorySystem] Memory synthesis LLM failed or returned invalid JSON; using deterministic guidance")
        logger.debug(f"[CIMemorySystem] Raw LLM output (first 500 chars): {raw[:500] if raw else 'None'}")
        return deterministic_doc

    parsed.setdefault("diagnosis", deterministic_doc.get("diagnosis", ""))
    parsed.setdefault("primary_files", deterministic_doc.get("primary_files", []))
    parsed.setdefault("full_scope", deterministic_doc.get("full_scope", {}))
    parsed.setdefault("linked_issues", deterministic_doc.get("linked_issues", []))
    parsed.setdefault("fix_approach", deterministic_doc.get("fix_approach", []))
    parsed.setdefault("post_fix_patterns", deterministic_doc.get("post_fix_patterns", []))
    parsed.setdefault("verification", deterministic_doc.get("verification", {}))
    parsed.setdefault("confidence", deterministic_doc.get("confidence", "low"))
    parsed.setdefault("confidence_reason", deterministic_doc.get("confidence_reason", ""))
    parsed.setdefault("summary", deterministic_doc.get("summary", ""))
    parsed.setdefault("relevant_candidates", deterministic_doc.get("relevant_candidates", []))
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Single synthesis entry point
# ─────────────────────────────────────────────────────────────────────────────

def _run_two_llm_gate(
    memory_result: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """
    Build the memory guidance used by the repair agent.

    Kept under the old function name for compatibility with existing callers.
    The production path uses one synthesis prompt over ranked L1/L2/L3 memory
    candidates, with deterministic guidance as a fallback.
    """
    result = _empty_selection()

    all_matches: List[Dict[str, Any]] = memory_result.get("matches") or []
    if not all_matches:
        return result

    guidance_document = _run_single_llm_synthesis(memory_result, llm)
    relevant_candidates = guidance_document.get("relevant_candidates", [])

    result["use_memory"]          = True
    result["relevant_candidates"] = relevant_candidates
    result["selected_items"]      = relevant_candidates
    result["guidance_document"]   = guidance_document
    result["analysis_summary"]    = guidance_document.get("summary", "")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Format memory context for the problem statement (Phase D)
# ─────────────────────────────────────────────────────────────────────────────

def format_memory_context(memory: Dict[str, Any]) -> str:
    """
    Convert Phase C output into a Markdown block for the agent's problem statement.

    Sections injected (when available):
      1. Diagnosis          — root cause beyond what the log shows
      2. Full scope         — all files that need changing (including hidden ones)
      3. Linked issues      — related failures from the same root cause
      4. Fix approach       — step-by-step repair guidance from past experience
      5. Post-fix patterns  — what might break after fixing
      6. Verification       — how to confirm the fix worked
      7. Confidence         — how much to trust this guidance
    """
    if not memory or not memory.get("enabled", False):
        return ""

    llm_sel = memory.get("llm_selection") or {}
    if not llm_sel.get("use_memory", False):
        return ""

    doc: Dict[str, Any] = llm_sel.get("guidance_document") or {}
    if not doc:
        return ""

    out: List[str] = ["## Memory Context — Repair Guidance from Past Experience\n"]

    confidence        = doc.get("confidence", "")
    confidence_reason = doc.get("confidence_reason", "")
    if confidence:
        badge = {"high": "🟢 HIGH", "medium": "🟡 MEDIUM", "low": "🔴 LOW"}.get(confidence, confidence)
        out.append(f"**Confidence:** {badge}  — {confidence_reason}\n")

    # ── 1. Diagnosis ──────────────────────────────────────────────────────────
    diagnosis = doc.get("diagnosis", "")
    if diagnosis:
        out.append(f"### What is really happening\n{diagnosis}\n")

    # ── 2. Full scope ─────────────────────────────────────────────────────────
    scope = doc.get("full_scope") or {}
    primary = scope.get("primary_files") or doc.get("primary_files") or []
    if primary:
        out.append("### Primary files to inspect first")
        for pf in primary:
            if isinstance(pf, dict):
                fname = pf.get("file", "")
                why = pf.get("reason", "")
                fix = pf.get("fix", "")
            else:
                fname, why, fix = str(pf), "", ""
            line = f"  - `{fname}`"
            if why:
                line += f"  — {why}"
            if fix:
                line += f"\n    → **prior fix pattern:** {fix}"
            out.append(line)
        out.append("")

    additional = scope.get("additional_files") or []
    if additional:
        out.append("### Files to fix — including those NOT in the log")
        for af in additional:
            fname = af.get("file", "")
            why   = af.get("reason", "")
            fix   = af.get("fix", "")
            line  = f"  - `{fname}`"
            if why:
                line += f"  — {why}"
            if fix:
                line += f"\n    → **fix:** {fix}"
            out.append(line)
        out.append("")

    followups = scope.get("l1_followup_queries") or []
    if followups:
        out.append("### Extra file-level memory to consult")
        for item in followups:
            if isinstance(item, dict):
                out.append(f"  - `{item.get('file', '')}` — {item.get('reason', '')}")
            else:
                out.append(f"  - `{item}`")
        out.append("")

    # ── 3. Linked issues ──────────────────────────────────────────────────────
    linked = doc.get("linked_issues") or []
    if linked:
        out.append("### Linked issues — same root cause, fix all affected files")
        for issue in linked:
            out.append(f"\n  **Root cause:** {issue.get('root_cause', '')}")
            if issue.get("missing_from_log"):
                out.append(f"  **What the log hides:** {issue['missing_from_log']}")
            if issue.get("fix_pattern"):
                out.append(f"  **Fix pattern:** {issue['fix_pattern']}")
            affected = issue.get("affected_files") or []
            if affected:
                out.append(f"  **Affected:** {', '.join(f'`{x}`' for x in affected)}")
        out.append("")

    # ── 4. Fix approach ───────────────────────────────────────────────────────
    steps = doc.get("fix_approach") or []
    if steps:
        out.append("### Fix approach (from past experience)")
        for i, step in enumerate(steps, 1):
            out.append(f"  {i}. {step}")
        out.append("")

    # ── 5. Post-fix patterns ──────────────────────────────────────────────────
    post_fix = doc.get("post_fix_patterns") or []
    if post_fix:
        out.append("### Watch for after fixing")
        for pf in post_fix:
            pat        = pf.get("pattern", "")
            likelihood = pf.get("likelihood", "")
            how        = pf.get("how_to_fix", "")
            line = f"  - [{likelihood}] {pat}"
            if how:
                line += f" → {how}"
            out.append(line)
        out.append("")

    # ── 6. Verification ───────────────────────────────────────────────────────
    verify = doc.get("verification") or {}
    if verify.get("command") or verify.get("expected_output"):
        out.append("### Verification")
        if verify.get("command"):
            out.append(f"  Run: `{verify['command']}`")
        if verify.get("expected_output"):
            out.append(f"  Expect: {verify['expected_output']}")
        files_to_check = verify.get("files_to_check") or []
        if files_to_check:
            out.append(f"  Also check: {', '.join(f'`{x}`' for x in files_to_check)}")
        out.append("")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    summary = doc.get("summary", "")
    if summary:
        out.append(f"**Summary:** {summary}")

    return "\n".join(out)
