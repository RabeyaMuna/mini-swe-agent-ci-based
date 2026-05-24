"""
ci_memory_system.py
===================
Thin wrapper that wires the local MemoryPlugin (memory_plugin.py) into the
ci_context.py pre-processing pipeline.

Phase C of the pipeline (Steps 1–5):
  1. Build a structured query document from CILogAnalyzer output
  2. Embed the query  (sentence-transformers or fastembed)
  3. Cosine similarity search across L1 / L2 / L3 memory stores
  4. Rank all candidates highest → lowest
  5. LLM relevance gate — keep only genuinely useful items

Public API used by ci_context.py:
    CIMemorySystem.create(memory_root, *, memory_enabled, ...)  → CIMemorySystem
    memory_system.build_and_retrieve(log_analysis_result, *, instance, llm)
    memory_system.save(*, task_id, sha_fail, repo_name, repo_owner,
                       workflow_path, workflow, log_analysis_result, diff, llm)
    format_memory_context(memory)  → str block for problem_statement
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
    """
    Call *llm* with *prompt* and return the text response.
    Compatible with LangChain ChatModels and plain callables.
    Returns "" on any failure.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Plugin import
# ─────────────────────────────────────────────────────────────────────────────

def _import_memory_plugin():
    """Import MemoryPlugin from the local memory_plugin.py module."""
    from minisweagent.run.benchmarks.utils.memory_plugin import MemoryPlugin  # type: ignore
    return MemoryPlugin


# ─────────────────────────────────────────────────────────────────────────────
# CIMemorySystem
# ─────────────────────────────────────────────────────────────────────────────

class CIMemorySystem:
    """
    Manages the three-level memory store (L1 / L2 / L3) for CI repair.

    Lifecycle
    ---------
    1.  ``CIMemorySystem.create(...)``         — instantiate with config
    2.  ``build_and_retrieve(...)``            — Phase C Steps 1–5
    3.  ``save(...)``                          — persist after a successful patch
    """

    def __init__(self, plugin: Any, *, memory_enabled: bool) -> None:
        self._plugin = plugin
        self._enabled = memory_enabled

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        memory_root: Optional[str],
        *,
        memory_enabled: bool = True,
        memory_top_k: int = 3,
        memory_ablation_levels: str = "L1+L2+L3",
        memory_plugin_path: Optional[str] = None,   # kept for API compat, unused
        llm: Any = None,
    ) -> "CIMemorySystem":
        """
        Create a CIMemorySystem.

        When *memory_enabled* is False or *memory_root* is None the returned
        instance is a no-op (``is_enabled()`` returns False).
        """
        if not memory_enabled or not memory_root:
            return cls(None, memory_enabled=False)

        import os
        os.makedirs(memory_root, exist_ok=True)

        MemoryPlugin = _import_memory_plugin()
        config = {
            "memory_enabled": True,
            "memory_top_k": memory_top_k,
            "memory_ablation_levels": memory_ablation_levels,
            "memory_backend": "json",
            "project_result_dir": memory_root,
        }
        plugin = MemoryPlugin(config, memory_root, llm=llm)
        return cls(plugin, memory_enabled=True)

    # ── Public helpers ────────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return (
            self._enabled
            and self._plugin is not None
            and self._plugin.is_enabled()
        )

    # ── Phase C: Steps 1–5 ───────────────────────────────────────────────────

    def build_and_retrieve(
        self,
        log_analysis_result: Dict[str, Any],
        *,
        instance: Optional[Dict[str, Any]] = None,
        llm: Any = None,
    ) -> Dict[str, Any]:
        """
        Run Steps 1–5 of Phase C.

        Returns a dict that is merged into the ``memory`` key of
        ``build_ci_context()``'s return value.  Always succeeds; returns a
        minimal disabled-result if memory is not enabled.
        """
        _empty = {
            "enabled": False,
            "weighted_similarity": 0.0,
            "selected_memory_levels": [],
            "llm_selection": {
                "use_memory": False,
                "selected_items": [],
                "analysis_summary": "",
            },
        }

        if not self.is_enabled():
            return _empty

        instance = instance or {}
        repo_owner = str(instance.get("repo_owner") or "")
        repo_name  = str(instance.get("repo_name") or "")
        repo_full  = f"{repo_owner}/{repo_name}".strip("/")

        # Step 1 — Build structured query from log analysis output
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

        # Steps 2–4 — Embed + cosine search + rank
        try:
            raw = self._plugin.retrieve(query)
        except Exception as exc:
            logger.error("[CIMemorySystem] retrieve failed: %s", exc)
            return _empty

        # Step 5 — LLM relevance gate
        effective_llm = llm or self._plugin.llm
        llm_selection = _llm_select_relevant_memory(raw, effective_llm)

        return {**raw, "llm_selection": llm_selection}

    # ── Post-patch persistence ────────────────────────────────────────────────

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
        """Persist L1 / L2 / L3 memory entries after a successful patch."""
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
# Step 5 — LLM relevance gate
# ─────────────────────────────────────────────────────────────────────────────

def _llm_select_relevant_memory(
    memory_result: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """
    Ask the LLM to keep only candidates that are genuinely relevant to the
    current CI failure.  Returns a structured selection dict.
    """
    _empty: Dict[str, Any] = {
        "use_memory": False,
        "selected_items": [],
        "analysis_summary": "",
    }

    all_matches: List[Dict[str, Any]] = memory_result.get("matches") or []
    if not all_matches or llm is None:
        return _empty

    # Build compact candidate lines for the LLM
    lines: List[str] = []
    for i, row in enumerate(all_matches[:12]):
        level   = row.get("memory_level", "")
        score   = float(row.get("similarity_score", 0.0))
        etype   = row.get("error_type", "")
        pattern = row.get("failure_pattern") or row.get("issue_type", "")
        reason  = (
            row.get("overall_failure_reason")
            or row.get("failure_reason")
            or row.get("principle")
            or ""
        )[:220]
        fix = (
            row.get("fix_strategy")
            or row.get("fix_approach")
            or row.get("fix_direction")
            or ""
        )[:220]
        lines.append(
            f"[{i}] {level} score={score:.2f} error_type={etype} "
            f"failure_pattern={pattern}\n"
            f"    reason={reason}\n"
            f"    fix={fix}"
        )

    q = memory_result.get("query") or {}
    current_etype   = q.get("error_type", "")
    current_pattern = q.get("failure_pattern", "")
    current_reason  = str(q.get("overall_failure_reason") or "")[:400]

    prompt = f"""You are a CI repair memory relevance analyst.

CURRENT FAILURE
- error_type: {current_etype}
- failure_pattern: {current_pattern}
- failure_reason: {current_reason}

RETRIEVED CANDIDATES (ranked highest → lowest similarity)
{chr(10).join(lines)}

Select only candidates GENUINELY relevant to the current failure.
A candidate is relevant if it offers matching error patterns, actionable fix
guidance, or file-level hints that directly apply to this CI failure.

Return STRICT JSON only (no markdown fences, no extra keys):
{{
  "use_memory": true,
  "selected_items": [
    {{
      "index": 0,
      "memory_level": "L1|L2|L3",
      "similarity_score": 0.0,
      "relevance": "high|medium|low",
      "key_insight": "<why relevant to the current failure>",
      "fix_direction": "<transferable fix strategy>"
    }}
  ],
  "analysis_summary": "<2-3 sentences on what the memory tells us about this failure>"
}}

If no candidate is genuinely relevant return use_memory=false and selected_items=[]."""

    try:
        raw = _call_llm(llm, prompt)
        if not raw:
            return _empty
        # Strip accidental markdown fences
        if raw.startswith("```"):
            fenced = raw.splitlines()[1:]
            if fenced and fenced[-1].strip() == "```":
                fenced = fenced[:-1]
            raw = "\n".join(fenced).strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return _empty
        parsed.setdefault("selected_items", [])
        parsed.setdefault("analysis_summary", "")
        return parsed
    except Exception as exc:
        logger.debug("[CIMemorySystem] LLM relevance gate failed: %s", exc)
        return _empty


# ─────────────────────────────────────────────────────────────────────────────
# Format memory context for the problem statement (Phase D input)
# ─────────────────────────────────────────────────────────────────────────────

def format_memory_context(memory: Dict[str, Any]) -> str:
    """
    Convert Phase C output into a Markdown block for the problem statement.

    Uses LLM-selected items when available; returns "" when memory is
    disabled or no relevant items were selected.
    """
    if not memory or not memory.get("enabled", False):
        return ""

    llm_sel = memory.get("llm_selection") or {}
    if not llm_sel.get("use_memory", False):
        return ""

    selected: List[Dict[str, Any]] = llm_sel.get("selected_items") or []
    if not selected:
        return ""

    # Map candidate index → raw retrieval row for extra detail
    all_matches: List[Dict[str, Any]] = memory.get("matches") or []
    row_by_index = {i: row for i, row in enumerate(all_matches[:12])}

    out: List[str] = ["## Memory Context (Similar Past Failures)"]

    summary = llm_sel.get("analysis_summary", "")
    if summary:
        out.append(f"\n{summary}")

    for item in selected:
        idx   = item.get("index")
        row   = row_by_index.get(idx, {}) if idx is not None else {}
        level = item.get("memory_level") or row.get("memory_level", "")
        score = float(item.get("similarity_score") or row.get("similarity_score") or 0.0)

        out.append(f"\n### [{level}] score={score:.2f}")

        if item.get("key_insight"):
            out.append(f"**Insight:** {item['key_insight']}")
        if item.get("fix_direction"):
            out.append(f"**Fix:** {item['fix_direction']}")

        # Supplement with raw row fields when present
        if row.get("file"):
            out.append(f"**File:** {row['file']}")
        etype   = row.get("error_type") or ""
        pattern = row.get("failure_pattern") or row.get("issue_type") or ""
        if etype or pattern:
            out.append(f"**Pattern:** {' / '.join(x for x in [etype, pattern] if x)}")

    return "\n".join(out)
