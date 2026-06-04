"""
ci_memory_system.py
===================
Wires MemoryPlugin into the ci_context.py pre-processing pipeline.

Phase C (Steps 1–5):
  1. Build structured query from CILogAnalyzer output
  2. Embed query (sentence-transformers or fastembed)
  3. Cosine similarity search across L1 / L2 / L3 memory stores
  4. Rank candidates highest → lowest
  5. Two-LLM gate:
       LLM 1 — Relevance Filter: fast binary selection of useful candidates
       LLM 2 — Experience Synthesizer: deep repair guidance document
               (only called when LLM 1 finds relevant candidates)

Design principles:
  - No arbitrary truncation — LLM sees all candidate data
  - Chunked processing when total context exceeds model limits
  - LLM 1 filters noise so LLM 2 works only on signal
  - LLM 2 produces a structured repair playbook, not just raw memories
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_PROMPT_CHARS = 28_000   # ~7k tokens — safe for most models
_CHUNK_SIZE       = 4        # candidates per chunk when prompt is too large


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
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
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
        "relevant_candidates":   [],   # LLM 1 output
        "guidance_document":     {},   # LLM 2 output
        "analysis_summary":      "",
    }


def _format_candidate(i: int, row: Dict[str, Any]) -> str:
    """Render one memory record for the LLM — all fields, no truncation."""
    level   = row.get("memory_level", "")
    score   = float(row.get("similarity_score") or 0.0)
    error   = row.get("error_type", "")
    pattern = row.get("failure_pattern") or row.get("issue_type", "")
    reason  = (
        row.get("overall_failure_reason")
        or row.get("failure_reason")
        or row.get("principle")
        or ""
    )
    fix = (
        row.get("fix_strategy")
        or row.get("fix_approach")
        or row.get("fix_direction")
        or ""
    )

    dep_files = row.get("dependent_files") or []
    dep_parts = []
    for d in dep_files:
        if isinstance(d, dict):
            f = d.get("file", "")
            r = d.get("reason", "")
            dep_parts.append(f"{f} ({r})" if r else f)
        elif d:
            dep_parts.append(str(d))
    dep_text = f"\n    dependent_files: {' | '.join(dep_parts)}" if dep_parts else ""

    changed = row.get("changed_files") or row.get("modified_files") or []
    changed_text = (
        f"\n    co_changed_files: {' | '.join(str(x) for x in changed)}"
        if changed else ""
    )

    files_list = row.get("files") or []
    files_parts = []
    for f in files_list:
        if isinstance(f, dict):
            name         = f.get("file", "")
            file_pattern = f.get("failure_pattern", "")
            file_fix     = f.get("fix_direction") or f.get("fix_strategy", "")
            files_parts.append(
                f"{name} [{file_pattern}] fix={file_fix}"
                if (file_pattern or file_fix) else name
            )
        elif f:
            files_parts.append(str(f))
    files_text = (
        f"\n    all_files_in_issue: {' | '.join(files_parts)}"
        if files_parts else ""
    )

    return (
        f"[{i}] {level} score={score:.3f} error_type={error} pattern={pattern}\n"
        f"    reason: {reason}\n"
        f"    fix: {fix}"
        f"{dep_text}"
        f"{changed_text}"
        f"{files_text}"
    )


def _current_failure_block(q: Dict[str, Any]) -> str:
    error   = q.get("error_type", "")
    pattern = q.get("failure_pattern", "")
    reason  = str(q.get("overall_failure_reason") or q.get("failure_reason") or "")
    files   = " | ".join(str(f) for f in (q.get("relevant_files") or [])) or "(none identified)"
    return (
        f"error_type      : {error}\n"
        f"failure_pattern : {pattern}\n"
        f"failure_reason  : {reason}\n"
        f"files_in_log    : {files}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM 1 — Relevance Filter
# ─────────────────────────────────────────────────────────────────────────────

def _build_filter_prompt(failure_block: str, candidate_lines: List[str]) -> str:
    return f"""You are a CI repair memory relevance filter.

CURRENT CI FAILURE
{failure_block}

RETRIEVED MEMORY CANDIDATES (ranked highest → lowest similarity)
{chr(10).join(candidate_lines)}

TASK
For each candidate decide: is it genuinely relevant to the current failure?
A candidate is relevant if it:
  - Matches the error type or failure pattern
  - Shows a fix, root cause analysis, or solution approach (even if the error matches the log)
  - Reveals additional files, dependencies, or context not visible in the current log
  - Shows a transferable fix that could directly apply

IMPORTANT:
  - A memory with the SAME error but a known FIX is highly relevant - select it!
  - Do NOT select based on similarity score alone - check if it adds value
  - Do NOT select if it ONLY describes the error without any fix, root cause, or actionable guidance

Return STRICT JSON only (no markdown):
{{
  "relevant_candidates": [
    {{
      "index": 0,
      "memory_level": "L1|L2|L3",
      "similarity_score": 0.0,
      "relevance": "high|medium|low",
      "why_relevant": "<one sentence: what this candidate adds beyond the log>"
    }}
  ]
}}

If nothing is relevant return: {{"relevant_candidates": []}}"""


def _run_llm1_filter(
    all_matches: List[Dict[str, Any]],
    failure_block: str,
    llm: Any,
) -> List[Dict[str, Any]]:
    """
    LLM 1 — fast relevance filter.
    Returns list of selected candidate dicts with their original index.
    Chunks if prompt is too large.
    """
    if not all_matches or llm is None:
        return []

    candidate_lines = [_format_candidate(i, row) for i, row in enumerate(all_matches)]

    def _call_chunk(lines: List[str]) -> List[Dict[str, Any]]:
        prompt = _build_filter_prompt(failure_block, lines)
        raw    = _call_llm(llm, prompt)
        parsed = _parse_json(raw)
        if not parsed:
            return []
        return parsed.get("relevant_candidates") or []

    full_prompt = _build_filter_prompt(failure_block, candidate_lines)
    if len(full_prompt) <= _MAX_PROMPT_CHARS:
        return _call_chunk(candidate_lines)

    # Chunked
    logger.info("[CIMemorySystem] LLM1 filter: chunking %d candidates", len(candidate_lines))
    results: List[Dict[str, Any]] = []
    for start in range(0, len(candidate_lines), _CHUNK_SIZE):
        chunk = candidate_lines[start : start + _CHUNK_SIZE]
        results.extend(_call_chunk(chunk))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LLM 2 — Experience Synthesizer
# ─────────────────────────────────────────────────────────────────────────────

def _build_synthesis_prompt(
    failure_block: str,
    relevant_candidate_lines: List[str],
    relevance_notes: List[str],
) -> str:
    candidates_text = "\n\n".join(relevant_candidate_lines)
    notes_text      = "\n".join(relevance_notes)

    return f"""You are a CI repair experience synthesizer. You have been given:
  1. A current CI failure (partial — the log is incomplete)
  2. Past repair experiences that were judged relevant to this failure
  3. Notes on WHY each past experience is relevant

Your job is to synthesize these into a complete repair guidance document
that fills in what the CI log does NOT show and tells the agent exactly
what to do, how to verify, and what to watch out for.

═══════════════════════════════════════════════════════
CURRENT CI FAILURE  (log is partial)
═══════════════════════════════════════════════════════
{failure_block}

═══════════════════════════════════════════════════════
RELEVANT PAST EXPERIENCES
═══════════════════════════════════════════════════════
{candidates_text}

═══════════════════════════════════════════════════════
RELEVANCE NOTES (from the filter step)
═══════════════════════════════════════════════════════
{notes_text}

═══════════════════════════════════════════════════════
SYNTHESIZE A REPAIR GUIDANCE DOCUMENT
═══════════════════════════════════════════════════════

Produce a structured document covering:

1. DIAGNOSIS — What is really happening beyond what the log shows?
   The log shows symptoms. What is the actual root cause based on past experience?

2. FULL SCOPE — What files need to change?
   The log may only mention one file. Based on past fixes, what other files
   are affected by the same root cause? Include files from dependent_files
   and co_changed_files in the past experiences.

3. LINKED ISSUES — Are there related failure patterns that come from the
   same root cause? If fixing file A typically also requires fixing file B,
   or if this error type tends to cascade into secondary failures, say so.

4. FIX APPROACH — How to fix it, step by step.
   Be specific: exact imports, function names, patterns. Use what the past
   fixes show, not general advice.

5. POST-FIX PATTERNS — What commonly breaks AFTER this fix?
   Based on past experience, what secondary failures or follow-up CI errors
   should the agent check for after applying the fix?

6. VERIFICATION — How to verify the fix worked before submitting.
   What command to run, what output to expect, what file to check.

7. CONFIDENCE — How confident are you based on the evidence?
   (high = multiple past fixes agree / low = only one partial match)

Return STRICT JSON only (no markdown fences):
{{
  "diagnosis": "<what is really happening — root cause beyond the log>",
  "full_scope": {{
    "files_in_log":     ["<files the log mentions>"],
    "additional_files": [
      {{
        "file":   "<path>",
        "reason": "<why this file needs changing>",
        "fix":    "<what specifically to change>"
      }}
    ]
  }},
  "linked_issues": [
    {{
      "root_cause":    "<shared root cause>",
      "affected_files": ["<file1>", "<file2>"],
      "fix_pattern":   "<common fix across all these files>",
      "missing_from_log": "<what the log hides about this linked change>"
    }}
  ],
  "fix_approach": [
    "<step 1>",
    "<step 2>",
    "<step 3>"
  ],
  "post_fix_patterns": [
    {{
      "pattern":     "<what might break next>",
      "likelihood":  "high|medium|low",
      "how_to_fix":  "<how to address it if it appears>"
    }}
  ],
  "verification": {{
    "command":         "<command to run to verify the fix>",
    "expected_output": "<what success looks like>",
    "files_to_check":  ["<file to inspect after fixing>"]
  }},
  "confidence":       "high|medium|low",
  "confidence_reason": "<why — number of agreeing past fixes, quality of match>",
  "summary": "<3-4 sentences: complete picture of the failure and the fix approach>"
}}"""


def _run_llm2_synthesis(
    all_matches: List[Dict[str, Any]],
    relevant_candidates: List[Dict[str, Any]],
    failure_block: str,
    llm: Any,
) -> Dict[str, Any]:
    """
    LLM 2 — deep synthesis.
    Only called with the relevant candidates (LLM 1 output).
    Produces a structured repair guidance document.
    """
    if not relevant_candidates or llm is None:
        return {}

    # Build candidate lines for only the relevant ones
    selected_lines  = []
    relevance_notes = []

    for item in relevant_candidates:
        idx = item.get("index")
        if idx is None or idx >= len(all_matches):
            continue
        row = all_matches[idx]
        selected_lines.append(_format_candidate(idx, row))
        note = item.get("why_relevant", "")
        if note:
            relevance_notes.append(f"  [candidate {idx}]: {note}")

    if not selected_lines:
        return {}

    prompt = _build_synthesis_prompt(failure_block, selected_lines, relevance_notes)
    raw    = _call_llm(llm, prompt)
    parsed = _parse_json(raw)

    if not parsed:
        logger.debug("[CIMemorySystem] LLM2 synthesis returned a response that could not be parsed")
        return {}

    # Ensure all expected keys exist
    parsed.setdefault("diagnosis",           "")
    parsed.setdefault("full_scope",          {"files_in_log": [], "additional_files": []})
    parsed.setdefault("linked_issues",       [])
    parsed.setdefault("fix_approach",        [])
    parsed.setdefault("post_fix_patterns",   [])
    parsed.setdefault("verification",        {})
    parsed.setdefault("confidence",          "low")
    parsed.setdefault("confidence_reason",   "")
    parsed.setdefault("summary",             "")
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Two-LLM gate entry point
# ─────────────────────────────────────────────────────────────────────────────

def _run_two_llm_gate(
    memory_result: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """
    Two-stage LLM gate:
      LLM 1 — fast relevance filter (all candidates → relevant subset)
      LLM 2 — deep synthesis (relevant subset → repair guidance document)
    """
    result = _empty_selection()

    all_matches: List[Dict[str, Any]] = memory_result.get("matches") or []
    if not all_matches or llm is None:
        return result

    q             = memory_result.get("query") or {}
    failure_block = _current_failure_block(q)

    # ── LLM 1: Relevance Filter ───────────────────────────────────────────────
    relevant_candidates = _run_llm1_filter(all_matches, failure_block, llm)
    logger.info(
        "[CIMemorySystem] LLM1 filter: %d/%d candidates selected as relevant",
        len(relevant_candidates), len(all_matches),
    )

    if not relevant_candidates:
        return result   # nothing relevant — skip LLM 2, return empty

    # ── LLM 2: Experience Synthesizer ────────────────────────────────────────
    guidance_document = _run_llm2_synthesis(
        all_matches, relevant_candidates, failure_block, llm
    )

    result["use_memory"]          = True
    result["relevant_candidates"] = relevant_candidates
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
