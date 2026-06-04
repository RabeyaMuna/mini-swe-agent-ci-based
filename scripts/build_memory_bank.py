#!/usr/bin/env python3
"""
build_memory_bank.py - CI-REPAIR-BENCH Style Memory Builder
==========================================================

Builds high-quality L1/L2/L3 memory using two-stage L2 construction:
  - L1: Per-file analysis (one LLM call per changed file)
  - L2: Two-stage process
    * Stage 1 (Identification): Separate files by distinct CI job/check that failed
    * Stage 2 (Enrichment): Chain-of-thought analysis per distinct failure
  - L3: Universal principles (cross-repo)

Key Differences from seed_memory.py (Runtime):
  ✅ Separates by CI job (one commit → multiple L2s if multiple failures)
  ✅ 4-field changed_files: [{file, reason, failure_reason, fix_strategy}]
  ✅ Chain-of-thought enrichment with cross-failure dependency analysis
  ✅ Proper structure for high-quality memory retrieval

Usage:
    # Build from eval issues
    python scripts/build_memory_bank.py \\
        --seed-file data/trs/eval_issues.json \\
        --output-dir data/trs_rebuilt \\
        --model minimax/minimax-m2.5

    # Test on single issue first
    python scripts/build_memory_bank.py \\
        --seed-file data/trs/eval_issues.json \\
        --output-dir test_memory \\
        --slice 0:1

    # Resume from checkpoint
    python scripts/build_memory_bank.py \\
        --seed-file data/trs/eval_issues.json \\
        --output-dir data/trs_rebuilt \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_memory")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minisweagent.models.litellm_model import LiteLLMModel

# Optional: Load pre-analyzed log details if available
_LOG_DETAILS_CACHE: Optional[Dict[str, Dict]] = None


def _load_log_details(log_details_path: Optional[Path] = None) -> Dict[str, Dict]:
    """Load pre-analyzed CI log details (speeds up memory building)."""
    global _LOG_DETAILS_CACHE

    if _LOG_DETAILS_CACHE is not None:
        return _LOG_DETAILS_CACHE

    if log_details_path is None:
        log_details_path = PROJECT_ROOT / "data" / "trs" / "log_details.json"

    if not log_details_path.exists():
        logger.info("[LOG_DETAILS] Not found, will analyze logs on-the-fly")
        _LOG_DETAILS_CACHE = {}
        return _LOG_DETAILS_CACHE

    try:
        with open(log_details_path) as f:
            log_details_list = json.load(f)

        # Index by SHA for fast lookup
        _LOG_DETAILS_CACHE = {
            entry["sha_fail"]: entry
            for entry in log_details_list
            if "sha_fail" in entry
        }

        logger.info(f"[LOG_DETAILS] Loaded {len(_LOG_DETAILS_CACHE)} pre-analyzed CI logs")
        return _LOG_DETAILS_CACHE

    except Exception as e:
        logger.warning(f"[LOG_DETAILS] Failed to load: {e}")
        _LOG_DETAILS_CACHE = {}
        return _LOG_DETAILS_CACHE


def _get_log_context(sha_fail: str, ci_log: str) -> str:
    """Get enhanced CI log context using pre-analyzed details if available."""
    log_details = _load_log_details()

    if sha_fail in log_details:
        details = log_details[sha_fail]

        context_parts = []

        # Add error context
        error_context = details.get("error_context", "")
        if error_context:
            context_parts.append(f"Error Context:\n{error_context}\n")

        # Add error types
        error_types = details.get("error_types", [])
        if error_types:
            context_parts.append(f"Error Types: {', '.join(error_types)}\n")

        # Add failed job
        failed_job = details.get("failed_job", "")
        if failed_job:
            context_parts.append(f"Failed Job: {failed_job}\n")

        # Add relevant files
        relevant_files = details.get("relevant_files", [])
        if relevant_files:
            files_list = [f["file"] for f in relevant_files if isinstance(f, dict) and "file" in f]
            if files_list:
                context_parts.append(f"Relevant Files: {', '.join(files_list[:10])}\n")

        if context_parts:
            enhanced = "".join(context_parts) + "\n" + "="*80 + "\n" + ci_log
            logger.debug(f"[LOG_CONTEXT] Using pre-analyzed details for {sha_fail[:12]}")
            return enhanced

    return ci_log


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_path(path: str) -> str:
    """Normalize file path."""
    return (path or "").strip().lstrip("/").replace("\\", "/")


def _clip(text: str, limit: int = 600) -> str:
    """Clip text to limit with ellipsis."""
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _dedupe(items: list) -> list:
    """Remove duplicates preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _parse_llm_json(raw: str) -> Dict:
    """Parse LLM output as JSON, stripping markdown fences."""
    raw = raw.strip()
    # Remove markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return {}


def _parse_llm_json_array(raw: str) -> List[Dict]:
    """Parse LLM output as JSON array."""
    parsed = _parse_llm_json(raw)
    if isinstance(parsed, list):
        return parsed
    return []


def _write_json(path: Path, data: Any) -> None:
    """Write JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> List[Dict]:
    """Load JSON from file."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  L1 CONSTRUCTION (PER FILE)
# ═══════════════════════════════════════════════════════════════════════════

_L1_SCHEMA = """{
  "file": "path/to/file.py",
  "error_type": "high-level category (Dependency Error, Type Checking, Code Formatting, etc)",
  "failure_pattern": "short kebab-case pattern (pip-pep440-violation, mypy-strict-optional, etc)",
  "failure_reason": "2-3 sentences: what in THIS file caused the CI failure, with evidence",
  "fix_strategy": "1-2 sentences: what was done to fix THIS file",
  "failed_tool": ["tool1", "tool2"],
  "dependent_files": [
    {
      "file": "other/file.py",
      "reason": "precise coupling explanation",
      "must_change": true
    }
  ]
}"""


def _build_l1_prompt(
    repo_name: str,
    sha_fail: str,
    file_path: str,
    file_diff: str,
    ci_log: str,
    log_context: Optional[str] = None,
) -> str:
    """Build prompt for L1 per-file analysis."""
    # Use enhanced log context if available
    if log_context:
        ci_log_section = f"""════════════════════════════════════════════
CI LOG ANALYSIS (Pre-analyzed)
════════════════════════════════════════════
{_clip(log_context, 1500)}

════════════════════════════════════════════
CI LOG (Raw excerpt)
════════════════════════════════════════════
{_clip(ci_log, 1000)}"""
    else:
        ci_log_section = f"""════════════════════════════════════════════
CI LOG (excerpt)
════════════════════════════════════════════
{_clip(ci_log, 2000)}"""

    return f"""You are a CI failure analyst.

════════════════════════════════════════════
REPOSITORY & FILE CONTEXT
════════════════════════════════════════════
Repository: {repo_name}
SHA (failing): {sha_fail}
File: {file_path}

════════════════════════════════════════════
FILE DIFF
════════════════════════════════════════════
{_clip(file_diff, 2000)}

{ci_log_section}

════════════════════════════════════════════
TASK
════════════════════════════════════════════
Analyze how THIS specific file ({file_path}) contributed to the CI failure.

Answer these questions:
1. What type of CI failure occurred? (dependency, type checking, formatting, etc)
2. What specifically in this file caused the failure? (be precise with evidence)
3. How was this file changed to fix the failure?
4. Which other files does this file depend on? (imports, config files, etc)

Return ONLY valid JSON matching this schema (no markdown fences, no extra keys):
{_L1_SCHEMA}
""".strip()


def extract_l1_per_file(
    repo_name: str,
    sha_fail: str,
    changed_files: List[str],
    diff: str,
    ci_log: str,
    llm,
) -> List[Dict]:
    """Extract L1 records (one per changed file)."""
    logger.info(f"[L1] Extracting per-file records for {len(changed_files)} files...")

    l1_records = []

    # Parse diff to extract per-file diffs
    file_diffs = {}
    current_file = None
    current_diff = []

    for line in diff.split("\n"):
        if line.startswith("diff --git"):
            if current_file and current_diff:
                file_diffs[current_file] = "\n".join(current_diff)
            # Extract filename
            match = re.search(r'b/(.+)$', line)
            if match:
                current_file = match.group(1)
                current_diff = [line]
        elif current_file:
            current_diff.append(line)

    if current_file and current_diff:
        file_diffs[current_file] = "\n".join(current_diff)

    # Get enhanced log context if available
    log_context = _get_log_context(sha_fail, ci_log)

    # Build L1 for each file
    for file_path in changed_files:
        file_diff = file_diffs.get(file_path, "")
        if not file_diff:
            logger.warning(f"[L1] No diff found for {file_path}, skipping")
            continue

        logger.info(f"[L1] Analyzing {file_path}...")

        prompt = _build_l1_prompt(repo_name, sha_fail, file_path, file_diff, ci_log, log_context)

        try:
            response = llm.invoke(prompt)
            l1_record = _parse_llm_json(response.content)

            if l1_record and l1_record.get("file"):
                # Add metadata
                l1_record["repo"] = repo_name
                l1_record["sha_fail"] = sha_fail
                l1_records.append(l1_record)
                logger.info(f"[L1] ✓ {file_path}")
            else:
                logger.warning(f"[L1] Failed to parse L1 for {file_path}")

        except Exception as e:
            logger.error(f"[L1] Error analyzing {file_path}: {e}")

    return l1_records


# ═══════════════════════════════════════════════════════════════════════════
#  L2 STAGE 1: IDENTIFICATION (GROUP FILES BY CI JOB)
# ═══════════════════════════════════════════════════════════════════════════

_L2_ID_SCHEMA = """[
  {
    "issue_type": "one of: formatting | test_failure | type_checking | dependency_or_env | workflow_config | import_or_module | other",
    "what_failed": "one sentence: which CI job/check failed and what made it fail",
    "files": ["exact/file/path.py"]
  }
]

Rules:
- Return JSON ARRAY — one object per DISTINCT CI job/check that failed
- Each file must appear in exactly ONE object's files list
- Use EXACT file paths from L1 summaries
- If all files belong to same CI failure → 1 object with all files
- If files belong to different CI failures → one object per failure"""


def _build_l2_id_prompt(repo_name: str, sha_fail: str, l1_records: List[Dict]) -> str:
    """Build prompt for L2 identification stage."""
    summaries = [
        {
            "file": r.get("file", ""),
            "error_type": r.get("error_type", ""),
            "failure_pattern": r.get("failure_pattern", ""),
            "failure_reason": _clip(str(r.get("failure_reason", "")), 120),
            "failed_tool": r.get("failed_tool", []),
        }
        for r in l1_records
    ]

    return f"""You are a CI failure analyst.

════════════════════════════════════════════
REPO CONTEXT
════════════════════════════════════════════
Repository: {repo_name}
SHA (failing): {sha_fail}

════════════════════════════════════════════
L1 FILE SUMMARIES (one per changed file)
════════════════════════════════════════════
{json.dumps(summaries, indent=2, ensure_ascii=False)}

════════════════════════════════════════════
TASK
════════════════════════════════════════════
A "distinct CI failure" is a specific CI job or check that failed independently:
  • "pip install" failing is one CI failure
  • "mypy --strict" failing is a SEPARATE CI failure
  • "ruff check" failing is a SEPARATE CI failure
  • "pytest" failing is a SEPARATE CI failure

Look at each file's error_type, failed_tool, and failure_reason.
Files that share the same CI check belong to the same CI failure.

Rules:
  • Every file must appear in exactly one object's "files" list
  • Use exact file paths as given above
  • If all files failed due to same CI check → return 1 object
  • If files failed in different CI checks → return one object per check

Return ONLY valid JSON array matching this schema (no markdown fences, no extra keys):
{_L2_ID_SCHEMA}""".strip()


# ═══════════════════════════════════════════════════════════════════════════
#  L2 STAGE 2: ENRICHMENT (CHAIN-OF-THOUGHT PER FAILURE)
# ═══════════════════════════════════════════════════════════════════════════

_L2_ENRICH_SCHEMA = """{
  "issue_type": "one of: formatting | test_failure | type_checking | dependency_or_env | workflow_config | import_or_module | other",
  "error_type": "high-level category (e.g. 'Dependency Error', 'Type Checking', 'Code Formatting')",
  "failure_pattern": "short human-readable description (e.g. 'Pip pep440 violation', 'Mypy strict optional')",
  "failure_reason": "2-3 sentences: root cause of THIS specific CI job/check failing — grounded in L1 evidence",
  "failed_tool": ["tools that failed for this CI failure"],
  "failed_cmd": ["exact CI commands that failed"],
  "fix_strategy": "1-2 sentences: what the overall fix did to resolve THIS CI failure",
  "changed_files": [
    {
      "file": "exact/file/path.py",
      "reason": "why this file was changed to fix THIS CI failure",
      "failure_reason": "what specifically in this file caused the CI failure",
      "fix_strategy": "what was done to fix this specific file — from L1 evidence"
    }
  ],
  "dependent_files": [
    {
      "file": "path/to/file from DIFFERENT CI failure that is causally linked",
      "reason": "why this file from another CI failure is linked"
    }
  ],
  "dependent_issues": [
    {
      "issue_type": "the other CI failure's issue_type",
      "direction": "caused_by OR causes",
      "reason": "precise causal explanation",
      "what_arose": "what specific problem appeared in THIS failure because of the link",
      "what_to_fix": "what must be done in linked failure to resolve dependency"
    }
  ],
  "dependency_note": "one sentence summary of causal chain, or empty string if standalone"
}"""


def _build_l2_enrich_prompt(
    repo_name: str,
    sha_fail: str,
    what_failed: str,
    issue_type: str,
    this_l1_records: List[Dict],
    other_failures_summary: List[Dict],
) -> str:
    """Build prompt for L2 enrichment stage."""
    # Strip internal fields from L1 for cleaner prompt
    this_l1_data = [
        {k: v for k, v in r.items()
         if k not in ("sha_fail", "repo", "issue_id")}
        for r in this_l1_records
    ]

    other_text = (
        json.dumps(other_failures_summary, indent=2, ensure_ascii=False)
        if other_failures_summary
        else "  (none — this is the only CI failure in this run)"
    )

    return f"""You are a CI failure analyst building a structured memory record for ONE CI failure.

════════════════════════════════════════════
REPO CONTEXT
════════════════════════════════════════════
Repository: {repo_name}
SHA (failing): {sha_fail}
Issue Type: {issue_type}
What Failed: {what_failed}

════════════════════════════════════════════
L1 RECORDS FOR THIS CI FAILURE
(full details for files that belong to this specific CI failure)
════════════════════════════════════════════
{_clip(json.dumps(this_l1_data, indent=2, ensure_ascii=False), 4000)}

════════════════════════════════════════════
OTHER CI FAILURES IN THE SAME RUN
(compact summaries — for cross-failure dependency analysis only)
════════════════════════════════════════════
{other_text}

════════════════════════════════════════════
CHAIN-OF-THOUGHT TASK
════════════════════════════════════════════
Answer these questions step by step, then produce the final JSON:

1. ROOT CAUSE: Why did this specific CI job/check fail?
   What is the fundamental root cause, grounded in the L1 evidence above?

2. PER-FILE CONTRIBUTION: For each file in the L1 records above:
   - What exactly in that file caused this CI failure?
   - What was done in the ground-truth fix to address it?

3. CROSS-FAILURE DEPENDENCIES: Looking at the other CI failures in this run:
   - Did any other CI failure CAUSE this one? (direction: "caused_by")
   - Did this failure CAUSE any other CI failure? (direction: "causes")
   - What files from those other failures are causally linked to this one?

4. CAUSAL CHAIN: Write a one-sentence summary of how this failure fits in the
   overall failure chain for this CI run (or "standalone" if no dependencies).

Now produce a complete L2 record for this CI failure:
  • changed_files: ALL files from the L1 records above, with:
      - reason: why it was changed for THIS failure
      - failure_reason: what in this file caused the CI failure (from L1)
      - fix_strategy: what the fix did in this file (from L1)
  • dependent_files: files from OTHER CI failures causally linked to THIS one
  • dependent_issues: causal links to other CI failures (from step 3 above)
  • dependency_note: one-sentence causal chain summary (from step 4 above)

Return ONLY valid JSON matching this schema exactly (no markdown fences, no extra keys):
{_L2_ENRICH_SCHEMA}
""".strip()


def _normalize_l2_changed_files(raw: Any) -> List[Dict]:
    """Normalize changed_files into 4-field structure."""
    if not raw or not isinstance(raw, list):
        return []

    result = []
    seen = set()

    for item in raw:
        if isinstance(item, str):
            fp = _normalize_path(item)
            if fp and fp not in seen:
                seen.add(fp)
                result.append({
                    "file": fp,
                    "reason": "",
                    "failure_reason": "",
                    "fix_strategy": ""
                })
        elif isinstance(item, dict):
            fp = _normalize_path(str(item.get("file", "")))
            if fp and fp not in seen:
                seen.add(fp)
                result.append({
                    "file": fp,
                    "reason": _clip(str(item.get("reason", "")), 300),
                    "failure_reason": _clip(str(item.get("failure_reason", "")), 300),
                    "fix_strategy": _clip(str(item.get("fix_strategy", "")), 300),
                })

    return result


def extract_l2_sub_issues(
    repo_name: str,
    sha_fail: str,
    l1_records: List[Dict],
    llm,
) -> List[Dict]:
    """
    Two-stage L2 extraction:
    1. Identification: Group files by distinct CI failure
    2. Enrichment: Chain-of-thought analysis per failure

    Returns list of L2 records (one per distinct CI failure).
    """
    if not l1_records:
        return []

    # Build file → L1 lookup
    l1_by_file = {}
    for r in l1_records:
        fp = _normalize_path(r.get("file", ""))
        if fp:
            l1_by_file[fp] = r
    all_files = set(l1_by_file.keys())

    # ═══ STAGE 1: IDENTIFICATION ═══
    logger.info(f"[L2 Stage1] Identifying distinct CI failures for {len(l1_records)} files...")

    id_prompt = _build_l2_id_prompt(repo_name, sha_fail, l1_records)

    try:
        response = llm.invoke(id_prompt)
        id_records = _parse_llm_json_array(response.content)
    except Exception as e:
        logger.error(f"[L2 Stage1] LLM error: {e}")
        id_records = []

    # Validate and clean
    assigned_files = set()
    valid_id = []

    for item in id_records:
        if not isinstance(item, dict):
            continue

        files = [_normalize_path(f) for f in (item.get("files") or []) if f]
        known = [f for f in files if f in all_files]

        if known:
            valid_id.append({
                "issue_type": str(item.get("issue_type", "other")),
                "what_failed": str(item.get("what_failed", "")),
                "files": known,
            })
            assigned_files.update(known)

    # Fallback for unassigned files
    unassigned = all_files - assigned_files
    if unassigned:
        logger.info(f"[L2 Stage1] {len(unassigned)} unassigned files → fallback group")
        valid_id.append({
            "issue_type": "other",
            "what_failed": "unclassified files not assigned to any identified CI failure",
            "files": sorted(unassigned),
        })

    if not valid_id:
        logger.info(f"[L2 Stage1] No valid groups — single-failure fallback")
        valid_id = [{
            "issue_type": "other",
            "what_failed": "single CI failure covering all changed files",
            "files": list(all_files),
        }]

    n_failures = len(valid_id)
    logger.info(f"[L2 Stage1] Identified {n_failures} distinct CI failure(s)")

    # ═══ STAGE 2: ENRICHMENT ═══
    l2_records = []

    for idx, failure in enumerate(valid_id):
        issue_type = failure["issue_type"]
        what_failed = failure["what_failed"]
        this_files = failure["files"]

        # Get full L1 records for this failure's files
        this_l1 = [l1_by_file[fp] for fp in this_files if fp in l1_by_file]

        # Compact summaries of other failures
        other_summaries = [
            {
                "issue_type": v["issue_type"],
                "what_failed": v["what_failed"],
                "files": v["files"],
            }
            for j, v in enumerate(valid_id) if j != idx
        ]

        logger.info(f"[L2 Stage2] ({idx+1}/{n_failures}) Enriching {issue_type} with {len(this_files)} files...")

        enrich_prompt = _build_l2_enrich_prompt(
            repo_name=repo_name,
            sha_fail=sha_fail,
            what_failed=what_failed,
            issue_type=issue_type,
            this_l1_records=this_l1,
            other_failures_summary=other_summaries,
        )

        try:
            response = llm.invoke(enrich_prompt)
            l2_record = _parse_llm_json(response.content)

            if not l2_record:
                # Fallback to L1 evidence
                logger.warning(f"[L2 Stage2] No record parsed — using L1 fallback")
                l2_record = {
                    "issue_type": issue_type,
                    "error_type": this_l1[0].get("error_type", "") if this_l1 else "",
                    "failure_pattern": this_l1[0].get("failure_pattern", "") if this_l1 else "",
                    "failure_reason": this_l1[0].get("failure_reason", "") if this_l1 else "",
                    "failed_tool": _dedupe([t for r in this_l1 for t in (r.get("failed_tool") or [])]),
                    "fix_strategy": this_l1[0].get("fix_strategy", "") if this_l1 else "",
                    "changed_files": [
                        {
                            "file": r.get("file", ""),
                            "reason": r.get("failure_reason", ""),
                            "failure_reason": r.get("failure_reason", ""),
                            "fix_strategy": r.get("fix_strategy", ""),
                        }
                        for r in this_l1 if r.get("file")
                    ],
                }

            # Normalize changed_files to 4-field structure
            l2_record["changed_files"] = _normalize_l2_changed_files(
                l2_record.get("changed_files", [])
            )

            # Add metadata
            l2_record["repo"] = repo_name
            l2_record["repo_name"] = repo_name
            l2_record["sha_fail"] = sha_fail

            l2_records.append(l2_record)
            logger.info(f"[L2 Stage2] ✓ {issue_type}")

        except Exception as e:
            logger.error(f"[L2 Stage2] Error enriching failure {idx}: {e}")

    return l2_records


# ═══════════════════════════════════════════════════════════════════════════
#  L3 CONSTRUCTION (UNIVERSAL PRINCIPLES)
# ═══════════════════════════════════════════════════════════════════════════

_L3_SCHEMA = """{
  "principle": "universal principle (no repo names, no file paths)",
  "root_cause_pattern": "generic root cause pattern",
  "fix_strategy": "generic fix strategy applicable to any codebase",
  "cascade_pattern": "how failures of this type can cascade to other failures"
}"""


def _build_l3_prompt(l2_records: List[Dict]) -> str:
    """Build prompt for L3 universal principles extraction."""
    # Strip repo-specific details
    l2_generic = []
    for l2 in l2_records:
        l2_generic.append({
            "issue_type": l2.get("issue_type", ""),
            "error_type": l2.get("error_type", ""),
            "failure_pattern": l2.get("failure_pattern", ""),
            "failure_reason": _clip(str(l2.get("failure_reason", "")), 200),
            "fix_strategy": _clip(str(l2.get("fix_strategy", "")), 200),
        })

    return f"""You are a CI failure analyst extracting universal principles.

════════════════════════════════════════════
L2 FAILURE SUMMARIES (repo-specific details removed)
════════════════════════════════════════════
{json.dumps(l2_generic, indent=2, ensure_ascii=False)}

════════════════════════════════════════════
TASK
════════════════════════════════════════════
Extract a UNIVERSAL principle from these CI failures that applies to ANY codebase.

IMPORTANT:
  • No repo names (camel, agno, flower, etc)
  • No specific file paths (.pre-commit-config.yaml, pyproject.toml, etc)
  • No project-specific details
  • Focus on general patterns and strategies

Return ONLY valid JSON matching this schema (no markdown fences, no extra keys):
{_L3_SCHEMA}
""".strip()


def extract_l3_principle(l2_records: List[Dict], llm) -> Optional[Dict]:
    """Extract L3 universal principle from L2 records."""
    if not l2_records:
        return None

    logger.info(f"[L3] Extracting universal principle from {len(l2_records)} L2 records...")

    prompt = _build_l3_prompt(l2_records)

    try:
        response = llm.invoke(prompt)
        l3_record = _parse_llm_json(response.content)

        if l3_record:
            logger.info(f"[L3] ✓ Extracted principle")
            return l3_record
        else:
            logger.warning(f"[L3] Failed to parse L3")
            return None

    except Exception as e:
        logger.error(f"[L3] Error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN BUILD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_memory_for_issue(
    issue: Dict,
    llm,
    output_dir: Path,
) -> Dict[str, int]:
    """Build L1/L2/L3 memory for a single issue."""
    repo_name = issue.get("repo_name", issue.get("repo", "unknown"))
    sha_fail = issue.get("sha_fail", "")
    issue_id = issue.get("id", "")

    logger.info(f"\n{'='*80}")
    logger.info(f"Building memory for: {repo_name} / {sha_fail[:12]} (ID: {issue_id})")
    logger.info(f"{'='*80}")

    # Extract data
    changed_files = issue.get("changed_files", [])
    diff = issue.get("diff", "")

    # CI logs - handle both list and string formats
    ci_log_raw = issue.get("logs", "")
    if isinstance(ci_log_raw, list):
        # Join list of log entries
        ci_log = "\n".join([
            entry if isinstance(entry, str) else json.dumps(entry)
            for entry in ci_log_raw
        ])
    else:
        ci_log = str(ci_log_raw)

    if not changed_files:
        logger.warning(f"No changed files for {issue_id}, skipping")
        return {"l1": 0, "l2": 0, "l3": 0}

    # Build L1 (per file)
    l1_records = extract_l1_per_file(
        repo_name=repo_name,
        sha_fail=sha_fail,
        changed_files=changed_files,
        diff=diff,
        ci_log=ci_log,
        llm=llm,
    )

    if not l1_records:
        logger.warning(f"No L1 records generated for {issue_id}")
        return {"l1": 0, "l2": 0, "l3": 0}

    # Build L2 (two-stage: identify then enrich)
    l2_records = extract_l2_sub_issues(
        repo_name=repo_name,
        sha_fail=sha_fail,
        l1_records=l1_records,
        llm=llm,
    )

    # Build L3 (universal principles)
    l3_record = extract_l3_principle(l2_records, llm)

    # Save to files
    l1_path = output_dir / "failure_memory.json"
    l2_path = output_dir / "repo_memory.json"
    l3_path = output_dir / "cross_memory.json"

    # Append to existing
    existing_l1 = _load_json(l1_path)
    existing_l2 = _load_json(l2_path)
    existing_l3 = _load_json(l3_path)

    _write_json(l1_path, existing_l1 + l1_records)
    _write_json(l2_path, existing_l2 + l2_records)

    if l3_record:
        # Add metadata
        l3_record["repo"] = repo_name
        l3_record["sha_fail"] = sha_fail
        _write_json(l3_path, existing_l3 + [l3_record])

    logger.info(f"✓ Saved: {len(l1_records)} L1, {len(l2_records)} L2, {1 if l3_record else 0} L3")

    return {
        "l1": len(l1_records),
        "l2": len(l2_records),
        "l3": 1 if l3_record else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Build high-quality memory bank")
    parser.add_argument("--seed-file", required=True, help="Path to seed issues JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for memory files")
    parser.add_argument("--model", default="minimax/minimax-m2.5", help="LLM model to use")
    parser.add_argument("--slice", help="Process subset (e.g., 0:5 for first 5 issues)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing memory")
    args = parser.parse_args()

    # Load seed issues
    seed_path = Path(args.seed_file)
    if not seed_path.exists():
        logger.error(f"Seed file not found: {seed_path}")
        return 1

    with open(seed_path) as f:
        issues = json.load(f)

    # Apply slice
    if args.slice:
        start, end = map(int, args.slice.split(":"))
        issues = issues[start:end]
        logger.info(f"Processing slice [{start}:{end}] = {len(issues)} issues")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip already processed SHAs
    if args.resume:
        existing_l2 = _load_json(output_dir / "repo_memory.json")
        processed_shas = {r.get("sha_fail") for r in existing_l2}
        issues = [i for i in issues if i.get("sha_fail") not in processed_shas]
        logger.info(f"Resuming: {len(issues)} issues remaining")

    # Initialize LLM
    logger.info(f"Initializing LLM: {args.model}")
    llm = LiteLLMModel(model_name=args.model)

    # Process issues
    total_stats = {"l1": 0, "l2": 0, "l3": 0}

    for idx, issue in enumerate(issues, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"Issue {idx}/{len(issues)}")
        logger.info(f"{'#'*80}")

        try:
            stats = build_memory_for_issue(issue, llm, output_dir)
            total_stats["l1"] += stats["l1"]
            total_stats["l2"] += stats["l2"]
            total_stats["l3"] += stats["l3"]

        except Exception as e:
            logger.error(f"Failed to process issue {issue.get('id')}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total L1 records: {total_stats['l1']}")
    logger.info(f"Total L2 records: {total_stats['l2']}")
    logger.info(f"Total L3 records: {total_stats['l3']}")
    logger.info(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
