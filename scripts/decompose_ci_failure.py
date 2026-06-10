#!/usr/bin/env python3
"""
decompose_ci_failure.py - Reverse Engineer CI Failures into Atomic Problems
===========================================================================

Based on professor's direction: Given CI failure (FIRST failure only) + ground truth diff,
use LLM to reverse engineer ALL hidden problems.

Key insight: CI stops at FIRST failure, but diff fixes MULTIPLE problems.
We need to infer hidden problems from the diff.

Usage:
    # Decompose single issue
    python scripts/decompose_ci_failure.py --issue-id 410

    # Decompose all eval issues
    python scripts/decompose_ci_failure.py --batch

References:
    - O-CRD: Backward reasoning from ground truth
    - STAIR: Multi-layer hierarchical abstraction
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import litellm
from dotenv import load_dotenv
from minisweagent.run.benchmarks.utils.ci_context import (
    _log_analysis_to_context,
    _run_log_analysis,
)
from minisweagent.run.benchmarks.utils.ci_workflow_aware_retrieval import (
    analyze_workflow_from_benchmark,
)

try:
    import demjson3  # type: ignore
except Exception:
    demjson3 = None  # type: ignore

DIFF_CHUNK_CHAR_LIMIT = 60000
CHUNK_FINDINGS_INLINE_CHAR_LIMIT = 22000
CHUNK_FINDINGS_BATCH_CHAR_LIMIT = 14000
LOGGER = logging.getLogger(__name__)
STRICT_JSON_RULES = """### Output Rules (STRICT) - CRITICAL FOR PARSING
- Output MUST be ONLY valid JSON - nothing else.
- Start your response immediately with { or [ - NO text before.
- End your response with } or ] - NO text after.
- Do NOT wrap in triple backticks (```json or ```).
- Do NOT add explanations, comments, or markdown.
- Do NOT start with phrases like "Looking at this", "Here is", "json", etc.
- Use double quotes for all JSON keys and string values.
- Do not emit trailing commas.
- Your entire response will be passed directly to json.loads() - ensure it's valid JSON."""

load_dotenv(PROJECT_ROOT / ".env", override=False)

if not os.getenv("OPENROUTER_API_KEY") and os.getenv("MINIMAX_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("MINIMAX_API_KEY", "")
if not os.getenv("OPENROUTER_BASE_URL") and os.getenv("MINIMAX_BASE_URL"):
    os.environ["OPENROUTER_BASE_URL"] = os.getenv("MINIMAX_BASE_URL", "")


class LitellmModel:
    """Small invoke-compatible wrapper for decomposition scripts."""

    def __init__(self, model_name: str):
        self.model_name = self._normalize_model_name(model_name)

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if (
            model_name.startswith("minimax/")
            and os.getenv("OPENROUTER_API_KEY")
            and (
                "openrouter.ai" in os.getenv("OPENROUTER_BASE_URL", "")
                or "openrouter.ai" in os.getenv("MINIMAX_BASE_URL", "")
            )
        ):
            return f"openrouter/{model_name}"
        return model_name

    def invoke(self, prompt: Any):
        if isinstance(prompt, list):
            messages = [
                {
                    "role": "user",
                    "content": str(getattr(message, "content", message)),
                }
                for message in prompt
            ]
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            temperature=0,
        )

        class Result:
            content = response.choices[0].message.content or ""

        return Result()

    def __call__(self, prompt: Any):
        """Make LitellmModel callable for compatibility with CILogAnalyzer."""
        result = self.invoke(prompt)
        return result.content


def _extract_json_from_text(content: str) -> str:
    """Extract JSON from LLM response that may include markdown fences or explanatory text."""
    content = str(content or "").strip()

    # Try to extract JSON from markdown code blocks
    json_fence_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_fence_pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find JSON object/array boundaries
    # Look for outermost { } or [ ]
    first_brace = content.find('{')
    first_bracket = content.find('[')

    if first_brace == -1 and first_bracket == -1:
        return content

    # Determine which comes first
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        # Start with {, find matching }
        start = first_brace
        open_char, close_char = '{', '}'
    else:
        # Start with [, find matching ]
        start = first_bracket
        open_char, close_char = '[', ']'

    # Find the matching closing bracket/brace
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(content)):
        char = content[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return content[start:i+1]

    # If we didn't find a match, return original
    return content


def _load_llm_json(content: str) -> Any:
    """Parse raw LLM JSON. Handles markdown fences and explanatory text."""
    content = str(content or "").strip()

    # First, try to extract JSON from potential markdown or text wrapper
    extracted = _extract_json_from_text(content)

    try:
        return json.loads(extracted)
    except json.JSONDecodeError as json_err:
        demjson3_err: Any = "demjson3 is not installed"
        try:
            if demjson3 is not None:
                return demjson3.decode(extracted)
        except Exception as exc:
            demjson3_err = exc

        # Log the problematic content for debugging
        preview = extracted[:500] if len(extracted) > 500 else extracted
        parse_error = ValueError(
            f"JSON parse failed: json={json_err}; demjson3={demjson3_err}\n"
            f"Content preview (first 500 chars):\n{preview}"
        )
        LOGGER.warning("%s", parse_error)
        return []


def _invoke_json(llm: Any, prompt: str) -> Any:
    response = llm.invoke(prompt)
    content = str(getattr(response, "content", response) or "").strip()
    return _load_llm_json(content)


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2)


def _extract_diff_file_sections(diff: str) -> List[Dict[str, Any]]:
    """Return complete `diff --git` file sections. Never splits a file diff."""
    diff = str(diff or "")
    if not diff.strip():
        return []

    matches = list(re.finditer(r"(?m)^diff --git ", diff))
    if not matches:
        return [{"files": [], "diff": diff, "char_count": len(diff)}]

    sections: List[Dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(diff)
        section = diff[start:end].rstrip()
        first_line = section.split('\n')[0]
        file_match = re.search(r"a/(.*?) b/", first_line)
        files = [file_match.group(1)] if file_match else []
        sections.append({"files": files, "diff": section, "char_count": len(section)})

    return sections


def chunk_diff_by_file(diff: str, max_chars: int = DIFF_CHUNK_CHAR_LIMIT) -> List[Dict[str, Any]]:
    """
    Group complete file diffs into prompt chunks.

    Important invariant:
    - A file diff is never split across chunks.
    - A file diff is never silently truncated.
    - If one file diff exceeds max_chars, it becomes one oversized chunk.
    """
    sections = _extract_diff_file_sections(diff)
    if not sections:
        return []

    chunks: List[Dict[str, Any]] = []
    current_sections: List[Dict[str, Any]] = []
    current_chars = 0

    def flush_current() -> None:
        nonlocal current_sections, current_chars
        if not current_sections:
            return
        chunks.append({
            "files": [
                file_path
                for section in current_sections
                for file_path in section.get("files", [])
            ],
            "diff": "\n\n".join(section.get("diff", "") for section in current_sections),
            "char_count": current_chars,
            "file_count": len(current_sections),
            "oversized_single_file": False,
        })
        current_sections = []
        current_chars = 0

    for section in sections:
        section_chars = int(section.get("char_count") or len(section.get("diff", "")))

        if section_chars > max_chars:
            flush_current()
            chunks.append({
                "files": section.get("files", []),
                "diff": section.get("diff", ""),
                "char_count": section_chars,
                "file_count": 1,
                "oversized_single_file": True,
            })
            continue

        if current_sections and current_chars + section_chars > max_chars:
            flush_current()

        current_sections.append(section)
        current_chars += section_chars

    flush_current()

    return chunks


def _repo_checkout_path(issue: Dict[str, Any]) -> str | None:
    """Return local checkout path for dependent workflow/config files if present."""
    explicit = issue.get("repo_path") or issue.get("checkout_path")
    if explicit and Path(str(explicit)).exists():
        return str(explicit)

    repo = str(issue.get("repo") or "").strip()
    repo_name = str(issue.get("repo_name") or "").strip()
    repo_owner = str(issue.get("repo_owner") or "").strip()
    candidates: List[Path] = []
    if "/" in repo:
        candidates.append(PROJECT_ROOT / "repo" / repo.replace("/", "__"))
    if repo_owner and repo_name:
        candidates.append(PROJECT_ROOT / "repo" / f"{repo_owner}__{repo_name}")
    if repo_name:
        candidates.append(PROJECT_ROOT / "repo" / repo_name)

    for path in candidates:
        if path.exists():
            return str(path)
    return None


def build_benchmark_ci_context(issue: Dict, llm: Any) -> Dict[str, Any]:
    """
    Build structured CI context for decomposition.

    Uses:
      - CILogAnalyzer/precomputed benchmark CI context for failure analysis
      - raw benchmark workflow YAML (`issue["workflow"]`)
      - LLM-extracted ordered validation sequence from that workflow

    This function does not pass raw CI logs into decomposition prompts.
    """
    raw_workflow = str(issue.get("workflow") or "")
    if not raw_workflow.strip():
        raise ValueError(f"Issue {issue.get('id')} has no workflow YAML in benchmark data")

    workflow_path = str(issue.get("workflow_path") or issue.get("workflow_filename") or "")
    repo_path = _repo_checkout_path(issue)
    model_name = str(getattr(llm, "model_name", "") or os.getenv("MEMCI_LLM_MODEL") or "")
    issue_id = str(issue.get("id") or issue.get("instance_id") or "")

    # Check cache first to avoid re-running CILogAnalyzer
    sha_fail = issue.get("sha_fail", "")
    cache_file = PROJECT_ROOT / "data" / "trs" / "log_details.json"
    cached_analysis = None

    if cache_file.exists() and sha_fail:
        try:
            with open(cache_file) as f:
                cache = json.load(f)
            cached_analysis = next((entry for entry in cache if entry.get("sha_fail") == sha_fail), None)
            if cached_analysis:
                print(f"  [1/2] Loading cached CI log analysis for {sha_fail[:12]}...")
        except Exception as e:
            print(f"  ⚠️  Cache load failed: {e}, will re-analyze")

    if cached_analysis:
        log_analysis = cached_analysis
    else:
        print("  [1/2] Building structured CI failure context with CILogAnalyzer...")
        log_analysis = _run_log_analysis(issue, llm=llm, model=model_name)

    context = _log_analysis_to_context(log_analysis, issue, workflow_profile={})

    # Check workflow validation cache
    validation_cache_file = PROJECT_ROOT / "data" / "trs" / "workflow_validation_cache.json"
    cached_validation = None

    if validation_cache_file.exists() and sha_fail:
        try:
            with open(validation_cache_file) as f:
                validation_cache = json.load(f)
            cached_validation = next((entry for entry in validation_cache if entry.get("sha_fail") == sha_fail), None)
            if cached_validation:
                print(f"  [2/2] Loading cached workflow validation sequence for {sha_fail[:12]}...")
        except Exception as e:
            print(f"  ⚠️  Validation cache load failed: {e}, will re-analyze")

    if cached_validation:
        validation_sequence = cached_validation.get("validation_sequence", [])
        workflow_validation_context = {
            "id": str(cached_validation.get("id") or issue_id),
            "sha_fail": str(cached_validation.get("sha_fail") or sha_fail),
            "workflow_path": str(cached_validation.get("workflow_path") or workflow_path),
            "dependent_files": cached_validation.get("dependent_files", []),
            "validation_sequence": validation_sequence,
        }
        print(f"       Found {len(validation_sequence)} validation steps (cached)")
    else:
        print("  [2/2] Analyzing workflow to extract validation sequence...")
        workflow_validation_context = analyze_workflow_from_benchmark(
            workflow_content=raw_workflow,
            workflow_path=workflow_path,
            repo_path=repo_path,
            llm=llm,
            issue_id=issue_id,
            sha_fail=str(sha_fail or ""),
        )
        validation_sequence = workflow_validation_context.get("validation_sequence", [])
        print(f"       Found {len(validation_sequence)} validation steps")

        # Save to global cache
        if sha_fail and validation_sequence:
            try:
                existing_cache = []
                if validation_cache_file.exists():
                    with open(validation_cache_file) as f:
                        existing_cache = json.load(f)

                # Update or append compact workflow-validation context.
                updated = False
                for entry in existing_cache:
                    if entry.get("sha_fail") == sha_fail:
                        entry.clear()
                        entry.update(workflow_validation_context)
                        updated = True
                        break

                if not updated:
                    existing_cache.append(workflow_validation_context)

                validation_cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(validation_cache_file, 'w') as f:
                    json.dump(existing_cache, f, indent=2)
                print(f"Saved workflow validation to cache")
            except Exception as e:
                print(f" Failed to save validation cache: {e}")
    if not validation_sequence:
        raise ValueError(
            f"Issue {issue.get('id')} has no CI workflow validation sequence; "
            "decomposition requires workflow analyzer output first"
        )

    return {
        "context": context,
        "log_analysis": log_analysis,
        "validation_sequence": validation_sequence,
        "workflow_validation_context": workflow_validation_context,
        "workflow_path": workflow_path,
        "workflow_name": str(issue.get("workflow_name") or ""),
        "repo_path": repo_path,
    }


def _has_structured_ci_context(benchmark_context: Dict[str, Any]) -> bool:
    """Return True when CILogAnalyzer produced usable structured failure context."""
    context = benchmark_context.get("context") or {}
    log_analysis = benchmark_context.get("log_analysis") or {}
    return bool(
        context.get("overall_failure_reasons")
        or context.get("overall_error_types")
        or context.get("effected_files")
        or context.get("failed_jobs")
        or log_analysis.get("error_context")
        or log_analysis.get("relevant_files")
        or log_analysis.get("error_types")
        or log_analysis.get("failed_job")
        or log_analysis.get("analysis_document")
        or log_analysis.get("overall_ci_summary")
        or log_analysis.get("chunk_summaries")
    )


def validate_required_ci_inputs(benchmark_context: Dict[str, Any]) -> bool:
    """Both CI analyzer context and workflow validation sequence are required."""
    has_ci_context = _has_structured_ci_context(benchmark_context)
    has_validation_sequence = bool(benchmark_context.get("validation_sequence"))
    if not has_ci_context:
        print("  ✗ Missing structured CI context from CILogAnalyzer; skipping decomposition")
    if not has_validation_sequence:
        print("  ✗ Missing CI workflow validation sequence; skipping decomposition")
    return has_ci_context and has_validation_sequence


def _compact_context_for_diff_analysis(
    issue: Dict,
    benchmark_context: Dict[str, Any],
) -> Dict[str, Any]:
    context = benchmark_context.get("context") or {}
    return {
        "issue_id": issue.get("id"),
        "repo": issue.get("repo_name", issue.get("repo")),
        "workflow_path": benchmark_context.get("workflow_path"),
        "overall_failure_reasons": context.get("overall_failure_reasons", []),
        "overall_error_types": context.get("overall_error_types", []),
        "failed_jobs": context.get("failed_jobs", []),
    }


def _batch_findings_by_size(findings: List[Dict[str, Any]], max_chars: int) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_size = 2

    for finding in findings:
        item_size = len(_json_text(finding))
        if current and current_size + item_size > max_chars:
            batches.append(current)
            current = [finding]
            current_size = item_size
        else:
            current.append(finding)
            current_size += item_size

    if current:
        batches.append(current)

    return batches


def _visible_failure_context_for_consolidation(compact_context: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only CI failure signals needed to label visible vs hidden candidates."""
    return {
        "overall_failure_reasons": compact_context.get("overall_failure_reasons", []),
        "overall_error_types": compact_context.get("overall_error_types", []),
        "failed_jobs": compact_context.get("failed_jobs", []),
    }


def _compact_chunk_findings_for_consolidation(
    chunk_findings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove bulky/redundant fields before consolidation prompts."""
    compact: List[Dict[str, Any]] = []
    for finding in chunk_findings:
        if not isinstance(finding, dict):
            continue
        compact.append({
            "chunk_index": finding.get("chunk_index"),
            "files_analyzed": finding.get("files_analyzed", []),
            "candidate_atomic_problems": finding.get("candidate_atomic_problems", []),
        })
    return compact


def consolidate_chunk_findings(
    issue: Dict,
    benchmark_context: Dict[str, Any],
    chunk_findings: List[Dict[str, Any]],
    llm: Any,
) -> Dict[str, Any]:
    """
    Compress chunk findings only when they are too large for the final prompt.

    This is a compression step, not the final reverse-engineering step. It keeps
    only candidate failure/fix groups that are relevant to CI validations.
    """
    if len(_json_text(chunk_findings)) <= CHUNK_FINDINGS_INLINE_CHAR_LIMIT:
        return {
            "consolidated": False,
            "compacted": False,
            "chunk_findings": chunk_findings,
        }

    compact_context = _compact_context_for_diff_analysis(issue, benchmark_context)
    visible_failure_context = _visible_failure_context_for_consolidation(compact_context)
    compact_findings = _compact_chunk_findings_for_consolidation(chunk_findings)
    validation_sequence = benchmark_context.get("validation_sequence") or []

    if len(_json_text(compact_findings)) <= CHUNK_FINDINGS_INLINE_CHAR_LIMIT:
        print(
            "  Using deterministic compact chunk findings "
            f"({len(_json_text(chunk_findings))} chars -> {len(_json_text(compact_findings))} chars)"
        )
        return {
            "consolidated": False,
            "compacted": True,
            "chunk_findings": compact_findings,
        }

    batches = _batch_findings_by_size(compact_findings, CHUNK_FINDINGS_BATCH_CHAR_LIMIT)
    batch_summaries: List[Dict[str, Any]] = []

    print(f"  Consolidating {len(chunk_findings)} chunk findings into {len(batches)} batch summaries...")

    for batch_index, batch in enumerate(batches, start=1):
        prompt = f"""You are consolidating chunk-level diff analysis for a CI repair.

Use ONLY the inputs provided. Do NOT invent information.

═══════════════════════════════════════════════════════════════════════════════
INPUTS
═══════════════════════════════════════════════════════════════════════════════

CI VALIDATION SEQUENCE (what runs in order):
{json.dumps(validation_sequence, indent=2)}

VISIBLE FAILURE CONTEXT:
{json.dumps(visible_failure_context, indent=2)}

COMPACT CHUNK FINDINGS (batch {batch_index}/{len(batches)}):
{json.dumps(batch, indent=2)}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK: Group Changes by CI Validation Step
═══════════════════════════════════════════════════════════════════════════════

For each group of related changes:

1. WHAT validation step does it fix?
   - Map to CI validation sequence (step 1, 2, 3...)
   - Validation command (e.g., "ruff check", "mypy .", "pytest")

2. WHY did it fail or would fail?
   - Root cause explanation

3. WHAT files are affected?
   - List all files in this group
   - What's wrong in each file

4. FIX STRATEGY?
   - What needs to be modified to make the validation pass
   - Use the diff only to infer the modification; do not quote long diff evidence

═══════════════════════════════════════════════════════════════════════════════
{STRICT_JSON_RULES}

IMPORTANT: Return a SINGLE JSON OBJECT (not an array). Your response must start with {{ and end with }}.

OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

{{
  "batch_index": {batch_index},

  "candidate_problems": [
    {{
      "candidate_id": "batch{batch_index}_step1",
      "validation_step": 1,
      "validation_cmd": "ruff check",
      "what_validates": "Code style and imports",
      "visibility": "visible_candidate | hidden_candidate | unclear",
      

      "why_failed": "Clear explanation of root cause",

      "affected_files": [
        {{
          "file": "path/to/file.py",
          "what_is_wrong": "Specific issue in this file",
          "fix_strategy": "What needs to be modified in this file"
        }}
      ]
    }}
  ]
}}

RULES:
- Group by CI validation step (step 1, step 2, etc.)
- Each candidate_id should be: batch{batch_index}_step<N>
- Preserve relevant files from input that map to a failure, hidden failure, validation step, or supporting fix
- Fix strategy required for each problem
- Clear, simple language
"""
        try:
            result = _invoke_json(llm, prompt)

            # Handle case where LLM returns array instead of object
            if isinstance(result, list):
                if len(result) == 1 and isinstance(result[0], dict):
                    result = result[0]
                else:
                    raise ValueError(
                        f"Batch consolidation returned array with {len(result)} items, expected single object"
                    )

            if not isinstance(result, dict):
                raise ValueError(f"Batch consolidation returned {type(result).__name__}, expected object")
            result.setdefault("batch_index", batch_index)
            result.setdefault("candidate_problems", [])
            batch_summaries.append(result)
        except Exception as exc:
            batch_summaries.append({
                "batch_index": batch_index,
                "candidate_problems": [],
                "summary": f"Batch consolidation failed: {exc}",
            })

    final_prompt = f"""You are creating the final diff analysis summary from batch summaries.

Use ONLY the inputs provided. Do NOT invent information.

═══════════════════════════════════════════════════════════════════════════════
INPUTS
═══════════════════════════════════════════════════════════════════════════════

CI VALIDATION SEQUENCE:
{json.dumps(validation_sequence, indent=2)}

VISIBLE FAILURE CONTEXT:
{json.dumps(visible_failure_context, indent=2)}

BATCH SUMMARIES:
{json.dumps(batch_summaries, indent=2)}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK: Merge Batches into Final Summary
═══════════════════════════════════════════════════════════════════════════════

Merge candidate problems from all batches:
- Combine duplicates (same validation step + same root cause)
- Keep different problems separate
- Preserve all files
- Preserve actionable fix strategies

═══════════════════════════════════════════════════════════════════════════════
{STRICT_JSON_RULES}

IMPORTANT: Return a SINGLE JSON OBJECT (not an array). Your response must start with {{ and end with }}.

OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

{{
  "problems_by_validation_step": [
    {{
      "validation_step": 1,
      "validation_cmd": "ruff check",
      "what_validates": "Code style and imports",
      "visibility": "visible | hidden | unclear",

      "problem_statement": "Clear problem description with root cause explanation why it failed at this step",


      "affected_files": [
        {{
          "file": "path/to/file.py",
          "what_is_wrong": "Specific issue",
          "fix_strategy": "What needs to be modified in this file"
        }}
      ]
    }}
  ],

  "summary": "One paragraph summary of all changes"
}}

RULES:
- One entry per validation step
- Merge duplicates within same step
- Preserve relevant files that map to a failure, hidden failure, validation step, or supporting fix
- Fix strategy required
- Clear, simple language
"""
    try:
        summary = _invoke_json(llm, final_prompt)

        # Handle case where LLM returns array instead of object
        if isinstance(summary, list):
            if len(summary) == 1 and isinstance(summary[0], dict):
                summary = summary[0]
            else:
                raise ValueError(
                    f"Final consolidation returned array with {len(summary)} items, expected single object"
                )

        if not isinstance(summary, dict):
            raise ValueError(f"Final consolidation returned {type(summary).__name__}, expected object")
        summary.setdefault("problems_by_validation_step", [])
        summary.setdefault("summary", "")
    except Exception as exc:
        summary = {
            "problems_by_validation_step": [],
            "summary": f"Final consolidation failed: {exc}",
        }

    return {
        "consolidated": True,
        "batch_count": len(batches),
        "batch_summaries": batch_summaries,
        "diff_analysis_summary": summary,
    }


def analyze_diff_chunks(
    issue: Dict,
    benchmark_context: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """
    Run independent file-aware diff chunk analysis before final reasoning.

    Each chunk sees only visible failure context, validation sequence, and its
    own complete file-diff sections. Cross-chunk reasoning happens later in the
    final decomposition prompt.
    """
    diff = str(issue.get("diff") or "")
    if not diff.strip():
        raise ValueError(f"Issue {issue.get('id')} has no ground-truth diff")

    chunks = chunk_diff_by_file(diff)
    if not chunks:
        raise ValueError(f"Issue {issue.get('id')} ground-truth diff could not be chunked")

    visible_failure_context = _compact_context_for_diff_analysis(issue, benchmark_context)
    validation_sequence = benchmark_context.get("validation_sequence") or []
    chunk_findings: List[Dict[str, Any]] = []

    print(f"  Analyzing ground-truth diff ({len(diff)} chars) in {len(chunks)} complete-file chunk(s)...")

    for index, chunk in enumerate(chunks, start=1):
        oversized_note = (
            "This chunk contains one complete file diff that exceeds the normal chunk size. "
            "It is intentionally NOT truncated."
            if chunk.get("oversized_single_file")
            else "This chunk contains complete file diff sections and is not truncated."
        )
        prompt = f"""You are pre-analyzing one chunk of a ground-truth diff for CI repair reverse engineering.

CRITICAL: Provide CONCRETE, ACTIONABLE details in your analysis:
- Extract line numbers from diff hunks (@@ -X,Y +A,B @@)
- Include before/after code snippets (2-3 lines, not full diff)
- Identify specific error messages or validation failures
- Explain WHY each change fixes the problem with technical reasoning
- If 5+ files have identical changes, note the PATTERN explicitly

Use ONLY the visible failure context, validation sequence, and current diff chunk.
Do NOT invent files, workflow steps, failures, or fixes.

Goal:
- Extract what this diff chunk appears to fix WITH CONCRETE EXAMPLES.
- Group changes into candidate atomic problems.
- Preserve exact file paths and concrete diff evidence.
- For each file: note line numbers, what was wrong, what changed, why it fixes the issue.
- Note whether each candidate seems related to the visible CI failure or a hidden downstream failure.
- Only return files/changes that have evidence linking them to:
  1. the visible CI failure context, or
  2. a command/stage in the provided CI validation sequence, or
  3. a hidden downstream CI validation failure that would appear after earlier fixes.
- If a file/change cannot be a reason for any provided CI validation command to pass/fail, ignore it completely.
- Do not include ignored files in files_analyzed, affected_files, or any candidate problem.
- A file is relevant when its path, file type, generated artifact, configuration role, dependency role, or test/code/docs role matches a command or target in the provided validation sequence.
- Decide relevance dynamically from the actual validation commands and their path arguments; do not rely on hardcoded file extensions or tool names.
- Include a file only when the diff evidence explains how that file could make one of the provided validation commands pass or fail.
- Large repairs may have many files for one atomic problem. Group same-pattern file changes, but do not merge unrelated problem types.

--- CI FAILURE CONTEXT ---
{json.dumps(visible_failure_context, indent=2)}

--- CI VALIDATION SEQUENCE ---
{json.dumps(validation_sequence, indent=2)}

--- CURRENT DIFF CHUNK {index}/{len(chunks)} ---
Files in this chunk: {', '.join(chunk.get("files", []))}
Chunk metadata: file_count={chunk.get("file_count", 0)}, char_count={chunk.get("char_count", len(chunk.get("diff", "")))}, oversized_single_file={bool(chunk.get("oversized_single_file"))}
{oversized_note}

{chunk["diff"]}

{STRICT_JSON_RULES}

IMPORTANT: Return a SINGLE JSON OBJECT (not an array). Your response must start with {{ and end with }}.

Return this JSON object with CONCRETE details:
{{
  "chunk_index": {index},
  "files_analyzed": ["only/files/with/ci-validation-evidence.py"],
  "candidate_atomic_problems": [
    {{
      "candidate_id": "c{index}_1",
      "problem_type": "short dynamic problem category",
      "visibility_guess": "visible_candidate | hidden_candidate | unclear",
      "symptom_inferred": "CONCRETE symptom with error message or line reference. Example: 'Line 12: F401 unused import Qux' not just 'import error'",
      "affected_files": [
        {{
          "file": "path/to/file.py",
          "line_range": "lines 45-52 or line 12",
          "what_was_wrong": "Specific issue at specific location. Example: 'Import statement includes unused Qux at line 12'",
          "before_snippet": "from foo import Bar, Baz, Qux",
          "after_snippet": "from foo import Bar, Baz",
          "why_wrong": "Root cause. Example: 'Ruff F401: Qux imported but never referenced'",
          "how_fixed": "What changed. Example: 'Removed Qux from import list'",
          "why_fix_works": "Technical reasoning. Example: 'Eliminating unused import satisfies ruff F401 rule'"
        }}
      ],
      "root_cause_hypothesis": "Evidence-based failure cause with technical explanation",
      "fix_summary": "What the changes made to fix it",
      "verification_command": "Command from validation_sequence to verify, or null",
      "pattern_detected": {{
        "is_pattern": false,
        "pattern_type": "remove_unused_imports | rst_title_fix | type_annotation_update | etc",
        "files_with_pattern": 1,
        "explanation": "If 5+ files have same fix, describe the pattern"
      }}
    }}
  ]
}}
"""
        try:
            finding = _invoke_json(llm, prompt)

            # Handle case where LLM returns array instead of object
            if isinstance(finding, list):
                if len(finding) == 1 and isinstance(finding[0], dict):
                    # LLM wrapped the object in an array, unwrap it
                    finding = finding[0]
                elif len(finding) > 1:
                    # LLM returned multiple objects, merge them
                    merged = {
                        "chunk_index": index,
                        "files_analyzed": [],
                        "candidate_atomic_problems": [],
                    }
                    for item in finding:
                        if isinstance(item, dict):
                            merged["files_analyzed"].extend(item.get("files_analyzed", []))
                            merged["candidate_atomic_problems"].extend(
                                item.get("candidate_atomic_problems", [])
                            )
                    finding = merged
                else:
                    # Empty array or invalid
                    raise ValueError(f"Chunk analysis returned empty or invalid array")

            if not isinstance(finding, dict):
                raise ValueError(
                    f"Chunk analysis returned {type(finding).__name__}, expected object. "
                    f"Value: {str(finding)[:200]}"
                )

            finding.setdefault("chunk_index", index)
            finding.setdefault("files_analyzed", chunk.get("files", []))
            finding.setdefault("candidate_atomic_problems", [])
        except Exception as exc:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Chunk {index}/{len(chunks)} JSON parsing failed: {exc}")
            print(f"    Error type: {type(exc).__name__}")
            print(f"    Full trace:\n{error_trace}")
            finding = {
                "chunk_index": index,
                "files_analyzed": [],
                "candidate_atomic_problems": [],
                "error_trace": error_trace,
            }
        chunk_findings.append(finding)

    consolidated = consolidate_chunk_findings(issue, benchmark_context, chunk_findings, llm)

    return {
        "mode": "file_chunk_diff_analysis",
        "diff_char_count": len(diff),
        "chunk_count": len(chunks),
        "chunk_findings": chunk_findings,
        "consolidated_chunk_findings": consolidated,
    }


def _build_decomposition_prompt(
    issue: Dict,
    benchmark_context: Dict[str, Any],
    *,
    diff_block_title: str,
    diff_instruction: str,
    diff_block: str,
    mode_rule: str,
) -> str:
    """
    Build prompt to REVERSE ENGINEER atomic problems from:
    - CI failure log (shows FIRST failure only)
    - Ground truth diff (fixes ALL problems)
    - Changed files
    - CI workflow context

    Key: CI stops at first failure, but diff fixes multiple problems.
    We must infer hidden problems!
    """

    repo = issue.get("repo_name", issue.get("repo", "unknown"))
    sha = issue.get("sha_fail", "")[:12]
    error_type = issue.get("error_type", [])
    if isinstance(error_type, list):
        error_type = ", ".join(error_type)

    # Get changed files
    changed_files = issue.get("changed_files", [])

    context = benchmark_context.get("context") or {}
    log_analysis = benchmark_context.get("log_analysis") or {}
    validation_sequence = benchmark_context.get("validation_sequence") or []
    workflow_path = benchmark_context.get("workflow_path") or ""
    workflow_name = benchmark_context.get("workflow_name") or ""

    analyzer_context = {
        "id": context.get("id"),
        "sha_fail": context.get("sha_fail"),
        "repo": context.get("repo"),
        "overall_failure_reasons": context.get("overall_failure_reasons", []),
        "overall_error_types": context.get("overall_error_types", []),
        "effected_files": context.get("effected_files", []),
        "failed_jobs": context.get("failed_jobs", []),
        "overall_ci_summary": log_analysis.get("overall_ci_summary", ""),
        "error_context": log_analysis.get("error_context", []),
        "relevant_files": log_analysis.get("relevant_files", []),
        "error_types": log_analysis.get("error_types", []),
        "failed_job": log_analysis.get("failed_job", []),
    }
    return f"""You are a CI failure analysis expert doing REVERSE ENGINEERING.

CRITICAL CONTEXT:
- The structured CI context below comes from CILogAnalyzer / benchmark CI analysis.
- It represents the FIRST observed CI failure context because CI stops at first error.
- The ground truth diff fixes ALL problems (visible + hidden)
- Your job: first build a clear case analysis, then infer HIDDEN problems from the diff
- Use ONLY the provided structured CI context, ground-truth diff, changed files, and validation_sequence.
- Do NOT invent workflow steps, commands, files, failures, or fixes. If evidence is missing, use null or explain the uncertainty.
- {mode_rule}

════════════════════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════════════════════

Repository: {repo}
Commit: {sha}
Error Type: {error_type}
Changed Files ({len(changed_files)}): {', '.join(changed_files)}
Workflow Name: {workflow_name}
Workflow Path: {workflow_path}

--- STRUCTURED CI FAILURE CONTEXT FROM CILogAnalyzer ---
{json.dumps(analyzer_context, indent=2)}

--- {diff_block_title} ---
{diff_instruction}

{diff_block}

--- CI VALIDATION SEQUENCE ---
{json.dumps(validation_sequence, indent=2)}

════════════════════════════════════════════════════════════════════════════════
	YOUR TASK: REVERSE ENGINEER ATOMIC PROBLEMS ONLY
	════════════════════════════════════════════════════════════════════════════════
	
	The structured CI context shows Problem 1. The chunk-level diff analysis shows what the
	ground-truth patch changed across all files. Your job is to analyze ALL chunk findings
	together against the ordered CI validation_sequence.
	
	Use validation_sequence as the ordering spine:
	- The first visible problem must map to the earliest failed validation evidenced by the structured CI context.
	- Hidden problems must map to later validation commands that would run only after earlier blockers are fixed.
	- If several problems map to the same validation command, keep them separate only when their root cause/fix pattern differs.
	- If a later file change does not correspond to any validation step, mark it supporting/unclear instead of inventing a failure.
	
	The diff changes {len(changed_files)} files. This may mean multiple problems were fixed.

Identify:
- What failed first?
- Which workflow step and command failed?
- Which files and diff hunks fix THIS problem?
- Look at changed files not explained by the visible failure.
- Infer which hidden problems those files fix.
	- Map each hidden problem to the ordered validation step that WOULD fail after earlier fixes.
	- Infer the sequence: after fixing each problem, which validation command runs next and what would fail/pass.
	- Evidence for each problem: CI analyzer evidence for visible problem, diff evidence for all problems, workflow evidence for mapped stages.
- Ignore files that do not map to visible or hidden CI validation failures.
- For large repairs with many changed files, group same-pattern changes into one atomic problem, but keep unrelated failures separate.
- Do not overfit to each file as a separate problem; atomic problem means one distinct failure/root-cause/fix pattern.

**Example reasoning:**
Structured CI context: "ERROR: No matching distribution for fish-audio-sdk>=2024.12.5"
→ Problem 1 (VISIBLE): Dependency constraint
→ Files: pyproject.toml
→ CI stage: pip install

Diff ALSO changes: src/audio/*.py (11 files, type hints changed)
→ Problem 2 (HIDDEN): Type errors after SDK upgrade
→ Would fail at: mypy src/
→ Depends on: Problem 1 (can't check types until SDK installable)

	Diff ALSO changes: tests/test_audio.py
	→ Problem 3 (HIDDEN): Test expectations outdated
	→ Would fail at: pytest tests/
	→ Depends on: Problem 2
	
	════════════════════════════════════════════════════════════════════════════════
	REQUIRED REASONING STEPS
	════════════════════════════════════════════════════════════════════════════════
	
	1. Read the validation_sequence in order.
	2. Match the structured CI failure context to the first failed validation command.
	3. Group chunk-level diff findings by validation command/root-cause/fix pattern.
	4. Decide which group fixes the visible failure.
	5. For remaining groups, infer hidden downstream failures by validation order.
	6. Build a workflow-ordered problem flow:
	   - current problem
	   - fix
	   - validation command to run
	   - expected result
	   - next problem exposed, if any
	7. Keep only files relevant to visible or hidden CI validation failures.

════════════════════════════════════════════════════════════════════════════════
{STRICT_JSON_RULES}

IMPORTANT: Return a SINGLE JSON OBJECT (not an array). Your response must start with {{ and end with }}.

OUTPUT FORMAT
════════════════════════════════════════════════════════════════════════════════

{{
  "total_problems": 2,
  "total_changed_files": {len(changed_files)},
  "problems": [
    {{
      "problem_id": 1,
      "visibility": "visible_in_log",
      "problem_type": "short dynamic category",
      "symptom": "specific visible or inferred failure",
      "evidence_in_ci_log": "structured CI evidence for visible problem, else null",
      "root_cause": "concise root cause",
      "how_fixed": "concise fix summary",
      "why_fix_works": "why this fix satisfies the validation",
      "affected_files": ["path/to/file.py"],
      "fix_strategy": "what needs to be modified to fix this problem",
      "workflow_validation_order": 1,
      "ci_workflow_step": "validation name from validation_sequence",
      "ci_command": "exact validation or install command",
      "workflow_evidence": "validation_sequence evidence, or null",
      "depends_on": null,
      "dependency_reason": null,
      "verification_after_fix": "exact command to run",
      "next_failure_after_fix": "next problem summary or null"
    }}
  ],
  "workflow_ordered_problem_flow": [
    {{
      "sequence_step": 1,
      "problem_id": 1,
      "validation_order": 1,
      "validation_command": "command from validation_sequence",
      "visibility": "visible_in_log | hidden",
      "fix_summary": "short fix summary",
      "expected_after_fix": "pass | would expose next problem | unknown",
      "next_problem_id": 2
    }}
  ],
  "overall_failure_summary": "one concise paragraph",
  "workflow_reasoning": "one concise paragraph explaining validation order and hidden failures"
}}

════════════════════════════════════════════════════════════════════════════════
CRITICAL RULES
════════════════════════════════════════════════════════════════════════════════

1. Problem 1 MUST have visibility="visible_in_log" and evidence_in_ci_log from structured CI context
2. Problems 2, 3, ... MUST have visibility="hidden" and evidence_in_ci_log=null
3. Infer hidden problems from:
   - Changed files not related to visible problem
   - Actual ordered CI validation sequence from the benchmark workflow
   - File types (config vs code vs tests)
4. If 10 files have same type of change (type hints), that's ONE atomic problem, not 10
5. Include depends_on/dependency_reason when a hidden problem only appears after an earlier problem is fixed
6. Keep output compact. Do not create per-file nested records.
7. Make each problem actionable through root_cause, how_fixed, affected_files, fix_strategy, and verification_after_fix.
8. If one repair has 3 distinct problems, return 3 distinct atomic problems; do not merge unrelated install/type/test/config fixes
9. workflow_evidence must cite the provided validation_sequence. If no matching workflow evidence exists, set workflow_evidence=null.
10. Do not account for every changed file; include only files relevant to inferred CI validation failures.
11. For documentation-formatting bursts, test-fixture updates, generated artifacts, or broad API migrations, group same-pattern files into one problem and list affected files.
12. workflow_validation_order must match the ordered validation_sequence when there is workflow evidence.
13. workflow_ordered_problem_flow must be ordered by validation_sequence and problem dependencies.
14. Do not order hidden problems by file order or chunk order. Chunk order is only evidence collection; CI validation_sequence determines repair order.
""".strip()


def format_diff_evidence_for_prompt(diff_context: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert chunk-level diff analysis into the evidence block used by the
    final decomposition prompt.
    """
    consolidated = diff_context.get("consolidated_chunk_findings") or {}
    if consolidated.get("consolidated"):
        findings_payload = {
            "diff_char_count": diff_context.get("diff_char_count"),
            "chunk_count": diff_context.get("chunk_count"),
            "chunks": diff_context.get("chunks", []),
            "diff_analysis_summary": consolidated.get("diff_analysis_summary", {}),
        }
        instruction = (
            "Use the consolidated file-aware diff summary below as the diff evidence. It was "
            "produced from all chunk findings and preserves candidate problems plus relevant "
            "file evidence."
        )
    else:
        findings_payload = {
            "diff_char_count": diff_context.get("diff_char_count"),
            "chunk_count": diff_context.get("chunk_count"),
            "chunks": diff_context.get("chunks", []),
            "chunk_findings": consolidated.get("chunk_findings", diff_context.get("chunk_findings", [])),
        }
        instruction = (
            "Use the file-aware chunk findings below as the diff evidence. Reconcile candidate "
            "problems across chunks before deciding the final atomic problems."
        )

    return {
        "title": "GROUND TRUTH DIFF CHUNK ANALYSIS",
        "instruction": instruction,
        "block": json.dumps(findings_payload, indent=2),
        "mode_rule": (
            "This prompt contains chunk-level diff analysis, not raw diff text. Use the chunk "
            "evidence and do not assume raw diff lines beyond those findings."
        ),
    }


def build_decomposition_prompt(
    issue: Dict,
    benchmark_context: Dict[str, Any],
    diff_context: Dict[str, Any],
) -> str:
    diff_evidence = format_diff_evidence_for_prompt(diff_context)
    return _build_decomposition_prompt(
        issue,
        benchmark_context,
        diff_block_title=diff_evidence["title"],
        diff_instruction=diff_evidence["instruction"],
        diff_block=diff_evidence["block"],
        mode_rule=diff_evidence["mode_rule"],
    )


def build_repair_trajectory_prompt(
    issue: Dict,
    benchmark_context: Dict[str, Any],
    atomic_result: Dict[str, Any],
) -> str:
    """Build a second prompt that derives repair trajectory from atomic problems."""
    validation_sequence = benchmark_context.get("validation_sequence") or []
    workflow_path = benchmark_context.get("workflow_path") or ""
    context = benchmark_context.get("context") or {}

    compact_context = {
        "workflow_path": workflow_path,
        "validation_sequence": validation_sequence,
        "failed_jobs": context.get("failed_jobs", []),
        "overall_failure_reasons": context.get("overall_failure_reasons", []),
    }

    return f"""You are building a CI repair trajectory from already extracted atomic problems.

Use ONLY:
- the atomic problems JSON
- workflow_ordered_problem_flow from the atomic problems JSON, if present
- the actual ordered validation_sequence extracted from the benchmark workflow
- the visible CI failure context

Do NOT invent new atomic problems. Do NOT change problem root causes or affected files.
Your job is only to order the problems through the CI pipeline and explain what runs/fails after each fix.
If workflow_ordered_problem_flow is present, use it as the primary ordering evidence and convert it into the detailed repair_trajectory_inference.
If multiple problems map to the same workflow stage, keep them as separate trajectory steps only if they have distinct root causes/fix patterns; otherwise preserve their single atomic problem.

--- ATOMIC PROBLEMS JSON ---
{json.dumps(atomic_result, indent=2)}

--- CI CONTEXT ---
{json.dumps(compact_context, indent=2)}

--- CI VALIDATION SEQUENCE ---
{json.dumps(validation_sequence, indent=2)}

{STRICT_JSON_RULES}

IMPORTANT: Return a SINGLE JSON OBJECT (not an array). Your response must start with {{ and end with }}.

Return this JSON object:
{{
  "ci_workflow_stages": [
    {{
      "stage_order": 1,
      "stage_name": "install | lint | type_check | test | build | other",
      "workflow_step": "Workflow step name",
      "command": "exact command from validation_sequence",
      "validation_purpose": "what this command validates",
      "would_run_after": null
    }}
  ],

  "repair_trajectory_inference": [
    {{
      "step": 1,
      "problem_fixed": 1,
      "problem": "p1",
      "problem_summary": "short summary of the atomic problem fixed in this step",
      "visible_failure": "failure seen or inferred at this step",
      "observability": "visible_in_log | hidden_inferred_from_diff",
      "ci_stage": "install | lint | type_check | test | build | other",
      "workflow_stage_order": 1,
      "validation_command_to_run_now": "command to verify this fix",
      "files_to_fix": [
        {{
          "file": "path/to/file.py",
          "change_needed": "actual fix from atomic problem files[]",
          "why_needed": "failure/risk in this file"
        }}
      ],
      "fix": "short fix summary from atomic problem",
      "result_after_fix": "success expected after this fix",
      "next_validation_to_run": {{
        "workflow_stage_order": 2,
        "command": "next command from validation_sequence",
        "validates": "what next command validates"
      }},
      "expected_result": "WOULD FAIL with p2 | PASS | unknown",
      "expected_next_failure": {{
        "problem_id": 2,
        "why_it_appears_next": "workflow order/dependency reason",
        "files_likely_involved": ["src/f1.py"]
      }}
    }}
  ],

  "problem_links": [
    {{
      "from_problem": 1,
      "to_problem": 2,
      "link_type": "unblocks | depends_on | sequence",
      "reason": "why fixing one problem exposes/enables the next workflow failure"
    }}
  ],

  "repair_trajectory_summary": "Concise explanation of the full repair order through the CI workflow."
}}

Rules:
1. Keep the same problem IDs from atomic problems.
2. The first trajectory step must be the visible CI failure.
3. Hidden problems should appear only after the earlier workflow-blocking problem is fixed.
4. validation and next_ci_stage_to_run must come from validation_sequence when available.
5. If a problem has no workflow command evidence, use null and explain uncertainty in the summary.
6. Do not add new problem IDs. The trajectory is an ordering of existing atomic problems only.
7. If two hidden problems would be exposed by the same validation command, order them by dependency evidence from depends_on; if no dependency exists, keep workflow_stage_order equal and explain they are same-stage findings.
8. files_to_fix must come from atomic problems' files[]/affected_files; do not invent files.
""".strip()


def add_coverage_diagnostics(issue: Dict, result: Dict[str, Any]) -> None:
    changed_files = [str(f) for f in issue.get("changed_files", []) if str(f).strip()]
    covered = set()
    for problem in result.get("problems", []):
        for file_path in problem.get("affected_files", []) or []:
            covered.add(str(file_path))
    for item in result.get("unmapped_or_supporting_files", []) or []:
        if isinstance(item, dict):
            covered.add(str(item.get("file") or ""))
        else:
            covered.add(str(item))

    missing = [f for f in changed_files if f not in covered]
    result["coverage_diagnostics"] = {
        "changed_files_total": len(changed_files),
        "covered_files_total": len([f for f in covered if f]),
        "missing_changed_files": missing,
        "coverage_complete": not missing,
    }


def finalize_decomposition_result(
    issue: Dict[str, Any],
    result: Dict[str, Any],
    trajectory_result: Dict[str, Any],
    benchmark_context: Dict[str, Any],
    diff_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach only the normalized fields needed by memory building/retrieval.

    The LLM-produced decomposition remains the source of truth. This helper
    only adds trajectory output, metadata, compact context, and compatibility
    aliases used by the current memory builder.
    """
    # Simple trajectory summary
    problems_list = result.get("problems", [])
    visible_count = sum(1 for p in problems_list if p.get("visibility") == "visible_in_log")

    result["trajectory_summary"] = {
        "total_problems": len(problems_list),
        "visible_problems": visible_count,
        "hidden_problems": len(problems_list) - visible_count,
        "repair_sequence": trajectory_result.get("repair_trajectory_summary", ""),
        "problem_links": trajectory_result.get("problem_links", []),
    }
    result.setdefault("total_changed_files", len(issue.get("changed_files", [])))
    add_coverage_diagnostics(issue, result)

    context = benchmark_context.get("context", {})
    log_analysis = benchmark_context.get("log_analysis", {})
    result.update({
        "original_issue_id": issue.get("id"),
        "sha_fail": issue.get("sha_fail"),
        "repo": issue.get("repo_name", issue.get("repo")),
        "original_error_type": issue.get("error_type"),
        "benchmark_ci_context": {
            "workflow_path": benchmark_context.get("workflow_path"),
            "workflow_name": benchmark_context.get("workflow_name"),
            "workflow_validation_context": benchmark_context.get("workflow_validation_context", {}),
            "validation_sequence": benchmark_context.get("validation_sequence", []),
            "overall_failure_reasons": context.get("overall_failure_reasons", []),
            "overall_error_types": context.get("overall_error_types", []),
            "effected_files": context.get("effected_files", []),
            "failed_jobs": context.get("failed_jobs", []),
            "log_analysis": {
                "error_context": log_analysis.get("error_context", []),
                "relevant_files": log_analysis.get("relevant_files", []),
                "error_types": log_analysis.get("error_types", []),
                "failed_job": log_analysis.get("failed_job", []),
                "overall_ci_summary": log_analysis.get("overall_ci_summary", ""),
                "analysis_document": log_analysis.get("analysis_document", ""),
                "chunk_summaries": log_analysis.get("chunk_summaries", []),
            },
        },
        "diff_analysis_context": {
            "mode": diff_context.get("mode"),
            "diff_char_count": diff_context.get("diff_char_count"),
            "chunk_count": diff_context.get("chunk_count", 0),
            "chunks": diff_context.get("chunks", []),
        },
    })
    return result


def reverse_engineer_atomic_problems(
    issue: Dict[str, Any],
    benchmark_context: Dict[str, Any],
    diff_context: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """Run the first reasoning pass: visible failure + hidden atomic problems."""
    prompt = build_decomposition_prompt(issue, benchmark_context, diff_context)
    return _invoke_json(llm, prompt)


def infer_repair_trajectory(
    issue: Dict[str, Any],
    benchmark_context: Dict[str, Any],
    atomic_result: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """Run the second reasoning pass: order atomic problems by CI validation sequence."""
    prompt = build_repair_trajectory_prompt(issue, benchmark_context, atomic_result)
    result = _invoke_json(llm, prompt)

    # Handle case where LLM returns array instead of object
    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], dict):
            result = result[0]
        else:
            # Invalid response, return empty structure
            print(f"  Warning: Trajectory inference returned array with {len(result)} items, expected object")
            return {
                "ci_workflow_stages": [],
                "repair_trajectory_inference": [],
                "problem_links": [],
                "repair_trajectory_summary": ""
            }

    if not isinstance(result, dict):
        print(f"  Warning: Trajectory inference returned {type(result).__name__}, expected object")
        return {
            "ci_workflow_stages": [],
            "repair_trajectory_inference": [],
            "problem_links": [],
            "repair_trajectory_summary": ""
        }

    return result


def decompose_issue(issue: Dict, llm) -> Dict:
    """
    Reverse engineer atomic problems from CI failure + ground truth diff.

    Returns decomposed problems with visibility markers:
    - visible_in_log: Problem 1 (from structured CI context)
    - hidden: Problems 2+ (inferred from diff)
    """

    issue_id = issue.get('id', '?')
    print(f"\n{'='*80}")
    print(f"Reverse Engineering Issue {issue_id}")
    print(f"  Repo: {issue.get('repo_name', issue.get('repo', '?'))}")
    print(f"  Changed files: {len(issue.get('changed_files', []))}")
    print(f"{'='*80}")

    try:
        print(f"  Fetching benchmark CI context and validation sequence...")
        benchmark_context = build_benchmark_ci_context(
            issue,
            llm=llm,
        )
        if not validate_required_ci_inputs(benchmark_context):
            return {}

        diff_context = analyze_diff_chunks(issue, benchmark_context, llm)
        print(f"  Calling LLM to reverse engineer atomic problems...")
        result = reverse_engineer_atomic_problems(issue, benchmark_context, diff_context, llm)
        if not result:
            print("  ✗ Atomic problem JSON parsing failed; returning empty result")
            return {}

        print(f"  Calling LLM to build repair trajectory...")
        trajectory_result = infer_repair_trajectory(issue, benchmark_context, result, llm)
        return finalize_decomposition_result(
            issue=issue,
            result=result,
            trajectory_result=trajectory_result,
            benchmark_context=benchmark_context,
            diff_context=diff_context,
        )

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"  ✗ Failed to decompose: {e}")
        print(f"\n--- FULL ERROR TRACE ---")
        print(error_trace)
        print(f"--- END TRACE ---\n")
        return {
            "error": "DECOMPOSITION_ERROR",
            "error_message": str(e),
            "error_trace": error_trace,
            "error_type": type(e).__name__,
            "original_issue_id": issue.get("id"),
            "sha_fail": issue.get("sha_fail"),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Reverse engineer CI failures into atomic problems (visible + hidden)"
    )
    parser.add_argument("--issue-id", help="Single issue ID to decompose")
    parser.add_argument("--batch", action="store_true", help="Decompose all eval issues")
    parser.add_argument("--eval-issues", default="data/trs/eval_issues.json", help="Path to eval issues")
    parser.add_argument("--output", default="data/trs/decomposed_issues.json", help="Output file")
    parser.add_argument(
        "--model",
        default="openrouter/minimax/minimax-m2.5",
        help="LLM model. Use openrouter/minimax/minimax-m2.5 for MiniMax M2.5 via OpenRouter.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of issues to process")
    args = parser.parse_args()

    # Load eval issues
    eval_path = Path(args.eval_issues)
    if not eval_path.exists():
        print(f"✗ Eval issues not found: {eval_path}")
        return 1

    with open(eval_path) as f:
        issues = json.load(f)

    print(f"Loaded {len(issues)} issues from {eval_path}")

    # Filter if specific issue requested
    if args.issue_id:
        issues = [i for i in issues if str(i.get("id")) == args.issue_id]
        if not issues:
            print(f"✗ Issue {args.issue_id} not found")
            return 1

    # Limit if requested
    if args.limit:
        issues = issues[:args.limit]
        print(f"Limited to first {args.limit} issues")

    # Initialize LLM
    print(f"\n{'='*80}")
    print(f"Initializing LLM: {args.model}")
    print(f"{'='*80}")
    llm = LitellmModel(model_name=args.model)

    # Decompose issues
    results = []
    errors = []

    for i, issue in enumerate(issues, 1):
        print(f"\nProgress: {i}/{len(issues)}")
        result = decompose_issue(issue, llm)

        if "error" in result:
            errors.append(result)

        results.append(result)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total issues processed: {len(results)}")
    print(f"Successful: {len(results) - len(errors)}")
    print(f"Errors: {len(errors)}")

    # Count problems
    successful = [r for r in results if "total_problems" in r]
    total_problems = sum(r.get("total_problems", 0) for r in successful)
    visible_problems = sum(
        sum(1 for p in r.get("problems", []) if p.get("visibility") == "visible_in_log")
        for r in successful
    )
    hidden_problems = total_problems - visible_problems

    print(f"\nAtomic problems identified:")
    print(f"  Total: {total_problems}")
    print(f"  Visible (in structured CI context): {visible_problems}")
    print(f"  Hidden (inferred): {hidden_problems}")

    if successful:
        avg_problems = total_problems / len(successful)
        print(f"  Average per issue: {avg_problems:.1f}")

    # Problem type distribution
    problem_types = {}
    for r in successful:
        for p in r.get("problems", []):
            ptype = p.get("problem_type", "unknown")
            problem_types[ptype] = problem_types.get(ptype, 0) + 1

    if problem_types:
        print(f"\nProblem type distribution:")
        for ptype, count in sorted(problem_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ptype}: {count}")

    print(f"\nOutput saved to: {output_path}")

    if errors:
        print(f"\n⚠️  {len(errors)} issues had errors")
        print(f"Issue IDs with errors: {[e.get('original_issue_id') for e in errors[:5]]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
