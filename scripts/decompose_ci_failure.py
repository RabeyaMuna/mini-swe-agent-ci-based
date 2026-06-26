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
import difflib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from deterministic_diff_parser import (
    parse_diff_to_structured,
    chunk_structured_diff,
    format_structured_for_llm
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import litellm
from dotenv import load_dotenv
from datasets import load_dataset
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

# Load memory issue IDs from workflow_validation_cache.json
def _load_memory_issue_ids() -> List[str]:
    """Load issue IDs from workflow_validation_cache.json."""
    cache_path = PROJECT_ROOT / "data" / "trs" / "workflow_validation_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
                issue_ids = [str(item['id']) for item in cache if 'id' in item]
                return issue_ids
        except Exception as e:
            print(f"Warning: Could not load workflow_validation_cache.json: {e}")
            return ['121']  # Fallback to default
    return ['121']  # Fallback to default

# Load memory issue IDs dynamically
MEMORY_ISSUE_IDS = _load_memory_issue_ids()

DIFF_CHUNK_CHAR_LIMIT = int(os.getenv("DIFF_CHUNK_CHAR_LIMIT", "30000"))
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

        try:
            import time
            start_time = time.time()

            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=16000,  # MiniMax M2.5 supports up to ~196k, using 16k for detailed analysis
            )

            elapsed = time.time() - start_time

            # Check for error or length finish_reason
            finish_reason = getattr(response.choices[0], 'finish_reason', None)

            # DEBUG: Always log finish_reason and token usage
            usage = getattr(response, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', '?')
                completion_tokens = getattr(usage, 'completion_tokens', '?')
                total_tokens = getattr(usage, 'total_tokens', '?')
                print(f"      [API] finish_reason={finish_reason}, tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
            else:
                print(f"      [API] finish_reason={finish_reason}, no usage data")

            if finish_reason == 'error':
                error_msg = getattr(response, 'error', 'Unknown error')
                if hasattr(response, '_hidden_params'):
                    error_msg += f" | Details: {response._hidden_params}"
                LOGGER.error(f"LLM error after {elapsed:.1f}s. Error: {error_msg}")
                print(f"    FAIL LLM Error ({elapsed:.1f}s): {error_msg}")
            elif finish_reason == 'length':
                # Log detailed info for length errors
                LOGGER.warning(f"Hit length limit: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
                print(f"    WARNING Length limit hit!")
                print(f"       Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
                print(f"       max_tokens setting: 16000")
                print(f"       Chunk too large - reduce max_changes_per_chunk or simplify prompt")

            class Result:
                content = response.choices[0].message.content or ""
                raw_response = response

            return Result()
        except Exception as e:
            LOGGER.error(f"LiteLLM API call failed: {type(e).__name__}: {e}")
            print(f"    FAIL API Error: {type(e).__name__}: {str(e)[:200]}")

            class Result:
                content = ""
                error = str(e)
                raw_response = None

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


def _clean_malformed_json(content: str) -> str:
    """Clean common LLM JSON formatting mistakes before parsing."""
    content = str(content or "").strip()
    content = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', content, flags=re.DOTALL)
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    content = re.sub(r',\s*,', ',', content)
    content = re.sub(r'}\s*{', '}, {', content)
    content = re.sub(r'}\s*\[', '}, [', content)
    content = re.sub(r']\s*{', '], {', content)
    return content.strip()


def _load_llm_json(content: str) -> Any:
    """Parse raw LLM JSON. Handles markdown fences and explanatory text."""
    content = str(content or "").strip()

    if not content:
        return []

    candidates = [
        content,
        _extract_json_from_text(content),
        _clean_malformed_json(content),
        _clean_malformed_json(_extract_json_from_text(content)),
    ]

    last_json_err: Any = None
    last_demjson3_err: Any = "demjson3 is not installed"

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_json_err = exc

        # Some models emit several JSON objects back-to-back instead of an array.
        try:
            decoder = json.JSONDecoder()
            objects = []
            idx = 0
            while idx < len(candidate):
                tail = candidate[idx:].lstrip()
                if not tail:
                    break
                obj, end = decoder.raw_decode(tail)
                objects.append(obj)
                idx += len(candidate[idx:]) - len(tail) + end
            if len(objects) > 1:
                return objects
            if len(objects) == 1:
                return objects[0]
        except Exception:
            pass

        try:
            if demjson3 is not None:
                return demjson3.decode(candidate)
        except Exception as exc:
            last_demjson3_err = exc

    preview = content[:500] if len(content) > 500 else content
    parse_error = ValueError(
        f"JSON parse failed: json={last_json_err}; demjson3={last_demjson3_err}\n"
        f"Content preview (first 500 chars):\n{preview}"
    )
    LOGGER.warning("%s", parse_error)
    return []


def _invoke_json(llm: Any, prompt: str) -> Any:
    """Invoke LLM and parse JSON with robust error handling."""
    # Check prompt size upfront
    prompt_size_kb = len(prompt) / 1024
    if prompt_size_kb > 80:
        print(f"        WARNING  Large prompt: {prompt_size_kb:.1f}KB - may cause API errors")

    try:
        response = llm.invoke(prompt)
        content = str(getattr(response, "content", response) or "").strip()
    except Exception as exc:
        error_msg = str(exc)

        # Provide specific guidance based on error type
        if "Unable to get json response" in error_msg or "Expecting value" in error_msg:
            print(f"        FAIL API returned malformed/truncated JSON")
            print(f"          Prompt size: {prompt_size_kb:.1f}KB")
            print(f"          -> Chunk too large, reduce max_files_per_chunk")
        elif "rate limit" in error_msg.lower():
            print(f"        FAIL Rate limit - increase delay (currently 3s)")
        else:
            print(f"        FAIL API Error: {type(exc).__name__}: {str(exc)[:200]}")

        LOGGER.error(f"LLM API call failed: {type(exc).__name__}: {exc}")
        return []  # Return empty, continue processing other chunks

    if not content:
        # Get detailed error info if available
        error_details = ""
        finish_reason = None
        if hasattr(response, 'error'):
            error_details = f"Error: {response.error}"
        elif hasattr(response, 'raw_response') and response.raw_response:
            raw = response.raw_response
            if hasattr(raw, 'error'):
                error_details = f"Error: {raw.error}"
            if hasattr(raw.choices[0], 'finish_reason'):
                finish_reason = raw.choices[0].finish_reason
                error_details += f" | Finish reason: {finish_reason}"

        print(f"        FAIL LLM returned empty content")
        print(f"          Prompt size: {len(prompt)} chars ({len(prompt)/1024:.1f}KB)")
        if error_details:
            print(f"          {error_details}")

        # Special handling for length limit: return signal to re-split
        if finish_reason == 'length':
            print(f"          -> Returning 'SPLIT_REQUIRED' signal for auto-retry")
            LOGGER.warning(f"Length limit hit, chunk needs splitting")
            return "SPLIT_REQUIRED"  # Signal to caller to split chunk

        LOGGER.warning(f"LLM empty content. Prompt size: {len(prompt)} chars. {error_details}")
        return []

    parsed = _load_llm_json(content)
    if parsed not in (None, [], {}):
        return parsed

    # One repair pass helps when the model produced almost-valid JSON or was
    # wrapped/truncated. Keep this prompt small.
    LOGGER.warning(f"Initial JSON parse failed, attempting repair. Content preview: {content[:200]}")
    repair_prompt = f"""{STRICT_JSON_RULES}

Repair the following model output into valid JSON only.
Preserve all recoverable keys and values.
If the output is truncated, close the current JSON structure conservatively and omit incomplete trailing items.

--- MODEL OUTPUT TO REPAIR ---
{content[:24000]}
"""
    try:
        repaired_response = llm.invoke(repair_prompt)
        repaired_content = str(getattr(repaired_response, "content", repaired_response) or "").strip()
        repaired = _load_llm_json(repaired_content)
        if repaired not in (None, [], {}):
            LOGGER.info("Recovered malformed JSON with repair prompt")
            return repaired
    except Exception as exc:
        LOGGER.warning("JSON repair prompt failed: %s", exc)

    LOGGER.warning(f"All JSON parsing attempts failed. Returning empty. Original content length: {len(content)}")
    return parsed


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2)


def _compact_diff_for_retry(diff: str, max_chars: int = 12000) -> str:
    """Compact a diff for JSON-retry prompts while preserving file/hunk signals."""
    kept: List[str] = []
    total = 0
    per_file_body_lines = 0
    for line in str(diff or "").splitlines():
        keep = False
        if line.startswith("diff --git "):
            per_file_body_lines = 0
            keep = True
        elif line.startswith(("+++ ", "--- ", "@@ ")):
            keep = True
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            if per_file_body_lines < 8:
                keep = True
                per_file_body_lines += 1
        if not keep:
            continue
        clipped = line[:240]
        kept.append(clipped)
        total += len(clipped) + 1
        if total >= max_chars:
            kept.append("...diff compacted...")
            break
    return "\n".join(kept)



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
            print(f"  WARNING:  Cache load failed: {e}, will re-analyze")

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
            print(f"  WARNING:  Validation cache load failed: {e}, will re-analyze")

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
        try:
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
        except Exception as e:
            print(f"       WARNING: Workflow extraction failed: {e}")
            print(f"       Using fallback: empty validation sequence")
            workflow_validation_context = {
                "workflow_path": str(workflow_path),
                "dependent_files": [],
                "validation_sequence": [],
            }
            validation_sequence = []

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

    # Allow empty validation_sequence as fallback (decomposition will work in simpler mode)
    if not validation_sequence:
        print(f"       WARNING: No validation sequence available, using fallback decomposition mode")

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
        print("  ERROR Missing structured CI context from CILogAnalyzer; skipping decomposition")
    if not has_validation_sequence:
        print("  ERROR Missing CI workflow validation sequence; skipping decomposition")
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


# ============================================================================
# CI-DIFF CORRELATION: Layered Analysis
# ============================================================================

def _is_dependency_file(file_path: str) -> bool:
    """Check if file is a dependency configuration file."""
    dep_files = ['pyproject.toml', 'package.json', 'requirements.txt',
                 'Cargo.toml', 'pom.xml', 'build.gradle', 'setup.py']
    return any(file_path.endswith(f) for f in dep_files)


def _extract_packages_from_error(error_message: str) -> List[str]:
    """Extract package names mentioned in error message."""
    if not error_message:
        return []

    packages = []
    # Common patterns in import errors
    patterns = [
        r"import ['\"]?(\w+)['\"]?",
        r"from ['\"]?(\w+)['\"]?",
        r"package ['\"]?(\w+)['\"]?",
        r"module ['\"]?(\w+)['\"]?",
        r"'(\w+)' not found",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, error_message.lower())
        packages.extend(matches)

    return list(set(packages))


def _extract_package_changes(changes: List[Dict]) -> List[str]:
    """Extract package names from dependency file changes."""
    packages = []
    for change in changes:
        before = change.get('before', '')
        after = change.get('after', '')

        # Look for package names in toml/json format
        for line in [before, after]:
            # Match: package = "version" or "package": "version"
            matches = re.findall(r'["\']?(\w+(?:-\w+)*)["\']?\s*[=:]\s*["\']', line)
            packages.extend(matches)

    return list(set(packages))


def extract_primary_ci_failures(ci_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    LAYER 1: Extract PRIMARY CI failures that actually happened.

    These are the failures that broke CI run.
    Other problems are hidden until these are fixed.
    """
    primary_failures = []
    failure_id = 1

    # From overall_error_types (structured CI analysis)
    for error_type in ci_context.get('overall_error_types', []):
        primary_failures.append({
            "failure_id": failure_id,
            "validation_order": None,  # Will be inferred from validation sequence
            "error_type": error_type,
            "error_category": error_type,
            "error_message": "",
            "severity": "primary",
        })
        failure_id += 1

    # From overall_failure_reasons (detailed descriptions)
    for reason in ci_context.get('overall_failure_reasons', []):
        # Try to extract validation info from reason
        validation_info = {
            "failure_id": failure_id,
            "validation_order": None,
            "error_type": "Unknown",
            "error_message": reason[:500],  # First 500 chars
            "severity": "primary",
        }

        # Try to infer error type from message
        if 'import' in reason.lower() or 'module' in reason.lower():
            validation_info['error_type'] = 'Import Error'
        elif 'type' in reason.lower() and 'check' in reason.lower():
            validation_info['error_type'] = 'Type Checking'
        elif 'format' in reason.lower():
            validation_info['error_type'] = 'Formatting'
        elif 'test' in reason.lower():
            validation_info['error_type'] = 'Test Failure'

        primary_failures.append(validation_info)
        failure_id += 1

    print(f"    Extracted {len(primary_failures)} primary CI failures")
    print(f"      From overall_error_types: {len(ci_context.get('overall_error_types', []))}")
    print(f"      From overall_failure_reasons: {len(ci_context.get('overall_failure_reasons', []))}")

    return primary_failures


def match_failures_with_diff_fixes(
    primary_failures: List[Dict[str, Any]],
    validation_groups: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    LAYER 2: For EACH validation group with changes, determine fix type.

    Simpler approach: Analyze each validation group independently.

    Categories:
    - DIRECT_FIX: Code changes only (normal fix)
    - ENABLEMENT_FIX: Dependency changes only (enables validation)
    - BOTH: Has both dependency and code changes (enablement + secondary)
    """
    matched = []
    groups_data = validation_groups.get('validation_groups', {})

    print(f"    Analyzing {len(groups_data)} validation groups with changes...")

    # For EACH validation group, classify the fix type
    for val_order_str, val_group in groups_data.items():
        all_changes = val_group.get('all_changes', [])

        if not all_changes:
            continue

        # Classify changes
        dep_changes = [c for c in all_changes if _is_dependency_file(c.get('file', ''))]
        code_changes = [c for c in all_changes if not _is_dependency_file(c.get('file', ''))]

        # Determine fix type based on what's present
        if dep_changes and code_changes:
            # Has BOTH - this is an enablement that reveals secondary
            fix_type = "ENABLEMENT_FIX"  # Mark as enablement, cascade detection will handle secondary
        elif dep_changes and not code_changes:
            # ONLY dependency changes - pure enablement
            fix_type = "ENABLEMENT_FIX"
        elif code_changes and not dep_changes:
            # ONLY code changes - direct fix
            fix_type = "DIRECT_FIX"
        else:
            # No changes (shouldn't happen)
            continue

        matched.append({
            "failure_id": len(matched) + 1,
            "fix_type": fix_type,
            "validation_order": int(val_order_str),
            "validation_cmd": val_group.get('validation_cmd', ''),
            "diff_changes": all_changes,
            "dependency_changes": dep_changes,
            "code_changes": code_changes,
            "error_type": val_group.get('failure_type', 'Unknown'),
            "error_message": f"Validation {val_order_str}: {val_group.get('validation_cmd', '')}"
        })

    return matched


def detect_enablement_cascades(
    matched_failures: List[Dict[str, Any]],
    validation_groups: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    LAYER 3: Detect which fixes ENABLE validations and REVEAL secondary failures.

    Logic: If validation has ENABLEMENT_FIX (dependency added)
    AND same validation has code changes too,
    THEN code changes are SECONDARY (revealed by enablement).
    """
    cascades = []
    groups_data = validation_groups.get('validation_groups', {})

    for failure in matched_failures:
        if failure.get('fix_type') != 'ENABLEMENT_FIX':
            continue

        val_order = failure.get('validation_order')
        if val_order is None:
            continue

        val_group = groups_data.get(str(val_order))
        if not val_group:
            continue

        # Check if there are code changes in same validation
        all_changes = val_group.get('all_changes', [])
        code_changes = [c for c in all_changes if not _is_dependency_file(c.get('file', ''))]

        if code_changes:
            # This enablement REVEALS secondary failures
            cascade = {
                "primary_failure_id": failure['failure_id'],
                "validation_order": val_order,
                "validation_cmd": val_group.get('validation_cmd', ''),
                "enablement_type": "dependency_installation",
                "reveals_secondary_failures": True,
                "secondary_changes": code_changes,
                "secondary_file_count": len(set(c.get('file') for c in code_changes)),
                "cascade_explanation": f"Enabling validation reveals {len(set(c.get('file') for c in code_changes))} files with violations"
            }
            cascades.append(cascade)

    return cascades

def merge_chunks_by_validation(
    chunk_findings: List[Dict[str, Any]],
    validation_sequence: List[Dict[str, Any]],
    structured_chunks: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Step 2: Merge chunk findings by validation (deterministic, no LLM).

    Groups all files and changes by validation_order.
    Detects cross-chunk patterns.
    Handles sub-problems (multiple failure types per validation).

    Args:
        chunk_findings: LLM classification output (which files go to which validations)
        validation_sequence: The validation sequence from CI workflow
        structured_chunks: Original parsed diff chunks with actual changes
    """
    validation_groups = {}

    for chunk in chunk_findings:
        for val_entry in chunk.get("validations_in_this_chunk", []):
            val_order = val_entry.get("validation_order")
            val_cmd = val_entry.get("validation_cmd")
            if val_order in (None, "unknown") or not val_cmd or val_cmd == "unknown":
                continue

            # Check if this validation has multiple failure types
            if val_entry.get("has_multiple_failure_types"):
                # Handle sub-problems
                for sub_problem in val_entry.get("sub_problems", []):
                    # Unique key: validation_order + sub_problem_id
                    sub_id = sub_problem.get("sub_problem_id", "")
                    key = f"{val_order}_{sub_id}" if sub_id else str(val_order)

                    if key not in validation_groups:
                        validation_groups[key] = {
                            "validation_order": val_order,
                            "validation_cmd": val_cmd,
                            "has_sub_problems": True,
                            "sub_problem_id": sub_id,
                            "failure_type": sub_problem.get("failure_type"),
                            "issue_type": sub_problem.get("issue_type") or sub_problem.get("failure_name"),
                            "error_code": sub_problem.get("error_code"),
                            "chunks": [],
                            "all_files": [],
                            "all_changes": [],
                            "has_pattern": False,
                            "total_files": 0,
                        }

                    validation_groups[key]["chunks"].append(chunk["chunk_index"])
                    validation_groups[key]["all_files"].extend(sub_problem.get("files", []))
                    # Don't add changes here - they'll be populated from structured_chunks below
                    validation_groups[key]["total_files"] += sub_problem.get("total_files", len(sub_problem.get("files", [])))

                    # Cross-chunk pattern detection
                    if sub_problem.get("has_pattern"):
                        validation_groups[key]["has_pattern"] = True
            else:
                # Single failure type - keep distinct issue subtypes separate.
                failure_type = str(val_entry.get("failure_type") or "").strip()
                issue_type = str(val_entry.get("issue_type") or val_entry.get("failure_name") or "").strip()
                key_parts = [str(val_order)]
                if failure_type:
                    key_parts.append(failure_type.lower().replace(" ", "_"))
                if issue_type:
                    key_parts.append(issue_type.lower().replace(" ", "_"))
                key = "::".join(key_parts)

                if key not in validation_groups:
                    validation_groups[key] = {
                        "validation_order": val_order,
                        "validation_cmd": val_cmd,
                        "failure_type": failure_type,
                        "issue_type": issue_type,
                        "has_sub_problems": False,
                        "chunks": [],
                        "all_files": [],
                        "all_changes": [],
                        "has_pattern": False,
                        "total_files": 0,
                    }

                validation_groups[key]["chunks"].append(chunk["chunk_index"])
                validation_groups[key]["all_files"].extend(val_entry.get("files", []))
                # Don't add changes here - they'll be populated from structured_chunks below
                validation_groups[key]["total_files"] += val_entry.get("total_files", len(val_entry.get("files", [])))

                # Cross-chunk pattern detection
                if val_entry.get("has_pattern"):
                    validation_groups[key]["has_pattern"] = True

    # CRITICAL FIX: Attach actual file changes to validation groups
    # Build a lookup of file -> changes from the original structured chunks
    file_changes_lookup = {}
    if structured_chunks:
        for chunk in structured_chunks:
            if "files" in chunk:
                for file_info in chunk["files"]:
                    file_path = file_info.get("path", "")
                    if file_path and "changes" in file_info:
                        if file_path not in file_changes_lookup:
                            file_changes_lookup[file_path] = []
                        file_changes_lookup[file_path].extend(file_info["changes"])

    # Now attach changes to each validation group
    for val_group in validation_groups.values():
        for file_path in val_group["all_files"]:
            if file_path in file_changes_lookup:
                # Add all changes for this file
                file_changes = file_changes_lookup[file_path]
                for change in file_changes:
                    # Add file context to each change
                    change_with_file = change.copy()
                    change_with_file["file"] = file_path
                    val_group["all_changes"].append(change_with_file)

    # Sort by validation order for sequential processing. LLMs should return
    # numeric orders, but keep this deterministic if a string slips through.
    def _validation_sort_key(item: tuple[str, Dict[str, Any]]) -> tuple[int, str]:
        order = item[1].get("validation_order")
        try:
            return (int(order), str(order))
        except (TypeError, ValueError):
            return (10**9, str(order))

    sorted_groups = dict(sorted(validation_groups.items(), key=_validation_sort_key))

    return {
        "validation_groups": sorted_groups,
        "total_validations": len(set(g["validation_order"] for g in sorted_groups.values())),
        "total_groups": len(sorted_groups),  # May be > total_validations if sub-problems exist
    }


def _chunk_validation_changes(
    val_group: Dict[str, Any],
    max_changes_per_chunk: int = 400
) -> List[Dict[str, Any]]:
    """
    Chunk a single validation if it has too many changes.

    Returns: List of chunks, where each chunk is a subset of val_group
    """
    all_changes = val_group.get('all_changes', [])

    if len(all_changes) <= max_changes_per_chunk:
        # Small enough, return as single chunk
        return [val_group]

    # Split into chunks
    chunks = []
    for start_idx in range(0, len(all_changes), max_changes_per_chunk):
        chunk_changes = all_changes[start_idx:start_idx + max_changes_per_chunk]

        chunk = {
            'validation_cmd': val_group.get('validation_cmd', ''),
            'failure_type': val_group.get('failure_type', ''),
            'issue_type': val_group.get('issue_type', ''),
            'all_files': val_group.get('all_files', []),  # Keep full file list
            'all_changes': chunk_changes,
            'chunk_info': f"Changes {start_idx + 1}-{start_idx + len(chunk_changes)} of {len(all_changes)} total"
        }
        chunks.append(chunk)

    return chunks

def _final_verify_config_files(
    validation_groups: Dict[str, Any],
    all_atomic_problems: List[Dict[str, Any]]
) -> None:
    """
    Final verification that ALL config files from ground truth are included.

    This runs after all problems are created to catch any config files that
    might have been filtered out during processing.
    """
    config_file_patterns = [
        'pyproject.toml', 'package.json', 'requirements.txt', 'setup.py',
        'Cargo.toml', 'pom.xml', 'build.gradle', 'Gemfile', 'go.mod'
    ]

    # Collect all config files from validation groups
    all_config_files = set()
    groups_data = validation_groups.get('validation_groups', {})

    for val_order, val_group in groups_data.items():
        for change in val_group.get('all_changes', []):
            file_path = change.get('file', '')
            if any(pattern in file_path for pattern in config_file_patterns):
                all_config_files.add(file_path)

    if not all_config_files:
        return  # No config files to verify

    # Collect all files from problems
    files_in_problems = set()
    for problem in all_atomic_problems:
        files_in_problems.update(problem.get('affected_files', []))

    # Check for missing config files
    missing_config_files = all_config_files - files_in_problems

    if missing_config_files:
        print(f"\n CRITICAL ERROR: Config files missing from decomposition!")
        print(f"     Config files in ground truth: {all_config_files}")
        print(f"     Missing from problems: {missing_config_files}")
        print(f"\n     These files MUST be included in enablement_fix problems.")
        print(f"     This is a violation of: NEVER remove config file changes from ground truth")
    else:
        print(f"  OK Config file verification: All {len(all_config_files)} config files included in problems")
        # If this happens frequently, we need to strengthen the prompt


def _normalize_chunk_finding(
    finding: Any,
    *,
    chunk_index: int,
    chunk: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize common LLM schema/key/path drift before validation."""
    if isinstance(finding, list):
        finding = {"chunk_index": chunk_index, "validations_in_this_chunk": finding}
    elif not isinstance(finding, dict):
        print(
            f"    Chunk {chunk_index} WARNING: LLM returned {type(finding).__name__}, "
            "treating as empty"
        )
        return {"chunk_index": chunk_index, "validations_in_this_chunk": []}

    if "validations_in_this_chunk" not in finding and "validations_this_chunk" in finding:
        finding["validations_in_this_chunk"] = finding.pop("validations_this_chunk")

    finding.setdefault("chunk_index", chunk_index)
    validations = finding.get("validations_in_this_chunk")
    if not isinstance(validations, list):
        finding["validations_in_this_chunk"] = []
        return finding

    actual_files = [
        str(file_info.get("path") or "")
        for file_info in chunk.get("files", [])
        if file_info.get("path")
    ]
    actual_set = set(actual_files)

    for validation in validations:
        if not isinstance(validation, dict):
            continue
        files = validation.get("files")
        if not isinstance(files, list):
            validation["files"] = []
            validation["total_files"] = 0
            continue

        normalized_files: List[str] = []
        for file_path in files:
            file_path = str(file_path or "").strip()
            if not file_path:
                continue
            if file_path in actual_set:
                normalized_files.append(file_path)
                continue

            match = difflib.get_close_matches(file_path, actual_files, n=1, cutoff=0.86)
            if match:
                normalized_files.append(match[0])
            else:
                print(
                    f"    Chunk {chunk_index} WARNING: dropping hallucinated file path: {file_path}"
                )

        validation["files"] = list(dict.fromkeys(normalized_files))
        validation["total_files"] = len(validation["files"])

    return finding


def analyze_validation_groups_with_reasoning(
    validation_groups: Dict[str, Any],
    validation_sequence: List[Dict[str, Any]],
    ci_context: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """
    Simple 3-step process:
    1. For each validation group
    2. If too large, chunk it
    3. Analyze each chunk with LLM -> create atomic problems
    """
    groups_data = validation_groups.get("validation_groups", {})
    print(f"  Processing {len(groups_data)} validation groups...")

    all_problems = []
    next_id = 1

    # Process each validation group
    def _group_order(item: tuple[str, Dict[str, Any]]) -> tuple[int, str]:
        order = item[1].get("validation_order")
        try:
            return (int(order), str(item[0]))
        except (TypeError, ValueError):
            return (10**9, str(item[0]))

    for val_order, val_group in sorted(groups_data.items(), key=_group_order):
        print(f"    Validation {val_order}: {val_group.get('validation_cmd', '')}")

        # Chunk if too large (40 changes per chunk with 16k max_tokens)
        # MiniMax M2.5 can handle large outputs, so we use bigger chunks for efficiency
        chunks = _chunk_validation_changes(val_group, max_changes_per_chunk=40)
        if len(chunks) > 1:
            print(f"      Splitting into {len(chunks)} chunks")

        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks, 1):
            # Add small delay before first chunk to avoid hitting API too fast
            if chunk_idx == 1:
                time.sleep(1)  # 1s before starting each validation

            # Extract validation info for this validation order
            validation_order = val_group.get("validation_order", val_order)
            val_info = next(
                (v for v in validation_sequence if str(v.get('order')) == str(validation_order)),
                {},
            )

            # Format changes - extract essential fields without losing important context
            all_changes = chunk.get('all_changes', [])
            compact_changes = []
            for c in all_changes:
                # Keep full context but remove redundant fields to save tokens
                compact_changes.append({
                    'file': c.get('file', ''),
                    'line': c.get('line_number') or c.get('line'),
                    'before': c.get('before', ''),  # FIX: Correct field name
                    'after': c.get('after', ''),    # FIX: Correct field name
                })

            changes_data = {
                'validation_cmd': chunk.get('validation_cmd', ''),
                'failure_type': chunk.get('failure_type', ''),
                'issue_type': chunk.get('issue_type', ''),
                'files_count': len(set(c.get('file', '') for c in all_changes)),
                'changes_count': len(all_changes),
                'changes': compact_changes
            }

            # Build detailed analysis prompt with reasoning guidance
            prompt = f"""Analyze validation {validation_order} and create atomic problems.

CI Failure Context:
{json.dumps(ci_context, indent=2)}

Validation Context:
- Order: {validation_order}
- Command: {val_info.get('validation_cmd', chunk.get('validation_cmd', ''))}
- Validates: {val_info.get('validates', 'Code quality/formatting')}
- Broad failure type: {chunk.get('failure_type', '')}
- Specific issue type: {chunk.get('issue_type', '')}

Changes in this chunk:
{json.dumps(changes_data, indent=2)}

ANALYSIS INSTRUCTIONS:

For EACH change, examine the 'before' and 'after' fields to understand:
1. What was the previous state (before)?
2. What is the new state (after)?
3. Why was the previous state broken/incompatible?
4. Why does the new state fix it?

## For DEPENDENCY/PACKAGE changes (requirements.txt, pyproject.toml, package.json, etc.):
- If ADDING a package:
  how_fixed: "Add [package]==[version] to [file] (was missing, needed by [reason])"
  root_cause: "Package [name] was not installed. [Tool/dependency] requires it for [functionality]"

- If UPDATING/CHANGING a package:
  how_fixed: "Update [package] from [old_version] to [new_version] in [file]"
  root_cause: "Version [old_version] is incompatible with [tool/dependency]. [Specific API/feature] changed in [new_version]"

- If REMOVING a package:
  how_fixed: "Remove [package]==[version] from [file]"
  root_cause: "Package [name] is no longer needed/conflicts with [reason]"

## GENERAL PATTERN (works for all change types):

how_fixed format:
"Change [what] from [before_state] to [after_state] in [file]. Reason: [why this specific change fixes the validation]"

root_cause format:
"Detailed technical explanation of why [before_state] was broken/incompatible and what caused the validation failure"

why_fix_works format:
"Explain the mechanism of how [after_state] resolves the issue and satisfies the validation requirement"

EXAMPLES by category:

Code changes (type annotations, imports, logic):
  - "Change type annotation from DTypeLike to np.dtype[Any] in ndarrays.py. Reason: DTypeLike is a private numpy type"
  - "Change import from 'beautysh' to 'mdformat_beautysh' in formatter.py. Reason: Module was moved in v2.0"
  - "Add null check before accessing property in handler.py. Reason: Validation requires defensive programming"

Dependency changes:
  - "Add beautysh==1.1.2 to requirements.txt. Reason: mdformat requires Beautify class from beautysh>=1.1.0"
  - "Update numpy from 1.20.0 to 1.24.0 in pyproject.toml. Reason: Version 1.20.0 lacks dtype[Any] support"
  - "Remove deprecated package 'oldlib' from requirements.txt. Reason: Replaced by 'newlib' with breaking changes"

Formatting/Config changes:
  - "Fix line length in 5 files by wrapping lines exceeding 100 chars. Reason: flake8 E501 rule violation"
  - "Update line-length from 88 to 100 in pyproject.toml. Reason: Match project's existing code style"

CRITICAL RULES:
OK BE SPECIFIC: Use exact versions, package names, and before->after states from the diff
OK how_fixed: State WHAT changed FROM what TO what, and WHY (include reasoning in the same field)
OK root_cause: Detailed technical explanation of why the old state failed validation
OK why_fix_works: Explain the mechanism of how the new state satisfies the validation
OK Group similar changes -> ONE problem (e.g., same fix pattern across multiple files)
OK NO VAGUE statements: Never say "install or update" - be precise: "Add X version Y" or "Update X from v1 to v2"
OK NO LINE NUMBERS: Avoid mentioning specific line numbers (they change across commits)

OUTPUT FORMAT:
{{
  "atomic_problems": [
    {{"problem_id": {next_id},
    "problem_type": "primary_failure|enablement_fix",
    "validation_order": {validation_order},
    "validation_cmd": "...",
    "failure_type": "...",
    "issue_type": "...",
    "problem": "...",
    "root_cause": "...",
    "how_fixed": "...",
    "why_fix_works": "...",
    "affected_files": [...]
    }},
    // Additional problems if multiple detected
  ]
}}

{STRICT_JSON_RULES}
"""

            # Debug: Log chunk info before LLM call
            print(f"      [DEBUG] Calling LLM for chunk {chunk_idx}/{len(chunks)}")
            print(f"              Prompt size: {len(prompt)} chars, {len(all_changes)} changes")

            # Get result from LLM
            result = _invoke_json(llm, prompt)

            # Handle SPLIT_REQUIRED signal (length limit hit)
            if result == "SPLIT_REQUIRED" and len(all_changes) > 1:
                print(f"      -> Auto-splitting chunk into 2 sub-chunks due to length limit...")

                # Split changes in half
                mid = len(all_changes) // 2
                sub_chunks = [
                    {**chunk, 'all_changes': all_changes[:mid]},
                    {**chunk, 'all_changes': all_changes[mid:]}
                ]

                # Retry with smaller chunks
                for sub_idx, sub_chunk in enumerate(sub_chunks, 1):
                    print(f"        Sub-chunk {sub_idx}/2: {len(sub_chunk['all_changes'])} changes")

                    sub_compact_changes = []
                    for c in sub_chunk['all_changes']:
                        sub_compact_changes.append({
                            'file': c.get('file', ''),
                            'line': c.get('line_number') or c.get('line'),
                            'before': c.get('before', ''),
                            'after': c.get('after', ''),
                        })

                    sub_changes_data = {
                        'validation_cmd': sub_chunk.get('validation_cmd', ''),
                        'failure_type': sub_chunk.get('failure_type', ''),
                        'issue_type': sub_chunk.get('issue_type', ''),
                        'files_count': len(set(c.get('file', '') for c in sub_chunk['all_changes'])),
                        'changes_count': len(sub_chunk['all_changes']),
                        'changes': sub_compact_changes
                    }

                    # Rebuild prompt with smaller chunk
                    sub_prompt = prompt.replace(
                        json.dumps(changes_data, indent=2),
                        json.dumps(sub_changes_data, indent=2)
                    )

                    sub_result = _invoke_json(llm, sub_prompt)
                    time.sleep(5)

                    if sub_result and isinstance(sub_result, dict) and sub_result.get("atomic_problems"):
                        for p in sub_result["atomic_problems"]:
                            p["problem_id"] = next_id
                            next_id += 1
                        chunk_problems.extend(sub_result["atomic_problems"])

                # Skip to next chunk after handling split
                print(f"      -> Split result: {len(chunk_problems)} problems from sub-chunks")
                continue

            # Debug: Check result
            if not result or (isinstance(result, dict) and not result.get("atomic_problems")):
                print(f"      [DEBUG] Empty or invalid result returned")
                print(f"              Result type: {type(result)}, value: {result}")

                # Save failing prompt for debugging
                debug_file = PROJECT_ROOT / "data" / "trs" / f"debug_prompt_val{val_order}_chunk{chunk_idx}.txt"
                debug_file.parent.mkdir(parents=True, exist_ok=True)
                debug_file.write_text(prompt, encoding='utf-8')
                print(f"              Saved prompt to: {debug_file}")

                # Show sample of changes that failed
                sample_files = list(set(c.get('file', '') for c in all_changes[:5]))
                print(f"              Failed files (sample): {sample_files}")

            # Add 5-second delay after each API call to avoid rate limiting
            # OpenRouter/MiniMax may have stricter rate limits than direct API
            time.sleep(5)

            # Normalize response format
            if isinstance(result, list):
                # LLM returned array: [{"problem_id": ...}, ...]
                problems = result
            elif isinstance(result, dict):
                # Check if it's a single problem object or a wrapper
                if "problem_id" in result and "atomic_problems" not in result:
                    # LLM returned single problem: {"problem_id": 1, "problem": ...}
                    problems = [result]
                else:
                    # LLM returned wrapper: {"atomic_problems": [...]}
                    problems = result.get("atomic_problems", [])
            else:
                # Invalid format
                problems = []

            # Check for empty result when chunk has changes
            if len(chunk.get('all_changes', [])) > 0 and len(problems) == 0:
                print(f"  WARNING: Chunk {chunk_idx} has changes but returned 0 problems!")
                LOGGER.warning(f"Validation {val_order} chunk {chunk_idx}: {len(chunk.get('all_changes', []))} changes but 0 problems")

            # Renumber IDs and verify required fields
            for p in problems:
                p["problem_id"] = next_id
                # Ensure validation_order is set (use the integer, not the composite key)
                if "validation_order" not in p:
                    p["validation_order"] = validation_order

                # No change_summary - how_fixed already describes changes
                # No detailed_changes - original diff has all details
                # Keep only essential problem description

                next_id += 1

            all_problems.extend(problems)
            print(f"      Chunk {chunk_idx}/{len(chunks)}: {len(problems)} problems")

    # Verify config files included
    _final_verify_config_files(validation_groups, all_problems)

    print(f"  OK Total: {len(all_problems)} problems created")
    return {"atomic_problems": all_problems}


def analyze_diff_chunks(
    issue: Dict,
    benchmark_context: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """
    Three-step diff analysis with deterministic pre-processing:
    0. Parse diff into structured format (deterministic, no LLM)
    1. Chunk and classify by validation (per chunk, LLM only for classification)
    2. Merge by validation (deterministic)
    3. Deep reasoning with full context (LLM)
    """
    diff = str(issue.get("diff") or "")
    if not diff.strip():
        raise ValueError(f"Issue {issue.get('id')} has no ground-truth diff")

    # Step 0: Deterministic diff parsing (NEW!)
    print(f"  Step 0: Parsing diff into structured format...")


    structured_diff = parse_diff_to_structured(diff)
    total_files = structured_diff["total_files"]
    total_changes = structured_diff["total_changes"]
    print(f"    Parsed {total_files} files with {total_changes} changes")

    # Chunk by file count (not char count) - cleaner and more predictable
    # Reduced from 15 to 8 to avoid API errors with large responses
    chunks = chunk_structured_diff(structured_diff, max_files_per_chunk=20)
    if not chunks:
        raise ValueError(f"Issue {issue.get('id')} ground-truth diff could not be chunked")

    visible_failure_context = _compact_context_for_diff_analysis(issue, benchmark_context)
    validation_sequence = benchmark_context.get("validation_sequence") or []
    chunk_findings: List[Dict[str, Any]] = []

    print(f"  Step 1: Classifying patch changes by repaired validation ({total_files} files in {len(chunks)} chunk(s))...")

    for index, chunk in enumerate(chunks, start=1):
        prompt = f"""Classify each changed file by the validation command that would catch the fixed issue.

## INPUT

CI failure context:
{json.dumps(visible_failure_context, indent=2)}

Available validations. Choose validation_order and validation_cmd from this list:
{json.dumps(validation_sequence, indent=2)}

Changed files, chunk {index}/{len(chunks)}:
{format_structured_for_llm(chunk)}

## TASK

For every file in this chunk:
1. Inspect the before/after changes.
2. Decide what kind of CI failure the change fixes.
3. Choose the validation command from VALIDATIONS that would detect that failure.
4. Group files that have the same validation command, broad failure category, and specific issue name.

Use two levels of failure naming:
- failure_type: broad category, from the allowed list below.
- issue_type: specific failure name/subtype inferred from the error or code change.

Examples:
- failure_type: "Type Checking", issue_type: "missing return annotation"
- failure_type: "Type Checking", issue_type: "incompatible argument type"
- failure_type: "Linting", issue_type: "unused import"
- failure_type: "Linting", issue_type: "undefined name"
- failure_type: "Formatting", issue_type: "line length"
- failure_type: "Dependencies", issue_type: "missing package"

Config files must be assigned to the validation they affect:
- Tool/dependency added or configured -> that tool's validation.
- Formatter config changed -> formatter validation.
- Type-checker config changed -> type-checking validation.
- Linter config changed -> linting validation.

---

## OUTPUT

Return ONLY a JSON array:

[
  {{
  "validation_order": <INT>, 
  "validation_cmd": "<exact>", 
  "failure_type": "<category>", 
  "issue_type": "<specific>", 
  "files": [...], 
  "total_files": <int>
  }},
  // Repeat for each validation group
]

## REQUIREMENTS

OK Return array only (no wrapper object)
OK validation_order = INTEGER from VALIDATIONS (never null)
OK validation_cmd = exact match from VALIDATIONS
OK Every file in at least one group
OK failure_type from allowed list only
OK issue_type = specific, short

{STRICT_JSON_RULES}
"""
        try:
            result = _invoke_json(llm, prompt)

            # Expect array directly (as specified in prompt)
            if isinstance(result, list):
                validations = result
            elif isinstance(result, dict):
                # LLM wrapped array in object - try to extract
                # Common patterns: {"validations": [...]} or {"groups": [...]} or {"result": [...]}
                for key in ["validations", "groups", "result", "data", "validation_groups"]:
                    if key in result and isinstance(result[key], list):
                        validations = result[key]
                        print(f"    INFO  Extracted array from dict.{key}")
                        break
                else:
                    # No valid array found
                    print(f"    WARNING: Expected array, got dict with keys: {list(result.keys())}")
                    validations = []
            else:
                # Invalid format
                print(f"    WARNING: Expected array, got {type(result).__name__}")
                validations = []

            # Fix missing validation_order
            for v in validations:
                if not v.get("validation_order"):
                    val_cmd = str(v.get("validation_cmd", "")).strip()
                    for seq_item in validation_sequence:
                        if seq_item.get("validation_cmd") == val_cmd:
                            v["validation_order"] = seq_item.get("order")
                            break

            # Keep only valid entries
            valid = [v for v in validations if v.get("validation_order")]

            print(f"    Chunk {index}/{len(chunks)}: {len(valid)} validation(s), orders={[v['validation_order'] for v in valid]}")

            chunk_findings.append({"chunk_index": index, "validations_in_this_chunk": valid})

        except Exception as exc:
            print(f"    Chunk {index}/{len(chunks)} FAILED: {exc}")
            import traceback
            traceback.print_exc()
            chunk_findings.append({"chunk_index": index, "validations_in_this_chunk": []})

    # Step 2: Merge by validation (deterministic)
    print(f"  Step 2: Merging chunks by validation...")
    validation_groups = merge_chunks_by_validation(chunk_findings, validation_sequence, chunks)
    print(f"    Found {validation_groups['total_groups']} groups from {validation_groups['total_validations']} validations")

    # Step 2.5: CI-Diff Correlation (deterministic)
    print(f"  Step 2.5: Analyzing CI-Diff correlation (layered structure)...")
    ci_context = _compact_context_for_diff_analysis(issue, benchmark_context)

    # Step 3: Deep reasoning with full context + correlation
    print(f"  Step 3: Deep reasoning with correlation context...")
    reasoning_result = analyze_validation_groups_with_reasoning(
        validation_groups,
        validation_sequence,
        ci_context,
        llm
    )
    # Log results
    atomic_problems = reasoning_result.get("atomic_problems", [])
    if atomic_problems:
        print(f"  OK Identified {len(atomic_problems)} atomic problems")
    else:
        print(f"  WARNING: No atomic problems identified")
    return {
        "mode": "structured_diff_3step",
        "total_files": total_files,
        "total_changes": total_changes,
        "chunk_count": len(chunks),
        "chunk_findings": chunk_findings,
        "validation_groups": validation_groups,
        "atomic_problems": atomic_problems,
        "sequential_workflow_metadata": reasoning_result.get("sequential_workflow_metadata", {}),
    }


def decompose_issue(issue: Dict, llm) -> Dict:
    """
    Three-step reverse engineering from CI failure + ground truth diff:

    1. Classify chunks by validation (per chunk)
    2. Merge by validation (deterministic)
    3. Deep reasoning with full context (ConRAD/STAIR style)

    Returns specific, actionable atomic problems for mini-swe-agent.
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

        # Three-step analysis
        diff_context = analyze_diff_chunks(issue, benchmark_context, llm)

        atomic_problems = diff_context.get("atomic_problems", [])
        sequential_metadata = diff_context.get("sequential_workflow_metadata", {})

        if not atomic_problems:
            print("  WARNING: No atomic problems identified")
            return {}

        print(f"  OK Identified {len(atomic_problems)} atomic problems")

        # Build final result
        context = benchmark_context.get("context", {})
        log_analysis = benchmark_context.get("log_analysis", {})

        result = {
            "original_issue_id": issue_id,
            "sha_fail": issue.get("sha_fail"),
            "repo": issue.get("repo_name", issue.get("repo")),
            "original_error_type": issue.get("error_type"),

            # Atomic problems from three-step analysis
            "problems": atomic_problems,
            "total_problems": len(atomic_problems),
            "total_changed_files": len(issue.get("changed_files", [])),

            # Benchmark CI context (cleaned - no redundancy)
            "benchmark_ci_context": {
                "workflow_path": benchmark_context.get("workflow_path"),
                "workflow_name": benchmark_context.get("workflow_name"),
                "validation_sequence": benchmark_context.get("validation_sequence", []),

                # Summary level (for quick access)
                "overall_failure_reasons": context.get("overall_failure_reasons", []),
                "overall_error_types": context.get("overall_error_types", []),

                # Detailed analysis (structured)
                "error_types": log_analysis.get("error_types", []),  # Detailed with subcategory + evidence
                "relevant_files": log_analysis.get("relevant_files", []),  # Files with line numbers
                "failed_jobs": log_analysis.get("failed_job", []),  # Job/step/command info
            },

            # Structured diff analysis metadata
            "diff_analysis_context": {
                "mode": diff_context.get("mode"),
                "total_files": diff_context.get("total_files", 0),
                "total_changes": diff_context.get("total_changes", 0),
                "chunk_count": diff_context.get("chunk_count", 0),
                "validation_groups_count": diff_context.get("validation_groups", {}).get("total_groups", 0),
                "validation_groups": diff_context.get("validation_groups", {}).get("validation_groups", {}),
            },
        }

        return result

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"  ERROR Failed to decompose: {e}")
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


def generate_l1_l2_l3_pipeline(decomposed_result: Dict, llm) -> Dict:
    """
    Full L1/L2/L3 generation pipeline.

    Stage 1: Deduplicate similar problems (mechanical)
    Stage 2: Detect dependencies (LLM)
    Stage 3: Generate L1 file-level (mechanical - reuse data)
    Stage 4: Generate L2 repair sequence (LLM)
    Stage 5: Generate L3 analysis (LLM)

    Args:
        decomposed_result: Output from decompose_issue()
        llm: LLM instance for prompts

    Returns:
        Dictionary with l1, l2, l3 sections
    """

    if "error" in decomposed_result or not decomposed_result.get("problems"):
        return decomposed_result

    issue_id = decomposed_result.get("original_issue_id", "?")
    print(f"\n{'='*80}")
    print(f"L1/L2/L3 Pipeline for Issue {issue_id}")
    print(f"{'='*80}")

    # Stage 1: Deduplicate (mechanical)
    print("\n[Stage 1/5] Deduplicating similar problems...")
    deduplicated = _stage1_deduplicate_problems(decomposed_result)
    print(f"  Input: {len(decomposed_result.get('problems', []))} problems")
    print(f"  Output: {len(deduplicated)} deduplicated problems")

    # Stage 2: Detect dependencies (LLM)
    print("\n[Stage 2/5] Detecting dependencies with LLM...")
    dependencies = _stage2_detect_dependencies_llm(deduplicated, llm)
    print(f"  Dependencies found: {len(dependencies.get('dependency_edges', []))}")
    print(f"  Repair order: {dependencies.get('repair_order', [])}")

    # Stage 3: Generate L1 (with LLM for detailed descriptions)
    print("\n[Stage 3/5] Generating L1 (file-level) with detailed descriptions...")
    repo = decomposed_result.get("repo", "unknown")
    workflow_path = decomposed_result.get("benchmark_ci_context", {}).get("workflow_path", "unknown")
    l1 = _stage3_generate_l1_with_llm(deduplicated, dependencies, decomposed_result, llm, repo, workflow_path)
    print(f"  L1 file-level problems: {len(l1)}")

    # Stage 4: Generate L2 (LLM)
    print("\n[Stage 4/5] Generating L2 (repair sequence) with LLM...")
    l2 = _stage4_generate_l2_llm(deduplicated, dependencies, llm, issue_id, decomposed_result.get("repo"))
    print(f"  L2 repair steps: {len(l2.get('problems', []))}")

    # Stage 5: Generate L3 (LLM)
    print("\n[Stage 5/5] Generating L3 (analysis) with LLM...")
    l3 = _stage5_generate_l3_llm(l1, l2, deduplicated, llm)
    print(f"  L3 insights generated")

    # Build final result
    result = {
        "issue_id": str(issue_id),
        "repo": decomposed_result.get("repo", "unknown"),
        "workflow_path": workflow_path,
        "l1_file_level": l1,
        "l2_repair_sequence": l2,
        "l3_analysis": l3,
        "metadata": {
            "original_problems_count": len(decomposed_result.get('problems', [])),
            "deduplicated_count": len(deduplicated),
            "l1_file_count": len(l1),
            "l2_step_count": len(l2.get('problems', [])),
            "total_time_minutes": l2.get('total_time_minutes', 0)
        }
    }

    print(f"\n{'='*80}")
    print(f"Pipeline Complete!")
    print(f"{'='*80}")

    return result


def _stage1_deduplicate_problems(decomposed_result: Dict) -> List[Dict]:
    """
    Stage 1: Deduplicate similar problems (mechanical).

    DON'T over-deduplicate! Keep sub_problems for file-level details.
    Group by validation command, but preserve all sub-problems.
    """
    problems = decomposed_result.get("problems", [])

    # Group by validation command only
    problem_groups = {}
    for prob in problems:
        validation = prob.get("validation_cmd", "unknown")
        if validation not in problem_groups:
            problem_groups[validation] = []
        problem_groups[validation].append(prob)

    # For each validation group, keep ALL sub-problems
    deduplicated = []
    for validation, group_problems in problem_groups.items():
        # Collect all unique files
        all_files = []
        for p in group_problems:
            all_files.extend(p.get("affected_files", []))
        all_files = list(dict.fromkeys(all_files))  # Deduplicate

        # Use first as template but keep ALL sub_problems for L1
        template = group_problems[0]
        merged = {
            "validation_cmd": validation,
            "validation_order": template.get("validation_order"),
            "problem_type": template.get("problem_type", "unknown"),
            "what_broke": template.get("what_broke", "Validation failed"),
            "root_cause": template.get("root_cause", "Unknown"),
            "how_fixed": template.get("how_fixed", ""),
            "why_fixed_works": template.get("why_fixed_works", ""),
            "affected_files": all_files,
            "issue_types": list(set(p.get("issue_type", "") for p in group_problems if p.get("issue_type"))),
            "sub_problems": group_problems  # CRITICAL: Keep ALL for L1 generation!
        }
        deduplicated.append(merged)

    return deduplicated


def _stage2_detect_dependencies_llm(problems: List[Dict], llm) -> Dict:
    """
    Stage 2: Detect dependencies between problems using LLM.

    Ask LLM:
    - Which problems must be fixed first?
    - Which problems enable others?
    - Which problems reveal others?
    - What is optimal repair order?
    """

    # Prepare summary for LLM
    problems_summary = []
    for idx, prob in enumerate(problems, 1):
        problems_summary.append({
            "id": idx,
            "validation": prob.get("validation_cmd", "unknown"),
            "type": prob.get("problem_type", "unknown"),
            "what_broke": prob.get("what_broke", "")[:200],
            "files_count": len(prob.get("affected_files", []))
        })

    prompt = f"""Analyze dependencies between these CI failure problems:

{json.dumps(problems_summary, indent=2)}

Determine:
1. Which problems are PRIMARY (must fix first)?
2. Which problems ENABLE others (e.g., installing tool enables validation)?
3. Which problems REVEAL others (consecutive failures)?
4. What is the OPTIMAL REPAIR ORDER?

Output JSON:
{{
  "dependency_edges": [
    {{"from": 1, "to": 3, "type": "enables", "reason": "Installing taplo enables TOML validation"}},
    {{"from": 2, "to": 4, "type": "reveals", "reason": "Fixing mdformat reveals formatting issues"}}
  ],
  "repair_order": [1, 2, 3, 4],
  "primary_problems": [1, 2],
  "enablement_problems": [3],
  "consecutive_problems": [4],
  "reasoning": "Fix primary failures first, then enable tools, then handle revealed issues"
}}

{STRICT_JSON_RULES}"""

    try:
        time.sleep(3)  # Rate limiting
        response = _invoke_json(llm, prompt)

        if isinstance(response, dict):
            return response
        else:
            print(f"  WARNING: LLM returned non-dict, using fallback")
            return _fallback_dependencies(problems)
    except Exception as e:
        print(f"  ERROR in dependency detection: {e}")
        return _fallback_dependencies(problems)


def _fallback_dependencies(problems: List[Dict]) -> Dict:
    """Fallback: simple dependency detection without LLM."""
    # Simple heuristic: primary first, then enablement, then consecutive
    primary = []
    enablement = []
    consecutive = []

    for idx, prob in enumerate(problems, 1):
        ptype = prob.get("problem_type", "")
        if "primary" in ptype:
            primary.append(idx)
        elif "enablement" in ptype:
            enablement.append(idx)
        else:
            consecutive.append(idx)

    repair_order = primary + enablement + consecutive

    return {
        "dependency_edges": [],
        "repair_order": repair_order,
        "primary_problems": primary,
        "enablement_problems": enablement,
        "consecutive_problems": consecutive,
        "reasoning": "Fallback: primary -> enablement -> consecutive"
    }


def _stage3_generate_l1_with_llm(deduplicated: List[Dict], dependencies: Dict, decomposed_result: Dict, llm, repo: str, workflow_path: str) -> List[Dict]:
    """
    Stage 3: Generate L1 (file-level) with LLM for detailed problem descriptions.

    L1 Format:
    [
      {
        "repo": "flower",
        "workflow": "python -m mypy py",
        "failure_type": "type_checking",
        "file": "path/to/file.py",
        "problem": "Detailed description of error, failure pattern, why failed, root cause",
        "fixes": "Before: ... -> After: ...",
        "why_fix": "How the fix works to solve the error",
        "dependent_files": [
          {
            "file": "other/file.py",
            "change": "What fixes were done, why, and how it's relevant"
          }
        ]
      }
    ]
    """

    print(f"  Generating detailed L1 descriptions with LLM...")

    l1_problems = []

    # Process each sub-problem individually to preserve ALL details
    for prob in deduplicated:
        sub_problems = prob.get("sub_problems", [])
        validation_cmd = prob.get("validation_cmd", "")

        # Process EACH sub-problem to avoid losing details
        for sub_prob in sub_problems:
            files = sub_prob.get("affected_files", [])

            # Get failure type and issue type from sub-problem
            failure_type = _infer_failure_type(sub_prob)
            issue_type = sub_prob.get("issue_type", "unknown error")

            # Create L1 entry for EACH file
            for file_path in files:
                # Build dependent files with detailed info
                dependent_files = []
                other_files = [f for f in files if f != file_path]

                for dep_file in other_files:
                    # Find the sub-problem that contains this dependent file to get its specific changes
                    dep_changes = _find_file_changes(dep_file, sub_problems)

                    dependent_files.append({
                        "file": dep_file,
                        "change": dep_changes if dep_changes else f"Part of same validation fix: {sub_prob.get('how_fixed', '')[:200]}",
                        "link_to": f"Both files needed for same validation ({validation_cmd}). Changes are interdependent to fix the validation failure."
                    })

                # Clean up why_fix (avoid "?" or "Unknown")
                why_fix = sub_prob.get("why_fixed_works", "")
                if why_fix in ["?", "Unknown", ""]:
                    why_fix = f"This fix resolves the validation failure by addressing: {sub_prob.get('root_cause', '')[:150]}"

                # Create L1 entry with full details from sub_problem
                l1_entry = {
                    "repo": repo,
                    "workflow": workflow_path,
                    "validation_cmd": validation_cmd,
                    "failure_type": failure_type,
                    "issue_type": issue_type,
                    "file": file_path,
                    "problem": f"{sub_prob.get('what_broke', 'Validation failed')}. Root cause: {sub_prob.get('root_cause', 'Unknown')}",
                    "fixes": sub_prob.get("how_fixed", "Unknown"),
                    "why_fix": why_fix,
                    "dependent_files": dependent_files
                }

                l1_problems.append(l1_entry)

    return l1_problems


def _find_file_changes(target_file: str, sub_problems: List[Dict]) -> str:
    """Find the specific changes for a target file from sub_problems."""
    for sub_prob in sub_problems:
        if target_file in sub_prob.get("affected_files", []):
            return sub_prob.get("how_fixed", "")[:300]
    return ""

    return l1_problems


def _infer_failure_type(prob: Dict) -> str:
    """Infer failure type from problem data (NO underscores)."""
    validation = prob.get("validation_cmd", "").lower()
    problem_type = prob.get("problem_type", "").lower()

    if "mypy" in validation or "type" in validation:
        return "type checking"
    elif "format" in validation or "lint" in validation:
        return "formatting"
    elif "test" in validation or "pytest" in validation:
        return "test failure"
    elif "config" in problem_type or "enable" in problem_type:
        return "config enablement"
    elif "import" in prob.get("what_broke", "").lower():
        return "import error"
    else:
        return "validation failure"


def _create_bulk_l1_entry(prob: Dict, repo: str, workflow_path: str, validation_cmd: str,
                           failure_type: str, issue_type: str, files: List[str]) -> Dict:
    """Create L1 entry for bulk operations."""
    return {
        "repo": repo,
        "workflow": workflow_path,
        "validation_cmd": validation_cmd,
        "failure_type": failure_type,
        "issue_type": issue_type,
        "file_pattern": "Multiple files with same pattern",
        "files": files,
        "file_count": len(files),
        "problem": f"{prob.get('what_broke', 'Validation failed')}. Root cause: {prob.get('root_cause', 'Unknown')[:200]}. Affects {len(files)} files with same pattern.",
        "fixes": prob.get("how_fixed", "Unknown")[:300],
        "why_fix": prob.get("why_fixed_works", "Unknown")[:200],
        "dependent_files": [],
        "is_bulk": True
    }


def _generate_file_l1_with_llm(file_path: str, repo: str, validation_cmd: str, failure_type: str,
                                 prob: Dict, file_details: Dict, all_files: List[str], llm) -> Dict:
    """Generate detailed L1 entry for a single file using LLM."""

    # Prepare context for LLM
    context = {
        "file": file_path,
        "validation": validation_cmd,
        "what_broke": prob.get("what_broke", ""),
        "root_cause": prob.get("root_cause", ""),
        "how_fixed": file_details.get("how_fixed", prob.get("how_fixed", "")),
        "why_fixed_works": file_details.get("why_fixed_works", prob.get("why_fixed_works", "")),
        "other_files": [f for f in all_files if f != file_path]
    }

    prompt = f"""Generate detailed L1 file-level problem description.

File: {file_path}
Validation: {validation_cmd}
Failure type: {failure_type}

Context:
- What broke: {context['what_broke']}
- Root cause: {context['root_cause']}
- How fixed: {context['how_fixed']}
- Why fix works: {context['why_fixed_works']}

Generate:
1. **Problem**: Detailed description including:
   - What error occurred
   - Failure pattern
   - Why it failed (root cause)
   - Technical details

2. **Fixes**: Before and after format:
   - Before: [original code/config]
   - After: [fixed code/config]
   - Be specific with line numbers if available

3. **Why fix works**: Explain how the fix solves the error

4. **Dependent files**: For each related file in {context['other_files'][:3]}, describe:
   - What changes were made
   - Why the change is needed
   - How it relates to the main file

Output JSON:
{{
  "problem": "Detailed problem description...",
  "fixes": "Before: ... -> After: ...",
  "why_fix": "How the fix works...",
  "dependent_files": [
    {{
      "file": "path",
      "change": "Detailed change description"
    }}
  ]
}}

{STRICT_JSON_RULES}"""

    try:
        time.sleep(3)  # Rate limiting
        response = _invoke_json(llm, prompt)

        if isinstance(response, dict):
            return {
                "repo": repo,
                "workflow": validation_cmd,
                "failure_type": failure_type,
                "file": file_path,
                "problem": response.get("problem", context['what_broke']),
                "fixes": response.get("fixes", context['how_fixed']),
                "why_fix": response.get("why_fix", context['why_fixed_works']),
                "dependent_files": response.get("dependent_files", [])
            }
    except Exception as e:
        print(f"    WARNING: LLM failed for {file_path}, using fallback")

    # Fallback: mechanical generation
    return _create_fallback_l1_entry(file_path, repo, validation_cmd, failure_type, context)


def _create_fallback_l1_entry(file_path: str, repo: str, workflow_path: str, validation_cmd: str,
                                failure_type: str, issue_type: str, context: Dict) -> Dict:
    """Create L1 entry without LLM (fallback)."""

    # Build dependent files list
    dependent_files = []
    for other_file in context['other_files'][:5]:
        dependent_files.append({
            "file": other_file,
            "change": f"Related fix in {other_file}: Part of the same validation ({validation_cmd}). {context['how_fixed'][:100]}"
        })

    return {
        "repo": repo,
        "workflow": workflow_path,
        "validation_cmd": validation_cmd,
        "failure_type": failure_type,
        "issue_type": issue_type,
        "file": file_path,
        "problem": f"{context['what_broke']}. Root cause: {context['root_cause']}",
        "fixes": context['how_fixed'],
        "why_fix": context['why_fixed_works'],
        "dependent_files": dependent_files
    }


def _stage3_generate_l1_mechanical(deduplicated: List[Dict], dependencies: Dict, decomposed_result: Dict) -> List[Dict]:
    """
    Stage 3: Generate L1 (file-level) mechanically from decomposed data.

    Reuse:
    - File info from decomposed data
    - Failure info from decomposed data
    - Fix info from decomposed data
    - Dependencies from Stage 2
    - Changes from diff

    No LLM needed!
    """

    l1_file_problems = []

    for prob in deduplicated:
        # Get all files affected by this problem
        files = prob.get("affected_files", [])

        # Group files by pattern if many files
        if len(files) > 10:
            # Bulk operation - create one L1 entry for the pattern
            l1_entry = {
                "file_pattern": "Multiple files with same pattern",
                "files": files,
                "problem": prob.get("what_broke", "Unknown"),
                "why_error": prob.get("root_cause", "Unknown"),
                "fixes": prob.get("how_fixed", "Unknown"),
                "why_fix_works": prob.get("why_fixed_works", "Unknown"),
                "dependent_files": [],
                "validation": prob.get("validation_cmd", ""),
                "is_bulk": True,
                "file_count": len(files)
            }
            l1_file_problems.append(l1_entry)
        else:
            # Individual files
            for file_path in files:
                # Find this file in sub_problems to get specific details
                file_details = _extract_file_details(file_path, prob.get("sub_problems", []))

                # Build detailed dependent files info
                dependent_files_list = []
                other_files = [f for f in files if f != file_path]

                if other_files:
                    # Get specific change details from sub_problems
                    for dep_file in other_files:
                        dep_details = _extract_file_details(dep_file, prob.get("sub_problems", []))

                        # Build detailed change description
                        change_desc = _build_dependency_description(
                            file_path,
                            dep_file,
                            file_details,
                            dep_details,
                            prob
                        )

                        dependent_files_list.append({
                            "file": dep_file,
                            "change": change_desc,
                            "relationship": "same_validation"
                        })

                l1_entry = {
                    "file": file_path,
                    "problem": prob.get("what_broke", "Unknown"),
                    "why_error": prob.get("root_cause", file_details.get("root_cause", "Unknown")),
                    "fixes": prob.get("how_fixed", file_details.get("how_fixed", "Unknown")),
                    "why_fix_works": prob.get("why_fixed_works", file_details.get("why_fixed_works", "Unknown")),
                    "dependent_files": dependent_files_list,
                    "validation": prob.get("validation_cmd", ""),
                    "is_bulk": False
                }
                l1_file_problems.append(l1_entry)

    return l1_file_problems


def _build_dependency_description(file1: str, file2: str, details1: Dict, details2: Dict, prob: Dict) -> str:
    """
    Build detailed description of how two files are related and what changes were made.

    Args:
        file1: First file path
        file2: Second file path (dependent file)
        details1: Change details for file1
        details2: Change details for file2
        prob: Problem context

    Returns:
        Detailed description of the dependency and changes
    """

    # Extract change information
    how_fixed_1 = details1.get("how_fixed", "")
    how_fixed_2 = details2.get("how_fixed", "")

    # Check if both files have similar changes
    if how_fixed_1 and how_fixed_2:
        # Try to identify the pattern
        if "same" in how_fixed_1.lower() or how_fixed_1 == how_fixed_2:
            # Same change pattern
            return f"Related fix in {file2}: {how_fixed_2[:200]}. Both files needed same change because they are part of the same validation ({prob.get('validation_cmd', '')})"
        else:
            # Different but related changes
            return f"Dependent change in {file2}: {how_fixed_2[:200]}. This file depends on changes in {file1} to pass validation."

    # Fallback: extract specific information from the problem
    what_broke = prob.get("what_broke", "")
    root_cause = prob.get("root_cause", "")

    if "config" in file2.lower() or "toml" in file2.lower():
        return f"Configuration change in {file2}: Modified to enable/configure the fix in {file1}. Root cause: {root_cause[:150]}"
    elif file1.endswith(".py") and file2.endswith(".py"):
        return f"Code change in {file2}: Related Python file that needed the same fix as {file1}. {how_fixed_2[:150] if how_fixed_2 else ''}"
    else:
        return f"Related change in {file2}: {how_fixed_2[:200] if how_fixed_2 else 'Part of the same validation fix'}"


def _extract_file_details(file_path: str, sub_problems: List[Dict]) -> Dict:
    """Extract specific details for a file from sub-problems."""
    for prob in sub_problems:
        if file_path in prob.get("affected_files", []):
            return {
                "root_cause": prob.get("root_cause", ""),
                "how_fixed": prob.get("how_fixed", ""),
                "why_fixed_works": prob.get("why_fixed_works", "")
            }
    return {}


def _stage4_generate_l2_llm(deduplicated: List[Dict], dependencies: Dict, llm, issue_id: str, repo: str) -> Dict:
    """
    Stage 4: Generate L2 (repair sequence) using LLM.

    Create optimal repair sequence with proper ordering and strategy.
    """

    # Prepare data for LLM
    problems_for_llm = []
    for idx, prob in enumerate(deduplicated, 1):
        # Get actual files from sub_problems
        sub_problems = prob.get("sub_problems", [])
        all_files = []
        for sp in sub_problems:
            all_files.extend(sp.get("affected_files", []))
        actual_files = list(dict.fromkeys(all_files))  # Deduplicate

        problems_for_llm.append({
            "id": idx,
            "validation_cmd": prob.get("validation_cmd", ""),
            "type": prob.get("problem_type", ""),
            "what_broke": prob.get("what_broke", ""),
            "root_cause": prob.get("root_cause", "")[:300],
            "how_fixed": prob.get("how_fixed", "")[:300],
            "files_count": len(actual_files),
            "actual_files": actual_files[:20],  # Show first 20 files to LLM
            "issue_types": prob.get("issue_types", [])
        })

    prompt = f"""Generate L2 REPAIR SEQUENCE for CI failure resolution.

Issue: {issue_id}
Repo: {repo}

Problems to fix:
{json.dumps(problems_for_llm, indent=2)}

Dependencies:
{json.dumps(dependencies, indent=2)}

Create optimal repair sequence in this EXACT format:
{{
  "problems": [
    {{
      "problem_id": 1,
      "verification_cmd": "python -m mypy py",
      "failure_type": "type_checking",
      "problem": "Clear description [error_code]",
      "root_cause": "Technical explanation. Scope: affected area",
      "pattern_detected": null or {{
        "type": "bulk_formatting",
        "rule": "Pattern description",
        "scope": "X files"
      }},
      "files": ["path/to/file-*.ext (5 files)", "other/specific/file.py"],
      "file_count": 5,
      "actual_files": ["path/to/file-1.ext", "path/to/file-2.ext", ...],
      "fix_strategy": "Approach: direct_fix | What: What to fix | How: Step-by-step | Why: Why it works | Time: Xmin"
    }}
  ],
  "total_problems": 3,
  "total_time_minutes": 40
}}

CRITICAL Rules:
- "files": Use patterns for clarity (e.g., "contributor-*.rst (6 files)")
- "file_count": MUST include explicit count
- "actual_files": List ACTUAL files from diff (max 50), NO speculation
- Order by repair sequence (use dependency_edges)
- Detect patterns for bulk operations (>10 files)
- Be specific and actionable

{STRICT_JSON_RULES}"""

    try:
        time.sleep(3)  # Rate limiting
        response = _invoke_json(llm, prompt)

        if isinstance(response, dict) and "problems" in response:
            return response
        else:
            print(f"  WARNING: LLM returned invalid L2, using fallback")
            return _fallback_l2(deduplicated, dependencies, issue_id, repo)
    except Exception as e:
        print(f"  ERROR in L2 generation: {e}")
        return _fallback_l2(deduplicated, dependencies, issue_id, repo)


def _generate_file_patterns(files: List[str]) -> List[str]:
    """Generate clear file patterns from file list."""
    from collections import defaultdict
    import os

    # Group files by directory and extension
    dir_groups = defaultdict(list)
    for f in files:
        dirname = os.path.dirname(f)
        basename = os.path.basename(f)
        dir_groups[dirname].append(basename)

    patterns = []
    for dirname, basenames in dir_groups.items():
        if len(basenames) > 3:
            # Find common prefix
            prefixes = defaultdict(list)
            for bn in basenames:
                # Get prefix before first dash or underscore
                parts = bn.replace('_', '-').split('-')
                prefix = parts[0] if len(parts) > 1 else ''
                prefixes[prefix].append(bn)

            for prefix, names in prefixes.items():
                if len(names) > 1 and prefix:
                    ext = os.path.splitext(names[0])[1]
                    patterns.append(f"{dirname}/{prefix}-*{ext} ({len(names)} files)")
                else:
                    for name in names:
                        patterns.append(f"{dirname}/{name}")
        else:
            for bn in basenames:
                patterns.append(f"{dirname}/{bn}")

    return patterns


def _fallback_l2(deduplicated: List[Dict], dependencies: Dict, issue_id: str, repo: str) -> Dict:
    """Fallback: mechanical L2 generation without LLM."""
    repair_order = dependencies.get("repair_order", list(range(1, len(deduplicated) + 1)))

    problems = []
    total_time = 0

    for idx in repair_order:
        prob = deduplicated[idx - 1] if idx <= len(deduplicated) else deduplicated[0]

        # Get ACTUAL files from sub_problems (not speculated)
        sub_problems = prob.get("sub_problems", [])
        all_files = []
        for sp in sub_problems:
            all_files.extend(sp.get("affected_files", []))
        # Deduplicate while preserving order
        files = list(dict.fromkeys(all_files))

        # Generate clear file patterns
        file_patterns = _generate_file_patterns(files) if len(files) > 10 else files

        pattern = None
        if len(files) > 10:
            pattern = {
                "type": "bulk_change",
                "rule": prob.get("how_fixed", "")[:100],
                "scope": f"{len(files)} files"
            }

        time_est = 5 if len(files) < 5 else (15 if len(files) < 20 else 30)
        total_time += time_est

        l2_prob = {
            "problem_id": len(problems) + 1,
            "verification_cmd": prob.get("validation_cmd", ""),
            "failure_type": prob.get("problem_type", "unknown").replace("_", " "),
            "problem": f"{prob.get('what_broke', 'Unknown')} [{prob.get('issue_types', ['unknown'])[0] if prob.get('issue_types') else 'unknown'}]",
            "root_cause": f"{prob.get('root_cause', 'Unknown')[:200]} Scope: {len(files)} files",
            "pattern_detected": pattern,
            "files": file_patterns,  # Clear patterns instead of raw paths
            "file_count": len(files),  # Explicit count
            "actual_files": files[:50] if len(files) > 50 else files,  # First 50 actual files
            "fix_strategy": f"Approach: direct_fix | How: {prob.get('how_fixed', '')[:150]} | Time: {time_est}min"
        }
        problems.append(l2_prob)

    return {
        "problems": problems,
        "total_problems": len(problems),
        "total_time_minutes": total_time
    }


def _stage5_generate_l3_llm(l1: List[Dict], l2: Dict, deduplicated: List[Dict], llm) -> List[Dict]:
    """
    Stage 5: Generate L3 (universal patterns) using LLM.

    Analyze DISTINCT problem patterns and extract universal fixes.
    Each independent problem = separate entry.
    """

    # Group L2 problems by verification command to identify distinct patterns
    l2_problems = l2.get("problems", [])

    prompt = f"""Analyze CI failure patterns and extract UNIVERSAL PROBLEM PATTERNS (L3).

L2 Repair Sequence:
{json.dumps(l2_problems, indent=2)[:10000]}

Task: Extract distinct, independent problem patterns with universal fixes.

CRITICAL RULES:
1. Each INDEPENDENT problem = separate entry
2. Only link problems if there's ACTUAL dependency
3. No forced grouping - mypy ≠ mdformat (separate entries)
4. Extract universal fixes that apply to similar future problems

Output JSON ARRAY of distinct patterns:
[
  {{
    "pattern_id": "numpy_private_type_annotation",
    "failure_type": "type checking",
    "verification_cmd": "python -m mypy py",
    "failure_pattern": "Private numpy type annotations fail without plugin",
    "problem": "< what is the problem, why it occurs, root cause >",
    "universal_fix": {{
      "approach": "Replace private types with public equivalents",
      "steps": [
        "1. Identify private type imports",
        "2. Find public equivalent (e.g., DTypeLike -> np.dtype[Any])",
        "3. Update annotations",
        "4. Remove plugin config"
      ],
      "applies_to": ["numpy private types", "pandas private types"]
    }},
    "examples": [
      {{
        "file": "ndarrays_arithmetic.py",
        "before": "dtype: DTypeLike = np.int64",
        "after": "dtype: np.dtype[Any] = np.int64"
      }}
    ],
    "dependent_problems": [
      {{
        "pattern_id": "pyproject_plugin_config",
        "relationship": "requires_config_change",
        "rationale": "Code change requires matching config update"
      }}
    ]
  }},
  {{
    "pattern_id": "rst_header_formatting",
    "failure_type": "formatting",
    "verification_cmd": "python -m mdformat --check",
    "failure_pattern": "RST headers with mismatched underlines",
    "problem": "...",
    "universal_fix": {{...}},
    "examples": [...],
    "dependent_problems": []
  }}
]

For EACH distinct validation/problem type, create separate entry.
Include dependencies ONLY if they actually exist.

{STRICT_JSON_RULES}"""

    try:
        time.sleep(3)  # Rate limiting
        response = _invoke_json(llm, prompt)

        if isinstance(response, list):
            return response
        else:
            print(f"  WARNING: LLM returned invalid L3, using fallback")
            return _fallback_l3(l2)
    except Exception as e:
        print(f"  ERROR in L3 generation: {e}")
        return _fallback_l3(l2)


def _fallback_l3(l2: Dict) -> List[Dict]:
    """Fallback: basic L3 pattern extraction without LLM."""
    problems = l2.get("problems", [])

    # Create one pattern per distinct problem
    patterns = []
    for p in problems:
        pattern = {
            "pattern_id": f"pattern_{p.get('problem_id', 'unknown')}",
            "failure_type": p.get("failure_type", "unknown"),
            "verification_cmd": p.get("verification_cmd", ""),
            "failure_pattern": p.get("problem", "")[:200],
            "problem": p.get("problem", "") + " " + p.get("root_cause", "")[:200],
            "universal_fix": {
                "approach": p.get("fix_strategy", "")[:100],
                "steps": ["Fallback - no detailed steps available"],
                "applies_to": ["Similar validation failures"]
            },
            "examples": [],
            "dependent_problems": []
        }
        patterns.append(pattern)

    return patterns


def convert_to_l2_format(decomposed_result: Dict) -> Dict:
    """
    Convert decomposed issue to clean L2 format.

    L2 Format:
    - Deduplicates similar problems (same error + fix pattern)
    - Detects bulk patterns (83 files with same fix)
    - Orders by repair sequence (primary -> consecutive -> enablement)
    - Compact strings (not nested objects)

    Args:
        decomposed_result: Output from decompose_issue()

    Returns:
        L2 format dictionary with deduplicated, ordered problems
    """

    if "error" in decomposed_result or not decomposed_result.get("problems"):
        return decomposed_result

    issue_id = decomposed_result.get("original_issue_id", "?")
    problems = decomposed_result.get("problems", [])

    print(f"\n  Converting to L2 format...")
    print(f"    Input: {len(problems)} problems")

    # Group problems by validation command ONLY (not by error type)
    # This merges all problems from same validation into ONE problem
    problem_groups = {}
    for prob in problems:
        validation = prob.get("validation_cmd", "unknown")
        # Key is just validation command - merge ALL problems from same validation
        key = validation

        if key not in problem_groups:
            problem_groups[key] = []
        problem_groups[key].append(prob)

    # Merge each group into one L2 problem
    l2_problems = []
    problem_id = 1

    for group_key, group_problems in problem_groups.items():
        if not group_problems:
            continue

        # Use first problem as template
        template = group_problems[0]

        # Collect all affected files
        all_files = []
        for p in group_problems:
            all_files.extend(p.get("affected_files", []))
        all_files = list(dict.fromkeys(all_files))  # Deduplicate preserving order

        # Detect pattern if multiple files with same fix
        pattern_detected = None
        if len(all_files) > 5:  # Bulk operation threshold
            # Extract pattern info
            issue_type = template.get("issue_type", "")
            how_fixed = template.get("how_fixed", "")

            # Determine pattern type
            pattern_type = "bulk_change"
            if "format" in issue_type.lower() or "format" in how_fixed.lower():
                pattern_type = "bulk_formatting"
            elif "config" in issue_type.lower() or "enable" in how_fixed.lower():
                pattern_type = "config_enablement"

            # Extract rule from how_fixed
            rule = how_fixed[:100] if how_fixed else "Pattern not specified"

            pattern_detected = {
                "type": pattern_type,
                "rule": rule,
                "scope": f"{len(all_files)} files in {template.get('validation_cmd', 'unknown')}"
            }

        # Build root cause string (summarize from all problems in group)
        root_causes = set()
        scopes = set()
        for p in group_problems:
            if p.get("root_cause"):
                root_causes.add(p["root_cause"][:150])
            if p.get("what_broke"):
                scopes.add(p["what_broke"][:80])

        root_cause_parts = []
        if root_causes:
            # Take first unique root cause (they're usually similar)
            root_cause_parts.append(list(root_causes)[0])
        if scopes:
            scope_str = f"Affects {len(all_files)} files"
            root_cause_parts.append(f"Scope: {scope_str}")
        root_cause = " ".join(root_cause_parts) if root_cause_parts else "Unknown root cause"

        # Build fix strategy string
        fix_parts = []

        # Approach
        problem_type = template.get("problem_type", "unknown")
        if problem_type == "primary_failure":
            approach = "direct_fix"
        elif problem_type == "enablement_fix":
            approach = "enablement"
        elif "workaround" in template.get("how_fixed", "").lower():
            approach = "workaround"
        else:
            approach = "direct_fix"
        fix_parts.append(f"Approach: {approach}")

        # What
        what_broke = template.get("what_broke", "")
        if what_broke:
            fix_parts.append(f"What: {what_broke[:100]}")

        # How
        how_fixed = template.get("how_fixed", "")
        if how_fixed:
            fix_parts.append(f"How: {how_fixed[:200]}")

        # Why
        why_works = template.get("why_fixed_works", "")
        if why_works:
            fix_parts.append(f"Why: {why_works[:100]}")

        # Time estimate (rough)
        time_estimate = 5  # default
        if len(all_files) > 50:
            time_estimate = 30
        elif len(all_files) > 10:
            time_estimate = 15
        elif len(all_files) > 2:
            time_estimate = 10
        fix_parts.append(f"Time: {time_estimate}min")

        fix_strategy = " | ".join(fix_parts)

        # Build problem string (summarize all error types in group)
        error_types = set(p.get("issue_type", "unknown") for p in group_problems)
        if len(error_types) == 1:
            error_code = list(error_types)[0]
        else:
            error_code = f"{len(error_types)} error types"

        what_broke = template.get("what_broke", "Validation failed")
        problem_str = f"{what_broke} [{error_code}]"

        # Build L2 problem
        l2_problem = {
            "problem_id": problem_id,
            "verification_cmd": template.get("validation_cmd", ""),
            "failure_type": template.get("problem_type", "unknown").replace("_", " "),
            "problem": problem_str,
            "root_cause": root_cause,
            "pattern_detected": pattern_detected,
            "files": all_files,
            "fix_strategy": fix_strategy
        }

        l2_problems.append(l2_problem)
        problem_id += 1

    # Sort by problem type (primary first, then enablement, then consecutive)
    def problem_priority(p):
        failure_type = p.get("failure_type", "")
        if "primary" in failure_type:
            return 0
        elif "enablement" in failure_type:
            return 2
        else:
            return 1

    l2_problems.sort(key=problem_priority)

    # Reassign problem IDs after sorting
    for idx, prob in enumerate(l2_problems, 1):
        prob["problem_id"] = idx

    # Calculate total time
    total_time = sum(
        int(p["fix_strategy"].split("Time: ")[-1].split("min")[0])
        for p in l2_problems
        if "Time: " in p["fix_strategy"]
    )

    print(f"    Output: {len(l2_problems)} problems (deduplicated)")
    print(f"    Total time estimate: {total_time} minutes")

    # Build L2 result
    l2_result = {
        "issue_id": str(issue_id),
        "repo": decomposed_result.get("repo", "unknown"),
        "problems": l2_problems,
        "total_problems": len(l2_problems),
        "total_time_minutes": total_time
    }

    return l2_result


def _save_to_memory_files(results: List[Dict], output_dir: str):
    """
    Save results to 3 memory files:
    1. failure_memory.json - L1 file-level problems (flat array)
    2. repo_memory.json - L2 repair sequences per issue
    3. cross_memory.json - L3 analysis per issue
    """

    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter successful results (skip errors)
    successful = [r for r in results if "l1_file_level" in r or "l2_repair_sequence" in r]

    # 1. failure_memory.json - Flat array of ALL file-level problems (L1)
    all_l1_problems = []
    for result in successful:
        l1_problems = result.get("l1_file_level", [])
        all_l1_problems.extend(l1_problems)

    # 2. l2_repair_sequences.json - Simplified format
    l2_sequences = []
    for result in successful:
        l2_data = result.get("l2_repair_sequence", {})

        # Fix failure_type underscores in problems
        l2_problems = []
        for prob in l2_data.get('problems', []):
            prob['failure_type'] = prob.get('failure_type', '').replace('_', ' ')
            l2_problems.append(prob)

        # Get workflow path
        workflow_path = result.get("workflow_path", "unknown")

        l2_sequences.append({
            "repo": result.get("repo"),
            "workflow": workflow_path,
            "problems": l2_problems
        })

    # Save memory files
    with open(output_dir / "failure_memory.json", "w") as f:
        json.dump(all_l1_problems, f, indent=2)

    with open(output_dir / "repo_memory.json", "w") as f:
        json.dump(l2_sequences, f, indent=2)

    # L3 analysis - merge all patterns from all issues
    all_l3_patterns = []
    for result in successful:
        l3_patterns = result.get("l3_analysis", [])
        if isinstance(l3_patterns, list):
            all_l3_patterns.extend(l3_patterns)

    with open(output_dir / "cross_memory.json", "w") as f:
        json.dump(all_l3_patterns, f, indent=2)

    print(f"  -> Saved to {output_dir}/")
    print(f"     - failure_memory.json ({len(all_l1_problems)} file problems)")
    print(f"     - repo_memory.json ({len(l2_sequences)} issues)")
    print(f"     - cross_memory.json ({len(all_l3_patterns)} patterns)")


def load_issues_from_huggingface(issue_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Load issues directly from HuggingFace dataset.

    Args:
        issue_ids: Optional list of issue IDs to filter. If None, loads all issues.

    Returns:
        List of issues matching the provided IDs (or all if no IDs specified)
    """
    print("Loading dataset from HuggingFace: ci-benchmark-user/ci-repair-bench")

    # Load with verification disabled to bypass feature compatibility issues
    try:
        # Delete cached dataset info to force reload
        import shutil
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "ci-benchmark-user___ci-repair-bench"
        if cache_dir.exists():
            info_file = cache_dir / "default" / "0.0.0" / "dataset_info.json"
            if info_file.exists():
                print("Removing cached dataset_info.json to force reload...")
                info_file.unlink()

        # Load without verification
        ds = load_dataset(
            "ci-benchmark-user/ci-repair-bench",
            verification_mode="no_checks",
            download_mode="reuse_cache_if_exists"
        )
        data = ds['train']
        print(f"Loaded {len(data)} issues from HuggingFace")

    except Exception as e:
        print(f"Dataset loading failed: {e}")
        raise RuntimeError(f"Could not load dataset from HuggingFace: {e}")

    if issue_ids:
        # Convert to set for faster lookup
        issue_ids_set = set(str(id) for id in issue_ids)

        # Filter to only requested IDs
        issues = []
        for item in data:
            if str(item.get('id')) in issue_ids_set:
                issues.append(dict(item))

        print(f"Filtered to {len(issues)} issues matching provided IDs")

        # Warn about missing IDs
        found_ids = set(str(i.get('id')) for i in issues)
        missing_ids = issue_ids_set - found_ids
        if missing_ids:
            print(f"WARNING: {len(missing_ids)} IDs not found in dataset: {sorted(missing_ids)}")
    else:
        # Load all issues
        issues = [dict(item) for item in data]
        print(f"Loaded all {len(issues)} issues from dataset")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Reverse engineer CI failures into atomic problems (visible + hidden)"
    )
    parser.add_argument("--issue-id", help="Single issue ID to decompose")
    parser.add_argument("--batch", action="store_true", help="Decompose all memory issues from HuggingFace")
    parser.add_argument("--use-huggingface", action="store_true", help="Load from HuggingFace instead of local JSON")
    parser.add_argument("--eval-issues", default="data/trs/eval_issues.json", help="Path to eval issues (legacy mode)")
    parser.add_argument("--output-dir", default="data/trs", help="Output directory for memory files")
    parser.add_argument(
        "--model",
        default="openrouter/minimax/minimax-m2.5",
        help="LLM model. Use openrouter/minimax/minimax-m2.5 for MiniMax M2.5 via OpenRouter.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of issues to process")
    args = parser.parse_args()

    # Determine loading mode
    if args.batch or args.use_huggingface or args.issue_id:
        # Load from HuggingFace
        print(f"\n{'='*80}")
        print("Loading issues from HuggingFace dataset")
        print(f"{'='*80}")

        if args.batch:
            # Load all memory issues
            print(f"Using MEMORY_ISSUE_IDS: {MEMORY_ISSUE_IDS}")
            issues = load_issues_from_huggingface(MEMORY_ISSUE_IDS)
        elif args.issue_id:
            # Load specific issue
            issues = load_issues_from_huggingface([args.issue_id])
            if not issues:
                print(f"ERROR Issue {args.issue_id} not found in HuggingFace dataset")
                return 1
        else:
            # Load all issues
            issues = load_issues_from_huggingface(None)
    else:
        # Legacy mode: Load from local JSON file
        print(f"\n{'='*80}")
        print(f"Loading issues from local file: {args.eval_issues}")
        print(f"{'='*80}")

        eval_path = Path(args.eval_issues)
        if not eval_path.exists():
            print(f"ERROR Eval issues not found: {eval_path}")
            print(f"TIP: Use --use-huggingface to load from HuggingFace instead")
            return 1

        with open(eval_path) as f:
            issues = json.load(f)

        print(f"Loaded {len(issues)} issues from {eval_path}")

    # Limit if requested
    if args.limit:
        issues = issues[:args.limit]
        print(f"Limited to first {args.limit} issues")

    # Initialize LLM
    print(f"\n{'='*80}")
    print(f"Initializing LLM: {args.model}")
    print(f"{'='*80}")
    llm = LitellmModel(model_name=args.model)

    # Prepare output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results if file exists (for resume capability)
    results = []
    errors = []
    processed_ids = set()
    l2_sequences_path = output_dir / "l2_repair_sequences.json"
    decomposed_issues_path = output_dir / "decomposed_issues.json"

    # Load existing decomposed issues (old format - need L1/L2/L3 conversion)
    decomposed_cache = {}
    if decomposed_issues_path.exists():
        try:
            with open(decomposed_issues_path) as f:
                existing_decomposed = json.load(f)
                if isinstance(existing_decomposed, list):
                    for item in existing_decomposed:
                        issue_id = str(item.get("original_issue_id", item.get("issue_id")))
                        decomposed_cache[issue_id] = item
                    print(f"Found {len(decomposed_cache)} decomposed issues (can reuse for L1/L2/L3)")
        except Exception as e:
            print(f"Warning: Could not load decomposed issues: {e}")

    # Load existing L2 sequences (new format - already has L1/L2/L3)
    if l2_sequences_path.exists():
        try:
            with open(l2_sequences_path) as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    # Build results from L2 sequences
                    results = existing
                    # Track already processed issues
                    for r in results:
                        if "issue_id" in r:
                            processed_ids.add(str(r["issue_id"]))
                    print(f"Loaded {len(results)} existing L1/L2/L3 results (will skip)")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    # Decompose issues with incremental saving
    for i, issue in enumerate(issues, 1):
        issue_id = str(issue.get("id"))

        # Skip if already in L1/L2/L3 format
        if issue_id in processed_ids:
            print(f"\nProgress: {i}/{len(issues)} - Issue {issue_id} already has L1/L2/L3, skipping")
            continue

        print(f"\nProgress: {i}/{len(issues)}")

        # Check if we have decomposed data (can skip decomposition)
        if issue_id in decomposed_cache:
            print(f"  Found in decomposed_issues.json - Building L1/L2/L3 directly (no decomposition needed)")
            decomposed_result = decomposed_cache[issue_id]
        else:
            # Need to decompose from scratch
            print(f"  Not found in cache - Running full decomposition...")
            decomposed_result = decompose_issue(issue, llm)

            # Save decomposed result to cache immediately
            decomposed_cache[issue_id] = decomposed_result

            # Save decomposed_issues.json after each decomposition
            decomposed_list = list(decomposed_cache.values())
            with open(decomposed_issues_path, "w") as f:
                json.dump(decomposed_list, f, indent=2)
            print(f"  OK Saved to decomposed_issues.json ({len(decomposed_list)} issues)")

        # Check for errors
        if "error" in decomposed_result:
            errors.append(decomposed_result)
            results.append(decomposed_result)
        else:
            # Run full L1/L2/L3 pipeline
            result = generate_l1_l2_l3_pipeline(decomposed_result, llm)
            results.append(result)

        # Incremental save after each issue - save to 3 memory files
        try:
            _save_to_memory_files(results, args.output_dir)
            print(f"  OK Saved progress ({len(results)} issues total)")
        except Exception as e:
            print(f"  WARNING: Could not save progress: {e}")

    # Final save - save to 3 memory files
    _save_to_memory_files(results, args.output_dir)

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

    print(f"\nOutput saved to: {output_dir}/")

    if errors:
        print(f"\nWARNING:  {len(errors)} issues had errors")
        print(f"Issue IDs with errors: {[e.get('original_issue_id') for e in errors[:5]]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
