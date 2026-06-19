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
from deterministic_diff_parser import (
    parse_diff_to_structured,
    chunk_structured_diff,
    format_structured_for_llm
)

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
    try:
        response = llm.invoke(prompt)
        content = str(getattr(response, "content", response) or "").strip()
    except Exception as exc:
        LOGGER.error(f"LLM API call failed: {type(exc).__name__}: {exc}")
        raise  # Re-raise so caller can handle it

    if not content:
        LOGGER.warning("LLM returned empty content")
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


def build_correlation_context(
    ci_context: Dict[str, Any],
    validation_groups: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build complete layered correlation context.

    This is DETERMINISTIC - no LLM needed.

    Returns structure showing:
    - Layer 1: Primary CI failures
    - Layer 2: How each was addressed (direct/enablement/missing/unresolved)
    - Layer 3: Enablement cascades
    - Layer 4: Secondary failures revealed
    """
    # Layer 1: Extract primary failures
    primary_failures = extract_primary_ci_failures(ci_context)

    # Layer 2: Match with diff fixes
    matched_failures = match_failures_with_diff_fixes(primary_failures, validation_groups)

    # Layer 3: Detect enablement cascades
    enablement_cascades = detect_enablement_cascades(matched_failures, validation_groups)

    # Categorize by fix type
    direct_fixes = [f for f in matched_failures if f.get('fix_type') == 'DIRECT_FIX']
    enablement_fixes = [f for f in matched_failures if f.get('fix_type') == 'ENABLEMENT_FIX']
    missing_fixes = [f for f in matched_failures if f.get('fix_type') == 'MISSING_FIX']
    unresolved = [f for f in matched_failures if f.get('fix_type') == 'UNRESOLVED']

    return {
        "primary_failures": primary_failures,
        "matched_failures": matched_failures,
        "direct_fixes": direct_fixes,
        "enablement_fixes": enablement_fixes,
        "missing_fixes": missing_fixes,
        "unresolved": unresolved,
        "enablement_cascades": enablement_cascades,
        "total_layers": {
            "layer_1_primary": len(primary_failures),
            "layer_2_direct_fixes": len(direct_fixes),
            "layer_2_enablements": len(enablement_fixes),
            "layer_2_missing": len(missing_fixes),
            "layer_3_cascades": len(enablement_cascades),
        }
    }


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
                            "error_code": sub_problem.get("error_code"),
                            "chunks": [],
                            "all_files": [],
                            "all_changes": [],
                            "has_pattern": False,
                            "total_files": 0,
                        }

                    validation_groups[key]["chunks"].append(chunk["chunk_index"])
                    validation_groups[key]["all_files"].extend(sub_problem.get("files", []))
                    validation_groups[key]["all_changes"].extend(sub_problem.get("changes", []))
                    validation_groups[key]["total_files"] += sub_problem.get("total_files", len(sub_problem.get("files", [])))

                    # Cross-chunk pattern detection
                    if sub_problem.get("has_pattern"):
                        validation_groups[key]["has_pattern"] = True
            else:
                # Single failure type - use validation_order as key
                key = str(val_order)

                if key not in validation_groups:
                    validation_groups[key] = {
                        "validation_order": val_order,
                        "validation_cmd": val_cmd,
                        "has_sub_problems": False,
                        "chunks": [],
                        "all_files": [],
                        "all_changes": [],
                        "has_pattern": False,
                        "total_files": 0,
                    }

                validation_groups[key]["chunks"].append(chunk["chunk_index"])
                validation_groups[key]["all_files"].extend(val_entry.get("files", []))
                validation_groups[key]["all_changes"].extend(val_entry.get("changes", []))
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
      "failed_type": "<failure category:e.g., syntax error, type error, test failure>",
      "issue_type": " <failure sub category> e.g., missing import, unused variable, test assertion failure>",
      "why_failed": "Clear explanation of root cause",

      "affected_files": [
        {{
          "file": "path/to/file.py",
          "what_is_wrong": "Specific issue in this file",
          "fix_strategy": "What needs to be modified overall in this file to fix",
          "why_fix_works": "Short explanation why this fix would resolve the validation failure"
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
          "why_fix_works": "Short explanation why this fix would resolve the validation failure"
        }}
      ]
    }}
  ],
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


def _estimate_tokens(data: Any) -> int:
    """
    Rough token estimation for data.

    Rule of thumb: 1 token ≈ 4 characters
    This is approximate but good enough for batching.
    """
    json_str = json.dumps(data)
    return len(json_str) // 4


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
            'all_files': val_group.get('all_files', []),  # Keep full file list
            'all_changes': chunk_changes,
            'chunk_info': f"Changes {start_idx + 1}-{start_idx + len(chunk_changes)} of {len(all_changes)} total"
        }
        chunks.append(chunk)

    return chunks


def _create_previous_findings_summary(problems: List[Dict[str, Any]]) -> str:
    """
    Create a compact summary of previous problems for context.
    """
    if not problems:
        return "No previous problems identified yet (this is the first validation)."

    summary_items = []
    for p in problems:
        item = (
            f"Problem {p.get('problem_id')}: "
            f"{p.get('problem_type', 'unknown')} - "
            f"{p.get('issue_type', 'N/A')} "
            f"(validation {p.get('validation_order')})"
        )

        # Add key relationships
        if p.get('enables_validation'):
            item += " [ENABLES future validations]"
        if p.get('enabled_by'):
            item += f" [ENABLED BY: {p.get('enabled_by')}]"
        if p.get('is_workaround'):
            item += f" [WORKAROUND for Problem {p.get('related_to')}]"

        summary_items.append(item)

    return "\n".join(summary_items)


def _verify_config_files_included(
    chunk: Dict[str, Any],
    chunk_problems: List[Dict[str, Any]],
    chunk_idx: int
) -> None:
    """
    CRITICAL: Verify that config/dependency files are not filtered out.

    This is a safety check to ensure ground truth config changes are preserved.
    """
    config_file_patterns = [
        'pyproject.toml', 'package.json', 'requirements.txt', 'setup.py',
        'Cargo.toml', 'pom.xml', 'build.gradle', 'Gemfile', 'go.mod',
        '.config.js', '.config.ts', 'tsconfig.json', 'webpack.config',
        '.github/workflows/', '.circleci/config.yml'
    ]

    # Get all files from chunk changes
    all_changes = chunk.get('all_changes', [])
    config_files_in_chunk = set()

    for change in all_changes:
        file_path = change.get('file', '')
        if any(pattern in file_path for pattern in config_file_patterns):
            config_files_in_chunk.add(file_path)

    if not config_files_in_chunk:
        return  # No config files in this chunk

    # Get all files from problems
    files_in_problems = set()
    for problem in chunk_problems:
        files_in_problems.update(problem.get('affected_files', []))

    # Check for missing config files
    missing_config_files = config_files_in_chunk - files_in_problems

    if missing_config_files:
        print(f"        ⚠️  WARNING: Config files found in chunk but NOT in problems!")
        print(f"            Missing: {missing_config_files}")
        print(f"            This violates the rule: NEVER remove config file changes")
        print(f"            These files MUST appear in enablement_fix problems")
        # This is a warning, not an error - the LLM should have included them


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
        print(f"\n  🚨 CRITICAL ERROR: Config files missing from decomposition!")
        print(f"     Config files in ground truth: {all_config_files}")
        print(f"     Missing from problems: {missing_config_files}")
        print(f"\n     These files MUST be included in enablement_fix problems.")
        print(f"     This is a violation of: NEVER remove config file changes from ground truth")
    else:
        print(f"  ✓ Config file verification: All {len(all_config_files)} config files included in problems")
        # If this happens frequently, we need to strengthen the prompt


def analyze_validation_groups_with_reasoning(
    validation_groups: Dict[str, Any],
    validation_sequence: List[Dict[str, Any]],
    ci_context: Dict[str, Any],
    correlation_context: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    """
    Step 3: Deep reasoning on merged validation groups with CI-Diff correlation.

    Creates specific, actionable problem statements for mini-swe-agent.
    Uses layered structure from correlation to understand:
    - Primary failures (Layer 1)
    - How each was fixed (Layer 2)
    - Enablement cascades (Layer 3)
    - Secondary failures (Layer 4)
    """

    groups_data = validation_groups.get("validation_groups", {})

    # Count validation groups
    validation_group_count = len(groups_data)
    validation_orders = sorted([int(k) for k in groups_data.keys()])

    # Extract correlation info for prompt
    direct_fixes = correlation_context.get('direct_fixes', [])
    enablement_fixes = correlation_context.get('enablement_fixes', [])
    missing_fixes = correlation_context.get('missing_fixes', [])
    unresolved = correlation_context.get('unresolved', [])
    cascades = correlation_context.get('enablement_cascades', [])

    # Process validations SEQUENTIALLY with context flow
    print(f"  Step 3a: Processing {validation_group_count} validation groups sequentially...")

    all_atomic_problems = []
    next_problem_id = 1

    # Sort by validation order (sequential processing)
    sorted_validations = sorted(groups_data.items(), key=lambda x: int(x[0]))

    for val_idx, (val_order, val_group) in enumerate(sorted_validations, 1):
        val_order_int = int(val_order)

        print(f"    Validation {val_idx}/{validation_group_count}: Order {val_order} ({val_group.get('validation_cmd', '')})")

        # Get correlation for THIS validation
        val_direct_fixes = [f for f in direct_fixes if f.get('validation_order') == val_order_int]
        val_enablement_fixes = [f for f in enablement_fixes if f.get('validation_order') == val_order_int]
        val_missing_fixes = [f for f in missing_fixes if f.get('validation_order') == val_order_int]
        val_cascades = [c for c in cascades if c.get('validation_order') == val_order_int]

        # Check if validation needs chunking
        num_changes = len(val_group.get('all_changes', []))
        chunks = _chunk_validation_changes(val_group, max_changes_per_chunk=400)

        if len(chunks) > 1:
            print(f"      Large validation ({num_changes} changes) → splitting into {len(chunks)} chunks")

        # Process each chunk of this validation
        validation_problems = []

        for chunk_idx, chunk in enumerate(chunks, 1):
            # Create summary of previous findings
            previous_findings = _create_previous_findings_summary(all_atomic_problems)

            chunk_suffix = f" - Chunk {chunk_idx}/{len(chunks)}" if len(chunks) > 1 else ""
            print(f"      Processing{chunk_suffix}...")

            prompt = f"""Create atomic problems for Validation {val_order}{chunk_suffix} (#{val_idx}/{validation_group_count})

CONTEXT:
- Start problem IDs from: {next_problem_id}
- Previous problems: {len(all_atomic_problems)}

{previous_findings if previous_findings else '(No previous problems yet)'}

CORRELATION (use to set problem_type):
- Direct: {len(val_direct_fixes)} | Enablement: {len(val_enablement_fixes)} | Missing: {len(val_missing_fixes)} | Cascades: {len(val_cascades)}

DATA:
Validation: {chunk.get('validation_cmd', '')}
Changes: {json.dumps(chunk.get('all_changes', []), indent=2)}

PROBLEM TYPE MAPPING:
- Direct fixes → primary_failure
- Enablement fixes → enablement_fix (config/dependency changes)
- Missing fixes → 2 problems: unresolved_root_cause + workaround_fix
- Cascades → 2 problems: enablement_fix (config) + secondary_failure (code)

CONFIG FILE RULES (CRITICAL):
1. If any config file in all_changes (pyproject.toml, package.json, requirements.txt, Cargo.toml, pom.xml, build.gradle, etc.) → MUST be in a problem's affected_files
2. Always quote actual changes: "Line X: changed from 'BEFORE' to 'AFTER'"
3. Create separate enablement_fix for all config changes

HOW TO UNDERSTAND AND EXPLAIN TOOLS:

Step 1: LOOK AT THE EVIDENCE
- validation_cmd shows: tool name + what it checks + which files/directories
  Example: "taplo fmt --check ../benchmarks ../examples"
  → Tool: taplo, Action: fmt --check (format checking), Target: benchmarks/examples dirs

Step 2: INFER TOOL PURPOSE from command structure
- "fmt --check" → formatting/style checking
- File extensions in target dirs → what file type (look at the actual directory)
- Tool name + command → specific capability
  Example: "taplo fmt --check" on dirs with .toml files → TOML formatter

Step 3: CHECK CI ERRORS for what actually failed
- CI error mentions the tool + what was wrong
- Use this to confirm/refine your understanding

Step 4: EXPLAIN THE COMPLETE PICTURE (all 4 required fields):

**problem**: What failed + WHY + what feature/capability was blocked
Format: "Validation '<cmd>' failed because <root issue>, preventing <capability> from running"
Example: "Validation 'taplo fmt --check ../benchmarks ../examples' failed because taplo dependency was disabled in pyproject.toml, preventing TOML formatting validation in benchmarks/examples directories from running"

**root_cause**: Technical reason WHY it failed (cite specific before state)
Format: "Dependency/file X was <before state> at line Y, causing <technical consequence>"
Example: "The taplo dependency was commented out in dev/pyproject.toml (line 18) and framework/pyproject.toml (line 98), preventing the taplo TOML formatter from being installed. Without taplo installed, the validation command 'taplo fmt --check' cannot execute."

**how_fixed**: Exact before→after changes with line numbers
Format: "File:line changed from '<before>' to '<after>'"
Example: "dev/pyproject.toml line 18: changed from '#taplo = \"==0.9.3\"' to 'taplo = \"==0.9.3\"'; framework/pyproject.toml line 98: changed from '#taplo = \"==0.9.3\"' to 'taplo = \"==0.9.3\"'"

**why_fix_works**: Technical explanation of HOW the fix resolves the issue + what capability it enables
Format: "Doing X enables Y, allowing Z validation to run on W files"
Example: "Uncommenting the taplo dependency enables installation of the taplo TOML formatter. With taplo installed, the 'taplo fmt --check' command can now validate TOML file formatting in ../benchmarks and ../examples directories, checking that TOML configuration files follow proper formatting standards."

GOOD vs BAD:
✓ GOOD: "Validation 'taplo fmt --check' failed because taplo was disabled, preventing TOML formatting validation"
✗ BAD: "Enabled taplo for RST validation" (wrong file type - inferred incorrectly)
✗ BAD: "Fixed configuration" (vague, no explanation)
✗ BAD: "Uncommented taplo" (missing what taplo does and why it matters)

QUALITY RULES:
- Be SPECIFIC and TECHNICAL in all explanations
- Use EVIDENCE from validation_cmd, all_changes, CI errors
- Quote exact before→after code in how_fixed
- Group same-pattern files into ONE problem

{STRICT_JSON_RULES}

OUTPUT:
{{
  "atomic_problems": [
    {{
      "problem_id": <int starting from {next_problem_id}>,
      "problem_type": "primary_failure|enablement_fix|secondary_failure|unresolved_root_cause|workaround_fix",
      "validation_order": {val_order},
      "validation_cmd": "<exact command>",
      "failure_type": "<Type Checking|Code Formatting|etc>",
      "issue_type": "<Import Error|Heading Style|etc>",
      "problem": "<what failed - be specific>",
      "root_cause": "<why failed - cite all_changes>",
      "how_fixed": "<quote before→after changes>",
      "why_fix_works": "<why fix works>",
      "affected_files": ["<exact files from all_changes>"],
      "depends_on": [<ids or empty>],
      "enables_validation": true, // if enablement_fix
      "reveals_secondary": true,  // if cascade
      "enabled_by": [<id>],       // if secondary_failure
      "unresolved": true,         // if unresolved_root_cause
      "proper_fix": "<what should have been done>",
      "is_workaround": true,      // if workaround_fix
      "related_to": <id>          // if workaround_fix
    }}
  ]
}}
"""

            chunk_result = _invoke_json(llm, prompt)

            # Normalize result
            if isinstance(chunk_result, list):
                if len(chunk_result) == 1 and isinstance(chunk_result[0], dict):
                    chunk_result = chunk_result[0]
                else:
                    chunk_result = {"atomic_problems": chunk_result if isinstance(chunk_result, list) else []}

            if not isinstance(chunk_result, dict):
                print(f"        Warning: Chunk {chunk_idx} returned {type(chunk_result).__name__}, expected dict")
                chunk_result = {"atomic_problems": []}

            # Extract problems from this chunk
            chunk_problems = chunk_result.get("atomic_problems", [])
            print(f"        ✓ Chunk {chunk_idx}: {len(chunk_problems)} atomic problems created")

            # CRITICAL: Verify config files are not filtered out
            _verify_config_files_included(chunk, chunk_problems, chunk_idx)

            # Renumber problem IDs sequentially
            for problem in chunk_problems:
                old_id = problem.get("problem_id", 1)
                problem["problem_id"] = next_problem_id

                # Update references to account for sequential numbering
                if "depends_on" in problem and isinstance(problem["depends_on"], list):
                    problem["depends_on"] = [dep_id + (next_problem_id - old_id) if dep_id < next_problem_id else dep_id for dep_id in problem["depends_on"]]

                if "enabled_by" in problem and isinstance(problem["enabled_by"], list):
                    problem["enabled_by"] = [en_id + (next_problem_id - old_id) if en_id < next_problem_id else en_id for en_id in problem["enabled_by"]]

                if "revealed_by" in problem and isinstance(problem["revealed_by"], int):
                    if problem["revealed_by"] < next_problem_id:
                        problem["revealed_by"] = problem["revealed_by"] + (next_problem_id - old_id)

                if "related_to" in problem and isinstance(problem["related_to"], int):
                    if problem["related_to"] < next_problem_id:
                        problem["related_to"] = problem["related_to"] + (next_problem_id - old_id)

                next_problem_id += 1

            # Add chunk problems to validation problems
            validation_problems.extend(chunk_problems)

        # Add validation problems to all problems
        all_atomic_problems.extend(validation_problems)
        print(f"      ✓ Validation {val_order}: {len(validation_problems)} total atomic problems")

    # Final result with all merged problems
    result = {
        "atomic_problems": all_atomic_problems,
        "sequential_workflow_metadata": {}  # Can merge from batches if needed
    }

    # CRITICAL: Final verification that config files are included
    _final_verify_config_files(validation_groups, all_atomic_problems)

    # Report on validation group coverage
    atomic_problems = all_atomic_problems
    print(f"  ✓ All {validation_group_count} validations processed sequentially → {len(atomic_problems)} total atomic problems created")

    if len(atomic_problems) < validation_group_count:
        # Show which validation orders are missing
        found_orders = set(p.get("validation_order") for p in atomic_problems)
        missing_orders = set(validation_orders) - found_orders
        if missing_orders:
            print(f"  ⚠️  WARNING: Some validation groups have no atomic problems!")
            print(f"         Missing validation orders: {sorted(missing_orders)}")
            print(f"         Expected: Changes in these groups should have problems too")

    # Show dependencies
    deps_count = sum(1 for p in atomic_problems if p.get("depends_on"))
    if deps_count > 0:
        print(f"  ✓ {deps_count} problem(s) have sequential dependencies (depends_on)")

    return result


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
    chunks = chunk_structured_diff(structured_diff, max_files_per_chunk=15)
    if not chunks:
        raise ValueError(f"Issue {issue.get('id')} ground-truth diff could not be chunked")

    visible_failure_context = _compact_context_for_diff_analysis(issue, benchmark_context)
    validation_sequence = benchmark_context.get("validation_sequence") or []
    chunk_findings: List[Dict[str, Any]] = []

    print(f"  Step 1: Classifying patch changes by repaired validation ({total_files} files in {len(chunks)} chunk(s))...")

    for index, chunk in enumerate(chunks, start=1):
        prompt = f"""# TASK: Match Changed Files to Validations That Were Repaired

You are analyzing a CI fix. The CI had failures, and this diff fixes them.
Your job: For EACH changed file, determine WHICH validation it repairs.

## CI FAILURE CONTEXT (What Actually Broke)

{json.dumps(visible_failure_context, indent=2)}

**These failures tell you WHICH validations broke and what are the issues or failures. Use this to match files!**

## VALIDATION SEQUENCE (All Validation Steps in Order)

{json.dumps(validation_sequence, indent=2)}

## CHANGED FILES IN THIS CHUNK ({index}/{len(chunks)})

{format_structured_for_llm(chunk)}

## MATCHING LOGIC

For EACH file, determine which validation it repairs:

**Step 1: What type of change?**
- Look at before → after examples
- Identify change type: type annotation, import, formatting, dependency, config, etc.

**Step 2: Which validation does this fix?**
- Check CI FAILURE CONTEXT first - which validations failed?
- Match change type to validation command:
  - Type annotation change → type checking validation (mypy, pyright)
  - Import/unused import → linting validation (ruff, pylint)
  - Formatting change → formatting validation (black, prettier, taplo)
  - Dependency change in pyproject.toml → validation that uses that tool
  - Config change → validation that uses that config

**Step 3: Find exact validation_order**
- Locate the validation in the sequence above
- Copy EXACT validation_order number and validation_cmd

**IMPORTANT RULES:**

1. **Don't skip files** - Every file repairs some validation
2. **Use CI failures as guide** - Prioritize matching to validations that failed
3. **Dependency files can affect multiple validations** - Include in all affected groups
4. **Be specific** - Match to the actual validation, not a guess
5. **Check file extension AND content** - .py file with type changes → type validation

## OUTPUT FORMAT

{{
  "chunk_index": {index},
  "validations_in_this_chunk": [
    {{
      "validation_order": <NUMBER from validation sequence>,
      "validation_cmd": "<EXACT command from validation sequence>",
      "failure_type": "<Type Checking | Formatting | Linting | etc>",
      "files": ["file1.py", "file2.py"],
      "total_files": 2,
      "has_pattern": true
    }}
  ]
}}

**Critical:**
- validation_order must be exact number from sequence
- validation_cmd must exactly match sequence
- Include ALL files from chunk in some validation group
- Don't return empty unless chunk truly has no matchable files (rare)

{STRICT_JSON_RULES}
"""
        try:
            finding = _invoke_json(llm, prompt)
            
            # Normalize LLM response to dict with validations_in_this_chunk
            if isinstance(finding, list):
                finding = {"chunk_index": index, "validations_in_this_chunk": finding}
            elif not isinstance(finding, dict):
                print(f"    Chunk {index}/{len(chunks)} WARNING: LLM returned {type(finding).__name__}, treating as empty")
                finding = {"chunk_index": index, "validations_in_this_chunk": []}

            # Ensure required fields exist
            finding.setdefault("chunk_index", index)
            finding.setdefault("validations_in_this_chunk", [])

            # Validate and log results
            validations = finding.get("validations_in_this_chunk", [])

            # Filter out validations with null/missing order (LLM didn't follow instructions)
            valid_validations = []
            invalid_count = 0
            for v in validations:
                order = v.get("validation_order")
                if order is not None and order != "null":
                    valid_validations.append(v)
                else:
                    invalid_count += 1

            # Update finding with only valid validations
            finding["validations_in_this_chunk"] = valid_validations

            if not valid_validations:
                if invalid_count > 0:
                    print(f"    Chunk {index}/{len(chunks)} WARNING: {invalid_count} validation(s) had null order (rejected)")
                else:
                    print(f"    Chunk {index}/{len(chunks)} WARNING: No validations found (LLM returned empty result)")
            else:
                val_orders = [v.get("validation_order") for v in valid_validations]
                if invalid_count > 0:
                    print(f"    Chunk {index}/{len(chunks)}: {len(valid_validations)} validation(s), orders={val_orders} ({invalid_count} rejected for null order)")
                else:
                    print(f"    Chunk {index}/{len(chunks)}: {len(valid_validations)} validation(s), orders={val_orders}")

        except Exception as exc:
            print(f"    Chunk {index}/{len(chunks)} FAILED with exception: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            finding = {
                "chunk_index": index,
                "validations_in_this_chunk": [],
                "error": str(exc),
            }

        chunk_findings.append(finding)

    # Step 2: Merge by validation (deterministic)
    print(f"  Step 2: Merging chunks by validation...")
    validation_groups = merge_chunks_by_validation(chunk_findings, validation_sequence, chunks)
    print(f"    Found {validation_groups['total_groups']} groups from {validation_groups['total_validations']} validations")

    # Step 2.5: CI-Diff Correlation (deterministic)
    print(f"  Step 2.5: Analyzing CI-Diff correlation (layered structure)...")
    ci_context = _compact_context_for_diff_analysis(issue, benchmark_context)
    correlation_context = build_correlation_context(ci_context, validation_groups)

    layers = correlation_context['total_layers']
    print(f"    Layer 1 (Primary CI failures): {layers['layer_1_primary']}")
    print(f"    Layer 2 (Direct fixes): {layers['layer_2_direct_fixes']}")
    print(f"    Layer 2 (Enablement fixes): {layers['layer_2_enablements']}")
    print(f"    Layer 2 (Missing fixes): {layers['layer_2_missing']}")
    print(f"    Layer 3 (Enablement cascades): {layers['layer_3_cascades']}")

    # Step 3: Deep reasoning with full context + correlation
    print(f"  Step 3: Deep reasoning with correlation context...")
    reasoning_result = analyze_validation_groups_with_reasoning(
        validation_groups,
        validation_sequence,
        ci_context,
        correlation_context,
        llm
    )
    # Log results
    atomic_problems = reasoning_result.get("atomic_problems", [])
    if atomic_problems:
        print(f"  ✓ Identified {len(atomic_problems)} atomic problems")
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

        print(f"  ✓ Identified {len(atomic_problems)} atomic problems")

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

            # Sequential workflow metadata
            "sequential_workflow_metadata": sequential_metadata,

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
        print(f"ERROR Eval issues not found: {eval_path}")
        return 1

    with open(eval_path) as f:
        issues = json.load(f)

    print(f"Loaded {len(issues)} issues from {eval_path}")

    # Filter if specific issue requested
    if args.issue_id:
        issues = [i for i in issues if str(i.get("id")) == args.issue_id]
        if not issues:
            print(f"ERROR Issue {args.issue_id} not found")
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

    # Prepare output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if file exists (for resume capability)
    results = []
    errors = []
    processed_ids = set()

    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    results = existing
                    # Track already processed issues
                    for r in results:
                        if "original_issue_id" in r:
                            processed_ids.add(str(r["original_issue_id"]))
                    print(f"Loaded {len(results)} existing results (will skip already processed issues)")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    # Decompose issues with incremental saving
    for i, issue in enumerate(issues, 1):
        issue_id = str(issue.get("id"))

        # Skip if already processed
        if issue_id in processed_ids:
            print(f"\nProgress: {i}/{len(issues)} - Issue {issue_id} already processed, skipping")
            continue

        print(f"\nProgress: {i}/{len(issues)}")
        result = decompose_issue(issue, llm)

        if "error" in result:
            errors.append(result)

        results.append(result)

        # Incremental save after each issue
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ Saved progress ({len(results)} issues total)")
        except Exception as e:
            print(f"  WARNING: Could not save progress: {e}")

    # Final save (redundant but ensures completion)
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
        print(f"\nWARNING:  {len(errors)} issues had errors")
        print(f"Issue IDs with errors: {[e.get('original_issue_id') for e in errors[:5]]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
