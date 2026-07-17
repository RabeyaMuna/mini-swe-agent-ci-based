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
from collections import defaultdict
from pathlib import Path
from typing import Any

from deterministic_diff_parser import (
    chunk_structured_diff,
    format_structured_for_llm,
    parse_diff_to_structured,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import litellm  # noqa: E402
from datasets import load_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from minisweagent.run.benchmarks.utils.ci_context import (  # noqa: E402
    _log_analysis_to_context,
    _run_log_analysis,
)

from scripts.ci_workflow_aware_retrieval import (  # noqa: E402
    analyze_workflow_from_benchmark,
)
from scripts.model_registry import (  # noqa: E402
    DEFAULT_MINIMAX_MODEL,
    configure_model_environment,
    resolve_model_alias,
)

try:
    import demjson3  # type: ignore
except Exception:
    demjson3 = None  # type: ignore


# Load memory issue IDs from workflow_validation_cache.json
def _load_memory_issue_ids() -> list[str]:
    """Load issue IDs from workflow_validation_cache.json."""
    cache_path = PROJECT_ROOT / 'data' / 'workflow_validation_cache.json'
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
                issue_ids = [str(item['id']) for item in cache if 'id' in item]
                return issue_ids
        except Exception as e:
            print(f'Warning: Could not load workflow_validation_cache.json: {e}')
            return ['121']  # Fallback to default
    return ['121']  # Fallback to default


# Load memory issue IDs dynamically
MEMORY_ISSUE_IDS = _load_memory_issue_ids()

DIFF_CHUNK_CHAR_LIMIT = int(os.getenv('DIFF_CHUNK_CHAR_LIMIT', '30000'))
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

load_dotenv(PROJECT_ROOT / '.env', override=False)

if not os.getenv('OPENROUTER_API_KEY') and os.getenv('MINIMAX_API_KEY'):
    os.environ['OPENROUTER_API_KEY'] = os.getenv('MINIMAX_API_KEY', '')
if not os.getenv('OPENROUTER_BASE_URL') and os.getenv('MINIMAX_BASE_URL'):
    os.environ['OPENROUTER_BASE_URL'] = os.getenv('MINIMAX_BASE_URL', '')


class LitellmModel:
    """Small invoke-compatible wrapper for decomposition scripts."""

    def __init__(self, model_name: str):
        self.model_name = self._normalize_model_name(model_name)

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        resolved = resolve_model_alias(model_name)
        if resolved:
            return resolved
        if (
            model_name.startswith('minimax/')
            and os.getenv('OPENROUTER_API_KEY')
            and (
                'openrouter.ai' in os.getenv('OPENROUTER_BASE_URL', '')
                or 'openrouter.ai' in os.getenv('MINIMAX_BASE_URL', '')
            )
        ):
            return f'openrouter/{model_name}'
        return model_name

    def invoke(self, prompt: Any, max_tokens: int | None = None):
        if isinstance(prompt, list):
            messages = [
                {
                    'role': 'user',
                    'content': str(getattr(message, 'content', message)),
                }
                for message in prompt
            ]
        else:
            messages = [{'role': 'user', 'content': str(prompt)}]

        try:
            import time

            start_time = time.time()

            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens
                or 16000,  # MiniMax M2.5 supports large outputs; cap per task.
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
                print(
                    f'      [API] finish_reason={finish_reason}, tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}'
                )
            else:
                print(f'      [API] finish_reason={finish_reason}, no usage data')

            if finish_reason == 'error':
                error_msg = getattr(response, 'error', 'Unknown error')
                if hasattr(response, '_hidden_params'):
                    error_msg += f' | Details: {response._hidden_params}'
                LOGGER.error(f'LLM error after {elapsed:.1f}s. Error: {error_msg}')
                print(f'    FAIL LLM Error ({elapsed:.1f}s): {error_msg}')
            elif finish_reason == 'length':
                # Log detailed info for length errors
                LOGGER.warning(
                    f'Hit length limit: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}'
                )
                print('    WARNING Length limit hit!')
                print(
                    f'       Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}'
                )
                print(f'       max_tokens setting: {max_tokens or 16000}')
                print(
                    '       Chunk too large - reduce max_changes_per_chunk or simplify prompt'
                )

            class Result:
                content = response.choices[0].message.content or ''
                raw_response = response

            return Result()
        except Exception as e:
            LOGGER.error(f'LiteLLM API call failed: {type(e).__name__}: {e}')
            print(f'    FAIL API Error: {type(e).__name__}: {str(e)[:200]}')

            class Result:
                content = ''
                error = str(e)
                raw_response = None

            return Result()

    def __call__(self, prompt: Any):
        """Make LitellmModel callable for compatibility with CILogAnalyzer."""
        result = self.invoke(prompt)
        return result.content


def _extract_json_from_text(content: str) -> str:
    """Extract JSON from LLM response that may include markdown fences or explanatory text."""
    content = str(content or '').strip()

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
                    return content[start : i + 1]

    # If we didn't find a match, return original
    return content


def _clean_malformed_json(content: str) -> str:
    """Clean common LLM JSON formatting mistakes before parsing."""
    content = str(content or '').strip()
    content = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', content, flags=re.DOTALL)
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    content = re.sub(r',\s*,', ',', content)
    content = re.sub(r'}\s*{', '}, {', content)
    content = re.sub(r'}\s*\[', '}, [', content)
    content = re.sub(r']\s*{', '], {', content)
    return content.strip()


def _load_llm_json(content: str) -> Any:
    """Parse raw LLM JSON. Handles markdown fences and explanatory text."""
    content = str(content or '').strip()

    if not content:
        return []

    candidates = [
        content,
        _extract_json_from_text(content),
        _clean_malformed_json(content),
        _clean_malformed_json(_extract_json_from_text(content)),
    ]

    last_json_err: Any = None
    last_demjson3_err: Any = 'demjson3 is not installed'

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
        f'JSON parse failed: json={last_json_err}; demjson3={last_demjson3_err}\n'
        f'Content preview (first 500 chars):\n{preview}'
    )
    LOGGER.warning('%s', parse_error)
    return []


def _invoke_json(llm: Any, prompt: str, max_tokens: int | None = None) -> Any:
    """Invoke LLM and parse JSON with robust error handling."""
    # Check prompt size upfront
    prompt_size_kb = len(prompt) / 1024
    if prompt_size_kb > 80:
        print(
            f'        WARNING  Large prompt: {prompt_size_kb:.1f}KB - may cause API errors'
        )

    try:
        try:
            response = (
                llm.invoke(prompt, max_tokens=max_tokens)
                if max_tokens
                else llm.invoke(prompt)
            )
        except TypeError:
            response = llm.invoke(prompt)
        content = str(getattr(response, 'content', response) or '').strip()
    except Exception as exc:
        error_msg = str(exc)

        # Provide specific guidance based on error type
        if 'Unable to get json response' in error_msg or 'Expecting value' in error_msg:
            print('        FAIL API returned malformed/truncated JSON')
            print(f'          Prompt size: {prompt_size_kb:.1f}KB')
            print('          -> Chunk too large, reduce max_files_per_chunk')
        elif 'rate limit' in error_msg.lower():
            print('        FAIL Rate limit - increase delay (currently 3s)')
        else:
            print(f'        FAIL API Error: {type(exc).__name__}: {str(exc)[:200]}')

        LOGGER.error(f'LLM API call failed: {type(exc).__name__}: {exc}')
        return []  # Return empty, continue processing other chunks

    if not content:
        # Get detailed error info if available
        error_details = ''
        finish_reason = None
        if hasattr(response, 'error'):
            error_details = f'Error: {response.error}'
        elif hasattr(response, 'raw_response') and response.raw_response:
            raw = response.raw_response
            if hasattr(raw, 'error'):
                error_details = f'Error: {raw.error}'
            if hasattr(raw.choices[0], 'finish_reason'):
                finish_reason = raw.choices[0].finish_reason
                error_details += f' | Finish reason: {finish_reason}'

        print('        FAIL LLM returned empty content')
        print(
            f'          Prompt size: {len(prompt)} chars ({len(prompt) / 1024:.1f}KB)'
        )
        if error_details:
            print(f'          {error_details}')

        # Special handling for length limit: return signal to re-split
        if finish_reason == 'length':
            print("          -> Returning 'SPLIT_REQUIRED' signal for auto-retry")
            LOGGER.warning('Length limit hit, chunk needs splitting')
            return 'SPLIT_REQUIRED'  # Signal to caller to split chunk

        LOGGER.warning(
            f'LLM empty content. Prompt size: {len(prompt)} chars. {error_details}'
        )
        return []

    parsed = _load_llm_json(content)
    if parsed not in (None, [], {}):
        return parsed

    # One repair pass helps when the model produced almost-valid JSON or was
    # wrapped/truncated. Keep this prompt small.
    LOGGER.warning(
        f'Initial JSON parse failed, attempting repair. Content preview: {content[:200]}'
    )
    repair_prompt = f"""{STRICT_JSON_RULES}

Repair the following model output into valid JSON only.
Preserve all recoverable keys and values.
If the output is truncated, close the current JSON structure conservatively and omit incomplete trailing items.

--- MODEL OUTPUT TO REPAIR ---
{content[:24000]}
"""
    try:
        repaired_response = llm.invoke(repair_prompt)
        repaired_content = str(
            getattr(repaired_response, 'content', repaired_response) or ''
        ).strip()
        repaired = _load_llm_json(repaired_content)
        if repaired not in (None, [], {}):
            LOGGER.info('Recovered malformed JSON with repair prompt')
            return repaired
    except Exception as exc:
        LOGGER.warning('JSON repair prompt failed: %s', exc)

    LOGGER.warning(
        f'All JSON parsing attempts failed. Returning empty. Original content length: {len(content)}'
    )
    return parsed


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2)


def _extract_diff_file_sections(diff: str) -> list[dict[str, Any]]:
    """Return complete `diff --git` file sections. Never splits a file diff."""
    diff = str(diff or '')
    if not diff.strip():
        return []

    matches = list(re.finditer(r'(?m)^diff --git ', diff))
    if not matches:
        return [{'files': [], 'diff': diff, 'char_count': len(diff)}]

    sections: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(diff)
        section = diff[start:end].rstrip()
        first_line = section.split('\n')[0]
        file_match = re.search(r'a/(.*?) b/', first_line)
        files = [file_match.group(1)] if file_match else []
        sections.append({'files': files, 'diff': section, 'char_count': len(section)})

    return sections


def chunk_diff_by_file(
    diff: str, max_chars: int = DIFF_CHUNK_CHAR_LIMIT
) -> list[dict[str, Any]]:
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

    chunks: list[dict[str, Any]] = []
    current_sections: list[dict[str, Any]] = []
    current_chars = 0

    def flush_current() -> None:
        nonlocal current_sections, current_chars
        if not current_sections:
            return
        chunks.append(
            {
                'files': [
                    file_path
                    for section in current_sections
                    for file_path in section.get('files', [])
                ],
                'diff': '\n\n'.join(
                    section.get('diff', '') for section in current_sections
                ),
                'char_count': current_chars,
                'file_count': len(current_sections),
                'oversized_single_file': False,
            }
        )
        current_sections = []
        current_chars = 0

    for section in sections:
        section_chars = int(section.get('char_count') or len(section.get('diff', '')))

        if section_chars > max_chars:
            flush_current()
            chunks.append(
                {
                    'files': section.get('files', []),
                    'diff': section.get('diff', ''),
                    'char_count': section_chars,
                    'file_count': 1,
                    'oversized_single_file': True,
                }
            )
            continue

        if current_sections and current_chars + section_chars > max_chars:
            flush_current()

        current_sections.append(section)
        current_chars += section_chars

    flush_current()

    return chunks


def _repo_checkout_path(issue: dict[str, Any]) -> str | None:
    """Return local checkout path for dependent workflow/config files if present."""
    explicit = issue.get('repo_path') or issue.get('checkout_path')
    if explicit and Path(str(explicit)).exists():
        return str(explicit)

    repo = str(issue.get('repo') or '').strip()
    repo_name = str(issue.get('repo_name') or '').strip()
    repo_owner = str(issue.get('repo_owner') or '').strip()
    candidates: list[Path] = []
    if '/' in repo:
        candidates.append(PROJECT_ROOT / 'repo' / repo.replace('/', '__'))
    if repo_owner and repo_name:
        candidates.append(PROJECT_ROOT / 'repo' / f'{repo_owner}__{repo_name}')
    if repo_name:
        candidates.append(PROJECT_ROOT / 'repo' / repo_name)

    for path in candidates:
        if path.exists():
            return str(path)
    return None


def build_benchmark_ci_context(issue: dict, llm: Any) -> dict[str, Any]:
    """
    Build structured CI context for decomposition.

    Uses:
      - CILogAnalyzer/precomputed benchmark CI context for failure analysis
      - raw benchmark workflow YAML (`issue["workflow"]`)
      - LLM-extracted ordered validation sequence from that workflow

    This function does not pass raw CI logs into decomposition prompts.
    """
    raw_workflow = str(issue.get('workflow') or '')
    if not raw_workflow.strip():
        raise ValueError(
            f'Issue {issue.get("id")} has no workflow YAML in benchmark data'
        )

    workflow_path = str(
        issue.get('workflow_path') or issue.get('workflow_filename') or ''
    )
    repo_path = _repo_checkout_path(issue)
    model_name = str(
        getattr(llm, 'model_name', '') or os.getenv('MEMCI_LLM_MODEL') or ''
    )
    issue_id = str(issue.get('id') or issue.get('instance_id') or '')

    # Check cache first to avoid re-running CILogAnalyzer
    sha_fail = issue.get('sha_fail', '')
    cache_file = PROJECT_ROOT / 'data' / 'log_details.json'
    cached_analysis = None

    if cache_file.exists() and sha_fail:
        try:
            with open(cache_file) as f:
                cache = json.load(f)
            cached_analysis = next(
                (entry for entry in cache if entry.get('sha_fail') == sha_fail), None
            )
            if cached_analysis:
                print(f'  [1/2] Loading cached CI log analysis for {sha_fail[:12]}...')
        except Exception as e:
            print(f'  WARNING:  Cache load failed: {e}, will re-analyze')

    if cached_analysis:
        log_analysis = cached_analysis
    else:
        print('  [1/2] Building structured CI failure context with CILogAnalyzer...')
        log_analysis = _run_log_analysis(issue, llm=llm, model=model_name)

    context = _log_analysis_to_context(log_analysis, issue, workflow_profile={})

    # Check workflow validation cache
    validation_cache_file = PROJECT_ROOT / 'data' / 'workflow_validation_cache.json'
    cached_validation = None

    if validation_cache_file.exists() and sha_fail:
        try:
            with open(validation_cache_file) as f:
                validation_cache = json.load(f)
            cached_validation = next(
                (
                    entry
                    for entry in validation_cache
                    if entry.get('sha_fail') == sha_fail
                ),
                None,
            )
            if cached_validation:
                print(
                    f'  [2/2] Loading cached workflow validation sequence for {sha_fail[:12]}...'
                )
        except Exception as e:
            print(f'  WARNING:  Validation cache load failed: {e}, will re-analyze')

    if cached_validation:
        validation_sequence = cached_validation.get('validation_sequence', [])
        workflow_validation_context = {
            'id': str(cached_validation.get('id') or issue_id),
            'sha_fail': str(cached_validation.get('sha_fail') or sha_fail),
            'workflow_path': str(
                cached_validation.get('workflow_path') or workflow_path
            ),
            'dependent_files': cached_validation.get('dependent_files', []),
            'validation_sequence': validation_sequence,
        }
        print(f'       Found {len(validation_sequence)} validation steps (cached)')
    else:
        print('  [2/2] Analyzing workflow to extract validation sequence...')
        try:
            workflow_validation_context = analyze_workflow_from_benchmark(
                workflow_content=raw_workflow,
                workflow_path=workflow_path,
                repo_path=repo_path,
                llm=llm,
                issue_id=issue_id,
                sha_fail=str(sha_fail or ''),
            )

            validation_sequence = workflow_validation_context.get(
                'validation_sequence', []
            )

            print(f'       Found {len(validation_sequence)} validation steps')
        except Exception as e:
            print(f'       WARNING: Workflow extraction failed: {e}')
            print('       Using fallback: empty validation sequence')
            workflow_validation_context = {
                'workflow_path': str(workflow_path),
                'dependent_files': [],
                'validation_sequence': [],
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
                    if entry.get('sha_fail') == sha_fail:
                        entry.clear()
                        entry.update(workflow_validation_context)
                        updated = True
                        break

                if not updated:
                    existing_cache.append(workflow_validation_context)

                validation_cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(validation_cache_file, 'w') as f:
                    json.dump(existing_cache, f, indent=2)
                print('Saved workflow validation to cache')
            except Exception as e:
                print(f' Failed to save validation cache: {e}')

    # Allow empty validation_sequence as fallback (decomposition will work in simpler mode)
    if not validation_sequence:
        print(
            '       WARNING: No validation sequence available, using fallback decomposition mode'
        )

    return {
        'context': context,
        'log_analysis': log_analysis,
        'validation_sequence': validation_sequence,
        'workflow_validation_context': workflow_validation_context,
        'workflow_path': workflow_path,
        'workflow_name': str(issue.get('workflow_name') or ''),
        'repo_path': repo_path,
    }


def _has_structured_ci_context(benchmark_context: dict[str, Any]) -> bool:
    """Return True when CILogAnalyzer produced usable structured failure context."""
    context = benchmark_context.get('context') or {}
    log_analysis = benchmark_context.get('log_analysis') or {}
    return bool(
        context.get('overall_failure_reasons')
        or context.get('overall_error_types')
        or context.get('effected_files')
        or context.get('failed_jobs')
        or log_analysis.get('error_context')
        or log_analysis.get('relevant_files')
        or log_analysis.get('error_types')
        or log_analysis.get('failed_job')
        or log_analysis.get('analysis_document')
        or log_analysis.get('overall_ci_summary')
        or log_analysis.get('chunk_summaries')
    )


def validate_required_ci_inputs(benchmark_context: dict[str, Any]) -> bool:
    """Both CI analyzer context and workflow validation sequence are required."""
    has_ci_context = _has_structured_ci_context(benchmark_context)
    has_validation_sequence = bool(benchmark_context.get('validation_sequence'))
    if not has_ci_context:
        print(
            '  ERROR Missing structured CI context from CILogAnalyzer; skipping decomposition'
        )
    if not has_validation_sequence:
        print('  ERROR Missing CI workflow validation sequence; skipping decomposition')
    return has_ci_context and has_validation_sequence


def _compact_context_for_diff_analysis(
    issue: dict,
    benchmark_context: dict[str, Any],
) -> dict[str, Any]:
    context = benchmark_context.get('context') or {}
    return {
        'issue_id': issue.get('id'),
        'repo': issue.get('repo_name', issue.get('repo')),
        'workflow_path': benchmark_context.get('workflow_path'),
        'overall_failure_reasons': context.get('overall_failure_reasons', []),
        'overall_error_types': context.get('overall_error_types', []),
        'failed_jobs': context.get('failed_jobs', []),
    }


def _batch_findings_by_size(
    findings: list[dict[str, Any]], max_chars: int
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
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


# ============================================================================
# CI-DIFF CORRELATION: Layered Analysis
# ============================================================================


def _is_dependency_file(file_path: str) -> bool:
    """Check if file is a dependency configuration file."""
    dep_files = [
        'pyproject.toml',
        'package.json',
        'requirements.txt',
        'Cargo.toml',
        'pom.xml',
        'build.gradle',
        'setup.py',
    ]
    return any(file_path.endswith(f) for f in dep_files)


def extract_primary_ci_failures(ci_context: dict[str, Any]) -> list[dict[str, Any]]:
    """
    LAYER 1: Extract PRIMARY CI failures that actually happened.

    These are the failures that broke CI run.
    Other problems are hidden until these are fixed.
    """
    primary_failures = []
    failure_id = 1

    # From overall_error_types (structured CI analysis)
    for error_type in ci_context.get('overall_error_types', []):
        primary_failures.append(
            {
                'failure_id': failure_id,
                'validation_order': None,  # Will be inferred from validation sequence
                'error_type': error_type,
                'error_category': error_type,
                'error_message': '',
                'severity': 'primary',
            }
        )
        failure_id += 1

    # From overall_failure_reasons (detailed descriptions)
    for reason in ci_context.get('overall_failure_reasons', []):
        # Try to extract validation info from reason
        validation_info = {
            'failure_id': failure_id,
            'validation_order': None,
            'error_type': 'Unknown',
            'error_message': reason[:500],  # First 500 chars
            'severity': 'primary',
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

    print(f'    Extracted {len(primary_failures)} primary CI failures')
    print(
        f'      From overall_error_types: {len(ci_context.get("overall_error_types", []))}'
    )
    print(
        f'      From overall_failure_reasons: {len(ci_context.get("overall_failure_reasons", []))}'
    )

    return primary_failures


def match_failures_with_diff_fixes(
    primary_failures: list[dict[str, Any]],
    validation_groups: dict[str, Any],
) -> list[dict[str, Any]]:
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

    print(f'    Analyzing {len(groups_data)} validation groups with changes...')

    # For EACH validation group, classify the fix type
    for val_order_str, val_group in groups_data.items():
        all_changes = val_group.get('all_changes', [])

        if not all_changes:
            continue

        # Classify changes
        dep_changes = [c for c in all_changes if _is_dependency_file(c.get('file', ''))]
        code_changes = [
            c for c in all_changes if not _is_dependency_file(c.get('file', ''))
        ]

        # Determine fix type based on what's present
        if dep_changes and code_changes:
            # Has BOTH - this is an enablement that reveals secondary
            fix_type = 'ENABLEMENT_FIX'  # Mark as enablement, cascade detection will handle secondary
        elif dep_changes and not code_changes:
            # ONLY dependency changes - pure enablement
            fix_type = 'ENABLEMENT_FIX'
        elif code_changes and not dep_changes:
            # ONLY code changes - direct fix
            fix_type = 'DIRECT_FIX'
        else:
            # No changes (shouldn't happen)
            continue

        matched.append(
            {
                'failure_id': len(matched) + 1,
                'fix_type': fix_type,
                'validation_order': int(val_order_str),
                'validation_cmd': val_group.get('validation_cmd', ''),
                'diff_changes': all_changes,
                'dependency_changes': dep_changes,
                'code_changes': code_changes,
                'error_type': val_group.get('failure_type', 'Unknown'),
                'error_message': f'Validation {val_order_str}: {val_group.get("validation_cmd", "")}',
            }
        )

    return matched


def detect_enablement_cascades(
    matched_failures: list[dict[str, Any]],
    validation_groups: dict[str, Any],
) -> list[dict[str, Any]]:
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
        code_changes = [
            c for c in all_changes if not _is_dependency_file(c.get('file', ''))
        ]

        if code_changes:
            # This enablement REVEALS secondary failures
            cascade = {
                'primary_failure_id': failure['failure_id'],
                'validation_order': val_order,
                'validation_cmd': val_group.get('validation_cmd', ''),
                'enablement_type': 'dependency_installation',
                'reveals_secondary_failures': True,
                'secondary_changes': code_changes,
                'secondary_file_count': len(set(c.get('file') for c in code_changes)),
                'cascade_explanation': f'Enabling validation reveals {len(set(c.get("file") for c in code_changes))} files with violations',
            }
            cascades.append(cascade)

    return cascades


def merge_chunks_by_validation(
    chunk_findings: list[dict[str, Any]],
    validation_sequence: list[dict[str, Any]],
    structured_chunks: list[dict[str, Any]] = None,
) -> dict[str, Any]:
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
    sequence_by_order = {
        str(item.get('order')): item
        for item in validation_sequence
        if item.get('order') is not None
    }

    for chunk in chunk_findings:
        for val_entry in chunk.get('validations_in_this_chunk', []):
            val_order = val_entry.get('validation_order')
            val_cmd = val_entry.get('validation_cmd')
            if (not val_cmd or val_cmd == 'unknown') and val_order not in (
                None,
                'unknown',
            ):
                seq_item = sequence_by_order.get(str(val_order), {})
                val_cmd = str(
                    seq_item.get('validation_cmd')
                    or seq_item.get('installation_cmd')
                    or seq_item.get('validates')
                    or ''
                ).strip()
                val_entry['validation_cmd'] = val_cmd
            if val_order in (None, 'unknown') or not val_cmd or val_cmd == 'unknown':
                continue

            # Check if this validation has multiple failure types
            if val_entry.get('has_multiple_failure_types'):
                # Handle sub-problems
                for sub_problem in val_entry.get('sub_problems', []):
                    # Unique key: validation_order + sub_problem_id
                    sub_id = sub_problem.get('sub_problem_id', '')
                    key = f'{val_order}_{sub_id}' if sub_id else str(val_order)

                    if key not in validation_groups:
                        validation_groups[key] = {
                            'validation_order': val_order,
                            'validation_cmd': val_cmd,
                            'has_sub_problems': True,
                            'sub_problem_id': sub_id,
                            'failure_type': sub_problem.get('failure_type'),
                            'issue_type': sub_problem.get('issue_type')
                            or sub_problem.get('failure_name'),
                            'error_code': sub_problem.get('error_code'),
                            'chunks': [],
                            'all_files': [],
                            'all_changes': [],
                            'has_pattern': False,
                            'total_files': 0,
                        }

                    validation_groups[key]['chunks'].append(chunk['chunk_index'])
                    validation_groups[key]['all_files'].extend(
                        sub_problem.get('files', [])
                    )
                    # Don't add changes here - they'll be populated from structured_chunks below
                    validation_groups[key]['total_files'] += sub_problem.get(
                        'total_files', len(sub_problem.get('files', []))
                    )

                    # Cross-chunk pattern detection
                    if sub_problem.get('has_pattern'):
                        validation_groups[key]['has_pattern'] = True
            else:
                # Single failure type - keep distinct issue subtypes separate.
                failure_type = str(val_entry.get('failure_type') or '').strip()
                issue_type = str(
                    val_entry.get('issue_type') or val_entry.get('failure_name') or ''
                ).strip()
                key_parts = [str(val_order)]
                if failure_type:
                    key_parts.append(failure_type.lower().replace(' ', '_'))
                if issue_type:
                    key_parts.append(issue_type.lower().replace(' ', '_'))
                key = '::'.join(key_parts)

                if key not in validation_groups:
                    validation_groups[key] = {
                        'validation_order': val_order,
                        'validation_cmd': val_cmd,
                        'failure_type': failure_type,
                        'issue_type': issue_type,
                        'has_sub_problems': False,
                        'chunks': [],
                        'all_files': [],
                        'all_changes': [],
                        'has_pattern': False,
                        'total_files': 0,
                    }

                validation_groups[key]['chunks'].append(chunk['chunk_index'])
                validation_groups[key]['all_files'].extend(val_entry.get('files', []))
                # Don't add changes here - they'll be populated from structured_chunks below
                validation_groups[key]['total_files'] += val_entry.get(
                    'total_files', len(val_entry.get('files', []))
                )

                # Cross-chunk pattern detection
                if val_entry.get('has_pattern'):
                    validation_groups[key]['has_pattern'] = True

    # CRITICAL FIX: Attach actual file changes to validation groups
    # Build a lookup of file -> changes from the original structured chunks
    file_changes_lookup = {}
    all_changed_files = set()
    if structured_chunks:
        for chunk in structured_chunks:
            if 'files' in chunk:
                for file_info in chunk['files']:
                    file_path = file_info.get('path', '')
                    if file_path:
                        all_changed_files.add(file_path)
                    if file_path and 'changes' in file_info:
                        if file_path not in file_changes_lookup:
                            file_changes_lookup[file_path] = []
                        file_changes_lookup[file_path].extend(file_info['changes'])

    # Now attach changes to each validation group
    for val_group in validation_groups.values():
        for file_path in val_group['all_files']:
            if file_path in file_changes_lookup:
                # Add all changes for this file
                file_changes = file_changes_lookup[file_path]
                for change in file_changes:
                    # Add file context to each change
                    change_with_file = change.copy()
                    change_with_file['file'] = file_path
                    val_group['all_changes'].append(change_with_file)

    # Sort by validation order for sequential processing. LLMs should return
    # numeric orders, but keep this deterministic if a string slips through.
    def _validation_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        order = item[1].get('validation_order')
        try:
            return (int(order), str(order))
        except (TypeError, ValueError):
            return (10**9, str(order))

    sorted_groups = dict(sorted(validation_groups.items(), key=_validation_sort_key))

    return {
        'validation_groups': sorted_groups,
        'total_validations': len(
            set(g['validation_order'] for g in sorted_groups.values())
        ),
        'total_groups': len(
            sorted_groups
        ),  # May be > total_validations if sub-problems exist
        'all_changed_files_from_diff': sorted(all_changed_files),
        'all_config_files_from_diff': sorted(
            file_path
            for file_path in all_changed_files
            if _is_dependency_file(file_path)
            or file_path.endswith(
                ('.toml', '.yaml', '.yml', '.json', '.ini', '.cfg', '.lock')
            )
        ),
    }


def _chunk_validation_changes(
    val_group: dict[str, Any], max_changes_per_chunk: int = 400
) -> list[dict[str, Any]]:
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
        chunk_changes = all_changes[start_idx : start_idx + max_changes_per_chunk]

        chunk = {
            'validation_cmd': val_group.get('validation_cmd', ''),
            'failure_type': val_group.get('failure_type', ''),
            'issue_type': val_group.get('issue_type', ''),
            'all_files': val_group.get('all_files', []),  # Keep full file list
            'all_changes': chunk_changes,
            'chunk_info': f'Changes {start_idx + 1}-{start_idx + len(chunk_changes)} of {len(all_changes)} total',
        }
        chunks.append(chunk)

    return chunks


ATOMIC_ANALYSIS_MAX_PROMPT_CHARS = 48000
ATOMIC_ANALYSIS_MAX_OUTPUT_TOKENS = (
    16000  # MiniMax M2.5 supports up to 64K, 16K is safe
)
VALIDATION_MERGE_MAX_PROMPT_CHARS = 32000
VALIDATION_MERGE_MAX_OUTPUT_TOKENS = 8000  # Increased from 6000


def _compact_text(value: Any, limit: int = 1200) -> str:
    """Compact text fields before placing them in prompts."""
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(' ', 1)[0] + '...'


def _compact_json(value: Any, limit: int = 6000) -> str:
    """Serialize JSON context with a hard character budget."""
    text = json.dumps(value, indent=2, default=str)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit('\n', 1)[0] + '\n  ...\n}'


def _final_verify_config_files(
    validation_groups: dict[str, Any], all_atomic_problems: list[dict[str, Any]]
) -> None:
    """
    Final verification that ALL config files from ground truth are included.

    This runs after all problems are created to catch any config files that
    might have been filtered out during processing.
    """
    config_file_patterns = [
        'pyproject.toml',
        'package.json',
        'requirements.txt',
        'setup.py',
        'Cargo.toml',
        'pom.xml',
        'build.gradle',
        'Gemfile',
        'go.mod',
    ]

    # Prefer original parsed-diff metadata. Falling back to validation_groups
    # alone misses files that were dropped during LLM classification.
    all_config_files = set(validation_groups.get('all_config_files_from_diff') or [])
    groups_data = validation_groups.get('validation_groups', {})

    if not all_config_files:
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
        print('\n CRITICAL ERROR: Config files missing from decomposition!')
        print(f'     Config files in ground truth: {all_config_files}')
        print(f'     Missing from problems: {missing_config_files}')
        print('\n     These files MUST be included in problems (primary or hidden).')
        print(
            '     This is a violation of: NEVER remove config file changes from ground truth'
        )
    else:
        print(
            f'  OK Config file verification: All {len(all_config_files)} config files included in problems'
        )
        # If this happens frequently, we need to strengthen the prompt


def _normalize_chunk_finding(
    finding: Any,
    *,
    chunk_index: int,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """Normalize common LLM schema/key/path drift before validation."""
    if isinstance(finding, list):
        finding = {'chunk_index': chunk_index, 'validations_in_this_chunk': finding}
    elif not isinstance(finding, dict):
        print(
            f'    Chunk {chunk_index} WARNING: LLM returned {type(finding).__name__}, '
            'treating as empty'
        )
        return {'chunk_index': chunk_index, 'validations_in_this_chunk': []}

    if (
        'validations_in_this_chunk' not in finding
        and 'validations_this_chunk' in finding
    ):
        finding['validations_in_this_chunk'] = finding.pop('validations_this_chunk')

    finding.setdefault('chunk_index', chunk_index)
    validations = finding.get('validations_in_this_chunk')
    if not isinstance(validations, list):
        finding['validations_in_this_chunk'] = []
        return finding

    actual_files = [
        str(file_info.get('path') or '')
        for file_info in chunk.get('files', [])
        if file_info.get('path')
    ]
    actual_set = set(actual_files)

    for validation in validations:
        if not isinstance(validation, dict):
            continue
        files = validation.get('files')
        if not isinstance(files, list):
            validation['files'] = []
            validation['total_files'] = 0
            continue

        normalized_files: list[str] = []
        for file_path in files:
            file_path = str(file_path or '').strip()
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
                    f'    Chunk {chunk_index} WARNING: dropping hallucinated file path: {file_path}'
                )

        validation['files'] = list(dict.fromkeys(normalized_files))
        validation['total_files'] = len(validation['files'])

    return finding


def _semantic_cluster_problems(
    problems: list[dict[str, Any]], threshold: float = 0.5
) -> list[list[dict[str, Any]]]:
    """
    Cluster problems by semantic similarity using embeddings.

    Groups problems with cosine similarity > threshold.
    Uses sentence-transformers for fast, quality embeddings.

    Args:
        problems: List of problems to cluster
        threshold: Cosine similarity threshold (0.85 = very similar, 0.75 = moderately similar)

    Returns:
        List of clusters, each cluster is a list of similar problems
    """
    if not problems:
        return []

    if len(problems) == 1:
        return [problems]

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        # Generate text representation for each problem
        problem_texts = []
        for prob in problems:
            # Combine key fields for similarity comparison
            # Note: validation_cmd should be same for all (already grouped by validation)
            text = f"""
Validation: {prob.get('validation_cmd', '')}
Problem: {prob.get('what_broke', '')}
Root Cause: {prob.get('root_cause', '')}
Fix: {prob.get('how_fixed', '')}
Failure Type: {prob.get('failure_type', '')}
Issue Type: {prob.get('issue_type', '')}
""".strip()
            problem_texts.append(text)

        # Compute embeddings (use lightweight model for speed)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(problem_texts, show_progress_bar=False)

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(embeddings)

        # Cluster using similarity threshold
        clusters = []
        visited = set()

        for i in range(len(problems)):
            if i in visited:
                continue

            # Start new cluster with this problem
            cluster = [problems[i]]
            visited.add(i)

            # Find all problems similar to this one
            for j in range(i + 1, len(problems)):
                if j in visited:
                    continue

                if similarity_matrix[i][j] >= threshold:
                    cluster.append(problems[j])
                    visited.add(j)

            clusters.append(cluster)

        return clusters

    except ImportError:
        # Fallback: no clustering if dependencies not available
        print('    [WARN] sentence-transformers not available, skipping clustering')
        return [[prob] for prob in problems]


def _llm_merge_decision(
    cluster: list[dict[str, Any]], validation_cmd: str, llm: Any
) -> dict[str, Any]:
    """
    LLM analyzes cluster and decides: MERGE, PARTIAL_MERGE, or SEPARATE.

    Decision based on:
    - Are problems describing the SAME underlying issue?
    - Are root causes IDENTICAL?
    - Are fixes following the SAME pattern?

    Returns:
        {
            "action": "merge" | "partial_merge" | "separate",
            "result": List of problems (merged or separate),
            "reasoning": "Why this decision was made"
        }
    """

    if len(cluster) == 1:
        # Single problem, no merge needed
        return {
            'action': 'keep_as_is',
            'result': cluster,
            'reasoning': 'Single problem in cluster',
        }

    # Build prompt for LLM
    problems_text = []
    for idx, prob in enumerate(cluster, 1):
        files_str = ', '.join(prob.get('affected_files', [])[:5])
        if len(prob.get('affected_files', [])) > 5:
            files_str += f' ... ({len(prob.get("affected_files", []))} total)'

        problems_text.append(
            f"""
Problem {idx}:
  Affected files: {files_str}
  What broke: {prob.get('what_broke', 'N/A')}
  Root cause: {prob.get('root_cause', 'N/A')}
  How fixed: {prob.get('how_fixed', 'N/A')}
  Failure type: {prob.get('failure_type', 'N/A')}
  Issue type: {prob.get('issue_type', 'N/A')}
""".strip()
        )

    prompt = f"""You are analyzing a cluster of SIMILAR problems to decide if they should be MERGED.

VALIDATION: {validation_cmd}
CLUSTER SIZE: {len(cluster)} problems

PROBLEMS:
{chr(10).join(problems_text)}

---

YOUR TASK: Decide ONE of these actions:

1. **MERGE** - All problems are essentially the SAME
   - Use when: Problems describe the same underlying issue in different files
   - Root causes are IDENTICAL (not just similar)
   - Fixes follow the EXACT SAME pattern
   - Example: "All files missing type hints for function parameters"

2. **PARTIAL_MERGE** - Some problems are same, others are distinct
   - Use when: Cluster has sub-groups of identical problems
   - Example: Problems 1,2,3 are same (merge), Problems 4,5 are different (separate)

3. **SEPARATE** - All problems are DISTINCT
   - Use when: Problems have different root causes or require different fixes
   - Even if similar, they're fundamentally different issues
   - Example: "Missing type hint" vs "Incorrect type hint"

MERGE CRITERIA (ALL must be true to merge):
OK Root causes express the SAME underlying issue
OK Fixes use the SAME pattern/approach
OK Files can be treated as "same problem in multiple places"
OK No interdependencies within cluster

SEPARATE if ANY true:
FAIL Root causes are DIFFERENT
FAIL Fixes require DIFFERENT approaches
FAIL Problems have different complexity levels

OUTPUT VALID JSON ONLY:

For MERGE:
{{
  "action": "merge",
  "reasoning": "Why all problems are the same",
  "merged_problem": {{
    "root_cause": "Unified description of root cause",
    "how_fixed": "Unified description of fix pattern",
    "why_fixed_works": "Why this fix solves all instances"
  }}
}}

For PARTIAL_MERGE:
{{
  "action": "partial_merge",
  "reasoning": "Why some merge, others separate",
  "sub_groups": [
    {{
      "problem_indices": [1, 2, 3],
      "action": "merge",
      "merged_problem": {{
        "root_cause": "...",
        "how_fixed": "...",
        "why_fixed_works": "..."
      }}
    }},
    {{
      "problem_indices": [4],
      "action": "separate",
      "reason": "Different root cause"
    }}
  ]
}}

For SEPARATE:
{{
  "action": "separate",
  "reasoning": "Why problems should stay separate"
}}

IMPORTANT: Be CONSERVATIVE. When in doubt, SEPARATE.
"""

    try:
        decision = _invoke_json(llm, prompt)

        # Process decision
        if decision.get('action') == 'merge':
            # Create merged problem
            all_files = []
            for prob in cluster:
                all_files.extend(prob.get('affected_files', []))

            merged_prob = decision.get('merged_problem', {})
            merged = {
                'affected_files': list(dict.fromkeys(all_files)),  # Deduplicate
                'root_cause': merged_prob.get(
                    'root_cause', cluster[0].get('root_cause')
                ),
                'how_fixed': merged_prob.get('how_fixed', cluster[0].get('how_fixed')),
                'why_fixed_works': merged_prob.get(
                    'why_fixed_works', cluster[0].get('why_fixed_works')
                ),
                'what_broke': merged_prob.get(
                    'root_cause', cluster[0].get('what_broke')
                ),
                'failure_type': cluster[0].get('failure_type'),
                'issue_type': cluster[0].get('issue_type'),
                'validation_cmd': cluster[0].get('validation_cmd'),
                'validation_order': cluster[0].get('validation_order'),
                'problem_type': cluster[0].get('problem_type'),
                'is_merged': True,
                'merged_from': [p.get('problem_id', i) for i, p in enumerate(cluster)],
                'merge_count': len(cluster),
            }

            return {
                'action': 'merge',
                'result': [merged],
                'reasoning': decision.get('reasoning', ''),
            }

        elif decision.get('action') == 'partial_merge':
            # Process sub-groups
            result = []
            for sub_group in decision.get('sub_groups', []):
                indices = sub_group.get('problem_indices', [])
                # Convert 1-based indices to 0-based
                sub_problems = [
                    cluster[i - 1] for i in indices if 0 < i <= len(cluster)
                ]

                if sub_group.get('action') == 'merge' and len(sub_problems) > 1:
                    # Merge this sub-group
                    all_files = []
                    for prob in sub_problems:
                        all_files.extend(prob.get('affected_files', []))

                    merged_prob = sub_group.get('merged_problem', {})
                    merged = {
                        'affected_files': list(dict.fromkeys(all_files)),
                        'root_cause': merged_prob.get(
                            'root_cause', sub_problems[0].get('root_cause')
                        ),
                        'how_fixed': merged_prob.get(
                            'how_fixed', sub_problems[0].get('how_fixed')
                        ),
                        'why_fixed_works': merged_prob.get(
                            'why_fixed_works', sub_problems[0].get('why_fixed_works')
                        ),
                        'what_broke': merged_prob.get(
                            'root_cause', sub_problems[0].get('what_broke')
                        ),
                        'failure_type': sub_problems[0].get('failure_type'),
                        'issue_type': sub_problems[0].get('issue_type'),
                        'validation_cmd': sub_problems[0].get('validation_cmd'),
                        'validation_order': sub_problems[0].get('validation_order'),
                        'problem_type': sub_problems[0].get('problem_type'),
                        'is_merged': True,
                        'merged_from': [
                            p.get('problem_id', i) for i, p in enumerate(sub_problems)
                        ],
                        'merge_count': len(sub_problems),
                    }
                    result.append(merged)
                else:
                    # Keep separate
                    result.extend(sub_problems)

            return {
                'action': 'partial_merge',
                'result': result,
                'reasoning': decision.get('reasoning', ''),
            }

        else:  # separate
            return {
                'action': 'separate',
                'result': cluster,
                'reasoning': decision.get('reasoning', ''),
            }

    except Exception as e:
        print(f'    [WARN] LLM merge decision failed: {e}, keeping problems separate')
        return {
            'action': 'separate',
            'result': cluster,
            'reasoning': f'Error: {str(e)}',
        }


def _cluster_and_merge_problems(
    problems: list[dict[str, Any]],
    validation_cmd: str,
    llm: Any,
    similarity_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """
    Cluster similar problems and let LLM decide whether to merge.

    Flow:
    1. Semantic clustering (cosine similarity on embeddings)
    2. LLM analyzes each cluster for merge decision
    3. Returns optimized problem list

    Args:
        problems: List of problems from one validation
        validation_cmd: The validation command for context
        llm: LLM instance
        similarity_threshold: Cosine similarity threshold for clustering

    Returns:
        Optimized list of problems (some merged, some separate)
    """

    if not problems:
        return problems

    if len(problems) == 1:
        return problems

    print(
        f'      Clustering {len(problems)} problems (threshold={similarity_threshold})...'
    )

    # Step 1: Semantic clustering
    clusters = _semantic_cluster_problems(problems, threshold=similarity_threshold)

    multi_problem_clusters = [c for c in clusters if len(c) > 1]
    single_problem_clusters = [c for c in clusters if len(c) == 1]

    print(
        f'        → {len(clusters)} clusters ({len(multi_problem_clusters)} multi-problem, {len(single_problem_clusters)} single)'
    )

    # Step 2: LLM merge decisions for multi-problem clusters
    optimized = []
    merge_stats = {'merged': 0, 'partial': 0, 'separate': 0}

    for cluster in clusters:
        if len(cluster) == 1:
            # Single problem, keep as-is
            optimized.extend(cluster)
            continue

        # Multi-problem cluster → LLM decision
        print(f'        Analyzing cluster of {len(cluster)} problems...')
        decision = _llm_merge_decision(cluster, validation_cmd, llm)

        if decision['action'] == 'merge':
            merge_stats['merged'] += len(cluster)
            print(f'          OK MERGED {len(cluster)} problems → 1')
        elif decision['action'] == 'partial_merge':
            merge_stats['partial'] += 1
            print(
                f'          ⊕ PARTIAL MERGE: {len(cluster)} → {len(decision["result"])}'
            )
        else:
            merge_stats['separate'] += len(cluster)
            print(f'          → KEPT SEPARATE: {decision.get("reasoning", "")[:80]}')

        optimized.extend(decision['result'])
        time.sleep(0.5)  # Rate limiting

    print(f'      Optimization: {len(problems)} → {len(optimized)} problems')
    if merge_stats['merged'] > 0:
        print(f'        Merged: {merge_stats["merged"]} problems')
    if merge_stats['partial'] > 0:
        print(f'        Partial merges: {merge_stats["partial"]}')
    if merge_stats['separate'] > 0:
        print(f'        Kept separate: {merge_stats["separate"]}')

    return optimized


def _reorder_by_repair_trajectory(
    problems: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Reorder problems by repair trajectory (LLM already assigned problem_type).

    Order:
    1. PRIMARY problems (files visible in CI logs) - sorted by validation_order
    2. HIDDEN problems (files not in CI logs) - sorted by validation_order
    """
    if not problems:
        return problems

    # Separate by problem_type (already assigned by LLM)
    primary_problems = []
    hidden_problems = []

    for prob in problems:
        ptype = prob.get('problem_type', 'primary')
        if ptype == 'primary':
            primary_problems.append(prob)
        else:
            hidden_problems.append(prob)

    # Sort each group by validation_order
    def get_validation_order(p):
        try:
            return int(p.get('validation_order', 999))
        except (TypeError, ValueError):
            return 999

    primary_problems.sort(key=get_validation_order)
    hidden_problems.sort(key=get_validation_order)

    # Reassign problem_ids in new order
    reordered = primary_problems + hidden_problems
    for idx, prob in enumerate(reordered, 1):
        prob['problem_id'] = idx
        prob['repair_sequence_index'] = idx

    print(f'    Primary problems: {len(primary_problems)} (files in CI logs)')
    print(f'    Hidden problems: {len(hidden_problems)} (files not in CI logs)')
    print(f'    Total reordered: {len(reordered)}')

    return reordered


def analyze_validation_groups_with_reasoning(
    validation_groups: dict[str, Any],
    validation_sequence: list[dict[str, Any]],
    ci_context: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    """
    Analyze validation groups into atomic problems.

    Flow:
    1. Grouped input is already scoped by validation_cmd + failure_type.
    2. Analyze each validation group, chunking only when needed.
    3. Merge chunk-level results back within the same validation so variants of
       one failure family become one atomic problem.
    """
    groups_data = validation_groups.get('validation_groups', {})
    print(f'  Processing {len(groups_data)} validation groups...')

    # Find the FIRST validation that failed (minimum validation_order with changes)
    failed_validation_order = None
    for val_order_str in groups_data.keys():
        try:
            val_order = int(val_order_str)
            if failed_validation_order is None or val_order < failed_validation_order:
                failed_validation_order = val_order
        except (ValueError, TypeError):
            pass

    print(f'  First failed validation: {failed_validation_order}')

    all_problems = []
    next_id = 1

    # Process each validation group
    def _group_order(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        order = item[1].get('validation_order')
        try:
            return (int(order), str(item[0]))
        except (TypeError, ValueError):
            return (10**9, str(item[0]))

    def _compact_changes(
        changes: list[dict[str, Any]], max_per_field: int = 600
    ) -> list[dict[str, Any]]:
        """Compress changes to fit more in one chunk. Reduce per-field size for large batches."""
        compact = []
        for change in changes:
            compact.append(
                {
                    'file': change.get('file', ''),
                    'before': _compact_text(change.get('before', ''), max_per_field),
                    'after': _compact_text(change.get('after', ''), max_per_field),
                }
            )
        return compact

    def _changes_data(chunk: dict[str, Any]) -> dict[str, Any]:
        changes = chunk.get('all_changes', [])
        # Dynamic compression: smaller per-field limit for larger chunks
        if len(changes) > 100:
            max_per_field = 400  # Aggressive compression for large chunks
        elif len(changes) > 60:
            max_per_field = 500
        else:
            max_per_field = 600  # Standard compression

        return {
            'validation_cmd': chunk.get('validation_cmd', ''),
            'failure_type': chunk.get('failure_type', ''),
            'issue_type': chunk.get('issue_type', ''),
            'files_count': len(
                {change.get('file', '') for change in changes if change.get('file')}
            ),
            'changes_count': len(changes),
            'changes': _compact_changes(changes, max_per_field),
        }

    def _build_atomic_prompt(
        *,
        validation_order: Any,
        val_info: dict[str, Any],
        chunk: dict[str, Any],
        changes_data: dict[str, Any],
    ) -> str:
        change_type = chunk.get('change_type', 'unknown')
        change_type_context = {
            'config': 'These are CONFIGURATION file changes (.toml, .yaml, .json, .ini). Pay special attention to tool settings, plugin configurations, and dependency specifications.',
            'dependency': 'These are DEPENDENCY-related changes (imports, packages, requirements). Focus on package installations, version updates, and import fixes.',
            'code': 'These are SOURCE CODE changes (.py, .rst, .md). Focus on code logic, formatting, type annotations, and documentation fixes.',
        }.get(change_type, '')

        return f"""Analyze this validation group and create atomic CI repair problems.

CI FAILURE CONTEXT:
{_compact_json(ci_context, 6000)}

VALIDATION CONTEXT:
- validation_order: {validation_order}
- validation_cmd: {val_info.get('validation_cmd', chunk.get('validation_cmd', ''))}
- validates: {val_info.get('validates', 'Code quality/formatting')}
- failure_type: {chunk.get('failure_type', '')}
- issue_type_hint: {chunk.get('issue_type', '')}
- change_type: {change_type.upper()}
- FAILED_VALIDATION_ORDER: {failed_validation_order} (CI stopped here)

{change_type_context}

CHANGES:
{json.dumps(changes_data, indent=2)}

TASK:
Use the before/after changes to infer the actual validation problems fixed by this group.
IMPORTANT: This group contains ONLY {change_type.upper()} changes. Do NOT lose any configuration details!

RULES:
1. Create separate atomic problems for EACH distinct fix, even if they're in the same validation.
2. Different files with different fixes = separate problems (e.g., ruff error in file A + config update in file B = 2 problems).
3. Same validation error across multiple files with THE SAME fix pattern = can merge into one problem.
4. Each problem must have a clear, specific root cause and fix that applies to its affected_files.
5. Be specific about package names, symbols, config keys, validators, before/after states, and affected file kinds.
6. Do not mention line numbers.
7. Do not use vague phrasing like "fixed issues" or "updated files".
8. affected_files must include ONLY the files directly involved in this specific problem's fix.

**CRITICAL - IGNORE GIT WORKFLOW ISSUES:**
9. DO NOT create problems for Git merge conflict markers (<<<<<<<, =======, >>>>>>>)
   - Merge conflicts are GIT workflow issues, NOT CI validation problems
   - If a file has merge conflicts RESOLVED in the diff, identify the REAL validation problem that was fixed
   - Example: If diff removes merge markers AND fixes RST formatting → Problem is "RST formatting", NOT "merge conflict"
   - Focus on the ACTUAL validation issue (type errors, format issues, imports, etc.), not the Git noise

EXAMPLES OF CORRECT GROUPING:
- SEPARATE problems (different fixes):
   - Problem 1: Unused import removed in file_a.py (affected_files: ["file_a.py"])
   - Problem 2: Config version updated in pyproject.toml (affected_files: ["pyproject.toml"])

- MERGED into ONE problem (same fix pattern):
   - Problem 1: Unused imports removed in 5 files (affected_files: ["a.py", "b.py", "c.py", "d.py", "e.py"])

- WRONG - DO NOT merge unrelated fixes:
   - Problem 1: Ruff errors + config update (affected_files: ["file.py", "pyproject.toml"]) ← WRONG!

FIELD GUIDANCE:
- problem: 1-2 sentences describing the failure category and important variants.
- root_cause: 1-2 sentences explaining why the previous state failed validation.
- how_fixed: 1-2 sentences saying what changed from old state to new state and why.
- why_fix_works: 1-2 sentences explaining how the new state satisfies the validator.
- problem_type: Check if affected_files appear in CI FAILURE CONTEXT above:
  * "primary" = Files ARE mentioned/visible in the CI failure logs (check relevant_files, error messages, stack traces)
  * "hidden" = Files are NOT mentioned in the CI failure logs

OUTPUT FORMAT (CRITICAL - Must match this EXACT structure):
{{
  "atomic_problems": [
    {{
      "problem_id": 1,
      "validation_order": {validation_order},
      "validation_cmd": "{val_info.get('validation_cmd', chunk.get('validation_cmd', ''))}",
      "failure_type": "{chunk.get('failure_type', '')}",
      "issue_type": "specific_error_code_or_type",
      "problem": "Brief description of what broke",
      "root_cause": "Why it failed",
      "how_fixed": "What changed",
      "why_fix_works": "Why the fix solves it",
      "affected_files": ["file1.py", "file2.py"],
      "problem_type": "primary"
    }},
   // Additional problems if any, each with incremented problem_id
  ]
}}

CRITICAL RULES:
1. Each problem MUST have "problem_id" as INTEGER (not nested objects!)
2. problem_id starts at 1 and increments
3. All keys must be strings in double quotes
4. affected_files is an ARRAY of file paths
5. problem_type MUST be either "primary" or "hidden":
   - "primary" = affected_files ARE visible in CI failure context (check relevant_files, error messages, traces)
   - "hidden" = affected_files are NOT visible in CI failure context

{STRICT_JSON_RULES}
"""

    def _extract_problems(result: Any) -> list[dict[str, Any]]:
        if isinstance(result, list):
            # Validate each problem has correct schema
            valid_problems = []
            for p in result:
                if isinstance(p, dict) and 'problem_id' in p:
                    valid_problems.append(p)
                elif isinstance(p, dict):
                    # Malformed: has nested problem_1, problem_2, etc.
                    print(
                        f'      WARNING: Skipping malformed problem (missing problem_id): {list(p.keys())[:3]}'
                    )
            return valid_problems
        if isinstance(result, dict):
            if 'problem_id' in result and 'atomic_problems' not in result:
                return [result]
            problems = result.get('atomic_problems', [])
            if isinstance(problems, list):
                # Validate each problem
                valid_problems = []
                for p in problems:
                    if isinstance(p, dict) and 'problem_id' in p:
                        valid_problems.append(p)
                    elif isinstance(p, dict):
                        # Malformed: has nested problem_1, problem_2, etc.
                        print(
                            f'      WARNING: Skipping malformed problem (missing problem_id): {list(p.keys())[:3]}'
                        )
                return valid_problems
            return []
        return []

    def _analyze_chunk(
        *,
        chunk: dict[str, Any],
        chunk_label: str,
        validation_order: Any,
        val_info: dict[str, Any],
        depth: int = 0,
    ) -> list[dict[str, Any]]:
        changes = chunk.get('all_changes', [])
        changes_data = _changes_data(chunk)
        prompt = _build_atomic_prompt(
            validation_order=validation_order,
            val_info=val_info,
            chunk=chunk,
            changes_data=changes_data,
        )

        print(f'      Calling LLM for {chunk_label}')
        print(f'        Prompt size: {len(prompt)} chars, {len(changes)} changes')

        # Proactive split checks:
        # 1. Prompt too large (>48KB)
        # 2. Too many changes (each change needs ~50 tokens output, limit is 16K tokens)
        #    16,000 tokens / 50 tokens per change = 320 max changes
        #    Use 200 as safe limit to leave headroom
        if (
            len(prompt) > ATOMIC_ANALYSIS_MAX_PROMPT_CHARS or len(changes) > 200
        ) and len(changes) > 1:
            if len(prompt) > ATOMIC_ANALYSIS_MAX_PROMPT_CHARS:
                print(
                    f'        Proactive split: prompt too large ({len(prompt)} chars)'
                )
            else:
                print(
                    f'        Proactive split: too many changes ({len(changes)} changes, max 200)'
                )
            result = 'SPLIT_REQUIRED'
        else:
            result = _invoke_json(
                llm, prompt, max_tokens=ATOMIC_ANALYSIS_MAX_OUTPUT_TOKENS
            )

        if result == 'SPLIT_REQUIRED' and len(changes) > 1:
            # Safety check: don't split if already very small
            if len(changes) <= 10:
                print(
                    f'      WARNING: Chunk too small to split further ({len(changes)} changes)'
                )
                print('        Returning empty to avoid infinite recursion')
                return []

            # Check depth limit (allow up to depth 5)
            MAX_RECURSION_DEPTH = 5
            if depth >= MAX_RECURSION_DEPTH:
                print(
                    f'      WARNING: Max recursion depth ({MAX_RECURSION_DEPTH}) reached'
                )
                print(
                    f'        Chunk {chunk_label} with {len(changes)} changes cannot be processed'
                )
                return []

            indent = '  ' * depth

            # SMART SPLITTING: Calculate optimal number of splits based on size
            # Target: each sub-chunk should be < 15KB prompt size
            TARGET_PROMPT_SIZE = 15000  # 15KB - safe size for most APIs
            current_prompt_size = len(prompt)

            # Estimate how many splits we need
            estimated_splits = max(2, (current_prompt_size // TARGET_PROMPT_SIZE) + 1)

            # Cap at reasonable limit (don't create too many tiny chunks)
            max_splits = min(estimated_splits, len(changes) // 10, 8)  # Max 8 splits

            # If only 2 splits needed, use traditional approach
            if max_splits <= 2:
                num_splits = 2
            else:
                num_splits = max_splits

            print(
                f'      {indent}Token limit for {chunk_label}; smart splitting analysis:'
            )
            print(
                f'      {indent}  Current size: {current_prompt_size} chars, {len(changes)} changes'
            )
            print(f'      {indent}  Target size per chunk: {TARGET_PROMPT_SIZE} chars')
            print(f'      {indent}  Estimated splits needed: {estimated_splits}')
            print(f'      {indent}  → Splitting into {num_splits} sub-chunks')

            # Calculate chunk sizes (distribute evenly)
            chunk_size = len(changes) // num_splits
            remainder = len(changes) % num_splits

            # Create sub-chunks
            sub_chunks = []
            start_idx = 0
            for i in range(num_splits):
                # Add 1 extra change to first 'remainder' chunks to distribute evenly
                size = chunk_size + (1 if i < remainder else 0)
                end_idx = start_idx + size

                sub_chunk = {**chunk, 'all_changes': changes[start_idx:end_idx]}
                sub_chunks.append(sub_chunk)

                print(
                    f'      {indent}  Sub-chunk {i + 1}: {size} changes (indices {start_idx}-{end_idx - 1})'
                )
                start_idx = end_idx

            # Process all sub-chunks
            split_problems = []
            for sub_idx, sub_chunk in enumerate(sub_chunks, 1):
                print(
                    f'      {indent}→ Processing sub-chunk {sub_idx}/{num_splits} (depth {depth + 1})...'
                )

                try:
                    sub_problems = _analyze_chunk(
                        chunk=sub_chunk,
                        chunk_label=f'{chunk_label}.{sub_idx}',
                        validation_order=validation_order,
                        val_info=val_info,
                        depth=depth + 1,
                    )

                    if not sub_problems:
                        print(
                            f'      {indent}  [WARN] Sub-chunk {sub_idx} returned 0 problems'
                        )
                    else:
                        print(
                            f'      {indent}  OK Sub-chunk {sub_idx} returned {len(sub_problems)} problems'
                        )

                    split_problems.extend(sub_problems)

                except Exception as e:
                    print(
                        f'      {indent}  FAIL ERROR in sub-chunk {sub_idx}: {type(e).__name__}: {str(e)[:200]}'
                    )
                    # Continue with other sub-chunks even if one fails

                # Exponential backoff: deeper splits wait longer
                sleep_time = min(2**depth, 8)  # 1s, 2s, 4s, 8s (max)
                if sub_idx < len(sub_chunks):  # Don't sleep after last sub-chunk
                    print(
                        f'      {indent}  Waiting {sleep_time}s before next sub-chunk...'
                    )
                    time.sleep(sleep_time)

            print(
                f'      {indent}Split complete: {len(split_problems)} total problems from {num_splits} sub-chunks'
            )
            return split_problems

        problems = _extract_problems(result)
        if changes and not problems:
            print(
                f'      WARNING {chunk_label} has {len(changes)} changes but returned 0 problems'
            )
            debug_file = (
                PROJECT_ROOT
                / 'data'
                / 'trs'
                / f'debug_prompt_val{validation_order}_{chunk_label.replace(".", "_")}.txt'
            )
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            debug_file.write_text(prompt, encoding='utf-8')
            print(f'        Saved prompt to: {debug_file}')

            # RETRY STRATEGY: If we have multiple changes and not at max depth, try splitting
            MAX_RECURSION_DEPTH = 5
            if len(changes) > 1 and depth < MAX_RECURSION_DEPTH:
                # Safety check
                if len(changes) <= 5:
                    print(
                        f'      WARNING: Chunk too small to split further ({len(changes)} changes)'
                    )
                    return []

                indent = '  ' * depth
                print(
                    f'      {indent}RETRY: Malformed response, applying smart split strategy'
                )

                # SMART SPLITTING for retry (same logic as token limit)
                TARGET_PROMPT_SIZE = 15000  # 15KB
                current_prompt_size = len(prompt)

                # Calculate optimal splits
                estimated_splits = max(
                    2, (current_prompt_size // TARGET_PROMPT_SIZE) + 1
                )
                max_splits = min(estimated_splits, len(changes) // 10, 8)

                if max_splits <= 2:
                    num_splits = 2
                else:
                    num_splits = max_splits

                print(
                    f'      {indent}  Current size: {current_prompt_size} chars, {len(changes)} changes'
                )
                print(
                    f'      {indent}  Estimated splits: {estimated_splits} → Using {num_splits} splits'
                )

                # Distribute changes evenly
                chunk_size = len(changes) // num_splits
                remainder = len(changes) % num_splits

                sub_chunks = []
                start_idx = 0
                for i in range(num_splits):
                    size = chunk_size + (1 if i < remainder else 0)
                    end_idx = start_idx + size
                    sub_chunk = {**chunk, 'all_changes': changes[start_idx:end_idx]}
                    sub_chunks.append(sub_chunk)
                    print(
                        f'      {indent}  Retry sub-chunk {i + 1}: {size} changes (indices {start_idx}-{end_idx - 1})'
                    )
                    start_idx = end_idx

                # Process sub-chunks
                split_problems = []
                for sub_idx, sub_chunk in enumerate(sub_chunks, 1):
                    print(f'      {indent}→ Retry sub-chunk {sub_idx}/{num_splits}...')

                    try:
                        sub_problems = _analyze_chunk(
                            chunk=sub_chunk,
                            chunk_label=f'{chunk_label}.retry{sub_idx}',
                            validation_order=validation_order,
                            val_info=val_info,
                            depth=depth + 1,
                        )

                        if not sub_problems:
                            print(
                                f'      {indent}  [WARN] Retry {sub_idx} returned 0 problems'
                            )
                        else:
                            print(
                                f'      {indent}  OK Retry {sub_idx} returned {len(sub_problems)} problems'
                            )

                        split_problems.extend(sub_problems)

                    except Exception as e:
                        print(
                            f'      {indent}  FAIL ERROR in retry {sub_idx}: {type(e).__name__}: {str(e)[:200]}'
                        )

                    # Exponential backoff
                    sleep_time = min(2**depth, 8)
                    if sub_idx < len(sub_chunks):
                        print(f'      {indent}  Waiting {sleep_time}s...')
                        time.sleep(sleep_time)

                if split_problems:
                    print(
                        f'      {indent}RETRY SUCCESS: Got {len(split_problems)} problems from {num_splits} splits'
                    )
                    return split_problems
                else:
                    # If recursive retry failed, return empty - don't create low-quality generic problems
                    print(
                        f'      {indent}RETRY FAILED: Split returned 0 problems after depth {depth} retries'
                    )
                    print(
                        f'      {indent}WARNING: Skipping this chunk - retry mechanism exhausted'
                    )
                    return []

        return problems

    def _normalize_problem_defaults(
        problem: dict[str, Any],
        *,
        validation_order: Any,
        val_group: dict[str, Any],
    ) -> dict[str, Any]:
        # Do NOT set problem_type here - it's assigned in _reorder_by_repair_trajectory
        problem['validation_order'] = (
            problem.get('validation_order') or validation_order
        )
        problem['validation_cmd'] = problem.get('validation_cmd') or val_group.get(
            'validation_cmd', ''
        )
        problem['failure_type'] = problem.get('failure_type') or val_group.get(
            'failure_type', ''
        )
        problem['issue_type'] = problem.get('issue_type') or val_group.get(
            'issue_type', ''
        )
        if not isinstance(problem.get('affected_files'), list):
            problem['affected_files'] = []
        problem['affected_files'] = list(
            dict.fromkeys(
                str(file_path)
                for file_path in problem.get('affected_files', [])
                if file_path
            )
        )
        for key in ['problem', 'root_cause', 'how_fixed', 'why_fix_works']:
            problem[key] = str(problem.get(key, '') or '').strip()
        return problem

    def _merge_validation_problems(
        problems: list[dict[str, Any]],
        *,
        validation_order: Any,
        val_group: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if len(problems) <= 1:
            return problems

        summaries = []
        for idx, problem in enumerate(problems, 1):
            summaries.append(
                {
                    'local_id': idx,
                    'problem_type': problem.get('problem_type', ''),
                    'validation_cmd': problem.get(
                        'validation_cmd', val_group.get('validation_cmd', '')
                    ),
                    'failure_type': problem.get(
                        'failure_type', val_group.get('failure_type', '')
                    ),
                    'issue_type': problem.get('issue_type', ''),
                    'problem': _compact_text(problem.get('problem', ''), 600),
                    'root_cause': _compact_text(problem.get('root_cause', ''), 600),
                    'how_fixed': _compact_text(problem.get('how_fixed', ''), 600),
                    'why_fix_works': _compact_text(
                        problem.get('why_fix_works', ''), 600
                    ),
                    'affected_files': problem.get('affected_files', [])[:12],
                    'affected_files_count': len(problem.get('affected_files', [])),
                }
            )

        prompt = f"""Merge atomic problems for one validation group.

VALIDATION:
- validation_order: {validation_order}
- validation_cmd: {val_group.get('validation_cmd', '')}
- failure_type: {val_group.get('failure_type', '')}

CHUNK-LEVEL PROBLEMS:
{_compact_json(summaries, VALIDATION_MERGE_MAX_PROMPT_CHARS - 8000)}

TASK:
Analyze the problems above and merge those that represent the same underlying issue and same general repair strategy..
Return clean atomic problems that are distinct and actionable.

CONTEXT-AWARE MERGE ANALYSIS:

Step 1: UNDERSTAND what each problem is actually about
- Read the root_cause to understand WHY it failed
- Read the how_fixed to understand WHAT was changed
- Read the why_fix_works to understand the MECHANISM

Step 2: COMPARE problems semantically (not just by keywords)
Ask: "Are these the SAME issue appearing in multiple places, or DIFFERENT issues?"

Consider:
a) ROOT CAUSE Equivalence:
   - Do they fail for the same underlying reason?
   - Would explaining one root cause cover both problems?
   - Are they consequences of the same missing/wrong pattern?

   MERGE if: The "why it failed" is fundamentally the same
   SEPARATE if: They fail for different reasons, even if symptoms look similar

b) REPAIR Strategy Equivalence:
   - Do they get fixed the same way?
   - Would the same change pattern apply to all affected files?
   - Does one fix require different reasoning/approach than the other?

   MERGE if: The "how to fix" follows the same logic/approach
   SEPARATE if: They require different types of changes (even if touching similar files)

c) PATTERN Recognition:
   - Is this an issue that repeats across multiple locations?
   - Would fixing all instances require the same understanding?
   - Are these independent issues that happen to coexist?

   MERGE if: It's the same pattern manifesting in multiple places
   SEPARATE if: They're coincidentally similar but independently occurring

Step 3: DECIDE based on semantic analysis

MERGE when:
→ All three (root cause + repair + pattern) indicate the SAME underlying issue
→ Combining them into one problem makes conceptual sense
→ A developer would think of these as "the same problem in multiple places"

KEEP SEPARATE when:
→ ANY of the three indicates DIFFERENT issues
→ Merging would confuse two distinct problems
→ A developer would need different mental models to understand each

Step 4: QUALITY CHECK your decision

Test: "If I explain problem A to a developer, does that explanation cover problem B?"
- If YES → they're likely the same issue → MERGE
- If NO → they're different issues → KEEP SEPARATE

IMPORTANT PRINCIPLES:

1. Analyze SEMANTICS, not syntax
   - Don't merge just because keywords match
   - Don't separate just because files differ
   - Focus on: "Is this fundamentally the same issue?"

2. Think like a developer
   - Would they group these mentally?
   - Would fixing one give insight to fix the other?
   - Are these the same bug/requirement manifesting differently?

3. Handle ANY failure/fix type
   - These principles work for linting, typing, logic, config, dependencies, etc.
   - No hardcoded patterns - analyze what's actually there
   - Trust the content of root_cause, how_fixed, why_fix_works

4. Preserve information when merging
   - Combine all variants in the merged fields
   - Keep all affected_files from merged problems
   - Don't lose detail - synthesize it

OUTPUT RULES:
1. Return distinct atomic problems (merged or separate based on analysis above)
2. Each problem should be conceptually atomic (one root cause → one fix approach)
3. Keep problem, root_cause, how_fixed, why_fix_works to 1-2 sentences each
4. Combine affected_files from merged problems
5. Be concise but preserve technical reasoning
6. Do not mention chunks, merging process, or meta-commentary

OUTPUT FORMAT:
{{
  "atomic_problems": [
    {{
      "problem_id": 1,
      "problem_type": "primary or hidden",
      "validation_order": {validation_order},
      "validation_cmd": "{val_group.get('validation_cmd', '')}",
      "failure_type": "{val_group.get('failure_type', '')}",
      "issue_type": "",
      "problem": "",
      "root_cause": "",
      "how_fixed": "",
      "why_fix_works": "",
      "affected_files": []
    }}
  ]
}}

{STRICT_JSON_RULES}
"""
        if len(prompt) > VALIDATION_MERGE_MAX_PROMPT_CHARS:
            print(
                '      WARNING Validation merge prompt too large; using chunk-level problems'
            )
            return problems

        result = _invoke_json(
            llm, prompt, max_tokens=VALIDATION_MERGE_MAX_OUTPUT_TOKENS
        )
        merged = _extract_problems(result)
        if not merged:
            print(
                '      WARNING Validation-level merge failed; using chunk-level problems'
            )
            return problems
        if len(merged) == 1 and not merged[0].get('affected_files'):
            all_files = []
            for problem in problems:
                all_files.extend(problem.get('affected_files', []))
            merged[0]['affected_files'] = list(dict.fromkeys(all_files))
        return merged

    for val_order, val_group in sorted(groups_data.items(), key=_group_order):
        print(f'    Validation {val_order}: {val_group.get("validation_cmd", "")}')

        validation_order = val_group.get('validation_order', val_order)
        val_info = next(
            (
                v
                for v in validation_sequence
                if str(v.get('order')) == str(validation_order)
            ),
            {},
        )

        # SMART GROUPING: Group by change_type first (config/dependency/code)
        all_changes = val_group.get('all_changes', [])

        change_type_groups = {'config': [], 'dependency': [], 'code': []}

        for change in all_changes:
            file_path = change.get('file', '')
            before = str(change.get('before', '')).lower()
            after = str(change.get('after', '')).lower()

            # Determine change type
            if file_path.endswith(('.toml', '.yaml', '.yml', '.json', '.ini', '.cfg')):
                change_type_groups['config'].append(change)
            elif (
                'import' in before
                or 'import' in after
                or 'package' in before
                or 'package' in after
            ):
                change_type_groups['dependency'].append(change)
            else:
                change_type_groups['code'].append(change)

        print(
            f'      Grouped: {len(change_type_groups["config"])} config, {len(change_type_groups["dependency"])} dependency, {len(change_type_groups["code"])} code changes'
        )

        validation_problems = []

        # Process each change type group separately
        for change_type in ['config', 'dependency', 'code']:
            group_changes = change_type_groups[change_type]
            if not group_changes:
                continue

            print(
                f'      Processing {change_type.upper()} group ({len(group_changes)} changes)...'
            )

            # Create chunk with this group
            group_chunk = {
                **val_group,
                'all_changes': group_changes,
                'change_type': change_type,
            }

            # Check if group fits in one chunk
            test_changes_data = _changes_data(group_chunk)
            test_prompt = _build_atomic_prompt(
                validation_order=validation_order,
                val_info=val_info,
                chunk=group_chunk,
                changes_data=test_changes_data,
            )

            if len(test_prompt) < ATOMIC_ANALYSIS_MAX_PROMPT_CHARS:
                # Fits in one chunk - analyze as-is
                chunk_problems = _analyze_chunk(
                    chunk=group_chunk,
                    chunk_label=f'{change_type}_group',
                    validation_order=validation_order,
                    val_info=val_info,
                )
                validation_problems.extend(chunk_problems)
                print(
                    f'        {change_type.upper()}: 1 chunk, {len(chunk_problems)} problems'
                )
                time.sleep(1)
            else:
                # Too big - split this group only
                print(f'        {change_type.upper()} group too large, splitting...')
                sub_chunks = _chunk_validation_changes(
                    group_chunk, max_changes_per_chunk=150
                )  # Reduced from 250 to fit better

                for sub_idx, sub_chunk in enumerate(sub_chunks, 1):
                    sub_chunk['change_type'] = change_type
                    chunk_problems = _analyze_chunk(
                        chunk=sub_chunk,
                        chunk_label=f'{change_type}_chunk_{sub_idx}_of_{len(sub_chunks)}',
                        validation_order=validation_order,
                        val_info=val_info,
                    )
                    validation_problems.extend(chunk_problems)
                    print(
                        f'        {change_type.upper()} chunk {sub_idx}/{len(sub_chunks)}: {len(chunk_problems)} problems'
                    )
                    time.sleep(1)

        validation_problems = [
            _normalize_problem_defaults(
                problem, validation_order=validation_order, val_group=val_group
            )
            for problem in validation_problems
        ]
        validation_problems = _merge_validation_problems(
            validation_problems,
            validation_order=validation_order,
            val_group=val_group,
        )

        # CLUSTER AND MERGE: Optimize problems using semantic similarity
        if len(validation_problems) > 1:
            print(
                f'      Optimizing {len(validation_problems)} problems via clustering...'
            )
            validation_problems = _cluster_and_merge_problems(
                validation_problems,
                validation_cmd=val_group.get('validation_cmd', ''),
                llm=llm,
                similarity_threshold=0.5,  # Tune this if needed
            )

        for problem in validation_problems:
            problem = _normalize_problem_defaults(
                problem,
                validation_order=validation_order,
                val_group=val_group,
            )
            problem['problem_id'] = next_id
            next_id += 1
            all_problems.append(problem)

        print(
            f'      Validation {val_order}: {len(validation_problems)} merged problems'
        )

    # Verify config files included
    _final_verify_config_files(validation_groups, all_problems)

    print(f'  OK Total: {len(all_problems)} problems created')

    # Reorder problems by repair trajectory: primary → hidden
    print('  Reordering problems by repair trajectory...')
    all_problems = _reorder_by_repair_trajectory(all_problems)

    return {'atomic_problems': all_problems}


def classify_chunk_with_fallback(
    chunk: dict,
    chunk_index: int,
    total_chunks: int,
    visible_failure_context: dict,
    validation_sequence: list,
    llm: Any,
    max_files: int = 50,
) -> list[dict]:
    """
    Classify a chunk with automatic fallback to smaller chunks if token limit hit.

    Strategy:
    - Try with current chunk size
    - If token limit -> split in half and retry recursively
    - Works down to 1 file if needed
    """

    files_in_chunk = chunk.get('total_files', 0)

    if files_in_chunk == 0:
        return []

    # Extract files visible in CI failure logs
    ci_visible_files = [
        rf.get('file', '') for rf in visible_failure_context.get('relevant_files', [])
    ]
    formatted_validations = []
    for item in validation_sequence:
        effective_cmd = str(
            item.get('validation_cmd')
            or item.get('installation_cmd')
            or item.get('validates')
            or ''
        ).strip()
        formatted_validations.append(
            {
                'order': item.get('order'),
                'validates': item.get('validates', ''),
                'validation_cmd': item.get('validation_cmd', ''),
                'installation_cmd': item.get('installation_cmd', ''),
                'effective_cmd': effective_cmd,
                'source': item.get('source', ''),
                'evidence': item.get('evidence', ''),
            }
        )

    prompt = f"""Classify each changed file by the validation command that would catch the fixed issue.

## INPUT

CI failure context:
{json.dumps(visible_failure_context, indent=2)}

FILES VISIBLE IN CI FAILURE LOGS (primary errors):
{json.dumps(ci_visible_files, indent=2) if ci_visible_files else '[]'}

Available validations:
{json.dumps(formatted_validations, indent=2)}

Changed files, chunk {chunk_index}/{total_chunks} ({files_in_chunk} files):
{format_structured_for_llm(chunk)}

## TASK

For every file:
1. Inspect before/after changes
2. Decide what CI failure the change fixes
3. Choose validation_order from VALIDATIONS
4. Use effective_cmd as the validation_cmd when validation_cmd is empty
5. Group files with same validation_cmd + failure_type + issue_type
6. Determine if file was VISIBLE in CI logs or HIDDEN

Use two levels:
- failure_type: broad category (Type Checking, Linting, Formatting, etc.)
- issue_type: specific failure (missing annotation, unused import, etc.)

Config files must be assigned to the validation they affect:
- Tool/dependency added or configured -> that tool's validation.
- Formatter config changed -> formatter validation.
- If a setup/install/dependency validation failed because a project
  configuration or dependency file was invalid, classify that changed config
  file under the setup/install/dependency validation.
- Distinguish executor files from root-cause files. Workflow files and local
  actions may run commands, while project config/dependency/source files may be
  the actual repair target. Use the CI evidence and the diff together.

VISIBILITY CLASSIFICATION:
- "primary" = file appears in "FILES VISIBLE IN CI FAILURE LOGS" above
- "hidden" = file does NOT appear in CI logs (enablement fix, cascaded fix)
- For setup/install failures, a config/dependency file can be primary when the
  CI evidence says that file's configuration caused the setup/install command
  to fail, even if the visible file list names the workflow action.

## OUTPUT

Return JSON array:
[
  {{
    "validation_order": <INT>,
    "validation_cmd": "<exact>",
    "failure_type": "<category>",
    "issue_type": "<specific>",
    "change_type": "<code|dependency|config>",
    "visibility": "<primary|hidden>",
    "files": [...],
    "total_files": <int>
  }},
  // Additional validation groups if multiple different issues detected
]

REQUIREMENTS:
- Array only (no wrapper)
- validation_order = INTEGER from VALIDATIONS
- validation_cmd = effective_cmd from VALIDATIONS when validation_cmd is empty
- visibility must be either "primary" or "hidden"
- Every file in at least one group

{STRICT_JSON_RULES}
"""

    try:
        # Try classification
        result = _invoke_json(llm, prompt, max_tokens=8000)

        # Expect array directly
        if isinstance(result, list):
            validations = result
        elif isinstance(result, dict):
            # Extract array from wrapper
            for key in ['validations', 'groups', 'result', 'data', 'validation_groups']:
                if key in result and isinstance(result[key], list):
                    validations = result[key]
                    break
            else:
                validations = []
        else:
            validations = []

        sequence_by_order = {
            str(item.get('order')): item
            for item in validation_sequence
            if item.get('order') is not None
        }

        # Fix missing validation_order and normalize install/setup commands.
        for v in validations:
            if not v.get('validation_order'):
                val_cmd = str(v.get('validation_cmd', '')).strip()
                for seq_item in validation_sequence:
                    effective_cmd = str(
                        seq_item.get('validation_cmd')
                        or seq_item.get('installation_cmd')
                        or seq_item.get('validates')
                        or ''
                    ).strip()
                    if (
                        seq_item.get('validation_cmd') == val_cmd
                        or effective_cmd == val_cmd
                    ):
                        v['validation_order'] = seq_item.get('order')
                        break
            seq_item = sequence_by_order.get(str(v.get('validation_order')), {})
            if seq_item and not str(v.get('validation_cmd') or '').strip():
                v['validation_cmd'] = str(
                    seq_item.get('validation_cmd')
                    or seq_item.get('installation_cmd')
                    or seq_item.get('validates')
                    or ''
                ).strip()

        # Keep only valid entries
        valid = [v for v in validations if v.get('validation_order')]

        # Enforce ground-truth diff coverage. The prompt asks the model to
        # include every file, but setup/config files are too important to trust
        # to prompt compliance alone.
        actual_files = [
            str(file_info.get('path') or '')
            for file_info in chunk.get('files', [])
            if file_info.get('path')
        ]
        classified_files = {
            str(file_path)
            for entry in valid
            for file_path in (entry.get('files') or [])
            if file_path
        }
        missing_files = [
            file_path for file_path in actual_files if file_path not in classified_files
        ]
        if missing_files:
            print(
                f'    WARNING Chunk {chunk_index}: classifier missed {len(missing_files)} changed file(s); '
                'adding deterministic fallback groups'
            )
        for file_path in missing_files:
            fallback_validation = validation_sequence[0] if validation_sequence else {}
            if not fallback_validation:
                continue
            effective_cmd = str(
                fallback_validation.get('validation_cmd')
                or fallback_validation.get('installation_cmd')
                or fallback_validation.get('validates')
                or ''
            ).strip()
            is_config = _is_dependency_file(file_path) or file_path.endswith(
                ('.toml', '.yaml', '.yml', '.json', '.ini', '.cfg', '.lock')
            )
            valid.append(
                {
                    'validation_order': fallback_validation.get('order'),
                    'validation_cmd': effective_cmd,
                    'failure_type': 'Configuration/Dependency'
                    if is_config
                    else 'Unclassified Changed File',
                    'issue_type': 'Diff-backed file omitted by classifier',
                    'change_type': 'config' if is_config else 'code',
                    'visibility': 'hidden',
                    'files': [file_path],
                    'total_files': 1,
                    '_fallback_reason': 'LLM omitted changed file from validation classification',
                }
            )

        print(
            f'    OK Chunk {chunk_index} ({files_in_chunk} files): {len(valid)} validation groups'
        )
        return valid

    except Exception as e:
        error_msg = str(e).lower()

        # Check if token limit error
        if any(
            keyword in error_msg
            for keyword in ['token', 'length', 'limit', 'too long', 'maximum']
        ):
            print(
                f'    WARNING Token limit hit with {files_in_chunk} files, splitting in half...'
            )

            # Can't split further
            if files_in_chunk <= 1:
                print('    FAIL Cannot split 1 file further, skipping')
                return []

            # Split chunk in half
            files_list = list(chunk['files'].items())
            half = len(files_list) // 2

            chunk1_files = dict(files_list[:half])
            chunk2_files = dict(files_list[half:])

            chunk1 = {
                'files': chunk1_files,
                'total_files': len(chunk1_files),
                'total_changes': sum(
                    len(f.get('changes', [])) for f in chunk1_files.values()
                ),
            }

            chunk2 = {
                'files': chunk2_files,
                'total_files': len(chunk2_files),
                'total_changes': sum(
                    len(f.get('changes', [])) for f in chunk2_files.values()
                ),
            }

            # Recursively process both halves
            result1 = classify_chunk_with_fallback(
                chunk1,
                chunk_index,
                total_chunks,
                visible_failure_context,
                validation_sequence,
                llm,
                max_files=half,
            )

            result2 = classify_chunk_with_fallback(
                chunk2,
                chunk_index,
                total_chunks,
                visible_failure_context,
                validation_sequence,
                llm,
                max_files=len(files_list) - half,
            )

            return result1 + result2
        else:
            # Other error - log and return empty
            print(f'    FAIL Chunk {chunk_index} failed: {str(e)[:100]}')
            return []


def analyze_diff_chunks(
    issue: dict,
    benchmark_context: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    """
    Three-step diff analysis with deterministic pre-processing:
    0. Parse diff into structured format (deterministic, no LLM)
    1. Chunk and classify by validation (per chunk, LLM with auto-fallback)
    2. Merge by validation (deterministic)
    3. Deep reasoning with full context (LLM)
    """
    diff = str(issue.get('diff') or '')
    if not diff.strip():
        raise ValueError(f'Issue {issue.get("id")} has no ground-truth diff')

    # Step 0: Deterministic diff parsing (NEW!)
    print('  Step 0: Parsing diff into structured format...')

    structured_diff = parse_diff_to_structured(diff)
    total_files = structured_diff['total_files']
    total_changes = structured_diff['total_changes']
    print(f'    Parsed {total_files} files with {total_changes} changes')

    # Chunk by file count (not char count) - cleaner and more predictable
    # Reduced from 15 to 8 to avoid API errors with large responses
    # Increased from 50 to 80 for better efficiency
    # classify_chunk_with_fallback will auto-split if token limit hit
    chunks = chunk_structured_diff(structured_diff, max_files_per_chunk=80)
    if not chunks:
        raise ValueError(
            f'Issue {issue.get("id")} ground-truth diff could not be chunked'
        )

    visible_failure_context = _compact_context_for_diff_analysis(
        issue, benchmark_context
    )
    validation_sequence = benchmark_context.get('validation_sequence') or []
    chunk_findings: list[dict[str, Any]] = []

    print(
        f'  Step 1: Classifying patch changes by repaired validation ({total_files} files in {len(chunks)} chunk(s))...'
    )

    for index, chunk in enumerate(chunks, start=1):
        # Use fallback function with automatic splitting
        validations = classify_chunk_with_fallback(
            chunk=chunk,
            chunk_index=index,
            total_chunks=len(chunks),
            visible_failure_context=visible_failure_context,
            validation_sequence=validation_sequence,
            llm=llm,
            max_files=80,  # Start with 80, will auto-split if needed
        )

        if validations:
            chunk_findings.append(
                {'chunk_index': index, 'validations_in_this_chunk': validations}
            )

    # Step 2: Merge by validation (deterministic)
    print('  Step 2: Merging chunks by validation...')
    validation_groups = merge_chunks_by_validation(
        chunk_findings, validation_sequence, chunks
    )
    print(
        f'    Found {validation_groups["total_groups"]} groups from {validation_groups["total_validations"]} validations'
    )

    # Step 2.5: CI-Diff Correlation (deterministic)
    print('  Step 2.5: Analyzing CI-Diff correlation (layered structure)...')
    ci_context = _compact_context_for_diff_analysis(issue, benchmark_context)

    # Step 3: Deep reasoning with full context + correlation
    print('  Step 3: Deep reasoning with correlation context...')
    reasoning_result = analyze_validation_groups_with_reasoning(
        validation_groups, validation_sequence, ci_context, llm
    )
    # Log results
    atomic_problems = reasoning_result.get('atomic_problems', [])
    if atomic_problems:
        print(f'  OK Identified {len(atomic_problems)} atomic problems')
    else:
        print('  WARNING: No atomic problems identified')
    return {
        'mode': 'structured_diff_3step',
        'total_files': total_files,
        'total_changes': total_changes,
        'chunk_count': len(chunks),
        'chunk_findings': chunk_findings,
        'validation_groups': validation_groups,
        'atomic_problems': atomic_problems,
        'sequential_workflow_metadata': reasoning_result.get(
            'sequential_workflow_metadata', {}
        ),
    }


def decompose_issue(issue: dict, llm) -> dict:
    """
    Three-step reverse engineering from CI failure + ground truth diff:

    1. Classify chunks by validation (per chunk)
    2. Merge by validation (deterministic)
    3. Deep reasoning with full context (ConRAD/STAIR style)

    Returns specific, actionable atomic problems for mini-swe-agent.
    """

    issue_id = issue.get('id', '?')
    print(f'\n{"=" * 80}')
    print(f'Reverse Engineering Issue {issue_id}')
    print(f'  Repo: {issue.get("repo_name", issue.get("repo", "?"))}')
    print(f'  Changed files: {len(issue.get("changed_files", []))}')
    print(f'{"=" * 80}')

    try:
        print('  Fetching benchmark CI context and validation sequence...')
        benchmark_context = build_benchmark_ci_context(
            issue,
            llm=llm,
        )
        if not validate_required_ci_inputs(benchmark_context):
            return {}

        # Three-step analysis
        diff_context = analyze_diff_chunks(issue, benchmark_context, llm)

        atomic_problems = diff_context.get('atomic_problems', [])
        diff_context.get('sequential_workflow_metadata', {})

        if not atomic_problems:
            print('  WARNING: No atomic problems identified')
            return {}

        print(f'  OK Identified {len(atomic_problems)} atomic problems')

        # Build final result
        context = benchmark_context.get('context', {})
        log_analysis = benchmark_context.get('log_analysis', {})

        result = {
            'original_issue_id': issue_id,
            'sha_fail': issue.get('sha_fail'),
            'repo': issue.get('repo_name', issue.get('repo')),
            'original_error_type': issue.get('error_type'),
            # Atomic problems from three-step analysis
            'problems': atomic_problems,
            'total_problems': len(atomic_problems),
            'total_changed_files': len(issue.get('changed_files', [])),
            # Benchmark CI context (cleaned - no redundancy)
            'benchmark_ci_context': {
                'workflow_path': benchmark_context.get('workflow_path'),
                'workflow_name': benchmark_context.get('workflow_name'),
                'validation_sequence': benchmark_context.get('validation_sequence', []),
                # Summary level (for quick access)
                'overall_failure_reasons': context.get('overall_failure_reasons', []),
                'overall_error_types': context.get('overall_error_types', []),
                # Detailed analysis (structured)
                'error_types': log_analysis.get(
                    'error_types', []
                ),  # Detailed with subcategory + evidence
                'relevant_files': log_analysis.get(
                    'relevant_files', []
                ),  # Files with line numbers
                'failed_jobs': log_analysis.get(
                    'failed_job', []
                ),  # Job/step/command info
            },
            # Structured diff analysis metadata
            'diff_analysis_context': {
                'mode': diff_context.get('mode'),
                'total_files': diff_context.get('total_files', 0),
                'total_changes': diff_context.get('total_changes', 0),
                'chunk_count': diff_context.get('chunk_count', 0),
                'validation_groups_count': diff_context.get(
                    'validation_groups', {}
                ).get('total_groups', 0),
                'validation_groups': diff_context.get('validation_groups', {}).get(
                    'validation_groups', {}
                ),
            },
        }

        return result

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f'  ERROR Failed to decompose: {e}')
        print('\n--- FULL ERROR TRACE ---')
        print(error_trace)
        print('--- END TRACE ---\n')
        return {
            'error': 'DECOMPOSITION_ERROR',
            'error_message': str(e),
            'error_trace': error_trace,
            'error_type': type(e).__name__,
            'original_issue_id': issue.get('id'),
            'sha_fail': issue.get('sha_fail'),
        }


def generate_l1_l2_l3_pipeline(decomposed_result: dict, llm) -> dict:
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

    if 'error' in decomposed_result or not decomposed_result.get('problems'):
        return decomposed_result

    issue_id = decomposed_result.get('original_issue_id', '?')
    print(f'\n{"=" * 80}')
    print(f'L1/L2/L3 Pipeline for Issue {issue_id}')
    print(f'{"=" * 80}')

    # Stage 1: Deduplicate (mechanical)
    print('\n[Stage 1/5] Clustering and optimizing problems...')
    original_problems = decomposed_result.get('problems', [])
    print(f'  Input: {len(original_problems)} problems')

    # Group by validation_cmd first
    from collections import defaultdict

    validation_groups = defaultdict(list)
    for prob in original_problems:
        validation_cmd = prob.get('validation_cmd', 'unknown')
        validation_groups[validation_cmd].append(prob)

    print(f'  Grouped into {len(validation_groups)} validations')

    # Cluster and merge within each validation
    optimized_problems = []
    for validation_cmd, val_problems in validation_groups.items():
        print(f'    {validation_cmd}: {len(val_problems)} problems')

        if len(val_problems) > 1:
            # Apply clustering + LLM merge
            optimized = _cluster_and_merge_problems(
                val_problems,
                validation_cmd=validation_cmd,
                llm=llm,
                similarity_threshold=0.5,
            )
            print(f'      → Optimized to {len(optimized)} problems')
            optimized_problems.extend(optimized)
        else:
            # Single problem, keep as-is
            optimized_problems.extend(val_problems)

    # Reorder: primary → hidden, sorted by validation_order
    deduplicated = _reorder_by_repair_trajectory(optimized_problems)

    print(
        f'  Output: {len(deduplicated)} optimized problems (after clustering & merge)'
    )
    print(f'  Reduction: {len(original_problems) - len(deduplicated)} problems merged')

    # Stage 2: Detect dependencies (LLM)
    print('\n[Stage 2/5] Detecting dependencies with LLM...')
    dependencies = _stage2_detect_dependencies_llm(deduplicated, llm)
    print(f'  Dependencies found: {len(dependencies.get("dependency_edges", []))}')
    print(f'  Repair order: {dependencies.get("repair_order", [])}')

    # Stage 3: Generate L1 (with LLM for detailed descriptions)
    print('\n[Stage 3/5] Generating L1 (file-level) with detailed descriptions...')
    repo = decomposed_result.get('repo', 'unknown')
    workflow_path = decomposed_result.get('benchmark_ci_context', {}).get(
        'workflow_path', 'unknown'
    )
    issue_id_for_l1 = decomposed_result.get('original_issue_id', issue_id)
    l1 = _stage3_generate_l1_with_llm(
        deduplicated,
        dependencies,
        decomposed_result,
        llm,
        repo,
        workflow_path,
        issue_id_for_l1,
    )
    print(f'  L1 file-level problems: {len(l1)}')

    # Stage 4: Generate L2 (LLM)
    print('\n[Stage 4/5] Generating L2 (repair sequence) with LLM...')
    l2 = _stage4_generate_l2_llm(
        deduplicated, dependencies, llm, issue_id, decomposed_result.get('repo')
    )
    print(f'  L2 repair steps: {len(l2.get("problems", []))}')

    # Stage 5: Generate L3 (LLM)
    print('\n[Stage 5/5] Generating L3 (analysis) with LLM...')
    l3 = _stage5_generate_l3_llm(l1, l2, deduplicated, llm)
    print('  L3 insights generated')

    # Build final result
    result = {
        'issue_id': str(issue_id),
        'repo': decomposed_result.get('repo', 'unknown'),
        'workflow_path': workflow_path,
        'l1_file_level': l1,
        'l2_repair_sequence': l2,
        'l3_analysis': l3,
        'metadata': {
            'original_problems_count': len(decomposed_result.get('problems', [])),
            'deduplicated_count': len(deduplicated),
            'l1_file_count': len(l1),
            'l2_step_count': len(l2.get('problems', [])),
            'total_time_minutes': l2.get('total_time_minutes', 0),
        },
    }

    print(f'\n{"=" * 80}')
    print('Pipeline Complete!')
    print(f'{"=" * 80}')

    return result


def _stage1_deduplicate_problems(decomposed_result: dict) -> list[dict]:
    """
    Stage 1: Deduplicate similar problems (semantic grouping).

    Groups problems by:
    1. Same validation_cmd AND
    2. Same issue_type (e.g., F401, E501) OR same root cause pattern

    This prevents merging DISTINCT problems that happen to be from same validation.
    Example: F401 (unused import) and E501 (line too long) both from ruff
    → Should stay separate because solutions are completely different!
    """
    problems = decomposed_result.get('problems', [])

    # First group by validation command
    validation_groups = {}
    for prob in problems:
        validation = prob.get('validation_cmd', 'unknown')
        if validation not in validation_groups:
            validation_groups[validation] = []
        validation_groups[validation].append(prob)

    # Then within each validation, group by issue_type or root cause pattern
    deduplicated = []
    for validation, val_problems in validation_groups.items():
        # Sub-group by issue_type (e.g., F401, E501, etc.)
        issue_groups = {}
        for prob in val_problems:
            # Use issue_type if available, otherwise use root cause pattern
            issue_types = prob.get('issue_type', [])
            if isinstance(issue_types, list) and issue_types:
                issue_key = issue_types[0]  # Primary issue type
            else:
                # Fallback: use first 50 chars of root cause as grouping key
                root_cause = prob.get('root_cause', 'unknown')[:50]
                issue_key = root_cause

            key = (validation, issue_key)
            if key not in issue_groups:
                issue_groups[key] = []
            issue_groups[key].append(prob)

        # For each issue group, merge similar problems
        for (val, issue), group_problems in issue_groups.items():
            # Collect all unique files
            all_files = []
            for p in group_problems:
                all_files.extend(p.get('affected_files', []))
            all_files = list(dict.fromkeys(all_files))  # Deduplicate

            # Use first as template but keep ALL sub_problems for L1
            template = group_problems[0]
            merged = {
                'validation_cmd': val,
                'validation_order': template.get('validation_order'),
                'problem_type': template.get('problem_type', 'unknown'),
                'what_broke': template.get('what_broke', 'Validation failed'),
                'root_cause': template.get('root_cause', 'Unknown'),
                'how_fixed': template.get('how_fixed', ''),
                'why_fixed_works': template.get('why_fixed_works', ''),
                'affected_files': all_files,
                'issue_types': list(
                    set(
                        p.get('issue_type', '')
                        for p in group_problems
                        if p.get('issue_type')
                    )
                ),
                'sub_problems': group_problems,  # CRITICAL: Keep ALL for L1 generation!
            }
        deduplicated.append(merged)

    return deduplicated


def _stage2_detect_dependencies_llm(problems: list[dict], llm) -> dict:
    """
    Stage 2: Detect dependencies between problems using LLM.

    Three-tier approach:
    - Tier 1: Try full LLM analysis (small data)
    - Tier 2: Grouped LLM per validation (large data)
    - Tier 3: Mechanical heuristics (LLM fails)
    """

    # Check data size
    problems_json_size = len(
        json.dumps(
            [
                {
                    'id': idx,
                    'validation_order': prob.get('validation_order'),
                    'validation_cmd': prob.get('validation_cmd'),
                    'problem_type': prob.get('problem_type'),
                    'problem': prob.get('problem', '')[:300],
                    'root_cause': prob.get('root_cause', '')[:300],
                    'affected_files': prob.get('affected_files', [])[:10],
                }
                for idx, prob in enumerate(problems, 1)
            ],
            indent=2,
        )
    )

    if problems_json_size < 20000 and len(problems) <= 20:
        # Small data - try full LLM
        print(
            f'  Dependencies: Full LLM approach ({problems_json_size} chars, {len(problems)} problems)'
        )
        return _stage2_dependencies_full_llm(problems, llm)
    else:
        # Large data - use grouped approach
        print(
            f'  Dependencies: Data too large ({problems_json_size} chars, {len(problems)} problems)'
        )
        print('  Using grouped LLM approach...')
        return _stage2_dependencies_grouped_llm(problems, llm)


def _stage2_dependencies_full_llm(problems: list[dict], llm) -> dict:
    """Full LLM dependency analysis for small/normal data."""

    # Prepare FULL context for LLM (not just summary)
    problems_summary = []
    for idx, prob in enumerate(problems, 1):
        problems_summary.append(
            {
                'id': idx,
                'validation_order': prob.get('validation_order', 'unknown'),
                'validation_cmd': prob.get('validation_cmd', 'unknown'),
                'problem_type': prob.get('problem_type', 'unknown'),
                'failure_type': prob.get('failure_type', 'unknown'),
                'problem': prob.get('problem', '')[:300],
                'root_cause': prob.get('root_cause', '')[:300],
                'how_fixed': prob.get('how_fixed', '')[:300],
                'why_fix_works': prob.get('why_fix_works', '')[:300],
                'affected_files': prob.get('affected_files', [])[:10],
                'files_count': len(prob.get('affected_files', [])),
            }
        )

    prompt = f"""Deep analysis of problem interdependencies and relationships.

PROBLEMS:
{json.dumps(problems_summary, indent=2)}

TASK: Analyze how these problems relate to each other.

CONTEXT-AWARE DEPENDENCY ANALYSIS:

For EACH pair of problems, ask:

1. BLOCKING Dependencies ("A must be fixed before B can be addressed"):
   - Does problem A block problem B?
   - Would fixing A allow B to be detected/fixed?
   - Are they in a validation sequence where A runs before B?

   Examples:
   - Config problem blocks validation (must install tool before it can validate)
   - Import error blocks type checking (must fix imports before types can be checked)
   - Formatting blocks parsing (must format before parser can read it)

2. CAUSALITY ("Fixing A causes/reveals B"):
   - Does fixing A reveal new problems?
   - Would A's fix change what B looks like?
   - Does A's root cause create the conditions for B?

   Examples:
   - Enabling a linter reveals new linting issues
   - Fixing imports reveals type errors that were hidden
   - Adding dependencies reveals compatibility issues

3. SHARED Context ("A and B affect each other"):
   - Do they modify the same files?
   - Do they fix different aspects of the same root issue?
   - Would fixing A require considering B?

   Examples:
   - Two problems in same file that interact
   - Config change + code change that must align
   - Test + implementation that must stay in sync

4. SIDE Effects ("Fixing A might affect B"):
   - Could A's fix break or change B?
   - Do they share assumptions?
   - Would fixing A first make B easier/harder?

5. INDEPENDENT ("A and B are unrelated"):
   - Can be fixed in any order
   - Don't interact or depend on each other
   - Separate validation stages, files, concerns

RELATIONSHIP TYPES:

Based on analysis above, identify these relationship types:

- "blocks": A must be fixed before B (strict dependency)
- "enables": Fixing A allows B to be detected (enablement)
- "reveals": Fixing A uncovers B (causality)
- "requires": A and B must be fixed together (shared context)
- "affects": Fixing A changes B (side effect)
- "independent": No relationship (can fix in any order)

For EACH relationship, provide:
- Type (one of above)
- Direction (from → to)
- Reason (WHY this relationship exists, based on root_cause/how_fixed)
- Strength (strong/medium/weak based on how critical the relationship is)

REPAIR ORDER ANALYSIS:

CRITICAL: Validation order and problem_type define the base repair sequence:
1. PRIMARY problems (problem_type="primary") ALWAYS come first - these are CI failures
2. HIDDEN problems (problem_type="hidden") ALWAYS come after - these are consecutive validations
3. Within each group, RESPECT validation_order (validation 8 before validation 11)
4. Dependencies can ONLY reorder within same validation_order + problem_type group

Example correct order:
  Problem A: validation_order=8, problem_type="primary"     ← 1st (primary, earliest validation)
  Problem B: validation_order=8, problem_type="primary"     ← 2nd (primary, same validation, use deps)
  Problem C: validation_order=11, problem_type="hidden"     ← 3rd (hidden, later validation)
  Problem D: validation_order=11, problem_type="hidden"     ← 4th (hidden, same validation, use deps)

WRONG order (NEVER do this):
  FAIL Hidden before primary
  FAIL validation_order=11 before validation_order=8
  FAIL Ignoring validation sequence for "semantic dependencies"

Based on dependencies, determine WITHIN each (problem_type, validation_order) group:
1. Which problems should be fixed first (block others in same group)
2. Which are intermediate (depend on some, enable others in same group)
3. Which can be in any order (independent within same group)

OUTPUT FORMAT:
{{
  "dependency_edges": [
    {{
      "from": 1,
      "to": 3,
      "type": "blocks",
      "reason": "Config file must declare tool before validation can run it",
      "strength": "strong"
    }},
    {{
      "from": 2,
      "to": 4,
      "type": "reveals",
      "reason": "Fixing formatter config reveals formatting issues in code",
      "strength": "medium"
    }},
    {{
      "from": 5,
      "to": 6,
      "type": "affects",
      "reason": "Both modify same file - changes may interact",
      "strength": "weak"
    }}
  ],
  "repair_order": [1, 2, 3, 4, 5, 6],
  "repair_stages": {{
    "stage_1_foundational": [1],
    "stage_2_intermediate": [2, 3],
    "stage_3_dependent": [4],
    "stage_4_independent": [5, 6]
  }},
  "problem_groups": [
    {{
      "problems": [5, 6],
      "relationship": "shared file context",
      "recommendation": "Consider fixing together to avoid conflicts"
    }}
  ],
  "reasoning": "Detailed explanation of the dependency structure and repair strategy"
}}

IMPORTANT:
- Analyze based on actual content (root_cause, how_fixed), not just keywords
- Think about what a developer would need to know about problem interactions
- Identify both obvious (validation order) and subtle (semantic) dependencies
- Be specific in reasons - explain WHY the relationship exists
- Consider the real-world implications of fix order

{STRICT_JSON_RULES}"""

    try:
        time.sleep(3)  # Rate limiting
        response = _invoke_json(llm, prompt)

        if isinstance(response, dict):
            return response
        else:
            print('  WARNING: LLM returned non-dict, using grouped fallback')
            return _stage2_dependencies_grouped_llm(problems, llm)
    except Exception as e:
        print(f'  ERROR in full LLM dependency detection: {e}')
        return _stage2_dependencies_grouped_llm(problems, llm)


def _stage2_dependencies_grouped_llm(problems: list[dict], llm) -> dict:
    """
    Grouped LLM dependency detection for large data.

    Strategy:
    1. Group problems by validation_cmd
    2. LLM analyzes dependencies within each validation
    3. Combine all dependencies
    4. Generate repair order
    """
    print('  Using validation-grouped dependency detection...')

    # Step 1: Group by validation_cmd
    validation_groups = {}
    for idx, prob in enumerate(problems, 1):
        cmd = prob.get('validation_cmd', 'unknown')
        if cmd not in validation_groups:
            validation_groups[cmd] = []
        validation_groups[cmd].append({'idx': idx, 'problem': prob})

    print(f'  Grouped into {len(validation_groups)} validation groups')

    # Step 2: Analyze dependencies within each validation group
    all_edges = []

    for validation_cmd, group in validation_groups.items():
        print(f'  Analyzing: {validation_cmd} ({len(group)} problems)')

        if len(group) == 1:
            # Single problem - no dependencies
            print('    → Single problem, no dependencies')
            continue

        try:
            # LLM analyzes this validation group
            group_edges = _analyze_validation_dependencies_llm(
                validation_cmd=validation_cmd, group=group, llm=llm
            )
            all_edges.extend(group_edges)
            print(f'    → Found {len(group_edges)} dependencies')

        except Exception as e:
            print(f'    → LLM failed: {e}, using mechanical heuristics')
            # Fallback to mechanical for this group
            group_edges = _detect_within_validation_dependencies_mechanical(group)
            all_edges.extend(group_edges)
            print(f'    → Mechanical: {len(group_edges)} dependencies')

    # Step 3: Generate repair order
    repair_order = _compute_repair_order_from_edges(
        problems=problems, dependency_edges=all_edges
    )

    return {
        'dependency_edges': all_edges,
        'repair_order': repair_order,
        'reasoning': 'Grouped LLM per validation + topological sort',
    }


def _analyze_validation_dependencies_llm(
    validation_cmd: str, group: list[dict], llm
) -> list[dict]:
    """
    Use LLM to analyze dependencies within a validation group.

    LLM analyzes:
    1. File-based: How file changes link to each other
    2. Problem-based: How one problem links to another
    3. Context-aware: Within this validation's context
    """

    # Prepare problem summaries for LLM
    problems_data = []
    for item in group:
        prob = item['problem']
        problems_data.append(
            {
                'id': item['idx'],
                'validation_cmd': validation_cmd,
                'validation_order': prob.get('validation_order'),
                'problem_type': prob.get('problem_type', 'unknown'),
                'problem': prob.get('problem', '')[:250],
                'root_cause': prob.get('root_cause', '')[:250],
                'how_fixed': prob.get('how_fixed', '')[:250],
                'affected_files': prob.get('affected_files', []),
                'file_count': len(prob.get('affected_files', [])),
            }
        )

    prompt = f"""Analyze dependencies within this validation group.

Validation: {validation_cmd}
Problems in this validation:
{json.dumps(problems_data, indent=2)}

TASK: Identify dependencies between these problems.

Analyze TWO types of relationships:

1. FILE-BASED DEPENDENCIES:
   - How do file changes link to each other?
   - Does changing file A affect file B?
   - Do files share context (same module, same functionality)?

   Examples:
   - Config file (pyproject.toml) affects code files
   - Import changes affect files that import from them
   - Files in same module/package interact

2. PROBLEM-BASED DEPENDENCIES:
   - How does one problem link to another?
   - Does fixing problem A enable/reveal/block problem B?
   - Are they part of same logical fix?

   Examples:
   - Fixing imports enables type checking
   - Config change enables linter to run
   - One problem reveals another (cascade)

OUTPUT format:
{{
  "dependencies": [
    {{
      "from": 1,
      "to": 2,
      "type": "blocks|enables|reveals|affects",
      "reason": "Explain how problem 'from' relates to problem 'to'",
      "file_link": "Optional: Explain file-based connection",
      "strength": "strong|medium|weak"
    }}
  ]
}}

Rules:
- Only include ACTUAL dependencies (not forced)
- Explain both file-based AND problem-based reasoning
- If no dependencies exist, return: {{"dependencies": []}}
- Be specific - explain WHY the dependency exists
- ALWAYS return valid JSON object with "dependencies" key

{STRICT_JSON_RULES}"""

    time.sleep(2)  # Rate limiting
    response = _invoke_json(llm, prompt)

    # Check if response is valid dict
    if not isinstance(response, dict):
        raise ValueError(f'LLM returned invalid response type: {type(response)}')

    dependencies = response.get('dependencies', [])

    # Convert to edge format
    edges = []
    for dep in dependencies:
        edges.append(
            {
                'from': dep.get('from'),
                'to': dep.get('to'),
                'type': dep.get('type', 'affects'),
                'reason': dep.get('reason', ''),
                'file_link': dep.get('file_link', ''),
                'strength': dep.get('strength', 'medium'),
            }
        )

    return edges


def _detect_within_validation_dependencies_mechanical(group: list[dict]) -> list[dict]:
    """
    Mechanical dependency detection within a validation group (no LLM).

    Heuristics:
    1. Config files affect code files
    2. File overlap >50% = dependency
    3. Fewer files before more files
    4. Same file = strong dependency
    """
    edges = []

    for i, item_a in enumerate(group):
        prob_a = item_a['problem']
        idx_a = item_a['idx']
        files_a = set(prob_a.get('affected_files', []))

        if not files_a:
            continue

        for j, item_b in enumerate(group):
            if i >= j:  # Skip self and already compared
                continue

            prob_b = item_b['problem']
            idx_b = item_b['idx']
            files_b = set(prob_b.get('affected_files', []))

            if not files_b:
                continue

            # Rule 1: Config files affect others
            if _is_config_problem(prob_a) and not _is_config_problem(prob_b):
                edges.append(
                    {
                        'from': idx_a,
                        'to': idx_b,
                        'type': 'enables',
                        'reason': 'Config change may affect validation behavior',
                        'strength': 'medium',
                    }
                )
                continue

            # Rule 2: File overlap
            overlap = files_a & files_b
            if overlap:
                overlap_ratio = len(overlap) / min(len(files_a), len(files_b))

                if overlap_ratio > 0.5:
                    # Significant overlap
                    edges.append(
                        {
                            'from': idx_a,
                            'to': idx_b,
                            'type': 'affects',
                            'reason': f'Shares {len(overlap)} files ({int(overlap_ratio * 100)}% overlap)',
                            'strength': 'strong' if overlap_ratio > 0.8 else 'medium',
                        }
                    )
                elif overlap_ratio > 0.2:
                    # Some overlap
                    edges.append(
                        {
                            'from': idx_a,
                            'to': idx_b,
                            'type': 'affects',
                            'reason': f'Shares {len(overlap)} files',
                            'strength': 'weak',
                        }
                    )

            # Rule 3: Fewer files before more files (if no other relationship)
            elif len(files_a) < len(files_b) and len(files_a) <= 3:
                edges.append(
                    {
                        'from': idx_a,
                        'to': idx_b,
                        'type': 'affects',
                        'reason': 'Simpler fix (fewer files) before complex',
                        'strength': 'weak',
                    }
                )

    return edges


def _is_config_problem(problem: dict) -> bool:
    """Check if problem involves config file changes."""
    files = problem.get('affected_files', [])

    config_patterns = [
        'pyproject.toml',
        'setup.py',
        'setup.cfg',
        'requirements.txt',
        '.yml',
        '.yaml',
        'Makefile',
        '.ini',
        '.cfg',
        'tox.ini',
    ]

    for f in files:
        for pattern in config_patterns:
            if pattern in f:
                return True

    return False


def _compute_repair_order_from_edges(
    problems: list[dict], dependency_edges: list[dict]
) -> list[int]:
    """
    Compute repair order from dependency edges.

    Strategy:
    1. Separate by problem_type (primary vs hidden)
    2. Topological sort within each group
    3. Use validation_order as tie-breaker
    4. Final: primary + hidden
    """

    # Separate by problem_type
    primary_indices = []
    hidden_indices = []

    for idx, prob in enumerate(problems, 1):
        ptype = prob.get('problem_type', 'primary')
        if ptype == 'primary':
            primary_indices.append(idx)
        else:
            hidden_indices.append(idx)

    # Topological sort each group
    primary_order = _topological_sort_with_validation(
        indices=primary_indices, problems=problems, edges=dependency_edges
    )

    hidden_order = _topological_sort_with_validation(
        indices=hidden_indices, problems=problems, edges=dependency_edges
    )

    return primary_order + hidden_order


def _topological_sort_with_validation(
    indices: list[int], problems: list[dict], edges: list[dict]
) -> list[int]:
    """
    Topological sort respecting dependencies + validation_order tie-breaker.
    """
    if not indices:
        return []

    # Build adjacency list for these indices only
    graph = {idx: [] for idx in indices}
    in_degree = {idx: 0 for idx in indices}

    for edge in edges:
        from_idx = edge.get('from')
        to_idx = edge.get('to')

        if from_idx in indices and to_idx in indices:
            graph[from_idx].append(to_idx)
            in_degree[to_idx] += 1

    # Topological sort with validation_order as tie-breaker
    result = []
    queue = []

    # Start with nodes that have no dependencies
    for idx in indices:
        if in_degree[idx] == 0:
            queue.append(idx)

    # Sort queue by validation_order
    queue.sort(key=lambda idx: problems[idx - 1].get('validation_order', 999))

    while queue:
        # Take node with smallest validation_order
        current = queue.pop(0)
        result.append(current)

        # Process neighbors
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

        # Keep queue sorted by validation_order
        queue.sort(key=lambda idx: problems[idx - 1].get('validation_order', 999))

    # If there are remaining nodes (cycle detected), add them sorted by validation_order
    if len(result) < len(indices):
        remaining = [idx for idx in indices if idx not in result]
        remaining.sort(key=lambda idx: problems[idx - 1].get('validation_order', 999))
        result.extend(remaining)

    return result


def _stage3_generate_l1_with_llm(
    deduplicated: list[dict],
    dependencies: dict,
    decomposed_result: dict,
    llm,
    repo: str,
    workflow_path: str,
    issue_id: str,
) -> list[dict]:
    """
    Stage 3: Generate L1 (problem-level) with descriptive IDs.

    NEW APPROACH (Option 2):
    - One L1 entry per PROBLEM (not per file)
    - Files stored as ARRAY (can be multiple)
    - Descriptive IDs: {repo}_{issue_id}_{problem_id}
    - Dependencies use descriptive IDs

    L1 Format:
    [
      {
        "id": "flower_117_24",  # Descriptive ID
        "repo": "flower",
        "workflow": ".github/workflows/framework.yml",
        "validation_cmd": "python -m mdformat",
        "failure_type": "Code Formatting",
        "issue_type": "markdown formatting",
        "file": ["file1.py", "file2.py", ...],  # ARRAY of files
        "problem": "Detailed description of root cause",
        "fixes": "How the problem was fixed",
        "why_fix": "Why this fix works",
        "enabled": ["flower_117_25", ...],  # Descriptive IDs
        "enabled_by": ["flower_117_23", ...]
      }
    ]
    """

    print('  Generating L1 (one entry per problem with descriptive IDs)...')

    # Extract dependency edges
    dependency_edges = dependencies.get('dependency_edges', [])

    # Create L1 entries - ONE per problem
    l1_problems = []

    # After clustering, deduplicated contains flattened problems directly
    for prob in deduplicated:
        problem_id = prob.get('problem_id')
        files = prob.get('affected_files', [])
        validation_cmd = prob.get('validation_cmd', '')

        # Get LLM-generated fields
        failure_type = prob.get('failure_type', 'unknown')
        issue_type = prob.get('issue_type', 'unknown')

        # Clean up why_fix
        why_fix = prob.get('why_fixed_works', '')
        if why_fix in ['?', 'Unknown', '']:
            why_fix = f'This fix resolves the validation failure by addressing: {prob.get("root_cause", "")[:150]}'

        # Generate descriptive ID: {repo}_{issue_id}_{problem_id}
        descriptive_id = f'{repo}_{issue_id}_{problem_id}'

        # Create L1 entry - ONE per problem
        l1_entry = {
            'id': descriptive_id,  # Descriptive ID (no separate problem_id)
            'repo': repo,
            'workflow': workflow_path,
            'validation_cmd': validation_cmd,
            'failure_type': failure_type,
            'issue_type': issue_type,
            'file': files,  # ARRAY of files (can be multiple)
            'problem': f'{prob.get("what_broke", "Validation failed")}. Root cause: {prob.get("root_cause", "Unknown")}',
            'fixes': prob.get('how_fixed', 'Unknown'),
            'why_fix': why_fix,
            'enabled': [],  # Will be filled below
            'enabled_by': [],
        }

        l1_problems.append(l1_entry)

    # Second pass: Add dependencies with descriptive IDs
    # Build mapping: problem_id → descriptive_id
    problem_id_to_descriptive = {}
    for l1_entry in l1_problems:
        # Extract problem_id from descriptive_id (format: repo_issue_problem)
        parts = l1_entry['id'].split('_')
        if len(parts) >= 3:
            problem_id = int(parts[-1])  # Last part is problem_id
            problem_id_to_descriptive[problem_id] = l1_entry['id']

    # Add dependencies using descriptive IDs
    for l1_entry in l1_problems:
        # Extract problem_id from descriptive_id
        parts = l1_entry['id'].split('_')
        if len(parts) >= 3:
            problem_id = int(parts[-1])

            # Find enabled problems
            enabled_problem_ids = [
                edge.get('to')
                for edge in dependency_edges
                if edge.get('from') == problem_id
            ]

            # Convert to descriptive IDs
            l1_entry['enabled'] = [
                problem_id_to_descriptive.get(pid)
                for pid in enabled_problem_ids
                if pid in problem_id_to_descriptive
            ]

            # Find enabled_by problems
            enabled_by_problem_ids = [
                edge.get('from')
                for edge in dependency_edges
                if edge.get('to') == problem_id
            ]

            # Convert to descriptive IDs
            l1_entry['enabled_by'] = [
                problem_id_to_descriptive.get(pid)
                for pid in enabled_by_problem_ids
                if pid in problem_id_to_descriptive
            ]

    print(f'    Created {len(l1_problems)} L1 entries (one per problem)')
    print(f'    Total files covered: {sum(len(e["file"]) for e in l1_problems)}')

    return l1_problems


def _find_file_changes(target_file: str, sub_problems: list[dict]) -> str:
    """Find the specific changes for a target file from sub_problems."""
    for sub_prob in sub_problems:
        if target_file in sub_prob.get('affected_files', []):
            return sub_prob.get('how_fixed', '')[:300]
    return ''


def _stage4_generate_l2_llm(
    deduplicated: list[dict], dependencies: dict, llm, issue_id: str, repo: str
) -> dict:
    """
    Stage 4: Generate L2 (repair sequence) with Three-Tier Strategy.

    Tier 1 (<10 problems): Single-pass LLM (best quality)
    Tier 2 (10-20 problems): Grouped LLM per validation (good quality)
    Tier 3 (>20 problems): Mechanical generation (guaranteed complete)
    """

    # After clustering in Stage 1, deduplicated already contains flattened problems
    # No need to flatten sub_problems - they ARE the problems
    flattened_problems = deduplicated

    actual_count = len(flattened_problems)
    prompt_size = len(json.dumps(flattened_problems, indent=2))

    print(f'  L2 Strategy: {actual_count} problems, {prompt_size} chars')

    # TIER SELECTION

    # Tier 1: Small dataset - Single-pass LLM
    if actual_count <= 10 and prompt_size < 20000:
        print('  → Tier 1: Single-pass LLM')
        try:
            result = _l2_tier1_single_pass(
                flattened_problems, dependencies, llm, issue_id, repo
            )

            # Validate: check if ALL problems returned
            if len(result.get('problems', [])) >= actual_count * 0.9:  # 90% threshold
                print(f'  OK Tier 1 success: {len(result["problems"])} problems')
                return result
            else:
                print(
                    f'  [WARN] Tier 1 incomplete ({len(result.get("problems", []))}/{actual_count})'
                )
                # Fall through to Tier 2
        except Exception as e:
            print(f'  [WARN] Tier 1 failed: {str(e)[:100]}')
            # Fall through to Tier 2

    # Tier 2: Medium dataset - Grouped LLM
    if actual_count <= 20:
        print('  → Tier 2: Grouped LLM per validation')
        try:
            result = _l2_tier2_grouped_llm(
                flattened_problems, dependencies, llm, issue_id, repo
            )

            # Validate
            if len(result.get('problems', [])) >= actual_count * 0.9:
                print(f'  OK Tier 2 success: {len(result["problems"])} problems')
                return result
            else:
                print(
                    f'  [WARN] Tier 2 incomplete ({len(result.get("problems", []))}/{actual_count})'
                )
                # Fall through to Tier 3
        except Exception as e:
            print(f'  [WARN] Tier 2 failed: {str(e)[:100]}')
            # Fall through to Tier 3

    # Tier 3: Large dataset or fallback - Mechanical (guaranteed)
    print('  → Tier 3: Mechanical generation (guaranteed complete)')
    result = _l2_tier3_mechanical(flattened_problems, dependencies, issue_id, repo)
    print(f'  OK Tier 3 success: {len(result["problems"])} problems')

    return result


def _l2_tier1_single_pass(
    problems: list[dict], dependencies: dict, llm, issue_id: str, repo: str
) -> dict:
    """Tier 1: Single-pass LLM for small datasets."""

    # Prepare data
    problems_for_llm = []
    for idx, prob in enumerate(problems, 1):
        problems_for_llm.append(
            {
                'id': idx,
                'validation_order': prob.get('validation_order', 999),
                'validation_cmd': prob.get('validation_cmd', ''),
                'problem_type': prob.get('problem_type', ''),
                'what_broke': prob.get('what_broke', ''),
                'root_cause': prob.get('root_cause', ''),
                'how_fixed': prob.get('how_fixed', ''),
                'files': prob.get('affected_files', []),
                'issue_type': prob.get('issue_type', ''),
            }
        )

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
      "fix_strategy": "Single comprehensive paragraph explaining: what to change, how to fix it, and why it works. Use the how_fixed and why_fixed_works from input data, combining them naturally into one flowing explanation.",
      "pattern_detected": null or {{
        "type": "bulk_formatting",
        "rule": "Pattern description",
        "scope": "X files"
      }},
      "files": ["path/to/file-1.ext", "path/to/file-2.ext", ...],
      "estimated_time_minutes": 5
    }},
    // Repeat for each problem in repair order
  ],
  "total_problems": <number of problems in repair sequence>,
}}

CRITICAL: fix_strategy must be a natural, flowing paragraph that:
1. Starts with WHAT and HOW to fix (from input how_fixed)
2. Continues with WHY it works (from input why_fixed_works)
3. NO labels like "Why this works:" - just natural sentences
4. Example: "Added .strip() method call to ensure whitespace is removed. The .strip() method removes leading/trailing whitespace, ensuring the value satisfies type expectations."

CRITICAL ORDERING RULES (MUST FOLLOW):
1. PRIMARY problems (problem_type="primary") MUST come FIRST
   - These are CI failures visible in logs
   - Fix these before any hidden problems

2. HIDDEN problems (problem_type="hidden") come AFTER primary
   - These are consecutive validations that never ran
   - Fix in validation_order sequence

3. Within each group (primary/hidden):
   - Respect validation_order (lower order = earlier in sequence)
   - Use dependencies to break ties (if problem A depends on B, do B first)
   - Independent problems at same validation can be in any order

4. NEVER reorder such that hidden comes before primary
5. NEVER violate validation sequence (validation_order 8 must come before 11)

OTHER RULES:
- "files": Array of ACTUAL file paths from diff (max 50), NO speculation or patterns
- Detect patterns for bulk operations (>10 files)
- Be specific and actionable

{STRICT_JSON_RULES}"""

    try:
        time.sleep(3)  # Rate limiting
        response = _invoke_json(llm, prompt)

        if isinstance(response, dict) and 'problems' in response:
            return response
        else:
            raise ValueError('Invalid response from LLM')
    except Exception as e:
        print(f'  Tier 1 error: {str(e)[:100]}')
        raise


def _l2_tier2_grouped_llm(
    problems: list[dict], dependencies: dict, llm, issue_id: str, repo: str
) -> dict:
    """Tier 2: Grouped LLM per validation for medium datasets."""

    # Group by validation_cmd
    validation_groups = defaultdict(list)
    for prob in problems:
        cmd = prob.get('validation_cmd', 'unknown')
        validation_groups[cmd].append(prob)

    print(f'    Grouped into {len(validation_groups)} validations')

    all_l2_problems = []

    for validation_cmd, group_problems in validation_groups.items():
        print(f'    Processing {validation_cmd}: {len(group_problems)} problems')

        # Prepare problems for this validation
        problems_for_llm = []
        for idx, prob in enumerate(group_problems, 1):
            problems_for_llm.append(
                {
                    'id': idx,
                    'validation_order': prob.get('validation_order', 999),
                    'validation_cmd': prob.get('validation_cmd', ''),
                    'problem_type': prob.get('problem_type', ''),
                    'what_broke': prob.get('what_broke', ''),
                    'root_cause': prob.get('root_cause', ''),
                    'how_fixed': prob.get('how_fixed', ''),
                    'files': prob.get('affected_files', []),
                    'issue_type': prob.get('issue_type', ''),
                }
            )

        prompt = f"""Organize repair sequence for validation: {validation_cmd}

Problems in this validation ({len(group_problems)}):
{json.dumps(problems_for_llm, indent=2)}

Generate L2 entries for these problems in this EXACT format:
{{
  "problems": [
    {{
      "problem_id": <original_id>,
      "verification_cmd": "{validation_cmd}",
      "failure_type": "...",
      "problem": "Clear description",
      "root_cause": "Technical explanation. Scope: affected area",
      "fix_strategy": "Single comprehensive paragraph: what to change, how to fix it, and why it works. Combine how_fixed and why_fixed_works from input naturally.",
      "pattern_detected": null or {{
        "type": "bulk_formatting",
        "rule": "Pattern description",
        "scope": "X files"
      }},
      "files": ["path/to/file.ext", ...],
      "estimated_time_minutes": 5
    }}
  ]
}}

CRITICAL: fix_strategy must combine input data naturally:
- Take how_fixed: "Added .strip() method call..."
- Take why_fixed_works: "The .strip() method removes whitespace..."
- Combine: "Added .strip() method call to ensure whitespace is removed. The .strip() method removes leading/trailing whitespace, ensuring the value satisfies type expectations."
- NO labels, just natural flowing text

{STRICT_JSON_RULES}"""

        try:
            time.sleep(2)
            response = _invoke_json(llm, prompt)

            if isinstance(response, dict) and 'problems' in response:
                # Add metadata back to problems
                for l2_prob in response['problems']:
                    orig_prob = group_problems[l2_prob.get('problem_id', 1) - 1]
                    l2_prob['problem_type'] = orig_prob.get('problem_type')
                    l2_prob['validation_order'] = orig_prob.get('validation_order', 999)

                all_l2_problems.extend(response['problems'])
            else:
                # Fallback: use mechanical for this validation
                print(f'      LLM failed for {validation_cmd}, using mechanical')
                for prob in group_problems:
                    all_l2_problems.append(_mechanical_l2_entry(prob))
        except Exception as e:
            print(f'      Error processing {validation_cmd}: {str(e)[:80]}')
            # Fallback: mechanical for this validation
            for prob in group_problems:
                all_l2_problems.append(_mechanical_l2_entry(prob))

    # Final organization: primary → hidden, by validation_order
    # Handle both "primary"/"hidden" and "primary_failure"/"hidden_failure" formats
    primary = [
        p
        for p in all_l2_problems
        if 'primary' in str(p.get('problem_type', '')).lower()
    ]
    hidden = [
        p for p in all_l2_problems if 'hidden' in str(p.get('problem_type', '')).lower()
    ]

    primary.sort(key=lambda p: p.get('validation_order', 999))
    hidden.sort(key=lambda p: p.get('validation_order', 999))

    return {'problems': primary + hidden, 'total_problems': len(all_l2_problems)}


def _l2_tier3_mechanical(
    problems: list[dict], dependencies: dict, issue_id: str, repo: str
) -> dict:
    """Tier 3: Mechanical generation - guaranteed complete, no LLM."""

    # Group by validation for organization
    validation_groups = defaultdict(list)
    for prob in problems:
        cmd = prob.get('validation_cmd', 'unknown')
        validation_groups[cmd].append(prob)

    print(f'    Processing {len(validation_groups)} validations mechanically')

    all_l2_problems = []

    for validation_cmd, group_problems in validation_groups.items():
        print(f'      {validation_cmd}: {len(group_problems)} problems')

        for prob in group_problems:
            l2_entry = _mechanical_l2_entry(prob)
            all_l2_problems.append(l2_entry)

    # Final organization: primary → hidden, by validation_order
    # Handle both "primary"/"hidden" and "primary_failure"/"hidden_failure" formats
    primary = [
        p
        for p in all_l2_problems
        if 'primary' in str(p.get('problem_type', '')).lower()
    ]
    hidden = [
        p for p in all_l2_problems if 'hidden' in str(p.get('problem_type', '')).lower()
    ]

    primary.sort(key=lambda p: p.get('validation_order', 999))
    hidden.sort(key=lambda p: p.get('validation_order', 999))

    return {
        'problems': primary + hidden,
        'total_problems': len(all_l2_problems),
        'approach': 'mechanical',
    }


def _mechanical_l2_entry(prob: dict) -> dict:
    """Generate a single L2 entry mechanically - preserves EXACT decomposition data."""

    files = prob.get('affected_files', [])
    is_merged = prob.get('is_merged', False)

    # Detect pattern for bulk/merged problems
    pattern = None
    if len(files) > 10 or is_merged:
        pattern = {
            'type': 'bulk_formatting' if len(files) > 10 else 'merged_problems',
            'rule': _detect_pattern_rule(prob),
            'scope': f'{len(files)} files' if is_merged else f'{len(files)} files',
        }

    # Use EXACT data from decomposition - NO templating
    how_fixed = prob.get('how_fixed', '')
    why_fixed_works = prob.get('why_fixed_works', '')

    # Create comprehensive fix_strategy: natural paragraph combining how + why
    if how_fixed and why_fixed_works:
        # Combine into single flowing paragraph
        fix_strategy = f'{how_fixed} {why_fixed_works}'
    elif how_fixed:
        fix_strategy = how_fixed
    else:
        fix_strategy = f'Address the root cause: {prob.get("root_cause", "See problem description")}'

    # Estimated time
    time_estimate = _estimate_time_mechanical(len(files))

    return {
        'problem_id': prob.get('problem_id'),
        'verification_cmd': prob.get('validation_cmd'),
        'failure_type': prob.get('failure_type', ''),
        'problem': prob.get('what_broke', '') or prob.get('problem', ''),
        'root_cause': prob.get('root_cause', ''),
        'fix_strategy': fix_strategy,  # Combined: how_fixed + why_it_works
        'pattern_detected': pattern,
        'files': files,  # ALL files, no limit
        'estimated_time_minutes': time_estimate,
        'problem_type': prob.get(
            'problem_type', 'unknown'
        ),  # CRITICAL: preserve problem_type
        'validation_order': prob.get('validation_order', 999),
    }


def _detect_pattern_rule(problem: dict) -> str:
    """Detect pattern rule mechanically based on issue type."""

    issue_type = problem.get('issue_type', '').lower()
    failure_type = problem.get('failure_type', '').lower()

    # Common patterns
    if 'type' in issue_type and 'hint' in issue_type:
        return 'Add type annotations for all untyped parameters'
    elif 'rst' in issue_type or 'underline' in issue_type:
        return 'RST section header underline length must match header text exactly'
    elif 'import' in issue_type:
        return 'Import sorting - standard library, third-party, then local imports'
    elif 'formatting' in failure_type or 'format' in issue_type:
        return f'Apply {failure_type} rules consistently'
    else:
        return f'{failure_type}: {issue_type}'


def _estimate_time_mechanical(file_count: int) -> int:
    """Estimate time based on file count."""
    if file_count == 1:
        return 5
    elif file_count <= 5:
        return 10
    elif file_count <= 10:
        return 15
    else:
        return 20


def _stage5_generate_l3_llm(
    l1: list[dict], l2: dict, deduplicated: list[dict], llm
) -> list[dict]:
    """
    Stage 5: Generate L3 (universal patterns) using LLM.

    Analyze DISTINCT problem patterns and extract universal fixes.
    Each independent problem = separate entry.

    Uses tiered approach:
    - Small data: Single-pass LLM
    - Large data: Grouped by validation + dependency analysis
    - LLM fails: Mechanical fallback with pattern detection
    """

    l2_problems = l2.get('problems', [])

    # Check data size
    l2_json_size = len(json.dumps(l2_problems, indent=2))

    if l2_json_size < 30000:
        # Small enough - use single-pass approach
        print(f'  L3: Single-pass approach ({l2_json_size} chars)')
        return _stage5_l3_single_pass(l2_problems, llm)
    else:
        # Too large - use grouped approach
        print(f'  L3: Data too large ({l2_json_size} chars), using grouped approach')
        return _stage5_l3_grouped(l2_problems, deduplicated, llm)


def _stage5_l3_single_pass(l2_problems: list[dict], llm) -> list[dict]:
    """Single-pass L3 generation for small/normal-sized data."""

    prompt = f"""Analyze CI failure patterns and extract UNIVERSAL PROBLEM PATTERNS (L3).

L2 Repair Sequence:
{json.dumps(l2_problems, indent=2)}

Task: Extract distinct, independent problem patterns with universal fixes.

CRITICAL RULES:
1. Each INDEPENDENT problem = separate entry
2. Only link problems if there's ACTUAL dependency
3. No forced grouping - mypy ≠ mdformat (separate entries)
4. Extract universal fixes that apply to similar future problems
5. **IGNORE Git workflow issues** - DO NOT create patterns for merge conflicts
   - Merge conflict markers are Git problems, NOT CI validation problems
   - Focus on ACTUAL validation issues (type errors, formatting, imports, etc.)

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
            print('  WARNING: LLM returned invalid L3, using fallback')
            return _fallback_l3_intelligent({'problems': l2_problems})
    except Exception as e:
        print(f'  ERROR in L3 single-pass: {e}')
        return _fallback_l3_intelligent({'problems': l2_problems})


def _stage5_l3_grouped(
    l2_problems: list[dict], deduplicated: list[dict], llm
) -> list[dict]:
    """
    Grouped L3 generation for large data.

    Strategy:
    1. Group problems by validation_cmd
    2. LLM extracts patterns per validation group (small prompts)
    3. LLM identifies cross-validation dependencies
    """
    print('  Using validation-grouped L3 approach...')

    # Step 1: Group by validation_cmd
    validation_groups = {}
    for prob in l2_problems:
        cmd = prob.get('validation_cmd', 'unknown')
        if cmd not in validation_groups:
            validation_groups[cmd] = []
        validation_groups[cmd].append(prob)

    print(f'  Grouped into {len(validation_groups)} validation groups')

    # Step 2: Extract patterns per validation group
    all_patterns = []

    for validation_cmd, group_problems in validation_groups.items():
        print(
            f'  Extracting patterns for: {validation_cmd} ({len(group_problems)} problems)'
        )

        try:
            group_patterns = _extract_patterns_for_validation_group(
                validation_cmd=validation_cmd, problems=group_problems, llm=llm
            )
            all_patterns.extend(group_patterns)
            print(f'    → Extracted {len(group_patterns)} patterns')
        except Exception as e:
            print(
                f'    → LLM failed for {validation_cmd}: {e}, using mechanical extraction'
            )
            # Fallback for this group
            mechanical_patterns = _extract_patterns_mechanical(
                validation_cmd, group_problems
            )
            all_patterns.extend(mechanical_patterns)

    # Step 3: Identify cross-validation dependencies
    print(f'  Analyzing dependencies across {len(all_patterns)} patterns...')

    try:
        final_patterns = _identify_cross_pattern_dependencies(all_patterns, llm)
        print('  → Found dependencies between patterns')
        return final_patterns
    except Exception as e:
        print(f'  → Dependency analysis failed: {e}, using file-based heuristic')
        # Fallback: use file overlap to detect dependencies
        return _detect_file_based_dependencies(all_patterns)


def _extract_patterns_for_validation_group(
    validation_cmd: str, problems: list[dict], llm
) -> list[dict]:
    """Extract patterns for a single validation group using LLM."""

    prompt = f"""Extract universal patterns from this validation group.

Validation: {validation_cmd}
Problems in this validation:
{json.dumps(problems, indent=2)}

Task: Identify DISTINCT problem patterns within this validation.

For EACH distinct pattern, extract:
1. Pattern ID (unique identifier)
2. Failure pattern (what breaks)
3. Problem (why it occurs, root cause)
4. Universal fix (approach + steps)
5. Examples (before/after)

Output JSON array of patterns:
[
  {{
    "pattern_id": "descriptive_unique_id",
    "failure_type": "type_checking",
    "verification_cmd": "{validation_cmd}",
    "failure_pattern": "Brief description of what fails",
    "problem": "Why this happens, root cause, context",
    "universal_fix": {{
      "approach": "High-level fix strategy",
      "steps": ["Step 1", "Step 2", "Step 3"],
      "applies_to": ["Similar cases where this applies"]
    }},
    "examples": [
      {{
        "file": "example_file.py",
        "before": "code before",
        "after": "code after"
      }}
    ],
    "dependent_problems": []
  }}
]

IMPORTANT:
- Group similar problems into ONE pattern (e.g., all F401 unused imports = 1 pattern)
- Separate patterns for different issues (F401 ≠ E501)
- Extract universal fixes that apply to similar future problems

{STRICT_JSON_RULES}"""

    time.sleep(2)  # Rate limiting
    response = _invoke_json(llm, prompt)

    if isinstance(response, list):
        # Add validation_cmd to each pattern
        for pattern in response:
            pattern['validation_cmd'] = validation_cmd
            pattern['validation_scope'] = 'single_validation'
        return response
    else:
        raise ValueError('Invalid response format')


def _identify_cross_pattern_dependencies(patterns: list[dict], llm) -> list[dict]:
    """Identify dependencies between patterns from different validations using LLM."""

    # Create compact pattern summaries for LLM
    pattern_summaries = []
    for p in patterns:
        pattern_summaries.append(
            {
                'pattern_id': p.get('pattern_id', 'unknown'),
                'validation_cmd': p.get('validation_cmd', 'unknown'),
                'failure_type': p.get('failure_type', 'unknown'),
                'failure_pattern': p.get('failure_pattern', '')[:150],
                'files_count': len(p.get('examples', [])),
                'fix_approach': p.get('universal_fix', {}).get('approach', '')[:100],
            }
        )

    prompt = f"""Analyze dependencies between these patterns from different validations.

Patterns:
{json.dumps(pattern_summaries, indent=2)}

Task: Identify which patterns depend on others.

Look for:
1. Config changes that enable/affect other patterns
2. Code changes that require config updates
3. Sequential dependencies (A must be fixed before B works)
4. Related patterns affecting same functionality

Output format:
{{
  "dependencies": [
    {{
      "from_pattern": "pattern_id_1",
      "to_pattern": "pattern_id_2",
      "relationship": "enables|requires|affects",
      "rationale": "Why this dependency exists"
    }}
  ]
}}

Only include dependencies that ACTUALLY exist.
Empty array if no cross-pattern dependencies.

{STRICT_JSON_RULES}"""

    time.sleep(2)  # Rate limiting
    response = _invoke_json(llm, prompt)

    # Add dependencies to patterns
    dependencies = response.get('dependencies', [])

    for dep in dependencies:
        from_id = dep.get('from_pattern')
        to_id = dep.get('to_pattern')

        # Find the pattern and add dependency
        for p in patterns:
            if p.get('pattern_id') == from_id:
                if 'dependent_problems' not in p:
                    p['dependent_problems'] = []
                p['dependent_problems'].append(
                    {
                        'pattern_id': to_id,
                        'relationship': dep.get('relationship', 'related'),
                        'rationale': dep.get('rationale', ''),
                    }
                )

    return patterns


def _extract_patterns_mechanical(
    validation_cmd: str, problems: list[dict]
) -> list[dict]:
    """Mechanical pattern extraction for a validation group (no LLM)."""

    # Group by problem_type within validation
    type_groups = {}
    for prob in problems:
        ptype = prob.get('problem_type', 'unknown')
        if ptype not in type_groups:
            type_groups[ptype] = []
        type_groups[ptype].append(prob)

    patterns = []
    for ptype, group_probs in type_groups.items():
        # Combine all files
        all_files = set()
        for p in group_probs:
            all_files.update(p.get('files', []))

        # Create pattern
        pattern = {
            'pattern_id': f'pattern_{validation_cmd.replace(" ", "_")}_{ptype}',
            'validation_cmd': validation_cmd,
            'problem_type': ptype,
            'failure_type': group_probs[0].get('failure_type', 'unknown'),
            'failure_pattern': f'{len(group_probs)} {ptype} problems in {validation_cmd}',
            'problem': _combine_problem_descriptions(group_probs),
            'universal_fix': {
                'approach': _extract_common_fix_approach(group_probs),
                'steps': ['Mechanical fallback - see individual problems for details'],
                'applies_to': [f'{validation_cmd} {ptype} issues'],
            },
            'examples': [],
            'affected_files': list(all_files),
            'file_count': len(all_files),
            'dependent_problems': [],
            'validation_scope': 'single_validation',
        }
        patterns.append(pattern)

    return patterns


def _combine_problem_descriptions(problems: list[dict]) -> str:
    """Combine problem descriptions from multiple problems."""
    if len(problems) == 1:
        return problems[0].get('problem', 'Unknown problem')

    # Take common theme
    first_prob = problems[0].get('problem', '')
    if len(first_prob) > 100:
        return f'{first_prob[:100]}... ({len(problems)} similar problems)'
    else:
        return f'Multiple {problems[0].get("failure_type", "validation")} issues ({len(problems)} problems)'


def _extract_common_fix_approach(problems: list[dict]) -> str:
    """Extract common fix approach from problems."""
    if len(problems) == 1:
        return problems[0].get('how_fixed', 'Fix validation failure')[:200]

    # Simple heuristic: use first problem's approach
    return problems[0].get('how_fixed', 'Fix validation failures')[:200]


def _detect_file_based_dependencies(patterns: list[dict]) -> list[dict]:
    """Detect dependencies between patterns based on file overlap."""

    for i, pattern_a in enumerate(patterns):
        files_a = set(pattern_a.get('affected_files', []))
        if not files_a:
            continue

        for j, pattern_b in enumerate(patterns):
            if i == j:
                continue

            files_b = set(pattern_b.get('affected_files', []))
            if not files_b:
                continue

            overlap = files_a & files_b

            # If significant overlap (>30% of either set)
            if overlap:
                overlap_ratio_a = len(overlap) / len(files_a)
                overlap_ratio_b = len(overlap) / len(files_b)

                if overlap_ratio_a > 0.3 or overlap_ratio_b > 0.3:
                    # Add dependency
                    if 'dependent_problems' not in pattern_a:
                        pattern_a['dependent_problems'] = []

                    pattern_a['dependent_problems'].append(
                        {
                            'pattern_id': pattern_b.get('pattern_id', 'unknown'),
                            'relationship': 'related_files',
                            'rationale': f'Shares {len(overlap)} files ({int(max(overlap_ratio_a, overlap_ratio_b) * 100)}% overlap)',
                        }
                    )

    return patterns


def _fallback_l3_intelligent(l2: dict) -> list[dict]:
    """
    Intelligent mechanical fallback for L3 (no LLM).

    Groups by (validation_cmd, problem_type) and detects dependencies by file overlap.
    NO truncation - preserves all data.
    """
    print('  Using intelligent mechanical L3 fallback...')

    problems = l2.get('problems', [])

    # Group by (validation_cmd, problem_type)
    pattern_groups = {}
    for prob in problems:
        validation = prob.get('validation_cmd', 'unknown')
        prob_type = prob.get('problem_type', 'unknown')
        key = (validation, prob_type)

        if key not in pattern_groups:
            pattern_groups[key] = []
        pattern_groups[key].append(prob)

    # Create patterns
    patterns = []
    for (validation, prob_type), group_probs in pattern_groups.items():
        # Collect all files
        all_files = set()
        for p in group_probs:
            all_files.update(p.get('files', []))

        pattern = {
            'pattern_id': f'pattern_{validation.replace(" ", "_")}_{prob_type}_{len(patterns) + 1}',
            'validation_cmd': validation,
            'problem_type': prob_type,
            'failure_type': group_probs[0].get('failure_type', 'unknown'),
            'failure_pattern': f'{len(group_probs)} {prob_type} problems in {validation}',
            'problem': _combine_problem_descriptions(group_probs),
            'universal_fix': {
                'approach': _extract_common_fix_approach(group_probs),
                'steps': ['Mechanical fallback - see L2 for detailed steps'],
                'applies_to': [f'{validation} validation failures'],
            },
            'examples': [],
            'affected_files': list(all_files),
            'file_count': len(all_files),
            'dependent_problems': [],
        }
        patterns.append(pattern)

    # Detect dependencies by file overlap
    patterns = _detect_file_based_dependencies(patterns)

    print(f'  Generated {len(patterns)} patterns (mechanical)')
    return patterns


def convert_to_l2_format(decomposed_result: dict) -> dict:
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

    if 'error' in decomposed_result or not decomposed_result.get('problems'):
        return decomposed_result

    issue_id = decomposed_result.get('original_issue_id', '?')
    problems = decomposed_result.get('problems', [])

    print('\n  Converting to L2 format...')
    print(f'    Input: {len(problems)} problems')

    # Group problems by validation command ONLY (not by error type)
    # This merges all problems from same validation into ONE problem
    problem_groups = {}
    for prob in problems:
        validation = prob.get('validation_cmd', 'unknown')
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
            all_files.extend(p.get('affected_files', []))
        all_files = list(dict.fromkeys(all_files))  # Deduplicate preserving order

        # Detect pattern if multiple files with same fix
        pattern_detected = None
        if len(all_files) > 5:  # Bulk operation threshold
            # Extract pattern info
            issue_type = template.get('issue_type', '')
            how_fixed = template.get('how_fixed', '')

            # Determine pattern type
            pattern_type = 'bulk_change'
            if 'format' in issue_type.lower() or 'format' in how_fixed.lower():
                pattern_type = 'bulk_formatting'
            elif 'config' in issue_type.lower() or 'enable' in how_fixed.lower():
                pattern_type = 'config_enablement'

            # Extract rule from how_fixed
            rule = how_fixed[:100] if how_fixed else 'Pattern not specified'

            pattern_detected = {
                'type': pattern_type,
                'rule': rule,
                'scope': f'{len(all_files)} files in {template.get("validation_cmd", "unknown")}',
            }

        # Build root cause string (summarize from all problems in group)
        root_causes = set()
        scopes = set()
        for p in group_problems:
            if p.get('root_cause'):
                root_causes.add(p['root_cause'][:150])
            if p.get('what_broke'):
                scopes.add(p['what_broke'][:80])

        root_cause_parts = []
        if root_causes:
            # Take first unique root cause (they're usually similar)
            root_cause_parts.append(list(root_causes)[0])
        if scopes:
            scope_str = f'Affects {len(all_files)} files'
            root_cause_parts.append(f'Scope: {scope_str}')
        root_cause = (
            ' '.join(root_cause_parts) if root_cause_parts else 'Unknown root cause'
        )

        # Build fix strategy string
        fix_parts = []

        # Approach
        problem_type = template.get('problem_type', 'unknown')
        if problem_type == 'primary':
            approach = 'direct_fix'
        elif problem_type == 'hidden':
            approach = 'consecutive'
        elif 'workaround' in template.get('how_fixed', '').lower():
            approach = 'workaround'
        else:
            approach = 'direct_fix'
        fix_parts.append(f'Approach: {approach}')

        # What
        what_broke = template.get('what_broke', '')
        if what_broke:
            fix_parts.append(f'What: {what_broke[:100]}')

        # How
        how_fixed = template.get('how_fixed', '')
        if how_fixed:
            fix_parts.append(f'How: {how_fixed[:200]}')

        # Why
        why_works = template.get('why_fixed_works', '')
        if why_works:
            fix_parts.append(f'Why: {why_works[:100]}')

        # Time estimate (rough)
        time_estimate = 5  # default
        if len(all_files) > 50:
            time_estimate = 30
        elif len(all_files) > 10:
            time_estimate = 15
        elif len(all_files) > 2:
            time_estimate = 10
        fix_parts.append(f'Time: {time_estimate}min')

        fix_strategy = ' | '.join(fix_parts)

        # Build problem string (summarize all error types in group)
        error_types = set(p.get('issue_type', 'unknown') for p in group_problems)
        if len(error_types) == 1:
            error_code = list(error_types)[0]
        else:
            error_code = f'{len(error_types)} error types'

        what_broke = template.get('what_broke', 'Validation failed')
        problem_str = f'{what_broke} [{error_code}]'

        # Build L2 problem
        l2_problem = {
            'problem_id': problem_id,
            'verification_cmd': template.get('validation_cmd', ''),
            'failure_type': template.get('problem_type', 'unknown').replace('_', ' '),
            'problem': problem_str,
            'root_cause': root_cause,
            'pattern_detected': pattern_detected,
            'files': all_files,
            'fix_strategy': fix_strategy,
        }

        l2_problems.append(l2_problem)
        problem_id += 1

    # Sort by problem type (primary first, then enablement, then consecutive)
    def problem_priority(p):
        failure_type = p.get('failure_type', '')
        if 'primary' in failure_type:
            return 0
        elif 'enablement' in failure_type:
            return 2
        else:
            return 1

    l2_problems.sort(key=problem_priority)

    # Reassign problem IDs after sorting
    for idx, prob in enumerate(l2_problems, 1):
        prob['problem_id'] = idx

    # Calculate total time
    total_time = sum(
        int(p['fix_strategy'].split('Time: ')[-1].split('min')[0])
        for p in l2_problems
        if 'Time: ' in p['fix_strategy']
    )

    print(f'    Output: {len(l2_problems)} problems (deduplicated)')
    print(f'    Total time estimate: {total_time} minutes')

    # Build L2 result
    l2_result = {
        'issue_id': str(issue_id),
        'repo': decomposed_result.get('repo', 'unknown'),
        'problems': l2_problems,
        'total_problems': len(l2_problems),
        'total_time_minutes': total_time,
    }

    return l2_result


def _save_to_memory_files(results: list[dict], output_dir: str):
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
    successful = [
        r for r in results if 'l1_file_level' in r or 'l2_repair_sequence' in r
    ]

    # 1. failure_memory.json - Flat array of ALL file-level problems (L1)
    all_l1_problems = []
    for result in successful:
        l1_problems = result.get('l1_file_level', [])
        all_l1_problems.extend(l1_problems)

    # 2. l2_repair_sequences.json - Simplified format
    l2_sequences = []
    for result in successful:
        l2_data = result.get('l2_repair_sequence', {})

        # Fix failure_type underscores in problems
        l2_problems = []
        for prob in l2_data.get('problems', []):
            prob['failure_type'] = prob.get('failure_type', '').replace('_', ' ')
            l2_problems.append(prob)

        # Get workflow path and issue ID
        workflow_path = result.get('workflow_path', 'unknown')
        issue_id = result.get('original_issue_id') or result.get('issue_id', 'unknown')

        l2_sequences.append(
            {
                'issue_id': issue_id,  # Identify which issue this repair sequence is for
                'repo': result.get('repo'),
                'workflow': workflow_path,
                'problems': l2_problems,
            }
        )

    # Save memory files
    with open(output_dir / 'failure_memory.json', 'w') as f:
        json.dump(all_l1_problems, f, indent=2)

    with open(output_dir / 'repo_memory.json', 'w') as f:
        json.dump(l2_sequences, f, indent=2)

    # L3 analysis - merge all patterns from all issues
    all_l3_patterns = []
    for result in successful:
        l3_patterns = result.get('l3_analysis', [])
        if isinstance(l3_patterns, list):
            all_l3_patterns.extend(l3_patterns)

    with open(output_dir / 'cross_memory.json', 'w') as f:
        json.dump(all_l3_patterns, f, indent=2)

    print(f'  -> Saved to {output_dir}/')
    print(f'     - failure_memory.json ({len(all_l1_problems)} file problems)')
    print(f'     - repo_memory.json ({len(l2_sequences)} issues)')
    print(f'     - cross_memory.json ({len(all_l3_patterns)} patterns)')


def load_issues_from_huggingface(issue_ids: list[str] = None) -> list[dict[str, Any]]:
    """
    Load issues directly from HuggingFace dataset.

    Args:
        issue_ids: Optional list of issue IDs to filter. If None, loads all issues.

    Returns:
        List of issues matching the provided IDs (or all if no IDs specified)
    """
    print('Loading dataset from HuggingFace: ci-benchmark-user/ci-repair-bench')

    # Load with verification disabled to bypass feature compatibility issues
    try:
        # Delete cached dataset info to force reload
        from pathlib import Path

        cache_dir = (
            Path.home()
            / '.cache'
            / 'huggingface'
            / 'datasets'
            / 'ci-benchmark-user___ci-repair-bench'
        )
        if cache_dir.exists():
            info_file = cache_dir / 'default' / '0.0.0' / 'dataset_info.json'
            if info_file.exists():
                print('Removing cached dataset_info.json to force reload...')
                info_file.unlink()

        # Load without verification
        ds = load_dataset(
            'ci-benchmark-user/ci-repair-bench',
            verification_mode='no_checks',
            download_mode='reuse_cache_if_exists',
        )
        data = ds['train']
        print(f'Loaded {len(data)} issues from HuggingFace')

    except Exception as e:
        print(f'Dataset loading failed: {e}')
        raise RuntimeError(f'Could not load dataset from HuggingFace: {e}')

    if issue_ids:
        # Convert to set for faster lookup
        issue_ids_set = set(str(id) for id in issue_ids)

        # Filter to only requested IDs
        issues = []
        for item in data:
            if str(item.get('id')) in issue_ids_set:
                issues.append(dict(item))

        print(f'Filtered to {len(issues)} issues matching provided IDs')

        # Warn about missing IDs
        found_ids = set(str(i.get('id')) for i in issues)
        missing_ids = issue_ids_set - found_ids
        if missing_ids:
            print(
                f'WARNING: {len(missing_ids)} IDs not found in dataset: {sorted(missing_ids)}'
            )
    else:
        # Load all issues
        issues = [dict(item) for item in data]
        print(f'Loaded all {len(issues)} issues from dataset')

    return issues


def main():
    parser = argparse.ArgumentParser(
        description='Reverse engineer CI failures into atomic problems (visible + hidden)'
    )
    parser.add_argument('--issue-id', help='Single issue ID to decompose')
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Decompose all memory issues from HuggingFace',
    )
    parser.add_argument(
        '--use-huggingface',
        action='store_true',
        help='Load from HuggingFace instead of local JSON',
    )
    parser.add_argument(
        '--dataset', help='Path to JSONL dataset file (filtered issues)'
    )
    parser.add_argument(
        '--eval-issues',
        default='data/trs/eval_issues.json',
        help='Path to eval issues (legacy mode)',
    )
    parser.add_argument(
        '--output-dir', default='data/trs', help='Output directory for memory files'
    )
    parser.add_argument(
        '--model',
        default=DEFAULT_MINIMAX_MODEL,
        help='LLM model or alias. Supported aliases include minimax2.5 and glm5.2.',
    )
    parser.add_argument('--limit', type=int, help='Limit number of issues to process')
    parser.add_argument(
        '--skip-memory',
        action='store_true',
        help='Skip building memory files (L1/L2/L3) - only save decomposed_issues.json. Use this when you plan to do similarity-based split later.',
    )
    args = parser.parse_args()

    # Determine loading mode
    if args.dataset:
        # Load from JSONL file (filtered issues)
        print(f'\n{"=" * 80}')
        print(f'Loading issues from JSONL file: {args.dataset}')
        print(f'{"=" * 80}')

        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f'ERROR Dataset file not found: {dataset_path}')
            return 1

        issues = []
        with open(dataset_path) as f:
            for line in f:
                if line.strip():
                    issues.append(json.loads(line))

        print(f'Loaded {len(issues)} issues from {dataset_path}')

    elif args.batch or args.use_huggingface or args.issue_id:
        # Load from HuggingFace
        print(f'\n{"=" * 80}')
        print('Loading issues from HuggingFace dataset')
        print(f'{"=" * 80}')

        if args.batch:
            # Load all memory issues
            print(f'Using MEMORY_ISSUE_IDS: {MEMORY_ISSUE_IDS}')
            issues = load_issues_from_huggingface(MEMORY_ISSUE_IDS)
        elif args.issue_id:
            # Load specific issue
            issues = load_issues_from_huggingface([args.issue_id])
            if not issues:
                print(f'ERROR Issue {args.issue_id} not found in HuggingFace dataset')
                return 1
        else:
            # Load all issues
            issues = load_issues_from_huggingface(None)
    else:
        # Legacy mode: Load from local JSON file
        print(f'\n{"=" * 80}')
        print(f'Loading issues from local file: {args.eval_issues}')
        print(f'{"=" * 80}')

        eval_path = Path(args.eval_issues)
        if not eval_path.exists():
            print(f'ERROR Eval issues not found: {eval_path}')
            print('TIP: Use --use-huggingface to load from HuggingFace instead')
            return 1

        with open(eval_path) as f:
            issues = json.load(f)

        print(f'Loaded {len(issues)} issues from {eval_path}')

    # Limit if requested
    if args.limit:
        issues = issues[: args.limit]
        print(f'Limited to first {args.limit} issues')

    # Initialize LLM
    print(f'\n{"=" * 80}')
    args.model = configure_model_environment(args.model) or args.model
    print(f'Initializing LLM: {args.model}')
    print(f'{"=" * 80}')
    llm = LitellmModel(model_name=args.model)

    # Prepare output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results if file exists (for resume capability)
    results = []
    errors = []
    processed_ids = set()
    l2_sequences_path = output_dir / 'l2_repair_sequences.json'
    decomposed_issues_path = output_dir / 'decomposed_issues.json'

    # Load existing decomposed issues (old format - need L1/L2/L3 conversion)
    decomposed_cache = {}
    if decomposed_issues_path.exists():
        try:
            with open(decomposed_issues_path) as f:
                existing_decomposed = json.load(f)
                if isinstance(existing_decomposed, list):
                    for item in existing_decomposed:
                        issue_id = str(
                            item.get('original_issue_id', item.get('issue_id'))
                        )
                        decomposed_cache[issue_id] = item
                    print(
                        f'Found {len(decomposed_cache)} decomposed issues (can reuse for L1/L2/L3)'
                    )
        except Exception as e:
            print(f'Warning: Could not load decomposed issues: {e}')

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
                        if 'issue_id' in r:
                            processed_ids.add(str(r['issue_id']))
                    print(
                        f'Loaded {len(results)} existing L1/L2/L3 results (will skip)'
                    )
        except Exception as e:
            print(f'Warning: Could not load existing results: {e}')

    # Decompose issues with incremental saving
    for i, issue in enumerate(issues, 1):
        issue_id = str(issue.get('id'))

        # Skip if already in L1/L2/L3 format
        if issue_id in processed_ids:
            print(
                f'\nProgress: {i}/{len(issues)} - Issue {issue_id} already has L1/L2/L3, skipping'
            )
            continue

        print(f'\nProgress: {i}/{len(issues)}')

        # Check if we have decomposed data (can skip decomposition)
        if issue_id in decomposed_cache:
            print(
                '  Found in decomposed_issues.json - Building L1/L2/L3 directly (no decomposition needed)'
            )
            decomposed_result = decomposed_cache[issue_id]
        else:
            # Need to decompose from scratch
            print('  Not found in cache - Running full decomposition...')
            decomposed_result = decompose_issue(issue, llm)

            # Save decomposed result to cache immediately
            decomposed_cache[issue_id] = decomposed_result

            # Save decomposed_issues.json after each decomposition
            decomposed_list = list(decomposed_cache.values())
            with open(decomposed_issues_path, 'w') as f:
                json.dump(decomposed_list, f, indent=2)
            print(
                f'  OK Saved to decomposed_issues.json ({len(decomposed_list)} issues)'
            )

        # Check for errors
        if 'error' in decomposed_result:
            errors.append(decomposed_result)
            results.append(decomposed_result)
        else:
            # Run full L1/L2/L3 pipeline (unless --skip-memory)
            if not args.skip_memory:
                result = generate_l1_l2_l3_pipeline(decomposed_result, llm)
                results.append(result)
            else:
                # Just save decomposed result without L1/L2/L3
                results.append(decomposed_result)

        # Incremental save after each issue - save to 3 memory files (unless --skip-memory)
        if not args.skip_memory:
            try:
                _save_to_memory_files(results, args.output_dir)
                print(f'  OK Saved progress ({len(results)} issues total)')
            except Exception as e:
                print(f'  WARNING: Could not save progress: {e}')

    # Final save - save to 3 memory files (unless --skip-memory)
    if not args.skip_memory:
        _save_to_memory_files(results, args.output_dir)
    else:
        print(f'\n{"=" * 80}')
        print('Skipping memory file generation (--skip-memory flag)')
        print(f'{"=" * 80}')
        print('Only decomposed_issues.json was saved.')
        print(
            'Use this file with prepare_memory_train_test_split.py for similarity-based split.'
        )

    # Summary statistics
    print(f'\n{"=" * 80}')
    print('SUMMARY')
    print(f'{"=" * 80}')
    print(f'Total issues processed: {len(results)}')
    print(f'Successful: {len(results) - len(errors)}')
    print(f'Errors: {len(errors)}')

    # Count problems
    successful = [r for r in results if 'total_problems' in r]
    total_problems = sum(r.get('total_problems', 0) for r in successful)
    visible_problems = sum(
        sum(1 for p in r.get('problems', []) if p.get('visibility') == 'visible_in_log')
        for r in successful
    )
    hidden_problems = total_problems - visible_problems

    print('\nAtomic problems identified:')
    print(f'  Total: {total_problems}')
    print(f'  Visible (in structured CI context): {visible_problems}')
    print(f'  Hidden (inferred): {hidden_problems}')

    if successful:
        avg_problems = total_problems / len(successful)
        print(f'  Average per issue: {avg_problems:.1f}')

    # Problem type distribution
    problem_types = {}
    for r in successful:
        for p in r.get('problems', []):
            ptype = p.get('problem_type', 'unknown')
            problem_types[ptype] = problem_types.get(ptype, 0) + 1

    if problem_types:
        print('\nProblem type distribution:')
        for ptype, count in sorted(
            problem_types.items(), key=lambda x: x[1], reverse=True
        ):
            print(f'  {ptype}: {count}')

    print(f'\nOutput saved to: {output_dir}/')

    if errors:
        print(f'\nWARNING:  {len(errors)} issues had errors')
        print(
            f'Issue IDs with errors: {[e.get("original_issue_id") for e in errors[:5]]}'
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
