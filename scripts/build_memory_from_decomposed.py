#!/usr/bin/env python3
"""
build_memory_from_decomposed.py - Build L1/L2/L3 Memory from Decomposed Problems
==================================================================================

Takes decomposed_issues.json (output from decompose_ci_failure.py) and builds:
- L1 (per-file memory with reasoning)
- L2 (per-issue with atomic problems)
- L3 (cross-repo with hierarchical abstraction)

Structure:
  L1: File-level (within-repo)
      - Each file with failure + fix + reasoning
      - Caller/callee dependencies

  L2: Issue-level (within-repo)
      - Multiple atomic problems per issue
      - Each with visibility (visible_in_log vs hidden)
      - Dependencies between problems
      - Repair trajectory inference

  L3: Universal principles (cross-repo)
      - Hierarchical abstraction (3 levels)
      - Evidence from multiple L2 problems

Usage:
    python scripts/build_memory_from_decomposed.py \\
        --decomposed data/trs/decomposed_issues.json \\
        --output-dir data/trs_memory_v2
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import litellm
from dotenv import load_dotenv

try:
    import demjson3  # type: ignore
except Exception:
    demjson3 = None  # type: ignore

load_dotenv(PROJECT_ROOT / ".env", override=False)
LOGGER = logging.getLogger(__name__)

if not os.getenv("OPENROUTER_API_KEY") and os.getenv("MINIMAX_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("MINIMAX_API_KEY", "")
if not os.getenv("OPENROUTER_BASE_URL") and os.getenv("MINIMAX_BASE_URL"):
    os.environ["OPENROUTER_BASE_URL"] = os.getenv("MINIMAX_BASE_URL", "")


class LitellmModel:
    """Small invoke-compatible wrapper for memory abstraction scripts."""

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

    def invoke(self, prompt: str):
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        class Result:
            content = response.choices[0].message.content or ""

        return Result()


def _search_document(level: str, fields: Dict[str, Any]) -> str:
    parts = [f"level: {level}"]
    for key, value in fields.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple)):
            value = " | ".join(
                json.dumps(x, sort_keys=True, ensure_ascii=False) if isinstance(x, dict) else str(x)
                for x in value
                if x not in (None, "")
            )
        elif isinstance(value, dict):
            value = json.dumps(value, sort_keys=True, ensure_ascii=False)
        parts.append(f"{key}: {value}")
    return "\n".join(parts)


def _extract_validation_sequence(issue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the CI validation/verification sequence from any supported decomposed schema."""
    benchmark_context = issue.get("benchmark_ci_context") or {}
    workflow_context = benchmark_context.get("workflow_validation_context") or {}
    candidates = (
        issue.get("verification_sequence"),
        issue.get("validation_sequence"),
        benchmark_context.get("validation_sequence"),
        workflow_context.get("validation_sequence"),
    )
    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            return [row for row in candidate if isinstance(row, dict)]
    return []


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _clean_malformed_json(content: str) -> str:
    """
    Clean common LLM JSON errors:
    - Trailing commas: {"key": "value",}
    - Missing commas: {"a": 1 "b": 2}
    - Extra commas: {"a": 1,, "b": 2}
    - Unquoted keys/values in some cases
    """
    import re

    # Remove markdown fences if present
    content = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', content, flags=re.DOTALL)

    # Fix trailing commas before } or ]
    content = re.sub(r',(\s*[}\]])', r'\1', content)

    # Fix missing commas between string values (heuristic)
    content = re.sub(r'"\s+"', '", "', content)

    # Fix double commas
    content = re.sub(r',\s*,', ',', content)

    # Fix missing commas between } and { or } and [
    content = re.sub(r'}\s*{', '}, {', content)
    content = re.sub(r'}\s*\[', '}, [', content)
    content = re.sub(r']\s*{', '], {', content)

    return content.strip()


def _load_llm_json(content: str) -> Any:
    """Parse raw LLM JSON. Prompts require no markdown fences."""
    content = str(content or "").strip()

    if not content:
        return []

    # First attempt: direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError as json_err:
        pass

    # Second attempt: clean and parse
    try:
        cleaned = _clean_malformed_json(content)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Third attempt: parse multiple consecutive JSON objects
    try:
        decoder = json.JSONDecoder()
        objects = []
        idx = 0
        # Try on cleaned content
        cleaned = _clean_malformed_json(content)

        while idx < len(cleaned):
            content_part = cleaned[idx:].lstrip()
            if not content_part:
                break
            try:
                obj, end = decoder.raw_decode(content_part)
                objects.append(obj)
                idx += len(cleaned[idx:]) - len(content_part) + end
            except json.JSONDecodeError:
                break

        if len(objects) > 1:
            LOGGER.warning(
                "LLM returned %d separate JSON objects instead of array; wrapping in array",
                len(objects)
            )
            return objects  # Return as array
        elif len(objects) == 1:
            return objects[0]
    except Exception:
        pass

    # Fourth attempt: demjson3 (lenient parser)
    demjson3_err: Any = "demjson3 is not installed"
    try:
        if demjson3 is not None:
            cleaned = _clean_malformed_json(content)
            return demjson3.decode(cleaned)
    except Exception as exc:
        demjson3_err = exc

    # Final fallback: try to extract JSON from text
    try:
        # Look for JSON-like structures
        import re
        # Find outermost { } or [ ]
        for pattern in [r'\{.*\}', r'\[.*\]']:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(0)
                cleaned = _clean_malformed_json(extracted)
                try:
                    return json.loads(cleaned)
                except:
                    if demjson3 is not None:
                        try:
                            return demjson3.decode(cleaned)
                        except:
                            pass
    except Exception:
        pass

    parse_error = ValueError(
        f"JSON parse failed after all attempts. First 200 chars: {content[:200]}"
    )
    LOGGER.warning("%s", parse_error)
    return []


def _record_key(record: Dict[str, Any], level: str) -> str:
    if level == "L1":
        return "|".join(str(record.get(k) or "") for k in ("issue_id", "atomic_problem_id", "file"))
    if level == "L2":
        return str(record.get("issue_id") or record.get("sha_fail") or "")
    if level == "L3":
        return str(record.get("principle_id") or record.get("pattern_id") or record.get("problem_type") or "")
    return json.dumps(record, sort_keys=True)


def _merge_skip_existing(
    existing: List[Dict[str, Any]],
    new_records: List[Dict[str, Any]],
    *,
    level: str,
) -> List[Dict[str, Any]]:
    keys = {_record_key(record, level) for record in existing}
    merged = list(existing)
    added = 0
    skipped = 0
    for record in new_records:
        key = _record_key(record, level)
        if key and key in keys:
            skipped += 1
            continue
        merged.append(record)
        keys.add(key)
        added += 1
    print(f"  {level}: added {added}, skipped existing {skipped}")
    return merged


def _build_l1_with_llm(issue: Dict, problem: Dict, llm) -> List[Dict]:
    """
    Use LLM to analyze decomposed problem and create clean L1 entries.

    Returns one L1 entry per affected file with clean structure.
    """
    issue_id = issue.get("original_issue_id")
    repo = issue.get("repo")
    workflow_path = issue.get("benchmark_ci_context", {}).get("workflow_path", "")

    # Extract problem data
    problem_id = problem.get("order") or problem.get("problem_id")
    issue_type = problem.get("issue_type") or problem.get("problem_type")
    ci_cmd = problem.get("ci_validation") or problem.get("ci_command", "")
    affected_files = problem.get("affected_files", [])
    evidence = problem.get("evidence", {})

    # Prepare compact input for LLM
    files_summary = []
    if affected_files and isinstance(affected_files[0], dict):
        for f in affected_files:
            files_summary.append({
                "file": f.get("file", ""),
                "role": f.get("role", ""),
                "what_wrong": f.get("what_is_wrong", ""),
                "why_wrong": f.get("why_this_file_is_part_of_problem", ""),
                "how_fix": f.get("how_to_fix", ""),
                "dependent_files": f.get("dependent_files", [])
            })
    else:
        for f in affected_files:
            files_summary.append({"file": f, "role": "primary", "what_wrong": "", "why_wrong": "", "how_fix": "", "dependent_files": []})

    prompt = f"""Create compact L1 memory entries from this decomposed CI problem.

Use ONLY the problem data below. Do not invent line numbers, code snippets, files, dependencies, or commands.
Create one entry per relevant affected file. If many files share the same exact fix pattern, you may create one pattern entry whose "file" is a concise glob/path group.

REPO: {repo}
ISSUE ID: {issue_id}
WORKFLOW: {workflow_path}

PROBLEM DETAILS:
- Type: {issue_type}
- Failed command: {ci_cmd}
- Problem description: {problem.get("problem") or problem.get("symptom", "")}
- Root cause: {problem.get("why") or problem.get("root_cause", "")}
- Fix summary: {problem.get("how_fixed") or problem.get("fix_strategy") or ""}
- Verification: {problem.get("verification_after_fix") or ci_cmd}
- CI log evidence: {evidence.get("from_ci_log", "") if isinstance(evidence, dict) else problem.get("evidence_in_ci_log", "")}

FILES AFFECTED:
{json.dumps(files_summary, indent=2)}

For each entry:
- "problem": concise explanation of the file-level failure/risk and root cause.
- "fix_strategy": actionable summary of what needs to be modified and how to verify it.
- "dependent_files": include only dependencies explicitly present in FILES AFFECTED.

### Output Rules (STRICT)
- Output MUST be a single raw JSON array.
- Do NOT wrap the JSON in markdown code fences.
- Use double quotes for every JSON key and string value.
- Do not emit trailing commas.
- Do NOT add any text before or after the JSON.

[
  {{
    "memory_level": "L1",
    "file": "path/to/file.py",
    "repo": "{repo}",
    "workflow_path": "{workflow_path}",
    "issue_type": "{issue_type}",
    "failed_cmd": "{ci_cmd}",
    "problem": "Concise file-level failure/risk and root cause.",
    "fix_strategy": "Actionable modification strategy and verification command.",
    "dependent_files": [
      {{"file": "related/file.py", "relationship": "why related"}}
    ]
  }}
]
"""

    # Call LLM
    try:
        result = llm.invoke(prompt)
        response = getattr(result, "content", str(result)).strip()
        l1_entries = _load_llm_json(response)

        if not l1_entries:
            print(f"Empty L1 JSON for issue {issue_id} problem {problem_id}; skipping")
            return []
        if not isinstance(l1_entries, list):
            print(f"LLM returned non-list L1 JSON for issue {issue_id} problem {problem_id}; skipping")
            return []

        return l1_entries

    except Exception as e:
        print(f"LLM analysis failed for issue {issue_id} problem {problem_id}: {e}")
        # Fallback
        return [{
            "memory_level": "L1",
            "file": f_summary['file'],
            "repo": repo,
            "workflow_path": workflow_path,
            "issue_type": issue_type,
            "failed_cmd": ci_cmd,
            "problem": f"{issue_type} in {f_summary['file']}. {f_summary.get('what_wrong', '')}",
            "fix_strategy": f_summary.get('how_fix', 'See diff'),
            "dependent_files": f_summary.get('dependent_files', [])
        } for f_summary in files_summary]


def build_l1_memory(decomposed_issues: List[Dict], llm=None) -> List[Dict]:
    """
    Build L1 (per-file) memory from decomposed atomic problems using LLM.

    Clean structure:
    - memory_level, file, repo, workflow_path
    - issue_type, failed_cmd
    - problem (complete statement in one string)
    - fix_strategy (complete fix in one string)
    - diff_evidence
    - dependent_files (if any)
    """

    l1_memories = []

    for issue in decomposed_issues:
        if "error" in issue:
            continue

        # Support both "atomic_problems" (new) and "problems" (old)
        problems = issue.get("atomic_problems") or issue.get("problems", [])

        for problem in problems:
            # Use LLM to build clean L1 entries
            problem_l1_entries = _build_l1_with_llm(issue, problem, llm)
            l1_memories.extend(problem_l1_entries)

    print(f"Built {len(l1_memories)} L1 (per-file) memory entries")
    return l1_memories


def _build_l2_with_llm(issue: Dict, llm) -> Dict:
    """
    Use LLM to analyze decompose results and create clean L2 structure.

    Simple, production-ready L2 with atomic_problems based on CI workflow order.
    """
    issue_id = issue.get("original_issue_id")
    repo = issue.get("repo")
    problems = issue.get("atomic_problems") or issue.get("problems", [])
    trajectory = issue.get("trajectory_summary", {})
    validation_sequence = _extract_validation_sequence(issue)

    # Prepare compact input for LLM
    problems_summary = []
    for i, p in enumerate(problems, 1):
        evidence = p.get("evidence", {})
        affected_files = p.get("affected_files", [])

        # Extract file changes
        file_changes = []
        if affected_files and isinstance(affected_files[0], dict):
            for f in affected_files:
                # Get fix from multiple possible field names in decomposed data
                fix_info = (
                    f.get("how_to_fix") or
                    f.get("change_made") or
                    f.get("fix_implementation") or
                    ""
                )
                file_changes.append({
                    "file": f.get("file", ""),
                    "what_wrong": f.get("what_is_wrong") or f.get("failure_or_risk_in_this_file") or "",
                    "how_fix": fix_info,
                    "diff": f.get("diff_evidence") or f.get("diff") or ""
                })
        else:
            file_changes = [{"file": f, "what_wrong": "", "how_fix": "", "diff": ""} for f in affected_files]

        # Get CI stage directly from decomposed data (no speculation)
        ci_stage = (
            p.get("would_fail_at_stage") or
            p.get("ci_workflow_step") or
            (p.get("workflow_validation", {}).get("validates", "").lower() if isinstance(p.get("workflow_validation"), dict) else "") or
            "unknown"
        )

        problems_summary.append({
            "order": i,
            "visibility": p.get("visibility"),
            "type": p.get("issue_type") or p.get("problem_type"),
            "ci_stage": ci_stage,  # Direct from data, no speculation
            "ci_cmd": p.get("ci_validation") or p.get("ci_command"),
            "problem_desc": p.get("problem") or p.get("symptom"),
            "why": p.get("why") or p.get("root_cause"),
            "ci_log_evidence": evidence.get("from_ci_log") if isinstance(evidence, dict) else p.get("evidence_in_ci_log"),
            "diff_evidence": evidence.get("from_diff") if isinstance(evidence, dict) else p.get("diff_evidence"),
            "file_changes": file_changes,
            "why_hidden": p.get("why_hidden", "")
        })

    prompt = f"""Analyze this CI failure decomposition and create a clean L2 memory structure.

Create a compact issue-level L2 memory record.
Use ONLY the provided decomposition. Do not invent files, commands, line numbers, or extra failures.
Keep text concise and actionable.

REPO: {repo}
ISSUE ID: {issue_id}

VALIDATION SEQUENCE:
{json.dumps(validation_sequence, indent=2)}

ATOMIC PROBLEMS (from decomposition):
{json.dumps(problems_summary, indent=2)}

TRAJECTORY SUMMARY:
{json.dumps(trajectory, indent=2)}

Return STRICT JSON only. No markdown, no code fences, no extra text.
{{
  "atomic_problems": [
    {{
      "problem_id": 1,
      "issue_type": "short dynamic problem type",
      "failed_cmd": "exact CI command that failed",
      "problem": "Concise issue-level problem and root cause.",
      "file_changes": [
        {{
          "file": "path/to/file1.py",
          "fix": "Actionable fix strategy for this file."
        }}
      ]
    }}
  ],
  "repair_trajectory_summary": "Concise repair order through the CI validation sequence."
}}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        llm_result = _load_llm_json(content)
        if not llm_result:
            print(f"  Warning: empty L2 JSON for {issue_id}; skipping")
            return {}
        return llm_result
    except Exception as e:
        print(f"  Warning: LLM analysis failed for {issue_id}: {e}")
        # Fallback to simple structure

        return {
            "atomic_problems": [
                {
                    "problem_id": i,
                    "issue_type": p.get("type", ""),
                    "failed_cmd": p.get("ci_cmd", ""),
                    "problem": p.get("problem_desc", ""),
                    "file_changes": [
                        {"file": fc.get("file", ""), "fix": fc.get("how_fix", "")}
                        for fc in p.get("file_changes", [])
                    ]
                }
                for i, p in enumerate(problems_summary, 1)
            ],
            "repair_trajectory_summary": trajectory.get("repair_sequence", "")
        }


def build_l2_memory(decomposed_issues: List[Dict], llm=None) -> List[Dict]:
    """
    Build L2 (per-issue) memory from decomposed problems.

    Uses LLM to analyze and create clean, simple L2 structure.
    Each L2 entry contains atomic_problems ordered by CI workflow validation sequence.
    """
    l2_memories = []

    print(f"Building L2 memory for {len(decomposed_issues)} issues...")

    for idx, issue in enumerate(decomposed_issues, 1):
        if "error" in issue:
            continue

        issue_id = issue.get("original_issue_id")
        repo = issue.get("repo")
        workflow_path = issue.get("benchmark_ci_context", {}).get("workflow_path", "")
        validation_sequence = _extract_validation_sequence(issue)
        benchmark_context = issue.get("benchmark_ci_context", {}) or {}
        workflow_validation_context = benchmark_context.get("workflow_validation_context", {}) or {}
        dependent_files = workflow_validation_context.get("dependent_files", []) if isinstance(workflow_validation_context, dict) else []

        print(f"  [{idx}/{len(decomposed_issues)}] Analyzing {repo} / {issue_id}...")

        # Use LLM to create clean L2 structure
        if llm:
            llm_result = _build_l2_with_llm(issue, llm)
            if not llm_result or not llm_result.get("atomic_problems"):
                print(f"  Warning: no usable L2 JSON for {issue_id}; skipping")
                continue
        else:
            # Fallback: use decompose output directly
            problems = issue.get("atomic_problems") or issue.get("problems", [])
            trajectory = issue.get("trajectory_summary", {})
            llm_result = {
                "atomic_problems": [
                    {
                        "problem_id": i,
                        "issue_type": p.get("issue_type") or p.get("problem_type", ""),
                        "failed_cmd": p.get("ci_validation") or p.get("ci_command", ""),
                        "problem": p.get("problem") or p.get("symptom", ""),
                        "file_changes": [
                            {
                                "file": f.get("file") if isinstance(f, dict) else f,
                                "fix": (
                                    f.get("how_to_fix") or
                                    f.get("change_made") or
                                    f.get("fix_implementation") or
                                    ""
                                ) if isinstance(f, dict) else ""
                            }
                            for f in (p.get("affected_files", []))
                        ]
                    }
                    for i, p in enumerate(problems, 1)
                ],
                "repair_trajectory_summary": trajectory.get("repair_sequence", "")
            }

        # Build clean L2 structure
        l2_memory = {
            "id": f"{repo.replace('/', '_')}_{issue_id}",
            "issue_id": issue_id,
            "sha_fail": issue.get("sha_fail", ""),
            "repo": repo,
            "workflow_path": workflow_path,
            "atomic_problems": llm_result.get("atomic_problems", []),
            "repair_trajectory_summary": llm_result.get("repair_trajectory_summary", ""),
            "dependent_files": dependent_files,
        }

        # Add search document for retrieval
        l2_memory["search_document"] = _search_document("L2", {
            "repo": repo,
            "issue_id": issue_id,
            "sha_fail": issue.get("sha_fail", ""),
            "workflow_path": workflow_path,
            "overall_failure_summary": issue.get("overall_failure_summary", ""),
            "workflow_reasoning": issue.get("workflow_reasoning", ""),
            "atomic_problems": llm_result.get("atomic_problems", []),
            "workflow_ordered_problem_flow": issue.get("workflow_ordered_problem_flow", []),
            "repair_trajectory": llm_result.get("repair_trajectory_summary", ""),
            "dependent_files": dependent_files,
        })

        l2_memories.append(l2_memory)

    print(f"Built {len(l2_memories)} L2 (per-issue) memory entries")
    return l2_memories




def _build_l3_with_llm(atomic_problem: Dict, llm) -> Dict:
    """
    Build ONE universal L3 pattern from ONE L2 atomic problem.

    L3 must be applicable to ANY Python project that encounters this pattern.
    Each atomic problem gets its own distinct L3 entry.
    """

    issue_type = atomic_problem.get("issue_type", "unknown")

    prompt = f"""You are analyzing a CI failure to create a UNIVERSAL repair pattern applicable to ANY Python project.

ISSUE TYPE: {issue_type}

L2 ATOMIC PROBLEM DATA (from real CI failure):
{json.dumps(atomic_problem, indent=2)}

YOUR TASK:
Analyze the L2 data deeply and extract a universal pattern that can help ANY Python project detect and fix this type of failure.

CRITICAL ANALYSIS REQUIREMENTS:

1. **ROOT CAUSE ANALYSIS**:
   - WHY does this failure occur at a fundamental level?
   - What ecosystem changes, constraints, or conditions trigger it?
   - Is it environment-specific, dependency-related, type system issue, etc.?

2. **FAILURE CONDITIONS**:
   - UNDER WHAT CONDITIONS does this failure appear?
   - What constraints or project states cause it?
   - When does it NOT occur? (important for understanding boundaries)

3. **FAILURE PATTERN DETECTION**:
   - What are the SPECIFIC error signals in CI logs?
   - Which files/configurations should be checked?
   - What hidden indicators suggest this problem exists?

4. **UNIVERSAL VALIDATION COMMAND**:
   - What command can ANY Python project run to verify this?
   - Generalize from examples (e.g., "uv install" → "pip install" or "poetry install")
   - Command should work across different tools/environments

5. **ACTIONABLE FIX STRATEGY**:
   - WHAT needs to be verified before fixing?
   - WHERE to apply the fix (which files, which sections)?
   - WHAT KIND of fix (version update, constraint change, code refactor)?
   - HOW does the fix address the root cause?
   - Clear step-by-step instructions

OUTPUT STRICT JSON (no markdown, no fences, no extra text):
{{
  "pattern_id": "descriptive_snake_case_id",
  "pattern_name": "Human Readable Pattern Name",
  "issue_type": "{issue_type}",
  "validation_cmd": "universal command to verify (e.g., 'pip install -r requirements.txt' or 'mypy .')",
  "problem": "Complete problem statement combining: ROOT CAUSE (WHY this happens fundamentally), CONDITIONS (WHEN and under what constraints), and FAILURE PATTERN (which cases trigger it, why it manifests, what constraints cause it). Make this a comprehensive paragraph explaining the universal problem.",
  "detection": {{
    "error_signals": ["Specific error message pattern 1", "Specific error pattern 2"],
    "where_to_check": ["File or config location 1", "File or config location 2"],
    "hidden_indicators": ["Indicator that suggests this problem", "Another subtle signal"]
  }},
  "universal_fix_strategy": "Complete fix strategy combining: WHAT TO VERIFY (checks before fixing), WHERE TO FIX (specific files/sections), WHAT KIND OF FIX (type of change), HOW IT WORKS (why it resolves root cause), and STEPS (numbered actionable steps). Make this a comprehensive paragraph with all fix information.",
  "applicability": "Complete applicability statement: Which Python projects this applies to and which it does NOT apply to. Be specific about constraints, tools, or conditions."
}}

REMEMBER: This pattern will be used by OTHER projects you've never seen. Make it universal, logical, and actionable.
"""

    try:
        result = llm.invoke(prompt)
        response = getattr(result, "content", str(result)).strip()

        # Extract JSON (handle cases where LLM adds explanation)
        first_brace = response.find("{")
        last_brace = response.rfind("}")
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            response = response[first_brace:last_brace + 1]

        return _load_llm_json(response)

    except Exception as e:
        print(f"⚠️  LLM L3 analysis failed for {issue_type}: {e}")
        print(f"   Response preview: {response[:500] if 'response' in locals() else 'N/A'}")

        # Fallback structure with flat strings
        failed_cmd = atomic_problem.get("failed_cmd") or atomic_problem.get("ci_command") or ""

        return {
            "pattern_id": f"{issue_type}_pattern",
            "pattern_name": issue_type.replace("_", " ").title(),
            "issue_type": issue_type,
            "validation_cmd": failed_cmd,
            "problem": f"Universal {issue_type} pattern. Analyze root cause, conditions, and failure patterns.",
            "detection": {
                "error_signals": [],
                "where_to_check": [],
                "hidden_indicators": []
            },
            "universal_fix_strategy": "Analyze the problem, verify conditions, identify fix location, apply appropriate fix, and verify resolution.",
            "applicability": "Python projects experiencing this issue type."
        }

def _llm_cluster_atomic_problems(problems: List[Dict], llm) -> List[List[Dict]]:
    """
    Use LLM to cluster similar atomic problems for more abstract L3 patterns.

    Returns: List of clusters, where each cluster is a list of similar problems.
    """
    if len(problems) <= 1:
        return [problems]  # No clustering needed

    # Prepare compact summary for LLM
    problems_summary = []
    for i, p in enumerate(problems):
        problems_summary.append({
            "id": i,
            "issue_type": p.get("issue_type", "unknown"),
            "ci_stage": p.get("ci_stage", ""),
            "failed_cmd": p.get("failed_cmd", ""),
            "symptom": p.get("symptom", "")[:150],  # Clip for token efficiency
            "root_cause": p.get("root_cause", "")[:150],
        })

    prompt = f"""Cluster these {len(problems)} atomic problems by similarity.

PROBLEMS:
{json.dumps(problems_summary, indent=2)}

Group problems that have:
- Similar root causes
- Similar fix strategies
- Similar failure conditions

Return STRICT JSON (no markdown):
{{
  "clusters": [
    {{"problem_ids": [0, 2], "similarity_reason": "Both version constraint issues"}},
    {{"problem_ids": [1], "similarity_reason": "Standalone type annotation issue"}}
  ]
}}
"""

    try:
        result = llm.invoke(prompt)
        response = getattr(result, "content", str(result)).strip()

        # Extract JSON
        first_brace = response.find("{")
        last_brace = response.rfind("}")
        if first_brace != -1 and last_brace != -1:
            response = response[first_brace:last_brace + 1]

        cluster_data = _load_llm_json(response)
        clusters_info = cluster_data.get("clusters", [])

        # Build actual problem clusters
        clusters = []
        for c_info in clusters_info:
            cluster_ids = c_info.get("problem_ids", [])
            cluster_problems = [problems[i] for i in cluster_ids if i < len(problems)]
            if cluster_problems:
                clusters.append(cluster_problems)

        return clusters if clusters else [problems]

    except Exception as e:
        print(f"⚠️  LLM clustering failed: {e}, falling back to individual L3s")
        return [[p] for p in problems]  # Each problem in its own cluster


def _build_l3_from_cluster(cluster: List[Dict], llm) -> Dict:
    """
    Build ONE L3 pattern from a CLUSTER of similar atomic problems.
    More abstract than single-problem L3.
    """
    if len(cluster) == 1:
        # Single problem - use existing logic
        return _build_l3_with_llm(cluster[0], llm)

    # Multiple similar problems - create more abstract pattern
    issue_type = cluster[0].get("issue_type", "unknown")

    # Collect all symptoms, root causes, and file changes
    all_symptoms = [p.get("symptom", "") for p in cluster if p.get("symptom")]
    all_root_causes = [p.get("root_cause", "") for p in cluster if p.get("root_cause")]
    all_affected_files = []
    for p in cluster:
        all_affected_files.extend(p.get("affected_files", []))

    prompt = f"""You are analyzing {len(cluster)} SIMILAR atomic problems to create ONE universal repair pattern.

CLUSTER OF SIMILAR PROBLEMS:
{json.dumps(cluster, indent=2)[:3000]}

Your task: Create ONE abstract L3 pattern that covers ALL these similar problems.

Answer these 5 questions with DEEP REASONING:

1. **UNIVERSAL ROOT CAUSE**:
   - What is the COMMON underlying cause across all {len(cluster)} problems?
   - Why does this fundamentally happen?

2. **GENERAL CONDITIONS**:
   - Under what conditions does this pattern appear?
   - What are the common constraints?

3. **DETECTION PATTERN**:
   - What error signals appear across these cases?
   - Where should we check?

4. **UNIVERSAL FIX STRATEGY**:
   - What is the GENERAL fix approach for this pattern?
   - How does it address the root cause?

5. **APPLICABILITY**:
   - Which projects does this pattern apply to?
   - Which projects it does NOT apply to?

OUTPUT STRICT JSON (no markdown):
{{
  "pattern_id": "descriptive_snake_case_id",
  "pattern_name": "Human Readable Pattern Name",
  "issue_type": "{issue_type}",
  "validation_cmd": "universal command to verify",
  "problem": "Universal problem statement covering all {len(cluster)} similar cases. ROOT CAUSE, CONDITIONS, PATTERN.",
  "detection": {{
    "error_signals": ["error pattern 1", "error pattern 2"],
    "where_to_check": ["file/config 1", "file/config 2"],
    "hidden_indicators": ["indicator 1", "indicator 2"]
  }},
  "universal_fix_strategy": "Universal fix strategy that works for all {len(cluster)} cases. VERIFY, FIX LOCATION, FIX TYPE, WHY IT WORKS, STEPS.",
  "applicability": "Which projects this applies to and which it does NOT."
}}
"""

    try:
        result = llm.invoke(prompt)
        response = getattr(result, "content", str(result)).strip()

        # Extract JSON
        first_brace = response.find("{")
        last_brace = response.rfind("}")
        if first_brace != -1 and last_brace != -1:
            response = response[first_brace:last_brace + 1]

        pattern = _load_llm_json(response)
        pattern["examples_count"] = len(cluster)  # Track how many problems this pattern covers
        return pattern

    except Exception as e:
        print(f"⚠️  LLM cluster analysis failed: {e}")
        # Fallback - use first problem in cluster
        return _build_l3_with_llm(cluster[0], llm)


def build_l3_memory(l2_memories: List[Dict], llm) -> List[Dict]:
    """
    Build L3 universal patterns from L2 atomic problems WITH CLUSTERING.

    Strategy:
    1. Collect all atomic problems
    2. Group by issue_type + ci_stage
    3. LLM clusters similar problems within each group
    4. Build more abstract L3 from each cluster

    Result: Fewer, more general L3 patterns that cover multiple similar problems.
    """

    print("Building L3 universal patterns with LLM clustering...")

    # Step 1: Collect all atomic problems with context
    all_problems = []
    for l2 in l2_memories:
        repo = l2.get("repo")
        issue_id = l2.get("id") or l2.get("issue_id")
        workflow_path = l2.get("workflow_path", "")

        for atomic_problem in l2.get("atomic_problems", []):
            issue_type = atomic_problem.get("issue_type", "unknown")
            if issue_type == "unknown" or not issue_type:
                continue

            enriched_problem = {
                **atomic_problem,
                "repo": repo,
                "issue_id": issue_id,
                "workflow_path": workflow_path,
            }
            all_problems.append(enriched_problem)

    print(f"  Collected {len(all_problems)} atomic problems")

    # Step 2: Group by issue_type + ci_stage
    by_type_stage = defaultdict(list)
    for p in all_problems:
        key = f"{p.get('issue_type')}::{p.get('ci_stage', 'unknown')}"
        by_type_stage[key].append(p)

    print(f"  Grouped into {len(by_type_stage)} type-stage groups")

    # Step 3: Cluster within each group and build L3
    l3_patterns = []
    for type_stage, problems in by_type_stage.items():
        print(f"  Processing {type_stage} ({len(problems)} problems)...")

        if len(problems) == 1:
            # Single problem - direct L3
            pattern = _build_l3_with_llm(problems[0], llm)
            if not pattern:
                print(f"    Warning: empty L3 JSON for {type_stage}; skipping")
                continue

            # Handle case where LLM returns array instead of object
            if isinstance(pattern, list):
                if len(pattern) == 1 and isinstance(pattern[0], dict):
                    pattern = pattern[0]  # Unwrap single-item array
                else:
                    print(f"    Warning: L3 LLM returned list with {len(pattern)} items for {type_stage}; using first or skipping")
                    if not pattern or not isinstance(pattern[0], dict):
                        continue
                    pattern = pattern[0]

            l3_entry = {
                "memory_level": "L3",
                "pattern_id": pattern.get("pattern_id", f"{problems[0].get('issue_type')}_single"),
                "pattern_name": pattern.get("pattern_name", ""),
                "issue_type": pattern.get("issue_type", ""),
                "validation_cmd": pattern.get("validation_cmd", ""),
                "problem": pattern.get("problem", ""),
                "detection": pattern.get("detection", {}),
                "universal_fix_strategy": pattern.get("universal_fix_strategy", ""),
                "applicability": pattern.get("applicability", ""),
                "examples_count": 1,
                "language": "Python",
                "source_repo": problems[0].get("repo", ""),
                "source_issue": problems[0].get("issue_id", ""),
            }
            l3_patterns.append(l3_entry)
        else:
            # Multiple problems - LLM cluster first
            clusters = _llm_cluster_atomic_problems(problems, llm)
            print(f"    → Clustered into {len(clusters)} pattern(s)")

            for cluster in clusters:
                pattern = _build_l3_from_cluster(cluster, llm)
                if not pattern:
                    print(f"    Warning: empty clustered L3 JSON for {type_stage}; skipping")
                    continue
                l3_entry = {
                    "memory_level": "L3",
                    "pattern_id": pattern.get("pattern_id", f"{cluster[0].get('issue_type')}_cluster"),
                    "pattern_name": pattern.get("pattern_name", ""),
                    "issue_type": pattern.get("issue_type", ""),
                    "validation_cmd": pattern.get("validation_cmd", ""),
                    "problem": pattern.get("problem", ""),
                    "detection": pattern.get("detection", {}),
                    "universal_fix_strategy": pattern.get("universal_fix_strategy", ""),
                    "applicability": pattern.get("applicability", ""),
                    "examples_count": pattern.get("examples_count", len(cluster)),
                    "language": "Python",
                    "source_repos": ", ".join(set(p.get("repo", "") for p in cluster)),
                    "source_issues": ", ".join(set(p.get("issue_id", "") for p in cluster)),
                }
                l3_patterns.append(l3_entry)

    print(f"Built {len(l3_patterns)} L3 universal patterns (covering {len(all_problems)} atomic problems)")
    return l3_patterns


def main():
    parser = argparse.ArgumentParser(
        description="Build L1/L2/L3 memory from decomposed issues"
    )
    parser.add_argument(
        "--decomposed",
        default="data/trs/decomposed_issues.json",
        help="Path to decomposed issues"
    )
    parser.add_argument(
        "--output-dir",
        default="data/trs",
        help="Output directory for L1/L2/L3 memory"
    )
    parser.add_argument(
        "--model",
        default="openrouter/minimax/minimax-m2.5",
        help="LLM model for L3 abstraction. Use openrouter/minimax/minimax-m2.5 for MiniMax M2.5 via OpenRouter."
    )
    parser.add_argument(
        "--append-skip-existing",
        action="store_true",
        help="Append to existing memory files and skip records with existing keys"
    )
    args = parser.parse_args()

    # Load decomposed issues
    decomposed_path = Path(args.decomposed)
    if not decomposed_path.exists():
        print(f"✗ Decomposed issues not found: {decomposed_path}")
        print(f"Run decompose_ci_failure.py first!")
        return 1

    print(f"{'='*80}")
    print(f"Building L1/L2/L3 Memory from Decomposed Issues")
    print(f"{'='*80}\n")

    with open(decomposed_path) as f:
        decomposed_issues = json.load(f)

    print(f"Loaded {len(decomposed_issues)} decomposed issues\n")

    # Build L1 (per-file)
    print("Step 1: Building L1 (per-file) memory with LLM...")
    # Initialize LLM for L1, L2 and L3 analysis
    llm = LitellmModel(model_name=args.model)

    l1_memories = build_l1_memory(decomposed_issues, llm)

    # Build L2 (per-issue with atomic problems) - uses LLM for clean structure
    print("\nStep 2: Building L2 (per-issue) memory with LLM analysis...")
    l2_memories = build_l2_memory(decomposed_issues, llm)

    # Build L3 (cross-repo with hierarchical abstraction)
    print("\nStep 3: Building L3 (cross-repo) memory with abstraction...")
    l3_principles = build_l3_memory(l2_memories, llm)

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.append_skip_existing:
        print("\nMerging with existing memory files (skip existing records)...")
        l1_memories = _merge_skip_existing(
            _load_json_list(output_dir / "failure_memory.json"),
            l1_memories,
            level="L1",
        )
        l2_memories = _merge_skip_existing(
            _load_json_list(output_dir / "repo_memory.json"),
            l2_memories,
            level="L2",
        )
        l3_principles = _merge_skip_existing(
            _load_json_list(output_dir / "cross_memory.json"),
            l3_principles,
            level="L3",
        )

    # Save L1 (failure_memory.json for compatibility)
    l1_path = output_dir / "failure_memory.json"
    with open(l1_path, "w") as f:
        json.dump(l1_memories, f, indent=2)
    print(f"\n✓ Saved L1 memory: {l1_path}")

    # Save L2 (repo_memory.json for compatibility)
    l2_path = output_dir / "repo_memory.json"
    with open(l2_path, "w") as f:
        json.dump(l2_memories, f, indent=2)
    print(f"✓ Saved L2 memory: {l2_path}")

    # Save L3 (cross_memory.json for compatibility)
    l3_path = output_dir / "cross_memory.json"
    with open(l3_path, "w") as f:
        json.dump(l3_principles, f, indent=2)
    print(f"✓ Saved L3 memory: {l3_path}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"L1 (per-file): {len(l1_memories)} entries")
    print(f"L2 (per-issue): {len(l2_memories)} entries")
    print(f"  Total atomic problems: {sum(len(l2.get('atomic_problems', [])) for l2 in l2_memories)}")
    print(f"L3 (cross-repo): {len(l3_principles)} principles")
    print(f"\nOutput directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
