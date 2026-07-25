#!/usr/bin/env python3
"""
trajectory_builder.py - Build and organize repair trajectory
"""

import json
from typing import Dict, List, Optional

try:
    from utilities.model_token_config import get_input_chunk_chars
except Exception:  # pragma: no cover - fallback for standalone use
    get_input_chunk_chars = None


def build_repair_trajectory(per_commit_analysis: List[Dict]) -> List[Dict]:
    """
    Build simple sequential repair trajectory from all commits

    Args:
        per_commit_analysis: List of commit analyses

    Returns:
        Flat list of all problems in sequence
    """

    problem_sequence = []

    for commit_analysis in per_commit_analysis:
        commit_sha = commit_analysis.get("commit_sha", "unknown")

        for problem in commit_analysis.get("problems", []):
            problem_entry = {
                "commit_sha": problem.get("commit_sha") or commit_sha,
                "commit_message": problem.get("commit_message", ""),
                "files": problem.get("files", []),
                "failure_type": problem.get("failure_type", ""),
                "issue_type": problem.get("issue_type", ""),
                "status": problem.get("status", ""),
                "problem": problem.get("problem", ""),
                "root_cause": problem.get("root_cause", ""),
                "changes_made": problem.get("changes_made", ""),
                "introduced": problem.get("introduced", bool(problem.get("introduces"))),
                "fixed": problem.get("fixed", bool(problem.get("fixes"))),
                "fix_strategy": problem.get("fix_strategy", ""),
                "why_this_fix_works": problem.get("why_this_fix_works", ""),
                "repair": problem.get("repair", problem.get("fixes", "")),
                "introduces": problem.get("introduces", ""),
                "fixes": problem.get("fixes", ""),
                "current_failed_jobs": problem.get("current_failed_jobs", []),
                "current_fixed_jobs": problem.get("current_fixed_jobs", []),
                "validation_cmd": problem.get("validation_cmd", ""),
                "group_context": problem.get("group_context", []),
                "commit": commit_sha,
                "sha_success": problem.get("sha_success"),
                "sha_fail": problem.get("sha_fail"),
            }

            problem_sequence.append(problem_entry)

    return problem_sequence


def organize_trajectory_with_llm(
    all_problems: List[Dict], validation_sequence: List[Dict], analyzer
) -> List[Dict]:
    """
    Group commit-level problem events by validation command and failure type,
    order those groups by CI validation order, then ask the LLM to synthesize
    final problem/repair records for each group.
    """

    if not all_problems:
        return []

    indexed_problems = []
    for idx, problem in enumerate(all_problems, 1):
        item = dict(problem)
        item["_input_index"] = idx
        indexed_problems.append(item)

    validation_order_by_cmd = _validation_order_by_cmd(validation_sequence)
    grouped_events = _group_problem_events(indexed_problems, validation_order_by_cmd)

    final_problems = []
    for group_idx, group in enumerate(grouped_events, 1):
        print(
            "      Final analysis group "
            f"{group_idx}/{len(grouped_events)}: "
            f"{group['validation_cmd'] or '<unknown cmd>'} / "
            f"{group['failure_type'] or '<unknown failure>'} "
            f"({len(group['events'])} event(s))"
        )
        final_problems.extend(
            _analyze_problem_group_with_llm(group, validation_sequence, analyzer)
        )

    return _normalize_final_problems(final_problems, validation_order_by_cmd)


def _validation_order_by_cmd(validation_sequence: List[Dict]) -> Dict[str, int]:
    """Map validation commands to their CI order."""
    order_by_cmd = {}
    for item in validation_sequence or []:
        order = item.get("order")
        if order is None:
            continue
        for field in ["validation_cmd", "installation_cmd"]:
            cmd = item.get(field, "")
            if cmd:
                order_by_cmd[cmd] = int(order)
    return order_by_cmd


def _group_problem_events(
    problems: List[Dict], validation_order_by_cmd: Dict[str, int]
) -> List[Dict]:
    """Cluster events by validation_cmd and failure_type, ordered by CI."""
    groups: Dict[tuple, List[Dict]] = {}
    for problem in problems:
        validation_cmd = problem.get("validation_cmd", "")
        failure_type = problem.get("failure_type", "")
        key = (
            validation_order_by_cmd.get(validation_cmd),
            validation_cmd,
            failure_type,
        )
        groups.setdefault(key, []).append(problem)

    def sort_key(pair: tuple) -> tuple:
        order, validation_cmd, failure_type = pair[0]
        normalized_order = order if order is not None else 9999
        return (normalized_order, validation_cmd, failure_type)

    ordered_groups = []
    for key, events in sorted(groups.items(), key=sort_key):
        validation_order, validation_cmd, failure_type = key
        ordered_groups.append(
            {
                "validation_order": validation_order,
                "validation_cmd": validation_cmd,
                "failure_type": failure_type,
                "events": events,
                "group_context": _collect_group_context(events),
            }
        )
    return ordered_groups


def _collect_group_context(events: List[Dict]) -> List[Dict]:
    """Collect commit-level validation/failure group context."""
    contexts = []
    for event in events:
        for context in event.get("group_context", []) or []:
            if not isinstance(context, dict):
                continue
            contexts.append(
                {
                    "commit_sha": event.get("commit_sha", ""),
                    "files": _files_from_context_group(context),
                    "validation_order": context.get("validation_order"),
                    "validation_cmd": context.get("validation_cmd", ""),
                    "failure_type": context.get("failure_type", ""),
                    "groups": context.get("groups", []),
                    "dependency_context": context.get("dependency_context", {}),
                }
            )
    return contexts


def _files_from_context_group(context: Dict) -> List[str]:
    """Return files from the compact validation/failure group context."""
    if context.get("files"):
        return context.get("files", [])
    files = []
    for subgroup in context.get("groups", []) or []:
        for file_path in subgroup.get("files", []) or []:
            if file_path not in files:
                files.append(file_path)
    return files


def _analyze_problem_group_with_llm(
    group: Dict, validation_sequence: List[Dict], analyzer
) -> List[Dict]:
    """Build final problems for one validation/failure group."""
    group_chunks = _chunk_validation_group(group, analyzer)
    if len(group_chunks) == 1:
        return _analyze_problem_group_chunk_with_llm(
            group_chunks[0], validation_sequence, analyzer
        )

    print(f"        Group is large; analyzing {len(group_chunks)} chunk(s)")
    chunk_problems = []
    for chunk_idx, chunk_group in enumerate(group_chunks, 1):
        problems = _analyze_problem_group_chunk_with_llm(
            chunk_group,
            validation_sequence,
            analyzer,
            chunk_index=chunk_idx,
            total_chunks=len(group_chunks),
        )
        for problem_idx, problem in enumerate(problems, 1):
            item = dict(problem)
            item["_chunk_problem_id"] = f"{chunk_idx}.{problem_idx}"
            item["_source_chunk_index"] = chunk_idx
            chunk_problems.append(item)

    return _merge_chunk_problem_outputs(group, chunk_problems, analyzer)


def _chunk_validation_group(group: Dict, analyzer) -> List[Dict]:
    """Split large validation groups by events while preserving group metadata."""
    max_group_chars = _validation_group_char_limit(analyzer)
    group_json = json.dumps(group, indent=2, default=str)
    if len(group_json) <= max_group_chars:
        return [group]

    chunks = []
    current_events = []

    for event in group.get("events", []):
        candidate_events = current_events + [event]
        candidate_group = _copy_group_with_events(group, candidate_events)
        candidate_size = len(json.dumps(candidate_group, indent=2, default=str))
        if current_events and candidate_size > max_group_chars:
            chunks.append(_copy_group_with_events(group, current_events))
            current_events = [event]
        else:
            current_events = candidate_events

    if current_events:
        chunks.append(_copy_group_with_events(group, current_events))

    return chunks or [group]


def _validation_group_char_limit(analyzer) -> int:
    """Use the configured model input chunk size for validation group chunking."""
    model_name = getattr(analyzer, "model_name", None)
    if get_input_chunk_chars:
        try:
            # Reserve space for the fixed prompt, validation sequence, and output.
            return max(20_000, int(get_input_chunk_chars(model_name) * 0.75))
        except Exception:
            pass
    return 210_000


def _copy_group_with_events(group: Dict, events: List[Dict]) -> Dict:
    """Copy validation group metadata and recalculate dependency context."""
    return {
        "validation_order": group.get("validation_order"),
        "validation_cmd": group.get("validation_cmd", ""),
        "failure_type": group.get("failure_type", ""),
        "events": events,
        "group_context": _collect_group_context(events),
    }


def _analyze_problem_group_chunk_with_llm(
    group: Dict,
    validation_sequence: List[Dict],
    analyzer,
    chunk_index: Optional[int] = None,
    total_chunks: Optional[int] = None,
) -> List[Dict]:
    """Ask the LLM to build atomic problems for one group chunk."""
    group_json = json.dumps(group, indent=2)
    validation_json = json.dumps(validation_sequence, indent=2)
    chunk_context = ""
    if chunk_index and total_chunks:
        chunk_context = (
            f"\nCHUNK CONTEXT:\n- chunk_index: {chunk_index}\n"
            f"- total_chunks: {total_chunks}\n"
            "- This is one chunk of a larger validation group. Extract atomic "
            "problems for this chunk only; a later pass will merge similar "
            "chunk problems across the full validation group.\n"
        )

    prompt = f"""Analyze this validation group and create atomic CI repair problems.

GROUPED COMMIT EVENTS:
{group_json}

CI VALIDATION SEQUENCE:
{validation_json}

VALIDATION CONTEXT:
- validation_order: {group.get("validation_order")}
- validation_cmd: {json.dumps(group.get("validation_cmd", ""))}
- failure_type: {json.dumps(group.get("failure_type", ""))}
{chunk_context}

TASK:
Infer the actual CI step problem fixed by these grouped commit events. The
events have already been grouped by validation_cmd and failure_type, and the
groups are processed in CI validation order. Use all events in this group
together with group_context and dependency_context to define atomic problems and
their corresponding repairs.

DECISION PROCESS:
1. Identify the CI step being repaired.
- validation_cmd may be an install/setup command, not only a final checker.
- Package metadata, dependency files, lockfiles, workflow setup, environment
  config, and tool installation changes belong to the relevant setup/install
  CI step.
- Source, docs, and test changes belong to the validator that directly checks
  them.
- Prefer the CI step that would fail without this specific change.

2. Decide merge vs split inside this validation group.
- Merge changes into one atomic problem when one explanation clearly covers all
  affected files: same validation_cmd, same validator/tool family, same repair
  family, repeated instances of the same validator problem, or variants of the
  same failure family.
- Split changes into separate atomic problems when one explanation would hide
  important differences: materially different root cause, repair strategy,
  affected file kind, setup/config enablement mixed with source/docs/test fixes,
  or different validator concern.

3. Handle repeated failures across files dynamically.
- Same validator plus same repair family across many files is one repeated
  problem pattern, even when files have variants.
- Formatter/linter/doc-style variants are usually one problem when the same
  tool normalizes them.
- For bulk changes, group by directory scope, file type, validator, and repair
  family.
- Mention directory scope and important variants in problem/root_cause/how_fixed.
- Do not list every file in prose because affected_files already contains exact
  paths.

4. Keep setup/install enablement separate.
- Examples: invalid pyproject metadata, missing dependency, wrong extras,
  incompatible tool version, broken pip/poetry install config, workflow setup
  command mismatch.
- If setup changes only enable a later formatter, linter, type checker, or test
  command, report the setup/install issue separately from later validation
  violations.

5. Handle cascading fixes.
- Cascading means one change caused or required another related change.
- If all affected files share the same CI validation and repair family, they may
  be one atomic problem.
- If related files are caught by different CI validations or require different
  repair strategies, split them into separate atomic problems.
- For cascading problems, explain the triggering relationship in problem,
  root_cause, how_fixed, or why_fix_works.

6. Handle merge-conflict cleanup correctly.
- Git conflict markers and conflict resolution mechanics are not CI problems.
- Never use "merge conflict" or "conflict resolution" as failure_type,
  issue_type, problem, root_cause, or how_fixed.
- Analyze the final before/after content around the conflict and classify the
  CI-relevant change that remained after resolution.
- If the only change is conflict marker removal with no CI-relevant behavior,
  formatting, config, docs, dependency, or workflow change, do not create a
  problem for that change.

Rules:
1. Use the grouped events as the source of truth. Do not invent files, jobs, or
   validation commands.
2. Use group_context and dependency_context to understand which commit-level
   selected groups produced each problem event and whether changes are connected
   through imports, calls, config reads, generated files, or tests.
3. Each problem must have a specific root cause and fix that applies to every
   affected file.
4. Be specific about packages, symbols, config keys, validators, commands,
   before/after states, directories, and affected file kinds.
5. Do not mention line numbers.
6. Do not use vague phrasing like "fixed issues" or "updated files".
7. affected_files must include only files directly involved in that problem's
   fix.
8. If no valid CI problem can be extracted, return {{"atomic_problems": []}}.

FIELD GUIDANCE:
- issue_type: Specific failure subtype, error code, rule, or
  validator-specific category.
- problem: 1-2 sentences describing what failed. Mention directory scope, file
  types, and important variants when relevant.
- root_cause: 1-2 sentences explaining what violated which rule, requirement,
  or expectation.
- how_fixed: 1-2 sentences describing what changed and why it was necessary.
- why_fix_works: 1-2 sentences explaining how the new state satisfies the CI
  step or validator.
- problem_type: "primary" when affected_files are visible in CI failure context;
  otherwise "hidden".

OUTPUT JSON FORMAT (valid JSON only, no markdown):
{{
  "atomic_problems": [
    {{
      "problem_id": 1,
      "validation_order": {json.dumps(group.get("validation_order"))},
      "validation_cmd": {json.dumps(group.get("validation_cmd", ""))},
      "failure_type": {json.dumps(group.get("failure_type", ""))},
      "issue_type": "specific_error_code_or_type",
      "problem": "Brief description of what broke",
      "root_cause": "Why it failed",
      "how_fixed": "What changed",
      "why_fix_works": "Why the fix solves it",
      "affected_files": ["file1.py", "file2.py"],
      "problem_type": "primary",
      "is_cascading": false,
      "dependency_type": "",
      "cascade_explanation": ""
    }},
    // Additional atomic problems if any
  ]
}}

OUTPUT REQUIREMENTS:
- Return only a JSON object with the "atomic_problems" array.
- If no problems can be extracted, return {{"atomic_problems": []}}.
- problem_id must be an integer starting at 1 and incrementing by 1.
- validation_order must be the integer validation_order from VALIDATION CONTEXT.
- validation_cmd must exactly match validation_cmd from VALIDATION CONTEXT.
- failure_type must match failure_type from VALIDATION CONTEXT unless the value
  is empty.
- affected_files must be an array of file path strings from GROUPED COMMIT EVENTS.
- Do not include files that are not directly involved in the problem.
- problem_type must be either "primary" or "hidden".
- is_cascading must be a boolean.
- dependency_type must be a string, empty string if not cascading.
- cascade_explanation must be a string, empty string if not cascading.
- String fields must be non-empty for every returned problem: issue_type,
  problem, root_cause, how_fixed, why_fix_works.
- Do not include JavaScript-style comments in JSON.
- Do not include markdown, explanations, or text outside the JSON object."""

    try:
        response = analyzer._call_llm(prompt)
        result = analyzer._parse_response(response)
        problems = result.get("atomic_problems", [])
        if problems:
            return [
                _attach_group_metadata(problem, group, result)
                for problem in problems
                if isinstance(problem, dict)
            ]
        return _fallback_group_problems(group)
    except Exception as e:
        print(f"    Warning: grouped LLM analysis failed: {e}")
        return _fallback_group_problems(group)


def _merge_chunk_problem_outputs(
    group: Dict, chunk_problems: List[Dict], analyzer
) -> List[Dict]:
    """Merge similar atomic problems found across chunks."""
    if not chunk_problems:
        return _fallback_group_problems(group)
    if len(chunk_problems) == 1:
        return [chunk_problems[0]]

    file_map = {
        problem.get("_chunk_problem_id"): problem.get("affected_files", [])
        for problem in chunk_problems
        if problem.get("_chunk_problem_id")
    }
    summaries = []
    for problem in chunk_problems:
        summaries.append(
            {
                "chunk_problem_id": problem.get("_chunk_problem_id"),
                "validation_order": problem.get("validation_order"),
                "validation_cmd": problem.get("validation_cmd"),
                "failure_type": problem.get("failure_type"),
                "issue_type": problem.get("issue_type"),
                "problem": problem.get("problem"),
                "root_cause": problem.get("root_cause"),
                "how_fixed": problem.get("how_fixed"),
                "why_fix_works": problem.get("why_fix_works"),
                "problem_type": problem.get("problem_type"),
                "is_cascading": problem.get("is_cascading"),
                "dependency_type": problem.get("dependency_type"),
                "cascade_explanation": problem.get("cascade_explanation"),
            }
        )

    prompt = f"""Merge atomic CI repair problems extracted from chunks of one validation group.

VALIDATION CONTEXT:
- validation_order: {group.get("validation_order")}
- validation_cmd: {json.dumps(group.get("validation_cmd", ""))}
- failure_type: {json.dumps(group.get("failure_type", ""))}

CHUNK ATOMIC PROBLEMS WITHOUT FILE LISTS:
{json.dumps(summaries, indent=2)}

FILE MAP BY CHUNK PROBLEM ID:
{json.dumps(file_map, indent=2)}

TASK:
Decide whether chunk problems are the same atomic CI problem. Merge them when
they have the same validation command, failure type, root cause, and repair
strategy. Keep them separate when the root cause or repair strategy is
materially different.

When merging, write one stronger problem/root_cause/how_fixed/why_fix_works
using evidence from all merged chunk problems. The code will union affected
files from merged_from_chunk_problem_ids, so choose those ids carefully.

OUTPUT JSON FORMAT (valid JSON only, no markdown):
{{
  "atomic_problems": [
    {{
      "problem_id": 1,
      "merged_from_chunk_problem_ids": ["1.1", "2.1"],
      "validation_order": {json.dumps(group.get("validation_order"))},
      "validation_cmd": {json.dumps(group.get("validation_cmd", ""))},
      "failure_type": {json.dumps(group.get("failure_type", ""))},
      "issue_type": "specific_error_code_or_type",
      "problem": "Brief description of what broke",
      "root_cause": "Why it failed",
      "how_fixed": "What changed across all merged chunks",
      "why_fix_works": "Why the combined fix solves it",
      "problem_type": "primary",
      "is_cascading": false,
      "dependency_type": "",
      "cascade_explanation": ""
    }}
  ]
}}

OUTPUT REQUIREMENTS:
- Return only a JSON object with the "atomic_problems" array.
- merged_from_chunk_problem_ids must only contain ids from CHUNK ATOMIC PROBLEMS.
- Do not include affected_files; they are merged deterministically from the file map.
- String fields must be non-empty: issue_type, problem, root_cause, how_fixed,
  why_fix_works.
- validation_order, validation_cmd, and failure_type must match VALIDATION CONTEXT.
- Do not include markdown or text outside JSON."""

    try:
        response = analyzer._call_llm(prompt)
        result = analyzer._parse_response(response)
        merged = result.get("atomic_problems", [])
        if not merged:
            return chunk_problems
        return [
            _attach_merged_chunk_files(problem, chunk_problems, file_map)
            for problem in merged
            if isinstance(problem, dict)
        ]
    except Exception as e:
        print(f"    Warning: chunk problem merge failed: {e}")
        return chunk_problems


def _attach_merged_chunk_files(
    problem: Dict, chunk_problems: List[Dict], file_map: Dict[str, List[str]]
) -> Dict:
    """Union files and source metadata for a merged chunk-level problem."""
    item = dict(problem)
    merged_ids = item.get("merged_from_chunk_problem_ids", [])
    if not merged_ids:
        merged_ids = [
            problem.get("_chunk_problem_id")
            for problem in chunk_problems
            if problem.get("_chunk_problem_id")
        ]

    affected_files = []
    input_indices = []
    commit_shas = []
    for chunk_problem in chunk_problems:
        chunk_problem_id = chunk_problem.get("_chunk_problem_id")
        if chunk_problem_id not in merged_ids:
            continue
        affected_files.extend(file_map.get(chunk_problem_id, []))
        input_indices.extend(chunk_problem.get("_input_indices", []))
        commit_sha = chunk_problem.get("commit_sha", [])
        if isinstance(commit_sha, list):
            commit_shas.extend(commit_sha)
        elif commit_sha:
            commit_shas.append(commit_sha)

    item["affected_files"] = sorted(
        {file_path for file_path in affected_files if file_path}
    )
    item["_input_indices"] = sorted({idx for idx in input_indices if idx})
    item["commit_sha"] = sorted({sha for sha in commit_shas if sha})
    return item


def _attach_group_metadata(problem: Dict, group: Dict, result: Dict) -> Dict:
    """Attach deterministic group metadata to an LLM-generated problem."""
    item = dict(problem)
    item["_input_indices"] = [
        event.get("_input_index")
        for event in group["events"]
        if event.get("_input_index")
    ]
    item["commit_sha"] = result.get("commit_sha") or sorted(
        {
            event.get("commit_sha", "")
            for event in group["events"]
            if event.get("commit_sha")
        }
    )
    item.setdefault("validation_cmd", group.get("validation_cmd", ""))
    item.setdefault("failure_type", group.get("failure_type", ""))
    item.setdefault("validation_order", group.get("validation_order"))
    return item


def _normalize_final_problems(
    problems: List[Dict], validation_order_by_cmd: Optional[Dict[str, int]] = None
) -> List[Dict]:
    """Ensure final problem sequence matches the requested output schema."""
    validation_order_by_cmd = validation_order_by_cmd or {}
    normalized = []
    for idx, problem in enumerate(problems or [], 1):
        if not isinstance(problem, dict):
            continue
        item = dict(problem)
        item.pop("_input_index", None)
        item.pop("_input_indices", None)
        item.pop("_chunk_problem_id", None)
        item.pop("_source_chunk_index", None)
        item.pop("commit_sha", None)
        item.pop("merged_from_chunk_problem_ids", None)
        item["problem_id"] = idx
        item["validation_order"] = item.get(
            "validation_order",
            validation_order_by_cmd.get(item.get("validation_cmd", "")),
        )
        item.setdefault("validation_cmd", "")
        item.setdefault("failure_type", "")
        item.setdefault("issue_type", "")
        item.setdefault("problem", "")
        item.setdefault("root_cause", "")
        item.setdefault("how_fixed", item.get("changes_made", ""))
        item.setdefault("why_fix_works", item.get("why_this_fix_works", ""))
        item.setdefault("affected_files", item.get("files", []))
        item.setdefault("problem_type", "primary")
        item.setdefault("is_cascading", False)
        item.setdefault("dependency_type", "")
        item.setdefault("cascade_explanation", "")
        item["repair_sequence_index"] = idx
        normalized.append(item)
    return normalized


def _fallback_group_problems(group: Dict) -> List[Dict]:
    """Build one deterministic problem for a group when LLM analysis fails."""
    events = group.get("events", [])
    files = []
    problem_text = []
    root_causes = []
    fixes = []
    reasons = []

    for event in events:
        files.extend(event.get("files", []))
        if event.get("problem"):
            problem_text.append(event["problem"])
        if event.get("root_cause"):
            root_causes.append(event["root_cause"])
        if event.get("fixed"):
            fixes.extend(
                [
                    event.get("fix_strategy", ""),
                    event.get("repair", ""),
                    event.get("changes_made", ""),
                ]
            )
            reasons.append(event.get("why_this_fix_works", ""))

    issue_types = [
        event.get("issue_type", "") for event in events if event.get("issue_type")
    ]
    return [
        {
            "_input_indices": [
                event.get("_input_index")
                for event in events
                if event.get("_input_index")
            ],
            "commit_sha": sorted(
                {
                    event.get("commit_sha", "")
                    for event in events
                    if event.get("commit_sha")
                }
            ),
            "validation_order": group.get("validation_order"),
            "validation_cmd": group.get("validation_cmd", ""),
            "failure_type": group.get("failure_type", ""),
            "issue_type": _join_unique(issue_types),
            "problem": _join_unique(problem_text),
            "root_cause": _join_unique(root_causes),
            "how_fixed": _join_unique(fixes),
            "why_fix_works": _join_unique(reasons),
            "affected_files": sorted({file_path for file_path in files if file_path}),
            "problem_type": "primary",
            "is_cascading": _has_dependency_context(group),
            "dependency_type": "READS" if _has_dependency_context(group) else "",
            "cascade_explanation": "",
        }
    ]


def _has_dependency_context(group: Dict) -> bool:
    """Return whether a final validation group has dependency evidence."""
    for context in group.get("group_context", []) or []:
        dependency_context = context.get("dependency_context")
        if dependency_context:
            return True
    return False


def _join_unique(values: List[str]) -> str:
    """Join unique non-empty strings in original order."""
    return "\n\n".join(dict.fromkeys(v for v in values if v))
