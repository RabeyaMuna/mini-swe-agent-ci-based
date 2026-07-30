#!/usr/bin/env python3
"""Resolve CI validation commands for GitHub Actions job/step metadata."""

import re
from copy import deepcopy
from typing import Any, Dict, List

try:
    import yaml
except Exception:  # pragma: no cover - optional runtime dependency
    yaml = None


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def enrich_ci_metadata_commands(
    ci_metadata: Dict,
    *,
    structured_failure: Dict | None = None,
    validation_sequence: List[Dict] | None = None,
    github_fetcher: Any = None,
    repo_owner: str = "",
    repo_name: str = "",
    commit_sha: str = "",
    workflow_path: str = "",
) -> Dict:
    """Add best-effort validation_cmd values to CI job/step records.

    The resolver is dynamic: it uses whatever evidence is available for the
    current repository/commit instead of hardcoded workflow or step names.
    """
    enriched = deepcopy(ci_metadata or {})
    structured_failure = structured_failure or {}
    validation_sequence = validation_sequence or []

    workflow_paths = _workflow_paths(enriched, workflow_path)
    candidates = []
    candidates.extend(_structured_failure_candidates(structured_failure))
    candidates.extend(_validation_sequence_candidates(validation_sequence))
    candidates.extend(
        _workflow_candidates(
            workflow_paths,
            github_fetcher=github_fetcher,
            repo_owner=repo_owner,
            repo_name=repo_name,
            commit_sha=commit_sha,
        )
    )

    for collection_name in ("failed_steps", "current_failed_jobs"):
        enriched[collection_name] = [
            _enrich_record(record, candidates)
            for record in enriched.get(collection_name, []) or []
        ]

    enriched["current_jobs_fixed"] = [
        _enrich_record(record, candidates, success_record=True)
        for record in enriched.get("current_jobs_fixed", []) or []
    ]

    if github_fetcher and repo_owner and repo_name:
        enriched["current_failed_jobs"] = [
            _enrich_from_job_log(
                record,
                github_fetcher=github_fetcher,
                repo_owner=repo_owner,
                repo_name=repo_name,
            )
            for record in enriched.get("current_failed_jobs", []) or []
        ]
        enriched["failed_steps"] = [
            _enrich_from_job_log(
                record,
                github_fetcher=github_fetcher,
                repo_owner=repo_owner,
                repo_name=repo_name,
            )
            for record in enriched.get("failed_steps", []) or []
        ]

    return enriched


def _workflow_paths(ci_metadata: Dict, workflow_path: str) -> List[str]:
    paths = []
    if workflow_path:
        paths.append(workflow_path)
    for run in ci_metadata.get("workflow_runs", []) or []:
        path = run.get("path")
        if path and path not in paths:
            paths.append(path)
    for record in (
        (ci_metadata.get("current_failed_jobs", []) or [])
        + (ci_metadata.get("current_jobs_fixed", []) or [])
        + (ci_metadata.get("failed_steps", []) or [])
    ):
        path = record.get("workflow_path")
        if path and path not in paths:
            paths.append(path)
    return paths


def _structured_failure_candidates(structured_failure: Dict) -> List[Dict]:
    candidates = []
    for item in structured_failure.get("failed_job", []) or []:
        if not isinstance(item, dict):
            continue
        command = item.get("command") or item.get("failed_cmd")
        if command:
            candidates.append(
                {
                    "command": str(command),
                    "source": "structured_failure.failed_job",
                    "job": item.get("job", ""),
                    "step": item.get("step", ""),
                    "text": " ".join(str(item.get(key, "")) for key in ("job", "step")),
                }
            )

    for item in structured_failure.get("relevant_files", []) or []:
        if not isinstance(item, dict):
            continue
        command = item.get("failed_cmd") or item.get("command")
        if command:
            candidates.append(
                {
                    "command": str(command),
                    "source": "structured_failure.relevant_files",
                    "tool": item.get("failed_tool", ""),
                    "text": " ".join(
                        str(item.get(key, ""))
                        for key in ("failed_tool", "issue_type", "reason", "file")
                    ),
                }
            )
    return candidates


def _validation_sequence_candidates(validation_sequence: List[Dict]) -> List[Dict]:
    candidates = []
    for item in validation_sequence or []:
        if not isinstance(item, dict):
            continue
        command = item.get("validation_cmd") or item.get("installation_cmd")
        if command:
            candidates.append(
                {
                    "command": str(command),
                    "source": "validation_sequence",
                    "text": " ".join(
                        str(item.get(key, ""))
                        for key in ("validates", "evidence", "source")
                    ),
                    "order": item.get("order"),
                }
            )
    return candidates


def _workflow_candidates(
    workflow_paths: List[str],
    *,
    github_fetcher: Any,
    repo_owner: str,
    repo_name: str,
    commit_sha: str,
) -> List[Dict]:
    if not yaml or not github_fetcher:
        return []

    candidates = []
    for path in workflow_paths:
        content = github_fetcher.get_file_content(
            repo_owner, repo_name, path, commit_sha
        )
        if not content:
            continue
        try:
            workflow = yaml.safe_load(content) or {}
        except Exception:
            continue
        jobs = workflow.get("jobs", {}) or {}
        if not isinstance(jobs, dict):
            continue
        for job_key, job_def in jobs.items():
            if not isinstance(job_def, dict):
                continue
            job_name = str(job_def.get("name") or job_key)
            for step in job_def.get("steps", []) or []:
                if not isinstance(step, dict):
                    continue
                command = step.get("run")
                if not command:
                    continue
                step_name = str(step.get("name") or command).strip()
                candidates.append(
                    {
                        "command": _compact_shell(command),
                        "source": f"workflow:{path}",
                        "job": job_name,
                        "step": step_name,
                        "text": f"{job_key} {job_name} {step_name}",
                    }
                )
    return candidates


def _enrich_record(
    record: Dict, candidates: List[Dict], success_record: bool = False
) -> Dict:
    item = dict(record)
    if item.get("validation_cmd"):
        return item

    match = _best_candidate(item, candidates)
    if match:
        item["validation_cmd"] = match["command"]
        item["command_source"] = match["source"]
        item["command_confidence"] = match["confidence"]
    else:
        item.setdefault("validation_cmd", "")
        item.setdefault("command_source", "")
        if not success_record:
            item.setdefault("command_confidence", 0)
    return item


def _best_candidate(record: Dict, candidates: List[Dict]) -> Dict | None:
    best = None
    best_score = 0
    record_text = _normalized(
        " ".join(
            str(record.get(key, ""))
            for key in ("workflow", "job", "step", "validation_cmd")
        )
    )
    step_text = _normalized(str(record.get("step", "")))
    job_text = _normalized(str(record.get("job", "")))

    for candidate in candidates:
        score = 0
        candidate_step = _normalized(str(candidate.get("step", "")))
        candidate_job = _normalized(str(candidate.get("job", "")))
        candidate_text = _normalized(str(candidate.get("text", "")))

        if step_text and candidate_step and step_text == candidate_step:
            score += 100
        elif (
            step_text
            and candidate_step
            and (step_text in candidate_step or candidate_step in step_text)
        ):
            score += 70

        if (
            job_text
            and candidate_job
            and (
                job_text == candidate_job
                or job_text in candidate_job
                or candidate_job in job_text
            )
        ):
            score += 20

        score += _token_overlap_score(record_text, candidate_text, limit=40)

        if score > best_score:
            best_score = score
            best = candidate

    if not best or best_score < 12:
        return None
    return {
        "command": best["command"],
        "source": best["source"],
        "confidence": min(best_score, 100),
    }


def _enrich_from_job_log(
    record: Dict,
    *,
    github_fetcher: Any,
    repo_owner: str,
    repo_name: str,
) -> Dict:
    if record.get("validation_cmd") or not record.get("job_id"):
        return record

    log_text = github_fetcher.get_job_log(repo_owner, repo_name, record.get("job_id"))
    command = _extract_command_from_log(log_text, str(record.get("step", "")))
    if not command:
        return record

    item = dict(record)
    item["validation_cmd"] = command
    item["command_source"] = "job_log"
    item["command_confidence"] = 60
    return item


def _extract_command_from_log(log_text: str, step_name: str) -> str:
    if not log_text:
        return ""
    lines = log_text.splitlines()
    normalized_step = _normalized(step_name)
    start_idx = 0
    if normalized_step:
        for idx, line in enumerate(lines):
            if normalized_step in _normalized(line):
                start_idx = idx
                break

    window = lines[start_idx : start_idx + 160]
    commands = []
    for line in window:
        clean = ANSI_RE.sub("", line).strip()
        if not clean:
            continue
        if clean.startswith("Run "):
            commands.append(clean[4:].strip())
        elif "\x1b[36;1m" in line:
            commands.append(clean)

    return _compact_shell("\n".join(commands[:8]))


def _compact_shell(command: Any) -> str:
    lines = [line.strip() for line in str(command).splitlines() if line.strip()]
    return " && ".join(lines)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _token_overlap_score(left: str, right: str, limit: int) -> int:
    left_tokens = {token for token in left.split() if len(token) > 2}
    right_tokens = {token for token in right.split() if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0
    return min(len(left_tokens & right_tokens) * 4, limit)
