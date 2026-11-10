#!/usr/bin/env python3
"""Build the Whole-Repair-History control used for reviewer jcJU Q1.

Each historical fail-to-pass PR remains one indivisible episode. This module
does not call ReCAP problem decomposition, problem/hunk attribution,
reconciliation, dependency, or problem-ordering code.

One historical PR produces exactly one L1 issue record (with one aggregate
``problems`` item), one L2 issue record (with one aggregate strategy), and one
flat L3 transferable pattern. Familiar schemas are retained so the existing
retrieval and repair pipeline can consume the control unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from commit_decomposition.github_fetcher import GitHubFetcher
from utilities.llm_invoker import STRICT_JSON_RULES, invoke_llm_with_retry
from utilities.llm_model import LitellmModel
from utilities.model_registry import configure_model_environment


DEFAULT_MEMORY_IDS = PROJECT_ROOT / "data" / "memory_issue_ids.json"
DEFAULT_LOG_DETAILS = PROJECT_ROOT / "data" / "log_details.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experimental_data" / "whole_repair_history"
DEFAULT_DATASET = "ci-benchmark-user/ci-repair-bench"


def issue_id(record: dict[str, Any]) -> str:
    """Return the canonical string ID used throughout the repository."""
    return str(
        record.get("id")
        or record.get("instance_id")
        or record.get("issue_id")
        or ""
    ).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _repo_parts(record: dict[str, Any]) -> tuple[str, str, str]:
    owner = str(record.get("repo_owner") or "").strip()
    name = str(record.get("repo_name") or "").strip()
    repo = str(record.get("repo") or "").strip()
    if repo and "/" in repo:
        owner = owner or repo.split("/", 1)[0]
        name = name or repo.split("/", 1)[1]
    repo = repo or "/".join(part for part in (owner, name) if part)
    return repo, owner, name


def _github_source_repo(record: dict[str, Any]) -> tuple[str, str]:
    """Prefer commit_link because benchmark rows may name a benchmark fork."""
    link = str(record.get("commit_link") or "")
    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:/|$)", link)
    if match:
        return match.group(1), match.group(2).removesuffix(".git")
    _, owner, name = _repo_parts(record)
    return owner, name


def load_id_list(path: Path) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"Expected a JSON array in {path}")
    ids = [str(value).strip() for value in values if str(value).strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found in {path}")
    return ids


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def load_benchmark_records(
    *, dataset_path: Path | None, dataset_name: str, revision: str | None
) -> tuple[list[dict[str, Any]], str | None]:
    """Load raw benchmark rows, never existing ReCAP memory artifacts."""
    if dataset_path is not None:
        return load_jsonl(dataset_path), revision

    # Prefer an already materialized Arrow cache. Besides avoiding unnecessary
    # network access, Dataset.from_file is read-only and therefore does not
    # require a writable Hugging Face cache lock.
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / dataset_name.replace("/", "___")
        / "default"
        / "0.0.0"
    )
    cached_candidates: list[Path] = []
    if cache_root.exists():
        for candidate in cache_root.iterdir():
            if not candidate.is_dir() or not (candidate / "dataset_info.json").exists():
                continue
            if revision and candidate.name != revision:
                continue
            if list(candidate.glob("*.arrow")):
                cached_candidates.append(candidate)
    if cached_candidates:
        from datasets import Dataset, concatenate_datasets

        selected = max(
            cached_candidates,
            key=lambda path: (path / "dataset_info.json").stat().st_mtime,
        )
        shards = [Dataset.from_file(str(path)) for path in sorted(selected.glob("*.arrow"))]
        dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        print(f"Using cached raw benchmark revision: {selected.name}")
        return [dict(row) for row in dataset], selected.name

    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": "train"}
    if revision:
        kwargs["revision"] = revision
    dataset = load_dataset(dataset_name, **kwargs)
    return [dict(row) for row in dataset], revision


def load_structured_failures(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Index log-only CI analysis by issue ID and failing SHA."""
    if not path.exists():
        return {}, {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
    by_id: dict[str, dict] = {}
    by_sha: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or "").strip()
        sha = str(row.get("sha_fail") or "").strip()
        if row_id:
            by_id[row_id] = row
        if sha:
            by_sha[sha] = row
    return by_id, by_sha


def _compact_commit(commit: dict[str, Any], order: int) -> dict[str, Any]:
    """Keep commit facts only; do not infer problem/repair associations."""
    return {
        "order": order,
        "sha": str(commit.get("sha") or ""),
        "message": str(commit.get("message") or ""),
        "author": str(commit.get("author") or ""),
        "date": str(commit.get("date") or ""),
        "html_url": str(commit.get("html_url") or ""),
    }


def collect_whole_pr_episode(
    record: dict[str, Any],
    *,
    structured_failure: dict[str, Any] | None,
    github_fetcher: GitHubFetcher | None,
    allow_incomplete: bool,
    fetch_final_ci: bool,
) -> dict[str, Any]:
    """Collect one raw fail-to-pass episode without decomposing or attributing it."""
    iid = issue_id(record)
    repo, repo_owner, repo_name = _repo_parts(record)
    source_owner, source_name = _github_source_repo(record)
    sha_fail = str(record.get("sha_fail") or "").strip()
    sha_success = str(record.get("sha_success") or "").strip()
    cumulative_patch = str(record.get("diff") or record.get("patch") or "")
    changed_files = _string_list(record.get("changed_files"))

    commits: list[dict[str, Any]] = []
    if github_fetcher is not None and source_owner and source_name and sha_fail and sha_success:
        fetched = github_fetcher.get_commits_between(
            source_owner, source_name, sha_success, sha_fail
        )
        commits = [_compact_commit(commit, index) for index, commit in enumerate(fetched, 1)]

    final_ci_api: dict[str, Any] = {}
    if fetch_final_ci and github_fetcher is not None and source_owner and source_name and sha_success:
        final_ci_api = github_fetcher.get_commit_ci_metadata(
            source_owner, source_name, sha_success
        )

    workflow_path = str(
        record.get("workflow_path") or record.get("workflow_filename") or ""
    )
    workflow_name = str(
        record.get("workflow_name")
        or record.get("workflow_filename")
        or Path(workflow_path).name
    )
    raw_logs = _as_list(record.get("logs"))
    initial_failure = {
        "sha": sha_fail,
        "raw_ci_logs": raw_logs,
        "structured_ci_failure": structured_failure or {},
        "error_types": _string_list(
            record.get("failure_types") or record.get("error_type")
        ),
        "failed_jobs": _as_list(record.get("failed_jobs")),
        "total_jobs": record.get("total_jobs"),
        "total_failed_jobs": record.get("total_failed_jobs"),
        "total_failed_steps": record.get("total_failed_steps"),
    }
    workflow_metadata = {
        "path": workflow_path,
        "name": workflow_name,
        "definition": str(record.get("workflow") or ""),
        "head_branch": str(record.get("head_branch") or ""),
        "language": str(record.get("language") or ""),
    }
    final_outcome = {
        "sha": sha_success,
        "conclusion": "success",
        "evidence_source": "benchmark fail-to-pass pair",
        "github_ci_metadata": final_ci_api,
    }
    completeness = {
        "initial_ci_evidence": bool(raw_logs or structured_failure),
        "repository_metadata": bool(repo and workflow_path),
        "ordered_commit_sequence": bool(commits),
        "cumulative_patch": bool(cumulative_patch),
        "changed_files": bool(changed_files),
        "successful_outcome": bool(sha_success),
    }
    missing = [name for name, present in completeness.items() if not present]
    if missing and not allow_incomplete:
        raise ValueError(
            f"Issue {iid} is incomplete ({', '.join(missing)}). "
            "Reviewer-grade memory requires all episode fields; rerun with "
            "working GitHub access or use --allow-incomplete only for debugging."
        )

    return {
        "issue_id": iid,
        "repo": repo,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "source_repository": f"{source_owner}/{source_name}".strip("/"),
        "workflow": workflow_path,
        "workflow_path": workflow_path,
        "workflow_name": workflow_name,
        "sha_fail": sha_fail,
        "sha_success": sha_success,
        "commit_link": str(record.get("commit_link") or ""),
        "initial_failure": initial_failure,
        "workflow_metadata": workflow_metadata,
        "ordered_commits": commits,
        "cumulative_patch": cumulative_patch,
        "all_changed_files": changed_files,
        "final_ci_outcome": final_outcome,
        "completeness": completeness,
    }


def _bounded_evidence(episode: dict[str, Any], max_chars: int) -> tuple[dict, bool]:
    """Bound only the LLM prompt; the stored L1 episode remains lossless."""
    evidence = {
        "issue_id": episode["issue_id"],
        "repo": episode["repo"],
        "workflow_metadata": episode["workflow_metadata"],
        "initial_failure": episode["initial_failure"],
        "ordered_commits": episode["ordered_commits"],
        "cumulative_patch": episode["cumulative_patch"],
        "all_changed_files": episode["all_changed_files"],
        "final_ci_outcome": episode["final_ci_outcome"],
    }
    encoded = json.dumps(evidence, ensure_ascii=False)
    if len(encoded) <= max_chars:
        return evidence, False

    compact = dict(evidence)
    initial = json.dumps(evidence["initial_failure"], ensure_ascii=False)
    patch = str(evidence["cumulative_patch"])
    fixed_size = len(encoded) - len(initial) - len(patch)
    available = max(2_000, max_chars - fixed_size)
    initial_budget = available // 2
    patch_budget = available - initial_budget
    compact["initial_failure"] = {
        "truncated_evidence": initial[:initial_budget],
        "original_characters": len(initial),
    }
    compact["cumulative_patch"] = patch[:patch_budget]
    compact["evidence_truncated_for_abstraction"] = True
    return compact, True


def build_abstraction_prompt(episode: dict[str, Any], max_chars: int) -> tuple[str, bool]:
    evidence, truncated = _bounded_evidence(episode, max_chars)
    prompt = f"""You are constructing a Whole-Repair-History experimental control.

Treat the complete fail-to-pass PR below as ONE INDIVISIBLE repair episode.
Do not decompose it into separate CI problems. Do not map commits, files, or
patch hunks to individual failures. Do not infer dependencies between problems.

WHOLE PR EVIDENCE:
```json
{json.dumps(evidence, indent=2, ensure_ascii=False)}
```

Return exactly one aggregate abstraction at each level:
1. L1: one concrete description of the complete PR repair.
2. L2: one repository-level strategy for the entire repair episode.
3. L3: one transferable cross-repository pattern for the entire episode.

Required JSON schema:
{{
  "l1": {{
    "failure_type": "aggregate failure category",
    "problem": "complete PR-level failure description",
    "root_cause": "aggregate root-cause explanation",
    "fix_strategy": "complete PR-level repair strategy",
    "why_fix_works": "why the cumulative repair produces a passing outcome",
    "verification_cmd": "overall CI verification command or workflow"
  }},
  "l2": {{
    "failure_identify": "one aggregate failure summary",
    "strategy": {{
      "failure_type": "aggregate failure category",
      "summary": "one whole-PR strategy summary",
      "intent": "overall repair intent",
      "reasoning": "reasoning across the complete episode",
      "rationale": "why this whole-episode strategy is appropriate",
      "when_to_apply": "conditions for applying this strategy",
      "signals": ["observable signal"],
      "key_actions": ["ordered whole-episode action"],
      "pitfalls": ["pitfall"],
      "validation_cmd": "overall validation command"
    }}
  }},
  "l3": {{
    "pattern_id": "stable-kebab-case-pattern-id",
    "failure_type": "general failure category",
    "failure_pattern": "generalized whole-episode failure pattern",
    "problem": "generalized problem",
    "reasoning": "why the generalized repair works",
    "when_to_apply": "transfer conditions",
    "signals": ["general signal"],
    "universal_fix": {{
      "approach": "general approach",
      "steps": ["general repair step"],
      "applies_to": ["applicable context"]
    }},
    "examples": [{{"file": "representative file", "before": "before", "after": "after"}}],
    "dependent_changes": []
  }}
}}

The output must contain one object, not arrays of problems, strategies, or
patterns. Never output per-problem or per-commit repair records.

{STRICT_JSON_RULES}
"""
    return prompt, truncated


def generate_whole_pr_abstractions(
    episode: dict[str, Any], llm: Any, max_evidence_chars: int
) -> tuple[dict[str, Any], bool]:
    prompt, truncated = build_abstraction_prompt(episode, max_evidence_chars)
    value = invoke_llm_with_retry(
        llm=llm, prompt=prompt, parse_json=True, max_tokens=16_000
    )
    if not isinstance(value, dict):
        raise ValueError(f"Issue {episode['issue_id']}: model did not return an object")
    validate_abstraction(value, episode["issue_id"])
    return value, truncated


def _required_text(mapping: dict[str, Any], key: str, context: str) -> str:
    value = str(mapping.get(key) or "").strip()
    if not value:
        raise ValueError(f"{context}: missing non-empty {key}")
    return value


def validate_abstraction(value: dict[str, Any], iid: str) -> None:
    """Reject empty outputs and attempts to reintroduce decomposition."""
    for level in ("l1", "l2", "l3"):
        if not isinstance(value.get(level), dict):
            raise ValueError(f"Issue {iid}: {level} must be one object")
    for field in ("failure_type", "problem", "root_cause", "fix_strategy", "why_fix_works"):
        _required_text(value["l1"], field, f"Issue {iid} L1")
    _required_text(value["l2"], "failure_identify", f"Issue {iid} L2")
    if not isinstance(value["l2"].get("strategy"), dict):
        raise ValueError(f"Issue {iid}: L2 strategy must be one object")
    _required_text(value["l2"]["strategy"], "summary", f"Issue {iid} L2 strategy")
    for field in ("pattern_id", "failure_type", "failure_pattern", "problem", "reasoning"):
        _required_text(value["l3"], field, f"Issue {iid} L3")
    if not isinstance(value["l3"].get("universal_fix"), dict):
        raise ValueError(f"Issue {iid}: L3 universal_fix must be an object")

    forbidden = {"problems", "repair_strategies", "universal_patterns"}
    for level in ("l1", "l2", "l3"):
        leaked = forbidden.intersection(value[level])
        if leaked:
            raise ValueError(
                f"Issue {iid}: {level} reintroduced plural records: {sorted(leaked)}"
            )


def build_memory_records(
    episode: dict[str, Any], abstraction: dict[str, Any], *, prompt_truncated: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Wrap three singular abstractions in existing retrieval schemas."""
    iid = episode["issue_id"]
    l1_abstract = abstraction["l1"]
    problem = {
        "problem_id": 1,
        "verification_cmd": str(l1_abstract.get("verification_cmd") or ""),
        "failure_type": str(l1_abstract.get("failure_type") or ""),
        "problem": str(l1_abstract.get("problem") or ""),
        "root_cause": str(l1_abstract.get("root_cause") or ""),
        "fix_strategy": str(l1_abstract.get("fix_strategy") or ""),
        "why_fix_works": str(l1_abstract.get("why_fix_works") or ""),
        "files": list(episode.get("all_changed_files", [])),
        "enabled": [],
    }
    l1 = {
        "issue_id": iid,
        "repo": episode["repo"],
        "repo_owner": episode["repo_owner"],
        "repo_name": episode["repo_name"],
        "workflow": episode["workflow"],
        "workflow_path": episode["workflow_path"],
        "workflow_name": episode["workflow_name"],
        "problems": [problem],
        "dependencies": [],
        "repair_sequence": [1],
        "representation": "whole_repair_history",
        "whole_pr_history": episode,
        "abstraction_input_truncated": prompt_truncated,
    }

    l2_abstract = abstraction["l2"]
    strategy = dict(l2_abstract["strategy"])
    strategy["step"] = 1
    strategy.setdefault("failure_type", problem["failure_type"])
    strategy.setdefault("signals", [])
    strategy.setdefault("key_actions", [])
    strategy.setdefault("pitfalls", [])
    l2 = {
        "issue_id": iid,
        "repo": episode["repo"],
        "workflow": episode["workflow"],
        "workflow_path": episode["workflow_path"],
        "workflow_name": episode["workflow_name"],
        "total_problems": 1,
        "failure_identify": [str(l2_abstract["failure_identify"])],
        "repair_strategies": [strategy],
        "representation": "whole_repair_history",
    }

    l3 = dict(abstraction["l3"])
    raw_pattern_id = str(l3.get("pattern_id") or "whole-repair-history")
    safe_pattern_id = re.sub(r"[^a-z0-9]+", "-", raw_pattern_id.lower()).strip("-")
    l3["pattern_id"] = f"whole-pr-{iid}-{safe_pattern_id}"
    l3["source_issue_id"] = iid
    l3["source_repo"] = episode["repo"]
    l3["representation"] = "whole_repair_history"
    l3.setdefault("dependent_changes", [])
    return l1, l2, l3


def validate_memory_sets(
    expected_ids: Iterable[str], l1_rows: list[dict], l2_rows: list[dict], l3_rows: list[dict]
) -> None:
    expected = [str(value) for value in expected_ids]
    l1_ids = [str(row.get("issue_id")) for row in l1_rows]
    l2_ids = [str(row.get("issue_id")) for row in l2_rows]
    l3_ids = [str(row.get("source_issue_id")) for row in l3_rows]
    if l1_ids != expected or l2_ids != expected or l3_ids != expected:
        raise ValueError("L1/L2/L3 IDs or ordering differ from requested memory IDs")
    for row in l1_rows:
        if len(row.get("problems", [])) != 1:
            raise ValueError(f"L1 issue {row.get('issue_id')} does not contain exactly one item")
        if row.get("dependencies"):
            raise ValueError(f"L1 issue {row.get('issue_id')} unexpectedly has dependencies")
    for row in l2_rows:
        if len(row.get("repair_strategies", [])) != 1:
            raise ValueError(f"L2 issue {row.get('issue_id')} does not contain exactly one strategy")
    if len(l3_ids) != len(set(l3_ids)):
        raise ValueError("L3 must contain exactly one pattern per source PR")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_by_id(rows: list[dict], key: str) -> dict[str, dict]:
    return {str(row.get(key)): row for row in rows if row.get(key) is not None}


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [row for row in value if isinstance(row, dict)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-ids", type=Path, default=DEFAULT_MEMORY_IDS)
    parser.add_argument("--dataset", type=Path, help="Raw benchmark JSONL")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", help="Optional pinned Hugging Face revision")
    parser.add_argument("--log-details", type=Path, default=DEFAULT_LOG_DETAILS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--issue-ids", help="Optional comma-separated subset of memory IDs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--collect-only", action="store_true", help="Collect raw episodes without LLM abstraction")
    parser.add_argument("--no-github", action="store_true", help="Disable commit collection (debug only)")
    parser.add_argument("--fetch-final-ci", action="store_true", help="Query final GitHub workflow/job metadata")
    parser.add_argument("--allow-incomplete", action="store_true", help="Permit incomplete debug episodes")
    parser.add_argument("--max-evidence-chars", type=int, default=120_000)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory_ids = load_id_list(args.memory_ids)
    if args.issue_ids:
        requested = {value.strip() for value in args.issue_ids.split(",") if value.strip()}
        unknown = requested.difference(memory_ids)
        if unknown:
            raise ValueError(f"Requested IDs are not in memory split: {sorted(unknown)}")
        memory_ids = [iid for iid in memory_ids if iid in requested]
    if args.limit is not None:
        memory_ids = memory_ids[: args.limit]

    records, effective_dataset_revision = load_benchmark_records(
        dataset_path=args.dataset,
        dataset_name=args.dataset_name,
        revision=args.dataset_revision,
    )
    by_id = {issue_id(row): row for row in records if issue_id(row)}
    missing_rows = [iid for iid in memory_ids if iid not in by_id]
    if missing_rows:
        raise ValueError(f"Raw benchmark is missing memory IDs: {missing_rows[:10]}")

    failures_by_id, failures_by_sha = load_structured_failures(args.log_details)
    github_fetcher = None if args.no_github else GitHubFetcher()
    output_dir = args.output_dir.resolve()
    memory_dir = output_dir / "bidirect_trs"
    raw_path = output_dir / "raw_episodes.json"
    l1_path = memory_dir / "failure_memory.json"
    l2_path = memory_dir / "repo_memory.json"
    l3_path = memory_dir / "cross_memory.json"

    raw_by_id = _rows_by_id(_load_json_list(raw_path), "issue_id")
    l1_by_id = _rows_by_id(_load_json_list(l1_path), "issue_id")
    l2_by_id = _rows_by_id(_load_json_list(l2_path), "issue_id")
    l3_by_id = _rows_by_id(_load_json_list(l3_path), "source_issue_id")

    # A full unauthenticated build exceeds GitHub's normal hourly allowance.
    # Fail early with a useful message instead of producing an unexplained
    # partial memory after rate limiting. Already-complete raw episodes do not
    # consume another request and are safe to resume.
    needs_github = [
        iid
        for iid in memory_ids
        if iid not in raw_by_id
        or not all((raw_by_id[iid].get("completeness") or {}).values())
        or args.rebuild
    ]
    if github_fetcher is not None and needs_github:
        rate = github_fetcher.check_rate_limit()
        minimum_requests = len(needs_github)
        if rate.get("remaining", 0) < minimum_requests:
            raise RuntimeError(
                "Insufficient GitHub API allowance for ordered commit collection: "
                f"need at least {minimum_requests} requests, have "
                f"{rate.get('remaining', 0)}. Set GITHUB_TOKEN (recommended) or "
                "collect a smaller subset into a separate output directory."
            )

    llm = None
    if not args.collect_only:
        model = configure_model_environment(args.model) or args.model
        llm = LitellmModel(model_name=model)

    for index, iid in enumerate(memory_ids, 1):
        complete = iid in l1_by_id and iid in l2_by_id and iid in l3_by_id
        if complete and not args.rebuild and not args.collect_only:
            print(f"[{index}/{len(memory_ids)}] {iid}: already complete")
            continue

        cached_episode = raw_by_id.get(iid)
        cached_is_usable = bool(
            cached_episode
            and (
                args.allow_incomplete
                or all((cached_episode.get("completeness") or {}).values())
            )
        )
        if cached_is_usable and not args.rebuild:
            episode = cached_episode
            print(f"[{index}/{len(memory_ids)}] {iid}: using cached whole PR")
        else:
            print(f"[{index}/{len(memory_ids)}] {iid}: collecting whole PR")
            row = by_id[iid]
            structured = failures_by_id.get(iid) or failures_by_sha.get(
                str(row.get("sha_fail") or "")
            )
            episode = collect_whole_pr_episode(
                row,
                structured_failure=structured,
                github_fetcher=github_fetcher,
                allow_incomplete=args.allow_incomplete,
                fetch_final_ci=args.fetch_final_ci,
            )
            raw_by_id[iid] = episode
            _write_json(
                raw_path, [raw_by_id[key] for key in memory_ids if key in raw_by_id]
            )

        if args.collect_only:
            continue

        print(f"[{index}/{len(memory_ids)}] {iid}: building singular L1/L2/L3")
        abstraction, was_truncated = generate_whole_pr_abstractions(
            episode, llm, args.max_evidence_chars
        )
        l1, l2, l3 = build_memory_records(
            episode, abstraction, prompt_truncated=was_truncated
        )
        l1_by_id[iid], l2_by_id[iid], l3_by_id[iid] = l1, l2, l3
        _write_json(l1_path, [l1_by_id[key] for key in memory_ids if key in l1_by_id])
        _write_json(l2_path, [l2_by_id[key] for key in memory_ids if key in l2_by_id])
        _write_json(l3_path, [l3_by_id[key] for key in memory_ids if key in l3_by_id])

    _write_json(output_dir / "memory_issue_ids.json", memory_ids)
    generated_ids = [
        iid for iid in memory_ids if iid in l1_by_id and iid in l2_by_id and iid in l3_by_id
    ]
    incomplete_ids = [
        iid
        for iid in memory_ids
        if iid in raw_by_id
        and not all((raw_by_id[iid].get("completeness") or {}).values())
    ]
    truncated_ids = [
        iid
        for iid in memory_ids
        if iid in l1_by_id and l1_by_id[iid].get("abstraction_input_truncated")
    ]
    if not args.collect_only and generated_ids == memory_ids:
        validate_memory_sets(
            memory_ids,
            [l1_by_id[iid] for iid in memory_ids],
            [l2_by_id[iid] for iid in memory_ids],
            [l3_by_id[iid] for iid in memory_ids],
        )

    manifest = {
        "experiment": "whole_repair_history",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset) if args.dataset else args.dataset_name,
        "requested_dataset_revision": args.dataset_revision,
        "effective_dataset_revision": effective_dataset_revision,
        "memory_ids_source": str(args.memory_ids),
        "requested_count": len(memory_ids),
        "raw_episode_count": sum(iid in raw_by_id for iid in memory_ids),
        "complete_memory_count": len(generated_ids),
        "model": None if args.collect_only else args.model,
        "collect_only": args.collect_only,
        "github_commit_collection": not args.no_github,
        "fetch_final_ci": args.fetch_final_ci,
        "allow_incomplete": args.allow_incomplete,
        "incomplete_issue_ids": incomplete_ids,
        "abstraction_input_truncated_issue_ids": truncated_ids,
        "invariants": {
            "problem_decomposition": False,
            "problem_repair_attribution": False,
            "dependency_expansion": False,
            "records_per_pr_per_level": 1,
        },
        "files": {},
    }
    for path in (raw_path, l1_path, l2_path, l3_path):
        if path.exists():
            manifest["files"][str(path.relative_to(output_dir))] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    _write_json(output_dir / "build_manifest.json", manifest)

    print(f"Raw episodes: {manifest['raw_episode_count']}/{len(memory_ids)}")
    print(f"Complete L1/L2/L3: {len(generated_ids)}/{len(memory_ids)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
