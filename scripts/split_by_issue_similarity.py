#!/usr/bin/env python3
"""Build recurrence metadata and a memory/eval split from issue similarity.

This is an offline benchmark-preparation tool. It may use ground-truth fields
such as the patch diff to decide which historical issues are recurring, but the
generated eval rows should still be run without exposing ground truth to agents.

Examples:
    python scripts/split_by_issue_similarity.py \
      --issues data/trs/eval_issues.json \
      --out-dir data/trs/similarity_split

    python scripts/split_by_issue_similarity.py \
      --issues data/trs/eval_issues.json \
      --mode ci \
      --backend lexical \
      --threshold 0.35
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [row for row in payload.values() if isinstance(row, dict)]
    raise ValueError(f"Unsupported issue payload in {path}")


def repo_key(row: dict[str, Any]) -> str:
    owner = str(row.get("repo_owner") or "").strip()
    name = str(row.get("repo_name") or row.get("repo") or "").strip()
    if owner and name and "/" not in name:
        return f"{owner}/{name}"
    return name or owner or "unknown"


def issue_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("instance_id") or row.get("sha_fail") or "")


def stable_key(row: dict[str, Any]) -> str:
    return f"{repo_key(row)}::{issue_id(row)}::{row.get('sha_fail', '')}"


def as_text(value: Any, *, limit: int | None = None) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = "\n".join(as_text(item) for item in value)
    elif isinstance(value, dict):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(text) > limit:
        return text[:limit]
    return text


def diff_summary(diff: str, *, limit: int = 6000) -> str:
    """Keep patch signal compact and remove noisy line bodies when possible."""
    if not diff:
        return ""
    lines: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("diff --git", "+++ ", "--- ", "@@ ")):
            lines.append(line)
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            stripped = line[1:].strip()
            if stripped:
                lines.append(stripped[:240])
        if sum(len(item) + 1 for item in lines) >= limit:
            break
    return "\n".join(lines)[:limit]


def build_similarity_document(row: dict[str, Any], *, mode: str, log_chars: int, diff_chars: int) -> str:
    parts = [
        f"repo: {repo_key(row)}",
        f"workflow_name: {as_text(row.get('workflow_name'))}",
        f"workflow_path: {as_text(row.get('workflow_path') or row.get('workflow_filename'))}",
        f"error_type: {as_text(row.get('error_type') or row.get('overall_error_types'))}",
        f"changed_files: {as_text(row.get('changed_files'))}",
        f"workflow: {as_text(row.get('workflow'), limit=5000)}",
        f"logs: {as_text(row.get('logs'), limit=log_chars)}",
    ]

    # Decomposed-memory rows use these fields; plain benchmark rows usually do not.
    parts.extend(
        [
            f"overall_failure_summary: {as_text(row.get('overall_failure_summary'))}",
            f"workflow_reasoning: {as_text(row.get('workflow_reasoning'))}",
            f"problems: {as_text(row.get('problems') or row.get('atomic_problems'), limit=6000)}",
            f"repair_trajectory: {as_text(row.get('trajectory_summary') or row.get('repair_trajectory_summary'), limit=4000)}",
        ]
    )

    if mode == "ci+ground-truth" and row.get("_trusted_ground_truth"):
        parts.append(f"ground_truth_diff: {diff_summary(as_text(row.get('diff')), limit=diff_chars)}")

    return "\n".join(part for part in parts if part.split(": ", 1)[1].strip())


def load_verified_patch_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows = load_rows(path)
    verified: dict[str, dict[str, str]] = {}
    for row in rows:
        patch = {
            "id": str(row.get("id") or ""),
            "sha_fail": str(row.get("sha_fail") or ""),
            "diff": str(row.get("diff") or ""),
        }
        for key in (patch["id"], patch["sha_fail"]):
            if key:
                verified[key] = patch
    return verified


def apply_ground_truth_policy(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    verified_patches: dict[str, dict[str, str]],
    verified_first_n: int,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out = dict(row)
        patch = verified_patches.get(issue_id(out)) or verified_patches.get(str(out.get("sha_fail") or ""))
        trusted = False
        source = ""

        if policy == "all":
            trusted = bool(out.get("diff"))
            source = "input_diff" if trusted else ""
        elif policy == "none":
            trusted = False
        else:
            if patch and patch.get("diff"):
                out["diff"] = patch["diff"]
                trusted = True
                source = "verified_patches"
            elif verified_first_n > 0 and idx < verified_first_n and out.get("diff"):
                trusted = True
                source = f"first_{verified_first_n}"

        out["_trusted_ground_truth"] = trusted
        out["_ground_truth_source"] = source
        prepared.append(out)
    return prepared


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./-]+", text.lower())


def lexical_vectors(docs: list[str]) -> list[Counter[str]]:
    vectors: list[Counter[str]] = []
    doc_freq: Counter[str] = Counter()
    raw_counts: list[Counter[str]] = []
    for doc in docs:
        counts = Counter(tokenize(doc))
        raw_counts.append(counts)
        doc_freq.update(counts.keys())
    n_docs = max(len(docs), 1)
    for counts in raw_counts:
        weighted: Counter[str] = Counter()
        for token, count in counts.items():
            weighted[token] = count * (math.log((1 + n_docs) / (1 + doc_freq[token])) + 1.0)
        vectors.append(weighted)
    return vectors


def sparse_cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def dense_embeddings(docs: list[str]) -> tuple[list[np.ndarray] | None, str]:
    try:
        from minisweagent.run.benchmarks.utils.memory_plugin import _EmbeddingProvider
    except Exception as exc:
        print(f"[similarity] Could not import embedding provider: {exc}")
        return None, "none"

    provider = _EmbeddingProvider.get()
    if getattr(provider, "_backend", "none") == "none":
        return None, "none"
    vectors: list[np.ndarray] = []
    for i, doc in enumerate(docs, 1):
        vec = provider.embed(doc)
        if vec is None:
            print(f"[similarity] Embedding failed for document {i}; falling back is required.")
            return None, "none"
        vectors.append(np.array(vec, dtype=np.float32))
    return vectors, str(getattr(provider, "_backend", "embedding"))


def cosine_matrix(docs: list[str], backend: str) -> tuple[list[list[float]], str]:
    if backend in {"embedding", "auto"}:
        vectors, resolved = dense_embeddings(docs)
        if vectors is not None:
            scores = [[0.0 for _ in docs] for _ in docs]
            for i in range(len(docs)):
                scores[i][i] = 1.0
                for j in range(i + 1, len(docs)):
                    score = float(np.dot(vectors[i], vectors[j]))
                    scores[i][j] = scores[j][i] = score
            return scores, resolved
        if backend == "embedding":
            raise RuntimeError("No embedding backend available. Use --backend lexical or install/cache an embedding model.")

    vectors = lexical_vectors(docs)
    scores = [[0.0 for _ in docs] for _ in docs]
    for i in range(len(docs)):
        scores[i][i] = 1.0
        for j in range(i + 1, len(docs)):
            score = sparse_cosine(vectors[i], vectors[j])
            scores[i][j] = scores[j][i] = score
    return scores, "lexical"


def connected_components(indices: list[int], edges: dict[int, set[int]]) -> list[list[int]]:
    unseen = set(indices)
    components: list[list[int]] = []
    while unseen:
        start = unseen.pop()
        queue: deque[int] = deque([start])
        comp = [start]
        while queue:
            cur = queue.popleft()
            for nxt in edges.get(cur, set()):
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
                    comp.append(nxt)
        components.append(sorted(comp))
    return components


def choose_representative(component: list[int], scores: list[list[float]], rows: list[dict[str, Any]]) -> int:
    return max(
        component,
        key=lambda idx: (
            sum(scores[idx][other] for other in component if other != idx),
            len([other for other in component if other != idx and scores[idx][other] > 0.0]),
            stable_hash(stable_key(rows[idx])),
        ),
    )


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def annotate_rows(
    rows: list[dict[str, Any]],
    scores: list[list[float]],
    threshold: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[int]]]:
    by_repo: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_repo[repo_key(row)].append(idx)

    edges: dict[int, set[int]] = defaultdict(set)
    pair_rows: list[dict[str, Any]] = []
    for repo, indices in by_repo.items():
        for pos, left in enumerate(indices):
            for right in indices[pos + 1 :]:
                score = scores[left][right]
                if score >= threshold:
                    edges[left].add(right)
                    edges[right].add(left)
                    pair_rows.append(
                        {
                            "repo": repo,
                            "issue1": issue_id(rows[left]),
                            "issue2": issue_id(rows[right]),
                            "sha1": rows[left].get("sha_fail", ""),
                            "sha2": rows[right].get("sha_fail", ""),
                            "similarity": round(score, 6),
                        }
                    )

    annotated: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        peers = []
        for other_idx, other in enumerate(rows):
            if idx == other_idx or repo_key(row) != repo_key(other):
                continue
            score = scores[idx][other_idx]
            if score >= threshold:
                peers.append(
                    {
                        "id": issue_id(other),
                        "sha_fail": other.get("sha_fail", ""),
                        "repo": repo_key(other),
                        "similarity": round(score, 6),
                    }
                )
        peers.sort(key=lambda item: (-float(item["similarity"]), str(item["id"])))
        all_repo_scores = [
            scores[idx][other_idx]
            for other_idx, other in enumerate(rows)
            if idx != other_idx and repo_key(row) == repo_key(other)
        ]
        threshold_scores = [float(peer["similarity"]) for peer in peers]
        enriched = dict(row)
        enriched["recurrence_count"] = 1 + len(peers)
        enriched["avg_similarity"] = round(sum(threshold_scores) / len(threshold_scores), 6) if threshold_scores else 0.0
        enriched["avg_repo_similarity"] = round(sum(all_repo_scores) / len(all_repo_scores), 6) if all_repo_scores else 0.0
        enriched["top_similar_issues"] = peers[:top_k]
        annotated.append(enriched)

    components: list[list[int]] = []
    for indices in by_repo.values():
        components.extend(connected_components(indices, edges))
    return annotated, sorted(pair_rows, key=lambda row: (-row["similarity"], row["repo"])), components


def select_memory_seed(
    annotated: list[dict[str, Any]],
    scores: list[list[float]],
    components: list[list[int]],
    memory_ratio: float,
) -> set[int]:
    target = int(round(len(annotated) * memory_ratio))
    if annotated and target == 0:
        target = 1
    target = min(max(target, 0), len(annotated))

    recurring_components = [comp for comp in components if len(comp) > 1]
    recurring_components.sort(
        key=lambda comp: (
            -len(comp),
            -sum(scores[i][j] for i in comp for j in comp if i < j) / max((len(comp) * (len(comp) - 1) / 2), 1),
            repo_key(annotated[comp[0]]),
        )
    )

    selected: list[int] = []
    for comp in recurring_components:
        if len(selected) >= target:
            break
        selected.append(choose_representative(comp, scores, annotated))

    if len(selected) > target:
        selected = selected[:target]

    selected_set = set(selected)
    if len(selected_set) < target:
        remaining = [idx for idx in range(len(annotated)) if idx not in selected_set]
        remaining.sort(
            key=lambda idx: (
                -int(annotated[idx].get("recurrence_count") or 1),
                -float(annotated[idx].get("avg_similarity") or 0.0),
                stable_hash(stable_key(annotated[idx])),
            )
        )
        selected_set.update(remaining[: target - len(selected_set)])

    return selected_set


def repo_summary(rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs_by_repo: Counter[str] = Counter()
    for row in rows:
        by_repo[repo_key(row)].append(row)
    for pair in pair_rows:
        pairs_by_repo[str(pair["repo"])] += 1

    summary: list[dict[str, Any]] = []
    for repo, repo_rows in sorted(by_repo.items()):
        recurring = [row for row in repo_rows if int(row.get("recurrence_count") or 1) > 1]
        avg_sim = [float(row.get("avg_similarity") or 0.0) for row in repo_rows]
        summary.append(
            {
                "repo": repo,
                "issues": len(repo_rows),
                "recurring_issues": len(recurring),
                "recurring_percent": round(100.0 * len(recurring) / len(repo_rows), 2) if repo_rows else 0.0,
                "similar_pairs": pairs_by_repo[repo],
                "avg_similarity": round(sum(avg_sim) / len(avg_sim), 6) if avg_sim else 0.0,
                "memory_seed": sum(1 for row in repo_rows if row.get("split_role") == "memory"),
                "evaluation": sum(1 for row in repo_rows if row.get("split_role") == "evaluation"),
            }
        )
    return summary


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict[str, Any]], total: int, seed_count: int, backend: str) -> None:
    print(f"[similarity] backend: {backend}")
    print(f"[similarity] issues: {total}  memory: {seed_count}  evaluation: {total - seed_count}")
    print("| Repo | Issues | Recurring | Pairs | Avg Similarity | Memory | Eval |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary:
        print(
            f"| {row['repo']} | {row['issues']} | {row['recurring_issues']} "
            f"({row['recurring_percent']:.2f}%) | {row['similar_pairs']} | "
            f"{row['avg_similarity']:.4f} | {row['memory_seed']} | {row['evaluation']} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze issue recurrence and create a memory/eval split.")
    parser.add_argument("--issues", type=Path, default=Path("data/trs/eval_issues.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/trs/similarity_split"))
    parser.add_argument("--memory-ratio", type=float, default=0.30)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["ci", "ci+ground-truth"], default="ci+ground-truth")
    parser.add_argument(
        "--ground-truth-policy",
        choices=["verified", "all", "none"],
        default="verified",
        help="Use repair/diff signal for all rows, no rows, or only verified rows.",
    )
    parser.add_argument(
        "--verified-patches",
        type=Path,
        default=None,
        help="JSON/JSONL with trusted {id, sha_fail, diff}; matching rows use this diff.",
    )
    parser.add_argument(
        "--verified-first-n",
        type=int,
        default=0,
        help="Also trust input diffs for the first N issue rows, useful for manually verified prefixes.",
    )
    parser.add_argument("--backend", choices=["auto", "embedding", "lexical"], default="auto")
    parser.add_argument("--log-chars", type=int, default=12000)
    parser.add_argument("--diff-chars", type=int, default=6000)
    args = parser.parse_args()

    if not 0.0 <= args.memory_ratio <= 1.0:
        raise ValueError("--memory-ratio must be between 0 and 1")

    rows = load_rows(args.issues)
    if not rows:
        raise ValueError(f"No issues loaded from {args.issues}")
    verified_patches = load_verified_patch_rows(args.verified_patches)
    rows = apply_ground_truth_policy(
        rows,
        policy=args.ground_truth_policy,
        verified_patches=verified_patches,
        verified_first_n=args.verified_first_n,
    )

    docs = [
        build_similarity_document(row, mode=args.mode, log_chars=args.log_chars, diff_chars=args.diff_chars)
        for row in rows
    ]
    scores, backend = cosine_matrix(docs, args.backend)
    annotated, pairs, components = annotate_rows(rows, scores, args.threshold, args.top_k)
    seed_indices = select_memory_seed(annotated, scores, components, args.memory_ratio)

    memory_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(annotated):
        out = dict(row)
        out["similarity_backend"] = backend
        out["similarity_mode"] = args.mode
        out["similarity_threshold"] = args.threshold
        out["trusted_ground_truth"] = bool(out.pop("_trusted_ground_truth", False))
        out["ground_truth_source"] = str(out.pop("_ground_truth_source", ""))
        if idx in seed_indices:
            out["split_role"] = "memory"
            memory_rows.append(out)
        else:
            out["split_role"] = "evaluation"
            eval_rows.append(out)

    all_rows = memory_rows + eval_rows
    summary = repo_summary(all_rows, pairs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "memory_seed_issues.json", memory_rows)
    write_json(args.out_dir / "eval_issues.json", eval_rows)
    write_jsonl(args.out_dir / "memory_seed_issues.jsonl", memory_rows)
    write_jsonl(args.out_dir / "eval_issues.jsonl", eval_rows)
    write_json(args.out_dir / "all_issues_with_similarity.json", all_rows)
    write_csv(args.out_dir / "similar_issue_pairs.csv", pairs)
    write_csv(args.out_dir / "repo_similarity_summary.csv", summary)

    print_summary(summary, len(rows), len(memory_rows), backend)
    print(f"[similarity] wrote outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
