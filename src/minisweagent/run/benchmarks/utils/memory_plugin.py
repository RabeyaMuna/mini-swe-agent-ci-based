from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except Exception:
    _NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _STModel
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

try:
    from fastembed import TextEmbedding as _FastEmbedModel
    _FASTEMBED_AVAILABLE = True
except Exception:
    _FASTEMBED_AVAILABLE = False

try:
    import chromadb
    _CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None
    _CHROMADB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Step 2 — Embedding Generation
# Generates dense vector representations of text for cosine similarity
# retrieval.
#
# Model priority (holistic, all features concatenated into one vector):
#   1. sentence-transformers/all-MiniLM-L6-v2  — high-quality, 384-dim,
#      fast, widely used; ideal for semantic sentence similarity
#   2. fastembed BAAI/bge-base-en-v1.5         — ONNX-optimised local model,
#      768-dim, strong cross-lingual retrieval
#
# NO TF-IDF FALLBACK — bag-of-words does not capture semantic similarity
# well enough for this task.  Install one of the packages above.
#
# All features (file path + error_type + failure_pattern + issue_type +
# failure_reason + failed_tool + failed_cmd) are concatenated into a single
# document string before embedding so one holistic vector represents the
# entire failure fingerprint.
# ---------------------------------------------------------------------------

class _EmbeddingProvider:
    """
    Singleton dense-embedding provider.

    Model priority:
      1. sentence-transformers/all-MiniLM-L6-v2
         - best for semantic sentence similarity
         - install: pip install sentence-transformers
      2. fastembed BAAI/bge-base-en-v1.5
         - ONNX-optimized, CPU-friendly, no API cost
         - install: pip install fastembed
      3. Neither available → returns 0.0 with a clear warning.
         TF-IDF fallback intentionally removed.
    """

    _instance: Optional["_EmbeddingProvider"] = None

    def __init__(self) -> None:
        self._model: Any = None
        self._backend: str = "none"
        self._cache: Dict[str, Any] = {}

        if _ST_AVAILABLE and _NUMPY_AVAILABLE:
            try:
                self._model = _STModel("all-MiniLM-L6-v2")
                self._backend = "sentence_transformers"
                print("[Memory] Embedding provider: sentence-transformers/all-MiniLM-L6-v2")
                return
            except Exception as exc:
                print(f"[Memory] sentence-transformers load failed ({exc}); trying fastembed…")

        if _FASTEMBED_AVAILABLE and _NUMPY_AVAILABLE:
            try:
                self._model = _FastEmbedModel("BAAI/bge-base-en-v1.5")
                self._backend = "fastembed"
                print("[Memory] Embedding provider: BAAI/bge-base-en-v1.5 (fastembed)")
                return
            except Exception as exc:
                print(f"[Memory] fastembed model load failed ({exc})")

        print(
            "[Memory] WARNING: No embedding model available. "
            "Install sentence-transformers (`pip install sentence-transformers`) "
            "or fastembed (`pip install fastembed`). "
            "Cosine similarity will return 0.0 — memory retrieval disabled."
        )

    @classmethod
    def get(cls) -> "_EmbeddingProvider":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, text: str):
        """Return a unit-norm numpy float32 vector for *text*, or None on failure."""
        if self._model is None or not text.strip():
            return None
        if text in self._cache:
            return self._cache[text]
        try:
            if self._backend == "sentence_transformers":
                # encode with normalisation so dot-product == cosine similarity
                vec = self._model.encode(text, normalize_embeddings=True)
                vec = np.array(vec, dtype=np.float32)
            else:
                # fastembed returns a generator
                vec = np.array(next(self._model.embed([text])), dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            self._cache[text] = vec
            return vec
        except Exception:
            return None

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """
        Dense cosine similarity between two text documents.
        Both texts should contain all relevant features concatenated
        (file path | error_type | failure_pattern | issue_type |
         failure_reason | failed_tool | failed_cmd).
        Returns 0.0 if either embedding fails — NO TF-IDF fallback.
        """
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        if vec_a is None or vec_b is None:
            return 0.0
        return float(np.dot(vec_a, vec_b))


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Pipeline Step 3 — Holistic Cosine Similarity.
    Embeds two fully-concatenated feature strings and returns their
    cosine similarity.  Uses sentence-transformers (primary) or
    fastembed (fallback).  No TF-IDF bag-of-words fallback.
    """
    return _EmbeddingProvider.get().cosine_similarity(text_a, text_b)


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_path(path: str) -> str:
    return (path or "").strip().lstrip("/").replace("\\", "/")


def _repo_matches(query_repo: str, memory_repo: str) -> bool:
    """
    Fuzzy repo matching to handle format inconsistencies.

    Query may be "owner/repo" (full) or "repo" (short).
    Memory may be "owner/repo" (full) or "repo" (short).

    Returns True if they refer to the same repository.

    Examples:
      _repo_matches("camel-ai/camel", "camel") → True
      _repo_matches("camel", "camel-ai/camel") → True
      _repo_matches("camel-ai/camel", "camel-ai/camel") → True
      _repo_matches("camel", "camel") → True
      _repo_matches("camel-ai/camel", "other/repo") → False
    """
    if not query_repo or not memory_repo:
        return False

    query_repo = query_repo.strip()
    memory_repo = memory_repo.strip()

    # Exact match
    if query_repo == memory_repo:
        return True

    # Extract repo names (after last /)
    query_short = query_repo.split("/")[-1]
    memory_short = memory_repo.split("/")[-1]

    # Match if repo names are the same (ignoring owner prefix)
    return query_short == memory_short


def _basename(path: str) -> str:
    return os.path.basename(_normalize_path(path))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_./-]+", (text or "").lower())


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    lset = {x for x in left if x}
    rset = {x for x in right if x}
    if not lset and not rset:
        return 1.0
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / len(lset | rset)


def _clip(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _structured_file_refs(value: Any) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    for item in _safe_list(value):
        if isinstance(item, dict):
            path = _normalize_path(str(item.get("file") or item.get("path") or ""))
            reason = str(item.get("reason") or "").strip()
            if path:
                refs.append({"file": path, "reason": reason})
        else:
            path = _normalize_path(str(item or ""))
            if path:
                refs.append({"file": path, "reason": ""})
    return refs


def _normalize_error_type_rows(error_types: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in _safe_list(error_types):
        if isinstance(item, dict):
            rows.append(
                {
                    "category": str(item.get("category") or "").strip(),
                    "subcategory": str(item.get("subcategory") or "").strip(),
                    "evidence": str(item.get("evidence") or "").strip(),
                }
            )
        else:
            text = str(item or "").strip()
            if text:
                rows.append({"category": text, "subcategory": "", "evidence": ""})
    return rows


def _primary_error_type(error_types: Any) -> str:
    rows = _normalize_error_type_rows(error_types)
    if not rows:
        return ""
    return rows[0]["category"] or rows[0]["subcategory"]


def _primary_failure_pattern(error_types: Any) -> str:
    rows = _normalize_error_type_rows(error_types)
    if not rows:
        return ""
    return rows[0]["subcategory"] or rows[0]["category"]


def _to_descriptive_issue_type(raw: str, error_type: str = "") -> str:
    """
    Convert a raw snake_case/kebab-case issue_type into a readable phrase
    without relying on a fixed label map. Falls back to error_type when the
    issue_type is missing.
    """
    if not raw:
        return (error_type or "General CI failure").strip()
    return raw.replace("_", " ").replace("-", " ").strip().capitalize()


def extract_files_from_diff(diff_text: str) -> List[str]:
    files: List[str] = []
    for match in re.finditer(r"^diff --git a/.+ b/(.+)$", diff_text or "", re.MULTILINE):
        path = _normalize_path(match.group(1).strip())
        if path:
            files.append(path)
    return files


def _extract_file_diff(diff_text: str, file_path: str) -> str:
    target = _normalize_path(file_path)
    current: List[str] = []
    capture = False
    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            if capture:
                break
            match = re.match(r"^diff --git a/.+ b/(.+)$", line)
            capture = bool(match and _normalize_path(match.group(1)) == target)
        if capture:
            current.append(line)
    return "\n".join(current)


def _extract_log_file_paths(log_details: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for item in log_details.get("relevant_files", []) or []:
        if isinstance(item, dict):
            path = item.get("file") or item.get("path")
            if path:
                paths.append(_normalize_path(path))
        elif isinstance(item, str):
            paths.append(_normalize_path(item))
    return [p for p in paths if p]


def _extract_log_file_details(log_details: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract full per-file detail dicts from log_analysis_result['relevant_files'],
    preserving per-file issue_type, failed_cmd, failed_tool, reason, and line_number.
    These fields are produced by the log analyzer LLM for each file it identifies as
    relevant to the CI failure, and are used to enrich memory retrieval queries.
    Returns a list of dicts keyed by normalized 'file' path.
    """
    details: List[Dict[str, Any]] = []
    for item in log_details.get("relevant_files", []) or []:
        if isinstance(item, dict):
            path = _normalize_path(str(item.get("file") or item.get("path") or ""))
            if not path:
                continue
            # failed_cmd and failed_tool may be str or list; normalize to list
            raw_cmd = item.get("failed_cmd")
            raw_tool = item.get("failed_tool")
            details.append({
                "file": path,
                "issue_type": str(item.get("issue_type") or "").strip(),
                "failed_cmd": _safe_list(raw_cmd) if raw_cmd else [],
                "failed_tool": _safe_list(raw_tool) if raw_tool else [],
                "reason": str(item.get("reason") or "").strip(),
                "line_number": item.get("line_number"),
            })
        elif isinstance(item, str):
            path = _normalize_path(item)
            if path:
                details.append({
                    "file": path,
                    "issue_type": "",
                    "failed_cmd": [],
                    "failed_tool": [],
                    "reason": "",
                    "line_number": None,
                })
    return details


def _extract_changed_file_paths(changed_files_info: Optional[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for item in (changed_files_info or {}).get("changed_files", []) or []:
        path = _normalize_path(item.get("file_path", "")) if isinstance(item, dict) else ""
        if path:
            paths.append(path)
    return paths


def _extract_failed_commands_and_tools(failed_jobs: Any) -> Tuple[List[str], List[str]]:
    commands: List[str] = []
    tools: List[str] = []
    for item in _safe_list(failed_jobs):
        if isinstance(item, dict):
            for key in ("command", "validation_command", "failed_command", "cmd"):
                value = str(item.get(key) or "").strip()
                if value and value not in commands:
                    commands.append(value)
            for key in ("tool", "tools", "validator", "name", "job", "job_name", "step"):
                value = item.get(key)
                if isinstance(value, list):
                    for entry in value:
                        text = str(entry).strip()
                        if text and text not in tools:
                            tools.append(text)
                else:
                    text = str(value or "").strip()
                    if text and text not in tools:
                        tools.append(text)
        else:
            text = str(item).strip()
            if text and text not in tools:
                tools.append(text)
    return commands, tools


def _first_error_type(log_analysis_result: Dict[str, Any]) -> str:
    error_types = _safe_list(log_analysis_result.get("error_types", []))
    return str(error_types[0]).strip() if error_types else ""


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _write_json_list(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    os.replace(temp, path)


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads_safe(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


class MemoryPlugin:
    """
    MemCI-style three-level memory for fault localization.

    L1: failure_memory  — per-file failure records (repo + workflow scoped)
    L2: repo_memory     — repo-scoped recurring patterns (repo + workflow scoped)
    L3: cross_memory    — cross-repo generalized principles (no filtering, generalized)

    Search scope:
      - L1 & L2: Search within the same repo AND workflow context
      - L3: Search across all repos and workflows (generalized knowledge)

    This plugin does deterministic retrieval only. Higher-level LLM filtering
    and synthesis happens in ci_memory_system.py after ranked candidates are
    returned.
    """

    def __init__(self, config, result_dir: str, llm=None):
        self.config = config
        self.result_dir = result_dir
        self.llm = llm
        self.enabled = bool(self._cfg("memory_enabled", False))
        self.top_k = int(self._cfg("memory_top_k", 3))
        self.memory_backend = str(self._cfg("memory_backend", "json")).strip().lower() or "json"

        # Thresholds are reported for observability. Retrieval itself keeps all
        # positive-similarity candidates and lets the downstream memory gate
        # decide what to use.
        _raw_levels = str(self._cfg("memory_ablation_levels", "L1+L2+L3"))
        _ablation_thresholds = {"L1": 0.10, "L1+L2": 0.10, "L1+L2+L3": 0.10}
        if _raw_levels in _ablation_thresholds:
            self.similarity_threshold = float(_ablation_thresholds[_raw_levels])
        else:
            self.similarity_threshold = float(self._cfg("memory_similarity_threshold", 0.45))

        project_result_dir = str(self._cfg("project_result_dir", result_dir))
        self.failure_memory_path = os.path.join(project_result_dir, "failure_memory.json")
        self.repo_memory_path = os.path.join(project_result_dir, "repo_memory.json")
        self.cross_memory_path = os.path.join(project_result_dir, "cross_memory.json")
        self.chroma_dir = str(
            self._cfg("memory_chroma_dir", os.path.join(project_result_dir, "chroma_memory"))
        )
        self.retrieval_log_path = str(
            self._cfg(
                "memory_retrieval_log_path",
                os.path.join(result_dir, "memory_retrieval_log.jsonl"),
            )
        )

        self.failure_memory = _load_json_list(self.failure_memory_path)
        self.repo_memory    = _load_json_list(self.repo_memory_path)
        self.cross_memory   = _load_json_list(self.cross_memory_path)

        # Pre-load stored embeddings into the _EmbeddingProvider cache so
        # retrieval never has to re-embed the same record twice, even across restarts.
        # Embeddings are stored in each record as "_embedding": [float, ...]
        # and injected into the provider's cache keyed by the record's search_document.
        self._load_stored_embeddings()
        self.level_thresholds = {"L1": 0.35, "L2": 0.40, "L3": 0.50}

        # Ablation: which memory levels are active (L1 / L1+L2 / L1+L2+L3)
        raw_levels = str(self._cfg("memory_ablation_levels", "L1+L2+L3"))
        self.active_levels = {lvl.strip() for lvl in raw_levels.split("+") if lvl.strip()} or {"L1", "L2", "L3"}

        # Renormalize weights so weighted_similarity is comparable across ablations.
        _base = {"L1": 0.50, "L2": 0.40, "L3": 0.10}
        _active_sum = sum(_base[lvl] for lvl in self.active_levels if lvl in _base)
        self.level_weights = {
            lvl: (_base[lvl] / _active_sum if lvl in self.active_levels and _active_sum > 0 else 0.0)
            for lvl in ("L1", "L2", "L3")
        }
        print(
            f"[Memory] active_levels={sorted(self.active_levels)}  "
            f"weights={{{', '.join(f'{k}:{v:.3f}' for k,v in self.level_weights.items())}}}  "
            f"threshold={self.similarity_threshold}"
        )

        self._per_file_analysis_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._chroma_client: Any = None
        self._chroma_collections: Dict[str, Any] = {}
        if self.memory_backend == "chroma" and not _CHROMADB_AVAILABLE:
            print("[Memory] chromadb is not available; falling back to json backend.")
            self.memory_backend = "json"
        if self.memory_backend == "chroma":
            self._init_chroma()

    def _cfg(self, key: str, default: Any) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:
            value = getattr(self.config, key, default)
        return default if value is None else value

    # ── Embedding persistence ──────────────────────────────────────────────────

    def _load_stored_embeddings(self) -> None:
        """
        Pre-populate the embedding provider's in-memory cache from stored
        '_embedding' fields on memory records.

        When records are first embedded (at retrieval time), their vectors are
        saved back to the JSON files via _persist_embeddings_for_level().
        On the next startup this method loads those vectors directly into the
        cache so no re-embedding is needed — first query is as fast as all
        subsequent ones.
        """
        if not _NUMPY_AVAILABLE:
            return
        provider = _EmbeddingProvider.get()
        loaded = 0
        for level_records, level_name in (
            (self.failure_memory, "L1"),
            (self.repo_memory,    "L2"),
            (self.cross_memory,   "L3"),
        ):
            for record in level_records:
                doc = str(record.get("search_document") or "").strip()
                vec = record.get("_embedding")
                if doc and vec and doc not in provider._cache:
                    try:
                        provider._cache[doc] = np.array(vec, dtype=np.float32)
                        loaded += 1
                    except Exception:
                        pass
        if loaded:
            print(f"[Memory] Loaded {loaded} pre-computed embeddings from memory bank (no re-embedding needed)")

    def _persist_new_embeddings(self) -> None:
        """
        After retrieval, write back any newly computed embeddings into the
        memory records and save to disk. Called once after the first query
        so subsequent runs skip all re-embedding.
        """
        if not _NUMPY_AVAILABLE:
            return
        provider = _EmbeddingProvider.get()
        changed = {"L1": False, "L2": False, "L3": False}

        for level_records, level_key in (
            (self.failure_memory, "L1"),
            (self.repo_memory,    "L2"),
            (self.cross_memory,   "L3"),
        ):
            for record in level_records:
                if "_embedding" in record:
                    continue   # already stored
                doc = str(record.get("search_document") or "").strip()
                if not doc:
                    # Build and store the search_document if missing (seeded records)
                    level_char = level_key
                    doc = self._build_search_document(record, level=level_char)
                    if doc:
                        record["search_document"] = doc
                        changed[level_key] = True
                vec = provider._cache.get(doc)
                if vec is not None:
                    record["_embedding"] = vec.tolist()
                    changed[level_key] = True

        if changed["L1"]:
            _write_json_list(self.failure_memory_path, self.failure_memory)
        if changed["L2"]:
            _write_json_list(self.repo_memory_path, self.repo_memory)
        if changed["L3"]:
            _write_json_list(self.cross_memory_path, self.cross_memory)

        total = sum(1 for r in self.failure_memory + self.repo_memory + self.cross_memory if "_embedding" in r)
        print(f"[Memory] Persisted embeddings: {total} records now have stored vectors")

    def is_enabled(self) -> bool:
        return self.enabled

    def set_llm(self, llm: Any) -> None:
        self.llm = llm

    def _init_chroma(self) -> None:
        os.makedirs(self.chroma_dir, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self._chroma_collections = {
            "L1": self._chroma_client.get_or_create_collection("memory_l1"),
            "L2": self._chroma_client.get_or_create_collection("memory_l2"),
            "L3": self._chroma_client.get_or_create_collection("memory_l3"),
        }

    def _collection_for_level(self, level: str):
        return self._chroma_collections.get(level)

    def _serialize_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for key, value in record.items():
            if key == "search_document":
                continue
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[f"{key}_json"] = _json_dumps_compact(value)
        return metadata

    def _deserialize_metadata(self, metadata: Dict[str, Any], document: str = "") -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if key.endswith("_json"):
                row[key[:-5]] = _json_loads_safe(value, [])
            else:
                row[key] = value
        if document:
            row["search_document"] = document
        return row

    def _build_record_id(self, level: str, row: Dict[str, Any]) -> str:
        if level == "L1":
            return (
                f"l1:{row.get('sha_fail','')}:{_normalize_path(str(row.get('file','')))}:"
                f"{str(row.get('failure_pattern','')).lower()}"
            )
        if level == "L2":
            return f"l2:{row.get('sha_fail','')}"
        return (
            f"l3:{str(row.get('error_type','')).lower().replace(' ','_')}:"
            f"{str(row.get('issue_type','')).lower().replace(' ','_')}"
        )

    def _build_search_document(self, record: Dict[str, Any], *, level: str) -> str:
        def _json_text(value: Any) -> str:
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(value)

        def _atomic_problem_text(value: Any) -> str:
            rows = []
            for problem in _safe_list(value):
                if not isinstance(problem, dict):
                    text = str(problem).strip()
                    if text:
                        rows.append(text)
                    continue
                file_changes = _safe_list(problem.get("file_changes", []))
                file_text = " ".join(
                    " ".join(
                        str(x).strip()
                        for x in [
                            item.get("file", ""),
                            item.get("fix", "") or item.get("fix_strategy", ""),
                            item.get("what_wrong", "") or item.get("what_is_wrong", ""),
                        ]
                        if str(x or "").strip()
                    )
                    for item in file_changes if isinstance(item, dict)
                )
                rows.append(
                    " ".join(
                        str(x).strip()
                        for x in [
                            problem.get("problem_id"),
                            problem.get("visibility"),
                            problem.get("issue_type") or problem.get("problem_type"),
                            problem.get("failed_cmd") or problem.get("ci_command"),
                            problem.get("problem") or problem.get("symptom"),
                            problem.get("root_cause"),
                            problem.get("fix_strategy") or problem.get("how_fixed"),
                            file_text,
                        ]
                        if str(x or "").strip()
                    )
                )
            return " | ".join(row for row in rows if row)

        dependent_files = _structured_file_refs(record.get("dependent_files", []))
        dep_text = " | ".join(
            " ".join(
                x for x in [
                    ref.get("file", "") or ref.get("path", ""),
                    ref.get("reason", ""),
                ] if x
            )
            for ref in dependent_files
        )
        file_entries = _safe_list(record.get("files", []))
        file_text = " | ".join(
            " ".join(
                x for x in [
                    str(item.get("file", "")).strip(),
                    str(item.get("reason", "") or item.get("failure_reason", "")).strip(),
                    str(item.get("failure_pattern", "")).strip(),
                    str(item.get("fix_strategy", "") or item.get("fix_direction", "")).strip(),
                ] if x
            )
            for item in file_entries if isinstance(item, dict)
        )
        example_files = _safe_list(record.get("example_files", []))
        example_text = " | ".join(
            " ".join(
                x for x in [
                    str(item.get("file", "")).strip(),
                    str(item.get("repo", "")).strip(),
                    str(item.get("failure_pattern", "")).strip(),
                ] if x
            )
            for item in example_files if isinstance(item, dict)
        )
        # Resolve field aliases — seeded data uses different key names than runtime-saved data.
        # Always check both names so the document is equally rich for both sources.
        file_failure_reason = (
            record.get("file_failure_reason")
            or record.get("failure_reason")   # seeded alias
            or record.get("reason")           # seeded alias
            or ""
        )
        overall_failure_reason = (
            record.get("overall_failure_reason")
            or record.get("failure_reason")   # seeded alias
            or record.get("reason")           # seeded alias
            or ""
        )
        fix = (
            record.get("fix_direction")
            or record.get("fix_strategy")     # seeded alias
            or ""
        )
        # fix_pattern is a list in seeded data — join into text
        fix_pattern_items = _safe_list(record.get("fix_pattern") or [])
        fix_pattern_text  = " | ".join(str(x) for x in fix_pattern_items)
        if fix_pattern_text and not fix:
            fix = fix_pattern_text

        # issue_subtype is seeded-data's finer-grained label — treat as extra failure_pattern signal
        issue_subtype = str(record.get("issue_subtype") or record.get("root_cause_category") or "")
        repair_trajectory = (
            record.get("repair_trajectory")
            or record.get("repair_trajectory_summary")
            or record.get("trajectory_summary")
            or ""
        )
        atomic_problems = record.get("atomic_problems") or record.get("problems") or []

        parts = [
            f"level: {level}",
            f"repo: {record.get('repo','')}",
            f"workflow: {record.get('workflow_name') or record.get('workflow_path') or ''}",
            f"sha_fail: {record.get('sha_fail','')}",
            f"issue_id: {record.get('issue_id') or record.get('id') or ''}",
            f"file: {record.get('file','')}",
            f"line: {record.get('line_number','')}",
            f"error_type: {record.get('error_type','')}",
            f"issue_type: {record.get('issue_type','')}",
            f"issue_subtype: {issue_subtype}",
            f"failure_pattern: {record.get('failure_pattern','')}",
            f"failed_tool: {' '.join(str(x) for x in _safe_list(record.get('failed_tool', [])))}",
            f"failed_cmd: {' '.join(str(x) for x in _safe_list(record.get('failed_cmd', [])))}",
            f"file_failure_reason: {file_failure_reason}",
            f"overall_failure_reason: {overall_failure_reason}",
            f"principle: {record.get('principle') or ''}",
            f"fix_strategy: {fix}",
            f"fix_direction: {fix}",
            f"repair_trajectory: {repair_trajectory if isinstance(repair_trajectory, str) else _json_text(repair_trajectory)}",
            f"repair_trajectory_summary: {record.get('repair_trajectory_summary') or ''}",
            f"atomic_problems: {_atomic_problem_text(atomic_problems)}",
            f"workflow_ordered_problem_flow: {_json_text(record.get('workflow_ordered_problem_flow', []))}",
            f"failure_examples: {' | '.join(str(x) for x in _safe_list(record.get('failure_examples', [])))}",
            f"files: {file_text}",
            f"example_files: {example_text}",
            f"dependent_files: {dep_text}",
        ]
        return "\n".join(part for part in parts if part.split(": ", 1)[1].strip())

    def _normalize_l1_record(
        self,
        row: Dict[str, Any],
        *,
        overall_failure_reason: str,
        log_file_detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        detail = log_file_detail or {}
        record = dict(row)
        record["record_id"] = record.get("record_id") or self._build_record_id("L1", record)
        record["memory_level"] = "L1"
        record["line_number"] = detail.get("line_number", record.get("line_number"))
        record["file_failure_reason"] = str(
            record.get("file_failure_reason")
            or record.get("failure_reason")
            or detail.get("reason")
            or ""
        ).strip()
        record["overall_failure_reason"] = str(
            record.get("overall_failure_reason") or overall_failure_reason or ""
        ).strip()
        record["fix_strategy"] = str(record.get("fix_strategy") or record.get("fix_direction") or "").strip()
        record["search_document"] = self._build_search_document(record, level="L1")
        return record

    def _normalize_l2_record(self, row: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(row)
        record["record_id"] = record.get("record_id") or self._build_record_id("L2", record)
        record["memory_level"] = "L2"
        record["file"] = record.get("file", "")
        record["line_number"] = record.get("line_number")
        record["file_failure_reason"] = str(record.get("file_failure_reason") or "").strip()
        record["overall_failure_reason"] = str(
            record.get("overall_failure_reason") or record.get("failure_reason") or ""
        ).strip()
        record["fix_strategy"] = str(record.get("fix_strategy") or record.get("fix_approach") or "").strip()
        record["search_document"] = self._build_search_document(record, level="L2")
        return record

    def _normalize_l3_record(self, row: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(row)
        record["record_id"] = record.get("record_id") or self._build_record_id("L3", record)
        record["memory_level"] = "L3"
        record["repo"] = record.get("repo", "")
        record["repo_name"] = record.get("repo_name", "")
        record["file"] = record.get("file", "")
        record["line_number"] = record.get("line_number")
        record["overall_failure_reason"] = str(
            record.get("overall_failure_reason")
            or " | ".join(str(x) for x in _safe_list(record.get("failure_reasons", [])))
        ).strip()
        record["file_failure_reason"] = str(record.get("file_failure_reason") or "").strip()
        record["failure_examples"] = _safe_list(record.get("failure_examples", [])) or _safe_list(record.get("failure_reasons", []))
        record["fix_strategy"] = str(record.get("fix_strategy") or "").strip()
        record["search_document"] = self._build_search_document(record, level="L3")
        return record

    def _build_query_document(self, query: Dict[str, Any]) -> str:
        file_rows = []
        for item in (query.get("relevant_files_details") or []):
            if not isinstance(item, dict):
                continue
            file_rows.append(
                " ".join(
                    x for x in [
                        str(item.get("file", "")).strip(),
                        str(item.get("line_number", "") or "").strip(),
                        str(item.get("issue_type", "")).strip(),
                        " ".join(str(x) for x in _safe_list(item.get("failed_tool", []))),
                        " ".join(str(x) for x in _safe_list(item.get("failed_cmd", []))),
                        str(item.get("reason", "")).strip(),
                    ] if x
                )
            )
        chunk_rows = []
        for item in (query.get("chunk_summaries") or []):
            if not isinstance(item, dict):
                continue
            chunk_rows.append(
                " ".join(
                    x for x in [
                        str(item.get("step_name", "")).strip(),
                        f"chunk {item.get('chunk_index')}/{item.get('chunk_total')}" if item.get("chunk_index") and item.get("chunk_total") else "",
                        str(item.get("summary", "")).strip(),
                        " ".join(str(x).strip() for x in _safe_list(item.get("token_keywords")) if str(x).strip()),
                        " ".join(str(x).strip() for x in _safe_list(item.get("code_context")) if str(x).strip()),
                        " ".join(str(x).strip() for x in _safe_list(item.get("relevant_failures")) if str(x).strip()),
                    ] if x
                )
            )
        parts = [
            f"repo: {query.get('repo','')}",
            f"workflow: {query.get('workflow_path') or query.get('workflow_text') or ''}",
            f"file: {query.get('file_path') or ''}",
            f"error_type: {query.get('error_type','')}",
            f"issue_type: {_to_descriptive_issue_type(str(query.get('failure_pattern') or ''), str(query.get('error_type') or ''))}",
            f"failure_pattern: {query.get('failure_pattern','')}",
            f"failed_tool: {' '.join(str(x) for x in _safe_list(query.get('failed_tool', [])))}",
            f"failed_cmd: {' '.join(str(x) for x in _safe_list(query.get('failed_cmd', [])))}",
            f"file_failure_reason: {' | '.join(file_rows)}",
            f"overall_ci_summary: {query.get('overall_ci_summary') or ''}",
            f"overall_failure_reason: {query.get('failure_reason') or query.get('error_context_summary') or ''}",
            f"ci_analysis_document: {query.get('analysis_document') or ''}",
            f"chunk_summaries: {' | '.join(row for row in chunk_rows if row)}",
        ]
        return "\n".join(part for part in parts if part.split(': ', 1)[1].strip())

    def _metadata_boost(self, query: Dict[str, Any], row: Dict[str, Any], level: str) -> float:
        boost = 0.0
        # Use fuzzy repo matching for metadata boost
        query_repo = str(query.get("repo") or "")
        row_repo = str(row.get("repo") or "")
        if query_repo and row_repo and _repo_matches(query_repo, row_repo):
            boost += 0.05
        query_error = str(query.get("error_type") or "").lower()
        row_error = str(row.get("error_type") or "").lower()
        if query_error and row_error and query_error == row_error:
            boost += 0.04
        query_pattern = str(query.get("failure_pattern") or "").lower()
        row_pattern = str(row.get("failure_pattern") or "").lower()
        if query_pattern and row_pattern and query_pattern == row_pattern:
            boost += 0.03
        tool_score = _jaccard(
            [str(x).lower() for x in _safe_list(query.get("failed_tool", []))],
            [str(x).lower() for x in _safe_list(row.get("failed_tool", []))]
        )
        cmd_score = _jaccard(
            [str(x).lower() for x in _safe_list(query.get("failed_cmd", []))],
            [str(x).lower() for x in _safe_list(row.get("failed_cmd", []))]
        )
        boost += 0.03 * max(tool_score, cmd_score)
        if level == "L1":
            query_file = _normalize_path(str(query.get("file_path") or ""))
            row_file = _normalize_path(str(row.get("file") or ""))
            if query_file and row_file and (query_file == row_file or _basename(query_file) == _basename(row_file)):
                boost += 0.05
        return min(boost, 0.15)

    def _similar_matches(self, scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Keep every candidate that has measurable similarity.

        The memory bank is already split by level and repo/workflow scope where
        appropriate, so retrieval should not silently drop useful files because
        of a fixed slice. Ranking is preserved for downstream prompt formatting.
        """
        scored.sort(key=lambda item: float(item.get("similarity_score", 0.0)), reverse=True)
        return [row for row in scored if float(row.get("similarity_score", 0.0)) > 0.0]

    def _query_collection(self, level: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        collection = self._collection_for_level(level)
        if collection is None:
            return []
        query_document = self._build_query_document(query)
        query_embedding = _EmbeddingProvider.get().embed(query_document)
        if query_embedding is None:
            return []
        try:
            try:
                n_results = max(int(collection.count()), 1)
            except Exception:
                n_results = self.top_k
            result = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            print(f"[Memory] Chroma query failed for {level}: {exc}")
            return []

        metadatas = (result.get("metadatas") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        rows: List[Dict[str, Any]] = []
        for metadata, document, distance in zip(metadatas, documents, distances):
            row = self._deserialize_metadata(metadata or {}, document or "")
            semantic_score = max(0.0, 1.0 - float(distance or 0.0))
            final_score = round(0.85 * semantic_score + self._metadata_boost(query, row, level), 4)
            row["memory_level"] = level
            row["similarity_score"] = final_score
            row["matched_on"] = {
                "semantic_score": round(semantic_score, 4),
            }
            rows.append(row)
        return self._similar_matches(rows)

    def _upsert_chroma_record(self, level: str, record: Dict[str, Any]) -> None:
        collection = self._collection_for_level(level)
        if collection is None:
            return
        document = str(record.get("search_document") or "").strip()
        embedding = _EmbeddingProvider.get().embed(document)
        if not document or embedding is None:
            return
        collection.upsert(
            ids=[str(record["record_id"])],
            documents=[document],
            metadatas=[self._serialize_metadata(record)],
            embeddings=[embedding.tolist()],
        )

    def build_query(
        self,
        *,
        task_id: str,
        sha_fail: str,
        repo_name: str,
        workflow_path: str,
        workflow: str,
        log_analysis_result: Dict[str, Any],
        changed_files_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        failed_jobs = log_analysis_result.get("failed_jobs", log_analysis_result.get("failed_job", []))
        failed_cmd, failed_tool = _extract_failed_commands_and_tools(failed_jobs)
        error_type = _primary_error_type(log_analysis_result.get("error_types", []))
        failure_pattern = _primary_failure_pattern(log_analysis_result.get("error_types", []))

        # overall failure_reason: joined error_context sentences (the holistic narrative).
        # This IS the error_context — all sentences joined into one string for embedding.
        error_context_items = [
            str(x).strip()
            for x in _safe_list(log_analysis_result.get("error_context", []))
            if str(x).strip()
        ]
        failure_reason = _clip(" | ".join(error_context_items), 1200)

        # Per-file detail dicts from the log analyzer (issue_type, failed_cmd, failed_tool, reason).
        # Keyed by normalized file path for O(1) lookup during retrieval.
        relevant_files_details = _extract_log_file_details(log_analysis_result)

        # all_file_reasons: aggregate of every file's individual failure reason from the log
        # analyzer output.  Used in L2/L3 query docs so per-file failure info is included
        # alongside the overall error_context in the holistic embedding.
        all_file_reasons = _clip(
            " | ".join(
                d["reason"] for d in relevant_files_details if d.get("reason")
            ),
            1000,
        )

        return {
            "task_id": task_id,
            "sha_fail": sha_fail,
            "repo": repo_name,
            "repo_name": repo_name,
            "workflow_path": workflow_path,
            "workflow_text": workflow or "",
            "error_type": error_type,
            "failure_pattern": failure_pattern,
            "error_types": _normalize_error_type_rows(log_analysis_result.get("error_types", [])),
            "failed_cmd": failed_cmd,
            "failed_tool": failed_tool,
            "relevant_files": [d["file"] for d in relevant_files_details],
            "relevant_files_details": relevant_files_details,
            # Per-file reasons aggregated — used in L2/L3 query docs
            "all_file_reasons": all_file_reasons,
            "changed_files": _extract_changed_file_paths(changed_files_info),
            # failure_reason = overall error_context (the holistic narrative)
            "failure_reason": failure_reason,
            "error_context_summary": _clip(
                json.dumps(log_analysis_result.get("error_context", []), ensure_ascii=False),
                1800,
            ),
            "analysis_document": str(log_analysis_result.get("analysis_document") or ""),
            "overall_ci_summary": str(log_analysis_result.get("overall_ci_summary") or ""),
            "chunk_summaries": log_analysis_result.get("chunk_summaries", []) or [],
        }

    def _llm_rerank(self, query: Dict[str, Any], candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        LLM re-ranks candidates by semantic relevance.

        Stage 2 of hybrid retrieval: after fast embedding search,
        use LLM to score semantic relevance and return top_k best matches.
        """
        if not self.llm or not candidates:
            return candidates[:top_k]

        if len(candidates) <= top_k:
            return candidates  # No re-ranking needed

        # Prepare compact summary for LLM
        candidates_summary = []
        for i, c in enumerate(candidates):
            candidates_summary.append({
                "index": i,
                "memory_level": c.get("memory_level", ""),
                "similarity_score": round(float(c.get("similarity_score", 0.0)), 3),
                "issue_type": c.get("issue_type", ""),
                "error_type": c.get("error_type", ""),
                "problem": _clip(str(c.get("problem", "") or c.get("overall_failure_reason", "") or c.get("failure_reason", "")), 200),
                "fix": _clip(str(c.get("universal_fix_strategy", "") or c.get("fix_strategy", "")), 150),
            })

        prompt = f"""Score how relevant each past CI failure memory is for this NEW failure.

NEW CI FAILURE:
- Repo: {query.get("repo", "")}
- Error Type: {query.get("error_type", "")}
- Failed Command: {" ".join(_safe_list(query.get("failed_cmd", [])))}
- Error Message: {_clip(str(query.get("failure_reason", "")), 200)}

CANDIDATE MEMORIES (from embedding search):
{json.dumps(candidates_summary, indent=2)[:2500]}

Your task: Score each candidate 0-100 on relevance to the NEW failure.
Consider:
- Does it address the same root cause?
- Would its fix strategy help here?
- Are the conditions/constraints similar?

Return STRICT JSON (no markdown):
{{
  "scores": [95, 80, 70, 65, 60, ...]
}}

IMPORTANT: Return exactly {len(candidates)} scores in the same order as candidates.
"""

        try:
            result = self.llm.invoke(prompt)
            response = getattr(result, "content", str(result)).strip()

            # Extract JSON
            first_brace = response.find("{")
            last_brace = response.rfind("}")
            if first_brace != -1 and last_brace != -1:
                response = response[first_brace:last_brace + 1]

            scores_data = json.loads(response)
            scores = scores_data.get("scores", [])

            # Combine scores with candidates and sort
            scored_candidates = []
            for i, candidate in enumerate(candidates):
                score = scores[i] if i < len(scores) else 0
                scored_candidates.append((score, candidate))

            # Sort by LLM score (descending) and return top_k
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored_candidates[:top_k]]

        except Exception as e:
            logger.warning(f"[Memory] LLM re-ranking failed: {e}, falling back to embedding scores")
            return candidates[:top_k]

    def retrieve(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hybrid retrieval: Fast embedding search (top 20) + LLM re-ranking (top 5).

        Stage 1: Embedding-based similarity (fast, broad)
        Stage 2: LLM semantic re-ranking (precise, focused)
        """
        if not self.enabled:
            return self._empty_result(query, "memory_disabled")

        # Stage 1: Fast embedding search - get top 20 candidates per level
        if self.memory_backend == "chroma":
            l1_candidates = self._query_collection("L1", query) if "L1" in self.active_levels else []
            l2_candidates = self._query_collection("L2", query) if "L2" in self.active_levels else []
            l3_candidates = self._query_collection("L3", query) if "L3" in self.active_levels else []
        else:
            # Get all candidates from embedding search (already sorted by similarity)
            l1_candidates = self._retrieve_l1(query) if "L1" in self.active_levels else []
            l2_candidates = self._retrieve_l2(query) if "L2" in self.active_levels else []
            l3_candidates = self._retrieve_l3(query) if "L3" in self.active_levels else []

        # Stage 2: LLM re-rank top 20 to top 5 (only if LLM is available)
        if self.llm:
            l1 = self._llm_rerank(query, l1_candidates[:20], top_k=5)
            l2 = self._llm_rerank(query, l2_candidates[:20], top_k=3)
            l3 = self._llm_rerank(query, l3_candidates[:20], top_k=2)
        else:
            # Fallback: just take top results from embedding search
            l1 = l1_candidates[:5]
            l2 = l2_candidates[:3]
            l3 = l3_candidates[:2]

        best_scores = {
            "L1": round(max((float(row.get("similarity_score", 0.0)) for row in l1), default=0.0), 4),
            "L2": round(max((float(row.get("similarity_score", 0.0)) for row in l2), default=0.0), 4),
            "L3": round(max((float(row.get("similarity_score", 0.0)) for row in l3), default=0.0), 4),
        }
        weighted_similarity = round(
            sum(self.level_weights[level] * best_scores[level] for level in ("L1", "L2", "L3")),
            4,
        )

        # Candidate files: prefer LLM's suspected_files, fall back to L1/L2 paths
        candidate_files: List[str] = []
        for row in l1:
            path = _normalize_path(row.get("file", ""))
            if path and path not in candidate_files:
                candidate_files.append(path)
        for row in l2:
            for file_row in (row.get("files", []) or row.get("modified_files", [])):
                path = _normalize_path(file_row.get("file", ""))
                if path and path not in candidate_files:
                    candidate_files.append(path)

        # High-level hints: coarse retrieval only; file-specific refinement happens later in FL.
        high_level_hints: List[str] = []
        for row in l2:
            # Use new field name with backward compat fallback
            reason = str(row.get("overall_failure_reason") or row.get("failure_reason") or "")
            if reason:
                high_level_hints.append(_clip(reason, 220))
        for row in l3:
            principle = str(row.get("principle") or row.get("fix_strategy") or "")
            if principle:
                high_level_hints.append(_clip(principle, 220))

        result = {
            "enabled": True,
            "query": {
                "task_id": query.get("task_id"),
                "sha_fail": query.get("sha_fail"),
                "repo": query.get("repo"),
                "workflow_path": query.get("workflow_path"),
                "error_type": query.get("error_type"),
                "failure_pattern": query.get("failure_pattern"),
                "overall_failure_reason": query.get("failure_reason") or query.get("error_context_summary") or "",
                "relevant_files": query.get("relevant_files", []) or [],
                "changed_files": query.get("changed_files", []) or [],
                "failed_cmd": query.get("failed_cmd", []) or [],
                "failed_tool": query.get("failed_tool", []) or [],
            },
            "query_file_details": query.get("relevant_files_details", []) or [],
            "thresholds": {
                "similarity_threshold": self.similarity_threshold,
                **self.level_thresholds,
            },
            "weights": dict(self.level_weights),
            "level_scores": best_scores,
            "weighted_similarity": weighted_similarity,
            "selected_memory_levels": [
                lvl for lvl, rows in (("L1", l1), ("L2", l2), ("L3", l3)) if rows
            ],
            "candidate_files": candidate_files,
            "high_level_hints": high_level_hints,
            "l1_matches": l1,
            "l2_matches": l2,
            "l3_matches": l3,
            "matches": [*l1, *l2, *l3],
            "retrieval_method": "hybrid" if self.llm else "embedding_only",
        }
        self._append_jsonl(self.retrieval_log_path, result)
        # Persist any newly computed embeddings back to disk so next run skips re-embedding
        self._persist_new_embeddings()
        return result

    def generate_repair_plan(
        self,
        issue: Dict[str, Any],
        retrieved_memories: Dict[str, Any],
        validation_sequence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate structured CI repair plan from retrieved memories.

        Input:
        - issue: Current CI failure details
        - retrieved_memories: Result from retrieve() with L1/L2/L3 matches
        - validation_sequence: CI workflow validation steps

        Output: Structured plan with:
        - problem_statement: For CI document
        - root_causes: Identified causes with evidence
        - repair_steps: Step-by-step fixes (WHAT, WHERE, HOW, WHY, VERIFY)
        - verification_order: Sequence to follow
        """
        if not self.llm:
            return {"error": "LLM not available for plan generation"}

        query = retrieved_memories.get("query", {})
        l1_matches = retrieved_memories.get("l1_matches", [])[:3]
        l2_matches = retrieved_memories.get("l2_matches", [])[:2]
        l3_matches = retrieved_memories.get("l3_matches", [])[:2]

        # Compact memory summary for LLM
        l1_summary = [
            {
                "file": m.get("file", ""),
                "problem": _clip(str(m.get("problem", "")), 150),
                "fix": _clip(str(m.get("fix_strategy", "")), 150),
            }
            for m in l1_matches
        ]

        l2_summary = [
            {
                "issue_type": m.get("issue_type", ""),
                "atomic_problems": [
                    {
                        "type": p.get("issue_type", ""),
                        "symptom": _clip(str(p.get("symptom", "")), 100),
                        "ci_stage": p.get("ci_stage", ""),
                    }
                    for p in (m.get("atomic_problems", []) or [])[:3]
                ],
            }
            for m in l2_matches
        ]

        l3_summary = [
            {
                "issue_type": m.get("issue_type", ""),
                "problem": _clip(str(m.get("problem", "")), 200),
                "fix_strategy": _clip(str(m.get("universal_fix_strategy", "")), 200),
            }
            for m in l3_matches
        ]

        prompt = f"""You are a CI repair expert. Generate a structured repair plan for this CI failure.

CI FAILURE:
- Repo: {query.get("repo", "")}
- Workflow: {query.get("workflow_path", "")}
- Failed Command: {" ".join(_safe_list(query.get("failed_cmd", [])))}
- Error: {_clip(str(query.get("overall_failure_reason", "")), 300)}

VALIDATION SEQUENCE (CI workflow order):
{json.dumps(validation_sequence, indent=2)[:1000]}

RETRIEVED MEMORIES FROM SIMILAR PAST FAILURES:

L1 (File-level fixes):
{json.dumps(l1_summary, indent=2)[:1500]}

L2 (Issue-level patterns):
{json.dumps(l2_summary, indent=2)[:1500]}

L3 (Universal patterns):
{json.dumps(l3_summary, indent=2)[:1500]}

YOUR TASK:
Generate a structured repair plan that:
1. Identifies WHAT needs to be fixed (root causes)
2. Specifies WHERE to fix (files, lines, sections)
3. Explains HOW to fix (specific changes)
4. Lists WHAT TO VERIFY after each fix
5. RESPECTS validation sequence order (fix P1 before P2 appears)

Return STRICT JSON (no markdown, no extra text):
{{
  "problem_statement": "Complete problem description for CI document. Be comprehensive.",
  "root_causes": [
    {{
      "cause": "Root cause description",
      "evidence": "Evidence from logs/errors",
      "validation_stage": 1
    }}
  ],
  "repair_steps": [
    {{
      "step_id": 1,
      "what_to_fix": "Specific problem description",
      "where_to_fix": "File path and location (e.g., pyproject.toml line 189)",
      "how_to_fix": "Exact change to make",
      "why_this_fixes": "Explanation of why this resolves the issue",
      "verify_by": "Command to verify (e.g., 'uv install' should succeed)",
      "validation_stage": 1,
      "depends_on": []
    }},
    {{
      "step_id": 2,
      "what_to_fix": "Next problem",
      "where_to_fix": "File location",
      "how_to_fix": "Change details",
      "why_this_fixes": "Reason",
      "verify_by": "Verification command",
      "validation_stage": 2,
      "depends_on": [1]
    }}
  ],
  "verification_order": [1, 2, 3],
  "estimated_complexity": "low|medium|high"
}}
"""

        try:
            result = self.llm.invoke(prompt)
            response = getattr(result, "content", str(result)).strip()

            # Extract JSON
            first_brace = response.find("{")
            last_brace = response.rfind("}")
            if first_brace != -1 and last_brace != -1:
                response = response[first_brace:last_brace + 1]

            plan = json.loads(response)
            return plan

        except Exception as e:
            logger.warning(f"[Memory] Plan generation failed: {e}")
            return {
                "problem_statement": str(query.get("overall_failure_reason", "CI failure")),
                "root_causes": [],
                "repair_steps": [],
                "verification_order": [],
                "estimated_complexity": "unknown",
                "error": str(e),
            }

    def _empty_result(
        self,
        query: Dict[str, Any],
        reason: str,
        level_scores: Optional[Dict[str, float]] = None,
        weighted_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "reason": reason,
            "query": {
                "task_id": query.get("task_id"),
                "sha_fail": query.get("sha_fail"),
                "repo": query.get("repo"),
                "error_type": query.get("error_type"),
            },
            "thresholds": {
                "similarity_threshold": self.similarity_threshold,
                **self.level_thresholds,
            },
            "weights": dict(self.level_weights),
            # Preserve actual computed scores even when suppressed so ablation
            # analysis can see how close each issue was to the threshold.
            "level_scores": level_scores if level_scores is not None else {"L1": 0.0, "L2": 0.0, "L3": 0.0},
            "weighted_similarity": weighted_similarity,
            "selected_memory_levels": [],
            "candidate_files": [],
            "high_level_hints": [],
            "l1_matches": [],
            "l2_matches": [],
            "l3_matches": [],
            "matches": [],
        }
        if self.enabled:
            self._append_jsonl(self.retrieval_log_path, result)
        return result

    def _retrieve_l1(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        repo = str(query.get("repo") or "")
        workflow = str(query.get("workflow_path") or query.get("workflow_text") or "")
        error_type = str(query.get("error_type") or "").lower()
        failure_pattern = str(query.get("failure_pattern") or "").lower()
        query_files = query.get("relevant_files", []) + query.get("changed_files", [])
        query_tools = [str(x).lower() for x in query.get("failed_tool", [])]
        query_cmds = [str(x).lower() for x in query.get("failed_cmd", [])]
        query_reason = str(query.get("failure_reason") or query.get("error_context_summary") or "")

        # Per-file detail map: file path → {issue_type, failed_cmd, failed_tool, reason}
        # from the log analyzer's per-file output. Used to enrich the query doc when
        # scoring L1 rows that match a specific file.
        file_details_map: Dict[str, Dict[str, Any]] = {
            d["file"]: d
            for d in (query.get("relevant_files_details") or [])
            if d.get("file")
        }

        # L1 query document — structured per-file features ONLY.
        # No narrative failure_reason — just the measurable signals:
        #   file_path | error_type | failure_pattern | issue_type | failed_tool | failed_cmd
        # Overall error_context is NOT included at L1 — L1 is file-precision level.
        # fix_direction/fix_pattern excluded — unknown at query time.
        query_issue_type = _to_descriptive_issue_type(failure_pattern, error_type).lower()
        base_query_doc = " | ".join(x for x in [
            error_type,
            failure_pattern,
            query_issue_type,
            " ".join(query_tools),
            " ".join(query_cmds),
        ] if x)

        # Normalized set of query file paths for O(1) file_score lookup
        query_file_norms = {_normalize_path(p) for p in query_files}

        scored: List[Dict[str, Any]] = []
        for row in self.failure_memory:
            if row.get("sha_fail") == query.get("sha_fail"):
                continue
            # L1: Filter by repo AND workflow to keep searches within the same repo+workflow context
            # Use fuzzy repo matching to handle "owner/repo" vs "repo" format inconsistencies
            if repo:
                row_repo = str(row.get("repo") or "")
                if not _repo_matches(repo, row_repo):
                    continue
            if workflow:
                row_workflow = str(row.get("workflow_path") or row.get("workflow_name") or "")
                if row_workflow and row_workflow != workflow:
                    continue

            row_error = str(row.get("error_type") or "").lower()
            row_pattern = str(row.get("failure_pattern") or "").lower()
            row_issue_type = str(row.get("issue_type") or "").lower()
            row_tools = [str(x).lower() for x in _safe_list(row.get("failed_tool", []))]
            row_cmds = [str(x).lower() for x in _safe_list(row.get("failed_cmd", []))]

            # Exact normalized path match — highest weight in L1.
            row_file_norm = _normalize_path(str(row.get("file") or ""))
            file_score = 1.0 if row_file_norm and row_file_norm in query_file_norms else 0.0

            # L1 row document: use stored search_document if present (richer, and
            # ensures the cache key matches _persist_new_embeddings lookup).
            # Fall back to compact inline doc for records that pre-date storage.
            row_doc = str(row.get("search_document") or "").strip() or " | ".join(x for x in [
                row_file_norm,
                row_error,
                row_pattern,
                row_issue_type,
                " ".join(row_tools),
                " ".join(row_cmds),
            ] if x)

            # Per-file enrichment: if this row's file has its own details from the log
            # analyzer, build a file-specific query_doc using that file's exact values
            # (issue_type, failed_tool, failed_cmd) for a more precise embedding match.
            file_detail = file_details_map.get(row_file_norm, {}) if row_file_norm else {}
            if file_detail:
                effective_tools = [str(x).lower() for x in _safe_list(file_detail.get("failed_tool") or [])] or query_tools
                effective_cmds = [str(x).lower() for x in _safe_list(file_detail.get("failed_cmd") or [])] or query_cmds
                effective_issue_type = str(file_detail.get("issue_type") or "").lower() or query_issue_type
                # Per-file query_doc: file_path | error | pattern | issue_type | tools | cmds
                query_doc = " | ".join(x for x in [
                    row_file_norm,
                    error_type,
                    failure_pattern,
                    effective_issue_type,
                    " ".join(effective_tools),
                    " ".join(effective_cmds),
                ] if x)
            else:
                effective_tools = query_tools
                effective_cmds = query_cmds
                query_doc = base_query_doc

            # Single holistic cosine similarity — one embedding, captures all semantic fields
            # (error_type + failure_pattern + issue_type + failure_reason) together.
            semantic_score = _semantic_similarity(query_doc, row_doc) if (query_doc and row_doc) else 0.0

            # Text-based: Jaccard on tools OR cmds separately — score if EITHER matches.
            # Not combined: a match on tool alone (e.g. 'pytest') is sufficient signal.
            tool_score = _jaccard(effective_tools, row_tools)
            cmd_score = _jaccard(effective_cmds, row_cmds)
            tool_cmd_score = max(tool_score, cmd_score)

            similarity = round(
                0.45 * file_score
                + 0.50 * semantic_score
                + 0.05 * tool_cmd_score,
                4,
            )
            scored.append(
                {
                    **row,
                    "memory_level": "L1",
                    "similarity_score": similarity,
                    "matched_on": {
                        "file_score": round(file_score, 4),
                        "semantic_score": round(semantic_score, 4),
                        "tool_score": round(tool_score, 4),
                        "cmd_score": round(cmd_score, 4),
                    },
                }
            )

        return self._similar_matches(scored)

    def _retrieve_l2(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        repo = str(query.get("repo") or "")
        workflow = str(query.get("workflow_path") or query.get("workflow_text") or "")
        error_type = str(query.get("error_type") or "").lower()
        failure_pattern = str(query.get("failure_pattern") or "").lower()
        query_tools = [str(x).lower() for x in query.get("failed_tool", [])]
        query_cmds = [str(x).lower() for x in query.get("failed_cmd", [])]
        # overall error_context — the holistic narrative of why CI failed
        query_reason = str(query.get("failure_reason") or query.get("error_context_summary") or "")
        query_issue_type = _to_descriptive_issue_type(failure_pattern, error_type).lower()

        # Build per-file structured block from relevant_files_details.
        # For each file: file_path + issue_type + per-file tools + per-file cmds.
        # This is the "PER FILE INFO" part of the L2 query:
        #   (file_path | error | pattern | tools | cmds | issue_type) per file, aggregated.
        query_per_file_parts: List[str] = []
        for d in (query.get("relevant_files_details") or []):
            f_path = d.get("file", "")
            f_issue = str(d.get("issue_type") or "").strip()
            f_tools = " ".join(str(t).lower() for t in _safe_list(d.get("failed_tool") or []))
            f_cmds  = " ".join(str(c).lower() for c in _safe_list(d.get("failed_cmd") or []))
            entry = " ".join(x for x in [f_path, error_type, failure_pattern, f_tools, f_cmds, f_issue] if x)
            if entry:
                query_per_file_parts.append(entry)
        query_per_file_block = " | ".join(query_per_file_parts)

        # L2 query document:
        #   PER FILE INFO (file_path | error | pattern | tools | cmds | issue_type) per file
        #   + overall error_context narrative
        #   + issue-level tools / cmds as fallback
        # fix_approach excluded: unknown at query time.
        query_doc = " | ".join(x for x in [
            query_per_file_block,
            error_type,
            failure_pattern,
            query_issue_type,
            query_reason,
            " ".join(query_tools),
            " ".join(query_cmds),
        ] if x)

        scored: List[Dict[str, Any]] = []
        for row in self.repo_memory:
            # L2: Filter by repo AND workflow to keep searches within the same repo+workflow context
            # Use fuzzy repo matching to handle "owner/repo" vs "repo" format inconsistencies
            if repo:
                row_repo = str(row.get("repo") or "")
                if not _repo_matches(repo, row_repo):
                    continue
            if workflow:
                row_workflow = str(row.get("workflow_path") or row.get("workflow_name") or "")
                if row_workflow and row_workflow != workflow:
                    continue

            row_error = str(row.get("error_type") or "").lower()
            row_pattern = str(row.get("failure_pattern") or row.get("pattern_name") or "").lower()
            row_issue_type = str(row.get("issue_type") or "").lower()
            # overall_failure_reason = the stored error_context for this past issue
            row_overall_reason = str(row.get("overall_failure_reason") or row.get("failure_reason") or "")
            row_tools = [str(x).lower() for x in _safe_list(row.get("failed_tool", []))]
            row_cmds = [str(x).lower() for x in _safe_list(row.get("failed_cmd", []))]

            # Build per-file structured block from the stored L2 files list.
            # Mirrors query_per_file_block: file_path + error + pattern + issue_type per file.
            row_per_file_parts: List[str] = []
            for f in _safe_list(row.get("files") or []):
                f_path    = f.get("file", "")
                f_issue   = str(f.get("issue_type") or "").strip()
                f_pattern = str(f.get("failure_pattern") or "").strip()
                entry = " ".join(x for x in [f_path, row_error, f_pattern or row_pattern, f_issue] if x)
                if entry:
                    row_per_file_parts.append(entry)
            row_per_file_block = " | ".join(row_per_file_parts)

            # L2 row document: use stored search_document if present.
            row_doc = str(row.get("search_document") or "").strip() or " | ".join(x for x in [
                row_per_file_block,
                row_error,
                row_pattern,
                row_issue_type,
                row_overall_reason,
                " ".join(row_tools),
                " ".join(row_cmds),
            ] if x)

            # Single holistic cosine similarity across all semantic failure fields.
            semantic_score = _semantic_similarity(query_doc, row_doc) if (query_doc and row_doc) else 0.0

            # Text-based: score if EITHER tool OR cmd matches — not required to match both.
            tool_score = _jaccard(query_tools, row_tools)
            cmd_score = _jaccard(query_cmds, row_cmds)
            tool_cmd_score = max(tool_score, cmd_score)

            similarity = round(
                0.90 * semantic_score + 0.10 * tool_cmd_score,
                4,
            )
            scored.append(
                {
                    **row,
                    "memory_level": "L2",
                    "similarity_score": similarity,
                    "matched_on": {
                        "semantic_score": round(semantic_score, 4),
                        "tool_score": round(tool_score, 4),
                        "cmd_score": round(cmd_score, 4),
                    },
                }
            )

        return self._similar_matches(scored)

    def _retrieve_l3(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # L3: Cross-repo generalized retrieval — NO repo or workflow filtering
        # Searches across all cross_memory records to find generalized principles
        # that apply across different repositories and workflows
        error_type = str(query.get("error_type") or "").lower()
        failure_pattern = str(query.get("failure_pattern") or "").lower()
        query_tools = [str(x).lower() for x in query.get("failed_tool", [])]
        query_cmds = [str(x).lower() for x in query.get("failed_cmd", [])]
        # overall error_context — the holistic narrative of why CI failed
        query_reason = str(query.get("failure_reason") or query.get("error_context_summary") or "")
        query_issue_type = _to_descriptive_issue_type(failure_pattern, error_type).lower()

        # Build per-file structured block — same logic as L2 (shared across query side).
        # For each file: file_path + error + pattern + tools + cmds + issue_type.
        query_per_file_parts: List[str] = []
        for d in (query.get("relevant_files_details") or []):
            f_path  = d.get("file", "")
            f_issue = str(d.get("issue_type") or "").strip()
            f_tools = " ".join(str(t).lower() for t in _safe_list(d.get("failed_tool") or []))
            f_cmds  = " ".join(str(c).lower() for c in _safe_list(d.get("failed_cmd") or []))
            entry = " ".join(x for x in [f_path, error_type, failure_pattern, f_tools, f_cmds, f_issue] if x)
            if entry:
                query_per_file_parts.append(entry)
        query_per_file_block = " | ".join(query_per_file_parts)

        # L3 query document:
        #   PER FILE INFO (file_path | error | pattern | tools | cmds | issue_type) per file
        #   + overall error_context narrative
        #   + issue-level tools / cmds as fallback
        # fix_strategies excluded: unknown at query time.
        query_doc = " | ".join(x for x in [
            query_per_file_block,
            error_type,
            failure_pattern,
            query_issue_type,
            query_reason,
            " ".join(query_tools),
            " ".join(query_cmds),
        ] if x)

        scored: List[Dict[str, Any]] = []
        # L3: No repo/workflow filtering — searches all cross-memory for generalized patterns
        for row in self.cross_memory:
            row_error = str(row.get("error_type") or "").lower()
            row_issue_type = str(row.get("issue_type") or "").lower()
            # failure_pattern or issue_type — L3 records may use either
            row_pattern = str(row.get("failure_pattern") or row_issue_type).lower()
            # all_failure_patterns: accumulated keyword variants across issues
            row_all_patterns = " | ".join(
                str(p) for p in _safe_list(row.get("failure_patterns", [])) if str(p).strip()
            )
            # failure_reasons: overall + per-file reasons accumulated across issues
            row_reasons = " | ".join(
                str(r) for r in _safe_list(row.get("failure_reasons", [])) if str(r).strip()
            )
            row_tools = [str(x).lower() for x in _safe_list(row.get("failed_tool", []))]
            row_cmds = [str(x).lower() for x in _safe_list(row.get("failed_cmd", []))]

            # Build per-file structured block from stored example_files.
            # example_files: [{file, issue_type, failure_pattern}] — concrete file examples
            # accumulated from past issues of the same error_type.
            row_per_file_parts: List[str] = []
            for f in _safe_list(row.get("example_files") or []):
                f_path    = f.get("file", "")
                f_issue   = str(f.get("issue_type") or "").strip()
                f_pattern = str(f.get("failure_pattern") or "").strip()
                entry = " ".join(x for x in [f_path, row_error, f_pattern or row_pattern, f_issue] if x)
                if entry:
                    row_per_file_parts.append(entry)
            row_per_file_block = " | ".join(row_per_file_parts)

            # L3 row document:
            #   PER FILE INFO (file_path | error | pattern | issue_type) from example_files
            #   + error_type | issue_type | all_failure_patterns
            #   + failure_reasons (overall + per-file, accumulated across issues)
            #   + tools | cmds
            # L3 row document: use stored search_document if present.
            row_doc = str(row.get("search_document") or "").strip() or " | ".join(x for x in [
                row_per_file_block,
                row_error,
                row_issue_type,
                row_pattern,
                row_all_patterns,
                row_reasons,
                " ".join(row_tools),
                " ".join(row_cmds),
            ] if x)

            # Single holistic cosine similarity across all semantic failure fields.
            semantic_score = _semantic_similarity(query_doc, row_doc) if (query_doc and row_doc) else 0.0

            # Text-based: score if EITHER tool OR cmd matches — not required to match both.
            tool_score = _jaccard(query_tools, row_tools)
            cmd_score = _jaccard(query_cmds, row_cmds)
            tool_cmd_score = max(tool_score, cmd_score)

            similarity = round(
                0.90 * semantic_score + 0.10 * tool_cmd_score,
                4,
            )
            scored.append(
                {
                    **row,
                    "memory_level": "L3",
                    "similarity_score": similarity,
                    "matched_on": {
                        "semantic_score": round(semantic_score, 4),
                        "tool_score": round(tool_score, 4),
                        "cmd_score": round(cmd_score, 4),
                    },
                }
            )

        return self._similar_matches(scored)

    def retrieve_for_file(
        self,
        file_path: str,
        file_context: str,
        issue_query: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Per-file memory retrieval pipeline (Steps 1–8).

        For each file being analyzed in FL, this method:
          Step 1  — builds a file-enriched query from the current failure context
                    (file_path + error_type + failure_pattern + failure_reason)
          Step 2  — generates a dense embedding of the combined file context
          Step 3  — retrieves similar entries from L1 / L2 / L3 via cosine similarity
          Step 4  — computes weighted similarity score across levels
          Step 5  — ranks retrieved entries highest → lowest similarity
          Steps 6–8 — (lazy) the result is passed to format_for_file_prompt() /
                      analyze_relevance_for_file() which call the LLM to select
                      only the memories relevant to this specific file and failure.
        """
        if not self.enabled:
            return self._empty_result(issue_query, "memory_disabled")

        # Step 1: File-specific query.
        # relevant_files = [this file only] so L1 file_score targets exactly this file.
        # changed_files cleared — it would otherwise contain ALL commit-changed files and
        # cause L1 to match records for unrelated files with the same file_score=1.0.
        file_query = dict(issue_query)
        norm_file_path = _normalize_path(file_path)
        file_query["relevant_files"] = [norm_file_path]
        file_query["changed_files"] = []
        file_query["file_path"] = norm_file_path

        # Enrich with per-file details from the log analyzer (issue_type, failed_cmd,
        # failed_tool, reason) when available for this specific file.
        # This makes the query doc targeted to this file rather than the whole issue,
        # improving L1 semantic matching for file-specific failure modes.
        file_details_map: Dict[str, Dict[str, Any]] = {
            d["file"]: d
            for d in (issue_query.get("relevant_files_details") or [])
            if d.get("file")
        }
        file_detail = file_details_map.get(norm_file_path, {})

        if file_detail:
            # Override tool/cmd with file-specific values (more precise than issue-level)
            if file_detail.get("failed_tool"):
                file_query["failed_tool"] = _safe_list(file_detail["failed_tool"])
            if file_detail.get("failed_cmd"):
                file_query["failed_cmd"] = _safe_list(file_detail["failed_cmd"])
            # Prepend file-specific issue_type to failure_reason so the embedding
            # is anchored to this file's exact failure mode, not the aggregate issue.
            if file_detail.get("issue_type"):
                existing_reason = file_query.get("failure_reason", "")
                file_issue_type = file_detail["issue_type"]
                file_query["failure_reason"] = (
                    f"{file_issue_type} | {existing_reason}"
                    if existing_reason else file_issue_type
                )

        # Further enrich failure_reason with file-level semantic context (code snippet)
        # so the embedding captures both issue and file content signals.
        existing_reason = file_query.get("failure_reason", "")
        file_snippet = _clip(file_context, 800)
        file_query["failure_reason"] = (
            f"{existing_reason} | file: {norm_file_path} | {file_snippet}"
            if file_snippet else existing_reason
        )

        # Steps 2–5: Embedding Generation → Cosine Similarity → Scoring → Ranking
        if self.memory_backend == "chroma":
            l1 = self._query_collection("L1", file_query) if "L1" in self.active_levels else []
            l2 = self._query_collection("L2", file_query) if "L2" in self.active_levels else []
            l3 = self._query_collection("L3", file_query) if "L3" in self.active_levels else []
        else:
            l1 = self._retrieve_l1(file_query) if "L1" in self.active_levels else []
            l2 = self._retrieve_l2(file_query) if "L2" in self.active_levels else []
            l3 = self._retrieve_l3(file_query) if "L3" in self.active_levels else []

        best_scores = {
            "L1": round(max((float(r.get("similarity_score", 0.0)) for r in l1), default=0.0), 4),
            "L2": round(max((float(r.get("similarity_score", 0.0)) for r in l2), default=0.0), 4),
            "L3": round(max((float(r.get("similarity_score", 0.0)) for r in l3), default=0.0), 4),
        }
        weighted_similarity = round(
            sum(self.level_weights[lvl] * best_scores[lvl] for lvl in ("L1", "L2", "L3")), 4
        )

        # No threshold gate — all retrieved entries are ranked and passed to the LLM.
        # The LLM uses similarity scores to decide what is relevant for this file.

        # Candidate files from L1 direct matches + L2 modified-file entries
        candidate_files: List[str] = []
        for row in l1:
            path = _normalize_path(row.get("file", ""))
            if path and path not in candidate_files:
                candidate_files.append(path)
        for row in l2:
            for file_row in (row.get("modified_files", []) or row.get("files", [])):
                path = _normalize_path(file_row.get("file", ""))
                if path and path not in candidate_files:
                    candidate_files.append(path)

        high_level_hints: List[str] = []
        for row in l2:
            # Use overall_failure_reason (new field) with fallback to failure_reason (legacy)
            reason = str(row.get("overall_failure_reason") or row.get("failure_reason") or "")
            if reason:
                high_level_hints.append(_clip(reason, 220))
        for row in l3:
            principle = str(row.get("principle") or row.get("fix_strategy") or "")
            if principle:
                high_level_hints.append(_clip(principle, 220))

        result = {
            "enabled": True,
            "query": {
                "task_id": file_query.get("task_id"),
                "sha_fail": file_query.get("sha_fail"),
                "repo": file_query.get("repo"),
                "error_type": file_query.get("error_type"),
                "failure_pattern": file_query.get("failure_pattern"),
                "file_path": file_path,
                "overall_failure_reason": file_query.get("failure_reason") or file_query.get("error_context_summary") or "",
            },
            "query_file_details": file_query.get("relevant_files_details", []) or [],
            "thresholds": {
                "similarity_threshold": self.similarity_threshold,
                **self.level_thresholds,
            },
            "weights": dict(self.level_weights),
            "level_scores": best_scores,
            "weighted_similarity": weighted_similarity,
            "selected_memory_levels": [
                lvl for lvl, rows in (("L1", l1), ("L2", l2), ("L3", l3)) if rows
            ],
            "candidate_files": candidate_files,
            "high_level_hints": high_level_hints,
            "l1_matches": l1,
            "l2_matches": l2,
            "l3_matches": l3,
            "matches": [*l1, *l2, *l3],
        }
        print(
            f"[Memory] file={_normalize_path(file_path)} "
            f"weighted_similarity={weighted_similarity} "
            f"retrieved_levels={result['selected_memory_levels']} "
            f"l1={len(l1)} l2={len(l2)} l3={len(l3)}"
        )
        self._append_jsonl(self.retrieval_log_path, result)
        return result

    def augment_suspicious_files(
        self,
        suspicious_files: List[Dict[str, Any]],
        changed_files_info: Optional[Dict[str, Any]],
        retrieval_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidate_files = retrieval_result.get("candidate_files", []) or []
        if not candidate_files:
            return suspicious_files

        existing = {_normalize_path(item.get("file") or item.get("path") or "") for item in suspicious_files}
        existing.discard("")

        augmented = list(suspicious_files)
        for item in (changed_files_info or {}).get("changed_files", []) or []:
            path = _normalize_path(item.get("file_path", "")) if isinstance(item, dict) else ""
            if not path or path in existing:
                continue
            if path in candidate_files or _basename(path) in {_basename(p) for p in candidate_files}:
                augmented.append({"file": path, "memory_source": "hierarchical_memory"})
                existing.add(path)
        return augmented

    def rank_files(self, candidate_files: List[Dict[str, Any]], retrieval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        l1 = retrieval_result.get("l1_matches", []) or []
        l2 = retrieval_result.get("l2_matches", []) or []

        direct_scores: Dict[str, float] = {}
        for row in l1:
            path = _normalize_path(row.get("file", ""))
            if path:
                direct_scores[path] = max(direct_scores.get(path, 0.0), float(row.get("similarity_score", 0.0)))
        for row in l2:
            boost = float(row.get("similarity_score", 0.0)) * 0.5
            for file_row in (row.get("modified_files", []) or row.get("files", []) or []):
                path = _normalize_path(file_row.get("file", ""))
                if path:
                    direct_scores[path] = max(direct_scores.get(path, 0.0), boost)

        ranked = []
        for index, item in enumerate(candidate_files):
            path = _normalize_path(item.get("file") or item.get("path") or "")
            score = direct_scores.get(path, direct_scores.get(_basename(path), 0.0))
            enriched = dict(item)
            if score:
                enriched["memory_rank_score"] = round(score, 4)
            ranked.append((score, -index, enriched))
        ranked.sort(reverse=True)
        return [item for _, _, item in ranked]

    def _filter_candidates_for_file(
        self,
        file_path: str,
        retrieval_result: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        normalized = _normalize_path(file_path)
        base = _basename(normalized)
        l1_rows = [
            row for row in (retrieval_result.get("l1_matches", []) or [])
            if _normalize_path(row.get("file", "")) == normalized or _basename(row.get("file", "")) == base
        ]
        l2_rows = []
        for row in (retrieval_result.get("l2_matches", []) or []):
            files = row.get("modified_files", []) or row.get("files", []) or []
            if any(_normalize_path(item.get("file", "")) == normalized or _basename(item.get("file", "")) == base for item in files):
                l2_rows.append(row)
        if not l2_rows:
            l2_rows = retrieval_result.get("l2_matches", []) or []
        l3_rows = retrieval_result.get("l3_matches", []) or []
        return {"l1": l1_rows, "l2": l2_rows, "l3": l3_rows}

    def _build_file_level_analysis_prompt(
        self,
        *,
        file_path: str,
        file_context: str,
        retrieval_result: Dict[str, Any],
        candidates: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        query = retrieval_result.get("query", {}) or {}
        query_file_details = [
            item for item in (retrieval_result.get("query_file_details") or [])
            if isinstance(item, dict) and _normalize_path(str(item.get("file", ""))) == _normalize_path(file_path)
        ]
        current_file_detail = query_file_details[0] if query_file_details else {}
        current_file_payload = {
            "file": file_path,
            "issue_type": current_file_detail.get("issue_type", ""),
            "line_number": current_file_detail.get("line_number"),
            "failed_tool": _safe_list(current_file_detail.get("failed_tool", [])),
            "failed_cmd": _safe_list(current_file_detail.get("failed_cmd", [])),
            "file_failure_reason": current_file_detail.get("reason", ""),
            "overall_failure_reason": query.get("overall_failure_reason", "") or query.get("failure_reason", "") or "",
        }

        def _summarize_row(level: str, idx: int, row: Dict[str, Any]) -> str:
            row_file = row.get("file", "")
            files = row.get("modified_files", []) or row.get("files", []) or []
            files_text = ", ".join(str(item.get("file", "")) for item in files if isinstance(item, dict))
            reason = row.get("failure_reason") or row.get("reason") or row.get("principle") or ""
            fix = row.get("fix_pattern") or row.get("fix_strategies") or row.get("fix_strategy") or ""

            # Base summary
            base = (
                f"  [{level}-{idx}] score={row.get('similarity_score', 0.0):.2f} "
                f"record_id={row.get('record_id', '')} "
                f"error_type={row.get('error_type', '')} "
                f"failure_pattern={row.get('failure_pattern') or row.get('issue_type') or row.get('pattern_name', '')}\n"
                f"      file={row_file}\n"
                f"      files={files_text}\n"
                f"      reason={_clip(str(reason), 260)}\n"
                f"      fix={_clip(str(fix), 220)}"
            )

            # FOR L2: Add full repair trajectory
            if level == "L2":
                traj = row.get("repair_trajectory", [])
                sequence = traj.get("sequence", []) if isinstance(traj, dict) else (traj if isinstance(traj, list) else [])

                if sequence:
                    traj_lines = [f"\n      TRAJECTORY ({len(sequence)} steps):"]
                    for s in sequence:
                        step_num = s.get("step", "?")
                        action = _clip(str(s.get("action", s.get("fix", ""))), 70)
                        result = _clip(str(s.get("result", s.get("expected_outcome", ""))), 50)
                        ci_stage = s.get("ci_stage", "")
                        traj_lines.append(f"        {step_num}[{ci_stage}]: {action} → {result}")
                    base += "\n".join(traj_lines)

                # Add atomic problems
                problems = row.get("atomic_problems", []) or row.get("all_problems", [])
                if problems:
                    prob_lines = [f"\n      PROBLEMS ({len(problems)}):"]
                    for p in problems:
                        p_num = p.get("problem_number", "?")
                        vis = "vis" if p.get("is_first_failure") else "hid"
                        symptom = _clip(str(p.get("problem_identification", {}).get("symptom", p.get("symptom", ""))), 50)
                        prob_lines.append(f"        P{p_num}[{vis}]: {symptom}")
                    base += "\n".join(prob_lines)

            return base

        candidate_lines: List[str] = []
        for level_key, rows in (("L1", candidates["l1"]), ("L2", candidates["l2"]), ("L3", candidates["l3"])):
            if rows:
                for idx, row in enumerate(rows):
                    candidate_lines.append(_summarize_row(level_key, idx, row))
            else:
                candidate_lines.append(f"  [{level_key}] none")

        candidate_block = "\n".join(candidate_lines)
        return f"""You are a TRAJECTORY-AWARE memory relevance analyst for CI fault localization.

Your task:
- Memory candidates below were retrieved by similarity (highest → lowest).
- Analyze them against CURRENT FAILURE and predict FUTURE FAILURES.
- Use similarity as a signal, but make your own relevance judgement.

CRITICAL FOR L2 MEMORY WITH TRAJECTORIES:
- L2 contains REPAIR TRAJECTORIES showing sequential failures (Problem 1 → Problem 2 → Problem 3)
- The CURRENT failure may match Problem 1, but you MUST analyze the ENTIRE trajectory
- Understand what will fail NEXT after fixing the current problem
- Identify ALL files that need checking across ALL trajectory steps
- Provide validation sequence (not just first fix)

WHY THIS MATTERS:
- CI stops at first error, so current failure shows only Problem 1
- But the trajectory reveals Problem 2 (hidden, will appear after fixing P1)
- And Problem 3 (hidden, will appear after fixing P2)
- Agent needs to know about ALL problems to fix efficiently

CURRENT FAILURE
- repo: {query.get("repo", "")}
- error_type: {query.get("error_type", "")}
- failure_pattern: {query.get("failure_pattern", "")}
- weighted_similarity: {retrieval_result.get("weighted_similarity", 0.0):.2f}

CURRENT FILE
{file_path}

CURRENT FILE RELEVANT DETAILS
{json.dumps(current_file_payload, indent=2, ensure_ascii=False)}

RETRIEVED CANDIDATES (ranked highest → lowest similarity score)
{candidate_block}

Selection criteria — select a candidate if it provides:
1. A matching or compatible error_type / failure_pattern for the CURRENT FILE
2. A directly applicable fault-localization guidance, fix strategy, or fix direction
3. Dependent files that likely need coordinated changes for this failure
4. Additional repo-level or cross-repo files worth inspecting when indirectly relevant

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAJECTORY-AWARE SELECTION FOR L2 MEMORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L2 contains REPAIR TRAJECTORIES showing sequential failures:
  Problem 1 → fix → Problem 2 appears → fix → Problem 3 appears

Your task is NOT just matching the current problem, but PREDICTING what comes next.

REQUIRED ANALYSIS FOR L2:
1. Match current failure to WHICH STEP in the trajectory
2. Identify ALL SUBSEQUENT problems that will appear after fixing current problem
3. List ALL FILES needed across ENTIRE trajectory (not just current step)
4. Provide VALIDATION SEQUENCE (what to check after each fix)
5. Warn about HIDDEN PROBLEMS that only appear after fixing visible problem

Example:
  Current failure: "uv install error in pyproject.toml"

  Trajectory shows:
    Step 1 [install]: Fix pyproject.toml → install succeeds
    Step 2 [lint]: Will fail on unused imports in src/*.py
    Step 3 [test]: Will fail on missing dependency

  YOU MUST INCLUDE:
    ✓ Current fix: pyproject.toml
    ✓ Next failure: lint will fail on src/main.py, src/utils.py
    ✓ After that: test will fail, need to update dependencies
    ✓ All files to check NOW: pyproject.toml, src/main.py, src/utils.py
    ✓ Validation sequence: uv install → ruff check → pytest

DO NOT just say "fix pyproject.toml" — that's single-problem matching.
DO include the ENTIRE trajectory so the agent knows what's coming.

Reject candidates that do not match the current file details, current failure signals, or do not provide actionable guidance.

Return STRICT JSON only:
{{
  "use_memory": true,
  "similarity_score": 0.0,
  "similarity_reason": "<why the selected memory is relevant for this file>",
  "selected_memory_levels": ["L1"],
  "selected_context": [
    {{
      "score": 0.0,
      "file": "{file_path}",
      "matched_memory_file": "<memory source file if available, else empty string>",
      "failure_pattern": "<compatible pattern>",
      "fl_guidance": "<file-specific fault-localization guidance for the current file>",
      "fix_strategy": "<transferable fix strategy>",
      "next_failures_expected": "<REQUIRED for L2: what will fail AFTER fixing current problem>",
      "validation_sequence": ["<step 1: check this>", "<step 2: then check this>"],
      "all_affected_files_in_trajectory": ["<file1 from step 1>", "<file2 from step 2>"]
    }}
  ],
  "dependent_files": [
    {{
      "file": "file_a",
      "reason": "<why this file likely needs coordinated change>",
      "failure_pattern": "<related pattern>",
      "fix_strategy": "<likely fix strategy>"
    }}
  ],
  "additional_files_to_inspect": [
    {{
      "file": "file_b",
      "reason": "<indirectly relevant file worth checking>",
      "failure_pattern": "<optional related pattern>",
      "fix_strategy": "<optional strategy>",
      "source": "L1|L2|L3",
      "confidence": "low|medium|high"
    }}
  ],
  "selected_items": [
    {{
      "memory_level": "L1|L2|L3",
      "candidate_key": "L1-0",
      "similarity_score": 0.0,
      "relevance": "high|medium|low",
      "justification": "<why this candidate is relevant for this file>",
      "failure_pattern": "<compatible pattern>",
      "failure_reason": "<relevant reason>",

      "trajectory_analysis": {{
        "current_failure_matches_step": 1,
        "total_trajectory_steps": 3,
        "next_failures": [
          {{"step": 2, "stage": "lint", "what_will_fail": "Unused imports", "files": ["src/main.py"]}},
          {{"step": 3, "stage": "test", "what_will_fail": "Missing dependency", "files": ["pyproject.toml"]}}
        ],
        "all_problems_in_trajectory": [
          {{"problem": 1, "visible": true, "symptom": "Install error"}},
          {{"problem": 2, "visible": false, "symptom": "Lint error"}},
          {{"problem": 3, "visible": false, "symptom": "Test error"}}
        ],
        "validation_sequence": [
          "Fix current problem → verify with 'uv install'",
          "EXPECT lint failure → check src/main.py for unused imports",
          "EXPECT test failure → check pyproject.toml for missing deps"
        ],
        "all_files_to_check_now": ["current_file", "file_from_step2", "file_from_step3"],
        "recommended_approach": "Fix all 3 problems at once OR fix sequentially expecting 2 more CI runs"
      }},

      "dependent_files": [
          {{"file": "file_a", "reason": "<why this file is relevant for the current file>"}},
          {{"file": "file_b", "reason": "<why this file is relevant for the current file>"}}
      ],
      "additional_localization_files": [
          {{"file": "file_c", "reason": "<why this file is relevant for the current file>"}}
      ],
      "localization_hint": "<what to inspect in the current file>",
      "fix_direction": "<transferable fix direction>"
    }}
  ],
  "diagnostic_summary": "<2-3 sentence file-specific memory summary>"
}}

Rules:
- If no candidate is useful for this file, return use_memory=false and all lists empty.
- Be file-specific. Do not keep a candidate just because it matches the issue globally.
- Prefer L1 over L2 over L3 when multiple candidates are compatible.
- No markdown fences. No extra keys."""

    def _build_step1_trajectory_analysis_prompt(
        self,
        *,
        file_path: str,
        retrieval_result: Dict[str, Any],
        candidates: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """
        STEP 1: Analyze L2 trajectory to identify all files and next failures.
        """
        query = retrieval_result.get("query", {}) or {}

        # Summarize L2 candidates with trajectories
        l2_summaries = []
        for idx, row in enumerate(candidates.get("l2", [])):
            traj = row.get("repair_trajectory", {})
            sequence = traj.get("sequence", []) if isinstance(traj, dict) else (traj if isinstance(traj, list) else [])
            problems = row.get("atomic_problems", []) or row.get("all_problems", [])

            summary = f"""[L2-{idx}] score={row.get('similarity_score', 0.0):.2f}
  issue_id={row.get('issue_id', '')} repo={row.get('repo', '')}
  overall_failure={_clip(str(row.get('overall_failure', '')), 150)}

  TRAJECTORY ({len(sequence)} steps):"""

            for s in sequence:
                step_num = s.get("step", "?")
                action = _clip(str(s.get("action", s.get("fix", ""))), 80)
                files = s.get("files_modified", s.get("affected_files", []))
                ci_stage = s.get("ci_stage", "")
                result = _clip(str(s.get("result", s.get("expected_outcome", ""))), 60)

                summary += f"\n    Step {step_num} [{ci_stage}]: {action}"
                summary += f"\n      Files: {files}"
                summary += f"\n      Result: {result}"

            summary += f"\n  \n  PROBLEMS ({len(problems)}):"
            for p in problems:
                p_num = p.get("problem_number", "?")
                vis = "visible" if p.get("is_first_failure") else "hidden"
                symptom = _clip(str(p.get("problem_identification", {}).get("symptom", p.get("symptom", ""))), 60)
                ci_stage = p.get("problem_identification", {}).get("ci_stage", p.get("ci_stage", ""))
                affected = p.get("affected_files", [])

                summary += f"\n    P{p_num} [{vis}, {ci_stage}]: {symptom}"
                summary += f"\n      Files: {affected}"

            l2_summaries.append(summary)

        l2_block = "\n\n".join(l2_summaries) if l2_summaries else "[No L2 trajectories]"

        return f"""STEP 1: TRAJECTORY ANALYSIS

You are analyzing CI repair trajectories to identify all affected files and predict next failures.

CURRENT FAILURE:
- repo: {query.get("repo", "")}
- file: {file_path}
- error_type: {query.get("error_type", "")}
- failure_pattern: {query.get("failure_pattern", "")}

L2 REPAIR TRAJECTORIES:
{l2_block}

YOUR TASK:
1. Find the L2 trajectory that best matches the current failure
2. Identify which STEP in the trajectory matches the current problem
3. Extract ALL files that will be affected across ALL steps
4. Predict NEXT failures that will appear after fixing current problem
5. Identify CI stages that will fail

Return STRICT JSON:
{{
  "best_match_l2_index": 0,
  "current_failure_matches_step": 1,
  "total_trajectory_steps": 3,

  "all_files_in_trajectory": [
    {{"file": "pyproject.toml", "step": 1, "stage": "install"}},
    {{"file": "src/main.py", "step": 2, "stage": "lint"}},
    {{"file": "src/utils.py", "step": 2, "stage": "lint"}}
  ],

  "next_failures": [
    {{"step": 2, "stage": "lint", "what_will_fail": "Unused imports", "files": ["src/main.py", "src/utils.py"]}},
    {{"step": 3, "stage": "test", "what_will_fail": "Missing dependency", "files": ["pyproject.toml"]}}
  ],

  "ci_stages_sequence": ["install", "lint", "test"],

  "validation_sequence": [
    "Fix step 1 → verify with 'uv install'",
    "EXPECT step 2 failure → check src/*.py for unused imports",
    "EXPECT step 3 failure → check dependencies"
  ]
}}

No markdown, just JSON."""

    def _fetch_l1_details_for_files(
        self,
        *,
        file_list: List[str],
        retrieval_result: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        STEP 2a: Fetch detailed L1 entries for specific files identified in trajectory.
        """
        l1_matches = retrieval_result.get("l1_matches", []) or []

        l1_by_file = {}
        for target_file in file_list:
            normalized_target = _normalize_path(target_file)
            base_target = _basename(normalized_target)

            # Find L1 entries for this file
            matching_l1 = [
                row for row in l1_matches
                if _normalize_path(row.get("file", "")) == normalized_target
                or _basename(row.get("file", "")) == base_target
            ]

            l1_by_file[target_file] = matching_l1

        return l1_by_file

    def _build_step2_deep_analysis_prompt(
        self,
        *,
        file_path: str,
        file_context: str,
        retrieval_result: Dict[str, Any],
        candidates: Dict[str, List[Dict[str, Any]]],
        step1_result: Dict[str, Any],
        l1_details: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """
        STEP 2b: Deep analysis with L1 file-level details to understand file connections.
        """
        query = retrieval_result.get("query", {}) or {}
        query_file_details = [
            item for item in (retrieval_result.get("query_file_details") or [])
            if isinstance(item, dict) and _normalize_path(str(item.get("file", ""))) == _normalize_path(file_path)
        ]
        current_file_detail = query_file_details[0] if query_file_details else {}
        current_file_payload = {
            "file": file_path,
            "issue_type": current_file_detail.get("issue_type", ""),
            "line_number": current_file_detail.get("line_number"),
            "failed_tool": _safe_list(current_file_detail.get("failed_tool", [])),
            "failed_cmd": _safe_list(current_file_detail.get("failed_cmd", [])),
            "file_failure_reason": current_file_detail.get("reason", ""),
            "overall_failure_reason": query.get("overall_failure_reason", "") or query.get("failure_reason", "") or "",
        }

        # Get best L2 trajectory
        best_l2_idx = step1_result.get("best_match_l2_index", 0)
        l2_trajectory = candidates["l2"][best_l2_idx] if candidates.get("l2") and best_l2_idx < len(candidates["l2"]) else {}

        # Format L1 details for each file
        l1_details_text = []
        for target_file, l1_entries in l1_details.items():
            if l1_entries:
                l1_details_text.append(f"\n  FILE: {target_file}")
                for idx, l1 in enumerate(l1_entries):
                    problem = _clip(str(l1.get("problem", l1.get("failure_reason", ""))), 100)
                    fix = _clip(str(l1.get("fix_strategy", l1.get("fix", ""))), 150)
                    l1_details_text.append(f"    [L1-{idx}] score={l1.get('similarity_score', 0.0):.2f}")
                    l1_details_text.append(f"      problem: {problem}")
                    l1_details_text.append(f"      fix_strategy: {fix}")

        l1_block = "\n".join(l1_details_text) if l1_details_text else "[No L1 details]"

        # L3 principles
        l3_summaries = []
        for idx, l3 in enumerate(candidates.get("l3", [])):
            principle = _clip(str(l3.get("universal_principle", l3.get("principle", ""))), 100)
            l3_summaries.append(f"  [L3-{idx}] {principle}")
        l3_block = "\n".join(l3_summaries) if l3_summaries else "[No L3 principles]"

        return f"""STEP 2: DEEP FILE ANALYSIS WITH L1 DETAILS

You are doing deep analysis to understand file connections and create a complete repair plan.

CURRENT FAILURE:
- repo: {query.get("repo", "")}
- file: {file_path}
- error_type: {query.get("error_type", "")}
- failure_pattern: {query.get("failure_pattern", "")}

CURRENT FILE DETAILS:
{json.dumps(current_file_payload, indent=2, ensure_ascii=False)}

STEP 1 TRAJECTORY ANALYSIS RESULTS:
{json.dumps(step1_result, indent=2, ensure_ascii=False)}

L2 TRAJECTORY (best match):
  issue_id: {l2_trajectory.get('issue_id', '')}
  repo: {l2_trajectory.get('repo', '')}
  overall_failure: {l2_trajectory.get('overall_failure', '')}

  repair_trajectory: {json.dumps(l2_trajectory.get('repair_trajectory', {}), indent=4)}

  atomic_problems: {json.dumps(l2_trajectory.get('atomic_problems', []), indent=4)}

DETAILED L1 FILE-LEVEL INFORMATION:
{l1_block}

L3 UNIVERSAL PRINCIPLES:
{l3_block}

YOUR TASK:
Analyze ALL information to understand:
1. How files are CONNECTED (why fixing A affects B)
2. Complete VALIDATION sequence (check A, then B, then C)
3. What to check in EACH file NOW (before any fixes)
4. Root cause analysis linking all problems

Return STRICT JSON:
{{
  "use_memory": true,
  "similarity_score": 0.85,
  "similarity_reason": "Matches install error with multi-step trajectory",
  "selected_memory_levels": ["L1", "L2"],

  "file_connection_analysis": {{
    "how_files_connected": "Fixing pyproject.toml updates dependencies → triggers lint checks → reveals unused imports",
    "why_fixing_current_affects_others": "Dependency update changes available imports, making some imports unused",
    "dependency_chain": ["pyproject.toml", "src/main.py", "src/utils.py"]
  }},

  "complete_validation_sequence": [
    {{
      "step": 1,
      "action": "Fix pyproject.toml line 379-382: remove default-groups",
      "verify_with": "uv install",
      "expected_result": "Install succeeds",
      "what_checks_now_enabled": "Linter can now run"
    }},
    {{
      "step": 2,
      "action": "Fix src/main.py line 5: remove unused import",
      "verify_with": "ruff check src/main.py",
      "expected_result": "Linter passes for main.py",
      "what_fails_if_skipped": "Linter fails on unused ChatAgent import"
    }}
  ],

  "all_files_to_check_now": [
    {{
      "file": "pyproject.toml",
      "why": "Current failure - invalid default-groups",
      "where": "Line 379-382, [tool.uv] section",
      "what_to_look_for": "default-groups config (deprecated)",
      "priority": "immediate"
    }},
    {{
      "file": "src/main.py",
      "why": "Will fail after fixing pyproject.toml",
      "where": "Line 5, imports",
      "what_to_look_for": "Unused ChatAgent import",
      "priority": "after_step1"
    }}
  ],

  "selected_context": [
    {{
      "score": 0.85,
      "file": "{file_path}",
      "matched_memory_file": "pyproject.toml",
      "failure_pattern": "uv install configuration error",
      "fl_guidance": "Check [tool.uv] section line 379-382 for deprecated default-groups",
      "fix_strategy": "Remove default-groups section (deprecated in uv 0.5+)",
      "next_failures_expected": "After fixing: lint will fail on unused imports in src/*.py files",
      "validation_sequence": ["uv install", "ruff check", "pytest"],
      "all_affected_files_in_trajectory": ["pyproject.toml", "src/main.py", "src/utils.py"]
    }}
  ],

  "dependent_files": [
    {{"file": "src/main.py", "reason": "Will have unused import after dependency update"}},
    {{"file": "src/utils.py", "reason": "Will have unused import after dependency update"}}
  ],

  "additional_files_to_inspect": [],

  "selected_items": [
    {{
      "memory_level": "L2",
      "candidate_key": "L2-0",
      "similarity_score": 0.85,
      "relevance": "high",
      "justification": "Complete trajectory matching install → lint → test sequence",
      "failure_pattern": "uv config + lint + test cascade",
      "failure_reason": "Dependency update cascades to multiple CI stages",

      "trajectory_analysis": {{
        "current_failure_matches_step": 1,
        "total_trajectory_steps": 3,
        "next_failures": [
          {{"step": 2, "stage": "lint", "what_will_fail": "Unused imports", "files": ["src/main.py", "src/utils.py"]}},
          {{"step": 3, "stage": "test", "what_will_fail": "Missing dev dependency", "files": ["pyproject.toml"]}}
        ],
        "all_problems_in_trajectory": [
          {{"problem": 1, "visible": true, "symptom": "Install error"}},
          {{"problem": 2, "visible": false, "symptom": "Lint error"}},
          {{"problem": 3, "visible": false, "symptom": "Test error"}}
        ],
        "validation_sequence": [
          "Fix pyproject.toml → verify with 'uv install'",
          "EXPECT lint failure → check src/main.py for unused imports",
          "EXPECT test failure → check pyproject.toml for missing deps"
        ],
        "all_files_to_check_now": ["pyproject.toml", "src/main.py", "src/utils.py"],
        "recommended_approach": "Fix all 3 problems at once to avoid 2 additional CI runs"
      }},

      "dependent_files": [
        {{"file": "src/main.py", "reason": "Unused import after dependency update"}},
        {{"file": "src/utils.py", "reason": "Unused import after dependency update"}}
      ],

      "localization_hint": "Start at [tool.uv] section, then check src/*.py imports",
      "fix_direction": "1. Remove deprecated config 2. Clean up imports 3. Update dependencies"
    }}
  ],

  "diagnostic_summary": "This is a multi-stage CI failure. Current install error will cascade to lint and test failures. Fix all 3 problems together for efficiency."
}}

Rules:
- Use ALL information: L2 trajectory + L1 file details + L3 principles
- Explain file CONNECTIONS (not just list files)
- Provide COMPLETE validation sequence
- Be specific about WHERE to look in each file
- No markdown, just JSON."""

    def analyze_relevance_for_file(
        self,
        *,
        file_path: str,
        file_context: str,
        retrieval_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        cache_key = (_normalize_path(file_path), _clip(file_context, 800))
        if cache_key in self._per_file_analysis_cache:
            return self._per_file_analysis_cache[cache_key]

        normalized_file = _normalize_path(file_path)
        empty: Dict[str, Any] = {
            "use_memory": False,
            "similarity_score": 0.0,
            "similarity_reason": "",
            "selected_memory_levels": [],
            "selected_context": [],
            "dependent_files": [],
            "additional_files_to_inspect": [],
            "selected_items": [],
            "diagnostic_summary": "",
        }

        candidates = self._filter_candidates_for_file(file_path, retrieval_result)
        if not (candidates["l1"] or candidates["l2"] or candidates["l3"]):
            self._per_file_analysis_cache[cache_key] = empty
            return empty

        selected_items: List[Dict[str, Any]] = []
        selected_context: List[Dict[str, Any]] = []
        dependent_files: List[Dict[str, str]] = []
        additional_files: List[Dict[str, str]] = []
        seen_files: set[str] = {normalized_file}

        def add_file(path: str, reason: str, target: List[Dict[str, str]]) -> None:
            normalized = _normalize_path(path)
            if not normalized or normalized in seen_files:
                return
            seen_files.add(normalized)
            target.append({"file": normalized, "reason": reason})

        for level_name, rows in (("L1", candidates["l1"]), ("L2", candidates["l2"]), ("L3", candidates["l3"])):
            for idx, row in enumerate(rows):
                score = float(row.get("similarity_score", 0.0))
                reason = str(
                    row.get("failure_reason")
                    or row.get("overall_failure_reason")
                    or row.get("reason")
                    or row.get("principle")
                    or ""
                )
                fix = str(
                    row.get("fix_strategy")
                    or row.get("fix_direction")
                    or row.get("fix_approach")
                    or row.get("principle")
                    or ""
                )
                key = f"{level_name}-{idx}"
                selected_items.append({
                    "memory_level": level_name,
                    "candidate_key": key,
                    "similarity_score": round(score, 4),
                    "relevance": "high" if score >= 0.65 else "medium" if score >= 0.35 else "low",
                    "justification": _clip(reason or fix, 260),
                    "failure_pattern": row.get("failure_pattern") or row.get("issue_type") or row.get("pattern_name", ""),
                    "failure_reason": reason,
                    "localization_hint": _clip(reason, 220),
                    "fix_direction": _clip(fix, 220),
                    "dependent_files": row.get("dependent_files", []),
                    "additional_localization_files": [],
                })
                selected_context.append({
                    "memory_level": level_name,
                    "score": round(score, 4),
                    "file": normalized_file,
                    "matched_memory_file": _normalize_path(str(row.get("file") or "")),
                    "failure_pattern": row.get("failure_pattern") or row.get("issue_type") or row.get("pattern_name", ""),
                    "fl_guidance": _clip(reason, 220),
                    "fix_strategy": _clip(fix, 220),
                })

                for ref in _structured_file_refs(row.get("dependent_files", [])):
                    add_file(ref["file"], ref.get("reason", "related memory file"), dependent_files)
                for file_row in _safe_list(row.get("files") or row.get("modified_files") or row.get("example_files") or []):
                    if isinstance(file_row, dict):
                        add_file(
                            str(file_row.get("file") or ""),
                            str(file_row.get("reason") or file_row.get("failure_reason") or "related memory file"),
                            additional_files,
                        )

        best_score = max((float(item.get("similarity_score", 0.0)) for item in selected_items), default=0.0)
        levels = [level for level in ("L1", "L2", "L3") if candidates[level.lower()]]
        parsed = {
            "use_memory": bool(selected_items),
            "similarity_score": round(best_score, 4),
            "similarity_reason": "Ranked memory candidates matched this file or its issue context.",
            "selected_memory_levels": levels,
            "selected_context": selected_context,
            "dependent_files": dependent_files,
            "additional_files_to_inspect": additional_files,
            "selected_items": selected_items,
            "diagnostic_summary": (
                f"Retrieved {len(selected_items)} memory candidates for {normalized_file}; "
                f"best score={best_score:.2f}."
            ),
        }
        self._per_file_analysis_cache[cache_key] = parsed
        return parsed

    def get_additional_files_for_file(
        self,
        *,
        file_path: str,
        file_context: str,
        retrieval_result: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        per_file = self.analyze_relevance_for_file(
            file_path=file_path,
            file_context=file_context,
            retrieval_result=retrieval_result,
        )
        out: List[Dict[str, str]] = []
        seen: set[str] = set()
        for ref in _structured_file_refs(per_file.get("dependent_files", [])):
            if ref["file"] and ref["file"] not in seen and ref["file"] != _normalize_path(file_path):
                seen.add(ref["file"])
                out.append(ref)
        for ref in _structured_file_refs(per_file.get("additional_files_to_inspect", [])):
            if ref["file"] and ref["file"] not in seen and ref["file"] != _normalize_path(file_path):
                seen.add(ref["file"])
                out.append(ref)
        for item in _safe_list(per_file.get("selected_items", [])):
            if not isinstance(item, dict):
                continue
            for ref in _structured_file_refs(item.get("dependent_files", [])):
                if ref["file"] and ref["file"] not in seen and ref["file"] != _normalize_path(file_path):
                    seen.add(ref["file"])
                    out.append(ref)
            for ref in _structured_file_refs(item.get("additional_localization_files", [])):
                if ref["file"] and ref["file"] not in seen and ref["file"] != _normalize_path(file_path):
                    seen.add(ref["file"])
                    out.append(ref)
        return out

    def format_for_file_prompt(self, file_path: str, retrieval_result: Dict[str, Any], file_context: str = "") -> str:
        file_path = _normalize_path(file_path)
        per_file = self.analyze_relevance_for_file(
            file_path=file_path,
            file_context=file_context,
            retrieval_result=retrieval_result,
        )
        selected_items = per_file.get("selected_items", []) or []
        selected_context = per_file.get("selected_context", []) or []

        lines = ["HIERARCHICAL MEMORY SYNTHESIS (L1/L2/L3):"]
        lines.append(f"  Levels used: {', '.join(per_file.get('selected_memory_levels', [])) or 'None'}")
        lines.append(
            "  Similarity scores: "
            f"L1={retrieval_result.get('level_scores', {}).get('L1', 0.0):.2f}, "
            f"L2={retrieval_result.get('level_scores', {}).get('L2', 0.0):.2f}, "
            f"L3={retrieval_result.get('level_scores', {}).get('L3', 0.0):.2f}, "
            f"weighted={retrieval_result.get('weighted_similarity', 0.0):.2f}, "
            f"file_relevance={per_file.get('similarity_score', 0.0):.2f}"
        )
        lines.append(f"  File-level relevance: {_clip(str(per_file.get('similarity_reason', '')), 220)}")
        diagnostic_summary = per_file.get("diagnostic_summary", "")
        if diagnostic_summary:
            lines.append(f"  Memory summary: {_clip(diagnostic_summary, 400)}")
        if not selected_items and not selected_context:
            lines.append("  No file-specific memory evidence selected.")
            return "\n".join(lines)

        for item in selected_context:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"  {item.get('memory_level', '')} selected_context: "
                f"record={item.get('source_record_id', '')} score={float(item.get('score', 0.0)):.2f}"
            )
            if item.get("matched_memory_file"):
                lines.append(f"    matched_memory_file={item.get('matched_memory_file', '')}")
            if item.get("failure_pattern"):
                lines.append(f"    failure_pattern={_clip(str(item.get('failure_pattern', '')), 220)}")
            if item.get("fl_guidance"):
                lines.append(f"    fl_guidance={_clip(str(item.get('fl_guidance', '')), 220)}")
            if item.get("fix_strategy"):
                lines.append(f"    fix_strategy={_clip(str(item.get('fix_strategy', '')), 220)}")

        for item in selected_items:
            level = str(item.get("memory_level", ""))
            candidate_key = str(item.get("candidate_key", ""))
            lines.append(
                f"  {level} selected: candidate={candidate_key} score={float(item.get('similarity_score', 0.0)):.2f} "
                f"relevance={item.get('relevance', '')}"
            )
            if item.get("failure_pattern"):
                lines.append(f"    failure_pattern={_clip(str(item.get('failure_pattern', '')), 220)}")
            if item.get("justification"):
                lines.append(f"    justification={_clip(str(item.get('justification', '')), 220)}")
            if item.get("failure_reason"):
                lines.append(f"    relevant_reason={_clip(str(item.get('failure_reason', '')), 220)}")
            if item.get("localization_hint"):
                lines.append(f"    localization_hint={_clip(str(item.get('localization_hint', '')), 220)}")
            if item.get("fix_direction"):
                lines.append(f"    fix_direction={_clip(str(item.get('fix_direction', '')), 220)}")
            dependent_files = _structured_file_refs(item.get("dependent_files", []))
            if dependent_files:
                lines.append(
                    "    dependent_files="
                    + ", ".join(
                        f"{ref['file']} ({_clip(ref['reason'], 80)})" if ref.get("reason") else ref["file"]
                        for ref in dependent_files
                    )
                )
            extra_files = _structured_file_refs(item.get("additional_localization_files", []))
            if extra_files:
                lines.append(
                    "    additional_localization_files="
                    + ", ".join(
                        f"{ref['file']} ({_clip(ref['reason'], 80)})" if ref.get("reason") else ref["file"]
                        for ref in extra_files
                    )
                )

        dependent_files = _structured_file_refs(per_file.get("dependent_files", []))
        if dependent_files:
            lines.append(
                "  dependent_files="
                + ", ".join(
                    f"{ref['file']} ({_clip(ref['reason'], 80)})" if ref.get("reason") else ref["file"]
                    for ref in dependent_files
                )
            )
        additional_files = _structured_file_refs(per_file.get("additional_files_to_inspect", []))
        if additional_files:
            lines.append(
                "  additional_files_to_inspect="
                + ", ".join(
                    f"{ref['file']} ({_clip(ref['reason'], 80)})" if ref.get("reason") else ref["file"]
                    for ref in additional_files
                )
            )

        return "\n".join(lines)

    def format_for_prompt(self, retrieval_result: Dict[str, Any]) -> str:
        lines = ["Retrieved hierarchical memory:"]
        for level_key in ("l1_matches", "l2_matches", "l3_matches"):
            for row in (retrieval_result.get(level_key, []) or []):
                score_str = f"score={row.get('similarity_score', 0.0):.2f}"
                lines.append(
                    f"  {row.get('memory_level','')} scores=({score_str}) "
                    f"error_type={row.get('error_type','')}"
                )
                
        
        return "\n".join(lines)

    def save_memory_entry(
        self,
        *,
        task_id: str,
        sha_fail: str,
        repo_name: str,
        repo_owner: str,
        workflow_path: str,
        workflow: str,
        log_analysis_result: Dict[str, Any],
        changed_files_info: Optional[Dict[str, Any]],
        fault_localizer: Optional[Dict[str, Any]],
        patch_generator: Optional[Dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return

        repo_id = repo_name
        error_type = _first_error_type(log_analysis_result)
        # issue-level failure_pattern from log analysis (subcategory of the primary error type)
        issue_failure_pattern = _primary_failure_pattern(log_analysis_result.get("error_types", []))
        failed_jobs = log_analysis_result.get("failed_jobs", log_analysis_result.get("failed_job", []))
        failed_cmd, failed_tool = _extract_failed_commands_and_tools(failed_jobs)
        diff_text = str((patch_generator or {}).get("diff", "") or "")
        ground_truth_files = extract_files_from_diff(diff_text)
        fl_data = (fault_localizer or {}).get("fault_localization_data", []) or []
        error_context_summary = _clip(
            json.dumps(log_analysis_result.get("error_context", []), ensure_ascii=False),
            1800,
        )
        # Derive workflow_name from path (e.g. ".github/workflows/test.yml" → "test")
        workflow_name = os.path.splitext(os.path.basename(workflow_path or ""))[0]

        # Per-file details from the log analyzer (issue_type, failed_cmd, failed_tool per file).
        # Used in _build_l1_rows to save file-specific cmd/tool instead of issue-level fallback.
        log_file_details_map: Dict[str, Dict[str, Any]] = {
            d["file"]: d
            for d in _extract_log_file_details(log_analysis_result)
            if d.get("file")
        }

        l1_rows = self._build_l1_rows(
            task_id=task_id,
            sha_fail=sha_fail,
            repo_id=repo_id,
            repo_name=repo_name,
            workflow_path=workflow_path,
            workflow_name=workflow_name,
            error_type=error_type,
            issue_failure_pattern=issue_failure_pattern,
            failed_cmd=failed_cmd,
            failed_tool=failed_tool,
            ground_truth_files=ground_truth_files,
            fl_data=fl_data,
            diff_text=diff_text,
            error_context_summary=error_context_summary,
            log_file_details_map=log_file_details_map,
            all_error_types=_normalize_error_type_rows(log_analysis_result.get("error_types", [])),
        )
        overall_failure_reason = _clip(
            " | ".join(
                str(x).strip() for x in _safe_list(log_analysis_result.get("error_context", [])) if str(x).strip()
            ),
            1200,
        )
        l1_rows = [
            self._normalize_l1_record(
                row,
                overall_failure_reason=overall_failure_reason,
                log_file_detail=log_file_details_map.get(_normalize_path(str(row.get("file", ""))), {}),
            )
            for row in l1_rows
        ]
        for row in l1_rows:
            self._upsert_list_record(
                self.failure_memory,
                row,
                keys=("sha_fail", "file", "error_type", "failure_pattern"),
                # One record per (issue, file, error_type, pattern) — distinct failure types
                # for the same file are stored separately, not overwritten.
            )
            if self.memory_backend == "chroma":
                self._upsert_chroma_record("L1", row)

        repo_row = self._build_l2_row(
            task_id=task_id,
            sha_fail=sha_fail,
            repo_id=repo_id,
            repo_name=repo_name,
            error_type=error_type,
            failed_cmd=failed_cmd,
            failed_tool=failed_tool,
            l1_rows=l1_rows,
            ground_truth_files=ground_truth_files,
            error_context_summary=error_context_summary,
        )
        repo_row = self._normalize_l2_record(repo_row)
        self._upsert_repo_memory(repo_row)
        if self.memory_backend == "chroma":
            self._upsert_chroma_record("L2", repo_row)

        cross_row = self._build_l3_row(
            task_id=task_id,
            repo_id=repo_id,
            repo_name=repo_name,
            error_type=error_type,
            failed_cmd=failed_cmd,
            failed_tool=failed_tool,
            repo_row=repo_row,
            ground_truth_files=ground_truth_files,
        )
        cross_row = self._normalize_l3_record(cross_row)
        self._merge_cross_memory(cross_row)
        if self.memory_backend == "chroma":
            self._upsert_chroma_record("L3", cross_row)

        _write_json_list(self.failure_memory_path, self.failure_memory)
        _write_json_list(self.repo_memory_path, self.repo_memory)
        _write_json_list(self.cross_memory_path, self.cross_memory)

    def _build_l1_rows(
        self,
        *,
        task_id: str,
        sha_fail: str,
        repo_id: str,
        repo_name: str,
        workflow_path: str,
        workflow_name: str,
        error_type: str,
        issue_failure_pattern: str,
        failed_cmd: List[str],
        failed_tool: List[str],
        ground_truth_files: List[str],
        fl_data: List[Dict[str, Any]],
        diff_text: str,
        error_context_summary: str,
        log_file_details_map: Optional[Dict[str, Dict[str, Any]]] = None,
        all_error_types: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build L1 records — one per (file, error_type) pair.

        A single CI failure can involve multiple distinct failure types in the
        same file (e.g. ImportError + TypeAnnotation + LintError). Storing one
        record per type means retrieval for any one of those types will find
        this memory, not just the primary type.
        """
        log_file_details_map = log_file_details_map or {}
        all_error_types = all_error_types or []
        rows: List[Dict[str, Any]] = []

        for entry in fl_data:
            file_path = _normalize_path(entry.get("file_path", ""))
            if not file_path:
                continue
            faults = entry.get("faults", []) or []

            log_file_detail = log_file_details_map.get(file_path, {})
            file_failed_tool = [str(x) for x in _safe_list(log_file_detail.get("failed_tool") or [])] or failed_tool
            file_failed_cmd  = [str(x) for x in _safe_list(log_file_detail.get("failed_cmd") or [])] or failed_cmd

            file_log_reason = str(log_file_detail.get("reason") or "").strip()

            file_diff = _extract_file_diff(diff_text, file_path)
            added_lines = [
                ln.lstrip("+").strip()
                for ln in file_diff.splitlines()
                if ln.startswith("+") and not ln.startswith("+++") and ln.lstrip("+").strip()
            ]
            fix_direction = _clip("; ".join(added_lines), 500) if added_lines else ""

            dep_files: List[Dict[str, str]] = [
                {"file": p, "reason": "Co-modified in the same fix"}
                for p in ground_truth_files if p != file_path
            ]

            # Collect all distinct (error_type, failure_pattern, reasons) for this file.
            # Sources in priority order:
            #   1. Per-fault issue_type from fault localization data (most specific)
            #   2. Per-file issue_type from log analyzer
            #   3. Issue-level error_types from log analysis (all types, not just primary)
            #   4. Issue-level primary error_type as final fallback
            type_pattern_pairs: List[Tuple[str, str, List[str]]] = []

            # Source 1: per-fault (issue_type, reason) pairs — one entry per distinct type
            fault_type_map: Dict[str, List[str]] = {}
            for f in faults:
                ft = str(f.get("issue_type") or "").strip()
                fr = str(f.get("reason") or "").strip()
                if ft:
                    fault_type_map.setdefault(ft, [])
                    if fr:
                        fault_type_map[ft].append(fr)

            for ft, ft_reasons in fault_type_map.items():
                type_pattern_pairs.append((error_type, ft, ft_reasons))

            # Source 2: per-file log analyzer issue_type (if not already covered)
            log_issue = str(log_file_detail.get("issue_type") or "").strip()
            if log_issue and not any(p == log_issue for _, p, _ in type_pattern_pairs):
                type_pattern_pairs.append((error_type, log_issue, [file_log_reason] if file_log_reason else []))

            # Source 3: issue-level error_types (use category as both error_type and pattern
            # when no finer-grained per-file type was found)
            if not type_pattern_pairs:
                for et_row in all_error_types:
                    cat = str(et_row.get("category") or "").strip()
                    sub = str(et_row.get("subcategory") or "").strip()
                    if cat:
                        pat = sub or cat
                        if not any(p == pat for _, p, _ in type_pattern_pairs):
                            type_pattern_pairs.append((cat, pat, []))

            # Source 4: primary error_type fallback
            if not type_pattern_pairs:
                type_pattern_pairs.append((
                    error_type,
                    issue_failure_pattern or error_type,
                    [file_log_reason] if file_log_reason else [],
                ))

            # Emit one L1 record per distinct (error_type, failure_pattern) pair
            for file_error_type, raw_pattern, type_reasons in type_pattern_pairs:
                issue_type_desc = _to_descriptive_issue_type(raw_pattern, file_error_type)
                all_reasons = list(dict.fromkeys(r for r in type_reasons + [file_log_reason] if r))
                failure_reason = _clip(" | ".join(all_reasons) or error_context_summary, 500)

                row = {
                    "sha_fail": sha_fail,
                    "issue_id": task_id,
                    "repo": repo_id,
                    "repo_name": repo_name,
                    "workflow_path": workflow_path,
                    "workflow_name": workflow_name,
                    "file": file_path,
                    "error_type": file_error_type,
                    "issue_type": issue_type_desc,
                    "failure_pattern": raw_pattern,
                    "failure_reason": failure_reason,
                    "fix_direction": fix_direction,
                    "failed_tool": file_failed_tool,
                    "failed_cmd": file_failed_cmd,
                    "dependent_files": dep_files,
                }
                rows.append(row)

        return rows

    def _build_l2_row(
        self,
        *,
        task_id: str,
        sha_fail: str,
        repo_id: str,
        repo_name: str,
        error_type: str,
        failed_cmd: List[str],
        failed_tool: List[str],
        l1_rows: List[Dict[str, Any]],
        ground_truth_files: List[str],
        error_context_summary: str,
    ) -> Dict[str, Any]:
        # Primary issue_type and failure_pattern from L1 rows (first non-empty wins)
        issue_type_desc = next((str(row.get("issue_type") or "").strip() for row in l1_rows if row.get("issue_type")), "")
        failure_pattern = next((str(row.get("failure_pattern") or "").strip() for row in l1_rows if row.get("failure_pattern")), "")

        # Per-file detail entries — one entry per file with full context
        file_entries: List[Dict[str, Any]] = []
        for row in l1_rows:
            file_entries.append({
                # Which file and what specific issue it has
                "file": row.get("file", ""),
                "issue_type": row.get("issue_type", ""),         # descriptive: "Dependency on environment"
                "failure_pattern": row.get("failure_pattern", ""), # keyword: "dependency_or_env"
                "failure_reason": row.get("failure_reason", ""),   # WHY this file failed
                # What was changed to fix this file
                "fix_direction": row.get("fix_direction", ""),
                # Files that directly or indirectly depend on this file
                "dependent_files": row.get("dependent_files", []),  # [{file, reason}]
            })

        # Overall failure reason connecting all files
        all_reasons = list(dict.fromkeys(
            r for row in l1_rows for r in [row.get("failure_reason", "")] if r
        ))
        overall_failure_reason = _clip(" | ".join(all_reasons) or error_context_summary, 600)

        # Fix approach: combine unique fix directions across files
        fix_directions = list(dict.fromkeys(
            row.get("fix_direction", "") for row in l1_rows if row.get("fix_direction")
        ))
        fix_approach = _clip("; ".join(fix_directions), 500)

        return {
            # Identity — keyed by sha_fail so each issue is its own L2 record
            "sha_fail": sha_fail,
            "issue_id": task_id,
            "repo": repo_id,
            "repo_name": repo_name,
            # Error classification
            "error_type": error_type,
            "issue_type": issue_type_desc,   # descriptive: "Dependency on environment"
            "failure_pattern": failure_pattern,  # keyword: "dependency_or_env"
            # Narrative context
            "overall_failure_reason": overall_failure_reason,
            "fix_approach": fix_approach,
            # Per-file details (each with issue_type, failure_reason, fix_direction, dependent_files)
            "files": file_entries,
            # Ground-truth patched files (for retrieval augmentation)
            "changed_files": ground_truth_files,
            "failed_tool": failed_tool,
            "failed_cmd": failed_cmd,
        }

    def _build_l3_row(
        self,
        *,
        task_id: str,
        repo_id: str,
        repo_name: str,
        error_type: str,
        failed_cmd: List[str],
        failed_tool: List[str],
        repo_row: Dict[str, Any],
        ground_truth_files: List[str],
    ) -> Dict[str, Any]:
        # issue_type: descriptive phrase carried forward from L2
        issue_type_desc = str(repo_row.get("issue_type") or "").strip()
        # failure_pattern: specific keyword from L2
        failure_pattern = str(repo_row.get("failure_pattern") or "").strip()

        # Failure reason for principle construction
        overall_reason = str(repo_row.get("overall_failure_reason") or repo_row.get("failure_reason") or "").strip()
        reason_snippet = _clip(overall_reason, 300) if overall_reason else ""

        # Build an abstract principle for this (error_type, issue_type) combination
        principle = (
            f"{error_type}"
            + (f" — {issue_type_desc}" if issue_type_desc else "")
            + (f" [{failure_pattern}]" if failure_pattern else "")
            + (f": {reason_snippet}" if reason_snippet else "")
            + f" (seen in repo: {repo_name})"
        )

        # Fix strategies from L2's fix_approach
        fix_approach = str(repo_row.get("fix_approach") or "").strip()
        fix_strategies = [fix_approach] if fix_approach else []

        # Failure reasons: collect from L2 file entries
        file_failure_reasons = list(dict.fromkeys(
            str(f.get("failure_reason") or "")
            for f in (repo_row.get("files") or [])
            if f.get("failure_reason")
        ))
        failure_reasons = ([overall_reason] + file_failure_reasons) if overall_reason else file_failure_reasons

        # Example files: top files from L2 for concrete retrieval context
        example_files = [
            {"file": f.get("file", ""), "issue_type": f.get("issue_type", ""), "failure_pattern": f.get("failure_pattern", "")}
            for f in (repo_row.get("files") or [])
            if f.get("file")
        ]

        # failure_patterns: all patterns observed across files in this issue
        file_patterns = list(dict.fromkeys(
            str(f.get("failure_pattern") or "")
            for f in (repo_row.get("files") or [])
            if f.get("failure_pattern")
        ))
        all_failure_patterns = list(dict.fromkeys([failure_pattern] + file_patterns)) if failure_pattern else file_patterns

        return {
            # Classification (merge key for cross-issue accumulation)
            "error_type": error_type,
            "issue_type": issue_type_desc,       # descriptive: "Dependency on environment"
            "failure_pattern": failure_pattern,   # primary keyword for this pattern
            # Abstract principle and fix guidance
            "principle": principle,
            "fix_strategies": fix_strategies,
            "fix_strategy": fix_strategies[0] if fix_strategies else "",
            # Accumulated patterns and reasons (grow as more issues are seen)
            "failure_patterns": all_failure_patterns,  # all specific patterns for this issue type
            "failure_reasons": failure_reasons,         # concrete failure reason examples
            # Provenance
            "repos": [repo_id],
            "repo_names": [repo_name],
            "failed_tool": failed_tool,
            "failed_cmd": failed_cmd,
            "evidence_issue_ids": [task_id],
            # Concrete file examples for retrieval grounding
            "example_files": example_files,
            "changed_files": ground_truth_files,
        }

    def _upsert_list_record(self, rows: List[Dict[str, Any]], record: Dict[str, Any], keys: Tuple[str, ...]) -> None:
        for index, row in enumerate(rows):
            if all(str(row.get(key) or "") == str(record.get(key) or "") for key in keys):
                rows[index] = record
                return
        rows.append(record)

    def _upsert_repo_memory(self, incoming: Dict[str, Any]) -> None:
        """
        Upsert L2 record keyed by sha_fail.
        Each issue gets its own clean L2 record — no cross-issue merging.
        This preserves the per-issue context (files, reasons, fix directions)
        without polluting entries with unrelated issues that share the same error_type.
        """
        sha = str(incoming.get("sha_fail") or "")
        if sha:
            for index, row in enumerate(self.repo_memory):
                if str(row.get("sha_fail") or "") == sha:
                    self.repo_memory[index] = incoming
                    return
        self.repo_memory.append(incoming)

    def _merge_cross_memory(self, incoming: Dict[str, Any]) -> None:
        """
        Merge L3 records by (error_type, issue_type).
        L3 is the abstract pattern level — multiple issues of the same type
        accumulate failure_patterns and fix_strategies here.
        """
        key = (
            str(incoming.get("error_type") or ""),
            str(incoming.get("issue_type") or ""),
        )
        for index, row in enumerate(self.cross_memory):
            row_key = (
                str(row.get("error_type") or ""),
                str(row.get("issue_type") or ""),
            )
            if row_key != key:
                continue
            merged = dict(row)
            # Accumulate list fields — deduplicated
            for field in ("repos", "repo_names", "failed_tool", "failed_cmd",
                          "failure_reasons", "fix_strategies", "failure_patterns",
                          "evidence_issue_ids", "changed_files"):
                merged[field] = list(dict.fromkeys(
                    (row.get(field) or []) + (incoming.get(field) or [])
                ))
            # Accumulate example_files (keyed by file path)
            merged["example_files"] = self._merge_dict_list(
                row.get("example_files", []), incoming.get("example_files", []), "file"
            )
            # Always take the latest principle and fix_strategy
            if incoming.get("principle"):
                merged["principle"] = incoming["principle"]
            if incoming.get("fix_strategy"):
                merged["fix_strategy"] = incoming["fix_strategy"]
            self.cross_memory[index] = merged
            return
        self.cross_memory.append(incoming)

    def _merge_dict_list(self, left: List[Dict[str, Any]], right: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for row in (left or []) + (right or []):
            value = str(row.get(key) or "")
            if value and value not in seen:
                seen.add(value)
                out.append(row)
        return out

    def _append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
