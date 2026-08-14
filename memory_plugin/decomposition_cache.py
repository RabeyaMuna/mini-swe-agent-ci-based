"""
Decomposition Cache - Centralized caching for CI failure decomposition
======================================================================

Caches STAGE 0 decomposition results so they can be reused across:
- Different memory levels (l1, l1+l2, l1+l2+l3)
- Different models (glm5.2, minimax2.5, gpt-5.4-mini)
- Different runs (avoid regenerating same decomposition)

Cache file: data/decomposition_cache.json

Format:
{
  "sha_fail": [
    {
      "problem": "...",
      "root_cause": "...",
      "files": [...],
      "failure_type": "...",
      "failure_signals": [...],
      "verification_cmd": "...",
      "query": {
        "l1": {
          "problem": "...",
          "root_cause": "...",
          "files": [...],
          "failure_types": [...],
          "repo": "...",
          "workflow_name": "...",  # From dataset
          "workflow_path": "...",  # From dataset
          "failure_signals": [...]
        },
        "l2": {
          "problem": "...",
          "root_cause": "...",
          "files": [...],
          "failure_types": [...],
          "repo": "...",
          "failure_signals": [...]
          # NO workflow_name (repo-level only)
        },
        "l3": {
          "problem": "...",  # Generic/abstract
          "root_cause": "...",  # Generic/abstract
          "failure_types": [...],
          "failure_signals": [...]
          # NO repo, NO workflow (universal)
        }
      }
    }
  ]
}

Note:
- Cache stores problems array DIRECTLY at sha_fail key
- workflow_name and workflow_path from dataset → stored in query.l1
- validation_sequence is automatically removed
- Old format is NOT supported - entries will be skipped
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import fcntl


class DecompositionCacheError(RuntimeError):
    """Raised when the decomposition cache cannot be read or persisted safely."""


class DecompositionCache:
    """Cache for STAGE 0 CI failure decomposition."""

    _thread_lock = threading.RLock()

    def __init__(self, cache_dir: str | Path = "data"):
        """
        Initialize decomposition cache.

        Args:
            cache_dir: Directory for cache file (default: data/)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "decomposition_cache.json"
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk (new format only)."""
        if not self.cache_file.exists():
            self._cache = {}
            return

        try:
            with open(self.cache_file) as f:
                loaded_cache = json.load(f)

            if not isinstance(loaded_cache, dict):
                raise DecompositionCacheError(
                    "decomposition_cache.json must contain a JSON object"
                )

            # NEW FORMAT ONLY: {"sha_fail": [problems]}
            new_cache = {}
            for sha_fail, entry in loaded_cache.items():
                if not isinstance(entry, list) or not all(
                    isinstance(problem, dict) for problem in entry
                ):
                    raise DecompositionCacheError(
                        "Malformed decomposition cache entry for "
                        f"{str(sha_fail)[:12]}: expected a list of problem objects"
                    )
                new_cache[sha_fail] = [
                    self._clean_problem(dict(problem)) for problem in entry
                ]

            self._cache = new_cache

        except DecompositionCacheError:
            raise
        except Exception as e:
            raise DecompositionCacheError(
                f"Failed to load {self.cache_file}: {e}"
            ) from e

    def _save_cache(self) -> None:
        """Atomically save the current cache while holding an exclusive lock."""
        with self._thread_lock, self._file_lock():
            self._save_cache_unlocked()

    def _save_cache_unlocked(self) -> None:
        """Atomically save while the caller holds the cache locks."""
        temporary_path: str | None = None
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=".decomposition_cache.",
                suffix=".tmp",
                dir=self.cache_file.parent,
            )
            with os.fdopen(descriptor, "w") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, self.cache_file)
            temporary_path = None
        except Exception as e:
            raise DecompositionCacheError(
                f"Failed to save {self.cache_file}: {e}"
            ) from e
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    @contextmanager
    def _file_lock(self):
        """Serialize cache reads and writes across benchmark processes."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_file.with_suffix(self.cache_file.suffix + ".lock")
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def get(self, sha_fail: str) -> list[dict[str, Any]] | None:
        """
        Get cached decomposition for a commit.

        Args:
            sha_fail: Commit SHA (full or short)

        Returns:
            Cached problems array (direct list), or None if not cached
        """
        if not sha_fail:
            return None

        # Reload under the same cross-process lock used by writers. A long-lived
        # worker must see entries generated by other workers or benchmark runs.
        with self._thread_lock, self._file_lock():
            self._load_cache()

            # Try exact match first
            if sha_fail in self._cache:
                return self._cache[sha_fail]

            # Try prefix match (for short vs full SHA)
            for cached_sha, entry in self._cache.items():
                if cached_sha.startswith(sha_fail) or sha_fail.startswith(cached_sha):
                    return entry

        return None

    def _clean_problem(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Remove unwanted fields from problem."""
        # Remove validation_sequence (not needed)
        problem.pop("validation_sequence", None)
        return problem

    def set(
        self,
        sha_fail: str,
        query: dict[str, Any],
        problems: list[dict[str, Any]],
        model: str = "unknown",
    ) -> None:
        """
        Cache decomposition result in simplified format.

        Args:
            sha_fail: Commit SHA
            query: Input query (CI failure data) - NOT saved, only used for reference
            problems: STAGE 0 output (decomposed problems)
            model: Model used for decomposition - NOT saved, only for backwards compatibility
        """
        if not sha_fail:
            return

        with self._thread_lock, self._file_lock():
            # Merge with the latest on-disk state instead of overwriting entries
            # written since this worker initialized its in-memory cache.
            self._load_cache()
            cleaned_problems = [self._clean_problem(dict(p)) for p in problems]
            self._cache[sha_fail] = cleaned_problems
            self._save_cache_unlocked()

    def has(self, sha_fail: str) -> bool:
        """
        Check if decomposition is cached.

        Args:
            sha_fail: Commit SHA

        Returns:
            True if cached, False otherwise
        """
        return self.get(sha_fail) is not None

    def clear(self) -> None:
        """Clear entire cache."""
        with self._thread_lock, self._file_lock():
            self._cache = {}
            self._save_cache_unlocked()

    def size(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def get_query(self, sha_fail: str) -> dict[str, Any] | None:
        """
        DEPRECATED: Query is no longer stored in cache (workflow_name/workflow_path
        are now in each problem's query.l1/l2 objects).

        Args:
            sha_fail: Commit SHA

        Returns:
            None (query no longer stored)
        """
        # Query is no longer stored - workflow_name and workflow_path
        # are embedded in each problem's query.l1 and query.l2 objects
        return None

    def get_problems(self, sha_fail: str) -> list[dict[str, Any]] | None:
        """
        Get cached problems for a commit.

        Args:
            sha_fail: Commit SHA

        Returns:
            Cached problems list, or None if not cached
        """
        # Cache now stores problems array directly
        return self.get(sha_fail)

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        if not self._cache:
            return {
                "total_entries": 0,
                "total_problems": 0,
            }

        total_problems = 0

        for entry in self._cache.values():
            # Entry is now a list of problems directly
            if isinstance(entry, list):
                total_problems += len(entry)

        return {
            "total_entries": len(self._cache),
            "total_problems": total_problems,
            "avg_problems_per_entry": (
                total_problems / len(self._cache) if self._cache else 0
            ),
        }


# Global cache instance (singleton pattern)
_global_cache: DecompositionCache | None = None


def get_global_cache(cache_dir: str | Path = "data") -> DecompositionCache:
    """
    Get global decomposition cache instance.

    Args:
        cache_dir: Directory for cache file

    Returns:
        Global DecompositionCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = DecompositionCache(cache_dir)
    return _global_cache
