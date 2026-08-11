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
  "sha_fail": {
    "query": {...},  # Input query (CI failure data)
    "problems": [...],  # STAGE 0 output (decomposed problems)
    "timestamp": "2024-01-15T10:30:00",
    "model": "glm-5.2"
  }
}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class DecompositionCache:
    """Cache for STAGE 0 CI failure decomposition."""

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
        """Load cache from disk."""
        if not self.cache_file.exists():
            self._cache = {}
            return

        try:
            with open(self.cache_file) as f:
                self._cache = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load decomposition cache: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save decomposition cache: {e}")

    def get(self, sha_fail: str) -> dict[str, Any] | None:
        """
        Get cached decomposition for a commit.

        Args:
            sha_fail: Commit SHA (full or short)

        Returns:
            Cached entry with 'query' and 'problems', or None if not cached
        """
        if not sha_fail:
            return None

        # Try exact match first
        if sha_fail in self._cache:
            return self._cache[sha_fail]

        # Try prefix match (for short vs full SHA)
        for cached_sha, entry in self._cache.items():
            if cached_sha.startswith(sha_fail) or sha_fail.startswith(cached_sha):
                return entry

        return None

    def set(
        self,
        sha_fail: str,
        query: dict[str, Any],
        problems: list[dict[str, Any]],
        model: str = "unknown",
    ) -> None:
        """
        Cache decomposition result.

        Args:
            sha_fail: Commit SHA
            query: Input query (CI failure data)
            problems: STAGE 0 output (decomposed problems)
            model: Model used for decomposition
        """
        if not sha_fail:
            return

        self._cache[sha_fail] = {
            "query": query,
            "problems": problems,
            "timestamp": datetime.now().isoformat(),
            "model": model,
        }
        self._save_cache()

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
        self._cache = {}
        self._save_cache()

    def size(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def get_query(self, sha_fail: str) -> dict[str, Any] | None:
        """
        Get cached query for a commit.

        Args:
            sha_fail: Commit SHA

        Returns:
            Cached query, or None if not cached
        """
        entry = self.get(sha_fail)
        return entry.get("query") if entry else None

    def get_problems(self, sha_fail: str) -> list[dict[str, Any]] | None:
        """
        Get cached problems for a commit.

        Args:
            sha_fail: Commit SHA

        Returns:
            Cached problems list, or None if not cached
        """
        entry = self.get(sha_fail)
        return entry.get("problems") if entry else None

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        if not self._cache:
            return {
                "total_entries": 0,
                "models": {},
                "total_problems": 0,
            }

        models = {}
        total_problems = 0

        for entry in self._cache.values():
            model = entry.get("model", "unknown")
            models[model] = models.get(model, 0) + 1
            total_problems += len(entry.get("problems", []))

        return {
            "total_entries": len(self._cache),
            "models": models,
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
