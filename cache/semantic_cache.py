from __future__ import annotations

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional


# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    query_text:       str
    query_embedding:  np.ndarray          # shape (d,), L2-normalised
    result:           Any                 # the payload we cached
    dominant_cluster: int                 # for bucket lookup
    timestamp:        float = field(default_factory=time.time)
    access_count:     int   = 0


class SemanticCache:
    """
    Semantic cache with cluster-bucketed lookup and a tunable similarity threshold.

    Parameters
    ----------
    threshold   : float — cosine similarity threshold for a cache hit (default 0.85)
    n_clusters  : int   — number of fuzzy clusters (used for bucket indexing)
    """

    def __init__(self, threshold: float = 0.85, n_clusters: int = 20):
        self.threshold   = threshold
        self.n_clusters  = n_clusters

        # cluster_buckets[c] = list of CacheEntry indices whose dominant_cluster == c
        self._buckets: dict[int, list[int]] = {c: [] for c in range(n_clusters)}
        self._entries: list[CacheEntry]     = []

        # Stats
        self._hit_count:  int = 0
        self._miss_count: int = 0

        # Thread-safety: reads are concurrent, writes are locked
        self._lock = threading.RLock()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        query_embedding: np.ndarray,
        query_cluster_memberships: np.ndarray,   # shape (n_clusters,)
    ) -> Optional[tuple[CacheEntry, float]]:
        """
        Look up the cache.

        Returns (entry, similarity_score) on hit, or None on miss.

        Steps:
          1. Find the query's top-2 dominant clusters.
          2. Scan only entries in those buckets (O(n/k) average case).
          3. Return the best match if similarity > threshold.
        """
        top2_clusters = np.argsort(query_cluster_memberships)[::-1][:2]

        best_score: float           = -1.0
        best_entry: Optional[CacheEntry] = None

        with self._lock:
            # Collect candidate indices from top-2 cluster buckets
            candidates: set[int] = set()
            for c in top2_clusters:
                candidates.update(self._buckets.get(int(c), []))

            for idx in candidates:
                entry = self._entries[idx]
                score = float(np.dot(query_embedding, entry.query_embedding))
                # Both are L2-normalised → dot product = cosine similarity
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_score >= self.threshold and best_entry is not None:
            with self._lock:
                best_entry.access_count += 1
            self._hit_count += 1
            return best_entry, best_score

        self._miss_count += 1
        return None

    def put(
        self,
        query_text:        str,
        query_embedding:   np.ndarray,
        result:            Any,
        cluster_memberships: np.ndarray,
    ) -> None:
        """
        Store a new entry.  dominant_cluster = argmax of membership vector.
        """
        dominant = int(np.argmax(cluster_memberships))
        entry = CacheEntry(
            query_text=query_text,
            query_embedding=query_embedding.astype(np.float32),
            result=result,
            dominant_cluster=dominant,
        )
        with self._lock:
            idx = len(self._entries)
            self._entries.append(entry)
            self._buckets.setdefault(dominant, []).append(idx)

    def flush(self) -> None:
        """Flush all entries and reset stats."""
        with self._lock:
            self._entries.clear()
            self._buckets = {c: [] for c in range(self.n_clusters)}
            self._hit_count  = 0
            self._miss_count = 0

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total = self._hit_count + self._miss_count
        return {
            "total_entries": len(self._entries),
            "hit_count":     self._hit_count,
            "miss_count":    self._miss_count,
            "hit_rate":      round(self._hit_count / total, 4) if total else 0.0,
            "threshold":     self.threshold,
            "n_clusters":    self.n_clusters,
        }

    def set_threshold(self, new_threshold: float) -> None:
        """Adjust threshold at runtime — no restart needed."""
        if not (0.0 < new_threshold <= 1.0):
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = new_threshold


# ─── SINGLETON ───────────────────────────────────────────────────────────────
# The API imports this single instance.  It is created lazily so that the
# number of clusters is set after the clustering step is complete.

_cache_instance: Optional[SemanticCache] = None


def get_cache(threshold: float = 0.85, n_clusters: int = 20) -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache(threshold=threshold, n_clusters=n_clusters)
    return _cache_instance
