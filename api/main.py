"""
Part 4: FastAPI Service
------------------------
Exposes the semantic search system as a REST API.

Endpoints:
  POST   /query        — semantic search with cache
  GET    /cache/stats  — cache statistics
  DELETE /cache        — flush cache

State management:
  All mutable state (the cache, hit/miss counters) lives in the
  SemanticCache singleton in cache/semantic_cache.py.
  FastAPI itself is stateless — it just orchestrates the components.

Startup:
  On startup we load:
    - SentenceTransformer model (for query embedding)
    - PCA model (same reduction used during clustering)
    - FuzzyCMeans model (for query cluster membership)
    - ChromaDB collection (for semantic search)
    - SemanticCache singleton

Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import os
import json
import pickle
import sys
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path so imports work regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer
from cache.semantic_cache import get_cache
from clustering.fuzzy_cmeans import FuzzyCMeans


# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH    = BASE_DIR / "data" / "chroma_db"
PCA_PATH       = BASE_DIR / "data" / "pca_model.pkl"
FCM_PATH       = BASE_DIR / "data" / "fcm_model.pkl"
CLUSTER_META   = BASE_DIR / "data" / "cluster_meta.json"
COLLECTION     = "newsgroups"
N_RESULTS      = 5           # docs returned per query
DEFAULT_THRESH = 0.85


# ─── APP STATE ───────────────────────────────────────────────────────────────

class AppState:
    embed_model: SentenceTransformer = None
    pca:         object               = None
    fcm:         object               = None
    collection:  object               = None
    n_clusters:  int                  = 20


state = AppState()


# ─── LIFESPAN ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once on startup; clean up on shutdown."""
    print("Loading embedding model …")
    state.embed_model = SentenceTransformer(EMBED_MODEL)

    print("Loading PCA model …")
    with open(PCA_PATH, "rb") as f:
        state.pca = pickle.load(f)

    print("Loading FCM model …")
    with open(FCM_PATH, "rb") as f:
        state.fcm = pickle.load(f)

    print("Connecting to ChromaDB …")
    client           = chromadb.PersistentClient(path=str(CHROMA_PATH))
    state.collection = client.get_collection(COLLECTION)

    print("Loading cluster metadata …")
    with open(CLUSTER_META) as f:
        cm = json.load(f)
    state.n_clusters = cm["n_clusters"]

    # Initialise the cache singleton
    get_cache(threshold=DEFAULT_THRESH, n_clusters=state.n_clusters)

    print(f"Service ready — {state.collection.count():,} docs, "
          f"{state.n_clusters} clusters, cache threshold={DEFAULT_THRESH}")
    yield
    # Shutdown: nothing to clean up for in-memory cache


# ─── APP ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Trademarkia Semantic Search",
    description="Fuzzy-clustered semantic search over 20 Newsgroups with semantic cache",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    """Embed and L2-normalise a single query string."""
    vec = state.embed_model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return vec.astype(np.float32)


def get_cluster_memberships(embedding: np.ndarray) -> np.ndarray:
    """
    Project the query into PCA space, then compute FCM membership distribution.
    This is the same pipeline used during corpus clustering.
    """
    reduced = state.pca.transform(embedding.reshape(1, -1))    # (1, 50)
    U       = state.fcm.predict_soft(reduced)                  # (1, k)
    return U[0]                                                # (k,)


def search_chroma(embedding: np.ndarray, n: int = N_RESULTS) -> list[dict]:
    """Run semantic search in ChromaDB; return top-n results with metadata."""
    results = state.collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":              doc[:300],
            "similarity":        round(1.0 - dist, 4),    # ChromaDB returns cosine distance
            "true_category":     meta.get("true_category"),
            "dominant_cluster":  meta.get("dominant_cluster"),
        })
    return hits


# ─── SCHEMAS ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:     str   = Field(..., min_length=3, description="Natural language query")
    threshold: Optional[float] = Field(
        None, gt=0.0, le=1.0,
        description="Override cache similarity threshold for this request"
    )
    n_results: Optional[int]   = Field(None, ge=1, le=20)


class QueryResponse(BaseModel):
    query:             str
    cache_hit:         bool
    matched_query:     Optional[str]
    similarity_score:  Optional[float]
    result:            list[dict]
    dominant_cluster:  int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count:     int
    miss_count:    int
    hit_rate:      float
    threshold:     float
    n_clusters:    int


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    POST /query
    -----------
    1. Embed the query.
    2. Compute fuzzy cluster memberships.
    3. Check the semantic cache (using cluster buckets for efficiency).
    4. On hit:  return cached result.
    5. On miss: run ChromaDB semantic search, store in cache, return result.
    """
    if not state.embed_model:
        raise HTTPException(503, "Models not yet loaded")

    cache = get_cache()
    if request.threshold is not None:
        # Temporarily honour per-request threshold override
        original = cache.threshold
        cache.set_threshold(request.threshold)

    n = request.n_results or N_RESULTS

    try:
        # Step 1 & 2: embed + cluster
        q_emb  = embed_query(request.query)
        q_memb = get_cluster_memberships(q_emb)
        dom_c  = int(np.argmax(q_memb))

        # Step 3: cache lookup
        hit = cache.get(q_emb, q_memb)

        if hit is not None:
            entry, score = hit
            return QueryResponse(
                query=request.query,
                cache_hit=True,
                matched_query=entry.query_text,
                similarity_score=round(score, 4),
                result=entry.result,
                dominant_cluster=dom_c,
            )

        # Step 4: miss — compute and store
        results = search_chroma(q_emb, n=n)
        cache.put(
            query_text=request.query,
            query_embedding=q_emb,
            result=results,
            cluster_memberships=q_memb,
        )

        return QueryResponse(
            query=request.query,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result=results,
            dominant_cluster=dom_c,
        )
    finally:
        if request.threshold is not None:
            cache.set_threshold(original)


@app.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats():
    """
    GET /cache/stats
    ----------------
    Returns current cache state: total entries, hit/miss counts, hit rate,
    active threshold, and cluster count.
    """
    return get_cache().stats


@app.delete("/cache")
def flush_cache():
    """
    DELETE /cache
    -------------
    Flushes all cache entries and resets hit/miss statistics.
    """
    get_cache().flush()
    return {"message": "Cache flushed successfully", "stats": get_cache().stats}


@app.get("/")
def root():
    return {
        "service": "Trademarkia Semantic Search",
        "status":  "ok",
        "docs_url": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "docs_indexed": state.collection.count() if state.collection else 0,
        "n_clusters":   state.n_clusters,
        "cache_stats":  get_cache().stats,
    }