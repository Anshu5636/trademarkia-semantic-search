# Trademarkia — AI & ML Engineer Task
## Semantic Search System over 20 Newsgroups

---

## Quick Start

```bash
# 1. Create and activate venv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Embed corpus and store in ChromaDB  (~5 min first run, cached after)
python 01_embed_and_store.py

# 4. Fuzzy cluster the corpus  (~3 min)
python 02_fuzzy_clustering.py

# 5. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

---

## Architecture Overview

```
20 Newsgroups Dataset
       │
       ▼
01_embed_and_store.py         ← Part 1
  • Clean posts (remove headers, quotes, URLs)
  • Embed with all-MiniLM-L6-v2
  • Store in ChromaDB (persistent vector DB)
       │
       ▼
02_fuzzy_clustering.py        ← Part 2
  • PCA: 384 → 50 dims
  • Sweep k ∈ {10,15,20,25,30} with FPC metric
  • Fit Fuzzy C-Means (hand-rolled)
  • Patch ChromaDB with cluster memberships
       │
       ▼
cache/semantic_cache.py       ← Part 3
  • Cluster-bucketed O(n/k) lookup
  • Tunable cosine threshold θ (default 0.85)
  • Thread-safe, no external dependencies
       │
       ▼
api/main.py                   ← Part 4
  • POST /query
  • GET  /cache/stats
  • DELETE /cache
```

---

## Part 1 — Embedding & Vector Database

### Cleaning decisions
| What we remove | Why |
|---|---|
| Header lines (From/Subject/Organization …) | Leak category via domain names — makes clustering trivial |
| Quoted reply lines (`> …`) | Repeat other people's posts; add noise |
| Email addresses & URLs | Not semantically meaningful tokens |
| Posts < 20 words after cleaning | Signature blocks and one-liners; no semantic content |
| Posts > 300 words | Truncated — long posts add marginal signal at high cost |

### Embedding model: `all-MiniLM-L6-v2`
- 22M parameters, 384-dim output, ~5 min for 18k docs on CPU
- Benchmarks at 68.1 on SBERT MTEB semantic similarity tasks
- Chosen over `all-mpnet-base-v2` (better but 3× slower) because the semantic
  similarity quality difference is small for this use case

### Vector store: ChromaDB
- Embedded (no separate server), file-persisted, supports metadata filtering
- Uses HNSW with cosine space — matches our L2-normalised embeddings
- Alternative: FAISS is faster for pure ANN but lacks persistence and metadata

---

## Part 2 — Fuzzy Clustering

### Why Fuzzy C-Means, not k-means
A post about gun legislation doesn't belong to *either* `talk.politics.guns` or
`rec.guns` — it belongs to *both*, to varying degrees.  FCM models this by
outputting a **membership distribution** over clusters rather than a hard label.

### Choosing k
We evaluate k ∈ {10, 15, 20, 25, 30} using:

**Fuzzy Partition Coefficient (FPC)**:
```
FPC = (1/n) Σᵢ Σⱼ uᵢⱼ²
```
FPC = 1 → perfectly hard clusters.  FPC = 1/k → random.
We pick the k where FPC shows an "elbow" — stops improving substantially.

**Average Membership Entropy**:
```
H = -(1/n) Σᵢ Σⱼ uᵢⱼ log(uᵢⱼ)
```
Lower entropy → crisper, more distinct clusters.

### PCA before clustering
FCM distance metrics degrade in high-dimensional space (curse of dimensionality).
We reduce 384 → 50 dims with PCA first, retaining ~95% of variance.

### Cluster analysis
For each cluster we identify:
- **Core documents**: top-10 by membership weight (what the cluster is about)
- **Boundary documents**: documents with >0.3 membership in a second cluster (the most interesting cases)
- **Dominant true category**: cross-check against ground-truth labels

---

## Part 3 — Semantic Cache

### Design
```
query text
    │  embed
    ▼
query_embedding (384-dim, L2-normalised)
    │  PCA + FCM
    ▼
cluster_memberships (k-dim)
    │
    ├─ top-2 clusters → narrow candidate set (O(n/k) average)
    │
    └─ cosine similarity scan → return best match if score ≥ θ
```

### The tunable parameter θ
| θ | Behaviour |
|---|---|
| 0.95+ | Only near-identical phrasings hit |
| **0.85** | **Paraphrases hit (default)** |
| 0.75 | Topically related queries hit |
| < 0.70 | Different-topic queries wrongly return cached results |

The API accepts a per-request `threshold` override so you can explore
what different values reveal.  Try `θ=0.90` vs `θ=0.80` with:
- "space station missions"
- "NASA spacecraft in orbit"
- "rocket launch schedule"

### No external dependencies
The cache is a plain Python list + dict.  No Redis, no Memcached, no vector DB.
Thread-safety via `threading.RLock`.

---

## Part 4 — FastAPI Endpoints

### `POST /query`
```json
Request:  { "query": "nuclear power plants safety",
            "threshold": 0.85,   // optional override
            "n_results": 5 }     // optional

Response (cache hit):
{
  "query": "nuclear power plants safety",
  "cache_hit": true,
  "matched_query": "are nuclear reactors safe?",
  "similarity_score": 0.91,
  "result": [...],
  "dominant_cluster": 7
}

Response (cache miss):
{
  "query": "nuclear power plants safety",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    { "text": "...", "similarity": 0.87,
      "true_category": "sci.med", "dominant_cluster": 7 }
  ],
  "dominant_cluster": 7
}
```

### `GET /cache/stats`
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "threshold": 0.85,
  "n_clusters": 20
}
```

### `DELETE /cache`
Flushes all entries and resets all stats.

---

## Docker (Bonus)

```bash
# Build and run everything with docker-compose
docker compose up

# Or build and run just the API (assuming data/ already populated)
docker build -t trademarkia-search .
docker run -p 8000:8000 -v $(pwd)/data:/app/data trademarkia-search
```

---

## File Structure
```
trademarkia_search/
├── 01_embed_and_store.py      # Part 1: clean, embed, store
├── 02_fuzzy_clustering.py     # Part 2: PCA + hand-rolled FCM
├── cache/
│   ├── __init__.py
│   └── semantic_cache.py      # Part 3: cluster-bucketed semantic cache
├── api/
│   ├── __init__.py
│   └── main.py                # Part 4: FastAPI service
├── data/                      # generated — gitignore this
│   ├── chroma_db/
│   ├── embeddings_cache.npz
│   ├── metadata_cache.json
│   ├── cluster_memberships.npz
│   ├── cluster_meta.json
│   ├── pca_model.pkl
│   └── fcm_model.pkl
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Design Philosophy

> "This task is intentionally open-ended in places. Your design decisions and how you justify them matter as much as the code."

Every decision in this codebase is justified in code comments at the point of decision:
- **Why MiniLM over MPNet** → `01_embed_and_store.py` EMBED_MODEL comment
- **Why ChromaDB over FAISS** → `01_embed_and_store.py` build_vector_store docstring
- **Why FCM over k-means** → `02_fuzzy_clustering.py` module docstring
- **Why PCA before clustering** → `02_fuzzy_clustering.py` run() function
- **What θ reveals** → `cache/semantic_cache.py` module docstring
- **Why O(n/k) bucket scan** → `cache/semantic_cache.py` get() docstring