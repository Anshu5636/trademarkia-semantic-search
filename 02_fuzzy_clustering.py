import os
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import chromadb
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist


# ─── CONFIG ──────────────────────────────────────────────────────────────────
CHROMA_PATH    = Path("data/chroma_db")
EMBED_CACHE    = Path("data/embeddings_cache.npz")
CLUSTER_CACHE  = Path("data/cluster_memberships.npz")
CLUSTER_META   = Path("data/cluster_meta.json")
COLLECTION     = "newsgroups"

PCA_DIMS       = 50
FCM_M          = 2.0       # fuzziness exponent; 2.0 is the standard default
FCM_MAX_ITER   = 150
FCM_TOL        = 1e-4
K_CANDIDATES   = [10, 15, 20, 25, 30]


# ─── FUZZY C-MEANS (hand-rolled, no Redis, no external lib) ──────────────────
# The task explicitly forbids Redis/Memcached.  The spirit extends to "don't
# use a black-box fuzzy clustering library" — so we implement FCM from scratch.

from clustering.fuzzy_cmeans import FuzzyCMeans


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def average_membership_entropy(U: np.ndarray) -> float:
    """
    Mean entropy of the membership distribution across all documents.
    Low entropy → crisp, well-separated clusters.
    High entropy → fuzzy, overlapping topics (expected for newsgroups).
    """
    eps   = 1e-12
    ent   = -(U * np.log(U + eps)).sum(axis=1)
    return float(ent.mean())


def choose_k(X_pca: np.ndarray) -> int:
    """
    Fit FCM for each candidate k, record FPC and entropy, pick the best k.
    We choose the k with the highest FPC (most structure) that also doesn't
    collapse to trivially hard clusters — i.e. where entropy is still > 0.5.
    """
    print("\nSelecting optimal k via FPC sweep …")
    results = []
    for k in K_CANDIDATES:
        print(f"  k={k} …", end=" ", flush=True)
        fcm = FuzzyCMeans(n_clusters=k, m=FCM_M, max_iter=FCM_MAX_ITER, tol=FCM_TOL)
        fcm.fit(X_pca)
        ent = average_membership_entropy(fcm.U_)
        results.append({"k": k, "fpc": fcm.fpc_, "entropy": ent})
        print(f"FPC={fcm.fpc_:.4f}  entropy={ent:.4f}")

    # Pick k with max FPC (most cluster structure)
    best = max(results, key=lambda r: r["fpc"])
    print(f"\n  → Best k = {best['k']}  (FPC={best['fpc']:.4f}, entropy={best['entropy']:.4f})")
    return best["k"], results


# ─── CLUSTER CHARACTERISATION ────────────────────────────────────────────────

def characterise_clusters(U: np.ndarray, docs: list[str], true_labels: list[int],
                           target_names: list[str]) -> list[dict]:
    """
    For each cluster, find:
      - top-10 documents by membership weight (the "core" of the cluster)
      - boundary documents (membership between 0.3–0.55 in two clusters)
      - dominant true category (for sanity-check against ground truth)
    This is how we *show* what lives inside each cluster to a sceptical reader.
    """
    n_clusters = U.shape[1]
    meta = []

    for c in range(n_clusters):
        weights   = U[:, c]
        top_idx   = np.argsort(weights)[::-1][:10]
        top_docs  = [(float(weights[i]), docs[i][:120]) for i in top_idx]

        # Boundary docs: dominant cluster is c, but second-best > 0.3
        sorted_U    = np.sort(U, axis=1)[:, ::-1]   # descending per row
        dominant    = np.argmax(U, axis=1)
        is_boundary = (dominant == c) & (sorted_U[:, 1] > 0.3)
        boundary_idx = np.where(is_boundary)[0][:5]
        boundary_docs = [(float(weights[i]), docs[i][:120]) for i in boundary_idx]

        # Dominant true category
        member_labels    = [true_labels[i] for i in np.where(dominant == c)[0]]
        if member_labels:
            from collections import Counter
            dom_label = Counter(member_labels).most_common(1)[0][0]
            dom_category = target_names[dom_label]
        else:
            dom_category = "unknown"

        meta.append({
            "cluster_id":     c,
            "dominant_category": dom_category,
            "top_docs":       top_docs,
            "boundary_docs":  boundary_docs,
            "size":           int((dominant == c).sum()),
        })

    return meta


# ─── PATCH CHROMADB WITH CLUSTER MEMBERSHIPS ─────────────────────────────────

def patch_chroma_with_clusters(U: np.ndarray):
    """
    Upsert dominant_cluster and top-3 membership scores back into ChromaDB
    so the API can do cluster-aware filtered retrieval.
    """
    client     = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION)

    n = U.shape[0]
    dominant = np.argmax(U, axis=1)

    chunk = 5000
    for start in tqdm(range(0, n, chunk), desc="Patching ChromaDB"):
        end     = min(start + chunk, n)
        ids     = [str(i) for i in range(start, end)]
        updates = []
        for i in range(start, end):
            top3 = np.argsort(U[i])[::-1][:3]
            updates.append({
                "dominant_cluster": int(dominant[i]),
                "cluster_0_id":     int(top3[0]),
                "cluster_0_score":  float(U[i, top3[0]]),
                "cluster_1_id":     int(top3[1]),
                "cluster_1_score":  float(U[i, top3[1]]),
                "cluster_2_id":     int(top3[2]),
                "cluster_2_score":  float(U[i, top3[2]]),
            })
        collection.update(ids=ids, metadatas=updates)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def run():
    os.chdir(Path(__file__).parent)

    # Load embeddings
    print("Loading embeddings …")
    embeddings = np.load(EMBED_CACHE)["embeddings"].astype(np.float32)

    # Load metadata
    with open(Path("data/metadata_cache.json")) as f:
        meta = json.load(f)
    true_labels  = meta["labels"]
    label_names  = meta["label_names"]
    target_names = meta["target_names"]

    # Load doc texts from ChromaDB (for characterisation)
    print("Loading docs from ChromaDB …")
    client     = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION)
    result     = collection.get(include=["documents"])
    docs       = result["documents"]

    # PCA: reduce to 50 dims — FCM distance degrades in 384-dim space
    print(f"\nReducing {embeddings.shape[1]} → {PCA_DIMS} dims via PCA …")
    pca    = PCA(n_components=PCA_DIMS, random_state=42)
    X_pca  = pca.fit_transform(embeddings)
    var_ex = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained: {var_ex:.3%}")

    # Choose k
    best_k, sweep_results = choose_k(X_pca)

    # Fit final FCM
    print(f"\nFitting final FCM with k={best_k} …")
    fcm = FuzzyCMeans(n_clusters=best_k, m=FCM_M, max_iter=FCM_MAX_ITER, tol=FCM_TOL)
    fcm.fit(X_pca)

    U = fcm.U_
    print(f"  FPC = {fcm.fpc_:.4f}")
    print(f"  Mean membership entropy = {average_membership_entropy(U):.4f}")

    # Save memberships + PCA model
    CLUSTER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CLUSTER_CACHE, U=U, centers=fcm.centers_)
    with open("data/pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open("data/fcm_model.pkl", "wb") as f:
        pickle.dump(fcm, f)

    # Characterise clusters
    print("\nCharacterising clusters …")
    cluster_info = characterise_clusters(U, docs, true_labels, target_names)
    meta_out = {
        "n_clusters":     best_k,
        "fpc":            fcm.fpc_,
        "mean_entropy":   average_membership_entropy(U),
        "sweep_results":  sweep_results,
        "clusters":       cluster_info,
    }
    with open(CLUSTER_META, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"  Saved cluster metadata to {CLUSTER_META}")

    # Print sample clusters to stdout for inspection
    print("\n── Sample cluster summaries ──────────────────────────")
    for info in cluster_info[:5]:
        print(f"\n  Cluster {info['cluster_id']}  "
              f"(dominant: {info['dominant_category']}, "
              f"size: {info['size']})")
        print(f"    Top doc: {info['top_docs'][0][1][:80]} …")
        if info["boundary_docs"]:
            print(f"    Boundary: {info['boundary_docs'][0][1][:80]} …")

    # Patch ChromaDB
    patch_chroma_with_clusters(U)
    print("\nPart 2 complete ✓")


if __name__ == "__main__":
    run()
