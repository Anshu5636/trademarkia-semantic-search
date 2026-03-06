import re
import os
import json
import zipfile
import urllib.request
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import chromadb


# ─── CONFIG ──────────────────────────────────────────────────────────────────
UCI_URL       = "https://archive.ics.uci.edu/static/public/113/twenty+newsgroups.zip"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR      = Path("data/raw")
CHROMA_PATH   = Path("data/chroma_db")
EMBED_CACHE   = Path("data/embeddings_cache.npz")
META_CACHE    = Path("data/metadata_cache.json")
COLLECTION    = "newsgroups"
BATCH_SIZE    = 512
MAX_TOKENS    = 300


# ─── CLEANING ────────────────────────────────────────────────────────────────
_HEADER_RE = re.compile(
    r"^(From|Subject|Organization|Lines|Nntp-Posting-Host|"
    r"Reply-To|Distribution|Newsgroups|Path|Message-ID|Date|"
    r"X-Newsreader|Xref|Summary|Keywords|In-Reply-To):\s.*$",
    re.MULTILINE | re.IGNORECASE,
)
_EMAIL_RE   = re.compile(r"\S+@\S+")
_URL_RE     = re.compile(r"http\S+|www\.\S+")
_QUOTE_RE   = re.compile(r"^>.*$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s{2,}")


def clean(text: str) -> str:
    """
    Remove boilerplate from a newsgroup post.
    - Headers removed: they leak category via domain names
    - Quoted lines removed: they repeat other posts
    - Emails/URLs removed: not semantically meaningful
    - Truncated to MAX_TOKENS words
    """
    text = _HEADER_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)
    text = _EMAIL_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = text.strip()
    words = text.split()
    if len(words) > MAX_TOKENS:
        text = " ".join(words[:MAX_TOKENS])
    return text


def is_useful(text: str) -> bool:
    """
    Filter out posts < 20 words after cleaning.
    These are signature blocks, one-liner replies, or corrupted entries.
    """
    return len(text.split()) >= 20


# ─── DOWNLOAD FROM UCI ───────────────────────────────────────────────────────

def download_uci_dataset():
    """
    Download directly from the UCI link specified in the task.
    https://archive.ics.uci.edu/dataset/113/twenty+newsgroups
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "twenty_newsgroups.zip"

    if zip_path.exists():
        print(f"  Already downloaded — using cached zip")
    else:
        print(f"  Downloading from UCI: {UCI_URL}")
        urllib.request.urlretrieve(UCI_URL, zip_path)
        print(f"  Saved to {zip_path}")

    extract_dir = DATA_DIR / "extracted"
    if not extract_dir.exists():
        print("  Extracting zip ...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        print(f"  Extracted to {extract_dir}")
    else:
        print(f"  Already extracted")

    return extract_dir


def find_newsgroups_root(base: Path) -> Path:
    """
    The UCI zip may nest folders differently depending on version.
    Walk until we find a folder containing 'alt.atheism'.
    """
    for path in base.rglob("alt.atheism"):
        return path.parent
    raise FileNotFoundError(
        f"Could not find newsgroup category folders inside {base}."
    )


def load_from_folder(news_dir: Path):
    categories = sorted([
        d.name for d in news_dir.iterdir() if d.is_dir()
    ])
    print(f"  Found {len(categories)} categories: {categories[:3]} ...")

    cat_to_label = {cat: i for i, cat in enumerate(categories)}
    docs, labels, label_names = [], [], []
    skipped = 0

    for category in tqdm(categories, desc="  Loading"):
        cat_dir = news_dir / category
        for filepath in cat_dir.iterdir():
            if not filepath.is_file():
                continue
            try:
                text    = filepath.read_text(encoding="utf-8", errors="replace")
                cleaned = clean(text)
                if not is_useful(cleaned):
                    skipped += 1
                    continue
                docs.append(cleaned)
                labels.append(cat_to_label[category])
                label_names.append(category)
            except Exception:
                skipped += 1

    print(f"  Kept {len(docs):,} docs  |  skipped {skipped:,}")
    return docs, labels, label_names, categories


# ─── LOAD DATASET ────────────────────────────────────────────────────────────

def load_dataset():
    """
    Primary:  UCI link from the task
    Fallback: sklearn mirror (identical data, different host)
    """
    print("\n── Loading 20 Newsgroups Dataset ──────────────────────")
    print(f"  Source: {UCI_URL}")

    # Try UCI first
    try:
        extract_dir = download_uci_dataset()
        news_dir    = find_newsgroups_root(extract_dir)
        docs, labels, label_names, target_names = load_from_folder(news_dir)
        if len(docs) > 100:
            return docs, labels, label_names, target_names
    except Exception as e:
        print(f"\n  UCI download failed ({e})")
        print(  "  Falling back to sklearn mirror — same dataset, different host")

    # Fallback: sklearn mirror
    from sklearn.datasets import fetch_20newsgroups
    raw = fetch_20newsgroups(subset="all", remove=(), shuffle=True, random_state=42)
    docs, labels, label_names = [], [], []
    skipped = 0
    for text, label in zip(raw.data, raw.target):
        cleaned = clean(text)
        if not is_useful(cleaned):
            skipped += 1
            continue
        docs.append(cleaned)
        labels.append(int(label))
        label_names.append(raw.target_names[label])
    print(f"  Kept {len(docs):,} docs  |  skipped {skipped:,}")
    return docs, labels, label_names, list(raw.target_names)


# ─── EMBED ───────────────────────────────────────────────────────────────────

def embed_corpus(docs: list) -> np.ndarray:
    """
    Load from cache if available — re-embedding 18k docs takes ~10 min.
    Cache stored in data/embeddings_cache.npz.
    """
    if EMBED_CACHE.exists():
        print(f"\nLoading cached embeddings from {EMBED_CACHE} ...")
        return np.load(EMBED_CACHE)["embeddings"]

    print(f"\nEmbedding {len(docs):,} docs with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        docs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBED_CACHE, embeddings=embeddings)
    print(f"  Saved to {EMBED_CACHE}")
    return embeddings


# ─── STORE IN CHROMADB ───────────────────────────────────────────────────────

def build_vector_store(docs, embeddings, labels, label_names):
    """
    Persist embeddings + metadata into ChromaDB.
    ChromaDB uses HNSW index with cosine space — matches our L2-normalised vecs.
    """
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids       = [str(i) for i in range(len(docs))]
    metadatas = [
        {"true_label": labels[i], "true_category": label_names[i]}
        for i in range(len(docs))
    ]

    chunk = 5000
    for start in tqdm(range(0, len(docs), chunk), desc="Storing in ChromaDB"):
        end = min(start + chunk, len(docs))
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    print(f"  ChromaDB collection '{COLLECTION}' -> {collection.count():,} docs")
    return collection


# ─── MAIN ────────────────────────────────────────────────────────────────────

def run():
    os.chdir(Path(__file__).parent)

    docs, labels, label_names, target_names = load_dataset()

    META_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_CACHE, "w") as f:
        json.dump({
            "labels":       labels,
            "label_names":  label_names,
            "target_names": list(target_names),
            "n_docs":       len(docs)
        }, f)

    embeddings = embed_corpus(docs)
    build_vector_store(docs, embeddings, labels, label_names)
    print("\nPart 1 complete ✓")


if __name__ == "__main__":
    run()
