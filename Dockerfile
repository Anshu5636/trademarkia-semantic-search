# ── Build stage: install dependencies ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Pre-download the embedding model so the container is self-contained
# (avoids a network call on first request in production)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Persist chroma DB and model caches as a named volume
VOLUME ["/app/data"]

EXPOSE 8000

# Run the pipeline scripts if data directory is empty, then start the API.
# In production you'd separate these into an init job + the API deployment.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]