FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=8080

WORKDIR /app

# Dependencies first so Docker can cache this layer. The extra index installs the
# CPU-only build of PyTorch (used by sentence-transformers) instead of the multi-GB GPU build.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Download the embedding model at build time so the first request is not slow.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY templates ./templates
COPY static ./static
COPY data/sample_kb ./data/sample_kb
COPY ingest.py ./

# Run as an unprivileged user. /app/state is where a persistent volume can be mounted
# (set AUDIT_DB_PATH=/app/state/audit.db and UPLOAD_DIR=/app/state/uploads).
RUN useradd --create-home appuser \
    && mkdir -p /app/uploads /app/state \
    && chown -R appuser /app
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8080\")}/api/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
