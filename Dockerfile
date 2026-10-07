# ---- Multi-stage build: small, cache-friendly, non-root image ----------------
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS deps
# Install deps first so code changes don't invalidate this (slow) layer.
COPY requirements.txt requirements-ml.txt ./
ARG WITH_ML=true
RUN pip install -r requirements.txt \
 && if [ "$WITH_ML" = "true" ]; then pip install -r requirements-ml.txt; fi

FROM deps AS runtime
ARG WITH_ML=true
# Bake the embedding model into the image -> no download at pod start.
RUN if [ "$WITH_ML" = "true" ]; then python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"; fi
COPY app ./app
COPY eval ./eval
COPY data/sample ./data/sample
RUN useradd -m appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
