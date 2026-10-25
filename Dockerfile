FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

RUN useradd --create-home --uid 1000 raguser
WORKDIR /app

COPY --from=builder /install /usr/local

# Pre-download local ONNX models into the image so containers start instantly
# and run in egress-restricted networks.
# HF_HOME is set before the build step so the tokenizer and Docling models
# land in /app/models too (the runtime user cannot read /root/.cache).
ENV ORT_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    MODEL_CACHE_DIR=/app/models \
    HF_HOME=/app/models/hf \
    DOCLING_ARTIFACTS_PATH=/app/models/docling

# Only the files the pull needs, so code edits keep this layer cached
COPY pipeline/__init__.py pipeline/config.py pipeline/models.py pipeline/
# fastembed models are required. Docling models are best effort: some come
# from www.modelscope.cn, and without them PDF parsing falls back to pypdf.
RUN python -c "from pipeline.models import pull_models; s = pull_models(); print(s); \
    import sys; sys.exit(any(v != 'ok' for k, v in s.items() if not k.startswith('docling')))"

# Every model is now in the image: block runtime downloads. A different
# LOCAL_EMBEDDING_MODEL in .env needs HF_HUB_OFFLINE=0 (or a rebuild).
ENV HF_HUB_OFFLINE=1

COPY pipeline/ pipeline/
COPY migrations/ migrations/
COPY static/ static/
COPY app.py worker.py main.py ./

RUN mkdir -p /app/uploads && chown -R raguser:raguser /app
USER raguser

EXPOSE 8000

# API by default; the worker container overrides the command.
CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", "--bind", "0.0.0.0:8000", "--timeout", "120", \
     "--access-logfile", "-"]
