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
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python - <<'PY'
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from tokenizers import Tokenizer
TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
print("Models cached.")
PY

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
