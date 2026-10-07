#!/usr/bin/env bash
set -e

echo "==================================================================="
echo "       ⚡ PRODUCTION RAG - 100% LOCAL ZERO-TOUCH LAUNCHER         "
echo "==================================================================="

# Check for Docker
if command -v docker >/dev/null 2>&1; then
    echo "[+] Docker detected. Launching local stack..."
    docker compose up -d
    echo ""
    echo "[+] Stack running!"
    echo "    🖥️  Web UI:     http://localhost:8000"
    echo "    📚  Swagger API: http://localhost:8000/docs"
    echo "    📊  Metrics:    http://localhost:8000/metrics"
    echo ""
    echo "[*] Ingest sample documents:"
    echo "    docker exec -it rag_api python main.py ingest /app/sample_documents/"
    echo ""
    echo "[*] Query sample documents:"
    echo "    docker exec -it rag_api python main.py query \"What is the annual lodging budget for L3 employees?\""
else
    echo "[!] Docker not found. Starting local Python environment..."
    pip install -e ".[all]"
    python main.py doctor
    python main.py serve
fi
