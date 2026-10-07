Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "       ⚡ PRODUCTION RAG - 100% LOCAL ZERO-TOUCH LAUNCHER         " -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "[+] Docker detected. Launching local stack..." -ForegroundColor Green
    docker compose up -d
    Write-Host ""
    Write-Host "[+] Stack running!" -ForegroundColor Green
    Write-Host "    🖥️  Web UI:     http://localhost:8000"
    Write-Host "    📚  Swagger API: http://localhost:8000/docs"
    Write-Host "    📊  Metrics:    http://localhost:8000/metrics"
    Write-Host ""
    Write-Host "[*] Ingest sample documents:" -ForegroundColor Yellow
    Write-Host "    docker exec -it rag_api python main.py ingest /app/sample_documents/"
    Write-Host ""
    Write-Host "[*] Query sample documents:" -ForegroundColor Yellow
    Write-Host "    docker exec -it rag_api python main.py query `"What is the annual lodging budget for L3 employees?`""
} else {
    Write-Host "[!] Docker not found. Starting local Python environment..." -ForegroundColor Yellow
    pip install -e ".[all]"
    python main.py doctor
    python main.py serve
}
