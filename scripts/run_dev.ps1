# PowerShell setup and development server runner for ai-data-agent
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Starting AI-Native Data Agent & RAG Microservice " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Virtual Environment Setup
if (-not (Test-Path ".\venv")) {
    Write-Host "[*] Creating Python virtual environment in .\venv..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "[*] Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

# 2. Dependencies
Write-Host "[*] Installing/updating requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# 3. Environment configuration check
if (-not (Test-Path ".env")) {
    Write-Host "[!] No .env file found. Copying .env.example to .env..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "[i] Please update .env with your GEMINI_API_KEY if using LLM reasoning." -ForegroundColor Gray
}

# 4. Start Uvicorn Server
Write-Host "[*] Starting Uvicorn development server on http://localhost:8000..." -ForegroundColor Cyan
Write-Host "[*] Swagger Documentation: http://localhost:8000/docs" -ForegroundColor Green
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
