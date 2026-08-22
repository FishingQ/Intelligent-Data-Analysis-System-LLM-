@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Wutong Gravity - AI Data Analysis

cd /d "%~dp0"
set "PROJECT_DIR=%CD%"

echo ========================================
echo   Wutong Gravity - AI Data Analysis
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found. Please install Python 3.10+.
    pause
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ is required.
    python --version
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating local virtual environment: .venv
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

set "PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "PATH=%PROJECT_DIR%\.venv\Scripts;%PATH%"

rem Clear global pip/proxy variables that can break dependency installation.
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "http_proxy="
set "https_proxy="
set "ALL_PROXY="
set "all_proxy="
set "PIP_NO_INDEX="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn"

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt was not found. Please run this file from the project root.
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
"%PYTHON%" -c "import streamlit, fastapi, pandas, watchfiles, uvicorn, langchain_openai, duckdb, sklearn, yaml" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies. First run may take several minutes...
    "%PYTHON%" -m pip install -r requirements.txt --prefer-binary --timeout 30 --retries 3 --index-url "%PIP_INDEX_URL%" --trusted-host "%PIP_TRUSTED_HOST%"
    if errorlevel 1 (
        echo [WARN] Full install failed. Retrying without optional prophet package...
        findstr /V /R /C:"^prophet" requirements.txt > "%TEMP%\wutong-requirements-core.txt"
        "%PYTHON%" -m pip install -r "%TEMP%\wutong-requirements-core.txt" --prefer-binary --timeout 30 --retries 3 --index-url "%PIP_INDEX_URL%" --trusted-host "%PIP_TRUSTED_HOST%"
        if errorlevel 1 (
            echo [ERROR] Dependency installation failed.
            echo Run this manually to see details:
            echo "%PYTHON%" -m pip install -r requirements.txt --prefer-binary --index-url "%PIP_INDEX_URL%" --trusted-host "%PIP_TRUSTED_HOST%"
            pause
            exit /b 1
        )
    )
)
echo       Dependencies are ready.

set "API_BASE=http://127.0.0.1:8000"

echo [2/3] Starting backend API on port 8000...
start "Wutong Backend" /D "%PROJECT_DIR%" cmd /k call "%PYTHON%" -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload

echo       Waiting for backend startup...
timeout /t 5 /nobreak >nul

echo [3/3] Starting frontend Web on port 8501...
start "Wutong Frontend" /D "%PROJECT_DIR%" cmd /k call "%PYTHON%" -m streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1

echo.
echo ========================================
echo   Started
echo   Backend API:  http://127.0.0.1:8000
echo   Frontend:     http://127.0.0.1:8501
echo   API Docs:     http://127.0.0.1:8000/api/docs
echo ========================================
echo.
echo If the frontend says the backend is disconnected, wait a few seconds and refresh.
echo Press any key to open the frontend...
pause >nul
start http://127.0.0.1:8501
