@echo off
chcp 65001 >nul
title 梧桐引力 - 智能数据分析系统

echo ========================================
echo   梧桐引力 - 智能数据分析系统
echo   Wutong Gravity - AI Data Analysis
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 检查依赖
echo [1/3] 检查依赖...
python -c "import streamlit, fastapi, pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] 正在安装依赖...
    pip install -r requirements.txt -q
    if %errorlevel% neq 0 (
        echo [ERROR] 依赖安装失败
        pause
        exit /b 1
    )
)
echo       依赖检查完成

:: 启动后端
echo [2/3] 启动后端 API (port 8000)...
start "梧桐引力-后端" cmd /c "cd /d %~dp0 && python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload"

:: 等待后端启动
echo       等待后端就绪...
timeout /t 3 /nobreak >nul

:: 启动前端
echo [3/3] 启动前端 Web (port 8501)...
start "梧桐引力-前端" cmd /c "cd /d %~dp0 && streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"

echo.
echo ========================================
echo   启动完成！
echo   后端 API:  http://localhost:8000
echo   前端界面:  http://localhost:8501
echo   API 文档:  http://localhost:8000/docs
echo ========================================
echo.
echo 按任意键打开前端界面...
pause >nul
start http://localhost:8501
