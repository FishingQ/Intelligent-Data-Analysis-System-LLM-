#!/usr/bin/env bash
set -e

# 梧桐引力 - 智能数据分析系统 启动脚本 (Linux/Mac)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  梧桐引力 - 智能数据分析系统"
echo "  Wutong Gravity - AI Data Analysis"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] 未找到 python3，请先安装 Python 3.10+"
    exit 1
fi

# 检查依赖
echo "[1/3] 检查依赖..."
if ! python3 -c "import streamlit, fastapi, pandas" &> /dev/null; then
    echo "[INFO] 正在安装依赖..."
    pip install -r requirements.txt -q
fi
echo "      依赖检查完成"

# 启动后端
echo "[2/3] 启动后端 API (port 8000)..."
python3 -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "      后端 PID: $BACKEND_PID"

# 等待后端就绪
sleep 3

# 启动前端
echo "[3/3] 启动前端 Web (port 8501)..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
echo "      前端 PID: $FRONTEND_PID"

echo ""
echo "========================================"
echo "  启动完成！"
echo "  后端 API:  http://localhost:8000"
echo "  前端界面:  http://localhost:8501"
echo "  API 文档:  http://localhost:8000/docs"
echo "========================================"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 捕获退出信号，清理子进程
cleanup() {
    echo ""
    echo "正在停止服务..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "已停止。"
}
trap cleanup EXIT INT TERM

# 等待子进程
wait
