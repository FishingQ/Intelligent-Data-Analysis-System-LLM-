#!/bin/bash
# ============================================================
# 云端启动脚本
# 同时启动 FastAPI 后端 (8000) 和 Streamlit 前端 (8501)
# ============================================================

set -e

echo "========================================="
echo "  智能数据分析系统 - 云端启动"
echo "  Provider: ${LLM_PROVIDER:-deepseek}"
echo "  Model:    ${LLM_MODEL:-deepseek-chat}"
echo "========================================="

# 确保上传目录存在
mkdir -p /app/data/uploads

# 启动后端 (后台)
cd /app
python -m uvicorn backend.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers ${WORKERS:-2} &

# 等待后端就绪
echo "等待后端就绪..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "后端已就绪"
        break
    fi
    sleep 2
done

# 启动前端 (前台)
export API_BASE="${API_BASE:-http://localhost:8000}"
echo "API_BASE=$API_BASE"
exec python -m streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
