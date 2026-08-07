# ============================================================
# 智能数据分析系统 - 云端部署 Dockerfile
# 构建: docker build -t ai-data-analysis .
# 运行: docker run -p 8000:8000 -p 8501:8501 -e DEEPSEEK_API_KEY="sk-xxx" ai-data-analysis
# ============================================================

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/data/uploads

EXPOSE 8000 8501

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
