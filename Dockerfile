# Railway / 通用容器部署镜像
# 特点：端口读 $PORT（平台注入）、构建期预热 ONNX 模型、数据随镜像内置
FROM python:3.11-slim

# PYTHONUNBUFFERED 让日志实时输出到 Railway 面板
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 构建期预热 Chroma 默认嵌入模型（all-MiniLM-L6-v2，约 83MB）
# 避免运行期首次向量检索时联网下载失败
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2 as M; M()(['warmup'])"

COPY . .

# 运行期会写入日志与评测结果，确保目录存在且可写
RUN mkdir -p /app/data/logs /app/data/eval && chmod -R 777 /app/data

EXPOSE 8501

# 平台不提供 curl，用 Python 标准库做健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8501')+'/_stcore/health',timeout=4)"

# 注意用 sh -c 以展开 ${PORT}；Railway 会注入 PORT，本地默认 8501
CMD ["sh", "-c", "python -m streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]
