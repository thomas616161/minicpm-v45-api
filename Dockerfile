FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

# HF 캐시 → 네트워크 볼륨(/runpod-volume)에 고정
ENV HF_HUB_CACHE=/runpod-volume/hf-cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf-cache

# 기본 하이퍼파라미터(필요시 콘솔에서 환경변수로 덮어쓰기)
ENV ATTN_IMPL=sdpa
ENV DTYPE=bfloat16
ENV MAX_SIDE=1280

COPY app.py ./

# Runpod LB 헬스체크는 /ping (PORT_HEALTH와 동일 포트)
ENV PORT=80
ENV PORT_HEALTH=80
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","80","--workers","1"]
