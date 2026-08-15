FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0 \
    FLAGS_use_mkldnn=false \
    FLAGS_cpu_deterministic=true \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=3838

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install paddlepaddle==3.2.0 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    pip install -r requirements.txt

COPY app ./app

EXPOSE 3838

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3838"]