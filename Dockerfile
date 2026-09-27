# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

WORKDIR /app

# تثبيت الأدوات الأساسية
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# تثبيت المكتبات مباشرة بدون نقل بيئات افتراضية
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -e .

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]