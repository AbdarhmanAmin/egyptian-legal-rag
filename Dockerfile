FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir .

FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/mizan/.cache/huggingface

RUN groupadd --system --gid 10001 mizan \
    && useradd --system --uid 10001 --gid mizan --create-home --shell /usr/sbin/nologin mizan \
    && mkdir -p /app/qdrant_storage \
    && chown -R mizan:mizan /app /home/mizan

WORKDIR /app
COPY --from=builder --chown=mizan:mizan /opt/venv /opt/venv
COPY --chown=mizan:mizan src ./src
COPY --chown=mizan:mizan frontend ./frontend
COPY --chown=mizan:mizan params.yaml ./params.yaml

USER mizan
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
