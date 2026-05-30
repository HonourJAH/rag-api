# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install \
    --timeout=300 \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefix=/install \
    --timeout=300 \
    -r requirements.txt

ENV HF_HOME=/app/.cache/huggingface


# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --create-home appuser

COPY --from=builder /install /usr/local

# Copy the sentence-transformer model cache from builder
# ~/.cache/huggingface is where sentence-transformers stores downloaded models

COPY --from=builder /app/.cache /home/appuser/.cache
RUN chown -R appuser:appgroup /home/appuser/.cache

COPY app/ ./app/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["./start.sh"]
