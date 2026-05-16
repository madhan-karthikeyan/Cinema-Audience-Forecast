FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

RUN groupadd -r forecast && useradd -r -g forecast forecast && \
    mkdir -p /app/data /app/models /app/feature_schema.json && \
    chown -R forecast:forecast /app

COPY --from=builder /root/.local /root/.local
COPY app/ app/
COPY feature_schema.json .
COPY models/ models/

ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

USER forecast

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/ready')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--limit-max-requests", "10000", "--timeout-keep-alive", "120"]
