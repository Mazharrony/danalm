# DanaLM service (Phase 7, D-034): FastAPI + ONNX Runtime on the CPU, no PyTorch.
# Models are never committed, so the model directory is mounted at run time:
#   docker build -t danalm-serve .
#   docker run --rm -p 8000:8000 -v "$PWD/artifacts/deploy/int8:/model:ro" danalm-serve
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DANALM_MODEL_DIR=/model \
    DANALM_THREADS=4

WORKDIR /app
COPY deploy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# only the torch-free modules the service imports
COPY src/danalm/__init__.py src/danalm/text.py /app/danalm/
COPY src/danalm/sft/__init__.py src/danalm/sft/format.py /app/danalm/sft/
COPY src/danalm/infer/ /app/danalm/infer/
COPY src/danalm/serve/ /app/danalm/serve/

RUN useradd --create-home --uid 10001 danalm
USER danalm
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "danalm.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
