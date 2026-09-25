"""DanaLM as an HTTP service (Phase 7, D-034), without PyTorch.

Run: DANALM_MODEL_DIR=artifacts/deploy/int8 uv run uvicorn danalm.serve.app:app --port 8000
- POST /predict {"message": "..."} -> intent, reply, confidence, route ("on_device" when the
  answer is valid JSON, its reply fits the customer's language (D-035) and its confidence clears
  the threshold, else "escalate"), the checks, the PII-masked message (what may leave the
  device), latency.
- GET /health -> the model variant, its SHA-256 and the threshold.
Settings come from the environment: DANALM_MODEL_DIR (a directory from
scripts/export_onnx.py or scripts/quantize_onnx.py) and DANALM_THREADS (ONNX Runtime threads,
default 4). Message texts are never logged.
"""

import os
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from danalm import __version__
from danalm.infer.predictor import MessageTooLong, Predictor


class PredictRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class PredictResponse(BaseModel):
    intent: str | None
    reply: str | None
    confidence: float
    route: Literal["on_device", "escalate"]
    valid_json: bool
    reply_fits: bool
    finished: bool
    message_masked: str
    latency_ms: float


@lru_cache(maxsize=1)
def predictor() -> Predictor:
    """The model, loaded on the first request (or at startup) from the environment."""
    return Predictor(
        os.environ["DANALM_MODEL_DIR"], threads=int(os.environ.get("DANALM_THREADS", "4"))
    )


app = FastAPI(title="DanaLM", version=__version__,
              description="Intent and reply for UAE customer-service messages, on the CPU.")  # fmt: skip


@app.get("/health")
def health() -> dict[str, object]:
    p = predictor()
    return {"status": "ok", "variant": p.meta.get("variant"),
            "model_sha256": p.meta.get("model_sha256"), "threshold": p.threshold}  # fmt: skip


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict[str, object]:
    try:
        return predictor().predict(req.message)
    except MessageTooLong as err:
        raise HTTPException(status_code=422, detail=f"message too long: {err}") from None
