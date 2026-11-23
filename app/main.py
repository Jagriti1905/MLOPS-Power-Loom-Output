"""
app/main.py — FastAPI prediction service for MP026 Power Loom Output.

Endpoints:
  GET  /health    → {"status": "ok"}
  POST /predict   → {"prediction": float}

Client rules (enforced at schema level, returning 422 on violation):
  - loom_age_yr <= 55
  - thread_count >= 20
"""
import logging
import json
import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ── JSON logging setup ──────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        return json.dumps(log_obj)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("mp026")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

# ── Load model ──────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
_model = joblib.load(MODEL_PATH)

FEATURES = ["loom_age_yr", "thread_count", "power_cuts_min",
            "weaver_exp_yr", "humidity_pct", "shift_hours", "fabric", "shed"]

# ── Pydantic schema ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    loom_age_yr:    float = Field(..., ge=0,  le=55,
                                  description="Loom age in years (max 55)")
    thread_count:   float = Field(..., ge=20,
                                  description="Threads per inch (min 20)")
    power_cuts_min: float = Field(..., ge=0)
    weaver_exp_yr:  float = Field(..., ge=0)
    humidity_pct:   float = Field(..., ge=0, le=100)
    shift_hours:    float = Field(..., gt=0)
    fabric:         str   = Field(...)
    shed:           str   = Field(...)


class PredictResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str


# ── App ─────────────────────────────────────────────────────────
app = FastAPI(title="MP026 Power Loom Output Predictor", version="1.0")


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = pd.DataFrame([{
        "loom_age_yr":    req.loom_age_yr,
        "thread_count":   req.thread_count,
        "power_cuts_min": req.power_cuts_min,
        "weaver_exp_yr":  req.weaver_exp_yr,
        "humidity_pct":   req.humidity_pct,
        "shift_hours":    req.shift_hours,
        "fabric":         req.fabric,
        "shed":           req.shed,
    }])
    prediction = float(_model.predict(row[FEATURES])[0])

    logger.info("prediction", extra={
        "fabric":         req.fabric,
        "shed":           req.shed,
        "shift_hours":    req.shift_hours,
        "prediction":     round(prediction, 4),
    })

    return {"prediction": round(prediction, 4)}
