"""
tests/test_api.py — pytest tests for the FastAPI service and data pipeline.

Tests do NOT use the CSV files — small inline data is built inside each test.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib
from fastapi.testclient import TestClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Ensure src is on path ───────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Build a tiny model and save it for the API tests ───────────
MODEL_PATH = "models/test_model.joblib"

NUM_FEATURES = ["loom_age_yr", "thread_count", "power_cuts_min",
                "weaver_exp_yr", "humidity_pct", "shift_hours"]
CAT_FEATURES = ["fabric", "shed"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

def _make_tiny_model():
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])
    pipe = Pipeline([("prep", prep), ("model", GradientBoostingRegressor(n_estimators=5, random_state=0))])
    # tiny synthetic data
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame({
        "loom_age_yr":    rng.integers(1, 30, n).astype(float),
        "thread_count":   rng.integers(20, 200, n).astype(float),
        "power_cuts_min": rng.uniform(0, 120, n),
        "weaver_exp_yr":  rng.integers(0, 30, n).astype(float),
        "humidity_pct":   rng.integers(30, 90, n).astype(float),
        "shift_hours":    rng.uniform(6, 12, n),
        "fabric":         rng.choice(["cotton", "silk", "polyester"], n),
        "shed":           rng.choice(["Shed 1", "Shed 2", "Shed 3"], n),
    })
    y = rng.uniform(20, 80, n)
    pipe.fit(X[ALL_FEATURES], y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a small temp model."""
    _make_tiny_model()
    os.environ["MODEL_PATH"] = MODEL_PATH
    # Import after setting env var
    import importlib
    import app.main as main_mod
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


# ── Test 1: health endpoint returns 200 and correct body ───────
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Test 2: valid predict request returns a numeric prediction ──
def test_predict_valid(client):
    payload = {
        "loom_age_yr":    10.0,
        "thread_count":   100.0,
        "power_cuts_min": 30.0,
        "weaver_exp_yr":  5.0,
        "humidity_pct":   60.0,
        "shift_hours":    8.0,
        "fabric":         "cotton",
        "shed":           "Shed 2",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


# ── Test 3: loom_age_yr > 55 triggers 422 (client rule 1) ──────
def test_predict_loom_age_too_high(client):
    payload = {
        "loom_age_yr":    56.0,          # > 55 → must be rejected
        "thread_count":   100.0,
        "power_cuts_min": 30.0,
        "weaver_exp_yr":  5.0,
        "humidity_pct":   60.0,
        "shift_hours":    8.0,
        "fabric":         "cotton",
        "shed":           "Shed 1",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── Test 4: thread_count < 20 triggers 422 (client rule 2) ─────
def test_predict_thread_count_too_low(client):
    payload = {
        "loom_age_yr":    10.0,
        "thread_count":   15.0,          # < 20 → must be rejected
        "power_cuts_min": 30.0,
        "weaver_exp_yr":  5.0,
        "humidity_pct":   60.0,
        "shift_hours":    8.0,
        "fabric":         "silk",
        "shed":           "Shed 3",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── Test 5: cleaning removes duplicates ────────────────────────
def test_clean_removes_duplicates():
    from mp026.clean import clean
    row = {
        "loom_age_yr": 10, "thread_count": 100, "power_cuts_min": 30.0,
        "weaver_exp_yr": 5, "humidity_pct": 60, "shift_hours": 8.0,
        "fabric": "cotton", "shed": "Shed 2",
        "metres_checked_at_folding": 45.0, "metres_per_shift": 45.0,
        "record_id": "R00001", "date": "2026-01-05",
    }
    df = pd.DataFrame([row, row])  # exact duplicate
    cleaned = clean(df)
    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) == 1


# ── Test 6: unit mixup fix — Shed 1 power_cuts_min divided by 60
def test_clean_unit_mixup():
    from mp026.clean import clean
    row = {
        "loom_age_yr": 10, "thread_count": 100, "power_cuts_min": 3600.0,
        "weaver_exp_yr": 5, "humidity_pct": 60, "shift_hours": 8.0,
        "fabric": "cotton", "shed": "Shed 1",
        "metres_checked_at_folding": 45.0, "metres_per_shift": 45.0,
        "record_id": "R00099", "date": "2026-01-05",
    }
    df = pd.DataFrame([row])
    cleaned = clean(df)
    # 3600 seconds / 60 = 60 minutes
    assert cleaned.loc[0, "power_cuts_min"] == pytest.approx(60.0)


# ── Test 7: sentinel fix — loom_age_yr 0 replaced ──────────────
def test_clean_preserves_zero_loom_age():
    from mp026.clean import clean
    rows = [
        {"loom_age_yr": 0,  "thread_count": 100, "power_cuts_min": 30.0,
         "weaver_exp_yr": 5, "humidity_pct": 60, "shift_hours": 8.0,
         "fabric": "cotton", "shed": "Shed 2",
         "metres_checked_at_folding": 45.0, "metres_per_shift": 45.0,
         "record_id": "R00001", "date": "2026-01-05"},
        {"loom_age_yr": 10, "thread_count": 80,  "power_cuts_min": 20.0,
         "weaver_exp_yr": 3, "humidity_pct": 50, "shift_hours": 7.0,
         "fabric": "silk",   "shed": "Shed 3",
         "metres_checked_at_folding": 40.0, "metres_per_shift": 40.0,
         "record_id": "R00002", "date": "2026-01-06"},
    ]
    df = pd.DataFrame(rows)
    cleaned = clean(df)
    assert (cleaned["loom_age_yr"] == 0).sum() == 1
