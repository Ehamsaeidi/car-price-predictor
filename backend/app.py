from __future__ import annotations
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# -------------------------------------------------
# Paths (relative to this file)
# -------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
META_PATH = HERE / "model_meta.json"

# -------------------------------------------------
# Load model and metadata
# -------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

features: list[str] = []
if META_PATH.exists():
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        # expected format: {"features": ["year", "mileage", ...]}
        features = meta.get("features") or meta.get("columns") or []
    except Exception:
        features = []

app = Flask(__name__)


# -------------------------------------------------
# Build a 1-row DataFrame in correct order
# -------------------------------------------------
def build_row(payload: dict) -> pd.DataFrame:
    """
    Convert a dict of {feature: value} into a single-row DataFrame.
    Missing features = 0, extra keys ignored.
    """
    row = {}
    if features:
        for f in features:
            v = payload.get(f, 0)
            try:
                row[f] = float(v)
            except Exception:
                row[f] = 0.0
        columns = features
    else:
        # If no metadata, fall back to whatever keys are provided
        for k, v in payload.items():
            try:
                row[k] = float(v)
            except Exception:
                row[k] = 0.0
        columns = list(row.keys())

    df = pd.DataFrame([row], columns=columns)
    return df


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"service": "car-price-predictor", "status": "running"})


@app.get("/")
def index():
    if not features:
        return (
            "<h2>Car Price Pre
