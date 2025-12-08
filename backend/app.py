from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

MODEL_PATH = Path("model.joblib")
META_PATH = Path("model_meta.json")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Load model at startup (fail fast if missing) ---
if not MODEL_PATH.exists():
    raise RuntimeError("model.joblib not found. Train the model first (see backend/train.py).")
pipe = joblib.load(MODEL_PATH)

# --- Load meta (optional) ---
model_meta = {}
if META_PATH.exists():
    try:
        model_meta = json.loads(META_PATH.read_text())
    except Exception:
        model_meta = {}

# --- Feature engineering (same as training) ---
def _apply_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Year" in df.columns:
        df["Age"] = datetime.now().year - pd.to_numeric(df["Year"], errors="coerce")
    if "Mileage" in df.columns:
        df["Mileage_log"] = np.log1p(pd.to_numeric(df["Mileage"], errors="coerce"))
    return df

# --- Normalize incoming payload to single-row DataFrame ---
def _as_row_df(payload: dict) -> pd.DataFrame:
    if "features" in payload and isinstance(payload["features"], dict):
        feats = payload["features"]
    else:
        feats = payload
    if not isinstance(feats, dict) or not feats:
        raise ValueError("No features provided. Send JSON with 'features': {...} or a flat mapping.")
    feats = {str(k): v for k, v in feats.items()}
    return pd.DataFrame([feats])

# --- Routes ---
@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True, silent=False)
        X = _as_row_df(data)
        X = _apply_engineered_features(X)
        yhat = pipe.predict(X)[0]
        return jsonify({"prediction": float(yhat)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/meta")
def meta():
    return jsonify(model_meta or {"message": "No meta available"})

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

# --- Entrypoint ---
if __name__ == "__main__":
    # Railway provides the port via the PORT env var
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
