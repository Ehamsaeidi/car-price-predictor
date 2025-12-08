from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import os

MODEL_PATH = Path("model.joblib")
META_PATH = Path("model_meta.json")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

if not MODEL_PATH.exists():
    raise RuntimeError("model.joblib not found. Train the model first (see backend/train.py).")

pipe = joblib.load(MODEL_PATH)

model_meta = {}
def _apply_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Year" in df.columns:
        df["Age"] = datetime.now().year - pd.to_numeric(df["Year"], errors="coerce")
    if "Mileage" in df.columns:
        df["Mileage_log"] = np.log1p(pd.to_numeric(df["Mileage"], errors="coerce"))
    return df

if META_PATH.exists():
    try:
        model_meta = json.loads(META_PATH.read_text())
    except Exception:
        model_meta = {}

def _as_row_df(payload: dict) -> pd.DataFrame:
    if "features" in payload and isinstance(payload["features"], dict):
        feats = payload["features"]
    else:
        feats = payload
    if not isinstance(feats, dict) or not feats:
        raise ValueError("No features provided. Send JSON with 'features': {...} or a flat mapping.")
    feats = {str(k): v for k, v in feats.items()}
    return pd.DataFrame([feats])

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
