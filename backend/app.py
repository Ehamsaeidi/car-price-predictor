from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json
from pathlib import Path

MODEL_PATH = Path("model.joblib")
META_PATH = Path("model_meta.json")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Lazy-load to avoid crash on import time
model = None
model_meta = None

def ensure_model():
    global model, model_meta
    if model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model file not found: model.joblib")
        model = joblib.load(MODEL_PATH)
    if model_meta is None and META_PATH.exists():
        model_meta = json.loads(META_PATH.read_text())

@app.get("/health")
def health():
    # Light health (no heavy load)
    return jsonify({"status": "ok"})

@app.get("/meta")
def meta():
    try:
        ensure_model()
        return jsonify(model_meta or {"message": "No meta available"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/predict")
def predict():
    try:
        ensure_model()
        data = request.get_json(force=True) or {}
        feats = data.get("features") or {}
        import pandas as pd
        X = pd.DataFrame([feats])
        yhat = model.predict(X)[0]
        return jsonify({"prediction": float(yhat)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Local dev only (gunicorn uses import path app:app)
    app.run(host="0.0.0.0", port=5000)
