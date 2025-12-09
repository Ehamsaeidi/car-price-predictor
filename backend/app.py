from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json
from pathlib import Path

# Paths
MODEL_PATH = Path("model.joblib")
META_PATH = Path("model_meta.json")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Lazy-load so import doesn't crash
model = None
model_meta = None

def ensure_model():
    global model, model_meta
    if model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model file not found: model.joblib")
        model = joblib.load(MODEL_PATH)
    if model_meta is None and META_PATH.exists():
        try:
            model_meta = json.loads(META_PATH.read_text())
        except Exception:
            model_meta = {}

@app.get("/health")
def health():
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
        feats = (data.get("features") or {}).copy()

        # ---- coerce numbers
        def to_num(x):
            try:
                if x is None or x == "":
                    return None
                v = float(x)
                # keep ints as int
                return int(v) if v.is_integer() else v
            except Exception:
                return x

        # numeric fields if present
        for k in ["Year", "Engine Size", "Mileage"]:
            if k in feats:
                feats[k] = to_num(feats[k])

        # ---- derive features if missing
        # Age = current_year - Year
        if "Age" not in feats and isinstance(feats.get("Year"), (int, float)):
            from datetime import datetime
            feats["Age"] = max(0, datetime.utcnow().year - int(feats["Year"]))

        # Mileage_log = log1p(Mileage)
        if "Mileage_log" not in feats and isinstance(feats.get("Mileage"), (int, float)):
            import math
            feats["Mileage_log"] = math.log1p(max(0.0, float(feats["Mileage"])))

        # ---- align to model columns if provided in meta
        import pandas as pd
        cols = (model_meta or {}).get("feature_columns")
        X = pd.DataFrame([feats])

        if cols:
            for c in cols:
                if c not in X.columns:
                    X[c] = pd.NA
            X = X[cols]  # reorder & keep only expected columns

        # ---- predict
        yhat = model.predict(X)[0]
        return jsonify({"prediction": float(yhat)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # for local dev only; in production gunicorn runs: app:app
    app.run(host="0.0.0.0", port=5000)
