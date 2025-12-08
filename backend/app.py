import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------------------
# App Setup
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# Load Model at Startup
# ----------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")

try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    pipe = None
    app.logger.error(f"Failed to load model from {MODEL_PATH}: {e}")

# ----------------------------
# Root Route (for Railway check)
# ----------------------------
@app.get("/")
def home():
    return jsonify({
        "service": "car-price-predictor",
        "status": "running"
    }), 200

# ----------------------------
# Health Check Route
# ----------------------------
@app.get("/health")
def health():
    status = "ok" if pipe is not None else "model_not_loaded"
    return jsonify({"status": status})

# ----------------------------
# Prediction Route
# ----------------------------
@app.post("/predict")
def predict():
    if pipe is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    try:
        data = request.get_json(force=True) or {}

        # Support both {"features": {...}} and flat {...}
        feats = data.get("features", data)

        # Convert to dataframe
        X = pd.DataFrame([feats])

        y = pipe.predict(X)[0]
        return jsonify({"predicted_price": float(y)})

    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 400

# ----------------------------
# Local Dev Server (Gunicorn is used in production)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
