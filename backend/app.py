import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- App setup ---
app = Flask(__name__)
CORS(app)

# --- Load model once at startup ---
MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    pipe = None
    app.logger.error(f"Failed to load model from {MODEL_PATH}: {e}")

# --- Routes ---
@app.get("/health")
def health():
    status = "ok" if pipe is not None else "model_not_loaded"
    return jsonify({"status": status})

@app.post("/predict")
def predict():
    if pipe is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    try:
        data = request.get_json(force=True) or {}

        # Support both {"features": {...}} and flat {...}
        feats = data.get("features", data)

        # Convert to single-row DataFrame (keeps column names with spaces)
        X = pd.DataFrame([feats])

        y = pipe.predict(X)[0]
        return jsonify({"predicted_price": float(y)})
    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 400

# --- Local dev server (production uses gunicorn) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
