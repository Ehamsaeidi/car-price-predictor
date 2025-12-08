# app.py
# Flask backend for car price prediction.
# This app loads a sklearn Pipeline (with OneHotEncoder)
# so it can accept raw text and numeric inputs directly.

from flask import Flask, request, jsonify, Blueprint, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
pipe = joblib.load(MODEL_PATH)

api = Blueprint("api", __name__)

@api.get("/health")
def health():
    # Simple health-check endpoint for Railway deployments
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}

@api.post("/predict")
def predict():
    # Expect JSON data containing all feature fields
    data = request.get_json(force=True) or {}
    feats = data.get("features", data)

    # Basic numeric validation
    try:
        if "Year" in feats:
            feats["Year"] = int(feats["Year"])
        if "Engine Size" in feats:
            feats["Engine Size"] = float(feats["Engine Size"])
        if "Mileage" in feats:
            feats["Mileage"] = float(feats["Mileage"])
    except Exception as e:
        return jsonify({"error": f"Invalid numeric field: {e}"}), 400

    # Convert input into DataFrame for the sklearn Pipeline
    X = pd.DataFrame([feats])

    # Attempt prediction
    try:
        y = pipe.predict(X)[0]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

    return jsonify({"predicted_price": float(y)})

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register API under /api
    app.register_blueprint(api, url_prefix="/api")

    # Serve frontend (index.html, app.js, style.css)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

    @app.get("/")
    def index_page():
        # Serve main HTML file
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/<path:path>")
    def static_files(path):
        # Serve JS, CSS, and other static assets
        return send_from_directory(FRONTEND_DIR, path)

    return app

app = create_app()

if __name__ == "__main__":
    # Railway will pass PORT env
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
