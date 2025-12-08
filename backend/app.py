# app.py
# Flask backend for car price prediction.
# Accepts raw text/numeric inputs from the frontend and relies on a
# saved sklearn Pipeline (with ColumnTransformer + OneHotEncoder)
# so that string features can be fed directly without manual label encoding.

from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Load model/pipeline once at startup.
# Tip: keep this a full sklearn Pipeline that includes preprocessing (e.g., OneHotEncoder)
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
pipe = joblib.load(MODEL_PATH)

api = Blueprint("api", __name__)

@api.get("/health")
def health():
    # Simple health endpoint for deployment checks
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}

@api.post("/predict")
def predict():
    """
    Expected JSON body:
    {
      "features": {
        "Brand": "...",
        "Model": "...",
        "Year": 2020,
        "Engine Size": 1.6,
        "Fuel Type": "...",
        "Transmission": "...",
        "Mileage": 45000,
        "Condition": "..."
      }
    }
    """
    data = request.get_json(force=True) or {}
    feats = data.get("features", data)

    # Light input validation & numeric coercion (keep strings as-is for OHE)
    try:
        # Coerce numeric fields if present
        if "Year" in feats:
            feats["Year"] = int(feats["Year"])
        if "Engine Size" in feats:
            feats["Engine Size"] = float(feats["Engine Size"])
        if "Mileage" in feats:
            feats["Mileage"] = float(feats["Mileage"])
    except Exception as e:
        return jsonify({"error": f"Invalid numeric field: {e}"}), 400

    # Convert to DataFrame so ColumnTransformer can access columns by name
    X = pd.DataFrame([feats])

    # Run prediction
    try:
        y = pipe.predict(X)[0]
    except Exception as e:
        # Most common issues: missing columns, wrong dtypes, pipeline mismatch
        return jsonify({"error": f"Prediction failed: {e}"}), 400

    # Return a plain number; format on the frontend if you want commas, currency, etc.
    return jsonify({"predicted_price": float(y)})

def create_app():
    app = Flask(__name__)
    # Enable CORS only if frontend is served from a different origin
    CORS(app)
    app.register_blueprint(api, url_prefix="/api")
    return app

app = create_app()

# -------- Serve frontend (index.html, app.js, style.css) --------
import os
from flask import send_from_directory

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

@app.get("/")
def index_html():
    # Serves the main HTML file
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.get("/<path:path>")
def static_files(path):
    # Serves static files (app.js, style.css, images, etc.)
    return send_from_directory(FRONTEND_DIR, path)
# ----------------------------------------------------------------


if __name__ == "__main__":
    # Respect PORT env var on platforms like Railway/Render/Heroku
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))


