from __future__ import annotations
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Paths
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
META_PATH = HERE / "model_meta.json"

# Load model & metadata
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

features: list[str] = []
if META_PATH.exists():
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        features = meta.get("features") or meta.get("columns") or []
    except Exception:
        features = []

app = Flask(__name__)


def build_row(payload: dict) -> pd.DataFrame:
    """
    Convert payload to a single-row dataframe.
    Keeps feature order from metadata.
    Unknown features ignored, missing ones filled with 0.
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
        for k, v in payload.items():
            try:
                row[k] = float(v)
            except Exception:
                row[k] = 0.0
        columns = list(row.keys())

    df = pd.DataFrame([row], columns=columns)
    return df


@app.get("/health")
def health():
    return jsonify({"service": "car-price-predictor", "status": "running"})


@app.get("/")
def index():
    if not features:
        return (
            "<h2>Car Price Predictor</h2>"
            "<p>No features found in model_meta.json.</p>"
            "<p>Send JSON POST to /predict</p>"
        )

    inputs_html = ""
    for f in features:
        inputs_html += (
            f'<div style="margin:8px 0;">'
            f'<label style="display:inline-block;width:120px">{f}</label>'
            f'<input name="{f}" type="number" step="any" required>'
            f"</div>"
        )

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Car Price Predictor</title>
      </head>
      <body style="font-family:Arial, sans-serif; max-width:720px; margin:40px auto;">
        <h2>Car Price Predictor</h2>
        <form method="POST" action="/predict">
          {inputs_html}
          <button type="submit" style="padding:8px 16px;">Predict</button>
        </form>
        <p style="margin-top:20px; color:#666">
          Or send JSON POST to <code>/predict</code>.
        </p>
      </body>
    </html>
    """
    return html


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = {k: v for k, v in request.form.items()}

    try:
        X = build_row(payload)
        y_pred = model.predict(X)
        predicted_value = float(y_pred[0])
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    if not request.is_json:
        return (
            f"<h3>Predicted price: {predicted_value:,.2f}</h3>"
            '<p><a href="/">Back</a></p>'
        )

    return jsonify({"ok": True, "prediction": predicted_value})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
