@app.post("/predict")
def predict():
    try:
        ensure_model()  # make sure model & meta are loaded
        data = request.get_json(force=True) or {}
        feats = (data.get("features") or {}).copy()

        # ---- Coerce + derive features ----
        # to numbers when possible
        def to_num(x):
            try:
                return float(x)
            except Exception:
                return x

        if "Year" in feats:
            feats["Year"] = to_num(feats["Year"])
        if "Engine Size" in feats:
            feats["Engine Size"] = to_num(feats["Engine Size"])
        if "Mileage" in feats:
            feats["Mileage"] = to_num(feats["Mileage"])

        # Derive Age if missing: current_year - Year
        if "Age" not in feats and "Year" in feats and isinstance(feats["Year"], (int, float)):
            from datetime import datetime
            current_year = datetime.utcnow().year
            feats["Age"] = max(0, current_year - int(feats["Year"]))

        # Derive Mileage_log if missing: log1p(Mileage)
        if "Mileage_log" not in feats and "Mileage" in feats and isinstance(feats["Mileage"], (int, float)):
            import math
            feats["Mileage_log"] = math.log1p(max(0.0, float(feats["Mileage"])))

        # ---- Reorder/complete columns as model expects ----
        import pandas as pd
        cols = (model_meta or {}).get("feature_columns")
        X = pd.DataFrame([feats])

        if cols:
            # add any missing expected columns with NA
            for c in cols:
                if c not in X.columns:
                    X[c] = pd.NA
            # keep only expected columns & order them
            X = X[cols]

        # ---- Predict ----
        yhat = model.predict(X)[0]
        return jsonify({"prediction": float(yhat)})

    except Exception as e:
        # return clear message to frontend
        return jsonify({"error": str(e)}), 400
