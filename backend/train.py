#!/usr/bin/env python3
import argparse, json
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_SEED = 42

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Year" in df.columns:
        df["Age"] = datetime.now().year - pd.to_numeric(df["Year"], errors="coerce")
    if "Mileage" in df.columns:
        df["Mileage_log"] = np.log1p(pd.to_numeric(df["Mileage"], errors="coerce"))
    return df

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)])
    model = RandomForestRegressor(n_estimators=400, random_state=RANDOM_SEED, n_jobs=-1)
    return Pipeline([("pre", pre), ("model", model)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../data/car_data.csv"))
    ap.add_argument("--model-out", type=Path, default=Path("model.joblib"))
    ap.add_argument("--meta-out", type=Path, default=Path("model_meta.json"))
    ap.add_argument("--target", type=str, default="Price")
    ap.add_argument("--drop-cols", type=str, default="Car ID")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    for c in [c.strip() for c in args.drop_cols.split(",") if c.strip()]:
        if c in df.columns:
            df = df.drop(columns=c)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found.")

    y = df[args.target]
    X = df.drop(columns=[args.target])
    X = make_features(X)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=RANDOM_SEED)
    pipe = build_pipeline(X)
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    metrics = {
        "r2": float(r2_score(yte, pred)),
        "mae": float(mean_absolute_error(yte, pred)),
        "rmse": float(mean_squared_error(yte, pred, squared=False)),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "feature_columns": X.columns.tolist(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": RANDOM_SEED,
        "sklearn_version": __import__("sklearn").__version__,
    }
    joblib.dump(pipe, args.model_out)
    Path(args.meta_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
