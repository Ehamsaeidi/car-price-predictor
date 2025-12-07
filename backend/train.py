
import re, json, joblib, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA = Path("../data/car_data.csv")
MODEL_PATH = Path("model.joblib")
META_PATH = Path("model_meta.json")

df = pd.read_csv(DATA)

target = None
for c in ["Price","price","selling_price","Selling_Price","car_price"]:
    if c in df.columns: target = c; break
if target is None:
    for c in df.columns:
        if re.search("price", c, flags=re.IGNORECASE):
            target = c; break
if target is None:
    raise ValueError("Price column not found.")

y = pd.to_numeric(df[target], errors="coerce")
X = df.drop(columns=[target])

drop_cols = [c for c in X.columns if re.fullmatch(r"\s*car\s*id\s*", str(c), flags=re.IGNORECASE)]
if drop_cols:
    X = X.drop(columns=drop_cols)

num = X.select_dtypes(include=["number"]).columns.tolist()
cat = [c for c in X.columns if c not in num]

numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])

pre = ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)])
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
pipe = Pipeline([("prep", pre), ("model", model)])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)

metrics = {
    "r2": float(r2_score(yte, pred)),
    "mae": float(mean_absolute_error(yte, pred)),
    "rmse": float(mean_squared_error(yte, pred, squared=False)),
    "features": X.columns.tolist(),
    "note": "Car ID removed" if drop_cols else ""
}

joblib.dump(pipe, MODEL_PATH)
META_PATH.write_text(json.dumps(metrics, indent=2))
print("Model trained.", json.dumps(metrics, indent=2))
