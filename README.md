# Car Price Predictor — Complete Project

This is a minimal full‑stack project for car price prediction.

## Structure
- `backend/` — Flask API (`/predict`, `/meta`, `/health`) + training script
- `frontend/` — Static UI (vanilla JS) to send predictions
- `data/` — Example dataset `car_data.csv`
- `docker-compose.yml` — Run frontend + backend together

## Quick Start (Docker)
```bash
docker-compose up --build
```
Then open http://localhost:8080 and try a prediction.

## Quick Start (Local)
```bash
cd backend
pip install -r requirements.txt
python train.py --data ../data/car_data.csv --model-out model.joblib --meta-out model_meta.json
python app.py
```
Open `frontend/index.html` in a local static server (e.g., `python -m http.server 8080`) and it will call the backend at `http://localhost:5000`.

## API
- `POST /predict` — JSON body with either:
  - flat payload: `{"Brand": "...", "Model":"...", "Year": 2018, ...}`
  - or nested: `{"features": { ... }}`
  - Response: `{"prediction": <float>}`
- `GET /meta` — returns model metrics and info
- `GET /health` — returns `{ "status": "ok" }`

## Notes
- Train first (creates `backend/model.joblib`), then run API.
- Update `CORS` or proxy settings if deploying on a different origin.
