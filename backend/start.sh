#!/usr/bin/env bash
set -e

# Use Railway's PORT or default to 8080
export PORT="${PORT:-8080}"

# If gunicorn is installed, use it; otherwise fallback to python
if command -v gunicorn >/dev/null 2>&1; then
    exec gunicorn -w 1 -b 0.0.0.0:"$PORT" app:app
else
    exec python app.py
fi
