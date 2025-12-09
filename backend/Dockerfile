# Backend Dockerfile (production-ready)
FROM python:3.10-slim

WORKDIR /app

# Faster installs + smaller image
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install deps
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt && pip install gunicorn

# Copy code and model files
COPY . /app

# Prod defaults
ENV PORT=5000
EXPOSE 5000

# Run with Gunicorn (more robust than `python app.py`)
# --timeout 120 helps for slow cold starts
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app", "--timeout", "120"]
