# backend/Dockerfile
FROM python:3.11-slim

# Create app dir
WORKDIR /app

# Install Python deps
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy BOTH backend and frontend into the image
COPY backend/ ./backend
COPY frontend/ ./frontend

# Railway provides PORT; we default to 8080
ENV PORT=8080
EXPOSE 8080

# Run from backend folder
WORKDIR /app/backend
CMD ["bash", "start.sh"]
