# backend/Dockerfile
FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy both backend and frontend into the image
COPY backend/ ./backend
COPY frontend/ ./frontend

# Railway sets the PORT env variable; default to 8080
ENV PORT=8080
EXPOSE 8080

# Run the app from the backend folder
WORKDIR /app/backend
CMD ["bash", "start.sh"]
