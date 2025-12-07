FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed by numpy/pandas/scikit)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Make shell script executable
RUN chmod +x start.sh

# Expose port (Render/Fly/Railway detect automatically)
EXPOSE 5000

# Start server
CMD ["./start.sh"]
