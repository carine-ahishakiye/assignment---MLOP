# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create necessary directories
RUN mkdir -p models retrain_data

# Copy model files (if they exist locally)
COPY models/ models/

# Expose port
EXPOSE 5000

# Set environment variables
ENV MODEL_PATH=models/final_model.h5
ENV SCALER_PATH=models/scaler.pkl
ENV ENCODER_PATH=models/label_encoder.pkl
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health', timeout=5)"

# Run the application
CMD ["python", "app.py"]