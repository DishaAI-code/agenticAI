# Use official Python image
FROM python:3.10-slim

# Install system dependencies including Git
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
