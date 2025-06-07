FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libportaudio2 \
    libsm6 \
    libxext6 \
    gcc \
    g++ \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies in two stages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]