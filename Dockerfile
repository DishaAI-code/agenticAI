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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install packages in stages to reduce memory pressure
COPY requirements.txt .

# 1. First install small foundational packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    setuptools \
    wheel \
    numpy \
    scipy

# 2. Then install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Finally install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]