FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git ffmpeg libportaudio2 libportaudiocpp0 \
    portaudio19-dev build-essential libsndfile1 \
    libgl1 cmake curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

CMD ["bash", "start.sh"]
