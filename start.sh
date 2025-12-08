#!/bin/bash

echo "Starting FastAPI server..."
uvicorn app:app --host 0.0.0.0 --port 8000 &

sleep 3

echo "Starting LiveKit Agent Worker..."
python app.py dev

wait
