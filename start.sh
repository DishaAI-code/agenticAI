#!/bin/bash

# Start FastAPI in background
echo "Starting API server..."
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
sleep 3

# Start LiveKit Agent Worker
echo "Starting LiveKit Agent Worker..."
python app.py dev   # only agent, NOT api mode
