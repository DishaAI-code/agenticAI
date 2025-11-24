#!/bin/bash
echo "Starting API server..."
uvicorn app:app --host 0.0.0.0 --port 8000 &
echo "Starting LiveKit Agent Worker..."
python app.py api
