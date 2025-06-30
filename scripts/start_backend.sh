#!/bin/bash

# Start the Iron Condor API Backend
echo "🚀 Starting Iron Condor API Backend..."

# Navigate to the backend directory
cd "$(dirname "$0")/../backend/api"

# Check if virtual environment exists
if [ ! -d "../../venv" ]; then
    echo "📦 Creating virtual environment..."
    cd ../..
    python3 -m venv venv
    cd backend/api
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../../venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r ../requirements.txt

# Start the API server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📖 API documentation available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server with auto-reload for development
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload