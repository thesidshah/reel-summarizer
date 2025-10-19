#!/bin/bash

# Reel Summarizer Setup Script

set -e

echo "🎬 Reel Summarizer Setup"
echo "========================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
echo "✅ Python 3 found: $(python3 --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi
echo "✅ Node.js found: $(node --version)"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg is not installed. Please install FFmpeg."
    exit 1
fi
echo "✅ FFmpeg found"

echo ""
echo "Setting up backend..."
cd server

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "✅ Backend setup complete!"

# Setup environment file
if [ ! -f ".env" ]; then
    echo ""
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit server/.env and add your OPENAI_API_KEY"
fi

cd ..

echo ""
echo "Setting up frontend..."
cd frontend

# Install Node dependencies
echo "Installing Node.js dependencies..."
npm install --silent

# Setup environment file
if [ ! -f ".env.local" ]; then
    echo "📝 Creating .env.local file..."
    cp .env.example .env.local
fi

echo "✅ Frontend setup complete!"

cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit server/.env and add your OPENAI_API_KEY"
echo "2. Start the backend: cd server && source venv/bin/activate && python server_claude.py"
echo "3. Start the frontend (in a new terminal): cd frontend && npm run dev"
echo "4. Open http://localhost:3000 in your browser"
echo ""
