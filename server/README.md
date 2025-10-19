# Reel Summarizer - Backend Server

Python Flask server that processes Instagram reels and generates PDF summaries.

## Setup

### Prerequisites
- Python 3.8 or higher
- FFmpeg installed and in PATH
- OpenAI API key

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running the Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the server
python server_claude.py
```

The server will start on `http://0.0.0.0:8080`

## API Endpoints

### GET /summarize?url=<reel_url>
Processes a reel with streaming progress updates (Server-Sent Events)

### GET /summarize-sync?url=<reel_url>
Processes a reel and returns PDF directly (no streaming)

### GET /download?url=<reel_url>
Downloads only the video file

### GET /download-pdf/<pdf_id>?filename=<name>
Downloads a previously generated PDF

### GET /health
Health check endpoint

## Files

- `server_claude.py` - Main Flask application with SSE streaming support
- `download_reel.py` - Instagram reel download utilities
- `requirements.txt` - Python package dependencies
- `.env.example` - Example environment configuration

## Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `PORT` - Server port (default: 8080)
