# Reel Summarizer

An AI-powered tool that automatically downloads Instagram reels, transcribes their audio, extracts recipes or key information, and generates a comprehensive PDF report with screenshots of key moments.

## Features

- **Instagram Reel Download**: Automatically downloads Instagram reel videos
- **Audio Transcription**: Uses OpenAI Whisper for accurate speech-to-text
- **AI-Powered Summarization**: Extracts recipes and key instructions using GPT
- **Key Frame Extraction**: Identifies and captures important moments from the video
- **PDF Report Generation**: Creates a professional PDF with recipe, instructions, and images
- **Real-time Progress Updates**: Streaming progress updates via Server-Sent Events (SSE)
- **Modern Web Interface**: Beautiful Next.js frontend with live progress tracking

## Architecture

The project consists of two main components:

### Backend (Python Flask)
- Flask API server with streaming SSE support
- OpenAI Whisper API for transcription
- OpenAI GPT for content summarization
- FFmpeg for video/audio processing
- ReportLab for PDF generation

### Frontend (Next.js)
- Modern React-based UI with TypeScript
- Real-time progress updates via EventSource (SSE)
- Tailwind CSS for styling
- Responsive design

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Node.js 18 or higher
- FFmpeg installed and available in PATH

### API Keys
- OpenAI API key for transcription and summarization

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/reel-summarizer.git
cd reel-summarizer
```

### 2. Backend Setup

```bash
cd server

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:8080" > .env.local
```

## Running the Application

### 1. Start the Backend Server

```bash
cd server
source venv/bin/activate  # Activate virtual environment
OPENAI_API_KEY='your-api-key' python server_claude.py
```

The server will start on `http://127.0.0.1:8080`

### 2. Start the Frontend

In a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Paste an Instagram reel URL (e.g., `https://www.instagram.com/reel/ABC123xyz/`)
3. Click "Process Reel"
4. Watch the real-time progress as the system:
   - Downloads the video
   - Extracts and transcribes audio
   - Analyzes content with AI
   - Extracts key frames
   - Generates a PDF report
5. Download your PDF report when processing completes

## API Endpoints

The backend server provides the following endpoints:

### GET /summarize?url=<reel_url>
Processes a reel with streaming progress updates (SSE)

**Parameters:**
- `url`: Instagram reel URL

**Returns:** Server-Sent Events stream with progress updates

### GET /summarize-sync?url=<reel_url>
Processes a reel and returns PDF directly (no streaming)

**Parameters:**
- `url`: Instagram reel URL

**Returns:** PDF file download

### GET /download?url=<reel_url>
Downloads only the video file

**Parameters:**
- `url`: Instagram reel URL

**Returns:** MP4 file download

### GET /download-pdf/<pdf_id>?filename=<name>
Downloads a previously generated PDF

**Parameters:**
- `pdf_id`: UUID of the generated PDF
- `filename`: (optional) Desired filename for download

**Returns:** PDF file download

### GET /health
Health check endpoint

**Returns:** JSON with server status

## Project Structure

```
reel-summarizer/
├── server/
│   ├── server.py      # Main Flask application
│   ├── download_reel.py       # Instagram download utilities
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── app/
│   │   ├── page.tsx          # Main UI component
│   │   ├── layout.tsx        # App layout
│   │   └── globals.css       # Global styles
│   ├── package.json          # Node.js dependencies
│   ├── tsconfig.json         # TypeScript config
│   ├── tailwind.config.ts    # Tailwind CSS config
│   └── next.config.mjs       # Next.js config
├── README.md
├── LICENSE
└── .gitignore
```

## Environment Variables

### Backend
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `PORT`: Server port (default: 8080)

### Frontend
- `NEXT_PUBLIC_API_URL`: Backend API URL (default: http://127.0.0.1:8080)

## How It Works

1. **Video Download**: The system fetches metadata from Instagram's GraphQL API and downloads the video file
2. **Audio Extraction**: FFmpeg extracts a mono WAV audio track at 16kHz
3. **Transcription**: OpenAI Whisper API transcribes the audio with timestamp information
4. **Analysis**: GPT analyzes the transcript to extract recipes, ingredients, and instructions
5. **Key Moments**: The AI identifies important timestamps in the video
6. **Frame Extraction**: FFmpeg captures screenshots at identified timestamps
7. **PDF Generation**: ReportLab creates a formatted PDF with the recipe and images

## Dependencies

### Backend
- Flask: Web framework
- flask-cors: CORS support
- openai: OpenAI API client
- reportlab: PDF generation
- python-dotenv: Environment variable management

### Frontend
- Next.js 14: React framework
- React 18: UI library
- TypeScript: Type safety
- Tailwind CSS: Styling
- lucide-react: Icons

## Troubleshooting

### FFmpeg not found
Make sure FFmpeg is installed and available in your PATH:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### OpenAI API Errors
- Verify your API key is set correctly
- Check your OpenAI account has sufficient credits
- Ensure you have access to the Whisper and GPT APIs

### CORS Issues
If you encounter CORS errors, make sure:
- The backend server is running
- The `NEXT_PUBLIC_API_URL` in frontend matches your backend URL
- flask-cors is properly installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for Whisper and GPT APIs
- Instagram for providing public reel access
- The open-source community for the various libraries used

## Disclaimer

This tool is for educational purposes only. Please respect Instagram's terms of service and only process content you have permission to use. Be mindful of API usage limits and costs associated with OpenAI services.
