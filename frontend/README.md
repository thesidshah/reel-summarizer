# Reel Summarizer - Frontend

Modern Next.js web interface for the Reel Summarizer application.

## Setup

### Prerequisites
- Node.js 18 or higher
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local and set NEXT_PUBLIC_API_URL to your backend URL
```

### Running the Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
npm run build
npm start
```

## Features

- Real-time progress updates via Server-Sent Events (SSE)
- Beautiful, responsive UI with Tailwind CSS
- Live progress tracking with detailed logs
- Recipe preview and PDF download
- Error handling and user feedback

## Environment Variables

- `NEXT_PUBLIC_API_URL` - Backend API URL (default: http://127.0.0.1:8080)

## Tech Stack

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Lucide React (icons)
