#!/usr/bin/env python3
"""
An improved Flask web server with streaming responses for reel summarization.

This server provides streaming updates using Server-Sent Events (SSE) to give
users real-time feedback during the summarization process.

Endpoints:
- /download: Downloads the video from a reel URL.
- /summarize: Creates a PDF summary with streaming progress updates (SSE).
- /summarize-sync: Non-streaming version (returns PDF directly).

Features:
- Real-time progress updates during:
  * Video download
  * Audio extraction
  * Transcription
  * AI summarization
  * Frame extraction
  * PDF generation
- Uses OpenAI GPT for summarization
- Graceful error handling with detailed messages

Requirements:
  pip install Flask openai reportlab python-dotenv

Environment Variables:
  OPENAI_API_KEY - Required for Whisper transcription and GPT summarization

To run:
  OPENAI_API_KEY="your-key" python server_claude.py

Example Usage:
  # Streaming with progress updates
  curl "http://127.0.0.1:8080/summarize?url=INSTAGRAM_REEL_URL"

  # Non-streaming (direct PDF download)
  curl "http://127.0.0.1:8080/summarize-sync?url=INSTAGRAM_REEL_URL" --output summary.pdf
"""
import os
import tempfile
import json
import time
import subprocess
import uuid
import shutil
from pathlib import Path
from typing import Generator, Optional, List, Dict

from flask import Flask, request, send_file, jsonify, Response, stream_with_context
from flask_cors import CORS


# Import functions from existing scripts
from download_reel import (
    extract_shortcode,
    fetch_metadata,
    download_video,
    ReelDownloadError,
)

app = Flask(__name__)
CORS(app)

# Directory to store temporary PDFs for download
TEMP_PDF_DIR = Path(tempfile.gettempdir()) / "reel_pdfs"
TEMP_PDF_DIR.mkdir(exist_ok=True)

def emit_progress(step: str, message: str, progress: int = 0) -> str:
    """Format a progress update as SSE data."""
    data = {
        "step": step,
        "message": message,
        "progress": progress,
        "timestamp": time.time()
    }
    return f"data: {json.dumps(data)}\n\n"


def extract_audio(video_path: Path, out_audio: Path) -> None:
    """Extract mono WAV audio track at 16kHz using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_audio),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def transcribe_with_openai(audio_path: Path) -> tuple[str, list]:
    """Transcribe audio using OpenAI Whisper API."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai package required")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    openai.api_key = api_key

    with audio_path.open("rb") as fh:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=fh,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        text = transcript.text
        segments = []
        if transcript.segments is not None:
            segments = [
                {'start': seg.start, 'end': seg.end, 'text': seg.text}
                for seg in transcript.segments
            ]
        return text, segments


def format_transcript_with_timestamps(segments: list) -> str:
    """Format transcript segments with timestamps."""
    lines = []
    for seg in segments:
        start_time = int(seg['start'])
        minutes = start_time // 60
        seconds = start_time % 60
        lines.append(f"[{minutes:02d}:{seconds:02d}] {seg['text'].strip()}")
    return "\n".join(lines)


def summarize_with_openai(transcript_with_ts: str, model: str = "gpt-3.5-turbo") -> tuple[str, List[dict]]:
    """
    Summarize transcript using OpenAI GPT API.
    Returns (recipe_text, key_moments).
    """
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai package required")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set for summarization")

    openai.api_key = api_key

    # Chunk if too large (simple char-based chunking)
    max_chars = 3000
    chunks = [transcript_with_ts[i:i+max_chars] for i in range(0, len(transcript_with_ts), max_chars)]

    all_key_moments: List[dict] = []
    summaries: List[str] = []

    for chunk in chunks:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant who creates recipes from video transcripts. "
                    "The user will provide a transcript with timestamps in the format [MM:SS]. "
                    "Your task is to:\n"
                    "1. Extract the recipe, listing ingredients first, then numbered instructions\n"
                    "2. Identify key moments - for each instruction step, provide the timestamp where that action happens\n\n"
                    "Format your response as:\n"
                    "RECIPE:\n"
                    "[ingredients and instructions here]\n\n"
                    "KEY_MOMENTS:\n"
                    "[MM:SS] Brief description of step 1\n"
                    "[MM:SS] Brief description of step 2\n"
                    "etc.\n\n"
                    "Be clear and concise. Ignore any conversational filler."
                ),
            },
            {"role": "user", "content": f"Here is the transcript:\n\n{chunk}"},
        ]
        resp = openai.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=500)
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned empty response")

        # Parse the response
        recipe_part, key_moments_part = parse_llm_response(content)
        summaries.append(recipe_part)
        all_key_moments.extend(key_moments_part)

    if len(summaries) == 1:
        return summaries[0], all_key_moments

    # If multiple chunks, combine and compress
    combined = "\n\n".join(summaries)
    combined_moments = "\n".join([f"[{m['timestamp']}] {m['description']}" for m in all_key_moments])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who creates recipes. "
                "Consolidate these recipe fragments into one coherent recipe with ingredients and numbered instructions. "
                "Also consolidate the key moments, removing duplicates."
            ),
        },
        {"role": "user", "content": f"RECIPES:\n{combined}\n\nKEY MOMENTS:\n{combined_moments}"},
    ]
    resp = openai.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=500)
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI returned empty response")

    recipe_part, key_moments_part = parse_llm_response(content)
    return recipe_part, key_moments_part


def parse_llm_response(response: str) -> tuple[str, List[dict]]:
    """Parse LLM response into recipe text and key moments."""
    import re

    parts = response.split("KEY_MOMENTS:")

    recipe = parts[0].replace("RECIPE:", "").strip()

    key_moments = []
    if len(parts) > 1:
        moments_text = parts[1].strip()
        for line in moments_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Match [MM:SS] or [M:SS] format
            match = re.match(r'\[(\d{1,2}):(\d{2})\]\s*(.+)', line)
            if match:
                minutes, seconds, description = match.groups()
                timestamp = int(minutes) * 60 + int(seconds)
                key_moments.append({
                    'timestamp': timestamp,
                    'description': description.strip()
                })

    return recipe, key_moments


def extract_key_frames_from_moments(video_path: Path, output_dir: Path, key_moments: List[dict]) -> List[tuple[Path, str]]:
    """Extract frames at timestamps identified by the LLM."""
    frames = []
    for i, moment in enumerate(key_moments):
        timestamp = moment['timestamp']
        frame_path = output_dir / f"key_frame_{i:03d}.jpg"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(frame_path)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            label = moment['description']
            frames.append((frame_path, label))
        except subprocess.CalledProcessError:
            continue

    return frames


def generate_pdf(recipe: str, frame_paths: List[tuple[Path, str]], output_path: Path, video_name: str) -> None:
    """Generate a PDF with recipe and video frames."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.colors import HexColor
    from reportlab.lib.utils import ImageReader

    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
        leftMargin=0.75*inch, rightMargin=0.75*inch
    )

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_LEFT
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=HexColor('#333333'),
        spaceAfter=8,
        leftIndent=0
    )

    # Title
    title = Paragraph(f"Video Recipe: {video_name}", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))

    # Recipe section
    summary_heading = Paragraph("Recipe", heading_style)
    elements.append(summary_heading)

    summary_lines = recipe.strip().split('\n')
    for line in summary_lines:
        if line.strip():
            if line.strip().startswith('- ') or line.strip().startswith('* '):
                line_text = '• ' + line.strip()[2:]
            elif line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                line_text = line.strip()
            else:
                line_text = line.strip()

            para = Paragraph(line_text, body_style)
            elements.append(para)

    elements.append(Spacer(1, 0.3*inch))

    # Frames section
    if frame_paths:
        frames_heading = Paragraph("Key Frames from Video", heading_style)
        elements.append(frames_heading)
        elements.append(Spacer(1, 0.1*inch))

        available_width = 6.5 * inch
        available_height = 4 * inch

        for idx, (frame_path, label) in enumerate(frame_paths):
            if frame_path.exists():
                img_reader = ImageReader(str(frame_path))
                img_width, img_height = img_reader.getSize()
                aspect = img_height / float(img_width)

                display_width = available_width
                display_height = display_width * aspect

                if display_height > available_height:
                    display_height = available_height
                    display_width = display_height / aspect

                img = Image(str(frame_path), width=display_width, height=display_height)
                elements.append(img)

                caption = Paragraph(f"<i>{label}</i>", body_style)
                elements.append(caption)
                elements.append(Spacer(1, 0.2*inch))

    doc.build(elements)


def process_reel_streaming(reel_url: str) -> Generator[str, None, None]:
    """
    Process a reel and yield progress updates as SSE.
    This is the core streaming logic.
    """
    try:
        yield emit_progress("init", "Starting reel processing...", 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Step 1: Extract shortcode
            yield emit_progress("metadata", "Extracting reel metadata...", 5)
            shortcode = extract_shortcode(reel_url)

            # Step 2: Fetch metadata
            yield emit_progress("metadata", "Fetching video information...", 10)
            metadata = fetch_metadata(shortcode)
            video_url = metadata.get("video_url")
            if not video_url:
                raise ReelDownloadError("Video URL not found in metadata")

            # Step 3: Download video
            yield emit_progress("download", f"Downloading video: {shortcode}.mp4", 15)
            video_path = tmp_path / f"{shortcode}.mp4"
            download_video(video_url, video_path)
            yield emit_progress("download", "Video downloaded successfully!", 25)

            # Step 4: Extract audio
            yield emit_progress("audio", "Extracting audio from video...", 30)
            audio_path = tmp_path / "audio.wav"
            extract_audio(video_path, audio_path)
            yield emit_progress("audio", "Audio extraction complete!", 35)

            # Step 5: Transcribe
            yield emit_progress("transcribe", "Transcribing audio (this may take a minute)...", 40)
            transcript, segments = transcribe_with_openai(audio_path)
            yield emit_progress("transcribe", f"Transcription complete! ({len(transcript)} characters)", 55)

            # Step 6: Summarize with OpenAI
            yield emit_progress("summarize", "Analyzing transcript with AI...", 60)
            transcript_with_ts = format_transcript_with_timestamps(segments)
            recipe, key_moments = summarize_with_openai(transcript_with_ts)
            yield emit_progress("summarize", f"Recipe extracted! Found {len(key_moments)} key moments", 75)

            # Step 7: Extract frames
            yield emit_progress("frames", f"Extracting {len(key_moments)} key frames from video...", 80)
            frame_paths = extract_key_frames_from_moments(video_path, tmp_path, key_moments)
            yield emit_progress("frames", f"Extracted {len(frame_paths)} frames successfully!", 85)

            # Step 8: Generate PDF
            yield emit_progress("pdf", "Generating PDF report...", 90)
            pdf_path = tmp_path / f"{shortcode}_summary.pdf"

            try:
                generate_pdf(recipe, frame_paths, pdf_path, shortcode)
                print(f"[DEBUG] PDF generated successfully at: {pdf_path}")
                print(f"[DEBUG] PDF file exists: {pdf_path.exists()}")
                print(f"[DEBUG] PDF file size: {pdf_path.stat().st_size if pdf_path.exists() else 'N/A'} bytes")
            except Exception as pdf_error:
                print(f"[ERROR] PDF generation failed: {pdf_error}")
                raise RuntimeError(f"PDF generation failed: {pdf_error}")

            yield emit_progress("pdf", "PDF generation complete!", 95)

            # Step 9: Save PDF to persistent temp location for download
            yield emit_progress("finalizing", "Preparing final result...", 98)

            try:
                # Generate unique ID for this PDF
                pdf_id = str(uuid.uuid4())
                persistent_pdf_path = TEMP_PDF_DIR / f"{pdf_id}.pdf"

                # Copy PDF to persistent location
                shutil.copy2(pdf_path, persistent_pdf_path)

                print(f"[DEBUG] PDF copied to persistent location: {persistent_pdf_path}")
                print(f"[DEBUG] PDF file size: {persistent_pdf_path.stat().st_size} bytes")
                print(f"[DEBUG] Recipe length: {len(recipe)} characters")
                print(f"[DEBUG] Key moments count: {len(key_moments)}")
            except Exception as save_error:
                print(f"[ERROR] PDF save failed: {save_error}")
                raise RuntimeError(f"PDF save failed: {save_error}")

            # Final result (WITHOUT base64 data - just send download URL)
            result = {
                "step": "complete",
                "message": "Processing complete!",
                "progress": 100,
                "pdf_id": pdf_id,
                "pdf_filename": f"{shortcode}_summary.pdf",
                "recipe": recipe,
                "key_moments": key_moments
            }

            print(f"[DEBUG] Final result structure:")
            print(f"  - step: {result['step']}")
            print(f"  - pdf_id: {result['pdf_id']}")
            print(f"  - pdf_filename: {result['pdf_filename']}")
            print(f"  - recipe length: {len(result['recipe'])}")
            print(f"  - key_moments count: {len(result['key_moments'])}")

            result_json = json.dumps(result)
            print(f"[DEBUG] JSON payload size: {len(result_json)} characters")

            yield f"data: {result_json}\n\n"
            print(f"[DEBUG] Final SSE message sent successfully")

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[ERROR] Exception in process_reel_streaming:")
        print(error_traceback)

        error_data = {
            "step": "error",
            "message": str(e),
            "error": True
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@app.route("/download", methods=["GET"])
def download_reel_endpoint():
    """
    Downloads an Instagram reel video.
    Expects a 'url' query parameter.
    """
    reel_url = request.args.get("url")
    if not reel_url:
        return jsonify({"error": "Missing 'url' query parameter."}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            shortcode = extract_shortcode(reel_url)
            metadata = fetch_metadata(shortcode)
            video_url = metadata.get("video_url")
            if not video_url:
                raise ReelDownloadError("Video URL not present in metadata response.")

            output_path = Path(tmpdir) / f"{shortcode}.mp4"
            download_video(video_url, output_path)

            return send_file(
                output_path, as_attachment=True, download_name=f"{shortcode}.mp4"
            )
    except ReelDownloadError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /download: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


@app.route("/summarize", methods=["GET"])
def summarize_reel_streaming():
    """
    Generates a PDF summary for an Instagram reel with streaming progress updates.
    Expects a 'url' query parameter.
    Returns Server-Sent Events (SSE) with progress updates.
    """
    reel_url = request.args.get("url")
    if not reel_url:
        return jsonify({"error": "Missing 'url' query parameter."}), 400

    return Response(
        stream_with_context(process_reel_streaming(reel_url)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.route("/summarize-sync", methods=["GET"])
def summarize_reel_sync():
    """
    Generates a PDF summary for an Instagram reel (non-streaming version).
    Expects a 'url' query parameter.
    Returns the PDF file directly.
    """
    reel_url = request.args.get("url")
    if not reel_url:
        return jsonify({"error": "Missing 'url' query parameter."}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shortcode = extract_shortcode(reel_url)
            video_path = tmp_path / f"{shortcode}.mp4"
            pdf_path = tmp_path / f"{shortcode}_summary.pdf"

            # Download video
            metadata = fetch_metadata(shortcode)
            video_url = metadata.get("video_url")
            if not video_url:
                raise ReelDownloadError("Video URL not present in metadata response.")
            download_video(video_url, video_path)

            # Extract audio
            audio_path = tmp_path / "audio.wav"
            extract_audio(video_path, audio_path)

            # Transcribe
            transcript, segments = transcribe_with_openai(audio_path)

            # Summarize with OpenAI
            transcript_with_ts = format_transcript_with_timestamps(segments)
            recipe, key_moments = summarize_with_openai(transcript_with_ts)

            # Extract frames
            frame_paths = extract_key_frames_from_moments(video_path, tmp_path, key_moments)

            # Generate PDF
            generate_pdf(recipe, frame_paths, pdf_path, shortcode)

            return send_file(pdf_path, as_attachment=True, download_name=f"{shortcode}_summary.pdf")

    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /summarize-sync: {e}")
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500


@app.route("/download-pdf/<pdf_id>", methods=["GET"])
def download_pdf(pdf_id):
    """
    Downloads a generated PDF by its ID.
    PDFs are stored temporarily after summarization.
    """
    try:
        # Validate pdf_id format (should be a UUID)
        try:
            uuid.UUID(pdf_id)
        except ValueError:
            return jsonify({"error": "Invalid PDF ID format"}), 400

        pdf_path = TEMP_PDF_DIR / f"{pdf_id}.pdf"

        if not pdf_path.exists():
            return jsonify({"error": "PDF not found. It may have expired."}), 404

        print(f"[DEBUG] Serving PDF: {pdf_path}")

        # Get the original filename from query param if provided
        filename = request.args.get("filename", "recipe_summary.pdf")

        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        app.logger.error(f"Error downloading PDF {pdf_id}: {e}")
        return jsonify({"error": f"Failed to download PDF: {e}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY"))
    })


if __name__ == "__main__":
    # Check required dependencies
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Transcription and summarization will fail.")
        print("Set it with: export OPENAI_API_KEY='your-key'")

    print("\nStarting streaming reel summarizer server...")
    print("Endpoints:")
    print("  - GET /download?url=<reel_url>")
    print("  - GET /summarize?url=<reel_url> (streaming SSE)")
    print("  - GET /summarize-sync?url=<reel_url> (direct PDF)")
    print("  - GET /download-pdf/<pdf_id>?filename=<name> (download generated PDF)")
    print("  - GET /health")
    print(f"\nTemporary PDF storage: {TEMP_PDF_DIR}")
    print()

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
