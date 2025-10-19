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
from reel_processing import (
    extract_audio,
    extract_key_frames,
    extract_key_frames_from_moments,
    generate_pdf,
    summarize_transcript,
    transcribe_audio,
    translate_recipe_and_moments,
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


def normalize_transcription_mode(mode: Optional[str]) -> str:
    """Validate and normalize transcription mode query values."""
    valid_modes = {"auto", "openai", "local"}
    if not mode:
        return "auto"
    mode_lower = mode.lower()
    return mode_lower if mode_lower in valid_modes else "auto"


def _is_truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def parse_translation_params(args) -> tuple[bool, Optional[str]]:
    """Determine translation preferences from request arguments."""
    lang_raw = args.get("lang") or args.get("language")
    translate_raw = args.get("translate")

    translate = False
    target_lang: Optional[str] = None

    if lang_raw:
        lang = lang_raw.strip()
        if lang and lang.lower() not in {"none", "original"}:
            target_lang = lang
            translate = True

    if translate_raw:
        if _is_truthy(translate_raw):
            translate = True
            if not target_lang:
                target_lang = "English"
        elif translate_raw.lower() in {"0", "false", "no", "off"}:
            translate = False
            if not (lang_raw and target_lang):
                target_lang = None

    if translate and not target_lang:
        target_lang = "English"

    return translate, target_lang


def update_frame_labels_from_moments(
    frame_paths: List[tuple[Path, str]],
    key_moments: List[Dict[str, object]],
) -> List[tuple[Path, str]]:
    """Return frames with labels updated from key moment descriptions."""
    if not key_moments:
        return frame_paths

    updated: List[tuple[Path, str]] = []
    for idx, (frame_path, label) in enumerate(frame_paths):
        if idx < len(key_moments):
            description = str(key_moments[idx].get("description", "")).strip()
            updated.append((frame_path, description or label))
        else:
            updated.append((frame_path, label))
    return updated


def apply_translation_if_needed(
    recipe: str,
    key_moments: List[Dict[str, object]],
    frame_paths: List[tuple[Path, str]],
    translate: bool,
    target_lang: Optional[str],
) -> tuple[str, List[Dict[str, object]], List[tuple[Path, str]], Optional[str], str]:
    """
    Translate recipe and key moments when requested.

    Returns (final_recipe, final_key_moments, frame_paths_for_pdf, language, status)
    where status is one of {"not_requested", "applied", "unchanged", "failed"}.
    """
    if not translate or not recipe.strip():
        return recipe, key_moments, frame_paths, None, "not_requested"

    translation_language = target_lang or "English"
    try:
        translated_recipe, translated_moments = translate_recipe_and_moments(
            recipe,
            key_moments,
            translation_language,
        )
        translation_changed = (
            translated_recipe != recipe or translated_moments != key_moments
        )
        updated_frames = (
            update_frame_labels_from_moments(frame_paths, translated_moments)
            if key_moments
            else frame_paths
        )
        status = "applied" if translation_changed else "unchanged"
        return translated_recipe, translated_moments, updated_frames, translation_language, status
    except Exception as exc:
        print(f"[ERROR] Translation failed: {exc}")
        return recipe, key_moments, frame_paths, translation_language, "failed"


def process_reel_streaming(
    reel_url: str,
    *,
    transcription_mode: str = "auto",
    translate: bool = False,
    target_lang: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Process a reel and yield progress updates as SSE.
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
            yield emit_progress(
                "transcribe",
                "Transcribing audio (this may take a minute)...",
                40,
            )
            transcript, segments = transcribe_audio(audio_path, mode=transcription_mode)
            yield emit_progress(
                "transcribe",
                f"Transcription complete! ({len(transcript)} characters)",
                55,
            )

            # Step 6: Summarize
            yield emit_progress("summarize", "Analyzing transcript with AI...", 60)
            recipe, key_moments = summarize_transcript(transcript, segments)
            original_recipe = recipe
            original_key_moments = [dict(moment) for moment in key_moments]
            yield emit_progress(
                "summarize",
                f"Recipe extracted! Found {len(original_key_moments)} key moments",
                75,
            )

            # Step 7: Extract frames
            fallback_frame_count = max(len(original_key_moments), 5)
            if original_key_moments:
                yield emit_progress(
                    "frames",
                    f"Extracting {len(original_key_moments)} key frames from video...",
                    80,
                )
                frame_paths = extract_key_frames_from_moments(
                    video_path,
                    tmp_path,
                    original_key_moments,
                    fallback_frames=fallback_frame_count,
                )
            else:
                yield emit_progress(
                    "frames",
                    "Extracting representative frames from video...",
                    80,
                )
                frame_paths = extract_key_frames(
                    video_path,
                    tmp_path,
                    segments,
                    original_recipe,
                    num_frames=fallback_frame_count,
                )
            yield emit_progress("frames", f"Extracted {len(frame_paths)} frames successfully!", 85)

            translation_hint = (target_lang or "English") if translate else None
            if translate and original_recipe.strip():
                yield emit_progress(
                    "translate",
                    f"Translating recipe to {translation_hint}...",
                    78,
                )

            (
                final_recipe,
                final_key_moments,
                frame_paths_for_pdf,
                applied_language,
                translation_status,
            ) = apply_translation_if_needed(
                original_recipe,
                original_key_moments,
                frame_paths,
                translate,
                target_lang,
            )

            if translation_status != "not_requested":
                if translation_status == "applied":
                    message = f"Translation to {applied_language} complete!"
                elif translation_status == "unchanged":
                    message = "Translation skipped; already in requested language."
                elif translation_status == "failed":
                    message = "Translation unavailable, continuing with original language."
                else:
                    message = "Translation step finished."
                yield emit_progress("translate", message, 79)

            # Step 8: Generate PDF
            yield emit_progress("pdf", "Generating PDF report...", 90)
            pdf_path = tmp_path / f"{shortcode}_summary.pdf"

            try:
                generate_pdf(final_recipe, frame_paths_for_pdf, pdf_path, shortcode)
                print(f"[DEBUG] PDF generated successfully at: {pdf_path}")
                print(f"[DEBUG] PDF file exists: {pdf_path.exists()}")
                print(
                    f"[DEBUG] PDF file size: {pdf_path.stat().st_size if pdf_path.exists() else 'N/A'} bytes"
                )
            except Exception as pdf_error:
                print(f"[ERROR] PDF generation failed: {pdf_error}")
                raise RuntimeError(f"PDF generation failed: {pdf_error}")

            yield emit_progress("pdf", "PDF generation complete!", 95)

            # Step 9: Save PDF to persistent temp location for download
            yield emit_progress("finalizing", "Preparing final result...", 98)

            try:
                pdf_id = str(uuid.uuid4())
                persistent_pdf_path = TEMP_PDF_DIR / f"{pdf_id}.pdf"
                shutil.copy2(pdf_path, persistent_pdf_path)

                print(f"[DEBUG] PDF copied to persistent location: {persistent_pdf_path}")
                print(f"[DEBUG] PDF file size: {persistent_pdf_path.stat().st_size} bytes")
                print(f"[DEBUG] Recipe length: {len(final_recipe)} characters")
                print(f"[DEBUG] Key moments count: {len(final_key_moments)}")
            except Exception as save_error:
                print(f"[ERROR] PDF save failed: {save_error}")
                raise RuntimeError(f"PDF save failed: {save_error}")

            result = {
                "step": "complete",
                "message": "Processing complete!",
                "progress": 100,
                "pdf_id": pdf_id,
                "pdf_filename": f"{shortcode}_summary.pdf",
                "recipe": final_recipe,
                "key_moments": final_key_moments,
                "translation_language": applied_language,
                "translation_status": translation_status,
            }

            result_json = json.dumps(result)
            print(f"[DEBUG] Final SSE message sent successfully (payload size {len(result_json)})")

            yield f"data: {result_json}\n\n"

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        print(f"[ERROR] Exception in process_reel_streaming:\n{error_traceback}")

        error_data = {
            "step": "error",
            "message": str(e),
            "error": True,
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

    translate, target_lang = parse_translation_params(request.args)
    transcription_mode = normalize_transcription_mode(request.args.get("mode"))

    return Response(
        stream_with_context(
            process_reel_streaming(
                reel_url,
                transcription_mode=transcription_mode,
                translate=translate,
                target_lang=target_lang,
            )
        ),
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

    translate, target_lang = parse_translation_params(request.args)
    transcription_mode = normalize_transcription_mode(request.args.get("mode"))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shortcode = extract_shortcode(reel_url)
            video_path = tmp_path / f"{shortcode}.mp4"
            pdf_path = tmp_path / f"{shortcode}_summary.pdf"

            metadata = fetch_metadata(shortcode)
            video_url = metadata.get("video_url")
            if not video_url:
                raise ReelDownloadError("Video URL not present in metadata response.")
            download_video(video_url, video_path)

            audio_path = tmp_path / "audio.wav"
            extract_audio(video_path, audio_path)

            transcript, segments = transcribe_audio(audio_path, mode=transcription_mode)
            recipe, key_moments = summarize_transcript(transcript, segments)
            original_recipe = recipe
            original_key_moments = [dict(moment) for moment in key_moments]

            fallback_frame_count = max(len(original_key_moments), 5)
            if original_key_moments:
                frame_paths = extract_key_frames_from_moments(
                    video_path,
                    tmp_path,
                    original_key_moments,
                    fallback_frames=fallback_frame_count,
                )
            else:
                frame_paths = extract_key_frames(
                    video_path,
                    tmp_path,
                    segments,
                    original_recipe,
                    num_frames=fallback_frame_count,
                )

            (
                final_recipe,
                final_key_moments,
                frame_paths_for_pdf,
                applied_language,
                _translation_status,
            ) = apply_translation_if_needed(
                original_recipe,
                original_key_moments,
                frame_paths,
                translate,
                target_lang,
            )

            generate_pdf(final_recipe, frame_paths_for_pdf, pdf_path, shortcode)

            response = send_file(
                pdf_path,
                as_attachment=True,
                download_name=f"{shortcode}_summary.pdf",
            )
            if applied_language:
                response.headers["X-Translation-Language"] = applied_language
            return response

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
