"""Reusable utilities for reel transcription, summarization, translation, and reporting."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

Segment = Dict[str, Any]
KeyMoment = Dict[str, Any]
FrameInfo = Tuple[Path, str]


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def extract_audio(video_path: Path, out_audio: Path) -> None:
    """Extract a mono WAV audio track at 16kHz using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(out_audio),
    ]
    logger.debug("Running audio extraction: %s", cmd)
    subprocess.run(cmd, check=True, capture_output=True)


def extract_frames(video_path: Path, output_dir: Path, num_frames: int = 5) -> List[Path]:
    """Extract evenly-spaced frames from the video using ffmpeg."""
    if num_frames <= 0:
        return []

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    logger.debug("Probing video duration: %s", cmd)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())

    frames: List[Path] = []
    for i in range(num_frames):
        timestamp = (duration / (num_frames + 1)) * (i + 1)
        frame_path = output_dir / f"frame_{i:03d}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]
        logger.debug("Extracting evenly spaced frame %s: %s", frame_path, cmd)
        subprocess.run(cmd, check=True, capture_output=True)
        frames.append(frame_path)

    return frames


def extract_key_frames(
    video_path: Path,
    output_dir: Path,
    segments: Sequence[Segment],
    summary: str,
    num_frames: int = 5,
) -> List[FrameInfo]:
    """Extract key frames by matching summary sentences to transcript segments."""
    if not segments:
        frames = extract_frames(video_path, output_dir, num_frames)
        return [(fp, f"Frame {idx + 1}") for idx, fp in enumerate(frames)]

    summary_sentences = [sentence.strip() for sentence in re.split(r"\n", summary.strip()) if sentence.strip()]
    key_segments: List[Segment] = []
    for sentence in summary_sentences:
        target = sentence.lower().strip(".!?")
        for seg in segments:
            text = str(seg.get("text", "")).lower()
            if not text:
                continue
            if target and target in text:
                key_segments.append(seg)
                break

    if len(key_segments) < num_frames:
        sorted_segments = sorted(
            (seg for seg in segments if seg not in key_segments),
            key=lambda s: len(str(s.get("text", ""))),
            reverse=True,
        )
        for seg in sorted_segments:
            key_segments.append(seg)
            if len(key_segments) >= num_frames:
                break

    key_segments = key_segments[:num_frames]
    frames: List[FrameInfo] = []
    for idx, seg in enumerate(key_segments):
        timestamp = float(seg.get("start", 0))
        frame_path = output_dir / f"key_frame_{idx:03d}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]
        logger.debug("Extracting key frame %s at %s seconds: %s", frame_path, timestamp, cmd)
        subprocess.run(cmd, check=True, capture_output=True)
        label = str(seg.get("text", "")).strip() or f"Frame {idx + 1}"
        frames.append((frame_path, label))
    return frames


def extract_key_frames_from_moments(
    video_path: Path,
    output_dir: Path,
    key_moments: Sequence[KeyMoment],
    fallback_frames: int = 5,
) -> List[FrameInfo]:
    """Extract frames at timestamps identified by the LLM, with fallback to even spacing."""
    if not key_moments:
        logger.info("No key moments provided; falling back to evenly spaced frames.")
        frames = extract_frames(video_path, output_dir, num_frames=fallback_frames)
        return [(fp, f"Frame {idx + 1}") for idx, fp in enumerate(frames)]

    frames: List[FrameInfo] = []
    for idx, moment in enumerate(key_moments):
        timestamp = float(moment.get("timestamp", 0))
        frame_path = output_dir / f"key_frame_{idx:03d}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]
        logger.debug("Extracting frame for key moment %s at %s seconds: %s", idx, timestamp, cmd)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            label = str(moment.get("description", "")).strip() or f"Moment {idx + 1}"
            frames.append((frame_path, label))
        except subprocess.CalledProcessError as exc:
            logger.warning("Failed to extract frame at %s seconds: %s", timestamp, exc)
            continue

    if not frames:
        logger.warning("All key moment frame extractions failed; using evenly spaced frames instead.")
        frames = extract_frames(video_path, output_dir, num_frames=fallback_frames)
        return [(fp, f"Frame {idx + 1}") for idx, fp in enumerate(frames)]

    return frames


def transcribe_with_openai(audio_path: Path) -> Tuple[str, List[Segment]]:
    """Transcribe audio using OpenAI Whisper API."""
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai package required for OpenAI transcription") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    openai.api_key = api_key

    with audio_path.open("rb") as fh:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=fh,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        text = transcript.text
        segments: List[Segment] = []
        if transcript.segments is not None:
            segments = [
                {"start": seg.start, "end": seg.end, "text": seg.text} for seg in transcript.segments
            ]
        return text, segments


def transcribe_with_local(audio_path: Path) -> Tuple[str, List[Segment]]:
    """Transcribe audio using local whisper backends."""
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments_iter, _info = model.transcribe(str(audio_path))
        parts: List[str] = []
        segment_list: List[Segment] = []
        for segment in segments_iter:
            parts.append(segment.text)
            segment_list.append({"start": segment.start, "end": segment.end, "text": segment.text})
        return "\n".join(parts), segment_list
    except Exception:
        logger.debug("faster-whisper transcription unavailable; falling back to whisper.")

    try:
        import whisper  # type: ignore

        model = whisper.load_model("small")
        result = model.transcribe(str(audio_path))
        text = result.get("text", "")
        segments_raw = result.get("segments", [])
        segments = [
            {"start": seg.get("start"), "end": seg.get("end"), "text": seg.get("text", "")}
            for seg in segments_raw
        ]
        return text, segments
    except Exception as exc:  # pragma: no cover - local transcription is optional
        raise RuntimeError(
            "No local transcription backend available (requires faster-whisper or whisper)."
        ) from exc


def transcribe_audio(audio_path: Path, mode: str = "auto") -> Tuple[str, List[Segment]]:
    """Transcribe audio using OpenAI, local whisper, or automatic selection."""
    transcript = ""
    segments: List[Segment] = []
    last_error: Optional[Exception] = None

    selected_mode = mode
    if selected_mode == "auto":
        selected_mode = "openai" if os.environ.get("OPENAI_API_KEY") else "local"

    if selected_mode == "openai":
        try:
            transcript, segments = transcribe_with_openai(audio_path)
        except Exception as exc:
            last_error = exc
            logger.warning("OpenAI transcription failed: %s", exc)

    if not transcript and selected_mode in {"local", "openai"}:
        try:
            transcript, segments = transcribe_with_local(audio_path)
        except Exception as exc:
            last_error = exc
            logger.error("Local transcription failed: %s", exc)

    if not transcript:
        raise RuntimeError(f"Transcription failed: {last_error}")

    return transcript, segments


def format_transcript_with_timestamps(segments: Sequence[Segment]) -> str:
    """Format transcript segments to a string with [MM:SS] timestamps."""
    lines: List[str] = []
    for seg in segments:
        start_time = int(float(seg.get("start", 0)))
        minutes = start_time // 60
        seconds = start_time % 60
        text = str(seg.get("text", "")).strip()
        if text:
            lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")
    return "\n".join(lines)


def summarize_with_openai(transcript_with_ts: str, model: str = "gpt-3.5-turbo") -> Tuple[str, List[KeyMoment]]:
    """Summarize transcript using OpenAI GPT API, returning recipe text and key moments."""
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai package required for OpenAI summarization") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set for summarization")
    openai.api_key = api_key

    max_chars = 3000
    chunks = [transcript_with_ts[i : i + max_chars] for i in range(0, len(transcript_with_ts), max_chars)]

    all_key_moments: List[KeyMoment] = []
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
        resp = openai.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=500  # type: ignore[arg-type]
        )
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned empty response during summarization")
        recipe_part, key_moments_part = parse_llm_response(content)
        summaries.append(recipe_part)
        all_key_moments.extend(key_moments_part)

    if len(summaries) == 1:
        return summaries[0], all_key_moments

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
    resp = openai.chat.completions.create(
        model=model, messages=messages, temperature=0.2, max_tokens=500  # type: ignore[arg-type]
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI returned empty response during consolidation")
    recipe_part, key_moments_part = parse_llm_response(content)
    return recipe_part, key_moments_part


def parse_llm_response(response: str) -> Tuple[str, List[KeyMoment]]:
    """Parse LLM response into recipe text and key moments."""
    parts = response.split("KEY_MOMENTS:")

    recipe = parts[0].replace("RECIPE:", "").strip()

    key_moments: List[KeyMoment] = []
    if len(parts) > 1:
        moments_text = parts[1].strip()
        for line in moments_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"\[(\d{1,2}):(\d{2})\]\s*(.+)", line)
            if match:
                minutes, seconds, description = match.groups()
                timestamp = int(minutes) * 60 + int(seconds)
                key_moments.append({"timestamp": timestamp, "description": description.strip()})

    return recipe, key_moments


def summarize_locally(text: str, bullets: int = 5) -> str:
    """Very small extractive summarizer: score sentences by word frequency."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    words = re.findall(r"\w+", text.lower())
    if not words or not sentences:
        return "(No transcript available to summarize)"

    freq = Counter(words)
    scored_sentences: List[Tuple[float, str]] = []
    for sentence in sentences:
        sentence_words = re.findall(r"\w+", sentence.lower())
        if not sentence_words:
            continue
        score = sum(freq[word] for word in sentence_words)
        scored_sentences.append((score / len(sentence_words), sentence))

    scored_sentences.sort(reverse=True, key=lambda item: item[0])
    top_sentences = [sentence for _, sentence in scored_sentences[:bullets]]
    return "\n".join([f"- {sentence.strip()}" for sentence in top_sentences])


def summarize_transcript(
    transcript: str,
    segments: Sequence[Segment],
    prefer_openai: bool = True,
    model: str = "gpt-3.5-turbo",
) -> Tuple[str, List[KeyMoment]]:
    """Summarize transcript using OpenAI when available, otherwise fall back to local summary."""
    recipe = ""
    key_moments: List[KeyMoment] = []

    if prefer_openai and os.environ.get("OPENAI_API_KEY"):
        try:
            transcript_with_ts = format_transcript_with_timestamps(segments)
            recipe, key_moments = summarize_with_openai(transcript_with_ts, model=model)
        except Exception as exc:
            logger.warning("OpenAI summarization failed; falling back to local summarizer: %s", exc)

    if not recipe:
        recipe = summarize_locally(transcript)

    return recipe, key_moments


def translate_summary(summary: str, target_lang: str = "English") -> str:
    """Translate text to the target language using OpenAI."""
    if not summary.strip():
        return summary

    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai package required for translation") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set for translation")
    openai.api_key = api_key

    messages = [
        {"role": "system", "content": f"You are a translator. Translate the following text to {target_lang}."},
        {"role": "user", "content": summary},
    ]
    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.2, max_tokens=500  # type: ignore[arg-type]
        )
        content = resp.choices[0].message.content
        return content.strip() if content else summary
    except Exception as exc:
        logger.warning("Translation failed; returning original text: %s", exc)
        return summary


def translate_recipe_and_moments(
    recipe: str, key_moments: Sequence[KeyMoment], target_lang: str
) -> Tuple[str, List[KeyMoment]]:
    """Translate recipe and key moment descriptions to the desired language."""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.info("Translation requested but OPENAI_API_KEY is not set; returning original text.")
        return recipe, list(key_moments)

    translated_recipe = translate_summary(recipe, target_lang=target_lang)
    translated_moments: List[KeyMoment] = []
    for moment in key_moments:
        description = str(moment.get("description", ""))
        translated_description = translate_summary(description, target_lang=target_lang) if description else description
        translated_moments.append(
            {"timestamp": moment.get("timestamp"), "description": translated_description}
        )
    return translated_recipe, translated_moments


def generate_pdf(recipe: str, frame_paths: Sequence[FrameInfo], output_path: Path, video_name: str) -> None:
    """Generate a PDF with recipe text and labeled video frames."""
    try:
        from reportlab.lib.colors import HexColor  # type: ignore
        from reportlab.lib.enums import TA_LEFT  # type: ignore
        from reportlab.lib.pagesizes import letter  # type: ignore
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
        from reportlab.lib.units import inch  # type: ignore
        from reportlab.lib.utils import ImageReader  # type: ignore
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "reportlab package is required for PDF generation. Install with: pip install reportlab"
        ) from exc

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    elements: List[Any] = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=HexColor("#1a1a1a"),
        spaceAfter=30,
        alignment=TA_LEFT,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=HexColor("#2c3e50"),
        spaceAfter=12,
        spaceBefore=12,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=11,
        textColor=HexColor("#333333"),
        spaceAfter=8,
        leftIndent=0,
    )

    title = Paragraph(f"Video Recipe: {video_name}", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2 * inch))

    recipe_heading = Paragraph("Recipe", heading_style)
    elements.append(recipe_heading)

    for line in recipe.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            line_text = "• " + stripped[2:]
        else:
            line_text = stripped
        para = Paragraph(line_text, body_style)
        elements.append(para)

    elements.append(Spacer(1, 0.3 * inch))

    if frame_paths:
        frames_heading = Paragraph("Key Frames from Video", heading_style)
        elements.append(frames_heading)
        elements.append(Spacer(1, 0.1 * inch))

        max_width = 6.5 * inch
        max_height = 4 * inch

        for frame_path, label in frame_paths:
            if not frame_path.exists():
                continue
            reader = ImageReader(str(frame_path))
            img_width, img_height = reader.getSize()
            aspect = img_height / float(img_width) if img_width else 1

            display_width = max_width
            display_height = display_width * aspect
            if display_height > max_height:
                display_height = max_height
                display_width = display_height / aspect if aspect else max_width

            elements.append(Image(str(frame_path), width=display_width, height=display_height))
            caption = Paragraph(f"<i>{label}</i>", body_style)
            elements.append(caption)
            elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    logger.debug("PDF generated at %s", output_path)


__all__ = [
    "check_ffmpeg",
    "extract_audio",
    "extract_frames",
    "extract_key_frames",
    "extract_key_frames_from_moments",
    "transcribe_audio",
    "transcribe_with_openai",
    "transcribe_with_local",
    "format_transcript_with_timestamps",
    "summarize_with_openai",
    "summarize_locally",
    "summarize_transcript",
    "parse_llm_response",
    "translate_summary",
    "translate_recipe_and_moments",
    "generate_pdf",
]
