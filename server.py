from __future__ import annotations

import json
import logging
import os
import re
import sys
import traceback

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ==========================================
# LOGGING  (stdout so Render captures it)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ==========================================
# CONFIG
# ==========================================
load_dotenv()

API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
if not API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY not set. Add it to .env or Render env vars.")

# 0-based offset applied to segment index in flat output
INDEX_BASE = int(os.environ.get("INDEX_BASE", "0"))

# Set DEBUG_SAVE_TRANSCRIPT=1 in env to persist flat transcript to disk (dev only)
DEBUG_SAVE_TRANSCRIPT = os.environ.get("DEBUG_SAVE_TRANSCRIPT", "0") == "1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac", "webm"}
MIME_MAP = {
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
    "m4a":  "audio/mp4",
    "ogg":  "audio/ogg",
    "flac": "audio/flac",
    "webm": "audio/webm",
}

# ==========================================
# APP
# ==========================================
client = ElevenLabs(api_key=API_KEY)

app = Flask(__name__)

# CORS — explicitly allow the Vercel frontend domain
CORS(
    app,
    resources={r"/api/*": {"origins": ["https://frontend-kappa-one-13.vercel.app", "http://localhost:3000", "http://localhost:8080"]}},
    supports_credentials=True,
)

# Upload limits and streaming
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.config["JSON_SORT_KEYS"] = False

# Werkzeug multipart form parser — stream to disk instead of buffering to memory
# This prevents timeouts on large file uploads
app.config["UPLOAD_FOLDER"] = "/tmp"

try:
    app.json.sort_keys = False  # Flask 2.3+ / 3.x JSON provider
except AttributeError:
    pass


# ==========================================
# HELPERS
# ==========================================

def allowed_file(filename: str) -> bool:
    """Return True if the file extension is an allowed audio type."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def extract_speaker_ids_from_filename(filename: str) -> list[str]:
        """
        Extract ordered speaker IDs from filename.

        Rules:
            - Remove file extension.
            - Remove trailing recording indicator such as " 1".
            - Treat both "-" and "_" as separators.
            - Keep only purely numeric tokens in their original order.

        Examples:
            "256_259_847102_847104_429 1.wav" -> ["256", "259", "847102", "847104", "429"]
            "256-259-847102_847104_429.wav"   -> ["256", "259", "847102", "847104", "429"]
        """
        base = os.path.splitext(filename)[0].strip()
        base = re.sub(r"\s+\d+$", "", base)
        parts = [part.strip() for part in re.split(r"[-_]", base) if part.strip()]
        return [part for part in parts if re.fullmatch(r"\d+", part)]


def _finalise_segment(seg: dict) -> None:
    """Collapse word tokens into clean segment text in-place."""
    seg["text"] = re.sub(
        r"\s+", " ", "".join(w["text"] for w in seg["words"])
    ).strip()


def process_response_to_segments(
    api_data: dict,
    speaker_ids: list[str] | None = None,
) -> dict:
    """
    Group ElevenLabs word-level transcription data into speaker segments.

    Returns:
        {
          "text":          str,   # full transcript
          "language_code": str,
          "segments":      list,  # rich format for frontend display
          "flat_segments": list,  # flat format for JSON download
        }
    """
    words         = api_data.get("words", [])
    language_code = api_data.get("language_code", "en")
    full_text     = api_data.get("text", "")
    speaker_ids   = speaker_ids or []

    segments: list[dict] = []
    current:  dict | None = None
    speaker_slots: dict[str, int] = {}

    for word in words:
        text    = word.get("text", "")
        start   = word.get("start", 0.0)
        end     = word.get("end", 0.0)
        raw_spk_value = word.get("speaker_id")
        raw_spk = str(raw_spk_value) if raw_spk_value not in (None, "") else "unknown"
        if raw_spk not in speaker_slots:
            speaker_slots[raw_spk] = len(speaker_slots)
        speaker_number = speaker_slots[raw_spk] + 1
        mapped_idx = speaker_slots[raw_spk]
        spk_id = speaker_ids[mapped_idx] if mapped_idx < len(speaker_ids) else raw_spk

        if current is None or current["_spk_key"] != raw_spk:
            if current:
                _finalise_segment(current)
                segments.append(current)
            current = {
                "_spk_key":   raw_spk,
                "text":       "",
                "start_time": start,
                "end_time":   end,
                "speaker":    {"id": spk_id, "name": f"Speaker {speaker_number}"},
                "words":      [{"text": text, "start_time": start, "end_time": end}],
            }
        else:
            current["words"].append({"text": text, "start_time": start, "end_time": end})
            current["end_time"] = end

    if current:
        _finalise_segment(current)
        segments.append(current)

    for seg in segments:
        seg.pop("_spk_key", None)

    flat_segments = [
        {
            "index":      INDEX_BASE + i,
            "speaker_id": str(seg["speaker"]["id"]),
            "start_time": round(float(seg["start_time"]), 3),
            "end_time":   round(float(seg["end_time"]),   3),
            "text":       seg["text"],
        }
        for i, seg in enumerate(segments)
    ]

    return {
        "text":          full_text,
        "language_code": language_code,
        "segments":      segments,
        "flat_segments": flat_segments,
    }


def enforce_filename_speaker_mapping(result: dict, speaker_ids: list[str]) -> dict:
    """Force speaker_id mapping using filename IDs in first-appearance order."""
    if not speaker_ids:
        return result

    speaker_slots: dict[str, int] = {}

    def map_speaker(raw_speaker: str) -> str:
        if raw_speaker not in speaker_slots:
            speaker_slots[raw_speaker] = len(speaker_slots)
        idx = speaker_slots[raw_speaker]
        return speaker_ids[idx] if idx < len(speaker_ids) else raw_speaker

    remapped_segments: list[dict] = []
    for seg in result.get("segments", []):
        speaker_obj = seg.get("speaker") or {}
        raw_speaker = str(speaker_obj.get("id") or speaker_obj.get("name") or "unknown")
        mapped_id = map_speaker(raw_speaker)

        seg_copy = dict(seg)
        seg_copy["speaker"] = {"id": mapped_id, "name": f"Speaker {mapped_id}"}
        remapped_segments.append(seg_copy)

    remapped_flat_segments: list[dict] = []
    for i, seg in enumerate(result.get("flat_segments", [])):
        raw_speaker = str(seg.get("speaker_id") or "unknown")
        mapped_id = map_speaker(raw_speaker)

        remapped_flat_segments.append(
            {
                "index": int(seg.get("index", INDEX_BASE + i)),
                "speaker_id": mapped_id,
                "start_time": round(float(seg.get("start_time", 0.0)), 3),
                "end_time": round(float(seg.get("end_time", 0.0)), 3),
                "text": str(seg.get("text", "")).strip(),
            }
        )

    result["segments"] = remapped_segments
    result["flat_segments"] = remapped_flat_segments
    return result


def _call_elevenlabs(stream, filename: str) -> dict:
    """
    Send the audio stream directly to ElevenLabs — no disk writes.

    The SDK accepts a (filename, file_obj, mime_type) tuple so we can
    pass the request stream without buffering it to disk first.
    """
    log.info("Starting ElevenLabs conversion for %s", filename)

    ext       = filename.rsplit(".", 1)[-1].lower()
    mime_type = MIME_MAP.get(ext, "application/octet-stream")

    try:
        transcription = client.speech_to_text.convert(
            file=(filename, stream, mime_type),
            model_id="scribe_v2",
            diarize=True,
            tag_audio_events=True,
        )
        log.info("ElevenLabs conversion completed for %s", filename)
    except Exception as e:
        log.error("ElevenLabs API error for %s: %s", filename, e)
        raise

    return {
        "text":          transcription.text,
        "language_code": transcription.language_code,
        "words": [
            {
                "text":       w.text,
                "start":      w.start,
                "end":        w.end,
                "speaker_id": w.speaker_id,
            }
            for w in transcription.words
        ],
    }


def _validate_upload(req) -> tuple[object, str, str, tuple | None]:
    """
    Validate the multipart file upload on an incoming request.

    Returns:
        (file, safe_filename, original_filename, None)               on success
        (None, None, None, (response, status))                       on validation failure
    """
    # Check Content-Length header before accessing multipart data
    content_length = req.content_length
    max_bytes = app.config["MAX_CONTENT_LENGTH"]

    if not content_length:
        return None, None, None, (jsonify({"success": False, "error": "Missing Content-Length header"}), 400)

    if content_length > max_bytes:
        return None, None, None, (jsonify({"success": False, "error": f"File too large. Maximum: {max_bytes // (1024 * 1024)} MB"}), 413)

    log.info("Parsing multipart form (size: %d bytes)", content_length)

    try:
        if "file" not in req.files:
            return None, None, None, (jsonify({"success": False, "error": "No file part"}), 400)

        file = req.files["file"]
        if not file or not file.filename:
            return None, None, None, (jsonify({"success": False, "error": "No file selected"}), 400)

        original_filename = file.filename.strip()
        filename = secure_filename(original_filename)

        if not allowed_file(original_filename):
            allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
            return None, None, None, (
                jsonify({"success": False, "error": f"Invalid file type. Allowed: {allowed}"}),
                415,
            )

        log.info("Multipart parsed successfully: %s", original_filename)
        return file, filename, original_filename, None

    except Exception as e:
        log.error("Multipart parsing failed: %s", e)
        return None, None, None, (jsonify({"success": False, "error": "Failed to parse upload"}), 400)


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(413)
def request_too_large(e):
    mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify({"success": False, "error": f"File too large. Maximum upload size is {mb} MB."}), 413


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def index():
    return (
        "<h2>Audio Insights Hub Backend</h2>"
        "<p>Status: Running</p>"
        "<p>Endpoints: "
        "POST /api/transcribe &nbsp;|&nbsp; "
        "POST /api/translate &nbsp;|&nbsp; "
        "GET /api/health</p>"
    )


@app.route("/api/transcribe", methods=["POST", "OPTIONS"])
def transcribe():
    """Transcribe audio file and return segment-based transcription with speaker IDs."""
    log.info("POST /api/transcribe — validating upload")

    file, filename, original_filename, err = _validate_upload(request)
    if err:
        return err

    log.info("Calling ElevenLabs for %s", filename)
    try:
        raw_data    = _call_elevenlabs(file.stream, filename)
        speaker_ids = extract_speaker_ids_from_filename(original_filename)
        result      = process_response_to_segments(raw_data, speaker_ids=speaker_ids)
        result      = enforce_filename_speaker_mapping(result, speaker_ids)

        if DEBUG_SAVE_TRANSCRIPT:
            out_path = os.path.join(BASE_DIR, "transcript_final.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result["flat_segments"], fh, indent=2, ensure_ascii=False)
            log.debug("Saved debug transcript -> %s", out_path)

        log.info("Transcription complete — %d segments", len(result["segments"]))
        return jsonify({"success": True, "data": result})

    except Exception as exc:
        log.error("Transcription failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/translate", methods=["POST", "OPTIONS"])
def translate():
    """Translate transcript using ElevenLabs text translation (if available) or return transcription."""
    log.info("POST /api/translate — validating upload")

    file, filename, original_filename, err = _validate_upload(request)
    if err:
        return err

    target_language = request.form.get("target_language", "en")
    log.info("Calling ElevenLabs for %s (target_language=%s)", filename, target_language)

    try:
        raw_data    = _call_elevenlabs(file.stream, filename)
        speaker_ids = extract_speaker_ids_from_filename(original_filename)
        result      = process_response_to_segments(raw_data, speaker_ids=speaker_ids)
        result      = enforce_filename_speaker_mapping(result, speaker_ids)

        log.info("Translation/transcription complete for %s", filename)
        return jsonify({"success": True, "data": result})

    except Exception as exc:
        log.error("Translation failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # debug=False is required when running under Gunicorn
    app.run(host="0.0.0.0", port=port, debug=False)
