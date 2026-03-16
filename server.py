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

# Allow CORS origins to be restricted via env var in production
CORS(app, resources={r"/api/*": {"origins": os.environ.get("CORS_ORIGINS", "*")}})

# 100 MB — sufficient for typical call recordings, prevents memory pressure
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Preserve JSON key insertion order (Flask 2.x and 3.x)
app.config["JSON_SORT_KEYS"] = False
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
    Extract the first two numeric tokens from the filename as an ordered list.

    ElevenLabs diarization returns IDs as "speaker_0", "speaker_1", etc.
    The numeric suffix is used as a 0-based index into this list.

    Examples:
      "252_255_847201-847222-430.wav"     -> ["252", "255"]
      "252-255-847201-847222-430.wav"     -> ["252", "255"]
      "252-255-847201-847222-430_DIN.wav" -> ["252", "255"]
    """
    base = os.path.splitext(filename)[0]
    return re.findall(r"\d+", base)[:2]


def _map_speaker_id(raw_id: str | None, speaker_ids: list[str]) -> str:
    """
    Resolve an ElevenLabs diarization ID to a real speaker ID from the filename.

      "speaker_0" -> speaker_ids[0]
      "speaker_1" -> speaker_ids[1]

    Falls back to the raw value when no mapping entry exists.
    """
    raw = str(raw_id) if raw_id else "unknown"
    m = re.match(r"speaker_(\d+)$", raw)
    if m:
        idx = int(m.group(1))
        if idx < len(speaker_ids):
            return speaker_ids[idx]
    return raw


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

    for word in words:
        text    = word.get("text", "")
        start   = word.get("start", 0.0)
        end     = word.get("end", 0.0)
        raw_spk = word.get("speaker_id") or "unknown"
        spk_id  = _map_speaker_id(raw_spk, speaker_ids)

        if current is None or current["_spk_key"] != raw_spk:
            if current:
                _finalise_segment(current)
                segments.append(current)
            current = {
                "_spk_key":   raw_spk,
                "text":       "",
                "start_time": start,
                "end_time":   end,
                "speaker":    {"id": spk_id, "name": f"Speaker {spk_id}"},
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
            "speaker_id": seg["speaker"]["id"],
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


def _call_elevenlabs(stream, filename: str) -> dict:
    """
    Send the audio stream directly to ElevenLabs — no disk writes.

    The SDK accepts a (filename, file_obj, mime_type) tuple so we can
    pass the request stream without buffering it to disk first.
    """
    ext       = filename.rsplit(".", 1)[-1].lower()
    mime_type = MIME_MAP.get(ext, "application/octet-stream")

    transcription = client.speech_to_text.convert(
        file=(filename, stream, mime_type),
        model_id="scribe_v2",
        diarize=True,
        tag_audio_events=True,
    )

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


def _validate_upload(req) -> tuple[object, str, tuple | None]:
    """
    Validate the multipart file upload on an incoming request.

    Returns:
        (file, filename, None)           on success
        (None, None, (response, status)) on validation failure
    """
    if "file" not in req.files:
        return None, None, (jsonify({"success": False, "error": "No file part"}), 400)

    file = req.files["file"]
    if not file or not file.filename:
        return None, None, (jsonify({"success": False, "error": "No file selected"}), 400)

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        return None, None, (
            jsonify({"success": False, "error": f"Invalid file type. Allowed: {allowed}"}),
            415,
        )

    return file, filename, None


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


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    file, filename, err = _validate_upload(request)
    if err:
        return err

    log.info("Transcribing '%s'", filename)
    try:
        raw_data    = _call_elevenlabs(file.stream, filename)
        speaker_ids = extract_speaker_ids_from_filename(filename)
        result      = process_response_to_segments(raw_data, speaker_ids=speaker_ids)

        if DEBUG_SAVE_TRANSCRIPT:
            out_path = os.path.join(BASE_DIR, "transcript_final.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result["flat_segments"], fh, indent=2, ensure_ascii=False)
            log.debug("Saved debug transcript -> %s", out_path)

        log.info("Done — %d segments produced", len(result["segments"]))
        return jsonify({"success": True, "data": result})

    except Exception as exc:
        log.error("Transcription failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/translate", methods=["POST"])
def translate():
    file, filename, err = _validate_upload(request)
    if err:
        return err

    target_language = request.form.get("target_language", "en")
    log.info("Translating '%s' -> language=%s", filename, target_language)
    try:
        raw_data    = _call_elevenlabs(file.stream, filename)
        speaker_ids = extract_speaker_ids_from_filename(filename)
        result      = process_response_to_segments(raw_data, speaker_ids=speaker_ids)
        return jsonify({"success": True, "data": result})

    except Exception as exc:
        log.error("Translation failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # debug=False is required when running under Gunicorn
    app.run(host="0.0.0.0", port=port, debug=False)
