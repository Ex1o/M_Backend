import os
import sys
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Fix Windows console Unicode encoding issue
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ==========================================
# CONFIG — load from .env if present
# ==========================================
load_dotenv()
API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
if not API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY not set. Add it to Backend/.env")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILENAME = os.path.join(BASE_DIR, "transcript_final.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac", "webm"}

client = ElevenLabs(api_key=API_KEY)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB
app.config["JSON_SORT_KEYS"] = False  # keep key order in JSON output
try:
    # Flask 2.3+/3.x JSON provider
    app.json.sort_keys = False
except Exception:
    pass

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# HELPERS
# ==========================================
def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def process_response_to_segments(api_data):
    """
    Group ElevenLabs word-level output into speaker segments.
    Returns the internal segment list AND the flat download format.

    Internal segment format (for frontend display):
      { text, start_time, end_time, speaker: { id, name }, words: [...] }

    Flat download format (for JSON download):
      { index, speaker_id, start_time, end_time, text }
    """
    input_words = api_data.get("words", [])
    language_code = api_data.get("language_code", "en")
    full_text = api_data.get("text", "")

    segments = []
    current_segment = None

    index_base = int(os.environ.get("INDEX_BASE", "0"))
    speaker_id_base = int(os.environ.get("SPEAKER_ID_BASE", "1316"))
    speaker_map = {}

    def map_speaker_id(raw_id):
        key = raw_id if raw_id else "unknown"
        if key not in speaker_map:
            speaker_map[key] = str(speaker_id_base + len(speaker_map))
        return speaker_map[key]

    for word in input_words:
        text = word.get("text", "")
        start = word.get("start", 0.0)
        end = word.get("end", 0.0)
        spk_id_raw = word.get("speaker_id", "unknown")
        spk_id = map_speaker_id(spk_id_raw)

        formatted_word = {"text": text, "start_time": start, "end_time": end}

        if current_segment is None or current_segment["speaker"]["id"] != spk_id:
            if current_segment:
                # Build clean segment text from word tokens
                current_segment["text"] = re.sub(
                    r"\s+",
                    " ",
                    "".join(w["text"] for w in current_segment["words"]),
                ).strip()
                segments.append(current_segment)
            current_segment = {
                "text": "",
                "start_time": start,
                "end_time": end,
                "speaker": {
                    "id": spk_id,
                    "name": f"Speaker {spk_id}",
                },
                "words": [formatted_word],
            }
        else:
            current_segment["words"].append(formatted_word)
            current_segment["end_time"] = end

    if current_segment:
        current_segment["text"] = re.sub(
            r"\s+",
            " ",
            "".join(w["text"] for w in current_segment["words"]),
        ).strip()
        segments.append(current_segment)

    # Build flat download format
    flat_segments = [
        {
            "index": index_base + i,
            "speaker_id": seg["speaker"]["id"],
            "start_time": round(float(seg["start_time"]), 3),
            "end_time": round(float(seg["end_time"]), 3),
            "text": seg["text"],
        }
        for i, seg in enumerate(segments)
    ]

    return {
        "text": full_text,
        "language_code": language_code,
        "segments": segments,           # rich format for display
        "flat_segments": flat_segments, # flat format for download
    }


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def index():
    return (
        "<h2>Audio Insights Hub Backend</h2>"
        "<p>Status: Running</p>"
        "<p>API endpoints: /api/transcribe, /api/translate, /api/health</p>"
    )

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({
            "success": False,
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        }), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        print(f"[INFO] Sending '{filename}' to ElevenLabs...")

        with open(file_path, "rb") as audio_file:
            transcription = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v2",
                diarize=True,
                tag_audio_events=True,
            )

        raw_data = {
            "text": transcription.text,
            "language_code": transcription.language_code,
            "words": [
                {
                    "text": w.text,
                    "start": w.start,
                    "end": w.end,
                    "speaker_id": w.speaker_id,
                }
                for w in transcription.words
            ],
        }

        print("[INFO] Processing segments...")
        result = process_response_to_segments(raw_data)

        # Save flat format to disk for debugging
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(result["flat_segments"], f, indent=2, ensure_ascii=False)

        print(f"[DONE] Segments: {len(result['segments'])}")

        return jsonify({"success": True, "data": result})

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"[ERROR] {e}\nTraceback:\n{tb_str}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": tb_str
        }), 500


@app.route("/api/translate", methods=["POST"])
def translate():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    target_language = request.form.get("target_language", "en")
    
    if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({
            "success": False,
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        }), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        print(f"[INFO] Translating '{filename}' to {target_language}...")

        with open(file_path, "rb") as audio_file:
            # Note: ElevenLabs Scribe v2 handles transcription.
            # For true translation, one might use a different model or process.
            # Here we follow the existing pattern but with fixed variable names.
            transcription = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v2",
                diarize=True,
                tag_audio_events=True,
            )

        raw_data = {
            "text": transcription.text,
            "language_code": transcription.language_code,
            "words": [
                {
                    "text": w.text,
                    "start": w.start,
                    "end": w.end,
                    "speaker_id": w.speaker_id,
                }
                for w in transcription.words
            ],
        }

        result = process_response_to_segments(raw_data)
        return jsonify({"success": True, "data": result})

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"[ERROR] {e}\nTraceback:\n{tb_str}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": tb_str
        }), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    # Default to 5000 to match Vite dev proxy (audio-insights-hub/vite.config.ts).
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
