"""
app.py
Flask backend – REST API for HR Policy Chatbot.

Routes:
  GET  /              → serve UI
  POST /api/chat      → RAG pipeline chat
  POST /api/upload    → ingest new HR document
  GET  /api/stats     → vector store statistics
  POST /api/clear     → reset conversation history
"""

import os
import logging

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from rag_pipeline import HRChatbot
from ingest import DocumentIngester
import config

# ── Bootstrap ────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(config.DATA_DIR,   exist_ok=True)
os.makedirs(config.CHROMA_PATH, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

# ── Initialise singletons ────────────────────────────────────
logger.info("Initialising ingester …")
ingester = DocumentIngester()
# DO NOT auto-ingest on startup
# ingester.ingest_directory(config.DATA_DIR)
logger.info("Initialising chatbot …")
chatbot = HRChatbot()


# ── Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True)
    if not body or not body.get("message", "").strip():
        return jsonify({"error": "No message provided."}), 400

    result = chatbot.chat(body["message"])
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type '{ext}' is not supported. Use .txt, .pdf, or .docx."}), 415

    filename = secure_filename(file.filename)
    save_path = os.path.join(config.DATA_DIR, filename)
    file.save(save_path)
    logger.info("Saved uploaded file: %s", save_path)

    added = ingester.ingest_file(save_path)

    # Refresh chatbot's collection reference so it sees the new chunks
    chatbot.collection = chatbot.chroma.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return jsonify({
        "message":      f"✔ Successfully ingested '{filename}' — {added} new chunks added.",
        "chunks_added": added,
        "filename":     filename,
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(chatbot.stats())


@app.route("/api/clear", methods=["POST"])
def clear_history():
    chatbot.clear_history()
    return jsonify({"message": "Conversation history cleared."})


# ── Error handlers ───────────────────────────────────────────
@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large. Maximum size is {config.MAX_UPLOAD_MB} MB."}), 413


@app.errorhandler(500)
def server_error(exc):
    logger.error("Internal server error: %s", exc)
    return jsonify({"error": "Internal server error. Please try again."}), 500


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  HR Policy Chatbot is running!")
    print("  Open this link in your browser:")
    print("  http://localhost:5000")
    print("="*50 + "\n")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)