#!/usr/bin/env python3
"""
Local LLM Logger v3 - Main Application
Modular version with clean separation of concerns

A Flask + Ollama + Claude + React/MUI application for logging
and tracking LLM conversations with comprehensive features.

Quick start:
1) Ensure Ollama is running with a model (e.g. `ollama pull qwen2.5:32b-instruct`)
2) `pip install flask requests markdown pypandoc python-dateutil`
3) `python main.py`
4) Open http://127.0.0.1:5005/

Features:
- Continuous conversations like ChatGPT
- Conversation-level tracking (ID, start/end times, duration, models used)
- Turn-level tracking (turn number, response times)
- Chat history display in UI
- Support for multiple models (Ollama + Claude)
- Export conversations (CSV, JSONL, MD, HTML, DOCX)
- File upload support (text, code, PDFs, notebooks)
- Context window management
- Token tracking
"""
from flask import Flask

from config import (
    OLLAMA_HOST, DEFAULT_MODEL, APP_PORT,
    LOGS_DIR, CONVERSATIONS_DIR
)
from models import initialize_csv_files
from routes import register_routes


def create_app() -> Flask:
    """Create and configure the Flask application"""
    app = Flask(__name__)

    # Initialize CSV log files
    initialize_csv_files()

    # Register all routes
    register_routes(app)

    return app


def main():
    """Main entry point"""
    app = create_app()

    print("=" * 60)
    print("Local LLM Logger v3 - Modular Edition")
    print("=" * 60)
    print(f"Server: http://127.0.0.1:{APP_PORT}")
    print(f"Ollama host: {OLLAMA_HOST}")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Logs directory: {LOGS_DIR}")
    print(f"Conversations directory: {CONVERSATIONS_DIR}")
    print("=" * 60)

    app.run(host="127.0.0.1", port=APP_PORT, debug=False)


if __name__ == "__main__":
    main()
