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


def cleanup_empty_conversations():
    """Remove empty conversation folders (folders with no turns)"""
    if not CONVERSATIONS_DIR.exists():
        return

    cleaned_count = 0
    for conv_dir in CONVERSATIONS_DIR.iterdir():
        if not conv_dir.is_dir() or conv_dir.name.startswith('.'):
            continue

        # Check if conversation has any content
        has_content = False

        # Check for conversation.json (saved conversations)
        if (conv_dir / "conversation.json").exists():
            has_content = True

        # Check for any turn directories
        if not has_content:
            turn_dirs = list(conv_dir.glob("turn_*"))
            if turn_dirs:
                has_content = True

        # If no content, remove the empty folder
        if not has_content:
            try:
                conv_dir.rmdir()
                cleaned_count += 1
                print(f"Cleaned empty conversation: {conv_dir.name}")
            except OSError:
                # Folder not empty (has hidden files or errors)
                pass

    if cleaned_count > 0:
        print(f"Cleaned up {cleaned_count} empty conversation folder(s)")

    return cleaned_count


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

    # Cleanup empty conversation folders on startup
    print("Checking for empty conversation folders...")
    cleanup_empty_conversations()
    print("=" * 60)

    app.run(host="127.0.0.1", port=APP_PORT, debug=False)


if __name__ == "__main__":
    main()
