"""
Configuration module for Local LLM Logger v3
Handles environment variables and application settings
"""
import os
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, environment variables must be set manually
    pass

# Check for optional dependencies
try:
    import pypandoc
    HAS_PYPANDOC = True
except Exception:
    HAS_PYPANDOC = False

try:
    import anthropic
    import tiktoken
    HAS_CLAUDE = True
except Exception:
    HAS_CLAUDE = False

# Ollama Configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b-instruct")

# Claude API Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Context Window Configuration
# How many recent turns to keep in context (0 = unlimited)
CONTEXT_WINDOW_SIZE = int(os.environ.get("CONTEXT_WINDOW_SIZE", 10))

# Server Configuration
APP_PORT = int(os.environ.get("PORT", 5005))

# Directory Configuration
BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = BASE_DIR / "sessions"
LOGS_DIR = BASE_DIR / "logs"
CONVERSATIONS_DIR = BASE_DIR / "conversations"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

# CSV log paths
CONVERSATIONS_CSV = LOGS_DIR / "conversations.csv"
TURNS_CSV = LOGS_DIR / "turns.csv"
