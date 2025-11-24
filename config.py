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

try:
    import google.generativeai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# Ollama Configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b-instruct")

# Claude API Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Gemini API Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

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

# Model Pricing Configuration (per 1M tokens in USD)
# Format: {"input": price_per_1M_input_tokens, "output": price_per_1M_output_tokens}
MODEL_PRICING = {
    # Claude models (Anthropic)
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250522": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251015": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},

    # Gemini models (Google) - approximate pricing
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}

def is_paid_model(model_name: str) -> bool:
    """Check if a model is a paid API model (not local/Ollama)"""
    return model_name.startswith("claude-") or model_name.startswith("gemini-")

def get_model_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a model call in USD"""
    if model_name not in MODEL_PRICING:
        return 0.0

    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
