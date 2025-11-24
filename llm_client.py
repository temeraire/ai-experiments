"""
LLM client module for Local LLM Logger v3
Handles API calls to Ollama and Claude
"""
import subprocess
from typing import List, Dict

import requests

from config import OLLAMA_HOST, DEFAULT_MODEL, ANTHROPIC_API_KEY, HAS_CLAUDE, GOOGLE_API_KEY, HAS_GEMINI


def call_chat(model: str, messages: List[Dict[str, str]]) -> str:
    """Call Ollama chat API"""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def call_claude(model: str, messages: List[Dict[str, str]]) -> str:
    """Call Claude API"""
    if not HAS_CLAUDE:
        raise RuntimeError("Claude support not available. Install: pip install anthropic tiktoken")

    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Claude model names: claude-3-5-sonnet-20241022, claude-opus-4-20250514, etc.
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=messages
    )

    # Extract text content from response
    return response.content[0].text


def is_claude_model(model: str) -> bool:
    """Check if a model name is a Claude model"""
    return model.startswith("claude-")


def is_gemini_model(model: str) -> bool:
    """Check if a model name is a Gemini model"""
    return model.startswith("gemini-")


def call_gemini(model: str, messages: List[Dict[str, str]]) -> str:
    """Call Gemini API"""
    if not HAS_GEMINI:
        raise RuntimeError("Gemini support not available. Install: pip install google-generativeai")

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")

    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)

    # Create model instance
    gemini_model = genai.GenerativeModel(model)

    # Convert messages to Gemini format
    # Gemini uses a different format: list of content dicts with 'role' and 'parts'
    gemini_messages = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_messages.append({
            "role": role,
            "parts": [msg["content"]]
        })

    # Start chat and send messages
    chat = gemini_model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])

    # Send the last message and get response
    last_message = messages[-1]["content"] if messages else ""
    response = chat.send_message(last_message)

    return response.text


def call_llm(model: str, messages: List[Dict[str, str]]) -> str:
    """Universal LLM caller - routes to appropriate API"""
    if is_claude_model(model):
        return call_claude(model, messages)
    elif is_gemini_model(model):
        return call_gemini(model, messages)
    else:
        return call_chat(model, messages)


def get_claude_models() -> List[str]:
    """Get list of available Claude models"""
    if not HAS_CLAUDE or not ANTHROPIC_API_KEY:
        return []

    # Return latest Claude models (2025)
    return [
        "claude-sonnet-4-5-20250929",      # Latest: Best for coding & agents
        "claude-opus-4-1-20250805",        # Most capable: Complex reasoning
        "claude-sonnet-4-20250522",        # Balanced: Good performance
        "claude-haiku-4-5-20251015",       # Fastest: Cost-effective
        "claude-3-5-sonnet-20241022",      # Legacy: Still available
        "claude-3-5-haiku-20241022",       # Legacy: Still available
    ]


def get_gemini_models() -> List[str]:
    """Get list of available Gemini models"""
    if not HAS_GEMINI or not GOOGLE_API_KEY:
        return []

    # Return available Gemini models
    return [
        "gemini-2.5-flash",        # Latest Flash: Fast and efficient
        "gemini-2.5-pro",          # Latest Pro: Best quality
        "gemini-2.0-flash",        # Previous generation Flash
        "gemini-1.5-pro",          # Previous generation Pro
        "gemini-1.5-flash",        # Previous generation Flash
    ]


def get_ollama_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return [DEFAULT_MODEL]

        # Parse output: NAME    ID    SIZE    MODIFIED
        lines = result.stdout.strip().split('\n')
        models = []
        for line in lines[1:]:  # Skip header
            if line.strip():
                # First column is the model name
                model_name = line.split()[0]
                models.append(model_name)

        return models if models else [DEFAULT_MODEL]
    except Exception as e:
        print(f"Error getting models: {e}")
        return [DEFAULT_MODEL]
