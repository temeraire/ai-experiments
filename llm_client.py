"""
LLM client module for Local LLM Logger v3
Handles API calls to Ollama and Claude
"""
import subprocess
from typing import List, Dict

import requests

from config import OLLAMA_HOST, DEFAULT_MODEL, ANTHROPIC_API_KEY, HAS_CLAUDE


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


def call_llm(model: str, messages: List[Dict[str, str]]) -> str:
    """Universal LLM caller - routes to appropriate API"""
    if is_claude_model(model):
        return call_claude(model, messages)
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
