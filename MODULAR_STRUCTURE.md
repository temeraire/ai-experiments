# Modular Structure Documentation

## Overview

The Local LLM Logger v3 has been refactored from a single 1595-line file into a modular architecture with clean separation of concerns. This makes the codebase easier to maintain, test, and extend.

## File Structure

```
ai_experiments/
├── main.py                          # Application entry point (67 lines)
├── config.py                        # Configuration and environment variables (58 lines)
├── models.py                        # Data models (Conversation class) (230 lines)
├── llm_client.py                    # LLM API clients (Ollama & Claude) (90 lines)
├── storage.py                       # File operations and artifact saving (140 lines)
├── routes.py                        # Flask API endpoints (450 lines)
├── frontend.py                      # HTML/React UI generation (580 lines)
└── local_llm_logger_v3_continuous_chat.py  # Original monolithic version (1595 lines)
```

## Module Responsibilities

### 1. `main.py` - Application Entry Point
**Purpose**: Bootstrap the application and start the Flask server

**Key Functions**:
- `create_app()` - Initialize Flask app and register routes
- `main()` - Entry point that starts the server

**Usage**:
```bash
python main.py
```

---

### 2. `config.py` - Configuration Management
**Purpose**: Centralize all configuration and environment variables

**Key Variables**:
- `OLLAMA_HOST` - Ollama API endpoint
- `DEFAULT_MODEL` - Default LLM model
- `ANTHROPIC_API_KEY` - Claude API key
- `CONTEXT_WINDOW_SIZE` - Number of recent turns to keep in context
- `APP_PORT` - Server port
- Directory paths: `LOGS_DIR`, `CONVERSATIONS_DIR`, etc.

**Dependencies Detected**:
- `HAS_PYPANDOC` - DOCX export support
- `HAS_CLAUDE` - Claude API support

---

### 3. `models.py` - Data Models
**Purpose**: Define the Conversation class and data structures

**Key Classes**:
- `Conversation` - Manages conversation state, turns, context, and token tracking

**Key Methods**:
- `add_turn()` - Add a new turn to the conversation
- `add_file()` - Attach files to conversation context
- `get_windowed_messages()` - Get messages with sliding window for context management
- `clear_context()` - Clear context while preserving history
- `end_conversation()` - Finalize and save conversation
- `get_token_stats()` - Get token usage statistics

**Functions**:
- `initialize_csv_files()` - Set up CSV log files

---

### 4. `llm_client.py` - LLM API Integration
**Purpose**: Handle API calls to different LLM providers

**Key Functions**:
- `call_llm()` - Universal LLM caller (auto-routes to correct API)
- `call_chat()` - Ollama API integration
- `call_claude()` - Claude API integration
- `is_claude_model()` - Detect if model is Claude
- `get_ollama_models()` - List available Ollama models
- `get_claude_models()` - List available Claude models

---

### 5. `storage.py` - File Operations
**Purpose**: Handle all file I/O, artifact saving, and exports

**Key Functions**:
- `save_turn_artifacts()` - Save turn data in multiple formats (JSONL, MD, HTML, DOCX)
- `export_conversation_to_markdown()` - Export full conversation to Markdown
- `export_conversation_to_docx()` - Export full conversation to DOCX
- `slugify()` - Convert text to filesystem-safe names
- `escape_html()` - HTML entity escaping

**Artifacts Created**:
- Turn-level: `turn.jsonl`, `turn.md`, `turn.html`, `turn.docx`
- Conversation-level: `conversation.json`, exports in various formats

---

### 6. `routes.py` - Flask API Endpoints
**Purpose**: Define all HTTP endpoints and request handling

**Endpoints**:

**Conversation Management**:
- `POST /conversation/new` - Start new conversation
- `POST /conversation/send` - Send message in conversation
- `POST /conversation/end` - End and save conversation
- `POST /conversation/clear-context` - Clear context (reduce tokens)
- `POST /conversation/restore` - Restore saved conversation

**Conversation Retrieval**:
- `GET /conversations/list` - List all saved conversations
- `GET /conversations/load/<conv_id>` - Load specific conversation
- `GET /conversation/export/<conv_id>` - Export conversation (supports ?fmt=json|md|docx)

**Models & Files**:
- `GET /models/list` - List available models
- `POST /upload` - Upload file to conversation context

**UI**:
- `GET /` - Serve main UI
- `GET /favicon` - Serve favicon

---

### 7. `frontend.py` - UI Generation
**Purpose**: Generate the HTML/React user interface

**Key Functions**:
- `generate_index_html()` - Generate complete React UI with Material-UI

**UI Features**:
- Conversation management (new, load, save, end)
- Real-time chat interface
- Model selection dropdown
- File upload support
- Token tracking display
- Context window indicator
- Message resend with different models

---

## Benefits of Modular Structure

### 1. **Maintainability**
- Each module has a single, clear responsibility
- Easier to locate and fix bugs
- Changes are isolated to specific modules

### 2. **Testability**
- Individual modules can be unit tested
- Mock dependencies easily
- Test coverage is easier to achieve

### 3. **Extensibility**
- Add new LLM providers in `llm_client.py`
- Add new export formats in `storage.py`
- Add new endpoints in `routes.py`
- Swap out frontend without touching backend

### 4. **Readability**
- No more scrolling through 1595 lines
- Clear imports show dependencies
- Documentation is easier to maintain

### 5. **Reusability**
- Modules can be reused in other projects
- `llm_client.py` works standalone
- `storage.py` is framework-agnostic

---

## Migration from Original File

The original `local_llm_logger_v3_continuous_chat.py` is preserved for reference. To use the modular version:

```bash
# Old way
python local_llm_logger_v3_continuous_chat.py

# New way (modular)
python main.py
```

Both versions are functionally identical and use the same data formats.

---

## Dependency Graph

```
main.py
  ├── config.py
  ├── models.py
  │   └── config.py
  ├── llm_client.py
  │   └── config.py
  ├── storage.py
  │   └── config.py
  ├── routes.py
  │   ├── config.py
  │   ├── models.py
  │   ├── llm_client.py
  │   ├── storage.py
  │   └── frontend.py
  └── frontend.py
      └── config.py
```

---

## Environment Variables

All configuration is managed through environment variables (or `.env` file):

```bash
# Ollama Configuration
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:32b-instruct

# Claude API (optional)
ANTHROPIC_API_KEY=your_api_key_here

# Context Management
CONTEXT_WINDOW_SIZE=10  # Number of recent turns to keep (0 = unlimited)

# Server Configuration
PORT=5005
```

---

## Development Workflow

### Adding a New LLM Provider

1. Add provider client in `llm_client.py`:
   ```python
   def call_new_provider(model: str, messages: List[Dict]) -> str:
       # Implementation
   ```

2. Update `call_llm()` to route to new provider
3. Add model listing function
4. Update `/models/list` endpoint in `routes.py`

### Adding a New Export Format

1. Add export function in `storage.py`:
   ```python
   def export_conversation_to_new_format(conv_id, summary, conv_dir):
       # Implementation
   ```

2. Update `/conversation/export/<conv_id>` in `routes.py`

### Adding a New Endpoint

1. Add route handler in `routes.py`
2. Update frontend in `frontend.py` if needed
3. Test the endpoint

---

## Testing

Each module can be tested independently:

```python
# Test config
import config
assert config.APP_PORT == 5005

# Test models
from models import Conversation
conv = Conversation("test_id")
conv.add_turn("model", "prompt", "response", 1.5, {})

# Test LLM client
from llm_client import is_claude_model
assert is_claude_model("claude-3-5-sonnet-20241022") == True

# Test storage
from storage import slugify
assert slugify("Hello World!") == "Hello-World"
```

---

## Future Improvements

1. **Add Type Hints**: Full typing for better IDE support
2. **Add Tests**: Unit tests for each module
3. **Add Logging**: Structured logging instead of print statements
4. **Configuration Classes**: Use dataclasses for config
5. **Async Support**: Convert to async/await for better performance
6. **API Documentation**: Add OpenAPI/Swagger docs
7. **Database Backend**: Replace CSV with SQLite/PostgreSQL
8. **Authentication**: Add user authentication and authorization

---

## Questions?

For issues or suggestions, please open an issue in the repository.
