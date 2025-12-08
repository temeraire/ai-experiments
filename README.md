# Local LLM Logger v3 - Multi-Model AI Chat Platform

A powerful Flask-based web application for interacting with multiple AI models (Ollama, Claude, Gemini) with comprehensive conversation logging, cost tracking, and auto-summarization.

## 🌟 Features

### Multi-Model Support
- **Local Models via Ollama** - Run models locally (qwen2.5, mixtral, gemma3, deepseek-r1, etc.)
- **Claude API Integration** - Access 6 Claude models (Sonnet 4.5, Opus 4.1, Haiku 4.5, etc.)
- **Gemini API Integration** - Use 5 Gemini models (2.5-flash/pro, 2.0-flash, 1.5-pro/flash)
- **Model Comparison Mode** - Run the same prompt across multiple models simultaneously
- **Real-time Streaming** - See responses appear in real-time for multi-model comparisons

### Intelligent Features
- **Auto-Summarization** - Every response gets a concise 2-3 sentence summary
- **Cost Tracking** - Real-time cost estimation for paid API models
- **Smart Naming** - Conversations and files named with model abbreviations
- **Context Management** - Sliding window keeps recent conversation context

### User Experience
- **ChatGPT-like Interface** - Natural conversation flow with full history
- **Color-Coded Models** - Green highlighting for paid models (light=available, dark=selected)
- **Collapsible Summaries** - Yellow summary boxes for quick scanning
- **File Upload Support** - Attach PDFs, DOCX, Jupyter notebooks to conversations
- **Auto-Save** - Conversations saved automatically when closing browser

### Data Management
- **DOCX Export** - Professional document output with summaries
- **JSONL Logging** - Structured data for analysis
- **Organized Storage** - Conversations stored with descriptive names
- **Empty Folder Cleanup** - Automatic cleanup on startup

## 🚀 Quick Start

### Prerequisites

1. **Ollama** (for local models)
```bash
# Install Ollama from https://ollama.ai
ollama serve

# Pull some models
ollama pull qwen2.5:32b-instruct
ollama pull gemma3:4b
```

2. **Python 3.8+**
```bash
python --version  # Should be 3.8 or higher
```

3. **API Keys** (optional, for Claude/Gemini)
```bash
# Create .env file
cp .env.example .env

# Add your API keys to .env
ANTHROPIC_API_KEY=your_claude_key_here
GOOGLE_API_KEY=your_gemini_key_here
```

### Installation

```bash
# Clone the repository
cd ai_experiments

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install flask requests markdown pypandoc python-dateutil anthropic google-generativeai

# Install pandoc for DOCX export (optional but recommended)
brew install pandoc  # macOS
# or
sudo apt-get install pandoc  # Linux
```

### Running the Application

```bash
python main.py
```

The server will start on **http://127.0.0.1:5005**

You should see:
```
============================================================
Local LLM Logger v3 - Modular Edition
============================================================
Server: http://127.0.0.1:5005
Ollama host: http://127.0.0.1:11434
Checking for empty conversation folders...
============================================================
```

Open your browser and navigate to **http://127.0.0.1:5005**

## 📖 How to Use

### Basic Chat
1. Click "New Conversation"
2. Select a model from the dropdown
3. Type your message and press Enter (or Shift+Enter for new line)
4. See the response with an auto-generated summary
5. Click "End & Save" when finished

### Multi-Model Comparison
1. Click "Compare Models" button
2. Select 2+ models from the chips
3. Enter your prompt
4. Watch responses appear in real-time
5. Mark the best response with the star button

### File Upload
1. Start a conversation
2. Click the "+" button to upload files
3. Supported formats: .txt, .py, .pdf, .docx, .ipynb
4. Files are available to all messages in that conversation

### Cost Tracking
- Paid models show with $ prefix and green highlighting
- Cumulative cost displayed in green chip
- Per-turn costs returned in API responses

## 🏗️ Architecture

### Modular Structure

The application follows a clean modular architecture with separated concerns:

```
ai_experiments/
├── main.py              # Application entry point
├── config.py            # Configuration & settings
├── models.py            # Data models (Conversation class)
├── llm_client.py        # LLM API integrations
├── storage.py           # File operations & exports
├── routes.py            # Flask API endpoints
├── frontend.py          # React/MUI UI generation
├── conversations/       # Saved conversation data
├── logs/               # CSV logs
└── .env                # API keys (not in git)
```

### Core Files

#### `main.py` - Application Entry Point
- Initializes Flask app
- Runs cleanup on startup
- Starts web server
- **Key Function:** `cleanup_empty_conversations()` - Removes empty folders

#### `config.py` - Configuration Management
- Environment settings (ports, hosts, directories)
- Model abbreviations (`CS45`, `CO41`, `G3P`, etc.)
- Pricing information for paid models
- Helper functions (`get_model_abbrev()`, `is_paid_model()`, `get_model_cost()`)

#### `models.py` - Data Models
- **`Conversation` class** - Main data structure
  - Tracks turns, messages, models used
  - Token estimation and tracking
  - Context window management
  - File attachment support
- **CSV initialization** - Creates log files if needed

#### `llm_client.py` - LLM Integration Layer
- **Ollama Integration**
  - `call_chat()` - Chat with local models
  - `get_ollama_models()` - List available models
- **Claude Integration**
  - `call_claude()` - Anthropic API calls
  - `get_claude_models()` - List Claude models
- **Gemini Integration**
  - `call_gemini()` - Google API calls
  - `get_gemini_models()` - List Gemini models
- **Universal Router**
  - `call_llm()` - Routes to correct API based on model name
  - `generate_summary()` - Creates 2-3 sentence summaries

#### `storage.py` - File Operations
- **Artifact Generation**
  - `save_turn_artifacts()` - Save single model responses
  - `save_comparison_artifacts()` - Save multi-model comparisons
  - Creates DOCX files with summaries at top
  - Generates JSONL for metadata
- **Export Functions**
  - `export_conversation_to_markdown()`
  - `export_conversation_to_docx()`
- **Naming Utilities**
  - `slugify()` - Create filesystem-safe names

#### `routes.py` - API Endpoints
- **Conversation Management**
  - `POST /conversation/new` - Start new conversation
  - `POST /conversation/send` - Send message to model
  - `POST /conversation/compare` - Compare multiple models (parallel)
  - `POST /conversation/compare-stream` - Streaming comparison (SSE)
  - `POST /conversation/end` - Save and end conversation
  - `POST /conversation/clear-context` - Clear context window
- **Data Access**
  - `GET /conversations/list` - List saved conversations
  - `GET /conversations/load/<id>` - Load conversation
  - `GET /conversation/export/<id>?fmt=docx` - Export conversation
- **Model Information**
  - `GET /models/list` - Get available models with pricing
- **File Upload**
  - `POST /upload` - Handle file attachments

#### `frontend.py` - UI Generation
- **React/Material-UI Interface**
  - Server-side React component generation
  - Material-UI components for modern design
- **Features**
  - Chat history display with summaries
  - Model selection with color coding
  - Multi-model comparison interface
  - Cost tracking display
  - File upload interface
  - Auto-save hooks (beforeunload, periodic)
- **Styling**
  - Yellow collapsible summary boxes
  - Green highlighting for paid models
  - Responsive layout

### Data Flow

```
User Input → Frontend (React/MUI)
    ↓
Flask Routes → LLM Client
    ↓
API Call (Ollama/Claude/Gemini)
    ↓
Response + Summary Generation
    ↓
Storage (DOCX + JSONL)
    ↓
Conversation Model Update
    ↓
Frontend Display (with summary)
```

### Conversation Storage Structure

```
conversations/
└── conv_20251124_130026_G1b_1model/
    ├── conversation.json              # Metadata
    └── turn_001_G1b_Explain-what-a-binary/
        ├── turn.jsonl                 # Structured data
        └── turn1_G1b.docx            # DOCX export with summary
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Optional: Override defaults
OLLAMA_HOST=http://127.0.0.1:11434
APP_PORT=5005
```

### Model Abbreviations

Models are abbreviated in file names for brevity:
- `CS45` - Claude Sonnet 4.5
- `CO41` - Claude Opus 4.1
- `CH45` - Claude Haiku 4.5
- `G25F` - Gemini 2.5 Flash
- `G25P` - Gemini 2.5 Pro
- `QW32` - Qwen 2.5:32b
- See `config.py` for full list

### Cost Tracking

Pricing per 1M tokens (input/output):
- **Claude Sonnet 4.5**: $3/$15
- **Claude Opus 4.1**: $15/$75
- **Claude Haiku 4.5**: $0.80/$4
- **Gemini 2.5 Flash**: $0.075/$0.30
- **Gemini 2.5 Pro**: $1.25/$5.00

## 📊 Recent Updates

### Phase 4: Auto-Summary Feature (PR #7)
- Automatic 2-3 sentence summaries for all responses
- Collapsible UI display with yellow styling
- Summaries placed at top of DOCX files
- Works for single and multi-model comparisons

### Phase 5: Bug Fixes (PR #8)
- Empty folder cleanup on startup
- Auto-save on page close
- Periodic activity checks
- Prevents data loss

### Phase 1-3: External Models & Cost Tracking (PR #6)
- Claude and Gemini API integration
- Real-time cost tracking and display
- Model abbreviation system
- Simplified file naming

## 🛠️ Development

### Project Structure
- **Modular Design** - Each file has a single responsibility
- **Type Hints** - Modern Python with type annotations
- **Error Handling** - Comprehensive try/catch blocks
- **Clean Code** - Following CLAUDE.md guidelines

### Adding a New Model
1. Add to `llm_client.py`:
   - Create `call_newmodel()` function
   - Add to `call_llm()` router
   - Add to `get_newmodel_models()`
2. Update `config.py`:
   - Add abbreviation to `MODEL_ABBREVS`
   - Add pricing to `MODEL_PRICING` (if paid)
3. Test with a conversation

### Running Tests
```bash
# Start the app
python main.py

# Open browser and test features manually
# Or use curl for API testing:
curl -X POST http://127.0.0.1:5005/conversation/new \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test"}'
```

## 📝 API Documentation

See `API_REFERENCE.md` for complete API documentation.

### Quick Examples

**Create Conversation:**
```bash
curl -X POST http://127.0.0.1:5005/conversation/new \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session"}'
```

**Send Message:**
```bash
curl -X POST http://127.0.0.1:5005/conversation/send \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_20251124_130026_068c611a",
    "model": "gemma3:4b",
    "prompt": "What is Python?"
  }'
```

**Compare Models:**
```bash
curl -X POST http://127.0.0.1:5005/conversation/compare \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_20251124_130026_068c611a",
    "models": ["gemma3:1b", "gemma3:4b"],
    "prompt": "Explain recursion"
  }'
```

## 🤝 Contributing

1. Follow the modular structure
2. Add type hints to functions
3. Update documentation
4. Test thoroughly
5. Follow CLAUDE.md guidelines (minimal changes, preserve features)

## 📄 License

See repository for license information.

## 🙏 Credits

- Built with Flask, React, and Material-UI
- Powered by Ollama, Anthropic Claude, and Google Gemini
- Auto-summarization using the same models that generate responses

---

**Questions or issues?** Check the existing documentation files:
- `CHANGELOG_v3.md` - Version history and migration guide
- `COMPARISON_v2_v3.md` - Comparison with previous version
- `API_REFERENCE.md` - Complete API documentation
- `MODULAR_STRUCTURE.md` - Detailed architecture guide
- `TODO.md` - Implementation roadmap and completed phases
