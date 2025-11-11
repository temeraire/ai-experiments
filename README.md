# Local LLM Logger v3 - Continuous Chat Edition

Transform your Ollama LLM interactions into persistent, trackable conversations with comprehensive logging.

## 📦 What's Included

- **Modular Python files** - Main application files (main.py, config.py, models.py, llm_client.py, storage.py, routes.py, frontend.py)
- **`CHANGELOG_v3.md`** - Detailed changelog and migration guide
- **`COMPARISON_v2_v3.md`** - Side-by-side comparison with v2
- **`API_REFERENCE.md`** - Complete API documentation
- **`MODULAR_STRUCTURE.md`** - Documentation of the modular architecture

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Ensure Ollama is running
ollama serve
# if you're using a venv (recommended)
source .venv/bin/activate
# Pull a model
ollama pull qwen2.5:32b-instruct
```

### 2. Install Dependencies
```bash
pip install flask requests markdown pypandoc python-dateutil

# Optional (for DOCX export)
brew install pandoc  # macOS
# or
sudo apt-get install pandoc  # Linux
```

### 3. Run the App
```bash
python main.py
```

### 4. Open Browser
Navigate to: **http://127.0.0.1:5005/**

## ✨ Key Features

### Continuous Conversations
- Chat naturally like ChatGPT
- Full conversation history visible in UI
- Context maintained throughout conversation
- Switch models mid-conversation

### Comprehensive Tracking
- **Conversation-level**: ID, start/end times, duration, total turns, models used
- **Turn-level**: Turn number, timestamp, prompt, response, response time
- **Model tracking**: See which models were used when

### Complete Logging
Every conversation generates:
- ✅ **CSV logs** - Structured data for analysis
- ✅ **JSONL** - Machine-readable per-turn data
- ✅ **Markdown** - Human-readable format
- ✅ **HTML** - Rendered output
- ✅ **DOCX** - Word documents (optional)

### Export Options
- Export individual conversations (JSON, Markdown, DOCX)
- List all conversations with metadata
- Full conversation summaries with turn-by-turn details

## 📊 Data Structure & Output Files

### Directory Layout

```
conversations/
├── conv_20250111_120530_a1b2c3d4/          # Conversation directory
│   ├── conversation.json                   # Summary of entire conversation
│   ├── turn_001_hello/
│   │   ├── turn.jsonl                      # Raw turn data (JSON Lines format)
│   │   ├── turn.md                         # Markdown format
│   │   ├── turn.html                       # Styled HTML
│   │   └── turn.docx                       # Word document (if pandoc installed)
│   ├── turn_002_what-is-quantum-computing/
│   │   ├── turn.jsonl
│   │   ├── turn.md
│   │   ├── turn.html
│   │   └── turn.docx
│   └── ...
└── ...

logs/
├── conversations.csv                       # Global conversation registry
└── turns.csv                               # Global turn registry
```

### File Formats & Details

| File Type | Location | When Created | Contents | Use Case |
|-----------|----------|--------------|----------|----------|
| **conversation.json** | `conversations/conv_{ID}/` | When conversation ends | Full conversation summary: metadata, all turns, models used, timing | Complete conversation snapshot |
| **turn.jsonl** | `conversations/conv_{ID}/turn_{N:03d}_{slug}/` | After each turn | Raw turn data: turn number, timestamp, model used, prompt (user), response, token counts | Machine-readable, detailed turn information |
| **turn.md** | `conversations/conv_{ID}/turn_{N:03d}_{slug}/` | After each turn | Formatted Markdown with headers, code blocks preserved, readable text | Human-readable documents |
| **turn.html** | `conversations/conv_{ID}/turn_{N:03d}_{slug}/` | After each turn | Styled HTML with CSS, syntax-highlighted code blocks | Web viewing, archival, sharing |
| **turn.docx** | `conversations/conv_{ID}/turn_{N:03d}_{slug}/` | After each turn (optional) | Word document with full formatting | Office integration, editing, printing |
| **conversations.csv** | `logs/` | After each conversation ends | Comma-separated values: conversation_id, start_time, end_time, duration_seconds, total_turns, models_used, directory | Analytics, conversation tracking |
| **turns.csv** | `logs/` | After each turn completes | Comma-separated values: conversation_id, turn_number, timestamp, model, response_time_seconds, artifact_paths | Turn-level analytics, performance tracking |

### Naming Conventions

**Conversation IDs:** `conv_YYYYMMDD_HHMMSS_XXXXXXXX`
- Example: `conv_20250111_120530_a1b2c3d4`
- Format: `conv_` + Date (YYYYMMDD) + Time (HHMMSS) + 8-character random UUID

**Turn Folders:** `turn_{NNN}_{slug}`
- Example: `turn_001_hello` or `turn_012_what-is-quantum-computing`
- Format: `turn_` + 3-digit zero-padded turn number + slug derived from first line of prompt
- Slug: First ~40 characters of prompt (sanitized, spaces replaced with hyphens)

**File Names:**
- `turn.jsonl` - Raw JSON Lines format (one JSON object per line)
- `turn.md` - Markdown format
- `turn.html` - HTML format
- `turn.docx` - Word document (only created if pypandoc is installed)

### Output Paths

**Default Paths:**
- Conversations: `./conversations/` (relative to project root)
- Logs: `./logs/` (relative to project root)
- CSV files: `./logs/conversations.csv` and `./logs/turns.csv`

These can be customized by modifying `CONVERSATIONS_DIR` and `LOGS_DIR` in `config.py`

## 🎯 Usage Flow

1. **Click "New Conversation"** to start
2. **Type your message** in the text field
3. **Press Enter** to send (Shift+Enter for multi-line)
4. **See the response** appear in chat history
5. **Continue naturally** - context is maintained
6. **Change models** if desired mid-conversation
7. **Click "End & Save"** when done to archive

## 🔄 Model Switching & Context Handling

### How Context is Carried Between Models

When you switch models mid-conversation, **the full conversation context IS maintained**. Here's how it works:

- **Context Window**: The system uses a sliding context window (default: **10 turns = 20 messages**) to manage memory
- **Every model gets the same context**: When you switch to a new model, it receives the same windowed messages as the previous model
- **Automatic persistence**: All previous exchanges are stored in the conversation's message history and turn logs
- **Growing context**: Each turn adds to the conversation history, so subsequent models can see the full conversation up to the context window limit

### Example Flow
```
Turn 1: You ask about quantum computing (Model: qwen2.5:32b)
        → AI responds, context grows

Turn 2: You ask follow-up question (Model: qwen2.5:32b)
        → AI responds with awareness of Turn 1

Turn 3: You switch to llama2:13b
        → New model SEES Turns 1-2 context automatically
        → You ask a question about the previous discussion
        → New model answers with full context from previous turns

Turn 4+: Continue with llama2:13b (or switch again)
         → Each model has access to the same context window
```

**Technical Details:**
- Context window size is configurable via `CONTEXT_WINDOW_SIZE` in `config.py` (default: 10 turns)
- When context exceeds the window, older turns are dropped to manage token usage
- The model selection is saved to browser localStorage so your preference persists across sessions

## 📝 Example Workflow

```bash
# Start conversation
Click "New Conversation" button

# Chat naturally
You: "What is quantum computing?"
AI: [Response appears with turn #1, model, response time]

You: "How does it compare to classical computing?"
AI: [Response appears with turn #2 - maintains context]

You: "What are the key challenges?"
AI: [Response appears with turn #3 - still maintains context]

# Switch models if desired
Change model dropdown to "llama2:13b"

You: "Summarize our discussion"
AI: [Response with new model, maintains full context]

# End conversation
Click "End & Save" button
→ Conversation archived with full metadata
```

## 🔍 Data Analysis

The CSV format makes it easy to analyze your conversations:

```python
import pandas as pd

# Load conversation data
conversations = pd.read_csv('logs/conversations.csv')
print(f"Total conversations: {len(conversations)}")
print(f"Average duration: {conversations['duration_seconds'].mean():.1f}s")
print(f"Average turns: {conversations['total_turns'].mean():.1f}")

# Load turn data
turns = pd.read_csv('logs/turns.csv')
print(f"Total turns: {len(turns)}")
print(f"Average response time: {turns['response_time_seconds'].mean():.2f}s")
print(f"Models used: {turns['model'].value_counts()}")
```

## 🛠️ Configuration

### Environment Variables
```bash
export OLLAMA_HOST="http://127.0.0.1:11434"  # Ollama API endpoint
export OLLAMA_MODEL="qwen2.5:32b-instruct"   # Default model
export PORT="5005"                            # Flask port
```

### Custom Port
```bash
PORT=8080 python main.py
```

## 📚 Documentation

### Full Documentation Files

1. **CHANGELOG_v3.md**
   - Detailed feature list
   - Migration from v2
   - Installation guide
   - Troubleshooting

2. **COMPARISON_v2_v3.md**
   - Side-by-side comparison
   - When to use which version
   - Migration strategies
   - Architecture differences

3. **API_REFERENCE.md**
   - Complete API documentation
   - Request/response examples
   - cURL examples
   - Python/Bash scripts

## 🔑 Key Differences from v2

| Feature | v2 | v3 |
|---------|----|----|
| **Model** | Single-turn or N-turn window | Full conversations |
| **UI** | Prompt → Result | Chat interface |
| **Context** | Auto-clears after N | Persists until ended |
| **Tracking** | Per-turn only | Per-turn + Per-conversation |
| **Export** | Daily batches | Individual conversations |

## 🎨 UI Features

- **Real-time chat history** - See all messages
- **Color-coded messages** - User vs Assistant
- **Metadata display** - Turn #, model, response time
- **Auto-scroll** - Always see latest message
- **Keyboard shortcuts** - Enter to send, Shift+Enter for newline
- **Loading states** - Visual feedback during generation
- **Status indicators** - Current turn count and model
- **Responsive design** - Works on mobile and desktop

## 📦 Export Examples

### Export as JSON
```bash
curl -O http://127.0.0.1:5005/conversation/export/conv_20250111_120530_a1b2c3d4?fmt=json
```

### Export as Markdown
```bash
curl -O http://127.0.0.1:5005/conversation/export/conv_20250111_120530_a1b2c3d4?fmt=md
```

### List All Conversations
```bash
curl http://127.0.0.1:5005/conversations/list | jq
```

## 🔒 Security Note

**This application is designed for LOCAL USE ONLY.**

For production deployment:
- Add authentication
- Implement rate limiting
- Sanitize inputs
- Use HTTPS
- Configure CORS properly

## 🐛 Troubleshooting

### "Conversation not found" error
→ Server was restarted. Click "New Conversation" to start fresh.

### Pandoc errors
→ DOCX export requires Pandoc. Either install it or ignore .docx files.

### Ollama connection errors
→ Ensure Ollama is running: `ollama serve`

### Model not found
→ Pull the model: `ollama pull model-name`

## 💡 Tips

1. **End conversations regularly** to free up memory
2. **Use model switching** to try different approaches
3. **Export important conversations** for archival
4. **Monitor turn counts** to track conversation complexity
5. **Check response times** to optimize model selection

## 🎯 Use Cases

- **Research interviews** - Track multi-turn explorations
- **Learning sessions** - Maintain educational context
- **Problem solving** - Keep context across complex questions
- **Creative writing** - Build narratives over time
- **Code assistance** - Maintain project context
- **Data analysis** - Track analytical conversations

## 📈 Performance

### Memory Usage
- Each active conversation stays in memory
- End conversations to free memory
- No hard limits, but monitor for very long conversations

### Response Time
- Grows slightly with conversation length (more context)
- Depends on model's context window handling
- Tracked per-turn for analysis

### Disk Usage
- One directory per conversation
- Subdirectories for each turn
- CSV logs append-only (efficient)

## 🔄 Updates & Extensions

### Potential Future Features
- Resume archived conversations
- Search through conversation history
- Conversation tagging/labeling
- Multi-conversation export
- Response regeneration
- Conversation branching
- Streaming responses
- Voice input/output

## 📄 License

This is a personal tool for logging LLM interactions. Use and modify as needed.

## 🙏 Credits

Built with:
- **Flask** - Web framework
- **Ollama** - Local LLM runtime
- **React + Material-UI** - Frontend
- **Pandoc** - Document conversion (optional)

---

## Quick Reference Card

```
┌─────────────────────────────────────────────┐
│  LOCAL LLM LOGGER V3 - QUICK REFERENCE      │
├─────────────────────────────────────────────┤
│  Start:    python main.py                   │
│  URL:      http://127.0.0.1:5005/          │
│  Logs:     logs/*.csv                       │
│  Data:     conversations/conv_*/            │
├─────────────────────────────────────────────┤
│  UI:       New Conversation → Chat → End   │
│  Keys:     Enter (send), Shift+Enter (line)│
│  Export:   /conversation/export/<id>?fmt=   │
│  List:     /conversations/list              │
└─────────────────────────────────────────────┘
```

**Ready to start logging? Run the script and open your browser!** 🚀
