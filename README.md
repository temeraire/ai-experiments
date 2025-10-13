# Local LLM Logger v3 - Continuous Chat Edition

Transform your Ollama LLM interactions into persistent, trackable conversations with comprehensive logging.

## 📦 What's Included

- **`local_llm_logger_v3_continuous_chat.py`** - Main application file
- **`CHANGELOG_v3.md`** - Detailed changelog and migration guide
- **`COMPARISON_v2_v3.md`** - Side-by-side comparison with v2
- **`API_REFERENCE.md`** - Complete API documentation

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Ensure Ollama is running
ollama serve

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
python local_llm_logger_v3_continuous_chat.py
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

## 📊 Data Structure

```
conversations/
├── conv_20250111_120530_a1b2c3d4/
│   ├── conversation.json          # Full conversation summary
│   ├── turn_001_hello/
│   │   ├── turn.jsonl
│   │   ├── turn.md
│   │   ├── turn.html
│   │   └── turn.docx (optional)
│   ├── turn_002_explain-quantum/
│   │   └── ...
│   └── ...
└── ...

logs/
├── conversations.csv  # Conversation-level metadata
└── turns.csv         # All turns across all conversations
```

## 🎯 Usage Flow

1. **Click "New Conversation"** to start
2. **Type your message** in the text field
3. **Press Enter** to send (Shift+Enter for multi-line)
4. **See the response** appear in chat history
5. **Continue naturally** - context is maintained
6. **Change models** if desired mid-conversation
7. **Click "End & Save"** when done to archive

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
PORT=8080 python local_llm_logger_v3_continuous_chat.py
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
│  Start:    python local_llm_logger_v3...py  │
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
