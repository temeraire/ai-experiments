# Local LLM Logger v3 - Continuous Chat Upgrade

## Overview
This version transforms your Flask app into a continuous chat application like ChatGPT, while preserving all existing logging capabilities and adding comprehensive conversation tracking.

## Key Changes

### 1. **Conversation Architecture**
- **New `Conversation` class**: Manages entire conversation lifecycle
  - Tracks conversation ID, start/end times, total duration
  - Maintains message history for chat continuity
  - Records all models used during conversation
  - Auto-saves conversation summary on completion

### 2. **Enhanced Logging System**

#### New CSV Files:
- **`conversations.csv`**: Conversation-level tracking
  - Fields: conversation_id, start_time, end_time, duration_seconds, total_turns, models_used, conversation_dir
  
- **`turns.csv`**: Turn-level tracking (replaces old log.csv)
  - Fields: conversation_id, turn_number, timestamp, model, prompt, response, response_time_seconds, paths

#### Directory Structure:
```
conversations/
  conv_20250101_120000_a1b2c3d4/
    conversation.json (summary)
    turn_001_hello/
      turn.jsonl
      turn.md
      turn.html
      turn.docx (optional)
    turn_002_explain-quantum/
      ...
```

### 3. **UI Improvements**

#### New Features:
- **"New Conversation" button**: Start fresh conversations
- **"End & Save" button**: Explicitly end and archive conversations
- **Real-time chat history**: See all messages in current conversation
- **Turn counter**: Visual indicator of conversation progress
- **Model display**: Shows active model as a chip
- **Response time display**: Per-message timing
- **Enter to send**: Shift+Enter for new lines
- **Auto-scroll**: Chat automatically scrolls to latest message

#### Visual Design:
- Color-coded messages (user vs assistant)
- Metadata shown for each message (turn #, model, response time)
- Scrollable chat history (max 60vh)
- Loading states during generation

### 4. **New API Endpoints**

#### `/conversation/new` (POST)
Start a new conversation
```json
Request: {"session_id": "session_123"}
Response: {"conversation_id": "conv_20250101_120000_abc", "start_time": "2025-01-01T12:00:00"}
```

#### `/conversation/send` (POST)
Send a message in active conversation
```json
Request: {
  "conversation_id": "conv_...",
  "model": "qwen2.5:32b-instruct",
  "prompt": "Hello!"
}
Response: {
  "conversation_id": "conv_...",
  "turn_number": 1,
  "model": "qwen2.5:32b-instruct",
  "response": "Hi! How can I help?",
  "response_time": 2.34,
  "paths": {...},
  "conversation_info": {...}
}
```

#### `/conversation/end` (POST)
End and save conversation
```json
Request: {"conversation_id": "conv_..."}
Response: {
  "conversation_id": "conv_...",
  "total_turns": 5,
  "duration_seconds": 123.45,
  "conversation_dir": "/path/to/conv"
}
```

#### `/conversation/export/<conv_id>` (GET)
Export specific conversation
- Query params: `fmt=json|md|docx`
- Returns: Downloadable file with full conversation

#### `/conversations/list` (GET)
List all completed conversations
```json
Response: {
  "conversations": [
    {
      "conversation_id": "conv_...",
      "start_time": "...",
      "end_time": "...",
      "duration_seconds": "123.45",
      "total_turns": "5",
      "models_used": "qwen2.5:32b-instruct"
    }
  ]
}
```

### 5. **Model Switching Mid-Conversation**
- Change model any time using the model field
- Each turn records which model was used
- Conversation tracks all models used
- No context loss when switching models

### 6. **Preserved Features**
All original features remain intact:
- ✅ CSV logging (enhanced with more detail)
- ✅ JSONL per turn
- ✅ Markdown export
- ✅ HTML export
- ✅ DOCX export (optional)
- ✅ Timestamps for everything
- ✅ Clean React/MUI interface

## Migration from v2

### What to Keep:
- Your existing `sessions/` directory (if you want to preserve old data)
- Your Ollama setup and models
- Environment variables (OLLAMA_HOST, OLLAMA_MODEL, PORT)

### What's New:
- `conversations/` directory will be created automatically
- `logs/conversations.csv` for conversation tracking
- `logs/turns.csv` replaces `logs/log.csv`

### Breaking Changes:
- Removed `keep_last` parameter (conversations now persist naturally)
- Session IDs now map to conversations instead of history arrays
- Old `/generate` endpoint removed (use `/conversation/*` endpoints)
- Old `/start_session` and `/clear_session` endpoints removed

## Installation & Usage

### 1. Install Dependencies
```bash
pip install flask requests markdown pypandoc python-dateutil
```

Optional:
```bash
brew install pandoc  # For DOCX export
```

### 2. Start Ollama
```bash
ollama pull qwen2.5:32b-instruct  # or your preferred model
ollama serve  # if not already running
```

### 3. Run the App
```bash
python local_llm_logger_v3_continuous_chat.py
```

### 4. Open Browser
Navigate to: http://127.0.0.1:5005/

## Usage Flow

1. **Click "New Conversation"** to start
2. **Type messages** in the text field
3. **Press Enter** to send (Shift+Enter for multi-line)
4. **Watch chat history** build up in real-time
5. **Change models** mid-conversation if desired
6. **Click "End & Save"** when done to archive the conversation

## File Organization

### Per Conversation:
- `conversations/conv_ID/conversation.json` - Full conversation summary
- `conversations/conv_ID/turn_NNN_slug/` - Individual turn artifacts
  - `turn.jsonl` - Structured data
  - `turn.md` - Markdown format
  - `turn.html` - Rendered HTML
  - `turn.docx` - Word document (optional)

### Global Logs:
- `logs/conversations.csv` - All conversations metadata
- `logs/turns.csv` - All turns across all conversations

## Export Options

### Export Single Conversation:
```bash
# JSON format (includes all metadata)
curl http://localhost:5005/conversation/export/conv_20250101_120000_abc?fmt=json

# Markdown format (readable)
curl http://localhost:5005/conversation/export/conv_20250101_120000_abc?fmt=md

# DOCX format (requires Pandoc)
curl http://localhost:5005/conversation/export/conv_20250101_120000_abc?fmt=docx
```

### List All Conversations:
```bash
curl http://localhost:5005/conversations/list
```

## Configuration

### Environment Variables:
```bash
export OLLAMA_HOST="http://127.0.0.1:11434"  # Ollama API endpoint
export OLLAMA_MODEL="qwen2.5:32b-instruct"   # Default model
export PORT="5005"                            # Flask port
```

## Data Analysis

With the new CSV structure, you can easily analyze your conversations:

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

# Models used
print("Models used:", turns['model'].value_counts())
```

## Tips

1. **Model Switching**: Try different models for different parts of conversation
2. **Long Conversations**: The UI auto-scrolls, making long chats easy to follow
3. **Keyboard Shortcuts**: Enter to send, Shift+Enter for new line
4. **Export Early**: You can export conversations even while they're active
5. **Session Persistence**: Your conversation ID is saved in localStorage

## Troubleshooting

### "Conversation not found" error:
- This happens if the server restarts. Click "New Conversation" to start fresh.

### Pandoc errors:
- DOCX export requires Pandoc. Either install it or ignore .docx files.

### Ollama connection errors:
- Ensure Ollama is running: `ollama serve`
- Check OLLAMA_HOST environment variable
- Verify model is pulled: `ollama list`

## Future Enhancements (Ideas)

- [ ] Search through past conversations
- [ ] Resume/continue archived conversations
- [ ] Conversation tagging/labeling
- [ ] Export multiple conversations to single file
- [ ] Response regeneration
- [ ] Conversation branching
- [ ] Streaming responses
- [ ] Voice input/output
- [ ] Multi-user support

## Technical Details

### Memory Management:
- Active conversations stored in `ACTIVE_CONVERSATIONS` dict
- Conversation removed from memory when ended
- All data persisted to disk immediately

### Response Timing:
- Measured using `time.time()` before/after API call
- Recorded in seconds with 2 decimal precision
- Included in every turn record

### Chat Context:
- Full message history maintained per conversation
- Sent to Ollama with each request for continuity
- No arbitrary limits (relies on model's context window)

---

**Version:** 3.0  
**Date:** October 2025  
**Author:** Enhanced from v2 Flask + Ollama Logger
