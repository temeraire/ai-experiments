# v2 vs v3 Feature Comparison

## Quick Reference: What Changed

| Feature | v2 (Original) | v3 (Enhanced) |
|---------|---------------|---------------|
| **Conversation Model** | Single-turn or N-turn rolling window | Full continuous conversations |
| **Context Persistence** | Auto-clears after N turns | Persists until "End & Save" clicked |
| **UI Pattern** | Prompt → Send → Result → Repeat | Chat interface with history |
| **Logging Level** | Per-turn only | Per-turn + Per-conversation |
| **CSV Files** | 1 file (`log.csv`) | 2 files (`conversations.csv`, `turns.csv`) |
| **Directory Structure** | `sessions/YYYY-MM-DD/HHMMSS_slug/` | `conversations/conv_ID/turn_NNN_slug/` |
| **Model Switching** | Supported per turn | Supported per turn + tracked in conversation |
| **Response Time** | Not tracked | Tracked per turn (seconds) |
| **Conversation Metadata** | None | Start/end times, duration, total turns |
| **Export** | Daily batch export | Individual conversation export + list all |
| **Session Management** | `/start_session`, `/clear_session` | `/conversation/new`, `/conversation/end` |
| **Main Endpoint** | `/generate` | `/conversation/send` |
| **Chat History** | Not visible | Displayed in real-time |
| **Keyboard Shortcuts** | None | Enter to send, Shift+Enter for newline |

## Detailed Comparison

### Data Structure

#### v2: Session-based Rolling History
```python
HISTORIES[session_id] = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    # Cleared after N turns
]
```

#### v3: Conversation Objects
```python
class Conversation:
    - id: str
    - start_time: datetime
    - end_time: datetime
    - turns: List[Turn]
    - messages: List[Message]  # For API
    - models_used: Set[str]
```

### UI Flow

#### v2 Flow:
1. Set model
2. Set keep_last N
3. Enter prompt
4. Click "Send"
5. View result below
6. (Repeat, context clears after N)

#### v3 Flow:
1. Click "New Conversation"
2. Set model (can change anytime)
3. Type message
4. Press Enter
5. See chat history build up
6. Continue conversation naturally
7. Click "End & Save" when done

### Logging Output

#### v2 Logs:
```
sessions/
  2025-01-01/
    120000_hello/
      record.jsonl
      result.md
      result.html
      result.docx

logs/
  log.csv (all turns mixed together)
```

#### v3 Logs:
```
conversations/
  conv_20250101_120000_a1b2c3d4/
    conversation.json (summary)
    turn_001_hello/
      turn.jsonl
      turn.md
      turn.html
      turn.docx
    turn_002_explain/
      turn.jsonl
      ...

logs/
  conversations.csv (conversation metadata)
  turns.csv (all turns with conversation_id)
```

### CSV Structure

#### v2 log.csv:
```csv
date_iso,time_local,model,prompt,result,jsonl_path,md_path,html_path,docx_path_optional
2025-01-01,12:00:00,qwen2.5:32b,hello,hi there,...
```

#### v3 conversations.csv:
```csv
conversation_id,start_time,end_time,duration_seconds,total_turns,models_used,conversation_dir
conv_20250101_120000_abc,2025-01-01T12:00:00,2025-01-01T12:05:30,330.45,5,qwen2.5:32b,...
```

#### v3 turns.csv:
```csv
conversation_id,turn_number,timestamp,model,prompt,response,response_time_seconds,jsonl_path,...
conv_20250101_120000_abc,1,2025-01-01T12:00:05,qwen2.5:32b,hello,hi,2.34,...
conv_20250101_120000_abc,2,2025-01-01T12:01:10,qwen2.5:32b,explain,sure...,3.21,...
```

### API Endpoints

#### v2 Endpoints:
- `POST /start_session` → Create session ID
- `POST /clear_session` → Clear history
- `POST /generate` → Send prompt, get response
- `GET /export?date=YYYY-MM-DD&fmt=docx` → Export day

#### v3 Endpoints:
- `POST /conversation/new` → Start new conversation
- `POST /conversation/send` → Send message in conversation
- `POST /conversation/end` → End and save conversation
- `GET /conversation/export/<conv_id>?fmt=json|md|docx` → Export conversation
- `GET /conversations/list` → List all conversations

## When to Use Which Version

### Use v2 If You Want:
- Simple turn-by-turn logging without chat history
- Auto-clearing context after N turns
- Minimal UI (prompt/result pattern)
- Day-based organization
- Stateless or semi-stateless operation

### Use v3 If You Want:
- Full chat conversations like ChatGPT
- Conversation-level analytics
- Chat history visible in UI
- Per-conversation exports
- Track conversation duration and turn count
- See which models were used in each conversation
- Natural conversation flow

## Migration Strategy

### Option 1: Fresh Start
1. Keep v2 running on port 5005
2. Run v3 on port 5006
3. Start using v3 for new conversations
4. Keep v2 data archived

### Option 2: Side-by-Side
1. Rename v2 to `local_llm_logger_v2.py`
2. Run v2 on port 5005
3. Run v3 on port 5006
4. Use both as needed

### Option 3: Full Migration
1. Backup v2 `sessions/` and `logs/` directories
2. Replace v2 with v3
3. Start fresh with new conversation model
4. Old data remains accessible in backup

## Performance Considerations

### Memory Usage:
- **v2**: Minimal (only last N messages in memory)
- **v3**: Higher (full conversation in memory until ended)
  - Mitigation: End conversations regularly
  - No hard limit, but monitor for very long chats

### Disk Usage:
- **v2**: One directory per turn
- **v3**: One directory per conversation, subdirectories per turn
  - More organized but slightly more nested

### Response Time:
- **v2**: Consistent (no history overhead when N=0)
- **v3**: Grows slightly with conversation length
  - Depends on model's context window handling

## Common Questions

### Q: Can I import v2 data into v3?
**A:** Not directly, but you can write a script to convert session directories into conversation format. The turn data is compatible.

### Q: Can I run both versions simultaneously?
**A:** Yes! Just use different ports:
```bash
PORT=5005 python local_llm_logger_v2.py &
PORT=5006 python local_llm_logger_v3_continuous_chat.py &
```

### Q: What happens if I don't click "End & Save"?
**A:** The conversation stays in memory. If the server restarts, you lose it. Always end conversations to preserve them.

### Q: Can I resume an ended conversation?
**A:** Not in v3.0, but you can:
1. Export the conversation
2. Start a new conversation
3. Summarize previous context in your first message

### Q: How long can conversations be?
**A:** Limited only by:
1. Your model's context window
2. Available memory
3. Practical usability (very long convos become hard to navigate)

Recommendation: End conversations at natural breakpoints (topic changes, end of session).

### Q: Can I delete conversations?
**A:** Yes, manually delete the conversation directory:
```bash
rm -rf conversations/conv_XXXXX
```
Then remove the row from `logs/conversations.csv` if desired.

## Code Architecture Differences

### v2 Architecture:
```
Flask App
├── In-memory: HISTORIES dict (ephemeral)
├── Endpoints: /generate (stateless or rolling)
└── Storage: Date-based directories
```

### v3 Architecture:
```
Flask App
├── In-memory: ACTIVE_CONVERSATIONS dict
│   └── Conversation objects (until ended)
├── Endpoints: /conversation/* (stateful)
└── Storage: Conversation-based directories
    ├── Per-conversation summaries
    └── Per-turn artifacts
```

### Key Class: Conversation

```python
class Conversation:
    def __init__(self, conv_id):
        # Initialize conversation
        self.id = conv_id
        self.start_time = datetime.now()
        self.turns = []
        self.messages = []  # For API continuity
        self.models_used = set()
    
    def add_turn(self, model, prompt, response, response_time, paths):
        # Add turn and update messages
        # Log to CSV
    
    def end_conversation(self):
        # Calculate duration
        # Log to conversations CSV
        # Save summary JSON
```

This class encapsulates all conversation logic, making the code cleaner and more maintainable.

## Summary

**v2** is great for **logging individual LLM calls** with optional short-term context.

**v3** is designed for **natural conversations** with full history, tracking, and export capabilities.

Both preserve all the original logging features you loved, just organized differently to match their use cases.

---

Choose based on your use case:
- Research/logging individual queries → v2
- Interactive conversations → v3
- Both → Run them side-by-side!
