# API Reference - Local LLM Logger v3

## Base URL
```
http://127.0.0.1:5005
```

## Endpoints

### 1. Start New Conversation

**Endpoint:** `POST /conversation/new`

**Description:** Creates a new conversation and returns a unique conversation ID.

**Request:**
```json
{
  "session_id": "session_1234567890"
}
```

**Response:**
```json
{
  "conversation_id": "conv_20250111_120530_a1b2c3d4",
  "start_time": "2025-01-11T12:05:30.123456"
}
```

**cURL Example:**
```bash
curl -X POST http://127.0.0.1:5005/conversation/new \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my_session"}'
```

---

### 2. Send Message

**Endpoint:** `POST /conversation/send`

**Description:** Sends a message in an active conversation and receives a response.

**Request:**
```json
{
  "conversation_id": "conv_20250111_120530_a1b2c3d4",
  "model": "qwen2.5:32b-instruct",
  "prompt": "What is quantum computing?"
}
```

**Response:**
```json
{
  "conversation_id": "conv_20250111_120530_a1b2c3d4",
  "turn_number": 1,
  "model": "qwen2.5:32b-instruct",
  "response": "Quantum computing is a type of computation that...",
  "response_time": 2.34,
  "paths": {
    "jsonl_path": "/path/to/turn_001_what-is-quantum/turn.jsonl",
    "md_path": "/path/to/turn_001_what-is-quantum/turn.md",
    "html_path": "/path/to/turn_001_what-is-quantum/turn.html",
    "docx_path": "/path/to/turn_001_what-is-quantum/turn.docx",
    "turn_dir": "/path/to/turn_001_what-is-quantum"
  },
  "conversation_info": {
    "total_turns": 1,
    "duration": 5.67,
    "models_used": ["qwen2.5:32b-instruct"]
  }
}
```

**cURL Example:**
```bash
curl -X POST http://127.0.0.1:5005/conversation/send \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_20250111_120530_a1b2c3d4",
    "model": "qwen2.5:32b-instruct",
    "prompt": "What is quantum computing?"
  }'
```

---

### 3. End Conversation

**Endpoint:** `POST /conversation/end`

**Description:** Ends an active conversation and saves all metadata.

**Request:**
```json
{
  "conversation_id": "conv_20250111_120530_a1b2c3d4"
}
```

**Response:**
```json
{
  "conversation_id": "conv_20250111_120530_a1b2c3d4",
  "total_turns": 5,
  "duration_seconds": 123.45,
  "conversation_dir": "/path/to/conversations/conv_20250111_120530_a1b2c3d4"
}
```

**cURL Example:**
```bash
curl -X POST http://127.0.0.1:5005/conversation/end \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "conv_20250111_120530_a1b2c3d4"}'
```

---

### 4. Export Conversation

**Endpoint:** `GET /conversation/export/<conversation_id>`

**Description:** Exports a specific conversation in various formats.

**Query Parameters:**
- `fmt` - Format: `json`, `md`, or `docx` (default: `json`)

**Response:** File download

**Examples:**

```bash
# Export as JSON (includes all metadata)
curl -O http://127.0.0.1:5005/conversation/export/conv_20250111_120530_a1b2c3d4?fmt=json

# Export as Markdown (readable format)
curl -O http://127.0.0.1:5005/conversation/export/conv_20250111_120530_a1b2c3d4?fmt=md

# Export as DOCX (requires Pandoc)
curl -O http://127.0.0.1:5005/conversation/export/conv_20250111_120530_a1b2c3d4?fmt=docx
```

**JSON Export Format:**
```json
{
  "conversation_id": "conv_20250111_120530_a1b2c3d4",
  "start_time": "2025-01-11T12:05:30.123456",
  "end_time": "2025-01-11T12:10:45.678901",
  "duration_seconds": 315.56,
  "total_turns": 5,
  "models_used": ["qwen2.5:32b-instruct", "llama2:13b"],
  "turns": [
    {
      "turn": 1,
      "timestamp": "2025-01-11T12:05:35.123456",
      "model": "qwen2.5:32b-instruct",
      "prompt": "What is quantum computing?",
      "response": "Quantum computing is...",
      "response_time": 2.34
    }
  ]
}
```

---

### 5. List All Conversations

**Endpoint:** `GET /conversations/list`

**Description:** Returns a list of all completed conversations.

**Request:** No parameters needed

**Response:**
```json
{
  "conversations": [
    {
      "conversation_id": "conv_20250111_120530_a1b2c3d4",
      "start_time": "2025-01-11T12:05:30.123456",
      "end_time": "2025-01-11T12:10:45.678901",
      "duration_seconds": "315.56",
      "total_turns": "5",
      "models_used": "qwen2.5:32b-instruct,llama2:13b"
    },
    {
      "conversation_id": "conv_20250111_140000_x9y8z7w6",
      "start_time": "2025-01-11T14:00:00.000000",
      "end_time": "2025-01-11T14:15:30.000000",
      "duration_seconds": "930.00",
      "total_turns": "12",
      "models_used": "qwen2.5:32b-instruct"
    }
  ]
}
```

**cURL Example:**
```bash
curl http://127.0.0.1:5005/conversations/list | jq
```

---

## Complete Workflow Example

### Python Script Example

```python
import requests
import time

BASE_URL = "http://127.0.0.1:5005"

# 1. Start a new conversation
response = requests.post(f"{BASE_URL}/conversation/new", json={
    "session_id": "my_research_session"
})
conv_id = response.json()["conversation_id"]
print(f"Started conversation: {conv_id}")

# 2. Send multiple messages
messages = [
    "What is quantum computing?",
    "How does it differ from classical computing?",
    "What are the main challenges?",
]

for i, msg in enumerate(messages, 1):
    print(f"\nTurn {i}: {msg}")
    response = requests.post(f"{BASE_URL}/conversation/send", json={
        "conversation_id": conv_id,
        "model": "qwen2.5:32b-instruct",
        "prompt": msg
    })
    data = response.json()
    print(f"Response: {data['response'][:100]}...")
    print(f"Response time: {data['response_time']:.2f}s")
    time.sleep(1)  # Be nice to the API

# 3. End the conversation
response = requests.post(f"{BASE_URL}/conversation/end", json={
    "conversation_id": conv_id
})
print(f"\nConversation ended:")
print(f"  Total turns: {response.json()['total_turns']}")
print(f"  Duration: {response.json()['duration_seconds']:.2f}s")

# 4. Export the conversation
export_url = f"{BASE_URL}/conversation/export/{conv_id}?fmt=md"
response = requests.get(export_url)
with open(f"{conv_id}.md", "wb") as f:
    f.write(response.content)
print(f"\nExported to {conv_id}.md")
```

### Bash Script Example

```bash
#!/bin/bash

BASE_URL="http://127.0.0.1:5005"

# Start conversation
CONV_ID=$(curl -s -X POST "$BASE_URL/conversation/new" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "bash_session"}' | jq -r '.conversation_id')

echo "Started conversation: $CONV_ID"

# Send message 1
echo -e "\n=== Turn 1 ==="
curl -s -X POST "$BASE_URL/conversation/send" \
  -H "Content-Type: application/json" \
  -d "{
    \"conversation_id\": \"$CONV_ID\",
    \"model\": \"qwen2.5:32b-instruct\",
    \"prompt\": \"What is machine learning?\"
  }" | jq -r '.response'

# Send message 2
echo -e "\n=== Turn 2 ==="
curl -s -X POST "$BASE_URL/conversation/send" \
  -H "Content-Type: application/json" \
  -d "{
    \"conversation_id\": \"$CONV_ID\",
    \"model\": \"qwen2.5:32b-instruct\",
    \"prompt\": \"Give me a simple example\"
  }" | jq -r '.response'

# End conversation
echo -e "\n=== Ending Conversation ==="
curl -s -X POST "$BASE_URL/conversation/end" \
  -H "Content-Type: application/json" \
  -d "{\"conversation_id\": \"$CONV_ID\"}" | jq

# Export
echo -e "\n=== Exporting ==="
curl -O "$BASE_URL/conversation/export/$CONV_ID?fmt=md"
echo "Saved to ${CONV_ID}.md"
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "missing prompt"
}
```

**Common causes:**
- Missing required fields
- Invalid conversation_id format

### 404 Not Found
```json
{
  "error": "conversation not found"
}
```

**Common causes:**
- Conversation doesn't exist
- Conversation was never ended (still in memory only)
- Server was restarted (active conversations lost)

### 500 Internal Server Error
```json
{
  "error": "ollama chat failed: Connection refused"
}
```

**Common causes:**
- Ollama not running
- Model not available
- Network issues

---

## Notes

### Session vs Conversation
- **Session ID**: Client-side identifier (stored in localStorage in UI)
- **Conversation ID**: Server-side unique identifier for each conversation
- Multiple conversations can belong to one session

### Conversation Lifecycle
1. **Created** → `POST /conversation/new`
2. **Active** → Multiple `POST /conversation/send` calls
3. **Ended** → `POST /conversation/end` (saves to disk, removes from memory)

### Important
- **Always end conversations** to persist them
- Active conversations are lost on server restart
- Use `/conversations/list` to see only *ended* conversations

---

## Data Persistence

### Automatic Saves (per turn):
- `turns.csv` - Appended immediately
- Turn artifacts (JSONL, MD, HTML, DOCX) - Created immediately

### Manual Save (on end):
- `conversations.csv` - Appended on `/conversation/end`
- `conversation.json` - Created on `/conversation/end`

This means individual turns are safe even if you forget to end the conversation, but conversation-level metadata requires explicit ending.

---

## Rate Limiting & Timeouts

### Timeouts:
- Ollama API calls: 600 seconds (10 minutes)
- Adjust in code if needed for very long responses

### Rate Limiting:
- None by default
- Add middleware if deploying publicly

---

## Security Considerations

### Production Deployment:
1. **Authentication**: Add auth middleware
2. **Rate limiting**: Prevent abuse
3. **Input validation**: Sanitize prompts
4. **File access**: Restrict export endpoints
5. **CORS**: Configure appropriately
6. **HTTPS**: Use reverse proxy (nginx, caddy)

**This version is designed for local use only!**
