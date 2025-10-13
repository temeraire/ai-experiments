# Local LLM Logger v3 - Continuous Chat with Comprehensive Tracking
# Flask + Ollama + React/MUI UI

# New in v3:
# - Continuous conversations like ChatGPT
# - Conversation-level tracking (ID, start/end times, duration, models used)
# - Turn-level tracking (turn number, response times)
# - Chat history display in UI
# - 'New Conversation' vs 'Continue' buttons
# - Export individual conversations or all data
# - Handle model switching mid-conversation
# - All original logging preserved (CSV, JSONL, MD, HTML, DOCX)

# Quick start:
# 1) Ensure Ollama is running with a model (e.g. `ollama pull qwen2.5:32b-instruct`)
# 2) `pip install flask requests markdown pypandoc python-dateutil`
# 3) `python local_llm_logger_v3_continuous_chat.py`
# 4) Open http://127.0.0.1:5005/

from __future__ import annotations
import os
import csv
import json
import re
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from flask import Flask, request, jsonify, Response, send_file
from markdown import markdown as md_to_html

try:
    import pypandoc
    HAS_PYPANDOC = True
except Exception:
    HAS_PYPANDOC = False

# --------------------------
# Configuration
# --------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b-instruct")
APP_PORT = int(os.environ.get("PORT", 5005))

BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = BASE_DIR / "sessions"
LOGS_DIR = BASE_DIR / "logs"
CONVERSATIONS_DIR = BASE_DIR / "conversations"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

# CSV logs
CONVERSATIONS_CSV = LOGS_DIR / "conversations.csv"
TURNS_CSV = LOGS_DIR / "turns.csv"

# Initialize conversations CSV
if not CONVERSATIONS_CSV.exists():
    with CONVERSATIONS_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "conversation_id", "start_time", "end_time", "duration_seconds",
            "total_turns", "models_used", "conversation_dir"
        ])

# Initialize turns CSV
if not TURNS_CSV.exists():
    with TURNS_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "conversation_id", "turn_number", "timestamp", "model",
            "prompt", "response", "response_time_seconds",
            "jsonl_path", "md_path", "html_path", "docx_path"
        ])

# In-memory conversation storage
class Conversation:
    def __init__(self, conv_id: str):
        self.id = conv_id
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.turns: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, str]] = []  # For chat API
        self.models_used: set = set()
        self.conv_dir = CONVERSATIONS_DIR / conv_id
        self.conv_dir.mkdir(parents=True, exist_ok=True)
    
    def add_turn(self, model: str, prompt: str, response: str, 
                 response_time: float, paths: Dict[str, str]):
        turn_num = len(self.turns) + 1
        turn = {
            "turn_number": turn_num,
            "timestamp": datetime.now(),
            "model": model,
            "prompt": prompt,
            "response": response,
            "response_time": response_time,
            "paths": paths
        }
        self.turns.append(turn)
        self.models_used.add(model)
        
        # Add to messages for chat continuity
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": response})
        
        # Log turn to CSV
        self._log_turn(turn)
        
    def _log_turn(self, turn: Dict[str, Any]):
        with TURNS_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.id,
                turn["turn_number"],
                turn["timestamp"].isoformat(),
                turn["model"],
                turn["prompt"],
                turn["response"],
                f"{turn['response_time']:.2f}",
                turn["paths"].get("jsonl_path", ""),
                turn["paths"].get("md_path", ""),
                turn["paths"].get("html_path", ""),
                turn["paths"].get("docx_path", "")
            ])
    
    def end_conversation(self):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Log conversation to CSV
        with CONVERSATIONS_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.id,
                self.start_time.isoformat(),
                self.end_time.isoformat(),
                f"{duration:.2f}",
                len(self.turns),
                ",".join(sorted(self.models_used)),
                str(self.conv_dir)
            ])
        
        # Save conversation summary
        self._save_summary()
    
    def _save_summary(self):
        summary = {
            "conversation_id": self.id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "total_turns": len(self.turns),
            "models_used": list(self.models_used),
            "turns": [
                {
                    "turn": t["turn_number"],
                    "timestamp": t["timestamp"].isoformat(),
                    "model": t["model"],
                    "prompt": t["prompt"],
                    "response": t["response"],
                    "response_time": t["response_time"]
                }
                for t in self.turns
            ]
        }
        
        summary_path = self.conv_dir / "conversation.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def to_dict(self):
        return {
            "id": self.id,
            "start_time": self.start_time.isoformat(),
            "total_turns": len(self.turns),
            "models_used": list(self.models_used),
            "turns": [
                {
                    "turn": t["turn_number"],
                    "timestamp": t["timestamp"].isoformat(),
                    "model": t["model"],
                    "prompt": t["prompt"][:100] + "..." if len(t["prompt"]) > 100 else t["prompt"],
                    "response": t["response"][:200] + "..." if len(t["response"]) > 200 else t["response"]
                }
                for t in self.turns
            ]
        }

# Active conversations by session_id
ACTIVE_CONVERSATIONS: Dict[str, Conversation] = {}

app = Flask(__name__)

# --------------------------
# Utilities
# --------------------------

def slugify(text: str, maxlen: int = 40) -> str:
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^a-zA-Z0-9-_]", "", text)
    return text[:maxlen] or "prompt"


def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_turn_artifacts(conv: Conversation, turn_num: int, model: str, 
                        prompt: str, result_md: str) -> Dict[str, str]:
    """Save artifacts for a single turn within a conversation"""
    slug = slugify(prompt.split("\n", 1)[0])
    turn_dir = conv.conv_dir / f"turn_{turn_num:03d}_{slug}"
    turn_dir.mkdir(parents=True, exist_ok=True)
    
    now = datetime.now()
    
    # JSONL
    rec = {
        "conversation_id": conv.id,
        "turn_number": turn_num,
        "timestamp": now.isoformat(),
        "model": model,
        "prompt": prompt,
        "result_markdown": result_md,
    }
    jsonl_path = turn_dir / "turn.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    # Markdown
    md_path = turn_dir / "turn.md"
    with md_path.open("w", encoding="utf-8") as mf:
        mf.write(f"# Turn {turn_num}\n\n## Prompt\n\n{prompt}\n\n---\n\n## Response\n\n{result_md}\n")
    
    # HTML
    html = md_to_html(result_md)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Turn {turn_num}</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.6}}pre,code{{background:#f5f5f5;padding:.2rem .4rem;border-radius:4px}}pre{{overflow-x:auto;padding:1rem}}h1,h2,h3{{line-height:1.25}}</style>
</head>
<body>
<h1>Turn {turn_num}</h1>
<h2>Prompt</h2>
<pre>{escape_html(prompt)}</pre>
<hr/>
<h2>Response</h2>
{html}
</body>
</html>"""
    html_path = turn_dir / "turn.html"
    with html_path.open("w", encoding="utf-8") as hf:
        hf.write(html_doc)
    
    # Optional DOCX
    docx_out = ""
    if HAS_PYPANDOC:
        try:
            from pypandoc import convert_text
            docx_path = turn_dir / "turn.docx"
            combined_md = f"# Turn {turn_num}\n\n## Prompt\n\n````\n{prompt}\n````\n\n---\n\n## Response\n\n{result_md}\n"
            convert_text(combined_md, "docx", format="md", outputfile=str(docx_path))
            docx_out = str(docx_path)
        except Exception:
            pass
    
    return {
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
        "html_path": str(html_path),
        "docx_path": docx_out,
        "turn_dir": str(turn_dir),
    }


def call_chat(model: str, messages: List[Dict[str, str]]) -> str:
    """Call Ollama chat API"""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def get_ollama_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        import subprocess
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

# --------------------------
# Frontend (React + MUI)
# --------------------------

@app.get("/")
def index() -> Response:
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Local LLM Logger v3</title>
<script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script crossorigin src="https://unpkg.com/@mui/material@5.15.14/umd/material-ui.development.js"></script>
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Inter:300,400,500,700&display=swap"/>
<style>
body{{margin:0;background:#fafafa}}
.container{{max-width:1200px;margin:32px auto;padding:0 16px}}
.chat-message{{margin:12px 0;padding:12px;border-radius:8px}}
.user-message{{background:#e3f2fd;text-align:right}}
.assistant-message{{background:#f5f5f5}}
.message-meta{{font-size:0.85em;color:#666;margin-top:4px}}
</style>
</head>
<body>
<div id="root"></div>
<script>
const e = React.createElement;
const {{useState, useEffect, useRef}} = React;
const {{Container, TextField, Button, Paper, Typography, Stack, Divider, Alert, Snackbar, Box, Chip, Card, CardContent, Select, MenuItem, FormControl, InputLabel, IconButton}} = MaterialUI;

function App() {{
  const [sessionId, setSessionId] = useState(localStorage.getItem('session_id') || '');
  const [conversationId, setConversationId] = useState(null);
  const [model, setModel] = useState('{escape_html(DEFAULT_MODEL)}');
  const [availableModels, setAvailableModels] = useState(['{escape_html(DEFAULT_MODEL)}']);
  const [prompt, setPrompt] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [status, setStatus] = useState('');
  const [snack, setSnack] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationInfo, setConversationInfo] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {{
    if (!sessionId) {{
      const newId = 'session_' + Date.now();
      localStorage.setItem('session_id', newId);
      setSessionId(newId);
    }}
    // Load available models
    fetch('/models/list').then(r=>r.json()).then(data=>{{
      if (data.models && data.models.length > 0) {{
        setAvailableModels(data.models);
        if (!model || model === '{escape_html(DEFAULT_MODEL)}') {{
          setModel(data.models[0]);
        }}
      }}
    }}).catch(err=>console.error('Failed to load models:', err));
  }}, []);

  useEffect(() => {{
    chatEndRef.current?.scrollIntoView({{ behavior: 'smooth' }});
  }}, [chatHistory]);

  const startNewConversation = async () => {{
    try {{
      const r = await fetch('/conversation/new', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ session_id: sessionId }})
      }});
      const data = await r.json();
      setConversationId(data.conversation_id);
      setChatHistory([]);
      setStatus('New conversation started');
      setSnack('New conversation started');
    }} catch (err) {{
      setSnack('Error starting conversation: ' + err);
    }}
  }};

  const sendMessage = async () => {{
    if (!prompt.trim()) {{ setSnack('Please enter a message'); return; }}
    if (!conversationId) {{ await startNewConversation(); return; }}
    
    setIsLoading(true);
    setStatus('Generating response...');
    
    // Add user message to UI immediately
    const userMsg = {{ role: 'user', content: prompt, model }};
    setChatHistory(prev => [...prev, userMsg]);
    const currentPrompt = prompt;
    setPrompt('');
    
    try {{
      const r = await fetch('/conversation/send', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          conversation_id: conversationId,
          model,
          prompt: currentPrompt
        }})
      }});
      
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'HTTP ' + r.status);
      
      // Add assistant response
      setChatHistory(prev => [...prev, {{
        role: 'assistant',
        content: data.response,
        model: data.model,
        turn: data.turn_number,
        response_time: data.response_time
      }}]);
      
      setStatus(`Turn ${{data.turn_number}} • ${{data.response_time.toFixed(2)}}s`);
      setConversationInfo(data.conversation_info);
    }} catch (err) {{
      setSnack('Error: ' + err);
      setStatus('Error');
    }} finally {{
      setIsLoading(false);
    }}
  }};

  const endConversation = async () => {{
    if (!conversationId) return;
    try {{
      await fetch('/conversation/end', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ conversation_id: conversationId }})
      }});
      setSnack('Conversation ended and saved');
      setConversationId(null);
      setChatHistory([]);
      setStatus('');
    }} catch (err) {{
      setSnack('Error ending conversation: ' + err);
    }}
  }};

  const handleKeyPress = (ev) => {{
    if (ev.key === 'Enter' && !ev.shiftKey) {{
      ev.preventDefault();
      sendMessage();
    }}
  }};

  const handleFileUpload = async (ev) => {{
    const file = ev.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {{
      const r = await fetch('/upload', {{method: 'POST', body: formData}});
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Upload failed');
      
      // Add file content to prompt with clear formatting
      const fileContent = `\\n\\n--- File: ${{file.name}} ---\\n${{data.content}}\\n--- End of file ---\\n\\n`;
      setPrompt(prev => prev + fileContent);
      setUploadedFiles(prev => [...prev, {{name: file.name, size: data.size}}]);
      setSnack(`Uploaded: ${{file.name}}`);
    }} catch (err) {{
      setSnack('Upload error: ' + err);
    }}
    // Reset input
    ev.target.value = '';
  }};

  return e('div', {{className: 'container'}}, [
    e(Paper, {{elevation: 1, sx: {{p: 3, borderRadius: 4, mb: 2}}}}, [
      e(Stack, {{direction: 'row', justifyContent: 'space-between', alignItems: 'center', mb: 2}}, [
        e(Typography, {{variant: 'h4'}}, 'Local LLM Logger v3'),
        e(Stack, {{direction: 'row', spacing: 1}}, [
          conversationId && e(Chip, {{
            label: `Turn: ${{chatHistory.filter(m => m.role === 'assistant').length}}`,
            color: 'primary',
            size: 'small'
          }}),
          conversationId && e(Chip, {{
            label: model,
            color: 'secondary',
            size: 'small'
          }})
        ])
      ]),
      
      e(Stack, {{direction: 'row', spacing: 2, mb: 2}}, [
        e(FormControl, {{sx: {{minWidth: 300}}}}, [
          e(InputLabel, {{id: 'model-label'}}, 'Model'),
          e(Select, {{
            labelId: 'model-label',
            label: 'Model',
            value: model,
            onChange: (ev) => setModel(ev.target.value),
            size: 'small'
          }}, availableModels.map(m => e(MenuItem, {{key: m, value: m}}, m)))
        ]),
        e(Button, {{
          variant: conversationId ? 'outlined' : 'contained',
          onClick: startNewConversation,
          disabled: isLoading
        }}, 'New Conversation'),
        conversationId && e(Button, {{
          variant: 'outlined',
          color: 'error',
          onClick: endConversation,
          disabled: isLoading
        }}, 'End & Save')
      ]),
      
      status && e(Alert, {{
        severity: status.startsWith('Error') ? 'error' : 'info',
        sx: {{mb: 2}}
      }}, status)
    ]),
    
    conversationId && e(Paper, {{elevation: 1, sx: {{p: 2, mb: 2, maxHeight: '60vh', overflowY: 'auto', borderRadius: 4}}}}, [
      chatHistory.length === 0 && e(Typography, {{color: 'text.secondary', align: 'center'}}, 'Start chatting...'),
      ...chatHistory.map((msg, idx) => 
        e('div', {{
          key: idx,
          className: msg.role === 'user' ? 'chat-message user-message' : 'chat-message assistant-message'
        }}, [
          e(Typography, {{variant: 'body1', sx: {{whiteSpace: 'pre-wrap'}}}}, msg.content),
          e('div', {{className: 'message-meta'}}, [
            msg.role === 'assistant' && msg.turn && `Turn ${{msg.turn}} • `,
            msg.model && `${{msg.model}}`,
            msg.response_time && ` • ${{msg.response_time.toFixed(2)}}s`
          ])
        ])
      ),
      e('div', {{ref: chatEndRef}})
    ]),
    
    conversationId && e(Paper, {{elevation: 1, sx: {{p: 2, borderRadius: 4}}}}, [
      e(Box, {{sx: {{position: 'relative'}}}}, [
        e(TextField, {{
          label: 'Your message',
          multiline: true,
          minRows: 3,
          maxRows: 8,
          fullWidth: true,
          value: prompt,
          onChange: (ev) => setPrompt(ev.target.value),
          onKeyPress: handleKeyPress,
          disabled: isLoading,
          placeholder: 'Type your message... (Shift+Enter for new line, or click + to upload file)'
        }}),
        e('input', {{
          ref: fileInputRef,
          type: 'file',
          style: {{display: 'none'}},
          onChange: handleFileUpload,
          accept: '.txt,.md,.py,.js,.json,.csv,.html,.css,.java,.cpp,.c,.h,.go,.rs,.sh,.yaml,.yml,.xml,.sql,.ipynb,.docx,.pdf'
        }}),
        e(IconButton, {{
          onClick: () => fileInputRef.current?.click(),
          disabled: isLoading,
          sx: {{position: 'absolute', right: 8, top: 8}},
          title: 'Upload file'
        }}, '+')
      ]),
      uploadedFiles.length > 0 && e(Box, {{sx: {{mt: 1, display: 'flex', flexWrap: 'wrap', gap: 1}}}}, 
        uploadedFiles.map((f, i) => e(Chip, {{
          key: i,
          label: `${{f.name}} (${{(f.size/1024).toFixed(1)}}KB)`,
          size: 'small',
          onDelete: () => setUploadedFiles(prev => prev.filter((_, idx) => idx !== i))
        }}))
      ),
      e(Stack, {{direction: 'row', spacing: 2, sx: {{mt: 2}}}}, [
        e(Button, {{
          variant: 'contained',
          onClick: sendMessage,
          disabled: isLoading || !prompt.trim()
        }}, isLoading ? 'Generating...' : 'Send'),
        e(Button, {{
          variant: 'outlined',
          onClick: () => {{ setPrompt(''); setUploadedFiles([]); }},
          disabled: isLoading
        }}, 'Clear')
      ])
    ]),
    
    !conversationId && e(Box, {{sx: {{textAlign: 'center', mt: 4}}}}, [
      e(Typography, {{variant: 'h6', color: 'text.secondary'}}, 'Click "New Conversation" to start'),
      e(Typography, {{variant: 'body2', color: 'text.secondary', mt: 1}}, 
        'All conversations are automatically logged with full tracking')
    ]),
    
    snack && e(Snackbar, {{
      open: true,
      autoHideDuration: 3000,
      onClose: () => setSnack(''),
      message: snack
    }})
  ]);
}}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html")

# --------------------------
# Conversation Endpoints
# --------------------------

@app.post("/conversation/new")
def new_conversation():
    """Start a new conversation"""
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    
    conv_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    conv = Conversation(conv_id)
    ACTIVE_CONVERSATIONS[session_id] = conv
    
    return jsonify({
        "conversation_id": conv_id,
        "start_time": conv.start_time.isoformat()
    })


@app.post("/conversation/send")
def send_message():
    """Send a message in an active conversation"""
    data = request.get_json(force=True)
    conv_id = data.get("conversation_id")
    model = data.get("model", DEFAULT_MODEL)
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "missing prompt"}), 400
    
    # Find conversation
    conv = None
    for sid, c in ACTIVE_CONVERSATIONS.items():
        if c.id == conv_id:
            conv = c
            break
    
    if not conv:
        return jsonify({"error": "conversation not found"}), 404
    
    # Call LLM
    start_time = time.time()
    try:
        result = call_chat(model, conv.messages + [{"role": "user", "content": prompt}])
        response_time = time.time() - start_time
    except Exception as e:
        return jsonify({"error": f"ollama chat failed: {e}"}), 500
    
    # Save artifacts
    turn_num = len(conv.turns) + 1
    paths = save_turn_artifacts(conv, turn_num, model, prompt, result)
    
    # Add turn to conversation
    conv.add_turn(model, prompt, result, response_time, paths)
    
    return jsonify({
        "conversation_id": conv.id,
        "turn_number": turn_num,
        "model": model,
        "response": result,
        "response_time": response_time,
        "paths": paths,
        "conversation_info": {
            "total_turns": len(conv.turns),
            "duration": (datetime.now() - conv.start_time).total_seconds(),
            "models_used": list(conv.models_used)
        }
    })


@app.post("/conversation/end")
def end_conversation():
    """End and save a conversation"""
    data = request.get_json(force=True)
    conv_id = data.get("conversation_id")
    
    # Find and remove conversation
    conv = None
    session_to_remove = None
    for sid, c in ACTIVE_CONVERSATIONS.items():
        if c.id == conv_id:
            conv = c
            session_to_remove = sid
            break
    
    if not conv:
        return jsonify({"error": "conversation not found"}), 404
    
    conv.end_conversation()
    
    if session_to_remove:
        del ACTIVE_CONVERSATIONS[session_to_remove]
    
    return jsonify({
        "conversation_id": conv.id,
        "total_turns": len(conv.turns),
        "duration_seconds": (conv.end_time - conv.start_time).total_seconds(),
        "conversation_dir": str(conv.conv_dir)
    })


@app.get("/conversation/export/<conv_id>")
def export_conversation(conv_id: str):
    """Export a specific conversation"""
    fmt = request.args.get("fmt", "json").lower()
    
    conv_dir = CONVERSATIONS_DIR / conv_id
    if not conv_dir.exists():
        return jsonify({"error": "conversation not found"}), 404
    
    summary_file = conv_dir / "conversation.json"
    if not summary_file.exists():
        return jsonify({"error": "conversation summary not found"}), 404
    
    if fmt == "json":
        return send_file(summary_file, as_attachment=True)
    
    # For other formats, compile all turns
    with summary_file.open() as f:
        summary = json.load(f)
    
    if fmt == "md":
        md_parts = [f"# Conversation {conv_id}\n"]
        md_parts.append(f"Started: {summary['start_time']}\n")
        md_parts.append(f"Total turns: {summary['total_turns']}\n")
        md_parts.append(f"Models used: {', '.join(summary['models_used'])}\n\n---\n\n")
        
        for turn in summary["turns"]:
            md_parts.append(f"## Turn {turn['turn']}\n\n")
            md_parts.append(f"**Model:** {turn['model']}\n\n")
            md_parts.append(f"**Prompt:**\n\n{turn['prompt']}\n\n")
            md_parts.append(f"**Response:**\n\n{turn['response']}\n\n")
            md_parts.append(f"*Response time: {turn['response_time']:.2f}s*\n\n---\n\n")
        
        md_file = conv_dir / f"{conv_id}.md"
        md_file.write_text("".join(md_parts))
        return send_file(md_file, as_attachment=True)
    
    if fmt == "docx" and HAS_PYPANDOC:
        # Create markdown first
        md_parts = [f"# Conversation {conv_id}\n"]
        for turn in summary["turns"]:
            md_parts.append(f"## Turn {turn['turn']}\n\n")
            md_parts.append(f"**Prompt:**\n\n{turn['prompt']}\n\n")
            md_parts.append(f"**Response:**\n\n{turn['response']}\n\n")
        
        try:
            from pypandoc import convert_text
            docx_file = conv_dir / f"{conv_id}.docx"
            convert_text("".join(md_parts), "docx", format="md", outputfile=str(docx_file))
            return send_file(docx_file, as_attachment=True)
        except Exception as e:
            return jsonify({"error": f"pandoc conversion failed: {e}"}), 500
    
    return jsonify({"error": "unsupported format; use json, md, or docx"}), 400


@app.get("/conversations/list")
def list_conversations():
    """List all conversations"""
    conversations = []
    
    if CONVERSATIONS_CSV.exists():
        with CONVERSATIONS_CSV.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                conversations.append({
                    "conversation_id": row["conversation_id"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "duration_seconds": row["duration_seconds"],
                    "total_turns": row["total_turns"],
                    "models_used": row["models_used"]
                })
    
    return jsonify({"conversations": conversations})


@app.get("/models/list")
def list_models():
    """List available Ollama models"""
    models = get_ollama_models()
    return jsonify({"models": models})


@app.post("/upload")
def upload_file():
    """Handle file upload and return content"""
    if 'file' not in request.files:
        return jsonify({"error": "no file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "no file selected"}), 400
    
    try:
        filename = file.filename.lower()
        content = file.read()
        
        # Handle different file types
        if filename.endswith('.pdf'):
            # Extract text from PDF
            try:
                import PyPDF2
                from io import BytesIO
                pdf_reader = PyPDF2.PdfReader(BytesIO(content))
                text_content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page.extract_text()
                if not text_content.strip():
                    return jsonify({"error": "PDF appears to be empty or contains only images"}), 400
            except ImportError:
                return jsonify({"error": "PyPDF2 not installed. Run: pip install PyPDF2"}), 500
            except Exception as e:
                return jsonify({"error": f"Failed to read PDF: {e}"}), 400
                
        elif filename.endswith('.docx'):
            # Extract text from DOCX
            try:
                import docx
                from io import BytesIO
                doc = docx.Document(BytesIO(content))
                text_content = ""
                for para in doc.paragraphs:
                    text_content += para.text + "\n"
                if not text_content.strip():
                    return jsonify({"error": "DOCX appears to be empty"}), 400
            except ImportError:
                return jsonify({"error": "python-docx not installed. Run: pip install python-docx"}), 500
            except Exception as e:
                return jsonify({"error": f"Failed to read DOCX: {e}"}), 400
                
        elif filename.endswith('.ipynb'):
            # Parse Jupyter notebook
            try:
                import json
                notebook = json.loads(content.decode('utf-8'))
                text_content = f"# Jupyter Notebook: {file.filename}\n\n"
                
                for i, cell in enumerate(notebook.get('cells', [])):
                    cell_type = cell.get('cell_type', 'unknown')
                    source = cell.get('source', [])
                    
                    # Handle source as list or string
                    if isinstance(source, list):
                        source_text = ''.join(source)
                    else:
                        source_text = source
                    
                    if cell_type == 'markdown':
                        text_content += f"\n## Markdown Cell {i+1}\n{source_text}\n"
                    elif cell_type == 'code':
                        text_content += f"\n## Code Cell {i+1}\n```python\n{source_text}\n```\n"
                        # Include outputs if available
                        outputs = cell.get('outputs', [])
                        if outputs:
                            text_content += "\n### Output:\n"
                            for output in outputs:
                                if 'text' in output:
                                    out_text = output['text']
                                    if isinstance(out_text, list):
                                        out_text = ''.join(out_text)
                                    text_content += f"```\n{out_text}\n```\n"
                                elif 'data' in output and 'text/plain' in output['data']:
                                    out_text = output['data']['text/plain']
                                    if isinstance(out_text, list):
                                        out_text = ''.join(out_text)
                                    text_content += f"```\n{out_text}\n```\n"
                
                if not text_content.strip():
                    return jsonify({"error": "Notebook appears to be empty"}), 400
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid Jupyter notebook format"}), 400
            except Exception as e:
                return jsonify({"error": f"Failed to read notebook: {e}"}), 400
        else:
            # Handle as regular text file
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_content = content.decode('latin-1')
                except:
                    return jsonify({"error": "unable to read file as text"}), 400
        
        return jsonify({
            "filename": file.filename,
            "content": text_content,
            "size": len(content)
        })
    except Exception as e:
        return jsonify({"error": f"failed to read file: {e}"}), 500


if __name__ == "__main__":
    print(f"Starting Local LLM Logger v3 on http://127.0.0.1:{APP_PORT}")
    print(f"Ollama host: {OLLAMA_HOST}")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Logs directory: {LOGS_DIR}")
    print(f"Conversations directory: {CONVERSATIONS_DIR}")
    app.run(host="127.0.0.1", port=APP_PORT, debug=False)
