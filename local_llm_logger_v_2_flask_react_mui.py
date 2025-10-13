#!/usr/bin/env python3
"""
Local LLM Logger + Auto-Clear (+ Optional Context N) — Flask + Ollama + React/MUI UI

What's new vs. v1:
- React + Material UI frontend (served from "/"): cleaner UX.
- Keep-context toggle: choose N prior turns to include; when N>0, app will
  *include* the last N turns and then **auto-clear** once it reaches N (so next
  turn starts fresh). N=0 = always stateless.
- "Copy HTML" button (copies rendered HTML for easy paste into Word).
- Batch export endpoint: bundle a whole day's turns into one file.
  • /export?date=YYYY-MM-DD&fmt=docx|pdf|md|zip
  • Requires Pandoc for docx/pdf. Falls back to .md/zip if Pandoc isn't present.
- Same per-turn artifacts (JSONL, MD, HTML, optional DOCX), plus CSV log.

Quick start (macOS):
1) Ensure Ollama is running, model pulled (e.g. `ollama pull qwen2.5:32b-instruct`).
2) (Optional) `python3 -m venv .venv && source .venv/bin/activate`
3) `pip install flask requests markdown pypandoc python-dateutil`
   Optional binaries: `brew install pandoc` (and a TeX engine for PDF if you want /export fmt=pdf)
4) `python app.py` → open http://127.0.0.1:5005/

"""
from __future__ import annotations
import os
import csv
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from flask import Flask, request, jsonify, Response, send_file
from markdown import markdown as md_to_html

try:
    import pypandoc  # type: ignore
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
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CSV_LOG = LOGS_DIR / "log.csv"

# Ensure CSV exists with header
if not CSV_LOG.exists():
    with CSV_LOG.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date_iso", "time_local", "model", "prompt", "result",
            "jsonl_path", "md_path", "html_path", "docx_path_optional"
        ])

# In-memory rolling histories by session_id
HISTORIES: Dict[str, List[Dict[str, str]]] = {}
MAX_TOTAL_HISTORY = 64  # hard cap per session for safety

app = Flask(__name__)

# --------------------------
# Utilities
# --------------------------

def slugify(text: str, maxlen: int = 40) -> str:
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^a-zA-Z0-9-_]", "", text)
    return text[:maxlen] or "prompt"


def ensure_session_dirs(now: datetime, slug: str) -> Path:
    date_dir = SESSIONS_DIR / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    turn_dir = date_dir / f"{now.strftime('%H%M%S')}_{slug}"
    turn_dir.mkdir(parents=True, exist_ok=True)
    return turn_dir


def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_artifacts(now: datetime, model: str, prompt: str, result_md: str) -> Dict[str, str]:
    slug = slugify(prompt.split("\n", 1)[0])
    turn_dir = ensure_session_dirs(now, slug)

    # JSONL
    rec = {
        "timestamp": now.isoformat(),
        "model": model,
        "prompt": prompt,
        "result_markdown": result_md,
    }
    jsonl_path = turn_dir / "record.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Markdown (combined)
    md_path = turn_dir / "result.md"
    with md_path.open("w", encoding="utf-8") as mf:
        mf.write(f"# Prompt\n\n{prompt}\n\n---\n\n# Response\n\n{result_md}\n")

    # HTML (from response markdown)
    html = md_to_html(result_md)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LLM Response</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.6}}pre,code{{background:#f5f5f5;padding:.2rem .4rem;border-radius:4px}}pre{{overflow-x:auto;padding:1rem}}h1,h2,h3{{line-height:1.25}}</style>
</head>
<body>
<h1>Prompt</h1>
<pre>{escape_html(prompt)}</pre>
<hr/>
<h1>Response</h1>
{html}
</body>
</html>"""
    html_path = turn_dir / "result.html"
    with html_path.open("w", encoding="utf-8") as hf:
        hf.write(html_doc)

    # Optional DOCX via Pandoc
    docx_out = ""
    if HAS_PYPANDOC:
        try:
            from pypandoc import convert_text  # type: ignore
            docx_path = turn_dir / "result.docx"
            combined_md = f"# Prompt\n\n````\n{prompt}\n````\n\n---\n\n# Response\n\n{result_md}\n"
            convert_text(combined_md, "docx", format="md", outputfile=str(docx_path))
            docx_out = str(docx_path)
        except Exception:
            docx_out = ""

    # Append to CSV
    with CSV_LOG.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            now.date().isoformat(), now.strftime("%H:%M:%S"), model, prompt, result_md,
            str(jsonl_path), str(md_path), str(html_path), docx_out
        ])

    return {
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
        "html_path": str(html_path),
        "docx_path": docx_out,
        "turn_dir": str(turn_dir),
    }

# --------------------------
# Ollama clients
# --------------------------

def call_generate(model: str, prompt: str, extra: Optional[Dict[str, Any]] = None) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if extra:
        payload.update(extra)
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")


def call_chat(model: str, messages: List[Dict[str, str]], extra: Optional[Dict[str, Any]] = None) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if extra:
        payload.update(extra)
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")

# --------------------------
# Frontend (React + MUI served inline)
# --------------------------

@app.get("/")
def index() -> Response:
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Local LLM Logger</title>
<!-- React & MUI via CDN -->
<script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script crossorigin src="https://unpkg.com/@mui/material@5.15.14/umd/material-ui.development.js"></script>
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Inter:300,400,500,700&display=swap"/>
<style>body{{margin:0;background:#fafafa}} .container{{max-width:980px;margin:32px auto;padding:0 16px}}</style>
</head>
<body>
<div id="root"></div>
<script>
const e = React.createElement;
const {{useState, useEffect}} = React;
const {{Container, TextField, Button, Paper, Typography, Stack, Divider, Alert, Snackbar, FormControl, InputLabel, Select, MenuItem, Box}} = MaterialUI;

function App() {{
  const [model, setModel] = useState('{escape_html(DEFAULT_MODEL)}');
  const [prompt, setPrompt] = useState('');
  const [keepLast, setKeepLast] = useState(0); // 0 = stateless
  const [sessionId, setSessionId] = useState(localStorage.getItem('session_id') || '');
  const [status, setStatus] = useState('');
  const [respMd, setRespMd] = useState('');
  const [respHtml, setRespHtml] = useState('');
  const [paths, setPaths] = useState({{}});
  const [snack, setSnack] = useState('');

  useEffect(() => {{
    if (!sessionId) {{
      fetch('/start_session', {{method:'POST'}}).then(r=>r.json()).then(j=>{{
        localStorage.setItem('session_id', j.session_id);
        setSessionId(j.session_id);
      }});
    }}
  }}, []);

  const send = async () => {{
    if (!prompt) {{ setSnack('Please enter a prompt'); return; }}
    setStatus('Running...');
    try {{
      const r = await fetch('/generate', {{
        method:'POST', headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{ model, prompt, keep_last: Number(keepLast), session_id: sessionId }})
      }});
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || ('HTTP '+r.status));
      setRespMd(data.result_markdown || '');
      setRespHtml(data.result_html || '');
      setPaths(data.paths || {{}});
      setStatus(data.note || 'Done.');
      if (data.auto_cleared) setSnack('Context auto-cleared (reached N).');
    }} catch (err) {{
      setStatus('Error'); setSnack(String(err));
    }}
  }};

  const clearSession = async () => {{
    await fetch('/clear_session', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{session_id: sessionId}})}});
    setSnack('Context cleared.');
  }};

  const copyHtml = async () => {{
    try {{ await navigator.clipboard.writeText(respHtml); setSnack('HTML copied to clipboard'); }} catch(e){{ setSnack('Copy failed: '+e); }}
  }};

  return e('div', {{className:'container'}}, [
    e(Paper, {{elevation:1, sx:{{p:3, borderRadius:4}}}}, [
      e(Typography, {{variant:'h4', gutterBottom:true}}, 'Local LLM Logger + Auto-Clear'),
      e(Stack, {{direction:'row', spacing:2, sx:{{mb:2}}}}, [
        e(TextField, {{label:'Model', fullWidth:true, value:model, onChange:(ev)=>setModel(ev.target.value)}}),
        e(FormControl, {{sx:{{width:180}}}}, [
          e(InputLabel, {{id:'kl'}} , 'Keep last N'),
          e(Select, {{labelId:'kl', label:'Keep last N', value:keepLast, onChange:(ev)=>setKeepLast(ev.target.value)}},
            [0,1,2,3,4,5,8,10].map(n=> e(MenuItem, {{key:n, value:n}}, String(n)))
          )
        ])
      ]),
      e(TextField, {{label:'Prompt', multiline:true, minRows:8, fullWidth:true, value:prompt, onChange:(ev)=>setPrompt(ev.target.value)}}),
      e(Stack, {{direction:'row', spacing:2, sx:{{mt:2}}}}, [
        e(Button, {{variant:'contained', onClick:send}}, 'Send (log + clear logic)'),
        e(Button, {{variant:'outlined', onClick:()=>setPrompt('')}}, 'Reset Fields'),
        e(Button, {{variant:'text', onClick:clearSession}}, 'Clear Context Now'),
        e(Button, {{variant:'text', onClick:copyHtml, disabled:!respHtml}}, 'Copy HTML')
      ]),
      status && e(Alert, {{severity: status.startsWith('Error')?'error':'info', sx:{{mt:2}}}}, status),
      respMd && e(Box, {{sx:{{mt:3}}}}, [
        e(MaterialUI.Divider),
        e(Typography, {{variant:'h6', sx:{{mt:2}}}}, 'Response (Markdown)'),
        e('pre', null, respMd),
        e(Typography, {{variant:'h6', sx:{{mt:2}}}}, 'Saved files'),
        e('ul', null, [
          e('li', null, 'JSONL: '+(paths.jsonl_path||'')),
          e('li', null, 'Markdown: '+(paths.md_path||'')),
          e('li', null, 'HTML: '+(paths.html_path||'')),
          e('li', null, 'DOCX (optional): '+(paths.docx_path||'')),
          e('li', null, 'Folder: '+(paths.turn_dir||'')),
        ]),
      ]),
    ]),
    snack && e(MaterialUI.Snackbar, {{open:true, autoHideDuration:2500, onClose:()=>setSnack(''), message:snack}})
  ]);
}}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html")

# --------------------------
# Session helpers / endpoints
# --------------------------

@app.post("/start_session")
def start_session():
    sid = str(uuid.uuid4())
    HISTORIES[sid] = []
    return jsonify({"session_id": sid})


@app.post("/clear_session")
def clear_session():
    data = request.get_json(force=True)
    sid = data.get("session_id")
    if not sid:
        return jsonify({"error": "missing session_id"}), 400
    HISTORIES[sid] = []
    return jsonify({"ok": True})

# --------------------------
# Generate endpoint (supports keep_last N)
# --------------------------

@app.post("/generate")
def generate():
    data = request.get_json(force=True)
    model = (data.get("model") or DEFAULT_MODEL).strip()
    prompt = data.get("prompt", "")
    keep_last = int(data.get("keep_last", 0))
    sid = data.get("session_id") or str(uuid.uuid4())
    HISTORIES.setdefault(sid, [])

    if not prompt:
        return jsonify({"error": "missing prompt"}), 400

    now = datetime.now()

    # Build messages depending on keep_last
    auto_cleared = False
    result = ""

    if keep_last <= 0:
        # Stateless call
        try:
            result = call_generate(model, prompt)
        except Exception as e:
            return jsonify({"error": f"ollama generate failed: {e}"}), 500
        # Do NOT store any history for N=0
        HISTORIES[sid] = []
    else:
        # Rolling chat; keep only last N pairs (user/assistant), then auto-clear after send
        hist = HISTORIES[sid]
        # Enforce safety cap
        if len(hist) > MAX_TOTAL_HISTORY:
            hist[:] = hist[-MAX_TOTAL_HISTORY:]
        messages = hist[-(2*keep_last):]  # approx N turns (user+assistant)
        messages = messages + [{"role": "user", "content": prompt}]
        try:
            result = call_chat(model, messages)
        except Exception as e:
            return jsonify({"error": f"ollama chat failed: {e}"}), 500
        # Update history with this turn
        hist.extend([{ "role": "user", "content": prompt }, { "role": "assistant", "content": result }])
        # If we hit/exceeded N turns, clear
        # Count turns as assistant messages
        turns = sum(1 for m in hist if m["role"] == "assistant")
        if turns >= keep_last:
            HISTORIES[sid] = []
            auto_cleared = True

    # Save artifacts
    paths = save_artifacts(now, model, prompt, result)

    return jsonify({
        "timestamp": now.isoformat(),
        "model": model,
        "result_markdown": result,
        "result_html": md_to_html(result),
        "paths": paths,
        "note": "No prior messages persisted" if keep_last<=0 else f"Kept last {keep_last} turns (then clear)",
        "auto_cleared": auto_cleared,
        "session_id": sid,
    })

# --------------------------
# Batch Export
# --------------------------

@app.get("/export")
def export_day():
    date_str = request.args.get("date")  # YYYY-MM-DD
    fmt = (request.args.get("fmt") or "docx").lower()  # docx|pdf|md|zip
    if not date_str:
        return jsonify({"error": "missing 'date' (YYYY-MM-DD)"}), 400

    day_dir = SESSIONS_DIR / date_str
    if not day_dir.exists() or not day_dir.is_dir():
        return jsonify({"error": f"no records for {date_str}"}), 404

    # Aggregate markdown in chronological folder order
    parts: List[str] = []
    for p in sorted(day_dir.iterdir()):
        md_file = p / "result.md"
        if md_file.exists():
            parts.append(md_file.read_text(encoding="utf-8"))
    if not parts:
        return jsonify({"error": "no markdown files found"}), 404

    combined_md = f"# Session Export — {date_str}\n\n" + "\n\n---\n\n".join(parts)

    # Prepare output
    out_dir = day_dir
    if fmt == "md":
        out_path = out_dir / f"export_{date_str}.md"
        out_path.write_text(combined_md, encoding="utf-8")
        return send_file(out_path, as_attachment=True)

    if fmt in {"docx", "pdf"}:
        if not HAS_PYPANDOC:
            return jsonify({"error": "Pandoc not available; install pandoc (and LaTeX for PDF)."}), 500
        try:
            from pypandoc import convert_text  # type: ignore
            out_path = out_dir / f"export_{date_str}.{fmt}"
            convert_text(combined_md, fmt, format="md", outputfile=str(out_path))
            return send_file(out_path, as_attachment=True)
        except Exception as e:
            return jsonify({"error": f"pandoc conversion failed: {e}"}), 500

    if fmt == "zip":
        import shutil
        out_path = out_dir / f"export_{date_str}.zip"
        if out_path.exists():
            out_path.unlink()
        shutil.make_archive(str(out_path.with_suffix("")), 'zip', day_dir)
        return send_file(out_path, as_attachment=True)

    return jsonify({"error": "unsupported fmt; use docx|pdf|md|zip"}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=APP_PORT, debug=False)