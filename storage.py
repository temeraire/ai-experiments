"""
Storage module for Local LLM Logger v3
Handles file operations, artifact saving, and exports
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

from markdown import markdown as md_to_html

from config import HAS_PYPANDOC


def slugify(text: str, maxlen: int = 40) -> str:
    """Convert text to filesystem-safe slug"""
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^a-zA-Z0-9-_]", "", text)
    return text[:maxlen] or "prompt"


def escape_html(s: str) -> str:
    """Escape HTML special characters"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_turn_artifacts(conv, turn_num: int, model: str,
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


def save_comparison_artifacts(conv, turn_num: int, prompt: str, results: list) -> Dict[str, str]:
    """Save artifacts for a comparison turn with multiple model results"""
    slug = slugify(prompt.split("\n", 1)[0])
    turn_dir = conv.conv_dir / f"turn_{turn_num:03d}_comparison_{slug}"
    turn_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()

    # JSONL - save all results
    rec = {
        "conversation_id": conv.id,
        "turn_number": turn_num,
        "timestamp": now.isoformat(),
        "type": "comparison",
        "prompt": prompt,
        "results": results,
    }
    jsonl_path = turn_dir / "comparison.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Markdown - side-by-side comparison
    md_parts = [f"# Turn {turn_num} - Model Comparison\n\n## Prompt\n\n{prompt}\n\n---\n\n"]
    for result in results:
        md_parts.append(f"## {result['model']}\n\n")
        if result.get('error'):
            md_parts.append(f"**ERROR:** {result['error']}\n\n")
        else:
            md_parts.append(f"{result['response']}\n\n")
        md_parts.append(f"*Response time: {result['response_time']:.2f}s*\n\n---\n\n")

    md_path = turn_dir / "comparison.md"
    with md_path.open("w", encoding="utf-8") as mf:
        mf.write("".join(md_parts))

    # HTML - nice comparison layout
    html_results = []
    for result in results:
        if result.get('error'):
            html_results.append(f"""
<div class="model-result error">
  <h3>{escape_html(result['model'])}</h3>
  <p class="error-message">ERROR: {escape_html(result['error'])}</p>
  <p class="timing">Response time: {result['response_time']:.2f}s</p>
</div>
""")
        else:
            response_html = md_to_html(result['response'])
            html_results.append(f"""
<div class="model-result">
  <h3>{escape_html(result['model'])}</h3>
  <div class="response">{response_html}</div>
  <p class="timing">Response time: {result['response_time']:.2f}s</p>
</div>
""")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Turn {turn_num} - Model Comparison</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;line-height:1.6}}
pre,code{{background:#f5f5f5;padding:.2rem .4rem;border-radius:4px}}
pre{{overflow-x:auto;padding:1rem}}
h1,h2,h3{{line-height:1.25}}
.prompt{{background:#f9f9f9;padding:1rem;border-radius:8px;margin:1rem 0}}
.comparison-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:1rem;margin:2rem 0}}
.model-result{{border:1px solid #ddd;padding:1rem;border-radius:8px;background:#fff}}
.model-result.error{{border-color:#e74c3c;background:#fff5f5}}
.error-message{{color:#e74c3c;font-weight:bold}}
.timing{{color:#666;font-size:0.9em;font-style:italic;margin-top:0.5rem}}
.response{{margin:1rem 0}}
</style>
</head>
<body>
<h1>Turn {turn_num} - Model Comparison</h1>
<h2>Prompt</h2>
<div class="prompt"><pre>{escape_html(prompt)}</pre></div>
<h2>Model Responses</h2>
<div class="comparison-grid">
{"".join(html_results)}
</div>
</body>
</html>"""

    html_path = turn_dir / "comparison.html"
    with html_path.open("w", encoding="utf-8") as hf:
        hf.write(html_doc)

    # Optional DOCX
    docx_out = ""
    if HAS_PYPANDOC:
        try:
            from pypandoc import convert_text
            docx_path = turn_dir / "comparison.docx"
            convert_text("".join(md_parts), "docx", format="md", outputfile=str(docx_path))
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


def export_conversation_to_markdown(conv_id: str, summary: Dict, conv_dir: Path) -> Path:
    """Export conversation to markdown file"""
    md_parts = [f"# Conversation {conv_id}\n"]
    md_parts.append(f"Started: {summary['start_time']}\n")
    md_parts.append(f"Total turns: {summary['total_turns']}\n")
    md_parts.append(f"Models used: {', '.join(summary['models_used'])}\n\n---\n\n")

    for turn in summary["turns"]:
        if turn.get('type') == 'comparison':
            # Handle comparison turns
            md_parts.append(f"## Turn {turn['turn']} - Model Comparison\n\n")
            md_parts.append(f"**Prompt:**\n\n{turn['prompt']}\n\n")
            for result in turn['results']:
                md_parts.append(f"### {result['model']}\n\n")
                if result.get('error'):
                    md_parts.append(f"**ERROR:** {result['error']}\n\n")
                else:
                    md_parts.append(f"{result['response']}\n\n")
                md_parts.append(f"*Response time: {result['response_time']:.2f}s*\n\n")
            md_parts.append("---\n\n")
        else:
            # Handle regular turns
            md_parts.append(f"## Turn {turn['turn']}\n\n")
            md_parts.append(f"**Model:** {turn['model']}\n\n")
            md_parts.append(f"**Prompt:**\n\n{turn['prompt']}\n\n")
            md_parts.append(f"**Response:**\n\n{turn['response']}\n\n")
            md_parts.append(f"*Response time: {turn['response_time']:.2f}s*\n\n---\n\n")

    md_file = conv_dir / f"{conv_id}.md"
    md_file.write_text("".join(md_parts))
    return md_file


def export_conversation_to_docx(conv_id: str, summary: Dict, conv_dir: Path) -> Path:
    """Export conversation to DOCX file"""
    if not HAS_PYPANDOC:
        raise RuntimeError("pypandoc not available")

    # Create markdown first
    md_parts = [f"# Conversation {conv_id}\n"]
    for turn in summary["turns"]:
        if turn.get('type') == 'comparison':
            # Handle comparison turns
            md_parts.append(f"## Turn {turn['turn']} - Model Comparison\n\n")
            md_parts.append(f"**Prompt:**\n\n{turn['prompt']}\n\n")
            for result in turn['results']:
                md_parts.append(f"### {result['model']}\n\n")
                if result.get('error'):
                    md_parts.append(f"**ERROR:** {result['error']}\n\n")
                else:
                    md_parts.append(f"{result['response']}\n\n")
                md_parts.append(f"*Response time: {result['response_time']:.2f}s*\n\n")
        else:
            # Handle regular turns
            md_parts.append(f"## Turn {turn['turn']}\n\n")
            md_parts.append(f"**Prompt:**\n\n{turn['prompt']}\n\n")
            md_parts.append(f"**Response:**\n\n{turn['response']}\n\n")

    from pypandoc import convert_text
    docx_file = conv_dir / f"{conv_id}.docx"
    convert_text("".join(md_parts), "docx", format="md", outputfile=str(docx_file))
    return docx_file
