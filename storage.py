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


def export_conversation_to_markdown(conv_id: str, summary: Dict, conv_dir: Path) -> Path:
    """Export conversation to markdown file"""
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
    return md_file


def export_conversation_to_docx(conv_id: str, summary: Dict, conv_dir: Path) -> Path:
    """Export conversation to DOCX file"""
    if not HAS_PYPANDOC:
        raise RuntimeError("pypandoc not available")

    # Create markdown first
    md_parts = [f"# Conversation {conv_id}\n"]
    for turn in summary["turns"]:
        md_parts.append(f"## Turn {turn['turn']}\n\n")
        md_parts.append(f"**Prompt:**\n\n{turn['prompt']}\n\n")
        md_parts.append(f"**Response:**\n\n{turn['response']}\n\n")

    from pypandoc import convert_text
    docx_file = conv_dir / f"{conv_id}.docx"
    convert_text("".join(md_parts), "docx", format="md", outputfile=str(docx_file))
    return docx_file
