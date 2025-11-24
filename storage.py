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

from config import HAS_PYPANDOC, get_model_abbrev


def slugify(text: str, maxlen: int = 40) -> str:
    """Convert text to filesystem-safe slug"""
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^a-zA-Z0-9-_]", "", text)
    return text[:maxlen] or "prompt"


def escape_html(s: str) -> str:
    """Escape HTML special characters"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_turn_artifacts(conv, turn_num: int, model: str,
                        prompt: str, result_md: str, summary: str = "") -> Dict[str, str]:
    """Save artifacts for a single turn within a conversation"""
    slug = slugify(prompt.split("\n", 1)[0])
    model_abbrev = get_model_abbrev(model)
    # Include model abbreviation in folder name
    turn_dir = conv.conv_dir / f"turn_{turn_num:03d}_{model_abbrev}_{slug}"
    turn_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()

    # JSONL - kept for internal use (metadata)
    rec = {
        "conversation_id": conv.id,
        "turn_number": turn_num,
        "timestamp": now.isoformat(),
        "model": model,
        "prompt": prompt,
        "result_markdown": result_md,
        "summary": summary,
    }
    jsonl_path = turn_dir / "turn.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Markdown - COMMENTED OUT per user request
    # md_path = turn_dir / "turn.md"
    # with md_path.open("w", encoding="utf-8") as mf:
    #     mf.write(f"# Turn {turn_num}\n\n## Prompt\n\n{prompt}\n\n---\n\n## Response\n\n{result_md}\n")
    md_path = ""

    # HTML - COMMENTED OUT per user request
    # html = md_to_html(result_md)
    # html_doc = f"""<!DOCTYPE html>
    # ... (HTML template removed)
    # """
    # html_path = turn_dir / "turn.html"
    # with html_path.open("w", encoding="utf-8") as hf:
    #     hf.write(html_doc)
    html_path = ""

    # DOCX - Primary output format with model name in filename
    docx_out = ""
    if HAS_PYPANDOC:
        try:
            from pypandoc import convert_text
            # New naming: turn{num}_{model_abbrev}.docx
            docx_path = turn_dir / f"turn{turn_num}_{model_abbrev}.docx"
            # Include summary at the top if available
            summary_section = f"## Summary\n\n{summary}\n\n---\n\n" if summary else ""
            combined_md = f"# Turn {turn_num}\n\n{summary_section}## Prompt\n\n````\n{prompt}\n````\n\n---\n\n## Response\n\n{result_md}\n"
            convert_text(combined_md, "docx", format="md", outputfile=str(docx_path))
            docx_out = str(docx_path)
        except Exception:
            pass

    return {
        "jsonl_path": str(jsonl_path),
        "md_path": md_path,
        "html_path": html_path,
        "docx_path": docx_out,
        "turn_dir": str(turn_dir),
    }


def save_comparison_artifacts(conv, turn_num: int, prompt: str, results: list) -> Dict[str, str]:
    """Save artifacts for a comparison turn with multiple model results"""
    slug = slugify(prompt.split("\n", 1)[0])
    # Get model abbreviations for all models in comparison
    model_abbrevs = [get_model_abbrev(r['model']) for r in results if not r.get('error')]
    models_str = "-".join(model_abbrevs) if model_abbrevs else "comparison"
    turn_dir = conv.conv_dir / f"turn_{turn_num:03d}_{models_str}_{slug}"
    turn_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()

    # JSONL - kept for internal use (metadata)
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

    # Build markdown content (used for DOCX conversion)
    md_parts = [f"# Turn {turn_num} - Model Comparison\n\n## Prompt\n\n{prompt}\n\n---\n\n"]
    for result in results:
        md_parts.append(f"## {result['model']}\n\n")
        # Add summary at the top of each model's section if available
        if result.get('summary'):
            md_parts.append(f"### Summary\n\n{result['summary']}\n\n---\n\n")
        if result.get('error'):
            md_parts.append(f"**ERROR:** {result['error']}\n\n")
        else:
            md_parts.append(f"{result['response']}\n\n")
        md_parts.append(f"*Response time: {result['response_time']:.2f}s*\n\n---\n\n")

    # Markdown file - COMMENTED OUT per user request
    # md_path = turn_dir / "comparison.md"
    # with md_path.open("w", encoding="utf-8") as mf:
    #     mf.write("".join(md_parts))
    md_path = ""

    # HTML - COMMENTED OUT per user request
    # (HTML generation code removed for brevity)
    html_path = ""

    # DOCX - Primary output format with model names in filename
    docx_out = ""
    if HAS_PYPANDOC:
        try:
            from pypandoc import convert_text
            # New naming: turn{num}_{model_abbrevs}.docx
            docx_path = turn_dir / f"turn{turn_num}_{models_str}.docx"
            convert_text("".join(md_parts), "docx", format="md", outputfile=str(docx_path))
            docx_out = str(docx_path)
        except Exception:
            pass

    return {
        "jsonl_path": str(jsonl_path),
        "md_path": md_path,
        "html_path": html_path,
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
