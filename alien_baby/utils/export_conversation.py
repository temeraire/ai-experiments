"""
export_conversation.py — Convert a Claude Code session JSONL into a clean
Markdown transcript.

Default output is "Option 4": user messages + assistant visible text replies
ONLY. System reminders, thinking blocks, tool calls, tool results, and
subagent (sidechain) traffic are all excluded.

Usage:
    python -m alien_baby.utils.export_conversation
    python -m alien_baby.utils.export_conversation --session-id 2c29b80f-e554-4cd2-ba39-84c89b88867b
    python -m alien_baby.utils.export_conversation --out /path/to/transcript.md

By default reads the most recently modified .jsonl in
~/.claude/projects/<encoded-cwd>/ and writes to
alien_baby/CONVERSATION_LOG.md.
"""
import argparse
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

PROJECT_DIR = pathlib.Path(__file__).parent.parent

# Same encoding Claude Code uses to name the project's transcript dir.
# Both "/" and "_" get replaced with "-" so e.g. /Users/.../ai_experiments
# becomes -Users-...-ai-experiments.
def _encoded_cwd(cwd: pathlib.Path) -> str:
    s = str(cwd).replace("_", "-").replace("/", "-")
    return s if s.startswith("-") else "-" + s


def _latest_jsonl(project_dir: pathlib.Path) -> pathlib.Path:
    transcript_dir = pathlib.Path.home() / ".claude" / "projects" / _encoded_cwd(project_dir.parent)
    if not transcript_dir.exists():
        raise FileNotFoundError(f"No transcript directory at {transcript_dir}")
    jsonls = list(transcript_dir.glob("*.jsonl"))
    if not jsonls:
        raise FileNotFoundError(f"No .jsonl files in {transcript_dir}")
    return max(jsonls, key=lambda p: p.stat().st_mtime)


SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
COMMAND_TAG_RE     = re.compile(r"<command-(name|message|args)>.*?</command-\1>", re.DOTALL)
LOCAL_COMMAND_RE   = re.compile(r"<local-command-stdout>.*?</local-command-stdout>", re.DOTALL)


def _clean_user_text(text: str) -> str:
    """Strip system reminders and runtime tags from a user message string."""
    text = SYSTEM_REMINDER_RE.sub("", text)
    text = COMMAND_TAG_RE.sub("", text)
    text = LOCAL_COMMAND_RE.sub("", text)
    return text.strip()


def _assistant_visible_text(content) -> str:
    """Concatenate only the 'text' blocks of an assistant message."""
    if not isinstance(content, list):
        return ""
    pieces = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            pieces.append(block.get("text", ""))
    return "\n\n".join(p for p in pieces if p.strip())


def _fmt_ts(iso: str) -> str:
    """Convert ISO UTC timestamp to local-ish display string."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        local = dt.astimezone()  # local timezone
        return local.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso


def export(jsonl_path: pathlib.Path, out_path: pathlib.Path) -> dict:
    """Walk the JSONL and write a clean Markdown transcript. Returns counts."""
    user_count = 0
    assistant_count = 0
    skipped_sidechain = 0
    skipped_empty = 0

    with open(jsonl_path) as f, open(out_path, "w") as out:
        out.write(f"# Conversation Transcript\n\n")
        out.write(f"_Source: `{jsonl_path}`_  \n")
        out.write(f"_Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}_  \n")
        out.write(f"_Scope: user messages and Claude's visible text replies only._  \n")
        out.write(f"_Excluded: system reminders, thinking blocks, tool calls, tool results, subagent traffic._\n\n")
        out.write("---\n\n")

        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip subagent / sidechain traffic — that's the agent-to-agent
            # back-and-forth from when we delegated to specialized agents.
            # It's not "between us" in the conversational sense.
            if r.get("isSidechain"):
                skipped_sidechain += 1
                continue

            t = r.get("type")
            ts = _fmt_ts(r.get("timestamp", ""))
            msg = r.get("message") or {}

            if t == "user":
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, str):
                    # tool_result blocks come through as user with list content; skip
                    skipped_empty += 1
                    continue
                cleaned = _clean_user_text(content)
                if not cleaned:
                    # Nothing but system reminders → skip
                    skipped_empty += 1
                    continue
                out.write(f"## {ts} — You\n\n")
                out.write(cleaned + "\n\n")
                user_count += 1
            elif t == "assistant":
                content = msg.get("content") if isinstance(msg, dict) else None
                visible = _assistant_visible_text(content)
                if not visible:
                    # Pure tool-call turn, no visible text → skip
                    skipped_empty += 1
                    continue
                out.write(f"## {ts} — Claude\n\n")
                out.write(visible + "\n\n")
                assistant_count += 1
            # else: permission-mode, file-history-snapshot, ai-title, etc.
            # All skipped.

    return {
        "user_messages": user_count,
        "assistant_messages": assistant_count,
        "skipped_sidechain": skipped_sidechain,
        "skipped_empty": skipped_empty,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session-id", default=None,
                   help="Specific session JSONL UUID (default: most recent).")
    p.add_argument("--out", default=str(PROJECT_DIR / "CONVERSATION_LOG.md"),
                   help="Output markdown path.")
    args = p.parse_args()

    if args.session_id:
        transcript_dir = pathlib.Path.home() / ".claude" / "projects" / _encoded_cwd(PROJECT_DIR.parent)
        jsonl = transcript_dir / f"{args.session_id}.jsonl"
        if not jsonl.exists():
            print(f"Session JSONL not found: {jsonl}", file=sys.stderr)
            sys.exit(1)
    else:
        jsonl = _latest_jsonl(PROJECT_DIR)

    out_path = pathlib.Path(args.out)
    print(f"Reading: {jsonl}  ({jsonl.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Writing: {out_path}")

    counts = export(jsonl, out_path)
    print(f"\nDone. {counts['user_messages']} user messages, {counts['assistant_messages']} assistant replies.")
    print(f"Skipped {counts['skipped_sidechain']} sidechain records, {counts['skipped_empty']} empty turns.")
    print(f"Output size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
