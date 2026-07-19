#!/usr/bin/env python3
"""Export a Claude Code session transcript (the .jsonl Claude Code already saves) to a
readable .txt — so the conversation is captured on disk without manual copy-paste.

Usage:
  python export_transcript.py                 # newest session for this project
  python export_transcript.py <session.jsonl> # a specific transcript
  python export_transcript.py --out FILE.txt  # choose output path

By default writes alongside the project at alien_baby/transcripts/<session>.txt.
Only the MAIN conversation is exported (subagent side-threads are skipped), matching
what you would otherwise copy from the terminal.
"""
import json, sys, glob, os

PROJ_DIR = os.path.expanduser(
    "~/.claude/projects/-Users-davidwolpe-Documents-DataScience-dev-ai-experiments")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "alien_baby", "transcripts")


def newest_transcript():
    js = sorted(glob.glob(os.path.join(PROJ_DIR, "*.jsonl")), key=os.path.getmtime)
    return js[-1] if js else None


def text_of(content):
    """Pull readable text + compact tool markers out of a message content field."""
    if isinstance(content, str):
        return content.strip()
    out = []
    for b in content:
        t = b.get("type")
        if t == "text":
            out.append(b.get("text", "").strip())
        elif t == "tool_use":
            name = b.get("name", "tool")
            out.append(f"    [→ tool: {name}]")
        elif t == "tool_result":
            # skip verbose tool output in the readable transcript
            continue
        elif t == "thinking":
            continue
    return "\n".join(s for s in out if s)


def _stop_hook_reentry():
    """When run as a Claude Code Stop hook, the harness passes JSON on stdin with a
    `stop_hook_active` flag. If it's already active we're in a stop re-entry loop, so
    exit cleanly — the hook fires once per turn and never re-triggers it. Returns False
    for manual runs (stdin is a tty, empty, or not the hook JSON), so `python
    export_transcript.py` from a terminal still works normally."""
    try:
        if sys.stdin.isatty():
            return False
        data = sys.stdin.read()
        if not data.strip():
            return False
        return bool(json.loads(data).get("stop_hook_active"))
    except Exception:
        return False


def main():
    if _stop_hook_reentry():
        return
    args = [a for a in sys.argv[1:]]
    out_path = None
    if "--out" in args:
        i = args.index("--out"); out_path = args[i + 1]; del args[i:i + 2]
    src = args[0] if args else newest_transcript()
    if not src or not os.path.exists(src):
        print(f"No transcript found (looked in {PROJ_DIR})"); sys.exit(1)
    if out_path is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        out_path = os.path.join(OUT_DIR, base + ".txt")

    lines, last_role = [], None
    for raw in open(src):
        try:
            d = json.loads(raw)
        except Exception:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue
        if d.get("isSidechain"):          # skip subagent side-threads
            continue
        msg = d.get("message", {})
        role = msg.get("role", d.get("type"))
        body = text_of(msg.get("content", ""))
        if not body:
            continue
        # drop system-reminder-only / injected user blocks that add noise
        if role == "user" and body.startswith("<") and "system-reminder" in body[:40]:
            continue
        label = "USER" if role == "user" else "CLAUDE"
        if label != last_role:                       # new speaker → new header
            lines.append(f"\n{'='*70}\n{label}:\n{'='*70}")
            last_role = label
        lines.append(body)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}  ({len(lines)} blocks from {os.path.basename(src)})")


if __name__ == "__main__":
    main()
