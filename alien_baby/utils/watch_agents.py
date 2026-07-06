#!/usr/bin/env python3
"""Live viewer for Claude Code subagent transcripts — one pane per agent role.

Claude Code writes each subagent's transcript as a JSONL file under the session
tasks directory. This script tails those files and pretty-prints the agent's
activity (its text, and which tools it calls) as it happens.

Pane/role matching: the orchestrating session starts each agent prompt with a
first line like "[role: experiment-strategist]". Run one watcher per role:

    python3 alien_baby/utils/watch_agents.py --role experiment-strategist

or watch everything in one stream:

    python3 alien_baby/utils/watch_agents.py --all

Use alien_baby/utils/agents_dashboard.sh to open an iTerm2 window with one
pane per project agent.
"""
import argparse
import glob
import json
import os
import textwrap
import time

TASKS_GLOB = os.path.expanduser(
    "/private/tmp/claude-*/-Users-*-ai-experiments*/*/tasks/*.output"
)

C_HDR = "\033[1;36m"   # bold cyan
C_TOOL = "\033[33m"    # yellow
C_TXT = "\033[0m"      # default
C_DIM = "\033[2m"
RESET = "\033[0m"


def first_line_matches(path, role):
    """True if the transcript's first user message carries our role tag (or,
    with --all, if the file looks like an agent transcript at all)."""
    try:
        with open(path) as f:
            head = f.readline()
        if not head.startswith("{"):
            return False
        rec = json.loads(head)
        content = rec.get("message", {}).get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        if role is None:
            return rec.get("isSidechain") is True
        return f"[role: {role}]" in content
    except Exception:
        return False


def render_record(line):
    """One JSONL record -> zero or more printable lines."""
    try:
        rec = json.loads(line)
    except Exception:
        return []
    msg = rec.get("message", {})
    out = []
    content = msg.get("content")
    if not isinstance(content, list):
        return out
    for block in content:
        btype = block.get("type")
        if btype == "text" and msg.get("role") == "assistant":
            for para in block["text"].split("\n"):
                out.extend(textwrap.wrap(para, 100) or [""])
        elif btype == "tool_use":
            inp = json.dumps(block.get("input", {}))
            if len(inp) > 120:
                inp = inp[:120] + "…"
            out.append(f"{C_TOOL}⚙ {block.get('name')}{RESET} {C_DIM}{inp}{RESET}")
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--role", help="agent role tag to follow (e.g. experiment-strategist)")
    g.add_argument("--all", action="store_true", help="follow every agent transcript")
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()
    role = None if args.all else args.role

    print(f"{C_HDR}=== watching for agent: {role or 'ALL'} ==={RESET}")
    print(f"{C_DIM}(waiting for a matching agent to be spawned…){RESET}")

    offsets = {}   # path -> bytes already printed
    known = set()  # paths already classified
    matched = set()
    while True:
        for path in glob.glob(TASKS_GLOB):
            if path not in known:
                known.add(path)
                if first_line_matches(path, role):
                    matched.add(path)
                    mtime = os.path.getmtime(path)
                    # Replay content only for fresh transcripts; for old ones
                    # just announce them and follow from the end.
                    fresh = (time.time() - mtime) < 600
                    offsets[path] = 0 if fresh else os.path.getsize(path)
                    stamp = time.strftime("%H:%M:%S", time.localtime(mtime))
                    label = "live" if fresh else "idle — following"
                    print(f"\n{C_HDR}━━━ agent {os.path.basename(path)} ({stamp}, {label}) ━━━{RESET}")
        for path in list(matched):
            try:
                size = os.path.getsize(path)
                if size > offsets[path]:
                    with open(path) as f:
                        f.seek(offsets[path])
                        chunk = f.read()
                        offsets[path] = f.tell()
                    for line in chunk.splitlines():
                        for rendered in render_record(line):
                            print(rendered, flush=True)
            except FileNotFoundError:
                matched.discard(path)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
