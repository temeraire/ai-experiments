#!/usr/bin/env python3
"""
Build Flashcards/app/public/cards.json from the markdown source of truth
(alien_baby/FLASHCARDS.md), plus the 12 HTML-only "Basics" intro cards
preserved from the legacy flashcards.html deck.

Handles both markdown card styles:
    Q: plain question            (original format, ~292 cards)
    **Q: bold question**         (newer glossary-rule entries)

Each card gets a stable id = sha1(question)[:12] so review history survives
re-syncs and re-orderings (natural key per DB_BACKEND_PLAN.md).

Usage:  python3 Flashcards/app/sync_cards_json.py
"""
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MD = ROOT / "alien_baby" / "FLASHCARDS.md"
LEGACY_HTML = ROOT / "Flashcards" / "flashcards.html"
OUT = pathlib.Path(__file__).resolve().parent / "public" / "cards.json"


def _clean_q(line):
    """Strip 'Q:' prefix in both plain and bold forms."""
    line = line.strip()
    if line.startswith("**Q:"):
        line = line[4:]
        if line.endswith("**"):
            line = line[:-2]
    elif line.startswith("Q:"):
        line = line[2:]
    return line.strip().rstrip("*").strip()


def parse_markdown(text):
    cards, section, q, a_lines, mode = [], None, None, [], None

    def flush():
        nonlocal q, a_lines, mode
        if q is not None and a_lines:
            cards.append({"s": section or "Misc", "q": q.strip(),
                          "a": " ".join(a_lines).strip()})
        q, a_lines, mode = None, [], None

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            flush(); section = line[3:].strip()
        elif line.startswith("Q:") or line.startswith("**Q:"):
            flush(); q = _clean_q(line); mode = "q"
        elif line.startswith("A:"):
            a_lines = [line[2:].strip()]; mode = "a"
        elif line.strip() in ("", "---") or line.startswith("#"):
            continue
        else:
            if mode == "a":
                a_lines.append(line.strip())
            elif mode == "q":
                q = (q + " " + line.strip()).strip()
    flush()
    return cards


def legacy_basics():
    """Pull the HTML-only 'Basics' cards out of the legacy deck (they have no
    markdown source). Parsed leniently from the JS object-literal lines."""
    try:
        html = LEGACY_HTML.read_text()
    except FileNotFoundError:
        return []
    m = re.search(r"const ALL_CARDS = \[(.*?)\n\];", html, re.S)
    if not m:
        return []
    out = []
    for ln in m.group(1).splitlines():
        if 's:"Basics"' not in ln:
            continue
        qm = re.search(r'q:("(?:[^"\\]|\\.)*")', ln)
        am = re.search(r'a:("(?:[^"\\]|\\.)*")', ln)
        if qm and am:
            out.append({"s": "Basics", "q": json.loads(qm.group(1)),
                        "a": json.loads(am.group(1))})
    return out


def main():
    cards = legacy_basics() + parse_markdown(MD.read_text())
    for c in cards:
        c["id"] = hashlib.sha1(c["q"].encode()).hexdigest()[:12]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cards": cards}, ensure_ascii=False, indent=1))
    sections = {}
    for c in cards:
        sections[c["s"]] = sections.get(c["s"], 0) + 1
    print(f"total cards: {len(cards)}  (unique ids: {len({c['id'] for c in cards})})")
    for s, n in sections.items():
        print(f"  {n:4}  {s}")
    print("wrote:", OUT)


if __name__ == "__main__":
    main()
