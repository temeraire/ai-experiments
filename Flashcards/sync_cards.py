#!/usr/bin/env python3
"""
Sync the flashcard deck embedded in flashcards.html from the markdown source
of truth (alien_baby/FLASHCARDS.md).

Why: flashcards.html hard-codes the deck in a JS `ALL_CARDS = [...]` array.
That array drifts out of date as we add cards to FLASHCARDS.md. This script
re-parses the markdown and rewrites ONLY that array, leaving all UI/logic
untouched. The 12 conversation-derived "Basics" intro cards live only in the
HTML (not in the markdown), so they are preserved verbatim.

Usage:  python Flashcards/sync_cards.py
Then:   open the file / redeploy to GitHub Pages.
"""
import json
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD = ROOT / "alien_baby" / "FLASHCARDS.md"
HTML = ROOT / "Flashcards" / "flashcards.html"


def parse_markdown(text):
    """Parse FLASHCARDS.md into a list of {s, q, a} dicts.

    A card starts at a `Q:` line and ends at the next `Q:`/`## `/EOF.
    `---` and blank lines are separators; multi-line answers are joined.
    """
    cards = []
    section = None
    q = None
    a_lines = []
    mode = None  # 'q' while reading question continuation, 'a' for answer

    def flush():
        nonlocal q, a_lines, mode
        if q is not None and a_lines:
            cards.append({
                "s": section or "Misc",
                "q": q.strip(),
                "a": " ".join(a_lines).strip(),
            })
        q, a_lines, mode = None, [], None

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
        elif line.startswith("Q:"):
            flush()
            q = line[2:].strip()
            mode = "q"
        elif line.startswith("A:"):
            a_lines = [line[2:].strip()]
            mode = "a"
        elif line.strip() in ("", "---") or line.startswith("#"):
            continue  # separators / top-level headings carry no card content
        else:  # continuation line
            if mode == "a":
                a_lines.append(line.strip())
            elif mode == "q":
                q = (q + " " + line.strip()).strip()
    flush()
    return cards


def main():
    html = HTML.read_text()
    md_cards = parse_markdown(MD.read_text())

    # Preserve the HTML-only "Basics" intro cards verbatim.
    m = re.search(r"const ALL_CARDS = \[(.*?)\n\];", html, re.S)
    if not m:
        raise SystemExit("Could not find `const ALL_CARDS = [...]` in the HTML")
    old_block = m.group(1)
    basics = [ln for ln in old_block.splitlines() if 's:"Basics"' in ln]

    out = ["  // ── Intro basics (conversation-derived; not in FLASHCARDS.md) ──"]
    out.extend(basics)
    cur = None
    for c in md_cards:
        if c["s"] != cur:
            cur = c["s"]
            out.append("")
            out.append(f"  // ── {cur} ──")
        out.append("  {{s:{}, q:{}, a:{}}},".format(
            json.dumps(c["s"], ensure_ascii=False),
            json.dumps(c["q"], ensure_ascii=False),
            json.dumps(c["a"], ensure_ascii=False),
        ))

    new_block = "const ALL_CARDS = [\n" + "\n".join(out) + "\n];"
    new_html = html[:m.start()] + new_block + html[m.end():]
    HTML.write_text(new_html)

    # Emit the array alone for a `node --check` validation step.
    (pathlib.Path("/tmp") / "cards_check.js").write_text(new_block + "\n")
    print(f"Basics preserved : {len(basics)}")
    print(f"Cards from md    : {len(md_cards)}")
    print(f"Total in deck    : {len(basics) + len(md_cards)}")
    print("Wrote:", HTML)
    print("Validate JS with : node --check /tmp/cards_check.js")


if __name__ == "__main__":
    main()
