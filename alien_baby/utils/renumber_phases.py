#!/usr/bin/env python3
"""One-shot: rewrite old phase names (A-H, Roman I-V, W, X) to the unified
Roman scheme defined in PHASES.md. Collision-safe: old Roman I-V map to IX-XIII
while the NEW namespace reuses I-VIII, so we map whole "Phase <tokens>" runs in a
single left-to-right pass (no in-place re-matching). Run once; revertible via git.
"""
import re
import sys

# old single-token -> new single-token
TOKEN_MAP = {
    "A": "I", "B": "II", "C2": "III.2", "C": "III", "D": "IV",
    "E2": "V.2", "E": "V", "F-2k": "VI.2", "F": "VI", "G": "VII",
    "H": "VIII", "I": "IX", "II": "X", "III": "XI", "IV": "XII",
    "V": "XIII", "W": "XIV", "X": "XV",
    # "0" intentionally omitted -> "Phase 0" left untouched
}
# longest / most-specific first so C2 beats C, F-2k beats F, III beats II beats I
ORDER = ["F-2k", "C2", "E2", "III", "IV", "II", "A", "B", "C", "D",
         "E", "F", "G", "H", "V", "W", "X", "I"]
TOK = "|".join(re.escape(t) for t in ORDER)
SEP = r"(?:\s*/\s*|\s+and\s+|\s*–\s*)"           # "/", " and ", en-dash
PATTERN = re.compile(
    r"\b(Phases?)(\s+)(" + TOK + r")((?:" + SEP + r"(?:" + TOK + r"))*)\b")
SEP_SPLIT = re.compile(r"(\s*/\s*|\s+and\s+|\s*–\s*)")


def _repl(m):
    kw, ws, first, rest = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    parts = SEP_SPLIT.split(first + rest)            # tokens even idx, seps odd
    out = [TOKEN_MAP.get(p, p) if i % 2 == 0 else p
           for i, p in enumerate(parts)]
    return kw + ws + "".join(out)


def main(files):
    for path in files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        new, n = PATTERN.subn(_repl, text)
        if n:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new)
        print(f"{n:4d} substitutions  {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
