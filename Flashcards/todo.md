# Flashcards: free-flip + drill-down glossary

## Plan
- [x] 1. Free flipping: make `flip()` toggle front↔back unlimited times; show flip button on front, ✗/✓ + follow-up on back. Decision only via ✗/✓.
- [x] 2. KaTeX from CDN (defer) for math rendering.
- [x] 3. CSS: `.term` (underlined tappable) + drill-down sheet styles.
- [x] 4. Drill modal markup (breadcrumb, title, levels, Go deeper / Back / Close).
- [x] 5. GLOSSARY data (broad first pass, ~40 terms, 1–3 levels each; project's own phrasing; L2 norm has the math level).
- [x] 6. Auto-link: `linkify()` wraps glossary terms in q/a text as tappable spans (skips `$math$`, escapes HTML); term clicks stopPropagation so they don't flip the card.
- [x] 7. Drill logic: openTerm/deeper/drillBack/closeDrill with breadcrumb stack; nested terms inside definitions are themselves tappable; render math per panel.
- [x] 8a. Test (stubbed DOM). PASS.
- [ ] 8b. Deploy to GitHub Pages. (left for human to deploy)

## Review (implemented 2026-06-17)
- BACKUP: `flashcards.html.bak_drilldown` written before any edit.
- Free flip: `flip()` now toggles front↔back unlimited times; flip button shows on front, ✗/✓ + follow-up on back. Card decision (right/wrong) is made ONLY via ✗/✓, never by flipping.
- KaTeX 0.16.11 loaded from jsDelivr (CSS `<link>` + deferred `<script>`) for typeset math in cards AND drill panels. Math is written `$...$`.
- 48-term GLOSSARY (separate `const GLOSSARY = {…}` structure, NOT folded into `ALL_CARDS`, so `sync_cards.py` keeps working), each term with 1–3 progressive levels in the project's own language drawn from `alien_baby/GLOSSARY.md`. "L2 norm" has the math level (`$\lVert v \rVert_2 = \sqrt{\sum v_i^2}$`); PoE, KL, CKA, MSE, σ, MICOA fusion, etc. also carry KaTeX formulas. Each term has an optional `aka` alias list (e.g. σ→sigma, proprio→proprioception, SAC→Soft Actor-Critic) for auto-linking.
- Auto-linking: `linkify()` underlines/makes-tappable any glossary term or alias in question AND answer text. Longest-match wins (`L2 norm` beats `L2`); HTML escaped; `$math$` segments left untouched; word-boundary aware so `step` doesn't match inside `timestep`. Term clicks `stopPropagation()` so they open the drill sheet WITHOUT flipping the card.
- Drill-down bottom-sheet: tap a term → Level 1 → "Go deeper ↓" reveals deeper levels; terms inside a definition are themselves tappable, pushing onto a breadcrumb STACK (`vision-ablation sensitivity › L2 norm`); ← Back pops (and dismisses at the root); tapping the backdrop or Close dismisses. KaTeX rendered per panel. Drill depth capped at 8.
- Verified: `node --check` on the page script + a 33-assertion stubbed-DOM test (`test_flashcards.js`): linkify wraps known terms, aka aliases, longest-match, HTML escaping, `$math$` skip, word-boundary, unknown-term safety, drill push/deeper/pop/back-to-close, depth cap, breadcrumb separator, unlimited flip toggle, ≥40 terms. ALL PASS (33/33). No card text contains `$`. `sync_cards.py` re-run cleanly afterward (12 Basics + 279 md = 291 cards) and the page still node --checks.
- Tradeoff: KaTeX is a CDN dependency, so math needs a network connection (fine on the hosted site; the deck, flipping, and drill-down all still work offline via `file://` — `typesetMath` no-ops when KaTeX is absent, leaving `$...$` visible).
- NOT DONE (per instructions): 8b GitHub Pages deploy — left for human to deploy. No git push / no remote touched.
