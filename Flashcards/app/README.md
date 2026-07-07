# Flashcards app (mobile, tracked)

Mobile-first flashcards with answer tracking. Zero-dependency Node server —
no npm install, no database daemon.

## Features
- **Tap to flip, up to 3 flips per card.** Question → answer → question → answer,
  then the card locks and you must commit. Flip dots show how many you've used;
  flips-used is recorded with each answer (a free confidence signal).
- **Right/Wrong tracking.** Every answer is appended to
  `data/reviews.jsonl` (one JSON line per review: ts, cardId, section,
  question, result, flips, time-to-answer, device). Stats tab shows lifetime
  accuracy, deck coverage, trouble cards, and recent history.
- **Session filters:** by section, shuffle, only-missed-before, only-never-seen.
- **Deck source of truth stays markdown:** `alien_baby/FLASHCARDS.md`
  (+ 12 legacy "Basics" cards from the old HTML deck). Regenerate with:
      python3 Flashcards/app/sync_cards_json.py
  Card ids are sha1(question), so history survives re-syncs.

## Run locally
    node Flashcards/app/server.js          # http://localhost:8642
    PORT=9000 node server.js               # custom port
    FLASHCARDS_DATA=/srv/flashcards/reviews.jsonl node server.js

Phone on the same wifi: http://<your-mac-ip>:8642

## Deploy on the droplet (AuthGateway /tools pattern)
The server routes on path *suffixes*, so it works unchanged behind a proxy
mount like `/tools/flashcards/`:

1. Copy `Flashcards/app/` to the droplet (e.g. `/srv/flashcards/`).
2. Run it under systemd (or pm2), e.g.:
       [Service]
       ExecStart=/usr/bin/node /srv/flashcards/server.js
       Environment=PORT=8642
       Environment=FLASHCARDS_DATA=/srv/flashcards/data/reviews.jsonl
       Restart=always
3. In the AuthGateway/nginx config, proxy the tool path to it:
       location /tools/flashcards/ { proxy_pass http://127.0.0.1:8642; }
   (Trailing slashes matter; the app fetches `cards.json` and `api/...`
   relative to the page URL.)
4. Back up `data/reviews.jsonl` with whatever backs up the other tools.

Updating the deck later = re-run the sync script locally, copy the new
`public/cards.json` up. Review history is untouched (append-only file).

## Upgrade path
If this ever needs multi-user/spaced repetition, see
`Flashcards/DB_BACKEND_PLAN.md` — the JSONL rows map 1:1 onto the
`review_state`/`reviews` tables sketched there.
