# Flashcards — Database-Backed Scaling Plan (proposal)

*Drafted 2026-06-16 overnight, for review. Nothing here has been provisioned or deployed —
it's a plan + recommendation, because standing up infra touches your Netlify/DigitalOcean
credentials and is not something to do unsupervised.*

---

## Where we are now (and why it doesn't scale)

- **Data lives inside the HTML.** Cards are a hard-coded `const ALL_CARDS = [...]` array baked
  into `flashcards.html`. `sync_cards.py` regenerates that array from `alien_baby/FLASHCARDS.md`.
- **Glossary** is about to be embedded the same way (a `const GLOSSARY = {...}` literal) by the
  drill-down build.
- **User-added cards** persist only in the browser's `localStorage`, per-device, and have to be
  manually exported and pasted back into the HTML to become permanent.
- **Deployed** as a static page (GitHub Pages: `temeraire.github.io/rl-flashcards`).

Pain points this creates:
1. **Update = regenerate + redeploy.** Every content change means running a script and pushing.
2. **No cross-device sync.** Cards you add on the phone don't exist on the laptop, and vice versa.
3. **Study progress isn't saved anywhere.** No record of what you've learned, no spaced
   repetition — every session starts cold.
4. **Three drifting copies of truth:** the markdown, the embedded JS, and per-device localStorage.

The single highest-value thing a backend unlocks is **#3 — persisted, cross-device study state
(spaced repetition)**. That, not raw storage, is the reason to bother with a DB.

---

## The options

### A. JSON files (no backend) — *do this first regardless*
Move cards + glossary out of the HTML into `cards.json` / `glossary.json`, fetched at runtime.
`sync_cards.py` writes JSON instead of rewriting the HTML.
- **Pros:** trivial, removes the embed-and-redeploy coupling, keeps current hosting, zero infra,
  one obvious source file per dataset.
- **Cons:** still no write API, no cross-device user cards, no study state. JSON still in git.
- **Effort:** ~1 hour. **Cost:** $0.

### B. Netlify static + managed Postgres (Supabase) — *recommended backend*
Static frontend stays on Netlify/Pages. A **managed Postgres** holds the data. **Supabase** is the
sweet spot: it gives you Postgres + an auto-generated REST/realtime API + auth + row-level
security out of the box, so you may not need to write a server at all. (Neon is a leaner
Postgres-only alternative if you'd rather write your own small API via Netlify Functions.)
- **Pros:** real DB, cross-device, auth if you want accounts, generous free tier, almost no ops,
  instant API (Supabase). Scales well past anything a flashcard app needs.
- **Cons:** another SaaS dependency; serverless cold-starts (negligible here).
- **Effort:** ~half a day. **Cost:** $0 on free tier.

### C. DigitalOcean droplet + Postgres + small API (self-hosted) — *if you prefer owning it*
Run Postgres + a small API (FastAPI or Express) on the droplet you already have; frontend calls
it. Co-locates with your other projects.
- **Pros:** full control, no per-row SaaS limits, consolidates on infra you already pay for, data
  stays yours.
- **Cons:** you own the ops — TLS/HTTPS, backups, uptime, security patching. More moving parts.
- **Effort:** ~1 day (server + API + TLS + deploy). **Cost:** $0 marginal (existing droplet).

---

## Recommended path (phased — each phase is independently useful)

**Phase 1 — Decouple data into JSON (today, no infra).**
Frontend fetches `cards.json` + `glossary.json`. `sync_cards.py` (and a new `sync_glossary.py`)
write those from the markdown source of truth. Immediate win: edit markdown → run sync → the app
updates with no HTML surgery. This is reversible and low-risk; I can do it as soon as the
drill-down build lands.

**Phase 2 — Stand up the database + API.** My recommendation is **Supabase (option B)** for
fastest time-to-value: managed Postgres, an API you don't have to write, free tier. If you'd
rather keep everything on the **droplet (option C)**, that's a clean choice too — it just trades
half a day of ops setup for full ownership. *This is the main decision I need from you* (see
below). Either way the data model and migration are the same.

**Phase 3 — Spaced repetition + cross-device study state.** Add a `review_state` table and an
SM-2 / FSRS scheduler so the app shows due cards and syncs progress across phone and laptop. This
is the payoff that justifies the backend.

---

## Data model (same regardless of B or C)

```
cards
  id            uuid pk
  section       text
  question      text          -- natural key for upsert-from-markdown
  answer        text
  source        text          -- 'markdown' | 'app' | 'import'
  created_at    timestamptz
  updated_at    timestamptz
  deleted       bool          -- soft delete

glossary_terms
  id            uuid pk
  term          text unique
  aka           text[]        -- aliases for auto-linking ("σ", "observations")
  levels        jsonb         -- [{level:1, body:"..."}, {level:2, body:"...", math:"..."}]

review_state            -- Phase 3
  id            uuid pk
  card_id       uuid fk -> cards
  user_id       text          -- or a device id if we skip real accounts
  ease          real
  interval_days int
  due_at        timestamptz
  last_reviewed timestamptz
```

**Authoring stays markdown-first.** `alien_baby/GLOSSARY.md` + `FLASHCARDS.md` remain the
human-authored source (so the standing glossary-upkeep rule still works); a sync script upserts
markdown → DB by natural key (question text / term). App-added cards write straight to the DB with
`source='app'`. One-time migration imports the current `ALL_CARDS` + any localStorage exports.

## API sketch (thin)
```
GET  /cards            -> all active cards (optionally ?section=)
POST /cards            -> add a card from the app
GET  /glossary         -> all terms with levels
POST /review           -> record a ✓/✗ result, get next due
GET  /due              -> cards due now for this user/device   (Phase 3)
```
With Supabase, most of these are auto-generated from the tables (+ row-level security); with the
droplet, it's ~120 lines of FastAPI.

---

## Decisions I need from you (the only blockers)

1. **Backend host: Supabase (managed, fastest) vs your DigitalOcean droplet (self-hosted, owned)?**
   My lean: Supabase to start — you can always migrate the Postgres to the droplet later, since
   it's just Postgres either way.
2. **Accounts/auth, or single-user (just you)?** If it's only ever you, we can skip real auth and
   key study-state by a device id — much simpler. Auth is easy to add later if you ever share it.
3. **Spaced-repetition algorithm:** classic **SM-2** (simple, proven) vs **FSRS** (modern, better
   scheduling, slightly more to implement). Default SM-2 unless you want FSRS.

Give me those three and I can do Phase 1 immediately and scaffold Phase 2/3 on approval. Until
then nothing is provisioned — this is a plan, not a deployment.
