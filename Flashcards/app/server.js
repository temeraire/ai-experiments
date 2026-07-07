#!/usr/bin/env node
/*
 * Flashcards server — zero-dependency Node (no npm install needed).
 *
 * Serves the mobile flashcards UI and a tiny JSON API. Review results are
 * persisted as append-only JSON-lines (one review per line) — durable,
 * human-readable, trivially backed up, and plenty for a single user.
 * (Upgrade path to Postgres documented in Flashcards/DB_BACKEND_PLAN.md.)
 *
 *   GET  <base>/            -> public/index.html
 *   GET  <base>/cards.json  -> the deck (synced from alien_baby/FLASHCARDS.md)
 *   POST <base>/api/review  -> {cardId, section, result:'right'|'wrong', flips, ms}
 *   GET  <base>/api/stats   -> per-card + overall aggregates
 *   GET  <base>/api/history?limit=100 -> recent reviews, newest first
 *
 * All URLs are handled relative to any mount path, so the app works equally
 * at http://localhost:8642/ and behind a proxy at /tools/flashcards/.
 *
 * Env: PORT (default 8642), FLASHCARDS_DATA (default ./data/reviews.jsonl)
 */
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = parseInt(process.env.PORT || "8642", 10);
const DATA = process.env.FLASHCARDS_DATA ||
  path.join(__dirname, "data", "reviews.jsonl");
const PUBLIC = path.join(__dirname, "public");

fs.mkdirSync(path.dirname(DATA), { recursive: true });

const MIME = { ".html": "text/html; charset=utf-8", ".json": "application/json",
  ".js": "text/javascript", ".css": "text/css", ".png": "image/png",
  ".svg": "image/svg+xml" };

function readReviews() {
  let lines = [];
  try {
    lines = fs.readFileSync(DATA, "utf8").split("\n").filter(Boolean);
  } catch (e) { /* no reviews yet */ }
  const out = [];
  for (const ln of lines) {
    try { out.push(JSON.parse(ln)); } catch (e) { /* skip corrupt line */ }
  }
  return out;
}

function sendJSON(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body) });
  res.end(body);
}

function handleReview(req, res) {
  let body = "";
  req.on("data", (c) => { body += c; if (body.length > 1e5) req.destroy(); });
  req.on("end", () => {
    let r;
    try { r = JSON.parse(body); } catch (e) {
      return sendJSON(res, 400, { error: "bad json" });
    }
    if (!r || typeof r.cardId !== "string" ||
        !["right", "wrong"].includes(r.result)) {
      return sendJSON(res, 400, { error: "need cardId + result right|wrong" });
    }
    const rec = {
      ts: new Date().toISOString(),
      cardId: r.cardId.slice(0, 40),
      section: String(r.section || "").slice(0, 120),
      question: String(r.question || "").slice(0, 300),
      result: r.result,
      flips: Math.max(0, Math.min(9, parseInt(r.flips, 10) || 0)),
      ms: Math.max(0, Math.min(3600e3, parseInt(r.ms, 10) || 0)),
      device: String(r.device || "").slice(0, 60),
    };
    fs.appendFile(DATA, JSON.stringify(rec) + "\n", (err) => {
      if (err) return sendJSON(res, 500, { error: "write failed" });
      sendJSON(res, 200, { ok: true });
    });
  });
}

function handleStats(res) {
  const reviews = readReviews();
  const perCard = {};
  let right = 0;
  for (const r of reviews) {
    const c = perCard[r.cardId] ||
      (perCard[r.cardId] = { seen: 0, right: 0, wrong: 0, lastResult: null,
        lastTs: null, section: r.section, question: r.question,
        flipsTotal: 0 });
    c.seen += 1; c[r.result] += 1; c.lastResult = r.result; c.lastTs = r.ts;
    c.flipsTotal += r.flips || 0;
    c.section = r.section || c.section;
    c.question = r.question || c.question;
    if (r.result === "right") right += 1;
  }
  sendJSON(res, 200, {
    total: reviews.length, right, wrong: reviews.length - right,
    accuracy: reviews.length ? right / reviews.length : null,
    cards: perCard,
  });
}

function handleHistory(res, query) {
  const limit = Math.max(1, Math.min(1000, parseInt(query.get("limit"), 10) || 100));
  const reviews = readReviews();
  sendJSON(res, 200, { reviews: reviews.slice(-limit).reverse() });
}

function serveStatic(res, rel) {
  const file = path.normalize(path.join(PUBLIC, rel));
  if (!file.startsWith(PUBLIC)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, buf) => {
    if (err) { res.writeHead(404); return res.end("not found"); }
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream" });
    res.end(buf);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  const p = url.pathname;
  // Route on path SUFFIX so any proxy mount prefix works transparently.
  if (req.method === "POST" && p.endsWith("/api/review")) return handleReview(req, res);
  if (req.method === "GET" && p.endsWith("/api/stats")) return handleStats(res);
  if (req.method === "GET" && p.endsWith("/api/history")) return handleHistory(res, url.searchParams);
  if (req.method === "GET") {
    if (p.endsWith("/cards.json")) return serveStatic(res, "cards.json");
    if (p.endsWith("/")) return serveStatic(res, "index.html");
    return serveStatic(res, path.basename(p));
  }
  res.writeHead(405); res.end();
});

server.listen(PORT, () => {
  console.log(`[flashcards] listening on :${PORT}, data -> ${DATA}`);
});
