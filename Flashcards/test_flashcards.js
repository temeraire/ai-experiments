#!/usr/bin/env node
/*
 * Stubbed-DOM test for flashcards.html's drill-down + linkify logic.
 * No browser: we stub the small slice of the DOM the page touches, eval the
 * page's <script>, then assert on the pure logic.
 *
 * Run:  node Flashcards/test_flashcards.js
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const HTML = path.join(__dirname, 'flashcards.html');

// ── Extract the main (no-src) <script> from the page ──────────────────────────
const html = fs.readFileSync(HTML, 'utf8');
const scripts = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
// In a vm context, top-level `const`/`let` bindings are NOT attached to the
// sandbox object — only `var`/functions/globals are. Append an explicit export
// so the test can reach the const-declared logic. This does not modify the page.
const EXPORTS = ['linkify','linkifySegment','escapeHtml','GLOSSARY','TERM_INDEX',
  'openTerm','deeper','drillBack','closeDrill','renderDrill','flip','showNext',
  'setRichText','typesetMath','current','flipped','deck'];
const exportTail = '\n;(function(){var __g=this;' +
  EXPORTS.map(n => 'try{__g.' + n + '=' + n + ';}catch(e){}').join('') + '}).call(this);\n';
const script = scripts[scripts.length - 1] + exportTail;

// ── Minimal DOM stub ──────────────────────────────────────────────────────────
function makeEl(id) {
  const el = {
    id: id || '', _text: '', _html: '', classList: { _set: new Set(),
      add(c){this._set.add(c);}, remove(c){this._set.delete(c);},
      contains(c){return this._set.has(c);},
      toggle(c,f){ if(f===undefined){ this._set.has(c)?this._set.delete(c):this._set.add(c); } else { f?this._set.add(c):this._set.delete(c); } return this._set.has(c);} },
    style: {}, dataset: {}, _listeners: [], children: [],
    addEventListener(t, fn){ this._listeners.push({t, fn}); },
    appendChild(c){ this.children.push(c); return c; },
    querySelectorAll(sel){
      // crude: find <span class="term"> spans in _html
      if (sel === '.term') {
        const terms = [...(this._html.matchAll(/data-term="([^"]*)"/g))].map(m => {
          const e = makeEl();
          e.dataset.term = m[1].replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'");
          return e;
        });
        return terms;
      }
      return [];
    },
    set textContent(v){ this._text = String(v); },
    get textContent(){ return this._text; },
    set innerHTML(v){ this._html = String(v); },
    get innerHTML(){ return this._html; },
  };
  return el;
}
const elements = {};
function getEl(id){ return elements[id] || (elements[id] = makeEl(id)); }

const documentStub = {
  getElementById: getEl,
  createElement: () => makeEl(),
  createDocumentFragment: () => makeEl(),
  createTextNode: (t) => ({ nodeType: 3, nodeValue: t }),
  createTreeWalker: () => ({ nextNode: () => null }),
  querySelectorAll: () => [],
};
const localStorageStub = { getItem: () => null, setItem: () => {}, };

const sandbox = {
  document: documentStub,
  localStorage: localStorageStub,
  window: {},
  navigator: {},
  console,
  NodeFilter: { SHOW_TEXT: 4 },
  alert: () => {},
  // katex intentionally undefined → typesetMath is a no-op (offline-safe path)
};
sandbox.window = sandbox;
vm.createContext(sandbox);
// Don't run the auto-init (startRound etc.) — strip trailing init calls by
// wrapping: we just need the function defs + data. Running them is harmless
// because the DOM is stubbed, but startRound shuffles CARDS which is fine.
vm.runInContext(script, sandbox);

// ── Tiny assert harness ───────────────────────────────────────────────────────
let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) { pass++; } else { fail++; console.log('  FAIL:', msg); } }
function eq(a, b, msg) { ok(a === b, msg + ' (got ' + JSON.stringify(a) + ', want ' + JSON.stringify(b) + ')'); }

const { linkify, GLOSSARY, openTerm, deeper, drillBack, closeDrill } = sandbox;

// ── 1. linkify wraps a known term ─────────────────────────────────────────────
ok(linkify('What is SAC?').includes('data-term="SAC"'), 'linkify wraps SAC');
ok(linkify('the actor network').includes('data-term="actor"'), 'linkify wraps actor');

// ── 2. aka aliases resolve to the canonical term ─────────────────────────────
ok(linkify('Soft Actor-Critic rocks').includes('data-term="SAC"'), 'alias Soft Actor-Critic → SAC');
ok(linkify('proprio sense').includes('data-term="proprioception"'), 'alias proprio → proprioception');

// ── 3. longest-match wins: "L2 norm" beats "L2" ──────────────────────────────
{
  const h = linkify('the L2 norm of the action');
  ok(h.includes('>L2 norm<') || h.includes('>L2 norm</span>'), 'L2 norm matched whole');
  ok(h.includes('data-term="L2 norm"'), 'L2 norm resolves to L2 norm term');
}

// ── 4. HTML is escaped ────────────────────────────────────────────────────────
{
  const h = linkify('a <b> & "x" tag');
  ok(h.includes('&lt;b&gt;'), 'escapes < >');
  ok(h.includes('&amp;'), 'escapes &');
  ok(h.includes('&quot;'), 'escapes "');
  ok(!h.includes('<b>'), 'no raw tag survives');
}

// ── 5. $math$ skipped (left verbatim, not linkified) ─────────────────────────
{
  const h = linkify('the norm $\\lVert v \\rVert_2$ here');
  ok(h.includes('$\\lVert v \\rVert_2$'), 'math token passed through verbatim');
  // A term name inside math must NOT be wrapped.
  const h2 = linkify('formula $SAC actor$ end');
  ok(h2.includes('$SAC actor$'), 'no terms wrapped inside math');
  ok(!h2.includes('data-term'), 'no data-term inside pure-math string');
}

// ── 6. word-boundary: "step" inside "timestep" is not double-wrapped ─────────
{
  const h = linkify('one timestep');
  // "timestep" is itself an aka of timestep; should match the whole word once.
  eq((h.match(/data-term/g) || []).length, 1, 'timestep wrapped once, not split');
}

// ── 7. unknown-term safety ────────────────────────────────────────────────────
ok(typeof openTerm === 'function', 'openTerm exists');
closeDrill(); // ensure starting closed
getEl('drill-backdrop').classList.add('hidden');
openTerm('NoSuchTerm12345'); // must not throw, must not open
ok(getEl('drill-backdrop').classList.contains('hidden'), 'unknown term does not open sheet');

// ── 8. drill push / deeper / pop / back-to-close ─────────────────────────────
closeDrill(); // reset
openTerm('vision-ablation sensitivity');
eq(getEl('drill-title').textContent, 'vision-ablation sensitivity', 'opened term shows title');
eq(getEl('drill-level-label').textContent, 'Level 1 of 3', 'starts at level 1');
deeper();
eq(getEl('drill-level-label').textContent, 'Level 2 of 3', 'deeper → level 2');
deeper();
eq(getEl('drill-level-label').textContent, 'Level 3 of 3', 'deeper → level 3');
deeper(); // cap: no level 4
eq(getEl('drill-level-label').textContent, 'Level 3 of 3', 'deeper capped at last level');

// push a nested term onto the breadcrumb stack
openTerm('L2 norm');
eq(getEl('drill-title').textContent, 'L2 norm', 'nested openTerm pushes new frame');
ok(getEl('drill-breadcrumb').innerHTML.includes('›'), 'breadcrumb shows separator after push');
ok(getEl('drill-breadcrumb').innerHTML.includes('vision-ablation sensitivity'), 'breadcrumb keeps parent');

// back pops to parent
drillBack();
eq(getEl('drill-title').textContent, 'vision-ablation sensitivity', 'back pops to parent');

// back at root closes the sheet
drillBack();
ok(getEl('drill-backdrop').classList.contains('hidden'), 'back at root closes sheet');

// ── 9. depth cap on the breadcrumb stack ─────────────────────────────────────
closeDrill();
for (let i = 0; i < 20; i++) openTerm('SAC'); // try to overgrow
// stack length is internal; assert the sheet is still functional (title set)
eq(getEl('drill-title').textContent, 'SAC', 'depth-capped stack still renders');
closeDrill();

// ── 10. flip toggles unlimited ────────────────────────────────────────────────
{
  const { flip } = sandbox;
  // showNext is needed to set `current`; but flip only toggles UI + `flipped`.
  // Set up the minimal DOM ids flip touches.
  ['card-inner','actions','followup-row','flip-btn-row'].forEach(getEl);
  const inner = getEl('card-inner');
  inner.classList.remove('flipped'); // normalize start state
  eq(inner.classList.contains('flipped'), false, 'starts unflipped');
  flip();
  eq(inner.classList.contains('flipped'), true, 'flip 1 → flipped');
  flip();
  eq(inner.classList.contains('flipped'), false, 'flip 2 → back to front');
  flip();
  eq(inner.classList.contains('flipped'), true, 'flip 3 → flipped again (unlimited)');
  flip();
  eq(inner.classList.contains('flipped'), false, 'flip 4 → unflipped (unlimited)');
}

// ── 11. glossary term count sanity ────────────────────────────────────────────
const termCount = Object.keys(GLOSSARY).length;
ok(termCount >= 40, 'glossary has >= 40 terms (got ' + termCount + ')');

// ── Report ────────────────────────────────────────────────────────────────────
console.log('\nGlossary terms:', termCount);
console.log('PASS:', pass, ' FAIL:', fail);
process.exit(fail === 0 ? 0 : 1);
