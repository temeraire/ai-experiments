"""
probe_align.py — alignment step of the concept-anchoring probe (GROUNDING_LLMS.md §7).

Compare the GEOMETRY of AB's grounded latents against several LLM/embedding systems'
representation of the WORDS for the same scene states, via RSA (basis-independent).
The result is the PATTERN ACROSS THE LADDER, not any single number.

Ladder (backends):
  mpnet  — all-mpnet-base-v2 sentence embedding (local)
  openai — OpenAI text-embedding API (sentence embedding)
  qwen   — Qwen2.5-1.5B-Instruct last-token HIDDEN STATES (text-only LLM, mid layer)
  clip   — CLIP text encoder pooled output (VISION-GROUNDED text — the contrast)

Two axis-isolation methods (both, per user):
  grid     — 2-D bearing x distance phrase cells (full-phrase stimulus)
  isolated — embed single-axis descriptors alone: bearing-only (7) and distance-only (4),
             so the two axes don't shadow each other in the embedding

RDM = pairwise (1 - cosine); RSA = Spearman on upper triangles; significance by label-shuffle
permutation. Also reports how much each side encodes bearing vs distance (structural refs).

Usage:
  PYTHONPATH=<repo> python -u -m alien_baby.grounding.probe_align --all
  PYTHONPATH=<repo> python -u -m alien_baby.grounding.probe_align --backend mpnet --stimulus grid
"""
import argparse
import json
import os
import pathlib

import numpy as np
from scipy.stats import spearmanr

RESULTS = pathlib.Path(__file__).parent.parent / "results"

# grid (2-D) bins
BEARING_GRID = [("far left", -68, -40), ("left", -40, -15), ("ahead", -15, 15),
                ("right", 15, 40), ("far right", 40, 68)]
DIST_GRID = [("close", 0.0, 0.40), ("mid-range", 0.40, 0.63), ("far away", 0.63, 1.2)]
# isolated 1-D bins (finer bearing for power; distance kept coarse)
BEARING_ISO = [("far to my left", -68, -48), ("to my left", -48, -29),
               ("slightly left", -29, -10), ("straight ahead", -10, 10),
               ("slightly right", 10, 29), ("to my right", 29, 48),
               ("far to my right", 48, 68)]
DIST_ISO = [("very close", 0.0, 0.28), ("close", 0.28, 0.46),
            ("far", 0.46, 0.66), ("very far", 0.66, 1.2)]


# ---------- geometry ----------
def cosine_rdm(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return 1.0 - Xn @ Xn.T


def upper(M):
    return M[np.triu_indices_from(M, k=1)]


def rsa(a, b):
    return float(spearmanr(upper(a), upper(b)).correlation)


def perm_p(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = rsa(a, b); k = b.shape[0]; null = np.empty(n)
    for i in range(n):
        p = rng.permutation(k); null[i] = rsa(a, b[np.ix_(p, p)])
    return obs, float((np.sum(null >= obs) + 1) / (n + 1))


# ---------- backends (text -> [n, dim]) ----------
def embed_mpnet(texts):
    import torch
    from transformers import AutoTokenizer, AutoModel
    name = "sentence-transformers/all-mpnet-base-v2"
    tok = AutoTokenizer.from_pretrained(name); m = AutoModel.from_pretrained(name).eval()
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        h = m(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return ((h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy()


def embed_openai(texts):
    from openai import OpenAI
    model = os.environ.get("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-3-small")
    r = OpenAI().embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in r.data], dtype=np.float32)


def embed_qwen(texts):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float32, output_hidden_states=True).eval()
    enc = tok(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        hs = m(**enc).hidden_states
    layer = len(hs) // 2                      # mid layer
    h = hs[layer]                             # [B, T, H]
    last = enc["attention_mask"].sum(1) - 1   # last non-pad token
    return h[torch.arange(h.shape[0]), last].cpu().numpy()


def embed_clip(texts):
    import torch
    from transformers import CLIPTokenizer, CLIPTextModel
    name = "openai/clip-vit-base-patch32"
    tok = CLIPTokenizer.from_pretrained(name); m = CLIPTextModel.from_pretrained(name).eval()
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        return m(**enc).pooler_output.cpu().numpy()


BACKENDS = {"mpnet": embed_mpnet, "openai": embed_openai,
            "qwen": embed_qwen, "clip": embed_clip}
TEXT_ONLY = {"mpnet", "openai", "qwen"}


# ---------- cell builders ----------
def cells_grid(lat, th, dist, min_cell=6):
    vecs, texts, bidx, didx = [], [], [], []
    for bi, (bl, blo, bhi) in enumerate(BEARING_GRID):
        for di, (dl, dlo, dhi) in enumerate(DIST_GRID):
            m = (th >= blo) & (th < bhi) & (dist >= dlo) & (dist < dhi)
            if m.sum() < min_cell:
                continue
            vecs.append(lat[m].mean(0)); texts.append(f"a ball {dl}, {bl} of me")
            bidx.append(bi); didx.append(di)
    return np.array(vecs), texts, np.array(bidx), np.array(didx)


def cells_iso(lat, th, dist, axis, min_cell=6):
    vecs, texts, idx = [], [], []
    if axis == "bearing":
        for i, (lbl, lo, hi) in enumerate(BEARING_ISO):
            m = (th >= lo) & (th < hi)
            if m.sum() < min_cell:
                continue
            vecs.append(lat[m].mean(0)); texts.append(f"a ball {lbl}"); idx.append(i)
    else:
        for i, (lbl, lo, hi) in enumerate(DIST_ISO):
            m = (dist >= lo) & (dist < hi)
            if m.sum() < min_cell:
                continue
            vecs.append(lat[m].mean(0)); texts.append(f"a ball {lbl}"); idx.append(i)
    return np.array(vecs), texts, np.array(idx)


def idx_rdm(idx):
    return (idx[:, None] != idx[None, :]).astype(float)


# ---------- runners ----------
def run_grid(lat, th, dist, backend, fn):
    ab, texts, bidx, didx = cells_grid(lat, th, dist)
    llm = fn(texts)
    ab_rdm, llm_rdm = cosine_rdm(ab), cosine_rdm(llm)
    r, p = perm_p(ab_rdm, llm_rdm)
    return {"backend": backend, "stimulus": "grid", "n_cells": len(texts),
            "rsa": r, "perm_p": p,
            "ab_vs_bearing": rsa(ab_rdm, idx_rdm(bidx)),
            "ab_vs_distance": rsa(ab_rdm, idx_rdm(didx)),
            "llm_vs_bearing": rsa(llm_rdm, idx_rdm(bidx)),
            "llm_vs_distance": rsa(llm_rdm, idx_rdm(didx))}


def run_iso(lat, th, dist, axis, backend, fn):
    ab, texts, idx = cells_iso(lat, th, dist, axis)
    llm = fn(texts)
    r, p = perm_p(cosine_rdm(ab), cosine_rdm(llm))
    return {"backend": backend, "stimulus": f"iso-{axis}", "n_cells": len(texts),
            "rsa": r, "perm_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default=str(RESULTS / "grounding" / "ab_latents.npz"))
    ap.add_argument("--field", default="latent128")
    ap.add_argument("--inview-deg", type=float, default=68.0)
    ap.add_argument("--backend", default=None, choices=list(BACKENDS))
    ap.add_argument("--stimulus", default="grid", choices=["grid", "iso-bearing", "iso-distance"])
    ap.add_argument("--all", action="store_true", help="run full ladder x both methods")
    args = ap.parse_args()

    d = np.load(args.latents)
    lat, th, dist = d[args.field], d["theta_deg"], d["dist"]
    keep = np.abs(th) < args.inview_deg
    lat, th, dist = lat[keep], th[keep], dist[keep]
    print(f"AB states: {len(th)} in-view of {len(d['theta_deg'])} total ({args.field})\n")

    backends = list(BACKENDS) if args.all else [args.backend]
    rows = []
    for b in backends:
        fn = BACKENDS[b]
        try:
            rows.append(run_grid(lat, th, dist, b, fn))
            rows.append(run_iso(lat, th, dist, "bearing", b, fn))
            rows.append(run_iso(lat, th, dist, "distance", b, fn))
            print(f"[{b}] done")
        except Exception as e:
            print(f"[{b}] FAILED: {type(e).__name__}: {e}")

    print("\n=== RSA LADDER (AB grounded latent vs text-model geometry) ===")
    print(f"{'backend':8s} {'vision?':8s} {'grid':>14s} {'iso-bearing':>14s} {'iso-distance':>14s}")
    for b in backends:
        vis = "text" if b in TEXT_ONLY else "VISION"
        g = next((r for r in rows if r["backend"] == b and r["stimulus"] == "grid"), None)
        ib = next((r for r in rows if r["backend"] == b and r["stimulus"] == "iso-bearing"), None)
        idd = next((r for r in rows if r["backend"] == b and r["stimulus"] == "iso-distance"), None)
        def cell(x):
            return f"{x['rsa']:+.2f}(p{x['perm_p']:.2f})" if x else "   —   "
        print(f"{b:8s} {vis:8s} {cell(g):>14s} {cell(ib):>14s} {cell(idd):>14s}")
    g0 = next((r for r in rows if r["stimulus"] == "grid"), None)
    if g0:
        print(f"\nStructural refs (constant): AB latent encodes  bearing={g0['ab_vs_bearing']:+.2f} "
              f" distance={g0['ab_vs_distance']:+.2f}")
        print("Per-backend, how much the WORD geometry encodes each axis (grid):")
        for r in rows:
            if r["stimulus"] == "grid":
                print(f"  {r['backend']:8s} word_vs_bearing={r['llm_vs_bearing']:+.2f}  "
                      f"word_vs_distance={r['llm_vs_distance']:+.2f}")

    out = RESULTS / "grounding" / "align_results.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
