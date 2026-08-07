"""Summarize the mismatched-shape decoy battery: choice_vs_true (->RED) per condition,
with Wilson CIs and a color-vs-shape read. Run from repo root with .venv python."""
import json, math, glob, os, sys

R = "alien_baby/results"

def wilson(r, n):
    if n == 0: return (None, None, None)
    p = r/n; z = 1.96; d = 1 + z*z/n
    c = (p + z*z/2/n)/d; h = z*math.sqrt(p*(1-p)/n + z*z/4/n/n)/d
    return (100*p, 100*(c-h), 100*(c+h))

def load(tag):
    f = f"{R}/decoy_shape_{tag}.json"
    if not os.path.exists(f): return None
    d = json.load(open(f))
    return d

def row(tag, label):
    d = load(tag)
    if d is None: return f"  {label:<34} (pending)"
    r, b, nei = d["red"], d["blue"], d["neither"]
    p, lo, hi = wilson(r, r+b)
    if p is None: return f"  {label:<34} no decisions"
    return (f"  {label:<34} {p:5.1f}%  [{lo:4.1f},{hi:4.1f}]   "
            f"{r:3d}R/{b:3d}B  neither={nei}")

def bearing_bins(tag):
    d = load(tag)
    if d is None: return None
    eps = d["episodes"]
    edges = [(-200,-40),(-40,40),(40,200)]; names=["farL","central","farR"]
    out=[]
    for (lo,hi),nm in zip(edges,names):
        sub=[e for e in eps if lo<=e['th_red']<hi]
        r=sum(e['out']=='red' for e in sub); b=sum(e['out']=='blue' for e in sub)
        out.append(f"{nm}:{(str(round(100*r/(r+b)))+'%') if (r+b) else '-'}")
    return " ".join(out)

for seed, tags in [("EXT_S0", [
        ("ext_s0_A_sph_sph", "A red=SPH blue=SPH (control)"),
        ("ext_s0_B_box_sph", "B red=BOX blue=SPH"),
        ("ext_s0_C_sph_box", "C red=SPH blue=BOX"),
        ("ext_s0_D_box_box", "D red=BOX blue=BOX"),
        ("ext_s0_E_cap_sph", "E red=CAP blue=SPH"),
        ("ext_s0_F_sph_cap", "F red=SPH blue=CAP"),
        ("ext_s0_A_sph_sph_abl", "A' ablated floor (sph/sph)"),
        ("ext_s0_B_box_sph_abl", "B' ablated floor (box/sph)"),
    ]), ("S2", [
        ("s2_A_sph_sph", "A red=SPH blue=SPH (control)"),
        ("s2_B_box_sph", "B red=BOX blue=SPH"),
        ("s2_C_sph_box", "C red=SPH blue=BOX"),
        ("s2_D_box_box", "D red=BOX blue=BOX"),
    ])]:
    print(f"\n==== {seed}: choice_vs_true (-> RED = target_geom), 200 ep ====")
    print(f"  {'condition':<34} {'P(red)':>6}  {'95% CI':>13}")
    for tag, label in tags:
        print(row(tag, label))
        bb = bearing_bins(tag)
        if bb and not tag.endswith("abl"): print(f"  {'':<34}   by red-bearing: {bb}")

print("""
READING GUIDE
  color-driven (shape-invariant): B ~= C ~= A  (always prefers red regardless of shape)
  sphere-preference:              B << 50 and C >~ A  (goes to whichever is the SPHERE)
  ablated floors A',B' should sit ~50%; if B' strays, the box creates a NON-visual
  (touch/physics/placement) asymmetry that must be subtracted before reading B.
""")
