"""A/B re-score of the prism aftereffect arc with the DIRECTION-CONDITIONAL metric
(the aggregate choice_vs_true hides a ~30 deg global shift). Pure re-analysis of
JSON already on disk. Run from repo root with .venv python.

A: conditional phantom-capture for aftereffect cells (s0/s2/frozen) + de-adaptation
   (washout) curve block-by-block.
B: generalization-of-offset — is the aftereffect a UNIFORM global bias or does it
   vary by bearing? (i) bin the s0 aftereffect episodes by red bearing; (ii) the
   bearing_gen eval (blue planted AT the phantom, swept across bearings).
"""
import json, math, glob, os

R = "alien_baby/results"
TRAIN_OFFSET = 30.0            # adaptation was +30 deg; aftereffect phantom = red - 30
PHANTOM = -TRAIN_OFFSET

def circdist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

def load(name):
    f = f"{R}/{name}.json"
    return json.load(open(f)) if os.path.exists(f) else None

def wilson(k, n):
    if n == 0: return "        -    "
    p = k/n; z = 1.96; d = 1 + z*z/n
    c = (p + z*z/2/n)/d; h = z*math.sqrt(p*(1-p)/n + z*z/4/n/n)/d
    return f"{100*p:4.0f}% [{100*(c-h):3.0f},{100*(c+h):3.0f}]"

def phantom_split(eps):
    """blue-capture rate when blue is nearer the PHANTOM red position than the true
    red position, vs when it isn't. A negative aftereffect => near >> far."""
    near_k = near_n = far_k = far_n = 0
    for e in eps:
        ph = e['th_red'] + PHANTOM
        blue_near = circdist(e['th_blue'], ph) < circdist(e['th_blue'], e['th_red'])
        cap = (e['out'] == 'blue')
        if blue_near: near_n += 1; near_k += cap
        else:         far_n += 1; far_k += cap
    return (near_k, near_n, far_k, far_n)

print("="*74)
print("A1 — AFTEREFFECT conditional phantom-capture (prism OFF; near>>far = aftereffect)")
print("="*74)
print(f"  {'cell':<26}{'aggP(red)':>9}   {'blue-cap|near-phantom':>22}   {'blue-cap|far':>14}")
for name, lab in [("prism_battery_aftereffect_off0","s0 aftereffect"),
                  ("prism_battery_aftereffect_s2_off0","s2 aftereffect"),
                  ("prism_battery_aftereffect_frozen_off0","frozen-enc aftereffect")]:
    d = load(name)
    if not d: print(f"  {lab:<26} MISSING"); continue
    nk,nn,fk,fn = phantom_split(d['episodes'])
    agg = 100*d['red']/(d['red']+d['blue'])
    print(f"  {lab:<26}{agg:8.0f}%   {wilson(nk,nn):>22}   {wilson(fk,fn):>14}")

print("\n" + "="*74)
print("A2 — DE-ADAPTATION (washout): does the phantom split decay to symmetric?")
print("="*74)
print(f"  {'deadapt step':>13}   {'blue-cap|near':>18}   {'blue-cap|far':>14}   near-far")
for f in sorted(glob.glob(f"{R}/prism_battery_washout_*.json"),
                key=lambda x:int(x.split('_')[-1].split('.')[0])):
    step = int(f.split('_')[-1].split('.')[0]); d = json.load(open(f))
    if not d.get('episodes'): print(f"  {step:>13}   (no episodes)"); continue
    nk,nn,fk,fn = phantom_split(d['episodes'])
    npct = 100*nk/nn if nn else 0; fpct = 100*fk/fn if fn else 0
    print(f"  {step:>13}   {wilson(nk,nn):>18}   {wilson(fk,fn):>14}   {npct-fpct:+5.0f}")

print("\n" + "="*74)
print("B1 — GENERALIZATION-OF-OFFSET: is s0's aftereffect uniform across bearing?")
print("     (bin s0 aftereffect episodes by |red bearing|; global bias => same split")
print("      in every bin; structured remap => split varies)")
print("="*74)
d = load("prism_battery_aftereffect_off0")
eps = d['episodes']
for lo, hi, lab in [(0,25,"red central |bearing|<25"),
                    (25,50,"red mid 25-50"),
                    (50,200,"red peripheral >50")]:
    sub = [e for e in eps if lo <= abs(e['th_red']) < hi]
    nk,nn,fk,fn = phantom_split(sub)
    npct = 100*nk/nn if nn else 0; fpct = 100*fk/fn if fn else 0
    print(f"  {lab:<26} n={len(sub):3d}  near-phantom {wilson(nk,nn)}  far {wilson(fk,fn)}  diff {npct-fpct:+.0f}")

print("\n" + "="*74)
print("B2 — bearing_gen probe (blue planted AT phantom, swept). adapted-control capture")
print("     per bearing = aftereffect magnitude; uniform => global, varying => structured")
print("="*74)
ctl = load("bearing_gen_control"); adp = load("bearing_gen_adapted")
if ctl and adp:
    print(f"  {'red bearing':>12}   {'control cap':>11}   {'adapted cap':>11}   {'aftereffect(adp-ctl)':>20}")
    for b in sorted(ctl, key=lambda x:int(x)):
        cc = ctl[b]['capture']*100; ac = adp[b]['capture']*100
        print(f"  {b:>12}   {cc:9.0f}%   {ac:9.0f}%   {ac-cc:+17.0f}")
else:
    print("  bearing_gen files missing")
