#!/usr/bin/env python3
"""Audit: re-score the ECCENTRICITY sweep under BROAD vs HAND-ONLY touch.

The Phase XIII/XIV headlines ("proprio generalizes lawfully", "20/20 at ecc=0",
the distance law) rest on R44 (proprio ecc sweep) and R45 (DroQ validation), both
on the same cart env as Phase XV. The Phase XV hand-touch finding (both_HAND 0/30)
raises the question: are those reach results also body/cart-delivered? This re-scores
the ecc sweep capturing both metrics so we can see how far the confound reaches.
"""
import numpy as np
from alien_baby.visualization.eval_phase_v import build_env_and_model, MAX_STEPS

RUNS = {"R44(XIII)": "phase_v_R44_proprio_static_randbox",
        "R45(XIV)":  "phase_w_R45_droq_proprio_validation"}
ECCS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
EPS, SEED = 30, 10_000


def episode(seed, ve, m):
    raw = ve.venv.envs[0]
    obs, _ = raw.reset(seed=seed); obs = ve.normalize_obs(obs.reshape(1, -1))
    b1 = b2 = h1 = h2 = False
    fh = None
    for step in range(MAX_STEPS):
        a, _ = m.predict(obs, deterministic=True)
        o, r, term, trunc, info = raw.step(a[0]); obs = ve.normalize_obs(o.reshape(1, -1))
        if (info["touched_ball1"] or info["touched_ball2"]) and fh is None and step > 0:
            fh = info["hand_touched_ball1"] or info["hand_touched_ball2"]
        b1 |= info["touched_ball1"]; b2 |= info["touched_ball2"]
        h1 |= info["hand_touched_ball1"]; h2 |= info["hand_touched_ball2"]
        if term or trunc: break
    return (b1 and b2), (h1 and h2), (h1 or h2), bool(fh)


print(f"\n{'='*72}\nECC sweep: BROAD vs HAND-ONLY  (eps={EPS}, seed={SEED}, default radius)\n{'='*72}")
print(f"{'run':>10} {'ecc':>5} | {'both_BROAD':>10} | {'both_HAND':>9} | {'any_HAND':>8} | {'1st=hand%':>9}")
print("-"*70)
for tag, rt in RUNS.items():
    for ecc in ECCS:
        ve, m, _ = build_env_and_model(rt, eccentricity=ecc, ball_speed=0.0,
                                       ball_radius=None, ball_y=0.35)
        res = [episode(SEED + i, ve, m) for i in range(EPS)]
        ve.close()
        bb = sum(r[0] for r in res); bh = sum(r[1] for r in res)
        ah = sum(r[2] for r in res); fh = np.mean([r[3] for r in res])
        print(f"{tag:>10} {ecc:>5.2f} | {bb:>4}/{EPS}     | {bh:>3}/{EPS}    | "
              f"{ah:>3}/{EPS}  | {100*fh:>7.0f}%")
print("\nboth_BROAD=headline metric; both_HAND=both reached by hand; any_HAND=>=1 hand touch.")
