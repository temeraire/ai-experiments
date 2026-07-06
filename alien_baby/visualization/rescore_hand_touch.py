#!/usr/bin/env python3
"""Re-score size sweep under BROAD (any body part) vs HAND-ONLY touch.

`touched_ball*` (used for success/reward/termination) fires on contact with ANY
MIMo geom -- incl. a leg, or the cart pushing the ball into a static body part.
The env also computes a stricter hand-only flag. This measures how much of the
"both-touched" success is actually hand reaches vs incidental/cart-delivered
contact, across the size band that showed the 0.075 trough.
"""
import numpy as np
from alien_baby.visualization.eval_phase_v import build_env_and_model, MAX_STEPS

RUNS = {"R46": "phase_x_R46_droq_proprio_sizevariety",
        "R48": "phase_x3_R48_droq_proprio_size_and_shape"}
SIZES = [0.053, 0.075, 0.090]
EPS, SEED, ECC = 30, 20_000, 0.10


def episode(seed, vec_env, model):
    raw = vec_env.venv.envs[0]
    obs, _ = raw.reset(seed=seed)
    obs = vec_env.normalize_obs(obs.reshape(1, -1))
    b1 = b2 = h1 = h2 = False
    first_is_hand = None
    for step in range(MAX_STEPS):
        a, _ = model.predict(obs, deterministic=True)
        o, r, term, trunc, info = raw.step(a[0])
        obs = vec_env.normalize_obs(o.reshape(1, -1))
        nb = info["touched_ball1"] or info["touched_ball2"]
        nh = info["hand_touched_ball1"] or info["hand_touched_ball2"]
        if nb and first_is_hand is None and step > 0:
            first_is_hand = nh
        b1 |= info["touched_ball1"]; b2 |= info["touched_ball2"]
        h1 |= info["hand_touched_ball1"]; h2 |= info["hand_touched_ball2"]
        if term or trunc:
            break
    return (b1 and b2), (h1 and h2), bool(first_is_hand)


print(f"\n{'='*70}\nBROAD vs HAND-ONLY touch  (eps={EPS}, ecc={ECC}, seed={SEED})\n{'='*70}")
print(f"{'run':>4} {'size':>6} | {'both_BROAD':>10} | {'both_HAND':>9} | {'1st-touch hand%':>15}")
print("-"*60)
for tag, rt in RUNS.items():
    for sz in SIZES:
        ve, m, _ = build_env_and_model(rt, eccentricity=ECC, ball_speed=0.0,
                                       ball_radius=sz, ball_y=0.35)
        res = [episode(SEED + i, ve, m) for i in range(EPS)]
        ve.close()
        broad = sum(r[0] for r in res)
        hand = sum(r[1] for r in res)
        fh = [r[2] for r in res]
        print(f"{tag:>4} {sz:>6.3f} | {broad:>4}/{EPS}     | {hand:>3}/{EPS}    | "
              f"{100*np.mean(fh):>13.0f}%")
print("\nboth_BROAD = success as currently scored; both_HAND = both reached with a hand.")
print("1st-touch hand% = fraction of episodes whose FIRST contact was a hand (not leg/cart).")
