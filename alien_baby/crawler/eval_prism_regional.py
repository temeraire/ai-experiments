"""eval_prism_regional.py — did realignment stay LOCAL to the band the creature acted in?

Scores the question the whole experiment exists for. For each trained encoder we sweep SINGLE balls
across ALL bearings (no distractor, so nothing to suppress), read the encoder's bearing estimate, and
report the SIGNED bias separately for the ACTED band and the SEEN-ONLY band.

READING THE NUMBERS. Under a lens of effective +theta, an UNrealigned eye reports the ball where the
ghost appears, i.e. bias ~ +theta. A fully realigned eye reports where the ball actually IS, i.e.
bias ~ 0. So realignment = bias travelling from +theta down toward 0.

  Taylor's prediction  : ACTED band -> ~0, SEEN-ONLY band -> stays ~+theta   (locality)
  Rival (weight-share) : BOTH bands -> ~0                                     (global realignment)
  Plumbing control     : arm=none -> BOTH stay ~+theta (the eye cannot change)

The readout is the encoder's OWN bearing head (`bearing_pred`), which is what the aux loss trains and
therefore what "the eye's estimate" means here -- not a fresh probe fitted post hoc, which would
measure decodability rather than the eye's own claim.

  python -m alien_baby.crawler.eval_prism_regional --models prism_act_s0 prism_clamp_s0 prism_none_s0
"""
import argparse
import numpy as np, torch
from stable_baselines3 import PPO
from alien_baby.crawler.prism_bearing_env import PrismBearingEnv, PROP_CUE


def sweep(model_path, prism_deg, band_hi, acted_band, n=400, seed=9000, device="cpu"):
    m = PPO.load(model_path, device=device)
    fe = m.policy.features_extractor.eval()
    env = PrismBearingEnv(prism_offset_deg=prism_deg, acted_band=acted_band,
                          band_hi=band_hi, seed=seed)
    rows = []
    for i in range(n):
        env.rng = np.random.default_rng(seed + i)
        o, _ = env.reset()
        with torch.no_grad():
            p = fe.bearing_pred(torch.as_tensor(o[None], dtype=torch.float32))[0].numpy()
        est = float(np.arctan2(p[0], p[1]))
        true = env.gaze_bearing_true()
        seen = env.gaze_bearing_seen()
        err = float(np.arctan2(np.sin(est - true), np.cos(est - true)))   # est minus TRUE
        rows.append((env.band, np.degrees(err), np.degrees(seen - true), np.degrees(true)))
    return np.array(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--prism-deg", type=float, default=15.0)
    p.add_argument("--band-hi", type=float, default=0.6)
    p.add_argument("--acted-band", type=int, default=1)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--ckpt", default="best_model.zip")
    args = p.parse_args()

    print("=== PRISM REGIONAL DISSOCIATION ===")
    print("signed bias = the eye's estimate MINUS the ball's TRUE bearing.")
    print("  ~ +theta  => NOT realigned (still reports where the ghost appears)")
    print("  ~ 0       => realigned     (reports where the ball actually is)\n")
    for tag in args.models:
        path = f"alien_baby/results/{tag}_final.zip"
        try:
            r = sweep(path, args.prism_deg, args.band_hi, args.acted_band, args.episodes)
        except Exception as e:
            print(f"{tag:18s} COULD NOT LOAD ({type(e).__name__}: {e})"); continue
        theta = np.median(r[:, 2])
        acted = r[r[:, 0] == args.acted_band]
        seen = r[r[:, 0] == -args.acted_band]
        ab, sb = np.median(acted[:, 1]), np.median(seen[:, 1])
        # realignment fraction: 1.0 = fully corrected, 0.0 = untouched
        fa = 1.0 - ab / theta if abs(theta) > 1e-6 else float("nan")
        fs = 1.0 - sb / theta if abs(theta) > 1e-6 else float("nan")
        print(f"{tag:18s} effective lens {theta:+5.1f} deg")
        print(f"    ACTED band     bias {ab:+6.2f} deg   realigned {100*fa:5.1f}%   n={len(acted)}")
        print(f"    SEEN-ONLY band bias {sb:+6.2f} deg   realigned {100*fs:5.1f}%   n={len(seen)}")
        gap = fa - fs
        print(f"    --> locality gap {100*gap:+5.1f} pts  "
              f"[{'LOCAL (Taylor)' if gap > 0.25 else 'GLOBAL / no dissociation'}]\n")


if __name__ == "__main__":
    main()
