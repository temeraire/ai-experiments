"""check_prism_baseline.py — the gate that the 2026-07-20 three-arm run needed and did not have.

WHAT VOIDED THAT RUN. The transplanted mildhead bearing head outputs in the CRAWLER'S BODY/TORSO
frame, which sits ~90 deg from functional forward. Measured on the walker before training, it read
-85 deg against the true bearing and -95 deg against the seen bearing. So when the aux loss ran, the
arms spent the whole run dragging the readout ~90 deg into the eye frame; the 12.6 deg prism shift
rode on top of that and was swamped. The near-zero final numbers meant "the head now speaks the eye
frame", NOT "the eye realigned under the lens."

WHY THE EARLIER GATE MISSED IT — the rule this script exists to enforce. Gate P3 validated the
baseline with a freshly-FITTED RIDGE PROBE, and a fitted probe silently absorbs any constant frame
offset, so it reported ~0.1 deg and looked immaculate. The result was then scored with the encoder's
OWN bearing head, which carries the offset. **A baseline validated with one readout does not license
a result scored with a different one.** This script therefore measures with the BEARING HEAD, exactly
what the prism eval scores with.

USE: between the lens-off pre-adaptation phase and the prism phase.
  PASS  -> bias within +-2 deg in BOTH bands, lens OFF. The eye speaks the eye frame; proceed.
  FAIL  -> do NOT start the prism phase. Any realignment measured afterwards would be confounded with
           frame conversion, which is a ~90 deg effect against a ~13 deg signal.

  python -m alien_baby.crawler.check_prism_baseline --model alien_baby/results/prism_pre_s0_final.zip
"""
import argparse
import numpy as np, torch
from stable_baselines3 import PPO
from alien_baby.crawler.prism_bearing_env import PrismBearingEnv

TOL_DEG = 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--band-hi", type=float, default=0.6)
    p.add_argument("--acted-band", type=int, default=1)
    p.add_argument("--seed", type=int, default=9500)
    args = p.parse_args()

    m = PPO.load(args.model, device="cpu")
    fe = m.policy.features_extractor.eval()
    # LENS OFF: this gate is about frame alignment, not about the prism.
    env = PrismBearingEnv(prism_offset_deg=0.0, acted_band=args.acted_band,
                          band_hi=args.band_hi, seed=args.seed)
    rows = []
    for i in range(args.episodes):
        env.rng = np.random.default_rng(args.seed + i)
        o, _ = env.reset()
        with torch.no_grad():
            v = fe.bearing_pred(torch.as_tensor(o[None], dtype=torch.float32))[0].numpy()
        est = float(np.arctan2(v[0], v[1]))
        t = env.gaze_bearing_true()
        rows.append((env.band, np.degrees(np.arctan2(np.sin(est - t), np.cos(est - t)))))
    r = np.array(rows)

    print("=== PRISM BASELINE GATE (lens OFF, measured with the BEARING HEAD) ===")
    print(f"model: {args.model}")
    out = {}
    for name, sgn in [("ACTED", args.acted_band), ("SEEN-ONLY", -args.acted_band)]:
        b = r[r[:, 0] == sgn][:, 1]
        out[name] = float(np.median(b))
        print(f"  {name:10s} bias {np.median(b):+7.2f} deg   "
              f"IQR {np.percentile(b,25):+.1f}..{np.percentile(b,75):+.1f}   n={len(b)}")
    ok = all(abs(v) <= TOL_DEG for v in out.values())
    print(f"  --> {'PASS' if ok else 'FAIL'} (need |bias| <= {TOL_DEG} deg in BOTH bands)")
    if not ok:
        print("      DO NOT start the prism phase. The head is not yet in the gaze frame, so any")
        print("      realignment measured later is confounded with frame conversion (~90 deg effect")
        print("      against a ~13 deg signal) -- the failure that voided the first three-arm run.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
