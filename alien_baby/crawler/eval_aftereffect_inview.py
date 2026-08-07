"""eval_aftereffect_inview.py — lens-OFF aftereffect at the eye/representation level,
measured ONLY on IN-VIEW samples (|theta_true| < FOV).

Why: eval_bearing_readout.py pools every step, including states where the ball is out of
the camera FOV and the bearing head necessarily outputs garbage (spawn cone 150 = +/-75 deg
> FOV ~ +/-68 deg). That garbage tanks circ_corr toward 0 and makes the signed bias
uninterpretable. Here we keep only samples where the ball is actually visible, so the eye
readout is measured where it can work.

AFTEREFFECT = signed(adapted @ offset 0) - signed(base @ offset 0), in-view only.
Predicted if the eye recalibrated under the +30 lens: the adapted eye keeps subtracting the
learned offset when the lens is gone, i.e. a NEGATIVE signed bias at offset 0 relative to base.
"""
import argparse, math
import numpy as np
import torch
from stable_baselines3 import PPO
from alien_baby.crawler.eval_prism_decoy import make_env


def cmean(a):
    return math.atan2(np.sin(a).sum(), np.cos(a).sum())


def circ_corr(A, B):
    am, bm = cmean(A), cmean(B)
    num = np.sum(np.sin(A - am) * np.sin(B - bm))
    den = math.sqrt(np.sum(np.sin(A - am) ** 2) * np.sum(np.sin(B - bm) ** 2))
    return num / den if den > 0 else float("nan")


def circ_dist(a, b):
    d = np.abs(a - b) % (2 * math.pi)
    return np.minimum(d, 2 * math.pi - d)


def rollout(model_path, offset, episodes, cone_deg, fov_deg, device):
    env = make_env(offset, cone_deg, [0.70, 0.80], 1000, 7)
    model = PPO.load(model_path, device=device)
    fe = model.policy.features_extractor
    th_vis, th_true = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        while True:
            with torch.no_grad():
                ot = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                sc = fe.bearing_pred(ot)[0].cpu().numpy()
            th_vis.append(math.atan2(sc[0], sc[1]))
            th_true.append(env._ball1_ego_bearing())
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
    V = np.array(th_vis); T = np.array(th_true)
    fov = math.radians(fov_deg)
    inview = np.abs(T) < fov
    Vi, Ti = V[inview], T[inview]
    return {
        "n_total": len(V), "n_inview": int(inview.sum()),
        "inview_frac": float(inview.mean()),
        "cc_all": circ_corr(V, T),
        "cc_inview": circ_corr(Vi, Ti) if inview.sum() > 5 else float("nan"),
        "mae_inview": float(np.degrees(circ_dist(Vi, Ti).mean())) if inview.sum() else float("nan"),
        "signed_inview": math.degrees(cmean(Vi - Ti)) if inview.sum() else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapted", required=True)
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--cone-deg", type=float, default=150.0)
    p.add_argument("--fov-deg", type=float, default=68.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    b = rollout(args.base, args.offset, args.episodes, args.cone_deg, args.fov_deg, args.device)
    a = rollout(args.adapted, args.offset, args.episodes, args.cone_deg, args.fov_deg, args.device)

    def show(tag, r):
        print(f"{tag}: n_inview={r['n_inview']}/{r['n_total']} ({r['inview_frac']*100:.0f}%)  "
              f"cc_all={r['cc_all']:+.3f} cc_inview={r['cc_inview']:+.3f}  "
              f"MAE_inview={r['mae_inview']:.1f}  signed_inview={r['signed_inview']:+.1f} deg")
    print(f"=== lens-off aftereffect, IN-VIEW only (|theta_true|<{args.fov_deg}), offset={args.offset} ===")
    show("BASE   ", b)
    show("ADAPTED", a)
    ae = a["signed_inview"] - b["signed_inview"]
    print(f"AFTEREFFECT (adapted - base signed bias, in-view) = {ae:+.1f} deg")
    print(f"  [negative => eye keeps subtracting the trained +offset with the lens off = genuine aftereffect]")
    print(f"  sanity: base cc_inview should approach the validated ~+0.5 if the eye is being read correctly")


if __name__ == "__main__":
    main()
