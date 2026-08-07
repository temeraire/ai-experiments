"""eval_bearing_readout.py — does the eye read the ball's direction from pixels?

Runs a mismatch-trained policy and, at each step, compares the encoder's bearing head
(theta_vis, "where the eye says the ball is") against the TRUE ego-bearing to the real
ball (theta_true = env._ball1_ego_bearing, matched to the same state). Reports:
  - circular correlation cc(theta_vis, theta_true)  [does the eye TRACK the ball's direction]
  - mean |theta_vis - theta_true|                   [absolute aiming error, degrees]
  - mean SIGNED (theta_vis - theta_true)            [constant bias; under a +offset lens this
                                                     is ~+offset before recalibration, ~0 after]

At offset 0 this measures whether the eye is grounded straight. At offset>0 it measures
whether the eye RE-ALIGNED (signed bias should collapse from +offset toward 0).
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prism-offset", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--cone-deg", type=float, default=150.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-tag", default=None)
    args = p.parse_args()

    env = make_env(args.prism_offset, args.cone_deg, [0.70, 0.80], 1000, 7)
    model = PPO.load(args.model, device=args.device)
    fe = model.policy.features_extractor

    th_vis, th_true = [], []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        while True:
            with torch.no_grad():
                ot = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
                sc = fe.bearing_pred(ot)[0].cpu().numpy()
            th_vis.append(math.atan2(sc[0], sc[1]))
            th_true.append(env._ball1_ego_bearing())   # true bearing for THIS obs/state
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break

    V = np.array(th_vis); T = np.array(th_true)
    cc = circ_corr(V, T)
    mae = float(np.degrees(circ_dist(V, T).mean()))
    signed = math.degrees(cmean(V - T))
    print(f"model={args.model.split('/')[-1]} offset={args.prism_offset} n_steps={len(V)}")
    print(f"  circ_corr(theta_vis, theta_true) = {cc:+.3f}   [eye tracks ball direction]")
    print(f"  mean |theta_vis - theta_true|    = {mae:.1f} deg  [absolute aiming error]")
    print(f"  mean signed (vis - true)         = {signed:+.1f} deg  [bias; ~+offset pre-realign, ~0 post]")


if __name__ == "__main__":
    main()
