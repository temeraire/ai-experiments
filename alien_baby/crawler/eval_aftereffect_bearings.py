"""
eval_aftereffect_bearings.py — generalization-of-offset test (no prism, no training).

Distinguishes "global -30deg bias" from "structured spatial remap": place RED at a
target bearing (jittered) and BLUE exactly at the aftereffect position (red-30deg),
measure P(capture blue) per bearing bin. A single global bias predicts a FLAT capture
profile across bearings; a structured remap predicts variation with bearing.
Run on both the adapted model and the pre-adaptation control (capture should be low
and uniform for the control).

  PYTHONPATH=<repo> python -u -m alien_baby.crawler.eval_aftereffect_bearings \
    --model alien_baby/results/prism_adapt_s0_final.zip --run-tag adapted
"""
import argparse
import json
import pathlib

import numpy as np

from stable_baselines3 import PPO
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES

XML = "alien_baby/crawler/mimo_crawler_pos_wide_hs.xml"
RESULTS = pathlib.Path(__file__).parent.parent / "results"
BEARINGS = [-60, -30, 0, 30, 60]          # red bearing bin centers (deg)
DECOY_SIDE = -30                          # blue planted at red+DECOY_SIDE (aftereffect side)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--eps-per-bearing", type=int, default=30)
    p.add_argument("--jitter-deg", type=float, default=8.0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-tag", default="bearing_gen")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = MimoCrawlerEnv(
        vision=True, stereo=True, target_obs=False, crawl_pose=CRAWL_POSES["arms_fwd"],
        action_mode="position_offset", xml_path=XML, max_steps=args.max_steps,
        step_cost=0.0, approach_reward_scale=10.0, velocity_bonus_scale=0.0,
        terminate_tilt_deg=50.0, tip_penalty=-5.0, decoy_ball=True,
        fixed_ball_positions=[(0.0, 0.75), (0.3, 0.7)],   # replaced per episode
    )
    env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"\n=== bearing generalization: {args.run_tag} ===  (blue at red{DECOY_SIDE:+}deg)")
    out = {}
    for b in BEARINGS:
        red = blue = none = 0
        for _ in range(args.eps_per_bearing):
            th = np.deg2rad(b + rng.uniform(-args.jitter_deg, args.jitter_deg))
            r = rng.uniform(0.70, 0.80)
            th_b = th + np.deg2rad(DECOY_SIDE)
            env.fixed_ball_positions = [
                (r * np.sin(th), r * np.cos(th)),
                (r * np.sin(th_b), r * np.cos(th_b)),
            ]
            obs, _ = env.reset()
            while True:
                act, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(act)
                if term or trunc:
                    break
            if info["touched_ball1"]:
                red += 1
            elif info["touched_ball2"]:
                blue += 1
            else:
                none += 1
        cap = blue / max(1, red + blue)
        out[b] = {"red": red, "blue": blue, "neither": none, "capture": cap}
        print(f"  red@{b:+3d}deg: capture {cap*100:5.1f}%  ({blue}B/{red}R, {none} neither)")

    f = RESULTS / f"bearing_gen_{args.run_tag}.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"  wrote {f}")


if __name__ == "__main__":
    main()
