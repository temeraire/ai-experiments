"""
eval_prism_decoy.py — the PRISM DISPLACEMENT battery on the two-ball decoy task.

Whole-field prism: both real balls stay solid at their true bearings but are
hidden from the head-cam; a red and a blue GHOST appear at the true bearings
rotated by --prism-offset. If vision drives target selection, choice should
follow the displaced picture; if vision only gates arousal, choice tracks the
true layout. Pre-registered decision rules: THEORY_LOG.md 2026-07-06 Q4 +
the strategist protocol (FINDINGS 2026-07-07).

Per-cell metrics:
  choice_vs_true      red/(red+blue) by REAL ball touched (blind floor 50%)
  wrong_ball          blue touches / episodes
  neither             timeouts+tips / episodes
  displayed_red_frac  of committed episodes, fraction whose net-CoM-displacement
                      bearing is nearest theta_red+offset among
                      {theta_red, theta_red+offset, theta_blue+offset}

Usage:
  PYTHONPATH=<repo> python -u -m alien_baby.crawler.eval_prism_decoy \
    --model alien_baby/results/decoy_v2_ext_s0_best/best_model.zip \
    --prism-offset 45 --eval-eps 100 --run-tag prism_ext_s0_off45
"""
import argparse
import json
import pathlib

import numpy as np
import mujoco

from stable_baselines3 import PPO
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES, VISION_DIM_STEREO

XML = "alien_baby/crawler/mimo_crawler_pos_wide_prism.xml"
RESULTS = pathlib.Path(__file__).parent.parent / "results"


def make_env(offset_deg, cone_deg, radius, max_steps, seed):
    env = MimoCrawlerEnv(
        vision=True, stereo=True, target_obs=False, crawl_pose=CRAWL_POSES["arms_fwd"],
        action_mode="position_offset", xml_path=XML, spawn_cone_deg=cone_deg,
        spawn_radius=tuple(radius), random_start_orientation=False, max_steps=max_steps,
        step_cost=0.0, approach_reward_scale=10.0, velocity_bonus_scale=0.0,
        terminate_tilt_deg=50.0, tip_penalty=-5.0, decoy_ball=True,
        prism_offset_deg=offset_deg,
    )
    env.reset(seed=seed)
    return env


def zero_pixels(obs):
    o = obs.copy(); o[-VISION_DIM_STEREO:] = 0.0; return o


def ball_bearing(env, joint):
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qa = env.model.jnt_qposadr[jid]
    x, y = env.data.qpos[qa:qa + 2]
    return float(np.arctan2(x, y))


def circ_dist(a, b):
    d = abs(a - b) % (2 * np.pi)
    return min(d, 2 * np.pi - d)


def run_cell(model, env, n_eps, max_steps, offset_rad, ablate=False):
    red = blue = neither = 0
    disp_red_hits = committed = 0
    episodes = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        th_red = ball_bearing(env, "target_free")
        th_blue = ball_bearing(env, "target2_free")
        root0 = obs[:2].copy()
        while True:
            o = zero_pixels(obs) if ablate else obs
            act, _ = model.predict(o, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        move = obs[:2] - root0
        heading = float(np.arctan2(move[0], move[1])) if np.linalg.norm(move) > 0.05 else None
        out = "red" if info["touched_ball1"] else ("blue" if info["touched_ball2"] else "neither")
        if out == "red":
            red += 1
        elif out == "blue":
            blue += 1
        else:
            neither += 1
        nearest = None
        if heading is not None:
            cands = {"true_red": th_red, "disp_red": th_red + offset_rad,
                     "disp_blue": th_blue + offset_rad}
            nearest = min(cands, key=lambda k: circ_dist(heading, cands[k]))
            if out != "neither":
                committed += 1
                disp_red_hits += int(nearest == "disp_red")
        episodes.append({"out": out, "th_red": round(np.rad2deg(th_red), 1),
                         "th_blue": round(np.rad2deg(th_blue), 1),
                         "heading": None if heading is None else round(np.rad2deg(heading), 1),
                         "nearest": nearest, "steps": info["step"]})
    dec = red + blue
    return {
        "n": n_eps, "red": red, "blue": blue, "neither": neither,
        "choice_vs_true": red / dec if dec else None,
        "wrong_ball": blue / n_eps,
        "displayed_red_frac": disp_red_hits / committed if committed else None,
        "committed": committed,
        "episodes": episodes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prism-offset", type=float, default=0.0)
    p.add_argument("--eval-eps", type=int, default=100)
    p.add_argument("--cone-deg", type=float, default=136.0)
    p.add_argument("--radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--run-tag", default="prism_decoy")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = make_env(args.prism_offset, args.cone_deg, args.radius, args.max_steps, args.seed)
    r = run_cell(model, env, args.eval_eps, args.max_steps,
                 np.deg2rad(args.prism_offset), ablate=args.ablate)

    cvt = r["choice_vs_true"]
    drf = r["displayed_red_frac"]
    se = (np.sqrt(cvt * (1 - cvt) / (r["red"] + r["blue"])) if cvt is not None else 0)
    print(f"\n=== {args.run_tag}  offset={args.prism_offset:.0f}deg"
          f"  {'ABLATED' if args.ablate else 'sighted'} ===")
    print(f"  choice_vs_true     : {cvt*100:.1f}% +/- {se*196:.1f}"
          f"  ({r['red']}R/{r['blue']}B, {r['neither']} neither)")
    print(f"  wrong_ball         : {r['wrong_ball']*100:.1f}%")
    print(f"  displayed_red_frac : "
          f"{'n/a' if drf is None else f'{drf*100:.1f}%'} (3-way chance 33%)"
          f"  [{r['committed']} committed]")
    out = RESULTS / f"prism_battery_{args.run_tag}.json"
    out.write_text(json.dumps({"args": vars(args), **r}, indent=1))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
