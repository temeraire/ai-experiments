"""eval_reach_aftereffect.py — SINGLE-TARGET, NO-FEEDBACK reach aftereffect probe.

Cleaner lens-off aftereffect than the two-ball decoy version. Two confounds removed:
  * SINGLE ball (decoy_ball=False) -> no target-SELECTION channel; heading is pure aim at
    the one target, not "which of two balls did vision pick".
  * EARLY-WINDOW heading (first --window steps, before the body reaches the ball) -> the
    "no-feedback" / ballistic aim, before touch-homing corrects it. The prior decoy probe
    measured whole-episode net heading, which touch-homing pulls back onto whatever ball got
    contacted, masking the eye's residual bias.

Design: full 2x2 {base,adapted} x {sighted,blind} at offset 0 (lens removed), same seed so
spawns are matched per episode index. Aftereffect = difference-in-differences of the signed
early-heading error (heading - true_bearing), which nets out any motor-bias drift between the
base and the fine-tuned checkpoint (the artifact that halved the last result). A NEGATIVE DiD
= the adapted eye still aims short/left by the trained +offset with the lens gone = genuine
vision-mediated aftereffect.

Usage:
  PYTHONPATH=<repo> python alien_baby/crawler/eval_reach_aftereffect.py \
    --base   alien_baby/results/stage1_ground_smoke_s0_best/best_model.zip \
    --adapted alien_baby/results/stage2_adapt_s0_best/best_model.zip \
    --offset 0 --episodes 300 --window 40 --run-tag reach_ae
"""
import argparse, json, math, pathlib
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
        terminate_tilt_deg=50.0, tip_penalty=-5.0, decoy_ball=False,   # SINGLE TARGET
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


def signed_deg(a, b):
    """signed circular (a-b) in degrees, in (-180,180]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(d)


def run_cell(model, env, n_eps, window, min_move, ablate):
    """Return per-episode list of {th_true, early_heading_err, full_heading_err} (deg)."""
    eps = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        th_true = ball_bearing(env, "target_free")
        root0 = obs[:2].copy()
        pos_at_window = None
        step = 0
        while True:
            o = zero_pixels(obs) if ablate else obs
            act, _ = model.predict(o, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            step += 1
            if step == window:
                pos_at_window = obs[:2].copy()
            if term or trunc:
                break
        # early (ballistic, no-feedback) heading: displacement over first `window` steps.
        # If the episode ENDED before `window` (fast reach / tip), there is no clean pre-contact
        # window -> undefined (do NOT fall back to the contaminated final position).
        if pos_at_window is None:
            early_h = None
        else:
            emove = pos_at_window - root0
            early_h = float(np.arctan2(emove[0], emove[1])) if np.linalg.norm(emove) > min_move else None
        # whole-episode heading (touch-homing-contaminated, kept for comparison)
        fmove = obs[:2] - root0
        full_h = float(np.arctan2(fmove[0], fmove[1])) if np.linalg.norm(fmove) > min_move else None
        eps.append({
            "th_true": round(math.degrees(th_true), 2),
            "early_err": None if early_h is None else round(signed_deg(early_h, th_true), 2),
            "full_err":  None if full_h  is None else round(signed_deg(full_h,  th_true), 2),
            "reached": bool(info.get("touched_ball1", False)),
            "steps": info["step"],
        })
    return eps


def col(eps, key):
    return np.array([e[key] for e in eps if e[key] is not None], float)


def summ(name, eps, key):
    v = col(eps, key)
    se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    print(f"  {name:16s}: mean {key} {v.mean():+6.1f} +/- {se:.1f}  (n={len(v)})")
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapted", required=True)
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--cone-deg", type=float, default=150.0)
    p.add_argument("--radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-move", type=float, default=0.03)
    p.add_argument("--run-tag", default="reach_ae")
    args = p.parse_args()

    cells = {}
    for pol, path in [("base", args.base), ("adapt", args.adapted)]:
        model = PPO.load(path, device="cpu")
        for vis, abl in [("sighted", False), ("blind", True)]:
            env = make_env(args.offset, args.cone_deg, args.radius, args.max_steps, args.seed)
            cells[(pol, vis)] = run_cell(model, env, args.episodes, args.window,
                                         args.min_move, abl)

    print(f"\n=== single-target no-feedback reach aftereffect (offset={args.offset}, "
          f"window={args.window} steps) ===")
    print("EARLY (ballistic, no-feedback) signed heading error, deg:")
    means = {}
    for pol in ("base", "adapt"):
        for vis in ("sighted", "blind"):
            means[(pol, vis)] = summ(f"{pol} {vis}", cells[(pol, vis)], "early_err")

    def m(k): return means[k].mean()
    vp_base = m(("base", "sighted")) - m(("base", "blind"))
    vp_adap = m(("adapt", "sighted")) - m(("adapt", "blind"))
    did = vp_adap - vp_base
    se_did = math.sqrt(sum((means[k].std(ddof=1)/math.sqrt(len(means[k])))**2 for k in means))
    z = did / se_did
    print(f"\n  visual pull (sighted-blind): base {vp_base:+.1f}  adapted {vp_adap:+.1f}")
    print(f"  motor-drift (adapt_blind - base_blind) = {m(('adapt','blind'))-m(('base','blind')):+.1f} deg")
    print(f"  AFTEREFFECT (diff-in-diff, early heading) = {did:+.1f} deg (SE {se_did:.1f}, z={z:+.2f})")
    print(f"  [negative => adapted eye still aims off by the trained +offset, lens removed = aftereffect]")

    out = RESULTS / f"reach_ae_{args.run_tag}.json"
    out.write_text(json.dumps({"args": vars(args),
                               "cells": {f"{p}_{v}": cells[(p, v)] for (p, v) in cells}},
                              indent=1))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
