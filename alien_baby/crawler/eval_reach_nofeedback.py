"""eval_reach_nofeedback.py — MINIMAL, human-faithful single-target aftereffect probe.

The human prism experiment needs no second object: reach for ONE target under the lens, miss,
adapt, remove the lens, and the aftereffect is that you now mis-reach the other way. This probe
matches that:
  * SINGLE target (decoy_ball=False) -> no selection channel.
  * At offset 0 the agent sees the REAL ball directly (the ghost only exists when offset!=0).
  * The target's COLLISION is disabled at runtime -> it is a VISUAL-ONLY marker. The agent steers
    toward what it sees with NO touch-homing to correct its aim and NO contact termination. The
    approach direction is therefore the pure vision-driven aim -- the "no-feedback reach".

Aftereffect = does the ADAPTED eye (trained under +30) aim off-true at lens-off, vs base?
Measured as a 2x2 {base,adapted} x {sighted,blind} difference-in-differences (nets out motor drift).

INSTRUMENT VALIDATION IS BUILT IN (the lesson from the failed early-window probe): we report the
approach heading under THREE definitions and, for each, R^2(true_bearing, heading) per cell. A
definition is only a valid AIM measure if sighted R^2 is high AND blind R^2 is low. Do not read an
aftereffect off a definition that fails that check.

Usage:
  PYTHONPATH=<repo> python alien_baby/crawler/eval_reach_nofeedback.py \
    --base   alien_baby/results/stage1_ground_smoke_s0_best/best_model.zip \
    --adapted alien_baby/results/stage2_adapt_s0_best/best_model.zip \
    --episodes 300 --run-tag nofb
"""
import argparse, json, math, pathlib
import numpy as np
import mujoco
from stable_baselines3 import PPO
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES, VISION_DIM_STEREO

XML = "alien_baby/crawler/mimo_crawler_pos_wide_prism.xml"
RESULTS = pathlib.Path(__file__).parent.parent / "results"


def make_env(offset_deg, cone_deg, radius, max_steps, seed, gaze_spawn=False):
    env = MimoCrawlerEnv(
        vision=True, stereo=True, target_obs=False, crawl_pose=CRAWL_POSES["arms_fwd"],
        action_mode="position_offset", xml_path=XML, spawn_cone_deg=cone_deg,
        spawn_radius=tuple(radius), random_start_orientation=False, max_steps=max_steps,
        step_cost=0.0, approach_reward_scale=10.0, velocity_bonus_scale=0.0,
        terminate_tilt_deg=50.0, tip_penalty=-5.0, decoy_ball=False,   # SINGLE target
        prism_offset_deg=offset_deg, gaze_spawn=gaze_spawn,
    )
    # Make the target a VISUAL-ONLY marker: disable collision so there is no touch-homing
    # and no contact termination -- the approach is a pure no-feedback vision-driven reach.
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    env.model.geom_contype[gid] = 0
    env.model.geom_conaffinity[gid] = 0
    env.reset(seed=seed)
    return env


def zero_pixels(obs):
    o = obs.copy(); o[-VISION_DIM_STEREO:] = 0.0; return o


def ball_xy(env, joint):
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qa = env.model.jnt_qposadr[jid]
    return np.array(env.data.qpos[qa:qa + 2], float)


def signed_deg(a, b):
    return math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi)


def heading_of(vec):
    return float(np.arctan2(vec[0], vec[1])) if np.linalg.norm(vec) > 1e-6 else None


def run_cell(model, env, n_eps, dist_thresh, min_move, ablate):
    eps = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        bxy = ball_xy(env, "target_free")
        root0 = obs[:2].copy()
        th_true = float(np.arctan2(bxy[0], bxy[1]))
        traj = [root0.copy()]
        pos_at_thresh = None
        while True:
            o = zero_pixels(obs) if ablate else obs
            act, _ = model.predict(o, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            p = obs[:2].copy(); traj.append(p)
            if pos_at_thresh is None and np.linalg.norm(p - root0) >= dist_thresh:
                pos_at_thresh = p.copy()
            if term or trunc:
                break
        traj = np.array(traj)
        # closest approach to the (static, non-collidable) ball
        dists = np.linalg.norm(traj - bxy, axis=1)
        closest = traj[int(np.argmin(dists))]
        h_close = heading_of(closest - root0)          # dir to closest-approach point
        h_thr = heading_of((pos_at_thresh - root0)) if pos_at_thresh is not None else None
        h_net = heading_of(traj[-1] - root0)           # net whole-episode displacement
        def err(h): return None if h is None else round(signed_deg(h, th_true), 2)
        eps.append({
            "th_true": round(math.degrees(th_true), 2),
            "err_close": err(h_close), "err_thresh": err(h_thr), "err_net": err(h_net),
            "min_dist": round(float(dists.min()), 3), "steps": info["step"],
        })
    return eps


def r2(eps, key):
    tt = np.array([e["th_true"] for e in eps if e[key] is not None], float)
    hd = np.array([e["th_true"] + e[key] for e in eps if e[key] is not None], float)
    if len(tt) < 5:
        return float("nan"), 0
    return float(np.corrcoef(tt, hd)[0, 1] ** 2), len(tt)


def mean_se(eps, key):
    v = np.array([e[key] for e in eps if e[key] is not None], float)
    return v.mean(), (v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")), len(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapted", required=True)
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--dist-thresh", type=float, default=0.30)
    p.add_argument("--cone-deg", type=float, default=150.0)
    p.add_argument("--radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-move", type=float, default=0.03)
    p.add_argument("--gaze-spawn", action="store_true",
                   help="corrected prism: place the visible target in the gaze cone")
    p.add_argument("--run-tag", default="nofb")
    args = p.parse_args()

    cells = {}
    for pol, path in [("base", args.base), ("adapt", args.adapted)]:
        model = PPO.load(path, device="cpu")
        for vis, abl in [("sighted", False), ("blind", True)]:
            env = make_env(args.offset, args.cone_deg, args.radius, args.max_steps, args.seed,
                           gaze_spawn=args.gaze_spawn)
            cells[(pol, vis)] = run_cell(model, env, args.episodes, args.dist_thresh,
                                         args.min_move, abl)

    print(f"\n=== single-target NO-FEEDBACK reach aftereffect (offset={args.offset}, "
          f"visual-only target, dist_thresh={args.dist_thresh}m) ===")
    print("STEP 1 -- INSTRUMENT VALIDATION: R^2(true bearing, heading) per definition per cell.")
    print("  A definition is a valid AIM measure only if SIGHTED R^2 high AND BLIND R^2 low.\n")
    keys = [("err_close", "closest-approach"), ("err_thresh", f"@{args.dist_thresh}m"),
            ("err_net", "net-episode")]
    valid = []
    for key, label in keys:
        print(f"  {label:18s}:")
        r2s = {}
        for pol in ("base", "adapt"):
            for vis in ("sighted", "blind"):
                rr, n = r2(cells[(pol, vis)], key)
                r2s[(pol, vis)] = rr
                print(f"      {pol} {vis:8s} R^2={rr:.3f} (n={n})")
        ok = (r2s[("base", "sighted")] > 0.25 and r2s[("base", "blind")] < 0.10)
        print(f"      -> {'VALID aim measure' if ok else 'NOT a clean aim measure'} "
              f"(sighted {r2s[('base','sighted')]:.2f} vs blind {r2s[('base','blind')]:.2f})")
        if ok:
            valid.append((key, label))

    print("\nSTEP 2 -- aftereffect (diff-in-diff, motor-drift controlled), per VALID definition:")
    if not valid:
        print("  NONE of the heading definitions is a valid aim measure -> the vision-driven aim")
        print("  cannot be isolated with this probe either. Report as instrument failure, not 'no aftereffect'.")
    for key, label in valid:
        ms = {k: mean_se(cells[k], key) for k in cells}
        m = lambda k: ms[k][0]
        vp_b = m(("base", "sighted")) - m(("base", "blind"))
        vp_a = m(("adapt", "sighted")) - m(("adapt", "blind"))
        did = vp_a - vp_b
        se = math.sqrt(sum(ms[k][1] ** 2 for k in cells))
        print(f"  [{label}] base sighted {m(('base','sighted')):+.1f} blind {m(('base','blind')):+.1f} | "
              f"adapt sighted {m(('adapt','sighted')):+.1f} blind {m(('adapt','blind')):+.1f}")
        print(f"    visual pull base {vp_b:+.1f} adapt {vp_a:+.1f}; motor-drift "
              f"{m(('adapt','blind'))-m(('base','blind')):+.1f}; "
              f"AFTEREFFECT (DiD) = {did:+.1f} deg (SE {se:.1f}, z={did/se:+.2f})")

    out = RESULTS / f"reach_nofb_{args.run_tag}.json"
    out.write_text(json.dumps({"args": vars(args),
                               "cells": {f"{p}_{v}": cells[(p, v)] for (p, v) in cells}}, indent=1))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
