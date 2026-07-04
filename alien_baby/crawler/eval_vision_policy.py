"""
eval_vision_policy.py — evaluate a Stage B (SlotFill) or Stage C (Residual) vision policy:
SUBSTRATE vs CAPABILITY + vision load-bearing (pixel ablation) + video. Also supports a
generalization battery (shape / radius sweeps) for the strengthening experiments.

Stage B/C envs use VecNormalize(norm_obs=False); the policy normalizes proprio internally,
so we feed RAW obs to model.predict. Vision ablation = zero the pixel block at eval.

Usage:
  PYTHONPATH=<repo> python -u -m alien_baby.crawler.eval_vision_policy \
    --model alien_baby/results/stage_c_v1_best/best_model.zip --run-tag stage_c_v1 --render-eps 3
"""
import argparse
import pathlib

import numpy as np
import mujoco
import imageio

from stable_baselines3 import PPO
# import the custom policies so PPO.load can reconstruct them
from alien_baby.crawler.residual_vision_policy import ResidualVisionPolicy  # noqa: F401
from alien_baby.crawler.slotfill_vision_policy import SlotFillVisionPolicy  # noqa: F401
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES, PROPRIO_DIM

RESULTS = pathlib.Path(__file__).parent.parent / "results"
XML = "alien_baby/crawler/mimo_crawler_pos_wide.xml"


def make_env(cone_deg, radius, max_steps, seed):
    env = MimoCrawlerEnv(
        vision=True, stereo=True, target_obs=False, crawl_pose=CRAWL_POSES["arms_fwd"],
        action_mode="position_offset", xml_path=XML, spawn_cone_deg=cone_deg,
        spawn_radius=tuple(radius), random_start_orientation=False, max_steps=max_steps,
        step_cost=0.0, approach_reward_scale=10.0, velocity_bonus_scale=0.0,
        terminate_tilt_deg=50.0, tip_penalty=-5.0,
    )
    env.reset(seed=seed)
    return env


def zero_pixels(obs, proprio_dim=PROPRIO_DIM):
    o = obs.copy(); o[proprio_dim:] = 0.0; return o


def evaluate(model, env, n_eps, max_steps, ablate=False):
    contacts = tips = 0
    disps, tow, speeds = [], [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        root0 = obs[:2].copy(); d0 = env._ball_dist()
        touched = False; steps = 0
        for _ in range(max_steps):
            o = zero_pixels(obs) if ablate else obs
            act, _ = model.predict(o, deterministic=True)
            obs, _, term, trunc, info = env.step(act); steps += 1
            if info.get("touched_ball1"):
                touched = True
            if term or trunc:
                break
        disps.append(float(np.linalg.norm(obs[:2] - root0))); speeds.append(disps[-1] / max(1, steps))
        tow.append(d0 - env._ball_dist()); contacts += int(touched); tips += int(term and not touched)
    return dict(contact=contacts / n_eps, tip=tips / n_eps, disp=float(np.mean(disps)),
                speed=float(np.mean(speeds)), toward=float(np.mean(tow)), n=n_eps)


def render(model, env, tag, n_eps, max_steps):
    vdir = RESULTS / "videos"; vdir.mkdir(parents=True, exist_ok=True)
    out = vdir / f"eval_{tag}.mp4"
    r = mujoco.Renderer(env.model, 480, 480)
    w = imageio.get_writer(str(out), fps=25, quality=8)
    for _ in range(n_eps):
        obs, _ = env.reset()
        for _ in range(max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            r.update_scene(env.data, camera="overhead"); over = r.render().copy()
            r.update_scene(env.data, camera="ringside"); ring = r.render().copy()
            w.append_data(np.concatenate([over, ring], axis=1))
            if term or trunc:
                break
    w.close(); r.close(); return str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cone-deg", type=float, default=44.0)
    p.add_argument("--radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--eval-eps", type=int, default=40)
    p.add_argument("--render-eps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-tag", default="vision_eval")
    p.add_argument("--battery", action="store_true", help="also sweep shapes/radii")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    print(f"\n=== eval {args.run_tag} ({type(model.policy).__name__}) ===\n  model={args.model}")

    env = make_env(args.cone_deg, args.radius, args.max_steps, args.seed)
    m = evaluate(model, env, args.eval_eps, args.max_steps, ablate=False)
    a = evaluate(model, env, args.eval_eps, args.max_steps, ablate=True)
    print("\n  --- CAPABILITY (blind floor 20%, teacher ceiling 77.5% on +/-22) ---")
    print(f"    contact         : {m['contact']*100:.1f}%   (pixels ablated: {a['contact']*100:.1f}%)")
    print(f"    mean_toward     : {m['toward']:+.3f} m   (ablated: {a['toward']:+.3f})")
    print(f"    VISION LOAD-BEARING gap (real - ablated contacts): {(m['contact']-a['contact'])*100:+.1f} pts")
    print("  --- SUBSTRATE (blind base: disp 0.219, tip 2.5%) ---")
    print(f"    mean_disp       : {m['disp']:.3f} m")
    print(f"    tip_rate        : {m['tip']*100:.1f}%")
    print(f"    mean_speed      : {m['speed']*1000:.2f} mm/step")

    if args.battery:
        print("\n  --- GENERALIZATION BATTERY (radius / cone sweeps) ---")
        for rad in [[0.55, 0.65], [0.85, 0.95], [1.00, 1.10]]:
            be = make_env(args.cone_deg, rad, args.max_steps, args.seed + 4)
            r = evaluate(model, be, max(20, args.eval_eps // 2), args.max_steps)
            print(f"    radius={rad}: contact {r['contact']*100:.1f}%  toward {r['toward']:+.3f}  tip {r['tip']*100:.1f}%")
            be.close()
        for cone in [30.0, 60.0]:
            be = make_env(cone, args.radius, args.max_steps, args.seed + 6)
            r = evaluate(model, be, max(20, args.eval_eps // 2), args.max_steps)
            print(f"    cone=+/-{cone/2:.0f}: contact {r['contact']*100:.1f}%  toward {r['toward']:+.3f}  tip {r['tip']*100:.1f}%")
            be.close()

    if args.render_eps > 0:
        renv = make_env(args.cone_deg, args.radius, args.max_steps, args.seed + 7)
        print(f"\n  video: {render(model, renv, args.run_tag, args.render_eps, args.max_steps)}")
        renv.close()
    env.close()


if __name__ == "__main__":
    main()
