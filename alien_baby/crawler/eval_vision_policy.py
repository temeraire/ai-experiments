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
from alien_baby.crawler.mimo_crawler_env import (
    MimoCrawlerEnv, CRAWL_POSES, PROPRIO_DIM, CAM_H, CAM_W, VISION_DIM_STEREO,
)

RESULTS = pathlib.Path(__file__).parent.parent / "results"
XML = "alien_baby/crawler/mimo_crawler_pos_wide.xml"


def make_env(cone_deg, radius, max_steps, seed, xml=XML, decoy=False):
    env = MimoCrawlerEnv(
        vision=True, stereo=True, target_obs=False, crawl_pose=CRAWL_POSES["arms_fwd"],
        action_mode="position_offset", xml_path=xml, spawn_cone_deg=cone_deg,
        spawn_radius=tuple(radius), random_start_orientation=False, max_steps=max_steps,
        step_cost=0.0, approach_reward_scale=10.0, velocity_bonus_scale=0.0,
        terminate_tilt_deg=50.0, tip_penalty=-5.0, decoy_ball=decoy,
    )
    env.reset(seed=seed)
    return env


def zero_pixels(obs, proprio_dim=PROPRIO_DIM):
    o = obs.copy(); o[proprio_dim:] = 0.0; return o


def evaluate(model, env, n_eps, max_steps, ablate=False):
    contacts = tips = decoys = 0
    disps, tow, speeds = [], [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        root0 = obs[:2].copy(); d0 = env._ball_dist()
        touched = wrong = False; steps = 0
        for _ in range(max_steps):
            o = zero_pixels(obs) if ablate else obs
            act, _ = model.predict(o, deterministic=True)
            obs, _, term, trunc, info = env.step(act); steps += 1
            if info.get("touched_ball1"):
                touched = True
            if info.get("touched_ball2"):
                wrong = True
            if term or trunc:
                break
        disps.append(float(np.linalg.norm(obs[:2] - root0))); speeds.append(disps[-1] / max(1, steps))
        tow.append(d0 - env._ball_dist()); contacts += int(touched)
        decoys += int(wrong and not touched)
        tips += int(term and not touched and not wrong)
    return dict(contact=contacts / n_eps, tip=tips / n_eps, decoy=decoys / n_eps,
                disp=float(np.mean(disps)),
                speed=float(np.mean(speeds)), toward=float(np.mean(tow)), n=n_eps)


def eye_panel(obs, size=480):
    """AB's-eye panel: slice the exact pixel block the policy received out of the obs
    vector and upscale nearest-neighbor (honestly blocky — no fake smoothing).
    Left|right eye side by side. All-black = pixels ablated / nothing visible."""
    px = np.clip(obs[-VISION_DIM_STEREO:], 0.0, 1.0)
    half = CAM_H * CAM_W * 3
    s = size // CAM_H

    def up(flat):
        img = (flat.reshape(CAM_H, CAM_W, 3) * 255).astype(np.uint8)
        return np.repeat(np.repeat(img, s, axis=0), s, axis=1)

    return np.concatenate([up(px[:half]), up(px[half:])], axis=1)


def _draw_line(img, p0, p1, color, thick=2):
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]), 1))
    for t in np.linspace(0.0, 1.0, n + 1):
        u = int(round(p0[0] + (p1[0] - p0[0]) * t))
        v = int(round(p0[1] + (p1[1] - p0[1]) * t))
        img[max(0, v - thick):v + thick, max(0, u - thick):u + thick] = color


def draw_heading_arrow(img, env, size=480, cam_z=5.5, fovy_deg=55.0):
    """Overlay a magenta arrow on the OVERHEAD panel showing which way the head
    points (display-only; AB's own pixels are untouched). Arrow base = head
    position; direction = head-frame +Z (prone 'forward'/gaze) projected to XY."""
    try:
        hid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        pos = env.data.xpos[hid]
        fwd = env.data.xmat[hid].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        fx, fy = fwd[0], fwd[1]
        norm = float(np.hypot(fx, fy))
        if norm < 0.2:      # head pointing near-vertically: heading undefined
            return
        fx, fy = fx / norm, fy / norm
        # overhead cam: straight down from (0,0,cam_z), +X right, +Y up in image
        ppm = (size / 2) / (np.tan(np.deg2rad(fovy_deg) / 2) * (cam_z - pos[2]))
        u0, v0 = size / 2 + pos[0] * ppm, size / 2 - pos[1] * ppm
        L = 42
        u1, v1 = u0 + fx * L, v0 - fy * L
        col = np.array([255, 0, 255], dtype=np.uint8)      # magenta
        _draw_line(img, (u0, v0), (u1, v1), col)
        # arrowhead: two barbs at ±150 deg from the shaft direction
        for a in (2.62, -2.62):
            bx = fx * np.cos(a) - fy * np.sin(a)
            by = fx * np.sin(a) + fy * np.cos(a)
            _draw_line(img, (u1, v1), (u1 + bx * 14, v1 - by * 14), col)
    except Exception:
        pass            # overlay must never kill a render


def render(model, env, tag, n_eps, max_steps, ablate=False):
    vdir = RESULTS / "videos"; vdir.mkdir(parents=True, exist_ok=True)
    out = vdir / f"eval_{tag}.mp4"
    r = mujoco.Renderer(env.model, 480, 480)
    w = imageio.get_writer(str(out), fps=25, quality=8)
    for _ in range(n_eps):
        obs, _ = env.reset()
        for _ in range(max_steps):
            o = zero_pixels(obs) if ablate else obs
            act, _ = model.predict(o, deterministic=True)
            r.update_scene(env.data, camera="overhead"); over = r.render().copy()
            draw_heading_arrow(over, env)
            r.update_scene(env.data, camera="ringside"); ring = r.render().copy()
            w.append_data(np.concatenate([over, ring, eye_panel(o)], axis=1))
            obs, _, term, trunc, _ = env.step(act)
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
    p.add_argument("--render-ablate", action="store_true",
                   help="render the video with pixels zeroed (blind policy; eye panel black)")
    p.add_argument("--decoy", action="store_true",
                   help="two-ball discrimination env: blue decoy active, wrong touch ends episode")
    p.add_argument("--xml", default=XML, help="env XML (swap for shape-generalization probes)")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    print(f"\n=== eval {args.run_tag} ({type(model.policy).__name__}) ===\n  model={args.model}\n  xml={args.xml}")

    env = make_env(args.cone_deg, args.radius, args.max_steps, args.seed, xml=args.xml,
                   decoy=args.decoy)
    m = evaluate(model, env, args.eval_eps, args.max_steps, ablate=False)
    a = evaluate(model, env, args.eval_eps, args.max_steps, ablate=True)
    print("\n  --- CAPABILITY (blind floor 20%, teacher ceiling 77.5% on +/-22) ---")
    print(f"    contact         : {m['contact']*100:.1f}%   (pixels ablated: {a['contact']*100:.1f}%)")
    if args.decoy:
        print(f"    WRONG-BALL rate : {m['decoy']*100:.1f}%   (pixels ablated: {a['decoy']*100:.1f}%)")
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
        renv = make_env(args.cone_deg, args.radius, args.max_steps, args.seed + 7, xml=args.xml,
                        decoy=args.decoy)
        vtag = args.run_tag + ("_ablated" if args.render_ablate else "")
        print(f"\n  video: {render(model, renv, vtag, args.render_eps, args.max_steps, ablate=args.render_ablate)}")
        renv.close()
    env.close()


if __name__ == "__main__":
    main()
