"""
eval_crawler.py — deterministic eval + video for a PPO crawler, reporting the
SUBSTRATE vs CAPABILITY split (todo.md measurement contract).

  SUBSTRATE (proprio, must be preserved): mean CoM displacement, mean speed, tip-rate.
  CAPABILITY (vision/bearing, must climb):  contact rate, mean dist-toward-ball.

Works for the blind base (proprio only) or a target_obs policy. Renders an
overhead+ringside composite mp4 for Gemini video review.

Usage:
  PYTHONPATH=<repo> python -m alien_baby.crawler.eval_crawler \
    --model alien_baby/results/crawl_ppo_blind_base_best/best_model.zip \
    --run-tag blind_base --eval-eps 40 --render-eps 2
"""
import argparse
import pathlib
import pickle

import numpy as np
import mujoco
import imageio

from stable_baselines3 import PPO
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES

RESULTS = pathlib.Path(__file__).parent.parent / "results"
XML = "alien_baby/crawler/mimo_crawler_pos_wide.xml"


def load_vecnorm(model_path):
    """Find the vec_normalize.pkl next to the model (best dir) or in the ckpt dir."""
    mp = pathlib.Path(model_path)
    cands = [mp.parent / "vec_normalize.pkl",
             mp.parent.parent / mp.parent.name.replace("_best", "") / "vec_normalize.pkl"]
    for c in cands:
        if c.exists():
            with open(c, "rb") as f:
                vn = pickle.load(f)
            mean, var = vn.obs_rms.mean.astype(np.float32), vn.obs_rms.var.astype(np.float32)
            clip, eps = float(vn.clip_obs), float(vn.epsilon)
            return lambda o: np.clip((o - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32), str(c)
    raise FileNotFoundError(f"no vec_normalize.pkl near {model_path}")


def make_env(target_obs, cone_deg, radius, max_steps, seed):
    env = MimoCrawlerEnv(
        vision=False, target_obs=target_obs, crawl_pose=CRAWL_POSES["arms_fwd"],
        action_mode="position_offset", xml_path=XML, spawn_cone_deg=cone_deg,
        spawn_radius=tuple(radius), random_start_orientation=False, max_steps=max_steps,
        step_cost=0.0, approach_reward_scale=10.0, velocity_bonus_scale=0.0,
        terminate_tilt_deg=50.0, tip_penalty=-5.0,
    )
    env.reset(seed=seed)
    return env


def evaluate(model, normalize, env, n_eps, max_steps):
    contacts = tips = 0
    disps, tow, speeds = [], [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        root0 = obs[:2].copy(); d0 = env._ball_dist()
        touched = False; steps = 0
        for _ in range(max_steps):
            act, _ = model.predict(normalize(obs), deterministic=True)
            obs, _, term, trunc, info = env.step(act); steps += 1
            if info.get("touched_ball1"):
                touched = True
            if term or trunc:
                break
        disp = float(np.linalg.norm(obs[:2] - root0))
        disps.append(disp); speeds.append(disp / max(1, steps))
        tow.append(d0 - env._ball_dist())
        contacts += int(touched)
        tips += int(term and not touched)   # ended early w/o contact = tip-terminated
    return dict(
        contact_rate=contacts / n_eps, tip_rate=tips / n_eps, n=n_eps,
        mean_disp=float(np.mean(disps)), mean_speed=float(np.mean(speeds)),
        mean_toward=float(np.mean(tow)),
    )


def render(model, normalize, env, tag, n_eps, max_steps):
    vdir = RESULTS / "videos"; vdir.mkdir(parents=True, exist_ok=True)
    out = vdir / f"eval_{tag}.mp4"
    rmain = mujoco.Renderer(env.model, 480, 480)
    writer = imageio.get_writer(str(out), fps=25, quality=8)
    for _ in range(n_eps):
        obs, _ = env.reset()
        for _ in range(max_steps):
            act, _ = model.predict(normalize(obs), deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            rmain.update_scene(env.data, camera="overhead")
            over = rmain.render().copy()
            rmain.update_scene(env.data, camera="ringside")
            ring = rmain.render().copy()
            writer.append_data(np.concatenate([over, ring], axis=1))
            if term or trunc:
                break
    writer.close(); rmain.close()
    return str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--target-obs", action="store_true")
    p.add_argument("--cone-deg", type=float, default=44.0)
    p.add_argument("--radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--eval-eps", type=int, default=40)
    p.add_argument("--render-eps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-tag", default="eval")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    normalize, vn_path = load_vecnorm(args.model)
    print(f"\n=== eval {args.run_tag} ===\n  model={args.model}\n  vecnorm={vn_path}")
    print(f"  cone=+/-{args.cone_deg/2:.0f} radius={args.radius} target_obs={args.target_obs}")

    env = make_env(args.target_obs, args.cone_deg, args.radius, args.max_steps, args.seed)
    m = evaluate(model, normalize, env, args.eval_eps, args.max_steps)
    print("\n  --- CAPABILITY (should climb with vision) ---")
    print(f"    contact_rate : {m['contact_rate']*100:.1f}%   ({m['n']} eps)")
    print(f"    mean_toward  : {m['mean_toward']:+.3f} m")
    print("  --- SUBSTRATE (must be preserved) ---")
    print(f"    mean_disp    : {m['mean_disp']:.3f} m")
    print(f"    mean_speed   : {m['mean_speed']*1000:.2f} mm/step")
    print(f"    tip_rate     : {m['tip_rate']*100:.1f}%")

    if args.render_eps > 0:
        renv = make_env(args.target_obs, args.cone_deg, args.radius, args.max_steps, args.seed + 5)
        path = render(model, normalize, renv, args.run_tag, args.render_eps, args.max_steps)
        print(f"\n  video: {path}")
        renv.close()
    env.close()


if __name__ == "__main__":
    main()
