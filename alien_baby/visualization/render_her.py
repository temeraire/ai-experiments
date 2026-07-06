"""
render_her.py — render an episode from a HER-trained MIMo crawler checkpoint.

Mirrors render_crawler.py but builds the env exactly the way train_crawler.py
does in --her mode: HERCrawlerWrapper → DummyVecEnv → VecNormalize with
norm_obs_keys=["observation"]. SAC.load picks the right MultiInputPolicy
automatically from the saved checkpoint.

Usage:
    python -m alien_baby.visualization.render_her \\
        --checkpoint alien_baby/results/mimo_phase_d_her/final_model.zip \\
        --seeds 0 1 2 --steps 800
"""
import argparse
import pathlib
import os, faulthandler
faulthandler.enable()

import numpy as np
import mujoco
import imageio

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.her_wrapper import HERCrawlerWrapper
from alien_baby.visualization.render_episodes import _add_label

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR   = RESULTS_DIR / "videos"
RENDER_W = 480
RENDER_H = 480


def _make_env(fixed_balls, random_orient, memory_obs, stereo, strength):
    return HERCrawlerWrapper(
        vision=False,
        strength_scale=strength,
        fixed_ball_positions=fixed_balls,
        random_start_orientation=random_orient,
        memory_obs=memory_obs,
        stereo=stereo,
        velocity_bonus_scale=0.0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--strength-scale", type=float, default=0.7)
    p.add_argument("--fixed-ball-positions", default="0.7,0.0;0.0,0.7")
    p.add_argument("--random-start-orientation", action="store_true", default=True)
    p.add_argument("--no-random-orient", dest="random_start_orientation",
                   action="store_false")
    p.add_argument("--memory-obs", action="store_true", default=True)
    p.add_argument("--mono", action="store_true", default=True)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    balls = []
    for chunk in args.fixed_ball_positions.split(";"):
        x, y = chunk.split(",")
        balls.append((float(x), float(y)))

    ckpt = pathlib.Path(args.checkpoint)
    label = args.label or ckpt.parent.name

    print(f"Loading: {ckpt}")
    vec_env = DummyVecEnv([lambda: _make_env(
        balls, args.random_start_orientation, args.memory_obs,
        stereo=not args.mono, strength=args.strength_scale)])

    vn_path = ckpt.parent / "vec_normalize.pkl"
    if vn_path.exists():
        vec_env = VecNormalize.load(str(vn_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"  VecNormalize stats: {vn_path}")

    model = SAC.load(str(ckpt), env=vec_env, device="cpu")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        out_path = VIDEO_DIR / f"crawler_{label}_seed{seed}.mp4"
        inner_env = _make_env(balls, args.random_start_orientation,
                              args.memory_obs, stereo=not args.mono,
                              strength=args.strength_scale)
        renderer = mujoco.Renderer(inner_env.inner.model, RENDER_H, RENDER_W)

        obs, _ = inner_env.reset(seed=seed)
        # Normalize the dict obs the same way the policy saw it during training
        if hasattr(vec_env, "obs_rms"):
            # VecNormalize wants a batched obs dict
            obs_batch = {k: np.asarray(v)[None] for k, v in obs.items()}
            obs_norm = vec_env.normalize_obs(obs_batch)
        else:
            obs_norm = {k: np.asarray(v)[None] for k, v in obs.items()}

        writer = imageio.get_writer(str(out_path), fps=25, quality=8)
        last_frame = None
        touched_b1 = False
        touched_b2 = False

        for step in range(args.steps):
            panels = []
            for cam, lbl in zip(["overhead", "ringside"], ["OVERHEAD", "RINGSIDE"]):
                renderer.update_scene(inner_env.inner.data, camera=cam)
                frame = renderer.render().copy()
                panels.append(_add_label(frame, lbl))
            composite = np.concatenate(panels, axis=1)
            writer.append_data(composite)
            last_frame = composite

            action, _ = model.predict(obs_norm, deterministic=True)
            action = action[0]
            obs, reward, terminated, truncated, info = inner_env.step(action)
            if hasattr(vec_env, "obs_rms"):
                obs_batch = {k: np.asarray(v)[None] for k, v in obs.items()}
                obs_norm = vec_env.normalize_obs(obs_batch)
            else:
                obs_norm = {k: np.asarray(v)[None] for k, v in obs.items()}

            touched_b1 = touched_b1 or inner_env.inner._ball1_touched
            touched_b2 = touched_b2 or inner_env.inner._ball2_touched

            if terminated or truncated:
                break

        if last_frame is not None:
            for _ in range(25):
                writer.append_data(last_frame)
        writer.close()
        renderer.close()
        inner_env.close()

        outcome = ("BALL1+BALL2" if touched_b1 and touched_b2
                   else "BALL1" if touched_b1
                   else "BALL2" if touched_b2
                   else "NONE")
        steps_taken = step + 1
        print(f"  {out_path.name}  ({steps_taken} steps, {outcome})")
        results.append((seed, outcome, steps_taken))

    print("\nSummary:")
    for seed, outcome, steps in results:
        print(f"  seed={seed:2d}  {outcome:11s}  {steps} steps")
    touch_count = sum(1 for _, o, _ in results if o != "NONE")
    print(f"\nTouched at least one ball: {touch_count}/{len(results)}")
    os._exit(0)


if __name__ == "__main__":
    main()
