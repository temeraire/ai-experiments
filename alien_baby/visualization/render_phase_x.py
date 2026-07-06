"""
render_phase_x.py — Phase X / X2 render gate.

Renders a handful of episodes with random_ball_radius or random_ball_shape active
so the human can confirm each variant spawns on the platform, sits at the right height,
doesn't intersect the cart/rails, and contacts register.

Usage:
  # Phase X (size variety) — use R45 checkpoint:
  python -m alien_baby.visualization.render_phase_x \
      --checkpoint alien_baby/results/phase_v_R44_proprio_static_randbox_best/best_model.zip \
      --mode size --seeds 0 1 2 3 4 --steps 300

  # Phase X2 (shape variety) — use any checkpoint (untrained is fine for visual gate):
  python -m alien_baby.visualization.render_phase_x \
      --checkpoint alien_baby/results/phase_x2_shape_smoke_best/best_model.zip \
      --mode shape --seeds 0 1 2 3 4 --steps 300
"""

import pathlib
import argparse
import numpy as np
import mujoco
import imageio

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR   = RESULTS_DIR / "videos"
RENDER_W = 480
RENDER_H = 480

# Phase X: training radii
SIZE_RADII = [0.040, 0.053, 0.075]
# Phase X2: all 5 primitive shapes (train set + held-out)
ALL_SHAPES  = ["sphere", "box", "cylinder", "ellipsoid", "capsule"]
FIXED_BALL_POSITIONS = [(0.0, 0.35), (0.0, -0.35)]


def _make_env_fn(mode):
    def _init():
        kwargs = dict(
            vision=False,
            max_steps=400,
            strength_scale=1.0,
            fixed_ball_positions=FIXED_BALL_POSITIONS,
            cart_speed=0.15,
            hip_actuation=False,
            memory_obs=True,
            random_ball_box=(0.08, 0.08),
        )
        if mode == "size":
            kwargs["random_ball_radius"] = SIZE_RADII
        else:
            kwargs["random_ball_shape"] = ALL_SHAPES
        return MimoCrawlerCartEnv(**kwargs)
    return _init


def _load_model(checkpoint_path, mode):
    ckpt = pathlib.Path(checkpoint_path)
    vec_env = DummyVecEnv([_make_env_fn(mode)])
    vn_path = ckpt.parent / "vec_normalize.pkl"
    if not vn_path.exists():
        run_dir = ckpt.parent.parent / ckpt.parent.name.replace("_best", "")
        vn_path = run_dir / "vec_normalize.pkl"
    if vn_path.exists():
        vec_env = VecNormalize.load(str(vn_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"  VecNormalize: {vn_path}")
    else:
        print("  WARNING: no vec_normalize.pkl — obs will be raw (expected for smoke ckpts)")
    model = SAC.load(str(ckpt), env=vec_env, device="cpu")
    return model, vec_env


def render_varied(model, vec_env, seeds, max_steps, mode, label):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_video = VIDEO_DIR / f"{label}_{mode}_gate.mp4"
    writer = imageio.get_writer(str(out_video), fps=20, quality=8)

    for seed in seeds:
        inner = MimoCrawlerCartEnv(
            vision=False,
            max_steps=max_steps,
            strength_scale=1.0,
            fixed_ball_positions=FIXED_BALL_POSITIONS,
            cart_speed=0.15,
            hip_actuation=False,
            memory_obs=True,
            random_ball_box=(0.08, 0.08),
            **({"random_ball_radius": SIZE_RADII} if mode == "size"
               else {"random_ball_shape": ALL_SHAPES}),
        )
        renderer = mujoco.Renderer(inner.model, RENDER_H, RENDER_W)
        obs, _ = inner.reset(seed=seed)
        variant = (f"r={inner._current_ball_radius:.3f}" if mode == "size"
                   else f"shape={inner._current_ball_shape}")
        if hasattr(vec_env, "obs_rms"):
            obs_norm = vec_env.normalize_obs(obs)
        else:
            obs_norm = obs

        touched_b1 = touched_b2 = False
        total_rew = 0.0
        for step in range(max_steps):
            renderer.update_scene(inner.data, camera="overhead")
            overhead = renderer.render().copy()
            renderer.update_scene(inner.data, camera="ringside")
            ringside = renderer.render().copy()
            frame = np.concatenate([overhead, ringside], axis=1)
            writer.append_data(frame)

            action, _ = model.predict(obs_norm.reshape(1, -1), deterministic=True)
            obs, rew, terminated, truncated, info = inner.step(action[0])
            total_rew += float(rew)
            if hasattr(vec_env, "obs_rms"):
                obs_norm = vec_env.normalize_obs(obs)
            else:
                obs_norm = obs
            if info["touched_ball1"]:
                touched_b1 = True
            if info["touched_ball2"]:
                touched_b2 = True
            if terminated or truncated:
                break

        renderer.close()
        inner.close()
        print(f"  seed={seed}  {variant}  steps={step+1}  reward={total_rew:+.1f}  "
              f"b1={'T' if touched_b1 else '-'} b2={'T' if touched_b2 else '-'}")

    writer.close()
    print(f"\nVideo: {out_video}")
    return str(out_video)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["size", "shape"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    label = args.label or pathlib.Path(args.checkpoint).parent.name
    print(f"Loading: {args.checkpoint}")
    print(f"  mode={args.mode}  seeds={args.seeds}  steps={args.steps}")

    model, vec_env = _load_model(args.checkpoint, args.mode)
    render_varied(model, vec_env, args.seeds, args.steps, args.mode, label)


if __name__ == "__main__":
    import os
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        os._exit(0)
