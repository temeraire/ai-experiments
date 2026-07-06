"""
render_crawler.py — render MIMo crawler episodes for evaluation.

Panels: OVERHEAD (top-down) + RINGSIDE (posture read).
Optionally add LEFT_EYE + RIGHT_EYE when --vision is passed.

Usage:
    python -m alien_baby.visualization.render_crawler \
        --checkpoint alien_baby/results/crawler_v2_strength07_prone_best/best_model.zip \
        --seeds 0 1 2 3 4 --steps 400
"""

import pathlib
import argparse
import numpy as np
import mujoco
import imageio

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv
from alien_baby.visualization.render_episodes import _add_label

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR   = RESULTS_DIR / "videos"
RENDER_W = 480
RENDER_H = 480


def _load_model_and_stats(checkpoint_path: str, strength_scale: float, n_envs: int = 1,
                          vision: bool = False, action_mode: str = "torque",
                          xml_path: str = None, spawn_radius=None):
    """Load SAC model; apply VecNormalize stats from sibling vec_normalize.pkl if present."""
    ckpt = pathlib.Path(checkpoint_path)
    env_fns = [lambda: MimoCrawlerEnv(vision=vision, strength_scale=strength_scale,
                                      action_mode=action_mode, xml_path=xml_path,
                                      spawn_radius=spawn_radius)]
    vec_env = DummyVecEnv(env_fns)

    vn_path = ckpt.parent / "vec_normalize.pkl"
    if not vn_path.exists():
        # try the non-_best directory
        run_dir = ckpt.parent.parent / ckpt.parent.name.replace("_best", "")
        vn_path = run_dir / "vec_normalize.pkl"

    if vn_path.exists():
        vec_env = VecNormalize.load(str(vn_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"  VecNormalize stats: {vn_path}")
    else:
        print("  No vec_normalize.pkl found — running without obs normalization")

    model = SAC.load(str(ckpt), env=vec_env, device="cpu")
    return model, vec_env


def render_crawler_episode(
    model,
    vec_env,
    seed: int = 0,
    max_steps: int = 400,
    label: str = "crawler",
    vision: bool = False,
    strength_scale: float = 0.7,
    action_mode: str = "torque",
    xml_path: str = None,
    spawn_radius=None,
):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"crawler_{label}_seed{seed}.mp4"

    inner_env = MimoCrawlerEnv(vision=vision, strength_scale=strength_scale,
                               action_mode=action_mode, xml_path=xml_path,
                               spawn_radius=spawn_radius)
    renderer  = mujoco.Renderer(inner_env.model, RENDER_H, RENDER_W)

    obs, _ = inner_env.reset(seed=seed)
    if hasattr(vec_env, "obs_rms"):
        # normalise the initial obs the same way the policy saw it during training
        obs_norm = vec_env.normalize_obs(obs)
    else:
        obs_norm = obs

    camera_names = ["overhead", "ringside"]
    cam_labels   = ["OVERHEAD", "RINGSIDE"]
    if vision:
        camera_names += ["left_eye", "right_eye"]
        cam_labels   += ["LEFT EYE", "RIGHT EYE"]

    writer = imageio.get_writer(str(out_path), fps=25, quality=8)
    last_frame = None
    steps_taken = 0
    touched = False

    for step in range(max_steps):
        panels = []
        for cam, lbl in zip(camera_names, cam_labels):
            renderer.update_scene(inner_env.data, camera=cam)
            frame = renderer.render().copy()
            panels.append(_add_label(frame, lbl))
        composite = np.concatenate(panels, axis=1)
        writer.append_data(composite)
        last_frame = composite

        action, _ = model.predict(obs_norm.reshape(1, -1), deterministic=True)
        action = action[0]
        obs, reward, terminated, truncated, info = inner_env.step(action)
        if hasattr(vec_env, "obs_rms"):
            obs_norm = vec_env.normalize_obs(obs)
        else:
            obs_norm = obs

        steps_taken = step + 1
        if terminated:
            touched = info.get("touched", False)
        if terminated or truncated:
            break

    # Hold last frame for 1 second
    if last_frame is not None:
        for _ in range(25):
            writer.append_data(last_frame)

    writer.close()
    renderer.close()
    inner_env.close()

    outcome = "TOUCHED" if touched else "TIMEOUT"
    print(f"  {out_path.name}  ({steps_taken} steps, {outcome})")
    return str(out_path), outcome, steps_taken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
        default=str(RESULTS_DIR / "crawler_v2_strength07_prone_best" / "best_model.zip"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--strength-scale", type=float, default=0.7)
    parser.add_argument("--label", default=None)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--action-mode", default="torque",
                        choices=["torque", "position_offset"],
                        help="Phase XVI R49: match the action mode used during training.")
    parser.add_argument("--xml", default=None,
                        help="Phase XVI R49: path to a custom MuJoCo XML (e.g. mimo_crawler_pos.xml). "
                             "When --action-mode position_offset, this defaults to mimo_crawler_pos.xml.")
    parser.add_argument("--spawn-radius", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Ball spawn radius range [min max] in metres. "
                             "Default: env default (0.5, 1.2).")
    args = parser.parse_args()

    label = args.label or pathlib.Path(args.checkpoint).parent.name
    spawn_radius = tuple(args.spawn_radius) if args.spawn_radius else None

    print(f"Loading: {args.checkpoint}")
    model, vec_env = _load_model_and_stats(args.checkpoint, args.strength_scale, vision=args.vision,
                                           action_mode=args.action_mode, xml_path=args.xml,
                                           spawn_radius=spawn_radius)

    results = []
    for seed in args.seeds:
        path, outcome, steps = render_crawler_episode(
            model, vec_env,
            seed=seed,
            max_steps=args.steps,
            label=label,
            vision=args.vision,
            strength_scale=args.strength_scale,
            action_mode=args.action_mode,
            xml_path=args.xml,
            spawn_radius=spawn_radius,
        )
        results.append((seed, outcome, steps))

    print("\nSummary:")
    for seed, outcome, steps in results:
        print(f"  seed={seed:2d}  {outcome:8s}  {steps} steps")

    touched_count = sum(1 for _, o, _ in results if o == "TOUCHED")
    print(f"\nTouch rate: {touched_count}/{len(results)}")


if __name__ == "__main__":
    main()
