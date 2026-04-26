"""
Render v8 composite: overhead + ringside + head-cam view.

Panels:
- OVERHEAD — experimenter's perspective (platform from above)
- RINGSIDE — low-angle posture read (upright vs. tipping)
- HEAD CAM — what the creature sees (temporally smoothed, optionally blurred)

Head-cam post-processing (display-side only; environment and observation
are unchanged):
- Temporal smoothing: EMA over successive frames to reduce body-shake jitter.
- Blur-to-sharpen overlay: Gaussian blur whose sigma decreases with a
  "sharpness" parameter in [0,1]. 1.0 = fully sharp, 0.0 = fully blurry.
  Can be supplied directly via --sharpness, or computed from the loaded
  model's vision-ablation sensitivity via --auto-sharpness.
"""

import pathlib
import numpy as np
import mujoco
import imageio
from scipy.ndimage import gaussian_filter

from alien_baby.envs.platform_creature_env import PlatformCreatureEnv
from alien_baby.agents.train_v5 import ConsistencySAC
from alien_baby.visualization.render_episodes import _add_label

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VIDEO_DIR = RESULTS_DIR / "videos"
RENDER_WIDTH = 480
RENDER_HEIGHT = 480

BLUR_SIGMA_MAX = 6.0  # sigma applied to head-cam at sharpness=0
DEFAULT_SMOOTH_ALPHA = 0.3  # EMA weight on the newest frame


def _render_cameras(renderer, data, cameras):
    frames = {}
    for name in cameras:
        renderer.update_scene(data, camera=name)
        frames[name] = renderer.render().copy()
    return frames


def _ema_blend(prev, curr, alpha):
    if prev is None:
        return curr.astype(np.float32)
    return alpha * curr.astype(np.float32) + (1.0 - alpha) * prev


def _apply_blur(frame_float, sharpness):
    sharpness = float(np.clip(sharpness, 0.0, 1.0))
    if sharpness >= 0.999:
        return frame_float
    sigma = BLUR_SIGMA_MAX * (1.0 - sharpness)
    return gaussian_filter(frame_float, sigma=(sigma, sigma, 0))


def measure_vision_ablation_sensitivity(model, n_states=40, seed=0, max_steps=120):
    """L2 distance between policy actions under full obs vs pixel-zeroed obs.

    Collects a fixed batch of observations by running the policy in a fresh
    vision env, then for each observation compares the deterministic action
    under the full obs to the action when the pixel columns are zeroed.
    Larger value = vision is load-bearing (policy "sees"); near 0 = vision
    is ignored (policy is still groping by proprio).
    """
    env = PlatformCreatureEnv(vision=True)
    from alien_baby.envs.platform_creature_env import CAM_HEIGHT, CAM_WIDTH
    proprio_dim = env.observation_space.shape[0] - CAM_HEIGHT * CAM_WIDTH * 3

    obs_list = []
    obs, _ = env.reset(seed=seed)
    obs_list.append(obs.copy())
    for _ in range(max_steps):
        if len(obs_list) >= n_states:
            break
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        obs_list.append(obs.copy())
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + len(obs_list))
    env.close()

    obs_batch = np.stack(obs_list[:n_states])
    obs_blind = obs_batch.copy()
    obs_blind[:, proprio_dim:] = 0.0

    actions_full, _ = model.predict(obs_batch, deterministic=True)
    actions_blind, _ = model.predict(obs_blind, deterministic=True)
    diffs = np.linalg.norm(actions_full - actions_blind, axis=-1)
    return float(diffs.mean())


def render_v8_episode(
    model,
    seed=0,
    max_steps=300,
    label=None,
    panels=3,
    smooth_alpha=DEFAULT_SMOOTH_ALPHA,
    sharpness=1.0,
):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"v8_{label or 'episode'}_seed{seed}.mp4"
    out_path = VIDEO_DIR / out_name

    env = PlatformCreatureEnv(vision=True)
    renderer = mujoco.Renderer(env.model, RENDER_HEIGHT, RENDER_WIDTH)

    if panels == 3:
        camera_names = ["overhead", "ringside", "head_cam"]
        labels = ["OVERHEAD", "RINGSIDE", "HEAD CAM"]
    elif panels == 2:
        camera_names = ["overhead", "head_cam"]
        labels = ["OVERHEAD", "HEAD CAM"]
    else:
        raise ValueError(f"panels must be 2 or 3, got {panels}")

    obs, _ = env.reset(seed=seed)
    writer = imageio.get_writer(str(out_path), fps=25, quality=8)

    head_ema = None
    last_composite = None
    steps_taken = 0
    touched = False
    fell = False
    for step in range(max_steps):
        frames = _render_cameras(renderer, env.data, camera_names)

        head_raw = frames["head_cam"]
        head_ema = _ema_blend(head_ema, head_raw, smooth_alpha)
        head_processed = _apply_blur(head_ema, sharpness)
        head_final = np.clip(head_processed, 0, 255).astype(np.uint8)
        frames["head_cam"] = head_final

        panels_out = [_add_label(frames[c], lbl)
                      for c, lbl in zip(camera_names, labels)]
        composite = np.concatenate(panels_out, axis=1)
        writer.append_data(composite)
        last_composite = composite

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps_taken = step + 1
        if terminated:
            touched = info.get("touched", False)
            fell = info.get("fell", False)
        if terminated or truncated:
            break

    if last_composite is not None:
        for _ in range(25):
            writer.append_data(last_composite)

    writer.close()
    renderer.close()
    env.close()
    outcome = "TOUCHED" if touched else ("FELL" if fell else "TIMEOUT")
    print(f"  Saved: {out_path}  ({steps_taken} steps, {outcome})")
    return str(out_path), outcome, steps_taken


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(RESULTS_DIR / "followon_v8_best" / "best_model.zip"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--panels", type=int, choices=[2, 3], default=3,
                        help="2 = overhead+head, 3 = overhead+ringside+head")
    parser.add_argument("--smooth-alpha", type=float, default=DEFAULT_SMOOTH_ALPHA,
                        help="EMA weight for head-cam smoothing (1.0 = no smoothing)")
    sharp_group = parser.add_mutually_exclusive_group()
    sharp_group.add_argument("--sharpness", type=float, default=None,
                             help="Head-cam sharpness in [0,1]. 1=sharp, 0=blurry.")
    sharp_group.add_argument("--auto-sharpness", action="store_true",
                             help="Derive sharpness from vision-ablation sensitivity")
    parser.add_argument("--ablation-max", type=float, default=0.5,
                        help="Sensitivity value that maps to sharpness=1 (calibrate after first run)")
    args = parser.parse_args()

    print(f"Loading: {args.checkpoint}")
    model = ConsistencySAC.load(args.checkpoint, device="cpu")

    if args.auto_sharpness:
        sensitivity = measure_vision_ablation_sensitivity(model)
        sharpness = float(np.clip(sensitivity / max(args.ablation_max, 1e-6), 0.0, 1.0))
        print(f"Vision-ablation sensitivity: {sensitivity:.4f}  "
              f"→ sharpness {sharpness:.2f} (normalized against {args.ablation_max})")
    elif args.sharpness is not None:
        sharpness = args.sharpness
        print(f"Sharpness: {sharpness:.2f} (manual)")
    else:
        sharpness = 1.0
        print("Sharpness: 1.00 (no blur)")

    results = []
    for seed in args.seeds:
        path, outcome, steps = render_v8_episode(
            model,
            seed=seed,
            max_steps=args.max_steps,
            label="best",
            panels=args.panels,
            smooth_alpha=args.smooth_alpha,
            sharpness=sharpness,
        )
        results.append((seed, outcome, steps))

    print("\nSummary:")
    for seed, outcome, steps in results:
        print(f"  seed={seed}  {outcome}  {steps} steps")
