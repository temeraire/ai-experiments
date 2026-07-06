"""
render_crawler_cart.py — render Phase G cart-substrate episodes.

Renders MimoCrawlerCartEnv rollouts at a forced ball offset so we can see
what the policy actually does on the curriculum's final task. Saves video
(mp4) plus 8 key PNG frames per seed so a vision-equipped reader can
inspect them with the Read tool.

Usage:
    python -m alien_baby.visualization.render_crawler_cart \
        --checkpoint alien_baby/results/mimo_phase_g_cart_v3_curriculum_best/best_model.zip \
        --offset 0.15 --seeds 0 1 2 --steps 400
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
FRAME_DIR   = RESULTS_DIR / "frames"
RENDER_W = 480
RENDER_H = 480


def _make_env_fn(offset, ball_y, max_steps, strength_scale, cart_speed,
                 hip_actuation, vision=False, memory_obs=False):
    def _init():
        env = MimoCrawlerCartEnv(
            vision=vision,
            max_steps=max_steps,
            strength_scale=strength_scale,
            fixed_ball_positions=[(offset, ball_y), (-offset, -ball_y)],
            cart_speed=cart_speed,
            hip_actuation=hip_actuation,
            memory_obs=memory_obs,
        )
        return env
    return _init


def _load_model_and_stats(checkpoint_path, offset, ball_y, max_steps,
                          strength_scale, cart_speed, hip_actuation,
                          vision=False, memory_obs=False):
    ckpt = pathlib.Path(checkpoint_path)
    vec_env = DummyVecEnv([_make_env_fn(offset, ball_y, max_steps,
                                        strength_scale, cart_speed,
                                        hip_actuation, vision=vision,
                                        memory_obs=memory_obs)])
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
        print("  WARNING: no vec_normalize.pkl found")
    model = SAC.load(str(ckpt), env=vec_env, device="cpu")
    return model, vec_env


def render_episode(model, vec_env, seed, offset, ball_y, max_steps,
                   strength_scale, cart_speed, hip_actuation, label,
                   dense_every=0, vision=False, memory_obs=False):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    inner_env = MimoCrawlerCartEnv(
        vision=vision,
        max_steps=max_steps,
        strength_scale=strength_scale,
        fixed_ball_positions=[(offset, ball_y), (-offset, -ball_y)],
        cart_speed=cart_speed,
        hip_actuation=hip_actuation,
        memory_obs=memory_obs,
    )
    inner_env.set_ball_positions((offset, ball_y), (-offset, -ball_y))
    renderer = mujoco.Renderer(inner_env.model, RENDER_H, RENDER_W)

    obs, _ = inner_env.reset(seed=seed)
    inner_env.set_ball_positions((offset, ball_y), (-offset, -ball_y))
    if hasattr(vec_env, "obs_rms"):
        obs_norm = vec_env.normalize_obs(obs)
    else:
        obs_norm = obs

    out_video = VIDEO_DIR / f"{label}_seed{seed}_off{offset:.2f}.mp4"
    writer = imageio.get_writer(str(out_video), fps=25, quality=8)
    composites = []
    total_reward = 0.0
    touched_b1 = False
    touched_b2 = False
    cart_xys = []
    last_step = 0

    for step in range(max_steps):
        # Two-panel: overhead + ringside
        renderer.update_scene(inner_env.data, camera="overhead")
        overhead = renderer.render().copy()
        renderer.update_scene(inner_env.data, camera="ringside")
        ringside = renderer.render().copy()
        composite = np.concatenate([overhead, ringside], axis=1)
        writer.append_data(composite)
        composites.append(composite)

        action, _ = model.predict(obs_norm.reshape(1, -1), deterministic=True)
        action = action[0]
        obs, reward, terminated, truncated, info = inner_env.step(action)
        total_reward += float(reward)
        if hasattr(vec_env, "obs_rms"):
            obs_norm = vec_env.normalize_obs(obs)
        else:
            obs_norm = obs

        cart_xys.append((info["cart_x"], info["cart_y"]))
        if info["touched_ball1"]:
            touched_b1 = True
        if info["touched_ball2"]:
            touched_b2 = True
        last_step = step + 1
        if terminated or truncated:
            break

    # Hold final frame for 1 second
    if composites:
        for _ in range(25):
            writer.append_data(composites[-1])
    writer.close()

    # Frame dump strategy:
    #   - 8 evenly-spaced "keyframes" (snapshot view)
    #   - dense sequential frames every `dense_every` steps (motion view).
    #     Per user suggestion, sequential frames reveal active reach vs
    #     passive sweep that single-frame snapshots can't distinguish.
    if composites:
        n_keys = min(8, len(composites))
        key_idxs = set(np.linspace(0, len(composites) - 1, n_keys, dtype=int).tolist())
        for k, idx in enumerate(sorted(key_idxs)):
            png_path = FRAME_DIR / f"{label}_seed{seed}_off{offset:.2f}_key{k}_t{idx}.png"
            imageio.imwrite(str(png_path), composites[idx])
        if dense_every > 0:
            for idx in range(0, len(composites), dense_every):
                if idx in key_idxs:
                    continue
                png_path = FRAME_DIR / f"{label}_seed{seed}_off{offset:.2f}_d_t{idx:04d}.png"
                imageio.imwrite(str(png_path), composites[idx])

    renderer.close()
    inner_env.close()

    print(f"  seed={seed}  steps={last_step}  reward={total_reward:+.1f}  "
          f"b1={'T' if touched_b1 else '-'} b2={'T' if touched_b2 else '-'}  "
          f"-> {out_video.name}")
    return {
        "seed": seed,
        "steps": last_step,
        "reward": total_reward,
        "b1": touched_b1,
        "b2": touched_b2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--offset", type=float, default=0.15,
                        help="Ball x-offset (default 0.15).")
    parser.add_argument("--ball-y", type=float, default=0.35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--strength-scale", type=float, default=0.7)
    parser.add_argument("--cart-speed", type=float, default=0.15)
    parser.add_argument("--hip-actuation", default="off", choices=["on", "off"])
    parser.add_argument("--label", default=None)
    parser.add_argument("--dense-every", type=int, default=0,
                        help="If >0, also dump every Nth frame as PNG (motion view).")
    parser.add_argument("--vision", action="store_true",
                        help="Build the env with stereo vision on (required for "
                             "models trained with --vision so the obs shape matches).")
    parser.add_argument("--memory-obs", action="store_true",
                        help="Include the 2-flag memory_obs in the proprio "
                             "(required for models trained with --memory-obs).")
    args = parser.parse_args()

    hip_act = (args.hip_actuation == "on")
    label = args.label or pathlib.Path(args.checkpoint).parent.name

    print(f"Loading: {args.checkpoint}")
    print(f"  offset={args.offset}  ball_y={args.ball_y}  hip_actuation={args.hip_actuation}")
    model, vec_env = _load_model_and_stats(
        args.checkpoint, args.offset, args.ball_y, args.steps,
        args.strength_scale, args.cart_speed, hip_act,
        vision=args.vision, memory_obs=args.memory_obs,
    )

    results = []
    for seed in args.seeds:
        results.append(render_episode(
            model, vec_env, seed,
            offset=args.offset,
            ball_y=args.ball_y,
            max_steps=args.steps,
            strength_scale=args.strength_scale,
            cart_speed=args.cart_speed,
            hip_actuation=hip_act,
            label=label,
            dense_every=args.dense_every,
            vision=args.vision,
            memory_obs=args.memory_obs,
        ))

    print("\n=== Summary ===")
    for r in results:
        touch = ("b1+b2" if r["b1"] and r["b2"]
                 else "b1" if r["b1"]
                 else "b2" if r["b2"]
                 else "-")
        print(f"  seed={r['seed']}  steps={r['steps']:>4}  "
              f"reward={r['reward']:>+9.1f}  touched={touch}")
    n_any = sum(1 for r in results if r["b1"] or r["b2"])
    n_both = sum(1 for r in results if r["b1"] and r["b2"])
    print(f"\nAny touch: {n_any}/{len(results)}   Both: {n_both}/{len(results)}")


if __name__ == "__main__":
    import os
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
