"""
render_decoy_shape.py — watchable video of the decoy policy choosing by COLOR across
SHAPES. Renders N episodes (overhead | ringside | agent-eye panel) to mp4 so the
shape-invariance result is visible, not just tabulated.

  PYTHONPATH=<repo> python -u -m alien_baby.crawler.render_decoy_shape \
    --model alien_baby/results/decoy_v2_ext_s0_best/best_model.zip \
    --red box --blue sphere --episodes 4 --out alien_baby/results/videos/decoy_shape_Rbox_Bsph.mp4
"""
import argparse
import numpy as np
import mujoco
import imageio.v2 as imageio
from stable_baselines3 import PPO
from alien_baby.crawler.eval_decoy_shape import make_env

CAMS = (("overhead", (420, 420)), ("ringside", (420, 420)))
EYE = (140, 140)


def _up(img, size):
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST))


def panel(env, eye_rgb):
    frames = []
    for cam, res in CAMS:
        r = mujoco.Renderer(env.model, res[0], res[1])
        r.update_scene(env.data, camera=cam)
        frames.append(r.render()); r.close()
    frames.append(_up(eye_rgb, res[0]))   # agent eye, upscaled, as the right panel
    return np.concatenate(frames, axis=1)


def eye_from_obs(env):
    # render the agent's left_eye at display res (the policy sees a 32x32 version)
    r = mujoco.Renderer(env.model, EYE[0], EYE[1])
    r.update_scene(env.data, camera="left_eye")
    out = r.render(); r.close(); return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--red", default="sphere")
    p.add_argument("--blue", default="sphere")
    p.add_argument("--target-color", default=None)
    p.add_argument("--decoy-color", default=None)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env = make_env(args.red, args.blue, 136.0, [0.70, 0.80], args.max_steps, 7,
                   target_color=args.target_color, decoy_color=args.decoy_color)
    model = PPO.load(args.model, device="cpu")
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=100 + ep)
        for _ in range(args.max_steps):
            writer.append_data(panel(env, eye_from_obs(env)))
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                # hold last frame briefly
                for _ in range(args.fps // 2):
                    writer.append_data(panel(env, eye_from_obs(env)))
                break
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
