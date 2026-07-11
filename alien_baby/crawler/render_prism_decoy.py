"""
render_prism_decoy.py — watchable video of the prism-decoy policy, so the aftereffect
decomposition (is it re-aiming, or a blind motor-search sweep?) can be SEEN, not just
tabulated. Renders N episodes (overhead | ringside | agent-eye panel) to mp4.

Reuses eval_prism_decoy.make_env (the ghost/displacement env) and its zero_pixels ablation.

  PYTHONPATH=<repo> python -u -m alien_baby.crawler.render_prism_decoy \
    --model alien_baby/results/prism_adapt_s2_rep_final.zip \
    --prism-offset 0 --episodes 4 --out alien_baby/results/videos/afterghost_s2_sighted_off0.mp4
"""
import argparse
import numpy as np
import mujoco
import imageio.v2 as imageio
from stable_baselines3 import PPO
from alien_baby.crawler.eval_prism_decoy import make_env, zero_pixels

CAMS = (("overhead", (420, 420)), ("ringside", (420, 420)))
EYE = (140, 140)


def _up(img, size):
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST))


def _ghost_geom_ids(env):
    ids = []
    for name in ("ghost_geom", "ghost2_geom"):
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            ids.append(gid)
    return ids


def _legend(img, offset):
    # Experimenter-eye annotation: solid ball = REAL/collidable, translucent = GHOST (seen).
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    if offset != 0.0:
        d = ImageDraw.Draw(im, "RGBA")
        d.text((6, 4), "solid = REAL (collidable/scored)", fill=(255, 255, 255, 255))
        d.text((6, 16), "translucent = GHOST (what AB sees, non-physical)", fill=(255, 255, 255, 255))
    return np.asarray(im)


def panel(env, eye_rgb, ghost_ids, offset):
    # Dim the ghost ONLY for the experimenter cameras (overhead/ringside), then restore —
    # the agent's eye (eye_rgb) renders separately at full opacity, so the policy's
    # observation is never touched by this display change.
    saved = {g: env.model.geom_rgba[g].copy() for g in ghost_ids}
    for g in ghost_ids:
        env.model.geom_rgba[g][3] = 0.30
    frames = []
    for cam, res in CAMS:
        r = mujoco.Renderer(env.model, res[0], res[1])
        r.update_scene(env.data, camera=cam)
        frames.append(r.render()); r.close()
    for g in ghost_ids:
        env.model.geom_rgba[g] = saved[g]
    frames.append(_up(eye_rgb, res[0]))
    return _legend(np.concatenate(frames, axis=1), offset)


def eye_from_obs(env):
    r = mujoco.Renderer(env.model, EYE[0], EYE[1])
    r.update_scene(env.data, camera="left_eye")
    out = r.render(); r.close(); return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prism-offset", type=float, default=0.0)
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env = make_env(args.prism_offset, 136.0, [0.70, 0.80], args.max_steps, 7)
    ghost_ids = _ghost_geom_ids(env)
    model = PPO.load(args.model, device="cpu")
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=100 + ep)
        for _ in range(args.max_steps):
            writer.append_data(panel(env, eye_from_obs(env), ghost_ids, args.prism_offset))
            o = zero_pixels(obs) if args.ablate else obs
            act, _ = model.predict(o, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                for _ in range(args.fps // 2):
                    writer.append_data(panel(env, eye_from_obs(env), ghost_ids, args.prism_offset))
                break
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
