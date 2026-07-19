"""
render_gaze_check.py — SEE where AB is actually looking, and check it against the world.

Draws, on the world cameras (overhead | ringside), two rays from the eye:
  MAGENTA = AB's true gaze direction (the left_eye camera's look axis in the world)
  CYAN    = the direction from the eye to the RED target ball
and shows the agent's own eye POV (left_eye) as the third panel. It also prints the
angle between gaze and the ball each step. When the magenta and cyan rays line up AND
the ball is centered in the POV panel, the world is being shown coherently: "his head is
turned X deg, the ball is X deg that way, and sure enough it's in his view."

This is a VISUALIZATION only — it reuses eval_prism_decoy.make_env and never touches the
agent's observation or the physics. Additive to render_prism_decoy.py (leaves it untouched).

  PYTHONPATH=<repo> python -u -m alien_baby.crawler.render_gaze_check \
    --model alien_baby/results/dist_holdout_vis_s0_best/best_model.zip \
    --gaze-spawn --cone-deg 136 --radius 0.60 0.80 --episodes 4 \
    --out scratch_render/gaze_check.mp4
"""
import argparse
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from alien_baby.crawler.eval_prism_decoy import make_env

CAM_RES = 460
EYE_RES = 200
RAY_LEN = 0.85


def _id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


def _gaze_world(data, cam_id):
    """Eye world position and unit look direction (camera looks along -Z of cam_xmat)."""
    eye = data.cam_xpos[cam_id].copy()
    m = data.cam_xmat[cam_id].reshape(3, 3)
    fwd = -m[:, 2]
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    return eye, fwd


def _add_ray(scene, p0, p1, rgba, width=0.006):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(p0, np.float64), np.asarray(p1, np.float64))
    scene.ngeom += 1


def world_panel(env, cam, eye_id, ball_id):
    eye, gaze = _gaze_world(env.data, eye_id)
    ball = env.data.xpos[ball_id].copy()
    to_ball = ball - eye
    to_ball_u = to_ball / (np.linalg.norm(to_ball) + 1e-9)
    r = mujoco.Renderer(env.model, CAM_RES, CAM_RES)
    r.update_scene(env.data, camera=cam)
    _add_ray(r.scene, eye, eye + RAY_LEN * gaze, (1.0, 0.0, 1.0, 1.0))       # magenta = gaze
    _add_ray(r.scene, eye, ball, (0.0, 0.9, 1.0, 1.0))                        # cyan = to ball
    img = r.render(); r.close()
    # angle between gaze and ball direction (full 3D)
    ang = np.degrees(np.arccos(np.clip(np.dot(gaze, to_ball_u), -1, 1)))
    return img, ang


def eye_panel(env, eye_id):
    r = mujoco.Renderer(env.model, EYE_RES, EYE_RES)
    r.update_scene(env.data, camera="left_eye")
    img = r.render(); r.close()
    return np.asarray(Image.fromarray(img).resize((CAM_RES, CAM_RES), Image.NEAREST))


def annotate(img, ang, ep, t):
    im = Image.fromarray(img); d = ImageDraw.Draw(im, "RGBA")
    d.text((6, 4),  "MAGENTA = gaze (where he looks)   CYAN = to red ball", fill=(255, 255, 255, 255))
    d.text((6, 16), f"gaze-to-ball angle: {ang:5.1f} deg   (0 = looking straight at it)", fill=(255, 255, 0, 255))
    d.text((6, 28), f"ep {ep}  step {t}", fill=(255, 255, 255, 255))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prism-offset", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cone-deg", type=float, default=136.0)
    p.add_argument("--radius", type=float, nargs=2, default=[0.60, 0.80])
    p.add_argument("--gaze-spawn", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env = make_env(args.prism_offset, args.cone_deg, list(args.radius), args.max_steps, 7,
                   gaze_spawn=args.gaze_spawn)
    # Make GHOST balls clearly distinct from REAL balls: 0.3 opacity vs 1.0 (David 2026-07-18).
    # Ghosts are the non-physical "what AB sees under the lens" markers; at offset 0 they're
    # inert but still present, so dimming them stops them reading as a second real ball.
    for gname in ("ghost_geom", "ghost2_geom"):
        gid = _id(env.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid >= 0:
            env.model.geom_rgba[gid][3] = 0.3
    eye_id = _id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_eye")
    ball_id = _id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target")   # red target body
    model = PPO.load(args.model, device="cpu")
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=100 + ep)
        for t in range(args.max_steps):
            over, ang = world_panel(env, "overhead", eye_id, ball_id)
            ring, _ = world_panel(env, "ringside", eye_id, ball_id)
            eye = eye_panel(env, eye_id)
            frame = np.concatenate([annotate(over, ang, ep, t), ring, eye], axis=1)
            writer.append_data(frame)
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                for _ in range(args.fps // 2):
                    writer.append_data(frame)
                break
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
