"""
verify_gaze_fov.py — prove the gaze instrument is honest.

For each step it computes, from geometry, whether the RED ball is inside the eye's
field of view (angle between the eye's look axis and the eye->ball direction < fovy/2),
then independently renders the eye POV and checks whether red ball pixels are actually
present. If the geometric "should see it" agrees with the pixel "does see it", the gaze
ray / FOV logic in render_gaze_check is trustworthy — i.e. "he can see what we expect."

Prints a confusion table. No video; just numbers.
"""
import argparse
import numpy as np
import mujoco
from stable_baselines3 import PPO
from alien_baby.crawler.eval_prism_decoy import make_env


def gaze_world(data, cam_id):
    eye = data.cam_xpos[cam_id].copy()
    fwd = -data.cam_xmat[cam_id].reshape(3, 3)[:, 2]
    return eye, fwd / (np.linalg.norm(fwd) + 1e-9)


def red_pixels(img):
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return int(np.sum((r > 150) & (g < 80) & (b < 80)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--cone-deg", type=float, default=136.0)
    p.add_argument("--radius", type=float, nargs=2, default=[0.60, 0.80])
    p.add_argument("--half-fov", type=float, default=60.0, help="fovy/2 in deg (fovy=120)")
    p.add_argument("--gaze-spawn", action="store_true")
    args = p.parse_args()

    env = make_env(0.0, args.cone_deg, list(args.radius), args.max_steps, 7,
                   gaze_spawn=args.gaze_spawn)
    eye_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_eye")
    ball_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target")
    model = PPO.load(args.model, device="cpu")
    rend = mujoco.Renderer(env.model, 96, 96)

    # confusion: predicted in-FOV (geometry) vs observed red pixels (POV)
    tp = fp = tn = fn = 0
    angs_seen, angs_unseen = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=100 + ep)
        for t in range(args.max_steps):
            eye, gaze = gaze_world(env.data, eye_id)
            to_ball = env.data.xpos[ball_id] - eye
            ang = np.degrees(np.arccos(np.clip(
                np.dot(gaze, to_ball / (np.linalg.norm(to_ball) + 1e-9)), -1, 1)))
            pred_in = ang < args.half_fov
            rend.update_scene(env.data, camera="left_eye")
            seen = red_pixels(rend.render()) >= 2
            (angs_seen if seen else angs_unseen).append(ang)
            if pred_in and seen: tp += 1
            elif pred_in and not seen: fp += 1
            elif not pred_in and not seen: tn += 1
            else: fn += 1
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
    rend.close()
    n = tp + fp + tn + fn
    agree = (tp + tn) / max(n, 1)
    print(f"steps={n}")
    print(f"  geometry says IN-FOV & POV shows red   (agree): {tp}")
    print(f"  geometry says OUT     & POV shows none  (agree): {tn}")
    print(f"  geometry IN  but POV none  (edge/occlusion): {fp}")
    print(f"  geometry OUT but POV red   (SHOULD NOT HAPPEN): {fn}")
    print(f"  agreement = {agree*100:.1f}%")
    if angs_seen:
        print(f"  angle when ball VISIBLE in POV:  mean={np.mean(angs_seen):.1f} max={np.max(angs_seen):.1f}")
    if angs_unseen:
        print(f"  angle when ball NOT visible:     mean={np.mean(angs_unseen):.1f} min={np.min(angs_unseen):.1f}")


if __name__ == "__main__":
    main()
