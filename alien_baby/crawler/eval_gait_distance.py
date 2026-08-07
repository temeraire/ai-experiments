"""eval_gait_distance.py — does the gait actually WALK (cover ground), not just hold a heading?

The v3 gait was declared "works" on a heading test (turn-left/right) that never measured how far
it travelled forward. This measures the thing that matters for a driver on top of it: at a
commanded forward speed, what forward (body-frame) velocity and net world displacement does it
actually produce? And does it turn on command while still moving?

  python -m alien_baby.crawler.eval_gait_distance --model alien_baby/results/ant_gait_v4_best/best_model.zip
"""
import argparse
import numpy as np, mujoco
from stable_baselines3 import PPO
from alien_baby.crawler.train_ant_gait import AntGaitEnv


def _fwd_vel_body(env):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, env.data.qpos[3:7]); R = R.reshape(3, 3)
    return float((R.T @ env.data.qvel[0:3])[0])


def roll(model, env, cmd, steps=400, seed=3):
    env.reset(seed=seed)
    p0 = env.data.qpos[0:2].copy()
    vxs, fell = [], False
    for _ in range(steps):
        env.cmd = np.array(cmd, np.float32)
        a, _ = model.predict(env._obs(), deterministic=True)
        _, _, term, trunc, _ = env.step(a)
        vxs.append(_fwd_vel_body(env))
        if term:
            fell = True; break
    disp = float(np.linalg.norm(env.data.qpos[0:2] - p0))
    secs = len(vxs) * env.n_sub * 0.01
    return dict(mean_fwd_vel=float(np.mean(vxs)), net_disp=disp,
                speed=disp / secs if secs else 0.0, secs=secs, fell=fell)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()
    model = PPO.load(args.model, device="cpu")
    env = AntGaitEnv(seed=3)
    NSEED = 5
    print(f"=== gait distance eval: {args.model}   ({NSEED} seeds/command) ===")
    print(f"{'command':>22} {'fwd_vel(mean)':>13} {'net_disp_m(mean+-sd)':>22} {'m/s':>6} {'falls':>6}")
    for cmd, label in [([0.6, 0.0], "walk 0.6, straight"),
                       ([1.0, 0.0], "walk 1.0, straight"),
                       ([1.2, 0.0], "walk 1.2, straight"),
                       ([0.8, 0.5], "walk 0.8, turn +0.5"),
                       ([0.8, -0.5], "walk 0.8, turn -0.5")]:
        rs = [roll(model, env, cmd, steps=args.steps, seed=s) for s in range(NSEED)]
        disp = np.array([r["net_disp"] for r in rs])
        fv = float(np.mean([r["mean_fwd_vel"] for r in rs]))
        secs = rs[0]["secs"]
        falls = sum(r["fell"] for r in rs)
        print(f"{label:>22} {fv:>13.3f} {disp.mean():>13.2f} +-{disp.std():>5.2f}   "
              f"{disp.mean()/secs:>6.2f} {falls:>3}/{NSEED}")
    print("\nPASS if STRAIGHT commands give net_disp growing with command (a real, controllable"
          " walker), fwd_vel clearly >0, few falls -- not the flat/inverse pattern v4 showed.")


if __name__ == "__main__":
    main()
