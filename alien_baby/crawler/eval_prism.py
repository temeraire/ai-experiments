"""
eval_prism.py — measure REACH DIRECTION under a prism, the objective signal of the prism
experiment (not the subjective percept). For a policy + offset, over N episodes:
  - real-contact rate (touches the hidden solid ball),
  - reach-to-REAL vs reach-to-GHOST fraction (net displacement projected onto the real-bearing
    vs ghost-bearing directions),
  - mean net-displacement bearing vs the real and ghost bearings.

Protocol (run this 3x):
  A baseline : offset 0, base policy     -> crawls to the ball (ghost==real).
  B pre      : offset D, base policy     -> should chase the GHOST (visual), miss the real.
  B adapted  : offset D, adapted policy  -> should re-map to the REAL side (contact recovers).
  C after    : offset 0, adapted policy  -> AFTEREFFECT: should now be biased to the wrong side.

Usage:
  PYTHONPATH=<repo> python -u -m alien_baby.crawler.eval_prism \
    --model <policy.zip> --prism-offset 30 --eval-eps 40 --run-tag prism_B_adapted [--render-eps 3]
"""
import argparse
import pathlib

import numpy as np
import mujoco
import imageio

from stable_baselines3 import PPO
from alien_baby.crawler.residual_vision_policy import ResidualVisionPolicy  # noqa: F401
from alien_baby.crawler.slotfill_vision_policy import SlotFillVisionPolicy  # noqa: F401
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES

RESULTS = pathlib.Path(__file__).parent.parent / "results"
XML = "alien_baby/crawler/mimo_crawler_pos_wide_prism.xml"


def make_env(offset, cone_deg, radius, max_steps, seed):
    env = MimoCrawlerEnv(
        vision=True, stereo=True, target_obs=False, crawl_pose=CRAWL_POSES["arms_fwd"],
        action_mode="position_offset", xml_path=XML, spawn_cone_deg=cone_deg,
        spawn_radius=tuple(radius), random_start_orientation=False, max_steps=max_steps,
        step_cost=0.0, approach_reward_scale=10.0, velocity_bonus_scale=0.0,
        terminate_tilt_deg=50.0, tip_penalty=-5.0, prism_offset_deg=offset,
    )
    env.reset(seed=seed)
    return env


def _real_ghost_xy(env):
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")
    q = env.model.jnt_qposadr[jid]
    real = env.data.qpos[q:q + 2].copy()
    ghost = env.data.mocap_pos[env._ghost_mocap_id][:2].copy() if env._ghost_mocap_id >= 0 else real
    return real, ghost


def evaluate(model, env, n_eps, max_steps):
    contacts = 0
    reach_real = 0; reach_defined = 0
    dr_list, dg_list = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        root0 = obs[:2].copy()
        real, ghost = _real_ghost_xy(env)
        touched = False
        for _ in range(max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if info.get("touched_ball1"):
                touched = True
            if term or trunc:
                break
        rootF = obs[:2].copy()
        disp = rootF - root0
        vr = real - root0; vg = ghost - root0
        nr, ng = np.linalg.norm(vr), np.linalg.norm(vg)
        if nr > 1e-6 and ng > 1e-6 and np.linalg.norm(disp) > 0.03:
            pr = float(disp @ (vr / nr)); pg = float(disp @ (vg / ng))
            reach_real += int(pr > pg); reach_defined += 1
        dr_list.append(float(np.linalg.norm(rootF - real)))
        dg_list.append(float(np.linalg.norm(rootF - ghost)))
        contacts += int(touched)
    return dict(
        contact=contacts / n_eps, n=n_eps,
        reach_real_frac=(reach_real / reach_defined) if reach_defined else float("nan"),
        reach_defined=reach_defined,
        end_dist_real=float(np.mean(dr_list)), end_dist_ghost=float(np.mean(dg_list)),
    )


def render(model, env, tag, n_eps, max_steps):
    """Debug render with the REAL ball made visible (enable geom group 3)."""
    vdir = RESULTS / "videos"; vdir.mkdir(parents=True, exist_ok=True)
    out = vdir / f"eval_{tag}.mp4"
    vopt = mujoco.MjvOption(); vopt.geomgroup[3] = 1   # show the hidden real ball for debugging
    r = mujoco.Renderer(env.model, 480, 480)
    w = imageio.get_writer(str(out), fps=25, quality=8)
    for _ in range(n_eps):
        obs, _ = env.reset()
        for _ in range(max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            r.update_scene(env.data, camera="overhead", scene_option=vopt); over = r.render().copy()
            r.update_scene(env.data, camera="ringside", scene_option=vopt); ring = r.render().copy()
            w.append_data(np.concatenate([over, ring], axis=1))
            if term or trunc:
                break
    w.close(); r.close(); return str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prism-offset", type=float, default=0.0)
    p.add_argument("--cone-deg", type=float, default=136.0)
    p.add_argument("--radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--eval-eps", type=int, default=40)
    p.add_argument("--render-eps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-tag", default="prism_eval")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    print(f"\n=== PRISM eval {args.run_tag} (offset +{args.prism_offset:.0f}) ===\n  model={args.model}")
    env = make_env(args.prism_offset, args.cone_deg, args.radius, args.max_steps, args.seed)
    m = evaluate(model, env, args.eval_eps, args.max_steps)
    print(f"  real-contact rate     : {m['contact']*100:.1f}%   ({m['n']} eps)")
    print(f"  reach-to-REAL fraction: {m['reach_real_frac']*100:.1f}%  "
          f"(vs ghost {100-m['reach_real_frac']*100:.1f}%; {m['reach_defined']} moved)")
    print(f"  end dist to REAL      : {m['end_dist_real']:.3f} m")
    print(f"  end dist to GHOST     : {m['end_dist_ghost']:.3f} m")
    print(f"  -> ends closer to {'REAL' if m['end_dist_real']<m['end_dist_ghost'] else 'GHOST'}")
    if args.render_eps > 0:
        renv = make_env(args.prism_offset, args.cone_deg, args.radius, args.max_steps, args.seed + 7)
        print(f"  video: {render(model, renv, args.run_tag, args.render_eps, args.max_steps)}")
        renv.close()
    env.close()


if __name__ == "__main__":
    main()
