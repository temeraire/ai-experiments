"""
v7 evaluation: v6 battery + new developmental metrics.

New metrics:
  - in_view_before_contact_frac: fraction of successful episodes where the
    target entered the head cam FOV at least once before fingertip contact.
    This is the "saw it before touching it" criterion.
  - mean_tracking_error: average angular distance (degrees) between the head
    camera's view direction and the direction to the target, over all steps.
    Lower = better tracking.
  - mean_time_to_contact: mean steps until termination, among successful
    episodes. Lower = more efficient.
"""

import pathlib
import re
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from alien_baby.envs import TabletopMovingGazeEnv
from alien_baby.envs.flatten_wrapper import FlattenVisionWrapper
from alien_baby.agents.train_v5 import ConsistencySAC

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
PROPRIO_DIM = 9
HEAD_CAM_FOV_DEG = 22.0  # v7 narrow FOV


def _hidden1(model, obs_np):
    with torch.no_grad():
        x = torch.as_tensor(obs_np, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return model.actor.latent_pi[1](model.actor.latent_pi[0](x)).numpy()


def _cka_linear(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def _cam_to_target_angle_deg(env_unwrapped):
    """Angle (degrees) between head cam view direction and direction to target."""
    data = env_unwrapped.data
    cam_id = env_unwrapped._head_cam_id
    tgt_body = env_unwrapped._target_body_ids[env_unwrapped.target_object]
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    view_dir = -cam_mat[:, 2]
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-10)
    to_tgt = data.xpos[tgt_body] - cam_pos
    to_tgt = to_tgt / (np.linalg.norm(to_tgt) + 1e-10)
    cos = np.clip(float(np.dot(view_dir, to_tgt)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def rollout_with_metrics(model, env, n_episodes=30, seed_base=0):
    """
    Roll out, gathering per-episode and per-step metrics.
    Returns dict with:
      observations, actions — concatenated across episodes
      episode_successes — list of bool
      episode_lengths — list of int
      in_view_per_step, near_per_step — per-step bool arrays
      cam_to_tgt_angles_deg — per-step float array
      episodes_saw_target_before_contact — list of bool (only for successful ones)
    """
    obs_all, act_all = [], []
    in_view_all, near_all, ang_all = [], [], []
    ep_success, ep_len = [], []
    ep_saw_before_contact = []

    unwrapped = env.unwrapped
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        saw_in_view = False
        for step in range(200):
            obs_all.append(obs.copy())
            ang = _cam_to_target_angle_deg(unwrapped)
            in_view = ang < HEAD_CAM_FOV_DEG / 2.0
            near = unwrapped._get_fingertip_target_dist() < 0.08
            in_view_all.append(in_view)
            near_all.append(near)
            ang_all.append(ang)
            if in_view:
                saw_in_view = True
            action, _ = model.predict(obs, deterministic=True)
            act_all.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated:
                ep_success.append(True)
                ep_len.append(step + 1)
                ep_saw_before_contact.append(saw_in_view)
                break
            if truncated:
                ep_success.append(False)
                ep_len.append(step + 1)
                break
        else:
            ep_success.append(False)
            ep_len.append(200)

    return {
        "observations": np.array(obs_all),
        "actions": np.array(act_all),
        "in_view": np.array(in_view_all),
        "near": np.array(near_all),
        "angles_deg": np.array(ang_all),
        "episode_successes": ep_success,
        "episode_lengths": ep_len,
        "saw_before_contact": ep_saw_before_contact,
    }


def summarize(result):
    successes = sum(result["episode_successes"])
    n_ep = len(result["episode_successes"])
    mean_time_to_contact = (
        np.mean([l for l, s in zip(result["episode_lengths"], result["episode_successes"]) if s])
        if successes > 0 else float("nan")
    )
    in_view_frac = float(result["in_view"].mean())
    in_view_near = float(result["in_view"][result["near"]].mean()) if result["near"].any() else float("nan")
    in_view_far = float(result["in_view"][~result["near"]].mean()) if (~result["near"]).any() else float("nan")
    mean_angle = float(result["angles_deg"].mean())
    saw_frac = (
        float(np.mean(result["saw_before_contact"]))
        if len(result["saw_before_contact"]) > 0 else float("nan")
    )
    return {
        "successes": successes,
        "n_episodes": n_ep,
        "mean_time_to_contact": mean_time_to_contact,
        "in_view_frac": in_view_frac,
        "in_view_frac_near": in_view_near,
        "in_view_frac_far": in_view_far,
        "mean_tracking_error_deg": mean_angle,
        "saw_before_contact_frac": saw_frac,
    }


def eval_success(model, noise_levels=(0.0, 1.0), n_episodes=20):
    env = Monitor(TabletopMovingGazeEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    results = {}
    for noise in noise_levels:
        # Reuse the noisy-vision collector from v6 approach
        successes = 0
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            for step in range(200):
                noisy = obs.copy()
                if noise > 0:
                    rng = np.random.RandomState(ep * 200 + step).randn(
                        len(noisy) - PROPRIO_DIM
                    ).astype(np.float32)
                    noisy[PROPRIO_DIM:] = noisy[PROPRIO_DIM:] * (1 - noise) + rng * noise
                action, _ = model.predict(noisy, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated:
                    successes += 1
                    break
                if truncated:
                    break
        results[noise] = successes
        print(f"  noise={noise:.1f}: {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")
    env.close()
    return results


def eval_cka(v7, stage1, n_episodes=30):
    env = Monitor(TabletopMovingGazeEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    result = rollout_with_metrics(v7, env, n_episodes=n_episodes)
    env.close()
    obs = result["observations"]
    h_v7 = _hidden1(v7, obs)
    h_s1 = _hidden1(stage1, obs[:, :PROPRIO_DIM])
    return _cka_linear(h_v7, h_s1), len(obs)


def eval_developmental_arc(stage1_model):
    """
    Run the rollout-with-metrics on every saved follow-on checkpoint and
    return a list of (step_count, summary_dict).
    """
    ckpt_dir = RESULTS_DIR / "followon_v7_checkpoints"
    paths = sorted(ckpt_dir.glob("followon_v7_*_steps.zip"))
    checkpoints = []
    for p in paths:
        m = re.search(r"(\d+)_steps\.zip$", p.name)
        if m:
            checkpoints.append((int(m.group(1)), str(p).replace(".zip", "")))
    checkpoints.sort()

    env = Monitor(TabletopMovingGazeEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)

    arc = []
    for steps, path in checkpoints:
        model = ConsistencySAC.load(path)
        result = rollout_with_metrics(model, env, n_episodes=20)
        summary = summarize(result)
        arc.append((steps, summary))
        print(
            f"  step {steps:>8}: succ={summary['successes']}/20, "
            f"tracking_err={summary['mean_tracking_error_deg']:.1f} deg, "
            f"in_view={summary['in_view_frac']*100:.0f}%, "
            f"saw-before-contact={summary['saw_before_contact_frac']*100:.0f}%"
        )
    env.close()
    return arc


def main():
    final_path = str(RESULTS_DIR / "followon_v7_checkpoint")
    best_path = str(RESULTS_DIR / "followon_v7_best" / "best_model")
    stage1_path = str(RESULTS_DIR / "stage1_v7_checkpoint")

    print("=" * 60)
    print("v7 evaluation (moving target + narrow-FOV gaze)")
    print("=" * 60)

    stage1 = SAC.load(stage1_path)

    # Evaluate both final and best
    for label, path in [("BEST", best_path), ("FINAL", final_path)]:
        if not pathlib.Path(path + ".zip").exists():
            continue
        print(f"\n--- {label} checkpoint ---")
        v7 = ConsistencySAC.load(path)

        env = Monitor(TabletopMovingGazeEnv(vision=True, target_object=0, hide_target_offset=True))
        env = FlattenVisionWrapper(env)
        result = rollout_with_metrics(v7, env, n_episodes=30)
        env.close()
        summary = summarize(result)

        print(f"  success: {summary['successes']}/{summary['n_episodes']}")
        print(f"  mean time-to-contact (successes only): {summary['mean_time_to_contact']:.1f} steps")
        print(f"  saw-target-before-contact (of successes): {summary['saw_before_contact_frac']*100:.0f}%")
        print(f"  in-view overall: {summary['in_view_frac']*100:.1f}%")
        print(f"  in-view when near: {summary['in_view_frac_near']*100:.1f}%")
        print(f"  in-view when far:  {summary['in_view_frac_far']*100:.1f}%")
        print(f"  mean tracking error: {summary['mean_tracking_error_deg']:.1f} deg (half-FOV={HEAD_CAM_FOV_DEG/2})")

        cka, n = eval_cka(v7, stage1)
        print(f"  CKA(hidden1, stage1): {cka:.4f} (n={n})")

        print("\n  Task success with vision noise:")
        eval_success(v7)

    # Developmental arc across all checkpoints
    print("\n" + "=" * 60)
    print("Developmental arc across checkpoints:")
    print("=" * 60)
    arc = eval_developmental_arc(stage1)

    # Save arc to a simple text file for later reference
    arc_path = RESULTS_DIR / "v7_developmental_arc.txt"
    with open(arc_path, "w") as f:
        f.write("step\tsuccess\ttime_to_contact\ttracking_err_deg\tin_view_frac\tin_view_near\tin_view_far\tsaw_before_contact\n")
        for steps, s in arc:
            f.write(
                f"{steps}\t{s['successes']}\t{s['mean_time_to_contact']:.1f}\t"
                f"{s['mean_tracking_error_deg']:.2f}\t{s['in_view_frac']:.3f}\t"
                f"{s['in_view_frac_near']:.3f}\t{s['in_view_frac_far']:.3f}\t"
                f"{s['saw_before_contact_frac']:.3f}\n"
            )
    print(f"\nDevelopmental arc written to {arc_path}")


if __name__ == "__main__":
    main()
