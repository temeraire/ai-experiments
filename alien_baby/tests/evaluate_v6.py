"""
v6 evaluation: success (clean/noisy), CKA vs stage1_v6, neighbor-consistency,
and new gaze-specific metrics (ball-in-view fraction, gaze-on-target when near).

Benchmarks v6 follow-on against stage1_v6 (the frozen proprio base).
"""

import pathlib
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from alien_baby.envs import TabletopGazeEnv
from alien_baby.envs.flatten_wrapper import FlattenVisionWrapper
from alien_baby.agents.train_v5 import ConsistencySAC

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
PROPRIO_DIM = 9
HEAD_CAM_FOV_DEG = 45.0  # must match tabletop_v6.xml


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


def _target_in_view(env_unwrapped):
    """True if the head camera's view direction is within FOV/2 of the target."""
    import mujoco
    data = env_unwrapped.data
    cam_id = env_unwrapped._head_cam_id
    tgt_body = env_unwrapped._target_body_ids[env_unwrapped.target_object]

    cam_pos = data.cam_xpos[cam_id]
    # MuJoCo stores camera frame in xmat as 3x3 row-major; cam looks along -z_cam.
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    cam_view_dir = -cam_mat[:, 2]
    cam_view_dir = cam_view_dir / (np.linalg.norm(cam_view_dir) + 1e-10)

    tgt_pos = data.xpos[tgt_body]
    to_tgt = tgt_pos - cam_pos
    to_tgt_norm = to_tgt / (np.linalg.norm(to_tgt) + 1e-10)

    cos_angle = float(np.dot(cam_view_dir, to_tgt_norm))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    # Check both horizontal and vertical within half-FOV. For simplicity use
    # isotropic half-FOV (the image is square so this is exact for the square FOV).
    half_fov_rad = np.deg2rad(HEAD_CAM_FOV_DEG / 2.0)
    return angle_rad < half_fov_rad


def collect_rollout_v6(model, env, n_episodes=20, noise=0.0, seed_base=0):
    """
    Roll out policy. Returns (obs_full, actions, successes, in_view_flags, near_flags).
    in_view_flags[i] = True iff target is in head cam FOV at step i.
    near_flags[i] = True iff fingertip-to-target distance < 0.08 at step i.
    """
    observations, actions = [], []
    in_view, near = [], []
    successes = 0
    unwrapped = env.unwrapped
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        for step in range(200):
            noisy = obs.copy()
            if noise > 0:
                vision = noisy[PROPRIO_DIM:]
                rng = np.random.RandomState(ep * 200 + step).randn(*vision.shape).astype(np.float32)
                noisy[PROPRIO_DIM:] = vision * (1 - noise) + rng * noise
            observations.append(noisy.copy())
            in_view.append(_target_in_view(unwrapped))
            near.append(unwrapped._get_fingertip_target_dist() < 0.08)
            action, _ = model.predict(noisy, deterministic=True)
            actions.append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            if terminated:
                successes += 1
                break
            if truncated:
                break
    return (
        np.array(observations),
        np.array(actions),
        successes,
        np.array(in_view),
        np.array(near),
    )


def eval_success(model, noise_levels=(0.0, 1.0), n_episodes=20):
    env = Monitor(TabletopGazeEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    results = {}
    for noise in noise_levels:
        _, _, successes, _, _ = collect_rollout_v6(model, env, n_episodes=n_episodes, noise=noise)
        results[noise] = successes
        print(f"  noise={noise:.1f}: {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")
    env.close()
    return results


def eval_cka_to_stage1(model, stage1_model, n_episodes=30):
    env = Monitor(TabletopGazeEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    obs_full, _, _, _, _ = collect_rollout_v6(model, env, n_episodes=n_episodes)
    env.close()

    h_model = _hidden1(model, obs_full)
    h_stage1 = _hidden1(stage1_model, obs_full[:, :PROPRIO_DIM])
    cka = _cka_linear(h_model, h_stage1)
    return cka, len(obs_full)


def eval_equivalence_class(model, n_episodes=30, k=5, n_random_pairs=2000):
    env = Monitor(TabletopGazeEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    obs, actions, _, _, _ = collect_rollout_v6(model, env, n_episodes=n_episodes)
    env.close()

    h = _hidden1(model, obs)
    n = len(h)
    d = np.linalg.norm(h[:, None, :] - h[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    neighbors = np.argsort(d, axis=1)[:, :k]

    neighbor_action_dists = []
    for i in range(n):
        for j in neighbors[i]:
            neighbor_action_dists.append(np.linalg.norm(actions[i] - actions[j]))
    mean_neighbor_dist = np.mean(neighbor_action_dists)

    rng = np.random.RandomState(0)
    idx_a = rng.randint(0, n, n_random_pairs)
    idx_b = rng.randint(0, n, n_random_pairs)
    mean_random_dist = np.mean(np.linalg.norm(actions[idx_a] - actions[idx_b], axis=1))

    ratio = mean_neighbor_dist / (mean_random_dist + 1e-10)
    return {
        "mean_neighbor_action_dist": float(mean_neighbor_dist),
        "mean_random_action_dist": float(mean_random_dist),
        "ratio": float(ratio),
        "n_samples": n,
    }


def eval_gaze(model, n_episodes=30):
    """
    Gaze-behavior metrics:
      - in_view_frac: fraction of all steps with target in head cam FOV
      - in_view_frac_near: conditional on fingertip within 0.08 of target
        (does the agent look at the ball when its hand is reaching it?)
      - in_view_frac_far: complement (do we look less when far from target?)
    """
    env = Monitor(TabletopGazeEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    _, _, _, in_view, near = collect_rollout_v6(model, env, n_episodes=n_episodes)
    env.close()

    total = len(in_view)
    near_mask = near
    far_mask = ~near
    return {
        "in_view_frac": float(in_view.mean()),
        "in_view_frac_near": float(in_view[near_mask].mean()) if near_mask.any() else float("nan"),
        "in_view_frac_far": float(in_view[far_mask].mean()) if far_mask.any() else float("nan"),
        "n_total": int(total),
        "n_near": int(near_mask.sum()),
    }


def main():
    v6_path = str(RESULTS_DIR / "followon_v6_checkpoint")
    stage1_path = str(RESULTS_DIR / "stage1_v6_checkpoint")

    print("=" * 60)
    print("v6 evaluation (gaze camera + consistency loss)")
    print("=" * 60)

    v6 = ConsistencySAC.load(v6_path)
    stage1 = SAC.load(stage1_path)

    print("\n--- Task success ---")
    v6_success = eval_success(v6)

    print("\n--- CKA(hidden1, stage1_v6.hidden1) ---")
    print("  (higher = vision stays on proprio's manifold)")
    v6_cka, n_v6 = eval_cka_to_stage1(v6, stage1)
    print(f"  v6: CKA = {v6_cka:.4f} (n={n_v6})")

    print("\n--- Equivalence-class neighbor-consistency ratio ---")
    print("  (lower = tighter classes; neighbors behave more similarly than chance)")
    v6_eq = eval_equivalence_class(v6)
    print(f"  v6: ratio = {v6_eq['ratio']:.3f} "
          f"(neighbor={v6_eq['mean_neighbor_action_dist']:.3f}, "
          f"random={v6_eq['mean_random_action_dist']:.3f}, n={v6_eq['n_samples']})")

    print("\n--- Gaze behavior ---")
    print("  (higher in_view_frac_near than in_view_frac_far means the agent")
    print("   learned to point its head at the target when approaching it)")
    v6_gaze = eval_gaze(v6)
    print(f"  overall in-view: {v6_gaze['in_view_frac']*100:.1f}% (n={v6_gaze['n_total']})")
    print(f"  in-view when near target: {v6_gaze['in_view_frac_near']*100:.1f}% (n={v6_gaze['n_near']})")
    print(f"  in-view when far from target: {v6_gaze['in_view_frac_far']*100:.1f}%")

    print("\n--- Summary ---")
    print(f"  v6 success: clean={v6_success[0.0]}/20, 100%noise={v6_success[1.0]}/20")
    print(f"  CKA to stage1: {v6_cka:.3f}")
    print(f"  Neighbor ratio: {v6_eq['ratio']:.3f}")
    print(f"  Gaze near/far in-view: {v6_gaze['in_view_frac_near']*100:.0f}% / "
          f"{v6_gaze['in_view_frac_far']*100:.0f}%")


if __name__ == "__main__":
    main()
