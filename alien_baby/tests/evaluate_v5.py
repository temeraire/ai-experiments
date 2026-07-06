"""
v5 evaluation: success (clean/noisy), CKA vs stage1_v3, neighbor-consistency.

Benchmarks v5 against v3 follow-on (same architecture, no consistency loss)
and stage1_v3 (the frozen proprio base).
"""

import pathlib
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from alien_baby.envs import TabletopReachEnv
from alien_baby.envs.flatten_wrapper import FlattenVisionWrapper
from alien_baby.agents.train_v5 import ConsistencySAC

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
PROPRIO_DIM = 7


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


def collect_rollout(model, env, n_episodes=20, noise=0.0, seed_base=0):
    """Roll out policy, return (obs_full, hidden1, actions, successes)."""
    observations, actions = [], []
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        for step in range(200):
            noisy = obs.copy()
            if noise > 0:
                vision = noisy[PROPRIO_DIM:]
                rng = np.random.RandomState(ep * 200 + step).randn(*vision.shape).astype(np.float32)
                noisy[PROPRIO_DIM:] = vision * (1 - noise) + rng * noise
            observations.append(noisy.copy())
            action, _ = model.predict(noisy, deterministic=True)
            actions.append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            if terminated:
                successes += 1
                break
            if truncated:
                break
    return np.array(observations), np.array(actions), successes


def eval_success(model, noise_levels=(0.0, 1.0), n_episodes=20):
    env = Monitor(TabletopReachEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    results = {}
    for noise in noise_levels:
        _, _, successes = collect_rollout(model, env, n_episodes=n_episodes, noise=noise)
        results[noise] = successes
        print(f"  noise={noise:.1f}: {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")
    env.close()
    return results


def eval_cka_to_stage1(model, stage1_model, n_episodes=30):
    """
    CKA between model.hidden1(full_obs) and stage1_v3.hidden1(obs[:7]).
    Matched samples: rollout model's policy, compute both hidden1s on each obs.
    """
    env = Monitor(TabletopReachEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    obs_full, _, _ = collect_rollout(model, env, n_episodes=n_episodes)
    env.close()

    h_model = _hidden1(model, obs_full)
    h_stage1 = _hidden1(stage1_model, obs_full[:, :PROPRIO_DIM])
    cka = _cka_linear(h_model, h_stage1)
    return cka, len(obs_full)


def eval_equivalence_class(model, n_episodes=30, k=5, n_random_pairs=2000):
    """
    Neighbor-consistency ratio: for each sample, find its k-NN in hidden1 space,
    measure mean action-distance among neighbors vs random pairs. Lower = tighter
    equivalence classes (neighbors behave more similarly than chance).
    """
    env = Monitor(TabletopReachEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    obs, actions, _ = collect_rollout(model, env, n_episodes=n_episodes)
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


def main():
    v5_path = str(RESULTS_DIR / "followon_v5_checkpoint")
    v3_path = str(RESULTS_DIR / "followon_v3_checkpoint")
    stage1_path = str(RESULTS_DIR / "stage1_v3_checkpoint")

    print("=" * 60)
    print("v5 evaluation (consistency loss)")
    print("=" * 60)

    v5 = ConsistencySAC.load(v5_path)
    v3 = SAC.load(v3_path)
    stage1 = SAC.load(stage1_path)

    print("\n--- Task success ---")
    print("v5:")
    v5_success = eval_success(v5)
    print("v3 (reference):")
    v3_success = eval_success(v3)

    print("\n--- CKA(hidden1, stage1_v3.hidden1) ---")
    print("  (higher = vision stays on proprio's manifold)")
    v5_cka, n_v5 = eval_cka_to_stage1(v5, stage1)
    print(f"  v5: CKA = {v5_cka:.4f} (n={n_v5})")
    v3_cka, n_v3 = eval_cka_to_stage1(v3, stage1)
    print(f"  v3: CKA = {v3_cka:.4f} (n={n_v3})")

    print("\n--- Equivalence-class neighbor-consistency ratio ---")
    print("  (lower = tighter classes; neighbors behave more similarly than chance)")
    v5_eq = eval_equivalence_class(v5)
    print(f"  v5: ratio = {v5_eq['ratio']:.3f} "
          f"(neighbor={v5_eq['mean_neighbor_action_dist']:.3f}, "
          f"random={v5_eq['mean_random_action_dist']:.3f}, n={v5_eq['n_samples']})")
    v3_eq = eval_equivalence_class(v3)
    print(f"  v3: ratio = {v3_eq['ratio']:.3f} "
          f"(neighbor={v3_eq['mean_neighbor_action_dist']:.3f}, "
          f"random={v3_eq['mean_random_action_dist']:.3f}, n={v3_eq['n_samples']})")

    print("\n--- Summary ---")
    print(f"  v5 success: clean={v5_success[0.0]}/20, 100%noise={v5_success[1.0]}/20")
    print(f"  v3 success: clean={v3_success[0.0]}/20, 100%noise={v3_success[1.0]}/20")
    print(f"  CKA to stage1:   v5={v5_cka:.3f}  vs  v3={v3_cka:.3f}")
    print(f"  Neighbor ratio:  v5={v5_eq['ratio']:.3f}  vs  v3={v3_eq['ratio']:.3f}")


if __name__ == "__main__":
    main()
