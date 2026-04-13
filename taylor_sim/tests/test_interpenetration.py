"""
Non-tautological interpenetration tests.

These tests ask whether vision and proprioception are genuinely entangled
in the staged agent's representations — not just whether it retained a
fallback proprioception skill from Stage 1.

Test A: Representation drift — Did the proprioceptive hidden activations
        CHANGE when vision was added in Stage 2? If they didn't change,
        there's no interpenetration.

Test B: Partial vision boost — Does the staged agent with degraded vision
        outperform the pure Stage 1 (proprio-only) agent? If interpenetration
        is real, even noisy vision should help.

Test C: Entanglement probe — When we zero out vision inputs, do the
        proprioceptive pathway activations change compared to normal vision?
        If they do, the modalities are genuinely entangled in the network.
"""

import pathlib
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from taylor_sim.envs import TabletopReachEnv
from taylor_sim.envs.flatten_wrapper import FlattenVisionWrapper

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
PROPRIO_DIM = 10


def _get_layer_activations(model, obs):
    """Get activations from each hidden layer of the actor network."""
    actor = model.actor
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        x = actor.features_extractor(obs_tensor)
        activations = [x.numpy().flatten()]
        for layer in actor.latent_pi:
            x = layer(x)
            activations.append(x.numpy().flatten())
    return activations


def _collect_proprio_activations(model, env, n_episodes=20, vision_model=False):
    """
    Run episodes and collect hidden activations.
    If vision_model, the env should be FlattenVisionWrapper'd.
    Returns: list of (activations_per_layer, proprio_obs) tuples.
    """
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        for step in range(200):
            acts = _get_layer_activations(model, obs)
            # Extract just the proprio portion of the observation
            if vision_model:
                proprio_obs = obs[:PROPRIO_DIM].copy()
            else:
                proprio_obs = obs.copy()
            results.append((acts, proprio_obs))
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return results


def _cka_linear(X, Y):
    """
    Centered Kernel Alignment (linear) between two activation matrices.
    X, Y: (n_samples, n_features) arrays.
    Returns scalar in [0, 1] — 1 means identical representations.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def test_representation_drift(n_episodes=20):
    """
    Test A: Did Stage 2 training change the proprioceptive representations?

    Compare hidden layer activations of Stage 1 vs Stage 2 models on
    identical proprioceptive inputs (vision zeroed out for Stage 2).
    If CKA is high, the proprio representations didn't change — no
    interpenetration, just a bolted-on vision pathway. If CKA is low,
    the representations are genuinely different.
    """
    print("\n" + "=" * 60)
    print("TEST A: Representation drift (Stage 1 vs Stage 2)")
    print("=" * 60)

    stage1_path = str(RESULTS_DIR / "stage1_checkpoint")
    stage2_path = str(RESULTS_DIR / "stage2_checkpoint")

    stage1_model = SAC.load(stage1_path)
    stage2_model = SAC.load(stage2_path)

    # Run Stage 1 model on proprio-only env
    env1 = Monitor(TabletopReachEnv(vision=False, target_object=0))
    s1_data = _collect_proprio_activations(stage1_model, env1, n_episodes)
    env1.close()

    # Run Stage 2 model on vision env but with vision zeroed out
    env2 = Monitor(TabletopReachEnv(vision=True, target_object=0))
    env2 = FlattenVisionWrapper(env2)
    s2_data = []
    for ep in range(n_episodes):
        obs, _ = env2.reset(seed=ep)
        for step in range(200):
            # Zero out vision — only proprio input active
            obs_proprio_only = obs.copy()
            obs_proprio_only[PROPRIO_DIM:] = 0.0
            acts = _get_layer_activations(stage2_model, obs_proprio_only)
            s2_data.append((acts, obs[:PROPRIO_DIM].copy()))
            action, _ = stage2_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env2.step(action)
            if terminated or truncated:
                break
    env2.close()

    # Compare activations at each hidden layer using CKA
    n_samples = min(len(s1_data), len(s2_data))
    n_layers = len(s1_data[0][0])

    print(f"  Comparing {n_samples} activation samples across {n_layers} layers")
    print(f"  (CKA=1.0 means identical representations, lower = more drift)")
    print()

    cka_scores = []
    for layer_idx in range(n_layers):
        s1_acts = np.array([s1_data[i][0][layer_idx] for i in range(n_samples)])
        s2_acts = np.array([s2_data[i][0][layer_idx] for i in range(n_samples)])
        cka = _cka_linear(s1_acts, s2_acts)
        cka_scores.append(cka)
        label = "input" if layer_idx == 0 else f"hidden {layer_idx}"
        print(f"  Layer {layer_idx} ({label}): CKA = {cka:.4f}")

    mean_hidden_cka = np.mean(cka_scores[1:])  # exclude input layer
    print(f"\n  Mean hidden-layer CKA: {mean_hidden_cka:.4f}")

    if mean_hidden_cka > 0.9:
        print("  --> Representations barely changed. Weak interpenetration.")
    elif mean_hidden_cka > 0.7:
        print("  --> Moderate representation drift. Some interpenetration.")
    else:
        print("  --> Substantial drift. Strong evidence of interpenetration.")

    return {"cka_scores": cka_scores, "mean_hidden_cka": mean_hidden_cka}


def test_partial_vision_boost(n_episodes=30):
    """
    Test B: Does degraded vision still help beyond pure proprioception?

    Compare:
    - Stage 1 agent (pure proprio, no vision at all)
    - Stage 2 agent with varying levels of vision noise

    If interpenetration is real, even noisy vision should boost performance
    beyond what pure proprioception achieves, because vision is integrated
    into the proprioceptive pathways rather than being a separate module.
    """
    print("\n" + "=" * 60)
    print("TEST B: Partial vision boost (Stage 2 + noise vs Stage 1)")
    print("=" * 60)

    stage1_model = SAC.load(str(RESULTS_DIR / "stage1_checkpoint"))
    stage2_model = SAC.load(str(RESULTS_DIR / "stage2_checkpoint"))

    # Stage 1 baseline (pure proprioception)
    env1 = Monitor(TabletopReachEnv(vision=False, target_object=0))
    s1_successes = 0
    s1_rewards = []
    for ep in range(n_episodes):
        obs, _ = env1.reset(seed=ep)
        ep_reward = 0
        for step in range(200):
            action, _ = stage1_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env1.step(action)
            ep_reward += reward
            if terminated or truncated:
                if terminated and info.get("touching", False):
                    s1_successes += 1
                break
        s1_rewards.append(ep_reward)
    env1.close()

    print(f"  Stage 1 (pure proprio): {s1_successes}/{n_episodes} success, "
          f"mean reward = {np.mean(s1_rewards):.2f}")

    # Stage 2 with varying noise levels
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    s2_results = {}

    for noise in noise_levels:
        env2 = Monitor(TabletopReachEnv(vision=True, target_object=0))
        env2 = FlattenVisionWrapper(env2)
        successes = 0
        rewards = []
        for ep in range(n_episodes):
            obs, _ = env2.reset(seed=ep)  # same seeds as Stage 1
            ep_reward = 0
            for step in range(200):
                noisy_obs = obs.copy()
                if noise > 0:
                    vision_part = noisy_obs[PROPRIO_DIM:]
                    noise_vec = np.random.RandomState(ep * 200 + step).randn(
                        *vision_part.shape
                    ).astype(np.float32) * noise
                    noisy_obs[PROPRIO_DIM:] = vision_part * (1 - noise) + noise_vec
                action, _ = stage2_model.predict(noisy_obs, deterministic=True)
                obs, reward, terminated, truncated, info = env2.step(action)
                ep_reward += reward
                if terminated or truncated:
                    if terminated and info.get("touching", False):
                        successes += 1
                    break
            rewards.append(ep_reward)
        env2.close()

        s2_results[noise] = {
            "successes": successes,
            "mean_reward": np.mean(rewards),
        }
        delta = np.mean(rewards) - np.mean(s1_rewards)
        marker = "+" if delta > 0 else ""
        print(f"  Stage 2 (noise={noise:.1f}): {successes}/{n_episodes} success, "
              f"mean reward = {np.mean(rewards):.2f} ({marker}{delta:.2f} vs Stage 1)")

    # The key question: at what noise level does Stage 2 drop BELOW Stage 1?
    s1_mean = np.mean(s1_rewards)
    crossover = None
    for noise in noise_levels:
        if s2_results[noise]["mean_reward"] < s1_mean:
            crossover = noise
            break

    print()
    if crossover is None or crossover > 0.6:
        print(f"  --> Stage 2 outperforms Stage 1 up to high noise. "
              f"Vision is genuinely helping via interpenetration.")
    elif crossover is not None:
        print(f"  --> Stage 2 drops below Stage 1 at noise={crossover:.1f}. "
              f"Partial interpenetration — noisy vision eventually hurts.")
    else:
        print(f"  --> Stage 2 never outperforms Stage 1. No evidence of useful interpenetration.")

    return {"stage1_mean": s1_mean, "stage2_results": s2_results, "crossover": crossover}


def test_entanglement_probe(n_episodes=20):
    """
    Test C: Are the modalities genuinely entangled in the network?

    For each vision-capable agent, compare hidden activations when:
    (a) Normal observation (proprio + vision)
    (b) Same observation but vision zeroed out

    If the HIDDEN layer activations change substantially, the modalities
    are entangled — proprioceptive processing depends on vision input
    even in intermediate layers. If only the input layer changes,
    vision is just an additive input that doesn't reshape the proprioceptive
    representations.
    """
    print("\n" + "=" * 60)
    print("TEST C: Entanglement probe (vision vs no-vision activations)")
    print("=" * 60)

    agents = {}
    for name, path in [
        ("staged", str(RESULTS_DIR / "stage2_checkpoint")),
        ("allatonce", str(RESULTS_DIR / "allatonce_checkpoint")),
        ("fusion", str(RESULTS_DIR / "fusion_checkpoint")),
    ]:
        if pathlib.Path(path + ".zip").exists():
            agents[name] = SAC.load(path)

    results = {}
    for name, model in agents.items():
        env = Monitor(TabletopReachEnv(vision=True, target_object=0))
        env = FlattenVisionWrapper(env)

        normal_acts = []  # activations with full observation
        zeroed_acts = []  # activations with vision zeroed

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            for step in range(50):  # first 50 steps of each episode
                # Normal activations
                acts_normal = _get_layer_activations(model, obs)
                normal_acts.append([a.copy() for a in acts_normal])

                # Zero-vision activations (same proprio, no vision)
                obs_zeroed = obs.copy()
                obs_zeroed[PROPRIO_DIM:] = 0.0
                acts_zeroed = _get_layer_activations(model, obs_zeroed)
                zeroed_acts.append([a.copy() for a in acts_zeroed])

                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        env.close()

        n_samples = len(normal_acts)
        n_layers = len(normal_acts[0])

        print(f"\n  {name} ({n_samples} samples):")
        layer_diffs = []
        for layer_idx in range(n_layers):
            norm_mat = np.array([normal_acts[i][layer_idx] for i in range(n_samples)])
            zero_mat = np.array([zeroed_acts[i][layer_idx] for i in range(n_samples)])

            # Relative activation change
            diff = np.linalg.norm(norm_mat - zero_mat, axis=1)
            baseline = np.linalg.norm(norm_mat, axis=1) + 1e-10
            relative_change = np.mean(diff / baseline)
            layer_diffs.append(relative_change)

            label = "input" if layer_idx == 0 else f"hidden {layer_idx}"
            print(f"    Layer {layer_idx} ({label}): "
                  f"relative activation change = {relative_change:.4f}")

        # Key metric: do deeper layers show entanglement?
        if len(layer_diffs) > 2:
            deep_change = np.mean(layer_diffs[2:])
            print(f"    Deep-layer mean change: {deep_change:.4f}")
        results[name] = layer_diffs

    # Compare across agents
    print("\n  Comparison (higher deep-layer change = more entanglement):")
    for name, diffs in results.items():
        deep = np.mean(diffs[2:]) if len(diffs) > 2 else np.mean(diffs[1:])
        print(f"    {name}: {deep:.4f}")

    return results


def run_interpenetration_tests():
    """Run all non-tautological interpenetration tests."""
    print("=" * 60)
    print("NON-TAUTOLOGICAL INTERPENETRATION TESTS")
    print("=" * 60)

    results = {}
    results["drift"] = test_representation_drift()
    results["boost"] = test_partial_vision_boost()
    results["entanglement"] = test_entanglement_probe()

    print("\n" + "=" * 60)
    print("INTERPENETRATION SUMMARY")
    print("=" * 60)

    drift_cka = results["drift"]["mean_hidden_cka"]
    crossover = results["boost"]["crossover"]
    entanglement = results["entanglement"]

    print(f"\n  A. Representation drift (Stage 1→2): CKA = {drift_cka:.4f}")
    if drift_cka < 0.7:
        print("     Proprioceptive representations substantially changed. ✓")
    elif drift_cka < 0.9:
        print("     Moderate change in proprioceptive representations.")
    else:
        print("     Representations barely changed — weak interpenetration.")

    print(f"\n  B. Partial vision crossover: {crossover}")
    if crossover is None or crossover > 0.6:
        print("     Degraded vision still helps — integrated, not bolted on. ✓")
    else:
        print("     Noisy vision hurts quickly — modalities may be separate.")

    staged_deep = np.mean(entanglement.get("staged", [0, 0, 0])[2:])
    aao_deep = np.mean(entanglement.get("allatonce", [0, 0, 0])[2:])
    print(f"\n  C. Deep-layer entanglement:")
    print(f"     Staged: {staged_deep:.4f}, All-at-once: {aao_deep:.4f}")
    if staged_deep > aao_deep:
        print("     Staged agent has more entangled representations. ✓")
    else:
        print("     All-at-once is equally or more entangled.")

    return results


if __name__ == "__main__":
    run_interpenetration_tests()
