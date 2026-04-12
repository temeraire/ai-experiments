"""
Test battery for comparing Taylor (staged) vs baseline agents.

Test 1: Action-based equivalence — do agents group objects by affordance?
Test 2: Cross-modal transfer — does seeing X activate same state as touching X?
Test 3: Staged vs all-at-once comparison — does developmental order matter?
Test 4: Graceful degradation — remove vision, measure decline curve.
"""

import pathlib
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from taylor_sim.envs import TabletopReachEnv
from taylor_sim.envs.flatten_wrapper import FlattenVisionWrapper

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"


def get_internal_activations(model, obs):
    """Extract activations from the actor network's hidden layers."""
    actor = model.actor
    # Get the MLP extractor
    features_extractor = actor.features_extractor
    latent_pi = actor.latent_pi

    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        features = features_extractor(obs_tensor)

        activations = [features.numpy().flatten()]
        x = features
        for layer in latent_pi:
            x = layer(x)
            activations.append(x.numpy().flatten())

    return activations


def test_action_equivalence(model, n_episodes=10):
    """
    Test 1: Action-based equivalence classes.

    Present novel objects (different shapes) and measure whether the agent
    groups them by affordance (graspability/reachability) rather than appearance.

    We test reaching to each of the 3 object types and compare internal
    representations — if interpenetration works, similar-size objects should
    have similar internal states regardless of shape.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Action-based equivalence classes")
    print("=" * 60)

    results = {}
    for obj_idx in range(3):
        env = Monitor(TabletopReachEnv(vision=True, target_object=obj_idx))
        env = FlattenVisionWrapper(env)

        activations_list = []
        rewards_list = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            for step in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            # Get final activations
            acts = get_internal_activations(model, obs)
            activations_list.append(acts[-1])  # last hidden layer
            rewards_list.append(total_reward)

        results[obj_idx] = {
            "mean_activation": np.mean(activations_list, axis=0),
            "mean_reward": np.mean(rewards_list),
        }
        env.close()
        print(f"  Object {obj_idx}: mean reward = {results[obj_idx]['mean_reward']:.2f}")

    # Compare activation similarity across objects
    for i in range(3):
        for j in range(i + 1, 3):
            cos_sim = np.dot(results[i]["mean_activation"], results[j]["mean_activation"]) / (
                np.linalg.norm(results[i]["mean_activation"]) * np.linalg.norm(results[j]["mean_activation"]) + 1e-8
            )
            print(f"  Activation similarity (obj {i} vs {j}): {cos_sim:.4f}")

    return results


def test_cross_modal_transfer(model, n_episodes=10):
    """
    Test 2: Cross-modal transfer (the big test).

    Compare internal representations when the agent:
    a) Sees an object (vision active, far away)
    b) Touches an object (vision zeroed out, close up)

    If interpenetration works, seeing and touching the same object should
    produce more similar internal states than seeing two different objects.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Cross-modal transfer")
    print("=" * 60)

    visual_acts = {}
    tactile_acts = {}

    for obj_idx in range(3):
        env = Monitor(TabletopReachEnv(vision=True, target_object=obj_idx))
        env = FlattenVisionWrapper(env)

        v_acts = []
        t_acts = []

        for _ in range(n_episodes):
            obs, _ = env.reset()

            # Visual observation: just after reset (far from object)
            v_act = get_internal_activations(model, obs)
            v_acts.append(v_act[-1])

            # Now run to touch the object
            for step in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            # Tactile observation: zero out vision portion, keep proprio
            proprio_dim = 10
            tactile_obs = obs.copy()
            tactile_obs[proprio_dim:] = 0.0  # zero vision
            t_act = get_internal_activations(model, tactile_obs)
            t_acts.append(t_act[-1])

        visual_acts[obj_idx] = np.mean(v_acts, axis=0)
        tactile_acts[obj_idx] = np.mean(t_acts, axis=0)
        env.close()

    # Key metric: does see(obj_X) correlate with touch(obj_X) more than
    # see(obj_X) with touch(obj_Y)?
    print("  Cross-modal similarity (see vs touch same object):")
    same_sims = []
    for i in range(3):
        sim = np.dot(visual_acts[i], tactile_acts[i]) / (
            np.linalg.norm(visual_acts[i]) * np.linalg.norm(tactile_acts[i]) + 1e-8
        )
        same_sims.append(sim)
        print(f"    Object {i}: {sim:.4f}")

    print("  Cross-modal similarity (see vs touch different object):")
    diff_sims = []
    for i in range(3):
        for j in range(3):
            if i != j:
                sim = np.dot(visual_acts[i], tactile_acts[j]) / (
                    np.linalg.norm(visual_acts[i]) * np.linalg.norm(tactile_acts[j]) + 1e-8
                )
                diff_sims.append(sim)
                print(f"    See {i} vs Touch {j}: {sim:.4f}")

    mean_same = np.mean(same_sims)
    mean_diff = np.mean(diff_sims)
    print(f"\n  Mean same-object cross-modal similarity: {mean_same:.4f}")
    print(f"  Mean diff-object cross-modal similarity: {mean_diff:.4f}")
    print(f"  Difference (higher = more interpenetration): {mean_same - mean_diff:.4f}")

    return {"same": same_sims, "diff": diff_sims}


def test_graceful_degradation(model, n_episodes=20):
    """
    Test 4: Graceful degradation.

    Progressively corrupt the vision input and measure performance decline.
    Interpenetrated representations should degrade gradually (vision is
    woven into proprioception). Fusion-based should cliff (lose a whole module).
    """
    print("\n" + "=" * 60)
    print("TEST 4: Graceful degradation (vision corruption)")
    print("=" * 60)

    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}

    for noise in noise_levels:
        env = Monitor(TabletopReachEnv(vision=True, target_object=0))
        env = FlattenVisionWrapper(env)

        successes = 0
        total_reward = 0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0
            for step in range(200):
                # Add noise to vision portion only
                noisy_obs = obs.copy()
                proprio_dim = 10
                if noise > 0:
                    vision_part = noisy_obs[proprio_dim:]
                    noise_vec = np.random.randn(*vision_part.shape).astype(np.float32) * noise
                    noisy_obs[proprio_dim:] = vision_part * (1 - noise) + noise_vec

                action, _ = model.predict(noisy_obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    if terminated and info.get("distance", 1.0) < 0.05:
                        successes += 1
                    break

            total_reward += ep_reward

        results[noise] = {
            "success_rate": successes / n_episodes,
            "mean_reward": total_reward / n_episodes,
        }
        env.close()
        print(f"  Noise {noise:.1f}: success={results[noise]['success_rate']:.0%}, reward={results[noise]['mean_reward']:.2f}")

    return results


def run_all_tests(staged_path=None, allatonce_path=None, fusion_path=None):
    """Run the full test battery on all available agents."""
    agents = {}

    if staged_path and pathlib.Path(staged_path + ".zip").exists():
        agents["staged"] = SAC.load(staged_path)
        print("Loaded staged (Taylor) agent")
    if allatonce_path and pathlib.Path(allatonce_path + ".zip").exists():
        agents["allatonce"] = SAC.load(allatonce_path)
        print("Loaded all-at-once agent")
    if fusion_path and pathlib.Path(fusion_path + ".zip").exists():
        agents["fusion"] = SAC.load(fusion_path)
        print("Loaded feature-fusion agent")

    if not agents:
        print("No trained agents found. Train agents first.")
        return

    all_results = {}
    for name, model in agents.items():
        print(f"\n{'#' * 60}")
        print(f"# Testing: {name}")
        print(f"{'#' * 60}")

        results = {}
        results["equivalence"] = test_action_equivalence(model)
        results["cross_modal"] = test_cross_modal_transfer(model)
        results["degradation"] = test_graceful_degradation(model)
        all_results[name] = results

    # Summary comparison
    if len(agents) > 1:
        print(f"\n{'=' * 60}")
        print("SUMMARY COMPARISON")
        print(f"{'=' * 60}")
        for name, results in all_results.items():
            cm = results["cross_modal"]
            deg = results["degradation"]
            print(f"\n  {name}:")
            print(f"    Cross-modal gap (same - diff): {np.mean(cm['same']) - np.mean(cm['diff']):.4f}")
            print(f"    Degradation (no noise vs full noise reward):")
            print(f"      Clean: {deg[0.0]['mean_reward']:.2f}")
            print(f"      Full noise: {deg[1.0]['mean_reward']:.2f}")
            print(f"      Drop: {deg[0.0]['mean_reward'] - deg[1.0]['mean_reward']:.2f}")

    return all_results


if __name__ == "__main__":
    results = run_all_tests(
        staged_path=str(RESULTS_DIR / "stage2_checkpoint"),
        allatonce_path=str(RESULTS_DIR / "allatonce_checkpoint"),
        fusion_path=str(RESULTS_DIR / "fusion_checkpoint"),
    )
