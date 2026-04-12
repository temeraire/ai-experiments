"""
Stage 1: Train proprioception-only SAC agent to reach objects by touch.
Stage 2: Expand to include vision, continue training from Stage 1 checkpoint.

This is the "Taylor" agent — developmental staging with interpenetration.
"""

import os
import pathlib
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from taylor_sim.envs import TabletopReachEnv

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class MetricsCallback(BaseCallback):
    """Log distance and touch metrics during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.distances = []
        self.touches = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "distance" in info:
                self.distances.append(info["distance"])
            if "touching" in info:
                self.touches.append(1.0 if info["touching"] else 0.0)
        return True


def train_stage1(total_timesteps=50_000, seed=42):
    """Stage 1: Proprioception only — learn to reach by touch."""
    print("=" * 60)
    print("STAGE 1: Proprioception-only reaching")
    print("=" * 60)

    env = Monitor(TabletopReachEnv(vision=False, target_object=0))
    eval_env = Monitor(TabletopReachEnv(vision=False, target_object=0))

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1000,
        verbose=1,
        seed=seed,
        device="cpu",  # MPS can be flaky with SB3
    )

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "stage1_best"),
        log_path=str(RESULTS_DIR / "stage1_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    checkpoint_path = str(RESULTS_DIR / "stage1_checkpoint")
    model.save(checkpoint_path)
    print(f"\nStage 1 checkpoint saved to {checkpoint_path}")

    # Evaluate
    successes = _evaluate(model, env, n_episodes=20)
    print(f"Stage 1 success rate: {successes}/{20} ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


def train_stage2(stage1_path=None, total_timesteps=50_000, seed=42):
    """
    Stage 2: Add vision to the trained proprioception agent.

    Key design choice: we DON'T use a separate vision encoder. The visual
    features get concatenated into the same MLP, forcing interpenetration —
    the network must integrate vision through the same pathways it already
    uses for proprioception.

    Since SB3's SAC with MlpPolicy can't natively handle Dict obs, we
    flatten the observation (proprio + downsampled vision) into a single vector.
    This is actually closer to Taylor's interpenetration idea — no separate
    "vision module."
    """
    print("=" * 60)
    print("STAGE 2: Adding vision (interpenetration)")
    print("=" * 60)

    env = Monitor(TabletopReachEnv(vision=True, target_object=0))
    eval_env = Monitor(TabletopReachEnv(vision=True, target_object=0))

    # Wrap to flatten Dict obs into single vector
    from taylor_sim.envs.flatten_wrapper import FlattenVisionWrapper
    env = FlattenVisionWrapper(env)
    eval_env = FlattenVisionWrapper(eval_env)

    if stage1_path is None:
        stage1_path = str(RESULTS_DIR / "stage1_checkpoint")

    # Load Stage 1 model — we need to create a new one with the larger obs space
    # but transfer the learned weights for the proprioception portion
    obs_dim = env.observation_space.shape[0]
    print(f"Stage 2 observation dim: {obs_dim} (proprio + flattened vision)")

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # lower LR to not destroy Stage 1 knowledge
        buffer_size=50_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=500,
        verbose=1,
        seed=seed,
        device="cpu",
    )

    # Transfer Stage 1 weights into the proprio portion of the new network
    stage1_model = SAC.load(stage1_path)
    _transfer_proprio_weights(stage1_model, model, proprio_dim=10)

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "stage2_best"),
        log_path=str(RESULTS_DIR / "stage2_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    checkpoint_path = str(RESULTS_DIR / "stage2_checkpoint")
    model.save(checkpoint_path)
    print(f"\nStage 2 checkpoint saved to {checkpoint_path}")

    successes = _evaluate(model, env, n_episodes=20)
    print(f"Stage 2 success rate: {successes}/{20} ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


def _transfer_proprio_weights(src_model, dst_model, proprio_dim):
    """
    Transfer weights from Stage 1 (proprio-only) network into the
    proprioception portion of the Stage 2 (larger) network.

    For the first layer: copy weights corresponding to the first `proprio_dim`
    input features, zero out the rest (vision features start blank).
    For subsequent layers: copy directly (same size).
    """
    import torch

    src_actor = dict(src_model.actor.named_parameters())
    dst_actor = dict(dst_model.actor.named_parameters())

    for name, dst_param in dst_actor.items():
        if name in src_actor:
            src_param = src_actor[name]
            if dst_param.shape == src_param.shape:
                dst_param.data.copy_(src_param.data)
            elif len(dst_param.shape) == 2 and dst_param.shape[1] > src_param.shape[1]:
                # First layer: larger input dim. Copy proprio weights, zero vision.
                dst_param.data.zero_()
                dst_param.data[:, :src_param.shape[1]] = src_param.data
            elif len(dst_param.shape) == 1 and dst_param.shape == src_param.shape:
                dst_param.data.copy_(src_param.data)

    # Do the same for critic networks
    for critic_name in ["critic", "critic_target"]:
        src_critic = dict(getattr(src_model, critic_name).named_parameters())
        dst_critic = dict(getattr(dst_model, critic_name).named_parameters())
        for name, dst_param in dst_critic.items():
            if name in src_critic:
                src_param = src_critic[name]
                if dst_param.shape == src_param.shape:
                    dst_param.data.copy_(src_param.data)
                elif len(dst_param.shape) == 2 and dst_param.shape[1] > src_param.shape[1]:
                    dst_param.data.zero_()
                    dst_param.data[:, :src_param.shape[1]] = src_param.data

    print(f"Transferred Stage 1 weights (proprio_dim={proprio_dim}) into Stage 2 network")


def _evaluate(model, env, n_episodes=20):
    """Count successful reaches (terminated by contact)."""
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if terminated and info.get("touching", False):
                successes += 1
    return successes


if __name__ == "__main__":
    print("Training Taylor (staged) agent...\n")
    model1, metrics1 = train_stage1(total_timesteps=50_000)
    model2, metrics2 = train_stage2(total_timesteps=50_000)
    print("\nDone! Staged agent trained.")
