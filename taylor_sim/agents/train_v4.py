"""
v4: precision follow-on.

Same frozen-weight architecture as v3, but the target ball is half its original
size (0.01m radius vs. 0.02m). The agent still gets the dense distance-to-target
reward, so learning is tractable — but contact now requires the fingertip to
land within a much tighter region. Expected: vision must sharpen its target
estimate to succeed.

Re-uses stage1_v3_checkpoint as the frozen proprio base (Stage 1 doesn't see
the target and is target-size-agnostic).
"""

import pathlib
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from taylor_sim.envs import TabletopReachEnv
from taylor_sim.envs.flatten_wrapper import FlattenVisionWrapper
from taylor_sim.agents.train_staged import (
    MetricsCallback,
    _transfer_proprio_weights,
    _evaluate,
)

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROPRIO_DIM = 7
TARGET_SIZE = 0.01  # half the default 0.02


def train_followon_v4(stage1_path=None, total_timesteps=200_000, seed=42):
    print("=" * 60)
    print(f"FOLLOW-ON v4: precision touch (target_size={TARGET_SIZE})")
    print("=" * 60)

    env = Monitor(TabletopReachEnv(
        vision=True, target_object=0,
        hide_target_offset=True, target_size=TARGET_SIZE,
    ))
    eval_env = Monitor(TabletopReachEnv(
        vision=True, target_object=0,
        hide_target_offset=True, target_size=TARGET_SIZE,
    ))
    env = FlattenVisionWrapper(env)
    eval_env = FlattenVisionWrapper(eval_env)

    if stage1_path is None:
        stage1_path = str(RESULTS_DIR / "stage1_v3_checkpoint")

    obs_dim = env.observation_space.shape[0]
    print(f"Obs dim: {obs_dim} (proprio={PROPRIO_DIM}, pixels={obs_dim - PROPRIO_DIM})")

    model = SAC(
        "MlpPolicy", env,
        learning_rate=1e-4,
        buffer_size=50_000, batch_size=256, tau=0.005, gamma=0.99,
        train_freq=1, gradient_steps=1, learning_starts=500,
        verbose=1, seed=seed, device="cpu",
    )

    stage1_model = SAC.load(stage1_path)
    _transfer_proprio_weights(stage1_model, model, proprio_dim=PROPRIO_DIM)

    candidates = [
        (name, p) for name, p in model.actor.named_parameters()
        if p.dim() == 2 and p.shape[1] == obs_dim
    ]
    assert len(candidates) == 1
    first_name, first_layer_weight = candidates[0]
    print(f"Actor first-layer weight: {first_name} {tuple(first_layer_weight.shape)}")

    for name, p in model.actor.named_parameters():
        if p is first_layer_weight:
            p.requires_grad = True
        else:
            p.requires_grad = False

    proprio_cols_snapshot = first_layer_weight.data[:, :PROPRIO_DIM].clone()
    mask = torch.ones_like(first_layer_weight)
    mask[:, :PROPRIO_DIM] = 0.0
    first_layer_weight.register_hook(lambda grad: grad * mask)

    trainable = sum(p.numel() for p in model.actor.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.actor.parameters())
    print(f"Actor trainable params: {trainable}/{total}")

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "followon_v4_best"),
        log_path=str(RESULTS_DIR / "followon_v4_logs"),
        eval_freq=5000, n_eval_episodes=10, deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    drift = (first_layer_weight.data[:, :PROPRIO_DIM] - proprio_cols_snapshot).abs().max().item()
    print(f"Proprio-column drift: {drift:.2e} (expected ~0)")
    assert drift < 1e-8

    checkpoint_path = str(RESULTS_DIR / "followon_v4_checkpoint")
    model.save(checkpoint_path)
    print(f"\nFollow-on v4 checkpoint saved to {checkpoint_path}")

    successes = _evaluate(model, env, n_episodes=20)
    print(f"Follow-on v4 success rate: {successes}/20 ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


if __name__ == "__main__":
    train_followon_v4(total_timesteps=200_000)
