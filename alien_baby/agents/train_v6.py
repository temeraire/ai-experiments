"""
v6: staged training on the gaze env.

Stage 1: 9-dim proprio (3 joint pos + 3 vel + 1 touch + 2 head angles), no
vision. Agent learns to reach by touch. Head actions 3-4 exist but don't
affect the reward, so head may flail during Stage 1.

Follow-on (v6): vision turns on, sourced from the head camera. Uses the v5
ConsistencySAC — frozen proprio columns + MSE(h_full, h_blind) consistency
loss. The agent must learn to point the head at the target to get useful
visual signal, since the head camera has a limited FOV.

Proprio dim: 9
Action dim: 5 (3 arm torque + 2 head position)
"""

import pathlib
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from alien_baby.envs import TabletopGazeEnv
from alien_baby.envs.flatten_wrapper import FlattenVisionWrapper
from alien_baby.agents.train_staged import (
    MetricsCallback,
    _transfer_proprio_weights,
    _evaluate,
)
from alien_baby.agents.train_v5 import ConsistencySAC

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROPRIO_DIM_V6 = 9


def train_stage1_v6(total_timesteps=200_000, seed=42):
    print("=" * 60)
    print("STAGE 1 v6: blind proprio + gaze (head flails, no vision)")
    print("=" * 60)

    env = Monitor(TabletopGazeEnv(vision=False, target_object=0, hide_target_offset=True))
    eval_env = Monitor(TabletopGazeEnv(vision=False, target_object=0, hide_target_offset=True))

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
        device="cpu",
    )

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "stage1_v6_best"),
        log_path=str(RESULTS_DIR / "stage1_v6_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    checkpoint_path = str(RESULTS_DIR / "stage1_v6_checkpoint")
    model.save(checkpoint_path)
    print(f"\nStage 1 v6 checkpoint saved to {checkpoint_path}")

    successes = _evaluate(model, env, n_episodes=20)
    print(f"Stage 1 v6 success rate: {successes}/20 ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


def train_followon_v6(
    stage1_path=None,
    total_timesteps=200_000,
    seed=42,
    lambda_consistency=0.1,
):
    print("=" * 60)
    print(f"FOLLOW-ON v6: gaze vision + consistency (lambda={lambda_consistency})")
    print("=" * 60)

    env = Monitor(TabletopGazeEnv(vision=True, target_object=0, hide_target_offset=True))
    eval_env = Monitor(TabletopGazeEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    eval_env = FlattenVisionWrapper(eval_env)

    if stage1_path is None:
        stage1_path = str(RESULTS_DIR / "stage1_v6_checkpoint")

    obs_dim = env.observation_space.shape[0]
    print(f"Obs dim: {obs_dim} (proprio={PROPRIO_DIM_V6}, pixels={obs_dim - PROPRIO_DIM_V6})")

    model = ConsistencySAC(
        "MlpPolicy",
        env,
        lambda_consistency=lambda_consistency,
        proprio_dim=PROPRIO_DIM_V6,
        learning_rate=1e-4,
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

    stage1_model = SAC.load(stage1_path)
    _transfer_proprio_weights(stage1_model, model, proprio_dim=PROPRIO_DIM_V6)

    candidates = [
        (name, p)
        for name, p in model.actor.named_parameters()
        if p.dim() == 2 and p.shape[1] == obs_dim
    ]
    assert len(candidates) == 1, f"Expected 1 actor first-layer weight, found: {candidates}"
    first_name, first_layer_weight = candidates[0]
    print(f"Actor first-layer weight: {first_name} {tuple(first_layer_weight.shape)}")

    for name, p in model.actor.named_parameters():
        if p is first_layer_weight:
            p.requires_grad = True
        else:
            p.requires_grad = False

    proprio_cols_snapshot = first_layer_weight.data[:, :PROPRIO_DIM_V6].clone()

    mask = torch.ones_like(first_layer_weight)
    mask[:, :PROPRIO_DIM_V6] = 0.0
    first_layer_weight.register_hook(lambda grad: grad * mask)

    trainable = sum(p.numel() for p in model.actor.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.actor.parameters())
    print(f"Actor trainable params: {trainable}/{total}")

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "followon_v6_best"),
        log_path=str(RESULTS_DIR / "followon_v6_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    drift = (first_layer_weight.data[:, :PROPRIO_DIM_V6] - proprio_cols_snapshot).abs().max().item()
    print(f"Proprio-column drift after training: {drift:.2e} (expected ~0)")
    assert drift < 1e-8, "Proprio columns drifted — freeze failed."

    checkpoint_path = str(RESULTS_DIR / "followon_v6_checkpoint")
    model.save(checkpoint_path)
    print(f"\nFollow-on v6 checkpoint saved to {checkpoint_path}")

    successes = _evaluate(model, env, n_episodes=20)
    print(f"Follow-on v6 success rate: {successes}/20 ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "followon", "both"], default="both")
    parser.add_argument("--stage1-steps", type=int, default=200_000)
    parser.add_argument("--followon-steps", type=int, default=200_000)
    parser.add_argument("--lambda-consistency", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stage in ("stage1", "both"):
        train_stage1_v6(total_timesteps=args.stage1_steps, seed=args.seed)
    if args.stage in ("followon", "both"):
        train_followon_v6(
            total_timesteps=args.followon_steps,
            lambda_consistency=args.lambda_consistency,
            seed=args.seed,
        )
