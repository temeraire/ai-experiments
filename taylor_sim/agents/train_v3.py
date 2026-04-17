"""
v3: the blind-proprio experiment.

Stage 1 is trained with hide_target_offset=True — the proprio observation drops
the 3-dim fingertip-to-target offset, so the agent can only feel its way to the
target via joint state + the binary touch bit. Expected outcome: low success
rate, long episodes, essentially blind groping.

The v3 follow-on then layers vision onto this blind Stage 1 with the usual
frozen-actor architecture. Because proprio no longer solves the task, vision
must do the target-localization work — this is where mutual confirmation
becomes structurally necessary rather than optional.

Proprio dim: 7 (was 10 in v1/v2).
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

PROPRIO_DIM_V3 = 7


def train_stage1_v3(total_timesteps=200_000, seed=42):
    print("=" * 60)
    print("STAGE 1 v3: blind proprio (no target offset)")
    print("=" * 60)

    env = Monitor(TabletopReachEnv(vision=False, target_object=0, hide_target_offset=True))
    eval_env = Monitor(TabletopReachEnv(vision=False, target_object=0, hide_target_offset=True))

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
        best_model_save_path=str(RESULTS_DIR / "stage1_v3_best"),
        log_path=str(RESULTS_DIR / "stage1_v3_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    checkpoint_path = str(RESULTS_DIR / "stage1_v3_checkpoint")
    model.save(checkpoint_path)
    print(f"\nStage 1 v3 checkpoint saved to {checkpoint_path}")

    successes = _evaluate(model, env, n_episodes=20)
    print(f"Stage 1 v3 success rate: {successes}/20 ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


def train_followon_v3(stage1_path=None, total_timesteps=200_000, seed=42):
    print("=" * 60)
    print("FOLLOW-ON v3: vision required (proprio is blind)")
    print("=" * 60)

    env = Monitor(TabletopReachEnv(vision=True, target_object=0, hide_target_offset=True))
    eval_env = Monitor(TabletopReachEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    eval_env = FlattenVisionWrapper(eval_env)

    if stage1_path is None:
        stage1_path = str(RESULTS_DIR / "stage1_v3_checkpoint")

    obs_dim = env.observation_space.shape[0]
    print(f"Obs dim: {obs_dim} (proprio={PROPRIO_DIM_V3}, pixels={obs_dim - PROPRIO_DIM_V3})")

    model = SAC(
        "MlpPolicy",
        env,
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
    _transfer_proprio_weights(stage1_model, model, proprio_dim=PROPRIO_DIM_V3)

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

    proprio_cols_snapshot = first_layer_weight.data[:, :PROPRIO_DIM_V3].clone()

    mask = torch.ones_like(first_layer_weight)
    mask[:, :PROPRIO_DIM_V3] = 0.0
    first_layer_weight.register_hook(lambda grad: grad * mask)

    trainable = sum(p.numel() for p in model.actor.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.actor.parameters())
    print(f"Actor trainable params: {trainable}/{total}")

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "followon_v3_best"),
        log_path=str(RESULTS_DIR / "followon_v3_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    drift = (first_layer_weight.data[:, :PROPRIO_DIM_V3] - proprio_cols_snapshot).abs().max().item()
    print(f"Proprio-column drift after training: {drift:.2e} (expected ~0)")
    assert drift < 1e-8, "Proprio columns drifted — freeze failed."

    checkpoint_path = str(RESULTS_DIR / "followon_v3_checkpoint")
    model.save(checkpoint_path)
    print(f"\nFollow-on v3 checkpoint saved to {checkpoint_path}")

    successes = _evaluate(model, env, n_episodes=20)
    print(f"Follow-on v3 success rate: {successes}/20 ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "followon", "both"], default="both")
    parser.add_argument("--stage1-steps", type=int, default=200_000)
    parser.add_argument("--followon-steps", type=int, default=200_000)
    args = parser.parse_args()

    if args.stage in ("stage1", "both"):
        train_stage1_v3(total_timesteps=args.stage1_steps)
    if args.stage in ("followon", "both"):
        train_followon_v3(total_timesteps=args.followon_steps)
