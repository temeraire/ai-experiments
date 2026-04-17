"""
v8: creature on a platform — survival-grounded perception.

Stage 1: blind crawling (proprio only). Learn to move around the platform,
find the target by feel, and avoid falling off the edge. Gravity is the
pencil tap.

Stage 2: vision added (head camera). Does the creature learn to look before
moving? Does proprio primacy emerge from consequence rather than architecture?

Stage 3: spectacles (future).
"""

import pathlib
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from taylor_sim.envs.platform_creature_env import PlatformCreatureEnv
from taylor_sim.agents.train_staged import MetricsCallback, _evaluate
from taylor_sim.agents.train_v5 import ConsistencySAC

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROPRIO_DIM_V8 = 29
N_ENVS_FOLLOWON = 8


def _make_followon_env():
    return Monitor(PlatformCreatureEnv(vision=True))


def train_stage1_v8(total_timesteps=500_000, seed=42, checkpoint_interval=50_000):
    print("=" * 60)
    print("STAGE 1 v8: blind proprio on platform (no vision)")
    print("=" * 60)

    env = Monitor(PlatformCreatureEnv(vision=False))
    eval_env = Monitor(PlatformCreatureEnv(vision=False))

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=5000,
        verbose=1,
        seed=seed,
        device="cpu",
    )

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "stage1_v8_best"),
        log_path=str(RESULTS_DIR / "stage1_v8_logs"),
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=str(RESULTS_DIR / "stage1_v8_checkpoints"),
        name_prefix="stage1_v8",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([metrics_cb, eval_cb, ckpt_cb]),
    )

    final_path = str(RESULTS_DIR / "stage1_v8_checkpoint")
    model.save(final_path)
    print(f"\nStage 1 v8 saved to {final_path}")

    # Evaluate
    successes = 0
    falls = 0
    for i in range(20):
        obs, _ = env.reset(seed=i + 1000)
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        if info.get("touched"):
            successes += 1
        if info.get("fell"):
            falls += 1

    print(f"Stage 1 v8: {successes}/20 touches, {falls}/20 falls")

    env.close()
    eval_env.close()
    return model, metrics_cb


def train_followon_v8(
    stage1_path=None,
    total_timesteps=1_000_000,
    seed=42,
    lambda_consistency=0.1,
    freeze_proprio=False,
    checkpoint_interval=50_000,
):
    freeze_label = "frozen" if freeze_proprio else "unfrozen"
    print("=" * 60)
    print(f"FOLLOW-ON v8: vision + platform ({freeze_label} proprio, lambda={lambda_consistency})")
    print("=" * 60)

    env = SubprocVecEnv(
        [_make_followon_env for _ in range(N_ENVS_FOLLOWON)],
        start_method="spawn",
    )
    eval_env = Monitor(PlatformCreatureEnv(vision=True))

    if stage1_path is None:
        stage1_path = str(RESULTS_DIR / "stage1_v8_checkpoint")

    obs_dim = env.observation_space.shape[0]
    print(f"Obs dim: {obs_dim} (proprio={PROPRIO_DIM_V8}, pixels={obs_dim - PROPRIO_DIM_V8})")

    model = ConsistencySAC(
        "MlpPolicy",
        env,
        lambda_consistency=lambda_consistency,
        proprio_dim=PROPRIO_DIM_V8,
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=N_ENVS_FOLLOWON,
        learning_starts=5000,
        verbose=1,
        seed=seed,
        device="cpu",
    )

    # Transfer stage 1 weights
    stage1_model = SAC.load(stage1_path)
    from taylor_sim.agents.train_staged import _transfer_proprio_weights
    _transfer_proprio_weights(stage1_model, model, proprio_dim=PROPRIO_DIM_V8)

    if freeze_proprio:
        # Find and freeze actor weights, only allow pixel columns to train
        candidates = [
            (name, p)
            for name, p in model.actor.named_parameters()
            if p.dim() == 2 and p.shape[1] == obs_dim
        ]
        assert len(candidates) == 1
        first_name, first_layer_weight = candidates[0]
        print(f"Actor first-layer: {first_name} {tuple(first_layer_weight.shape)}")

        for name, p in model.actor.named_parameters():
            if p is first_layer_weight:
                p.requires_grad = True
            else:
                p.requires_grad = False

        proprio_cols_snapshot = first_layer_weight.data[:, :PROPRIO_DIM_V8].clone()
        mask = torch.ones_like(first_layer_weight)
        mask[:, :PROPRIO_DIM_V8] = 0.0
        first_layer_weight.register_hook(lambda grad: grad * mask)
    else:
        proprio_cols_snapshot = None
        print("Proprio weights UNFROZEN — letting the world establish hierarchy")

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "followon_v8_best"),
        log_path=str(RESULTS_DIR / "followon_v8_logs"),
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=str(RESULTS_DIR / "followon_v8_checkpoints"),
        name_prefix="followon_v8",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([metrics_cb, eval_cb, ckpt_cb]),
    )

    if freeze_proprio and proprio_cols_snapshot is not None:
        drift = (first_layer_weight.data[:, :PROPRIO_DIM_V8] - proprio_cols_snapshot).abs().max().item()
        print(f"Proprio-column drift: {drift:.2e}")

    final_path = str(RESULTS_DIR / "followon_v8_checkpoint")
    model.save(final_path)
    print(f"\nFollow-on v8 saved to {final_path}")

    # Evaluate on a fresh single env (VecEnv has a different reset API)
    final_eval_env = PlatformCreatureEnv(vision=True)
    successes = 0
    falls = 0
    for i in range(20):
        obs, _ = final_eval_env.reset(seed=i + 1000)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = final_eval_env.step(action)
            done = terminated or truncated
        if info.get("touched"):
            successes += 1
        if info.get("fell"):
            falls += 1

    print(f"Follow-on v8: {successes}/20 touches, {falls}/20 falls")

    env.close()
    eval_env.close()
    final_eval_env.close()
    return model, metrics_cb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "followon", "both"], default="both")
    parser.add_argument("--stage1-steps", type=int, default=500_000)
    parser.add_argument("--followon-steps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--lambda-consistency", type=float, default=0.1)
    parser.add_argument("--freeze-proprio", action="store_true",
                        help="Freeze proprio weights in follow-on (default: unfrozen)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stage in ("stage1", "both"):
        train_stage1_v8(
            total_timesteps=args.stage1_steps,
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
        )
    if args.stage in ("followon", "both"):
        train_followon_v8(
            total_timesteps=args.followon_steps,
            lambda_consistency=args.lambda_consistency,
            freeze_proprio=args.freeze_proprio,
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
        )
