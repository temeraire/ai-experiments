"""
Stage 2 FOLLOW-ON: vision layered onto a frozen Stage 1 actor.

Unlike train_staged.train_stage2 — which lets SAC rewrite the entire actor when
vision is added — this variant freezes every Stage 1-derived weight in the actor.
Only the new pixel-input weights (columns of the first hidden layer corresponding
to pixel inputs) receive gradients.

Consequences:
  - With pixels zeroed, the actor produces exactly Stage 1's action.
  - Vision can only perturb hidden1 activations through the new pixel columns;
    those perturbations then flow through frozen hidden1->hidden2->action weights.
  - Critics train normally over the full observation space.

This implements the "vision must align with proprioception, not overwrite it"
reading of the interpenetration idea principle.
"""

import pathlib
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from alien_baby.envs import TabletopReachEnv
from alien_baby.envs.flatten_wrapper import FlattenVisionWrapper
from alien_baby.agents.train_staged import (
    MetricsCallback,
    _transfer_proprio_weights,
    _evaluate,
)

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def train_stage2_followon(
    stage1_path=None,
    total_timesteps=200_000,
    seed=42,
    proprio_dim=10,
):
    print("=" * 60)
    print("STAGE 2 FOLLOW-ON: vision on frozen Stage 1 actor")
    print("=" * 60)

    env = Monitor(TabletopReachEnv(vision=True, target_object=0))
    eval_env = Monitor(TabletopReachEnv(vision=True, target_object=0))
    env = FlattenVisionWrapper(env)
    eval_env = FlattenVisionWrapper(eval_env)

    if stage1_path is None:
        stage1_path = str(RESULTS_DIR / "stage1_checkpoint")

    obs_dim = env.observation_space.shape[0]
    print(f"Obs dim: {obs_dim} (proprio={proprio_dim}, pixels={obs_dim - proprio_dim})")

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
    _transfer_proprio_weights(stage1_model, model, proprio_dim=proprio_dim)

    # Locate the first-layer weight in the actor (the only one whose input dim == obs_dim)
    candidates = [
        (name, p)
        for name, p in model.actor.named_parameters()
        if p.dim() == 2 and p.shape[1] == obs_dim
    ]
    assert len(candidates) == 1, f"Expected 1 actor first-layer weight, found: {candidates}"
    first_name, first_layer_weight = candidates[0]
    print(f"Actor first-layer weight: {first_name} {tuple(first_layer_weight.shape)}")

    # Freeze all actor params EXCEPT the first-layer weight; that one stays
    # trainable but gets its proprio-column gradients zeroed by a hook.
    for name, p in model.actor.named_parameters():
        if p is first_layer_weight:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # Snapshot of the proprio columns so we can verify they never change.
    proprio_cols_snapshot = first_layer_weight.data[:, :proprio_dim].clone()

    mask = torch.ones_like(first_layer_weight)
    mask[:, :proprio_dim] = 0.0
    first_layer_weight.register_hook(lambda grad: grad * mask)

    actor_trainable = sum(p.numel() for p in model.actor.parameters() if p.requires_grad)
    actor_total = sum(p.numel() for p in model.actor.parameters())
    print(f"Actor trainable params: {actor_trainable}/{actor_total}")

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "followon_best"),
        log_path=str(RESULTS_DIR / "followon_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    # Sanity: proprio columns must be untouched.
    drift = (first_layer_weight.data[:, :proprio_dim] - proprio_cols_snapshot).abs().max().item()
    print(f"Proprio-column weight drift after training: {drift:.2e} (expected ~0)")
    assert drift < 1e-8, "Proprio-input columns drifted during training — freeze failed."

    checkpoint_path = str(RESULTS_DIR / "followon_checkpoint")
    model.save(checkpoint_path)
    print(f"\nFollow-on checkpoint saved to {checkpoint_path}")

    successes = _evaluate(model, env, n_episodes=20)
    print(f"Follow-on success rate: {successes}/{20} ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


if __name__ == "__main__":
    train_stage2_followon(total_timesteps=200_000)
