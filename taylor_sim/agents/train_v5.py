"""
v5: consistency-loss follow-on.

Same frozen-proprio / trainable-vision-columns architecture as v3, but the
actor loss gains an auxiliary term that pulls the hidden1 activation under
real pixels toward the hidden1 activation with pixels zeroed. Vision can
only *confirm* proprio's manifold, not steer away from it.

Loss = SAC_actor_loss + lambda * MSE(h_full, h_blind)
  h_full  = ReLU(W x_full  + b)   — full obs (pixels + proprio)
  h_blind = ReLU(W x_blind + b)   — proprio only, pixels zeroed

Gradient still only flows through the pixel columns (others frozen, proprio
columns masked out), so the loss physically constrains what vision is
allowed to contribute.
"""

import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update

from taylor_sim.envs import TabletopReachEnv
from taylor_sim.envs.flatten_wrapper import FlattenVisionWrapper
from taylor_sim.agents.train_staged import (
    MetricsCallback,
    _transfer_proprio_weights,
    _evaluate,
)

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROPRIO_DIM_V5 = 7


class ConsistencySAC(SAC):
    """SAC with auxiliary hidden1 consistency loss on the actor."""

    def __init__(self, *args, lambda_consistency=0.1, proprio_dim=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_consistency = lambda_consistency
        self.proprio_dim = proprio_dim

    def _hidden1(self, obs):
        feats = self.actor.extract_features(obs, self.actor.features_extractor)
        return self.actor.latent_pi[1](self.actor.latent_pi[0](feats))

    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, consistency_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_base_loss = (ent_coef * log_prob - min_qf_pi).mean()

            obs = replay_data.observations
            obs_blind = obs.clone()
            obs_blind[:, self.proprio_dim:] = 0.0
            h_full = self._hidden1(obs)
            h_blind = self._hidden1(obs_blind)
            consistency_loss = F.mse_loss(h_full, h_blind)
            consistency_losses.append(consistency_loss.item())

            actor_loss = actor_base_loss + self.lambda_consistency * consistency_loss
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/consistency_loss", np.mean(consistency_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


def train_followon_v5(
    stage1_path=None,
    total_timesteps=200_000,
    seed=42,
    lambda_consistency=0.1,
):
    print("=" * 60)
    print(f"FOLLOW-ON v5: consistency loss (lambda={lambda_consistency})")
    print("=" * 60)

    env = Monitor(TabletopReachEnv(vision=True, target_object=0, hide_target_offset=True))
    eval_env = Monitor(TabletopReachEnv(vision=True, target_object=0, hide_target_offset=True))
    env = FlattenVisionWrapper(env)
    eval_env = FlattenVisionWrapper(eval_env)

    if stage1_path is None:
        stage1_path = str(RESULTS_DIR / "stage1_v3_checkpoint")

    obs_dim = env.observation_space.shape[0]
    print(f"Obs dim: {obs_dim} (proprio={PROPRIO_DIM_V5}, pixels={obs_dim - PROPRIO_DIM_V5})")

    model = ConsistencySAC(
        "MlpPolicy",
        env,
        lambda_consistency=lambda_consistency,
        proprio_dim=PROPRIO_DIM_V5,
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
    _transfer_proprio_weights(stage1_model, model, proprio_dim=PROPRIO_DIM_V5)

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

    proprio_cols_snapshot = first_layer_weight.data[:, :PROPRIO_DIM_V5].clone()

    mask = torch.ones_like(first_layer_weight)
    mask[:, :PROPRIO_DIM_V5] = 0.0
    first_layer_weight.register_hook(lambda grad: grad * mask)

    trainable = sum(p.numel() for p in model.actor.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.actor.parameters())
    print(f"Actor trainable params: {trainable}/{total}")

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "followon_v5_best"),
        log_path=str(RESULTS_DIR / "followon_v5_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    drift = (first_layer_weight.data[:, :PROPRIO_DIM_V5] - proprio_cols_snapshot).abs().max().item()
    print(f"Proprio-column drift after training: {drift:.2e} (expected ~0)")
    assert drift < 1e-8, "Proprio columns drifted — freeze failed."

    checkpoint_path = str(RESULTS_DIR / "followon_v5_checkpoint")
    model.save(checkpoint_path)
    print(f"\nFollow-on v5 checkpoint saved to {checkpoint_path}")

    successes = _evaluate(model, env, n_episodes=20)
    print(f"Follow-on v5 success rate: {successes}/20 ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--lambda-consistency", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_followon_v5(
        total_timesteps=args.steps,
        lambda_consistency=args.lambda_consistency,
        seed=args.seed,
    )
