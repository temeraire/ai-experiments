"""
train_dialogue.py — Train MimoCrawlerEnv with the dialogue architecture.

Two-stream actor (proprio + vision) trained jointly with SAC, plus a
consistency loss penalizing disagreement between the per-stream action proposals.

Usage:
    python -m alien_baby.crawler.train_dialogue \\
        --steps 300000 --n-envs 4 --mps --learning-rate 1e-4 --ent-coef 0.2 \\
        --consistency-lambda 0.05 \\
        --spawn-cone-deg 360 --max-steps 2000 \\
        --run-tag mimo_dialogue_v1
"""

import argparse
import datetime
import pathlib

import numpy as np
import torch as th

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv
from alien_baby.crawler.dialogue_policy import DialogueSACPolicy, DialogueActor

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"


def make_env(rank, seed, vision=True, max_steps=2000, spawn_cone_deg=360,
             strength_scale=0.7, velocity_bonus_scale=0.0, approach_reward_scale=0.0):
    def _init():
        env = MimoCrawlerEnv(
            vision=vision,
            strength_scale=strength_scale,
            spawn_cone_deg=spawn_cone_deg,
            max_steps=max_steps,
            n_substeps=4,
            approach_reward_scale=approach_reward_scale,
            velocity_bonus_scale=velocity_bonus_scale,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


class DialogueSAC(SAC):
    """SAC variant that adds a consistency loss to the actor objective.

    L_actor_total = L_actor_sac + λ · MSE(μ_p, μ_v)
    """

    def __init__(self, *args, consistency_lambda: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.consistency_lambda = consistency_lambda

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Most of this is a verbatim copy of SAC.train(), plus the consistency-loss term.
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        consistency_losses = []
        dialogue_stats = {"gate_mean": [], "disagreement_mean": []}

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Action sampling using the dialogue actor
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Entropy coefficient (auto or fixed)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Critic targets
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Critic loss
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(th.nn.functional.mse_loss(c, target_q_values) for c in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Actor loss (with consistency penalty)
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            sac_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            consistency = self.actor.consistency_loss()
            consistency_losses.append(consistency.item())
            actor_loss = sac_actor_loss + self.consistency_lambda * consistency
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Diagnostics
            diag = self.actor.diagnostics()
            for k, v in diag.items():
                short = k.split("/")[-1]
                dialogue_stats.setdefault(short, []).append(v)

            # Polyak update of target critic
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        self.logger.record("train/consistency_loss", float(np.mean(consistency_losses)))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))
        for k, vals in dialogue_stats.items():
            if vals:
                self.logger.record(f"dialogue/{k}", float(np.mean(vals)))


def train(args):
    tag = args.run_tag or (
        "mimo_dialogue_" + datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    )
    out_dir = RESULTS_DIR / tag
    best_dir = RESULTS_DIR / (tag + "_best")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Dialogue training: {tag} ===")
    print(f"  steps={args.steps}  n_envs={args.n_envs}  λ_consistency={args.consistency_lambda}")
    print(f"  spawn_cone={args.spawn_cone_deg}°  max_steps={args.max_steps}")
    print(f"  velocity_bonus={args.velocity_bonus_scale}  approach_reward={args.approach_reward_scale}")

    train_env = DummyVecEnv([
        make_env(i, args.seed,
                 max_steps=args.max_steps,
                 spawn_cone_deg=args.spawn_cone_deg,
                 strength_scale=args.strength_scale,
                 velocity_bonus_scale=args.velocity_bonus_scale,
                 approach_reward_scale=args.approach_reward_scale)
        for i in range(args.n_envs)
    ])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([
        make_env(0, args.seed + 1000,
                 max_steps=args.max_steps,
                 spawn_cone_deg=args.spawn_cone_deg,
                 strength_scale=args.strength_scale,
                 velocity_bonus_scale=args.velocity_bonus_scale,
                 approach_reward_scale=args.approach_reward_scale)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                             training=False)

    device = "mps" if args.mps else "cpu"
    model = DialogueSAC(
        DialogueSACPolicy,
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
        device=device,
        consistency_lambda=args.consistency_lambda,
    )

    eval_cb  = EvalCallback(eval_env, best_model_save_path=str(best_dir),
                             log_path=str(out_dir), eval_freq=max(args.steps // 20, 5000),
                             n_eval_episodes=20, deterministic=True)
    ckpt_cb  = CheckpointCallback(save_freq=max(args.steps // 5, 25000),
                                    save_path=str(out_dir), name_prefix="ckpt")

    model.learn(total_timesteps=args.steps, callback=CallbackList([eval_cb, ckpt_cb]),
                progress_bar=True)

    model.save(out_dir / "final_model.zip")
    train_env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"\nFinal model: {out_dir/'final_model.zip'}")
    print(f"Best model:  {best_dir/'best_model.zip'}")

    # macOS done-sound
    try:
        import subprocess
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
    except Exception:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mps", action="store_true")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--ent-coef", default="0.2")  # string to allow "auto"
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--consistency-lambda", type=float, default=0.05)
    p.add_argument("--spawn-cone-deg", type=float, default=360.0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--strength-scale", type=float, default=0.7)
    p.add_argument("--velocity-bonus-scale", type=float, default=0.0)
    p.add_argument("--approach-reward-scale", type=float, default=0.0)
    p.add_argument("--run-tag", default=None)
    args = p.parse_args()
    if args.ent_coef != "auto":
        args.ent_coef = float(args.ent_coef)
    train(args)
