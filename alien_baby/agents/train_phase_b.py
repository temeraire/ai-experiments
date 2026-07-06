"""
Phase B: vision + locomotion together, ball back in play.

Initializes from the Phase A locomotion checkpoint and immediately
introduces vision — not as a later add-on, but as a co-learner from
step one. This is the mutual-confirmation setup: the body already knows
how to move (Phase A), and now vision and locomotion train together so
they can confirm each other.

Reward structure:
  - Contact (+200, sparse): the actual goal
  - PBRS approach gradient (dense): pulls directed locomotion toward ball
  - Speed bonus (tiny, dense): keeps body moving; prevents reversion to
    the static attractor now that there is a richer reward landscape

Eval metric to watch: touch rate (touches / episodes). Random-search
floor from the overnight sweep was ~10%. If Phase B exceeds this AND
improves over training, directed locomotion (not just random coverage)
is emerging.

Usage:
    python -m alien_baby.agents.train_phase_b
    python -m alien_baby.agents.train_phase_b --steps 50000 --run-tag smoke
    python -m alien_baby.agents.train_phase_b --phase-a-checkpoint <path>
"""

import argparse
import pathlib

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from alien_baby.envs.platform_creature_env import PlatformCreatureEnv
from alien_baby.agents.train_v8 import _best_device, _play_done_sound

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PHASE_A_DEFAULT_CHECKPOINT = str(RESULTS_DIR / "phase_a_best" / "best_model.zip")
PROPRIO_DIM = 29
N_ENVS = 16

PHASE_B_INFO_KEYWORDS = (
    "touched", "fell", "tilted", "mean_dist", "hunger_sum",
    "attract_sum", "shaping_sum",
)


class SpeedBonusWrapper(gym.Wrapper):
    """Adds a small per-step speed bonus to the env's existing reward.

    r_total = r_env + scale * ||v_xy||

    Keeps the body moving without overriding the contact + PBRS signals.
    Scale is intentionally tiny (default 0.01): at max observed speed
    (0.38 m/s) the bonus is 0.0038/step — ~1.5 total per episode, vs
    contact reward of 200. It nudges, not dominates.
    """

    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        qvel = self.env.unwrapped.data.qvel
        speed = float(np.hypot(qvel[0], qvel[1]))
        return obs, reward + self.scale * speed, terminated, truncated, info


def _transfer_actor_weights(src_model, dst_model):
    """Transfer Phase A actor weights into Phase B actor.

    Phase A obs_dim=29, Phase B obs_dim=3101. First layer of Phase B
    actor has 3101 input columns; we copy Phase A's 29 proprio columns
    and zero the 3072 pixel columns. Later layers are same size — copied
    directly. Critic starts fresh (different reward landscape).
    """
    src_params = dict(src_model.actor.named_parameters())
    dst_params = dict(dst_model.actor.named_parameters())
    transferred = 0
    for name, dst_p in dst_params.items():
        if name not in src_params:
            continue
        src_p = src_params[name]
        if dst_p.shape == src_p.shape:
            dst_p.data.copy_(src_p.data)
            transferred += 1
        elif dst_p.dim() == 2 and dst_p.shape[1] > src_p.shape[1]:
            # First layer: proprio columns → copy, pixel columns → zero
            dst_p.data.zero_()
            dst_p.data[:, :src_p.shape[1]] = src_p.data
            transferred += 1
    print(f"Transferred {transferred} actor parameter tensors from Phase A.")


class TouchRateCallback(BaseCallback):
    """Log touch rate and mean_dist from ep_info_buffer."""

    def _on_step(self) -> bool:
        buf = self.model.ep_info_buffer
        if not buf:
            return True
        touches = [float(bool(ep["touched"])) for ep in buf if "touched" in ep]
        if touches:
            self.logger.record("rollout/touch_rate", float(np.mean(touches)))
        dists = [ep["mean_dist"] for ep in buf if "mean_dist" in ep]
        if dists:
            self.logger.record("rollout/mean_dist", float(np.mean(dists)))
        return True


def _make_env(pbrs_alpha, max_steps, speed_bonus_scale):
    def _make():
        env = PlatformCreatureEnv(
            vision=True, stage=1,
            pbrs_alpha=pbrs_alpha,
            max_steps_override=max_steps,
        )
        env = SpeedBonusWrapper(env, scale=speed_bonus_scale)
        return Monitor(env, info_keywords=PHASE_B_INFO_KEYWORDS)
    return _make


def train_phase_b(total_timesteps, seed, checkpoint_interval, run_tag,
                  ent_coef, phase_a_checkpoint, pbrs_alpha, max_steps,
                  speed_bonus_scale):
    suffix = f"_{run_tag}" if run_tag else ""
    print("=" * 60)
    print(f"PHASE B: vision + locomotion, ball in play")
    print(f"  steps={total_timesteps:,}  envs={N_ENVS}  seed={seed}")
    print(f"  phase_a_checkpoint={phase_a_checkpoint}")
    print(f"  reward = contact + PBRS(α={pbrs_alpha}) + speed×{speed_bonus_scale}")
    print(f"  episode_length={max_steps} steps ({max_steps*0.01:.0f}s)")
    print(f"  ent_coef={ent_coef}")
    print("=" * 60)

    env = SubprocVecEnv(
        [_make_env(pbrs_alpha, max_steps, speed_bonus_scale)
         for _ in range(N_ENVS)],
        start_method="spawn",
    )
    eval_env = Monitor(
        SpeedBonusWrapper(
            PlatformCreatureEnv(
                vision=True, stage=1,
                pbrs_alpha=pbrs_alpha,
                max_steps_override=max_steps,
            ),
            scale=speed_bonus_scale,
        ),
        info_keywords=PHASE_B_INFO_KEYWORDS,
    )

    # Build Phase B model (vision obs = 3101 dims)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,       # lower than Phase A — warm-started policy
        buffer_size=500_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=N_ENVS,
        learning_starts=5000,
        ent_coef=ent_coef,
        verbose=1,
        seed=seed,
        device=_best_device(),
    )

    # Transfer Phase A actor weights (proprio columns only; critic starts fresh)
    phase_a = SAC.load(phase_a_checkpoint, device="cpu")
    _transfer_actor_weights(phase_a, model)
    del phase_a

    touch_cb = TouchRateCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / f"phase_b{suffix}_best"),
        log_path=str(RESULTS_DIR / f"phase_b{suffix}_logs"),
        eval_freq=max(10000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(checkpoint_interval // N_ENVS, 1),
        save_path=str(RESULTS_DIR / f"phase_b{suffix}_checkpoints"),
        name_prefix=f"phase_b{suffix}",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([touch_cb, eval_cb, ckpt_cb]),
    )

    final_path = str(RESULTS_DIR / f"phase_b{suffix}_checkpoint")
    model.save(final_path)
    print(f"\nPhase B saved to {final_path}")

    # Final touch-rate eval
    touches = 0
    for i in range(20):
        obs, _ = eval_env.reset(seed=i + 2000)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        if info.get("touched"):
            touches += 1
    print(f"Final touch rate: {touches}/20 ({touches/20*100:.0f}%)")
    print(f"(Blind floor from overnight sweep: ~10%)")

    env.close()
    eval_env.close()
    _play_done_sound()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--phase-a-checkpoint", default=PHASE_A_DEFAULT_CHECKPOINT,
                        help="Path to Phase A best_model.zip. Default: "
                             "results/phase_a_best/best_model.zip")
    parser.add_argument("--pbrs-alpha", type=float, default=0.3,
                        help="PBRS approach gradient coefficient. 0.3 provides "
                             "a dense pull toward the ball without changing the "
                             "optimal policy (Ng et al. 1999).")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Episode length in sim steps (default 1000 = 10s). "
                             "Longer than Stage 1 default of 300 so a moving AB "
                             "has time to cover the platform and find the ball.")
    parser.add_argument("--speed-bonus", type=float, default=0.01,
                        help="Scale on per-step speed bonus. Keeps body moving. "
                             "0.01 = ~1.5 total per episode vs contact=200.")
    parser.add_argument("--ent-coef", default="auto")
    args = parser.parse_args()

    try:
        ent_coef = float(args.ent_coef)
    except ValueError:
        ent_coef = args.ent_coef

    train_phase_b(
        total_timesteps=args.steps,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        run_tag=args.run_tag,
        ent_coef=ent_coef,
        phase_a_checkpoint=args.phase_a_checkpoint,
        pbrs_alpha=args.pbrs_alpha,
        max_steps=args.max_steps,
        speed_bonus_scale=args.speed_bonus,
    )
