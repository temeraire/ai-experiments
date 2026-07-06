"""
Baseline agents for comparison with the staged agent.

1. All-at-once: Same architecture, all modalities from the start.
2. Feature-fusion: Separate encoders merged at a fusion layer.
"""

import pathlib
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from alien_baby.envs import TabletopReachEnv
from alien_baby.envs.flatten_wrapper import FlattenVisionWrapper
from alien_baby.agents.train_staged import MetricsCallback, _evaluate

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def train_all_at_once(total_timesteps=100_000, seed=42):
    """
    All-at-once baseline: same total training budget as staged agent,
    all modalities available from the start, same architecture.
    """
    print("=" * 60)
    print("BASELINE: All-at-once (vision + proprio from start)")
    print("=" * 60)

    env = Monitor(TabletopReachEnv(vision=True, target_object=0))
    eval_env = Monitor(TabletopReachEnv(vision=True, target_object=0))
    env = FlattenVisionWrapper(env)
    eval_env = FlattenVisionWrapper(eval_env)

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
        best_model_save_path=str(RESULTS_DIR / "allatonce_best"),
        log_path=str(RESULTS_DIR / "allatonce_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    checkpoint_path = str(RESULTS_DIR / "allatonce_checkpoint")
    model.save(checkpoint_path)
    print(f"\nAll-at-once checkpoint saved to {checkpoint_path}")

    successes = _evaluate(model, env, n_episodes=20)
    print(f"All-at-once success rate: {successes}/{20} ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


def train_feature_fusion(total_timesteps=100_000, seed=42):
    """
    Feature-fusion baseline: separate processing for vision and proprio,
    merged at a later layer. Uses custom network architecture.

    Since SB3 doesn't natively support this with MlpPolicy, we approximate
    by using a larger network where the first hidden layer is double-width,
    effectively giving each modality its own "encoder" portion.
    """
    print("=" * 60)
    print("BASELINE: Feature-fusion (separate encoders)")
    print("=" * 60)

    env = Monitor(TabletopReachEnv(vision=True, target_object=0))
    eval_env = Monitor(TabletopReachEnv(vision=True, target_object=0))
    env = FlattenVisionWrapper(env)
    eval_env = FlattenVisionWrapper(eval_env)

    # Wider first layer to approximate separate encoders
    policy_kwargs = dict(net_arch=[512, 256, 256])

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
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="cpu",
    )

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / "fusion_best"),
        log_path=str(RESULTS_DIR / "fusion_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    checkpoint_path = str(RESULTS_DIR / "fusion_checkpoint")
    model.save(checkpoint_path)
    print(f"\nFeature-fusion checkpoint saved to {checkpoint_path}")

    successes = _evaluate(model, env, n_episodes=20)
    print(f"Feature-fusion success rate: {successes}/{20} ({successes/20*100:.0f}%)")

    env.close()
    eval_env.close()
    return model, metrics_cb


if __name__ == "__main__":
    print("Training baseline agents...\n")
    model_aao, metrics_aao = train_all_at_once(total_timesteps=100_000)
    model_ff, metrics_ff = train_feature_fusion(total_timesteps=100_000)
    print("\nDone! Baselines trained.")
