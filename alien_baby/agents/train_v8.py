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
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, VecNormalize, DummyVecEnv, sync_envs_normalization,
)

from alien_baby.envs.platform_creature_env import PlatformCreatureEnv
from alien_baby.envs.mirror_wrapper import MirrorWrapper
from alien_baby.agents.train_staged import MetricsCallback, _evaluate
from alien_baby.agents.train_v5 import ConsistencySAC

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROPRIO_DIM_V8 = 29
N_ENVS_FOLLOWON = 16  # was 8; 16 cores on Apple Silicon → 2x env parallelism


def _best_device():
    """Return 'mps' on Apple Silicon, 'cuda' if available, else 'cpu'.
    SB3's 'auto' doesn't pick mps, so we select explicitly."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


FOLLOWON_INFO_KEYWORDS = (
    "hunger_sum", "attract_sum", "mean_dist", "touched", "fell", "ball_lost",
    "tilted", "spawn_angle", "spawn_left", "shaping_sum", "closure_sum",
    "ctrl_cost_sum", "mirrored",
)


class RadiusAnnealCallback(BaseCallback):
    """Linear radius anneal curriculum. Holds `radius_start` until
    `anneal_begin_step`, linearly interpolates to `radius_end` across
    [anneal_begin_step, anneal_end_step], then holds `radius_end` forever.

    Forces locomotion to emerge before the policy settles into a turret
    attractor: start radius is just outside arm reach so the creature must
    move, but close enough that random wheel jitter under pinned entropy
    occasionally produces contact. Once the locomotion engram is wired
    (touch rate stabilizes), the anneal extends the spatial horizon.

    Implementation note: the radius bucket is rounded to 1 cm so env IPC
    only fires ~20 times over the whole anneal, not every step."""

    def __init__(self, radius_start, radius_end,
                 anneal_begin_step, anneal_end_step, verbose=1):
        super().__init__(verbose)
        self.r0 = float(radius_start)
        self.r1 = float(radius_end)
        self.t0 = int(anneal_begin_step)
        self.t1 = int(anneal_end_step)
        self._last_r = None

    def _current_radius(self, t):
        if t < self.t0:
            return self.r0
        if t >= self.t1:
            return self.r1
        frac = (t - self.t0) / (self.t1 - self.t0)
        return self.r0 + frac * (self.r1 - self.r0)

    def _on_step(self) -> bool:
        r = round(self._current_radius(self.num_timesteps), 2)
        if r != self._last_r:
            self.model.env.env_method("set_target_radius", (r, r))
            self.logger.record("rollout/spawn_radius", r)
            if self.verbose:
                print(f"[RadiusAnneal] step {self.num_timesteps}: radius → {r:.2f} m")
            self._last_r = r
        return True


class Stage1RadiusAnnealCallback(BaseCallback):
    """Anneal the UPPER bound of the stage-1 spawn radius over a relative
    step window. Lower bound stays pinned at `radius_lo`.

    Unlike `RadiusAnnealCallback` (which pins lo = hi = r and drives a single
    value absolute-step), this callback keeps a floor and stretches the
    ceiling — producing a mixed-range distribution where easy close-spawns
    stay present while far spawns gradually enter the training set. Intended
    for transitioning a "contact-reflex" policy into a locomotion policy
    without collapsing the existing behavior.

    Step accounting is RELATIVE to the first _on_step call so warm-start
    runs (via --stage1-init-from) start their anneal at step 0 of the new
    phase, not whatever `num_timesteps` the loaded checkpoint had.
    """

    def __init__(self, radius_lo, radius_hi_start, radius_hi_end,
                 anneal_steps, verbose=1):
        super().__init__(verbose)
        self.lo = float(radius_lo)
        self.hi0 = float(radius_hi_start)
        self.hi1 = float(radius_hi_end)
        self.anneal_steps = int(anneal_steps)
        self._init_ts = None
        self._last_hi = None

    def _current_hi(self, t_rel):
        if t_rel >= self.anneal_steps:
            return self.hi1
        frac = t_rel / max(self.anneal_steps, 1)
        return self.hi0 + frac * (self.hi1 - self.hi0)

    def _on_step(self) -> bool:
        if self._init_ts is None:
            self._init_ts = int(self.num_timesteps)
        t_rel = int(self.num_timesteps) - self._init_ts
        hi = round(self._current_hi(t_rel), 2)
        if hi != self._last_hi:
            self.model.env.env_method("set_target_radius", (self.lo, hi))
            self.logger.record("rollout/spawn_radius_hi", hi)
            if self.verbose:
                print(f"[Stage1RadiusAnneal] rel_step {t_rel}: "
                      f"(lo,hi) → ({self.lo:.2f}, {hi:.2f}) m")
            self._last_hi = hi
        return True


class RewardComponentCallback(BaseCallback):
    """Log per-rollout means of the env's component signals to SB3's logger so
    they appear in the rollout/ block. Reads from model.ep_info_buffer, which
    Monitor populates via info_keywords. Needed for the 50K abort criterion
    (mean_dist downward trend, touch rate) to be observable during training."""

    def _on_step(self) -> bool:
        buf = self.model.ep_info_buffer
        if not buf:
            return True
        for key in ("hunger_sum", "attract_sum", "mean_dist", "shaping_sum",
                    "closure_sum", "ctrl_cost_sum"):
            vals = [ep[key] for ep in buf if key in ep]
            if vals:
                self.logger.record(f"rollout/{key}_mean", float(np.mean(vals)))
        for key in ("touched", "fell", "ball_lost", "tilted", "mirrored"):
            vals = [float(bool(ep[key])) for ep in buf if key in ep]
            if vals:
                self.logger.record(f"rollout/{key}_frac", float(np.mean(vals)))
        # Per-hemisphere touch rates — watch for specialization collapse
        # (policy only touches in one hemisphere while ignoring the other).
        left_touches = [float(bool(ep["touched"])) for ep in buf
                        if "spawn_left" in ep and ep["spawn_left"]]
        right_touches = [float(bool(ep["touched"])) for ep in buf
                         if "spawn_left" in ep and not ep["spawn_left"]]
        if left_touches:
            self.logger.record("rollout/touched_left_frac", float(np.mean(left_touches)))
        if right_touches:
            self.logger.record("rollout/touched_right_frac", float(np.mean(right_touches)))
        return True


def _make_followon_env(v9=False, target_radius_override=None, pbrs_alpha=0.0,
                       mirror_augmentation=False):
    def _make():
        env = PlatformCreatureEnv(
            vision=True, v9=v9,
            target_radius_override=target_radius_override,
            pbrs_alpha=pbrs_alpha,
        )
        if mirror_augmentation:
            env = MirrorWrapper(env)
        return Monitor(env, info_keywords=FOLLOWON_INFO_KEYWORDS)
    return _make


class VecNormalizeSaveCallback(BaseCallback):
    """Save VecNormalize statistics alongside model checkpoints. SB3's
    CheckpointCallback only persists the model; the running mean/std of the
    obs/reward normalizer must be saved separately or eval at load time
    sees denormalized observations."""

    def __init__(self, vec_env, save_freq, save_dir, name_prefix, verbose=1):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.save_freq = int(save_freq)
        self.save_dir = pathlib.Path(save_dir)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path = self.save_dir / (
                f"{self.name_prefix}_vecnorm_{self.num_timesteps}_steps.pkl"
            )
            self.vec_env.save(str(path))
            if self.verbose:
                print(f"[VecNormSave] {path.name}")
        return True


def _play_done_sound():
    """Play a chime + spoken 'training done' when training finishes.
    macOS only; silent elsewhere. Tries multiple methods since a backgrounded
    subprocess with a piped stdout can sometimes make afplay silently fail."""
    import platform, subprocess, sys
    if platform.system() != "Darwin":
        return
    print(">>> TRAINING COMPLETE — chime triggered", flush=True)
    # Try afplay (Glass chime) and `say` (TTS) in sequence; if one fails the
    # other should still fire.
    # afplay -v is a volume multiplier; values >1 amplify. Using 3 is loud
    # enough to hear from another room. Hero.aiff is more fanfare-like than
    # Glass.aiff — harder to miss.
    for cmd in (
        ["afplay", "-v", "3", "/System/Library/Sounds/Hero.aiff"],
        ["afplay", "-v", "3", "/System/Library/Sounds/Hero.aiff"],
        ["say", "-v", "Samantha", "-r", "200", "training complete"],
    ):
        try:
            subprocess.run(cmd, check=False, timeout=10,
                           stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"    ({cmd[0]} failed: {e})", file=sys.stderr, flush=True)


def train_stage1_v8(total_timesteps=500_000, seed=42, checkpoint_interval=50_000,
                    curriculum_stage=1, init_from=None, ent_coef="auto",
                    v9=False, target_radius_override=None, run_tag=None,
                    pbrs_alpha=0.0, mirror_augmentation=False,
                    stage1_anneal_max_radius=None, stage1_anneal_steps=None,
                    closure_bonus=0.0):
    # Build output suffix. If run_tag is supplied, use it; otherwise fall back
    # to the curriculum_stage-based naming so older callers keep working.
    if run_tag:
        suffix = f"_{run_tag}"
    else:
        suffix = "" if curriculum_stage == 1 else f"_c{curriculum_stage}"
    print("=" * 60)
    version_label = "v9" if v9 else "v8"
    print(f"STAGE 1 {version_label}: blind proprio on platform "
          f"(curriculum stage {curriculum_stage})")
    if init_from:
        print(f"Warm-starting from: {init_from}")
    if target_radius_override is not None:
        print(f"Spawn radius override: {target_radius_override}")
    if pbrs_alpha > 0:
        print(f"PBRS enabled: alpha={pbrs_alpha} (Φ(s)=-α·body_ball_dist)")
    if mirror_augmentation:
        print("Mirror augmentation: ENABLED (per-episode L/R coin flip)")
    if stage1_anneal_max_radius is not None:
        print(f"Stage-1 radius anneal: hi → {stage1_anneal_max_radius} "
              f"over {stage1_anneal_steps} steps (relative to run start)")
    if closure_bonus > 0.0:
        print(f"Closure bonus: ENABLED "
              f"(scale={closure_bonus} · max(0, Δ body→ball dist))")
    print("=" * 60)

    train_inner = PlatformCreatureEnv(
        vision=False, stage=curriculum_stage, v9=v9,
        target_radius_override=target_radius_override,
        pbrs_alpha=pbrs_alpha,
        closure_bonus_scale=closure_bonus,
    )
    if mirror_augmentation:
        train_inner = MirrorWrapper(train_inner)
    env = Monitor(train_inner, info_keywords=FOLLOWON_INFO_KEYWORDS)
    # Eval env is NEVER mirror-wrapped: we want eval rewards measured on the
    # raw env so best_model.zip reflects policy quality on the unmodified
    # task, comparable across runs with/without the augmentation. Closure
    # bonus IS applied to eval — it's part of the env's true reward now.
    eval_env = Monitor(PlatformCreatureEnv(
        vision=False, stage=curriculum_stage, v9=v9,
        target_radius_override=target_radius_override,
        pbrs_alpha=pbrs_alpha,
        closure_bonus_scale=closure_bonus,
    ))

    if init_from:
        model = SAC.load(init_from, env=env, device="cpu")
        model.set_env(env)
        # Fresh learning_starts so the warm-started policy is used from step 1,
        # not overwritten by purely random actions.
        model.learning_starts = 0
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=500_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            learning_starts=5000,
            ent_coef=ent_coef,
            verbose=1,
            seed=seed,
            device=_best_device(),
        )
    print(f"Entropy coefficient: {ent_coef} "
          f"{'(auto-tuned)' if ent_coef == 'auto' else '(PINNED — no decay)'}")

    metrics_cb = MetricsCallback()
    component_cb = RewardComponentCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / f"stage1_v8{suffix}_best"),
        log_path=str(RESULTS_DIR / f"stage1_v8{suffix}_logs"),
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=str(RESULTS_DIR / f"stage1_v8{suffix}_checkpoints"),
        name_prefix=f"stage1_v8{suffix}",
    )

    callbacks = [metrics_cb, component_cb]
    if stage1_anneal_max_radius is not None:
        if stage1_anneal_steps is None or stage1_anneal_steps <= 0:
            raise ValueError(
                "--stage1-anneal-steps must be set and positive when "
                "--stage1-anneal-max-radius-to is provided."
            )
        lo_start, hi_start = (
            target_radius_override if target_radius_override else (0.2, 0.7)
        )
        callbacks.append(Stage1RadiusAnnealCallback(
            radius_lo=lo_start,
            radius_hi_start=hi_start,
            radius_hi_end=stage1_anneal_max_radius,
            anneal_steps=stage1_anneal_steps,
        ))
    callbacks.extend([eval_cb, ckpt_cb])
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
    )

    final_path = str(RESULTS_DIR / f"stage1_v8{suffix}_checkpoint")
    model.save(final_path)
    print(f"\nStage 1 v8 (curriculum stage {curriculum_stage}) saved to {final_path}")

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

    print(f"Stage 1 v8 (curriculum stage {curriculum_stage}): {successes}/20 touches, {falls}/20 falls")

    env.close()
    eval_env.close()
    _play_done_sound()
    return model, metrics_cb


def train_followon_v8(
    stage1_path=None,
    total_timesteps=1_000_000,
    seed=42,
    lambda_consistency=0.1,
    freeze_proprio=False,
    checkpoint_interval=50_000,
    run_tag=None,
    v9=False,
    from_scratch=False,
    ent_coef="auto",
    target_radius_override=None,
    radius_anneal=None,
    followon_init_from=None,
    pbrs_alpha=0.0,
    mirror_augmentation=False,
    vec_normalize=False,
):
    freeze_label = "frozen" if freeze_proprio else "unfrozen"
    # Auto-tag output dirs so back-to-back frozen/unfrozen runs don't clobber.
    if run_tag is None:
        prefix = "v9_" if v9 else ""
        prefix += "scratch_" if from_scratch else ""
        run_tag = prefix + freeze_label
    suffix = f"_{run_tag}"
    print("=" * 60)
    print(f"FOLLOW-ON v8: vision + platform ({freeze_label} proprio, lambda={lambda_consistency})")
    print(f"Run tag: {run_tag}")
    print("=" * 60)

    # Dump the full launch config next to the other run artifacts so future
    # diagnostics don't require grepping shell history / stdout logs.
    import json
    config_path = RESULTS_DIR / f"followon_v8{suffix}_config.json"
    config_path.write_text(json.dumps({
        "stage1_path": stage1_path,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "lambda_consistency": lambda_consistency,
        "freeze_proprio": freeze_proprio,
        "checkpoint_interval": checkpoint_interval,
        "run_tag": run_tag,
        "v9": v9,
        "from_scratch": from_scratch,
        "ent_coef": ent_coef,
        "n_envs": N_ENVS_FOLLOWON,
        "target_radius_override": target_radius_override,
        "radius_anneal": radius_anneal,
        "followon_init_from": followon_init_from,
        "pbrs_alpha": pbrs_alpha,
        "mirror_augmentation": mirror_augmentation,
        "vec_normalize": vec_normalize,
    }, indent=2))
    print(f"Config written to {config_path}")
    if target_radius_override is not None:
        print(f"Spawn radius override: {target_radius_override}")
    if radius_anneal is not None:
        print(f"Radius anneal: start={radius_anneal['radius_start']} → "
              f"end={radius_anneal['radius_end']} across "
              f"[{radius_anneal['anneal_begin_step']}, "
              f"{radius_anneal['anneal_end_step']}]")
    if pbrs_alpha > 0:
        print(f"PBRS enabled: alpha={pbrs_alpha} (Φ(s)=-α·body_ball_dist)")

    # Anneal semantics:
    #   - VecEnv workers START at radius_start; the callback mutates them in
    #     place as training progresses (set_target_radius via env_method).
    #   - eval env is PINNED to radius_end for the whole run, so best_model.zip
    #     captures the best policy measured at the target distribution (not
    #     whatever radius the anneal happens to be at when the eval fires).
    vec_env_radius = target_radius_override
    eval_env_radius = target_radius_override
    if radius_anneal is not None:
        vec_env_radius = (float(radius_anneal["radius_start"]),) * 2
        eval_env_radius = (float(radius_anneal["radius_end"]),) * 2

    env = SubprocVecEnv(
        [_make_followon_env(v9=v9, target_radius_override=vec_env_radius,
                            pbrs_alpha=pbrs_alpha,
                            mirror_augmentation=mirror_augmentation)
         for _ in range(N_ENVS_FOLLOWON)],
        start_method="spawn",
    )
    if vec_normalize:
        # If --followon-init-from is paired with a sibling vec_normalize.pkl,
        # load existing stats; otherwise start fresh. Mismatch (init-from set
        # but no pkl found) gets a loud warning since the policy was trained
        # against normalized obs but will see nearly-raw obs until stats
        # converge.
        existing_vecnorm = None
        if followon_init_from is not None:
            candidate = pathlib.Path(followon_init_from).parent / "vec_normalize.pkl"
            if candidate.exists():
                existing_vecnorm = str(candidate)
        if existing_vecnorm:
            print(f"VecNormalize: loading stats from {existing_vecnorm}")
            env = VecNormalize.load(existing_vecnorm, env)
            env.training = True
            env.norm_reward = True
        else:
            if followon_init_from is not None:
                print(f"VecNormalize: WARNING — no vec_normalize.pkl alongside "
                      f"{followon_init_from}; starting with fresh stats. "
                      f"Loaded policy will see denormalized obs until stats "
                      f"warm up (~1k env steps).")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        # Eval env: VecNormalize'd with training=False. EvalCallback's
        # sync_envs_normalization() copies stats from training env before
        # each eval automatically.
        eval_env_inner = DummyVecEnv([lambda: Monitor(PlatformCreatureEnv(
            vision=True, v9=v9, target_radius_override=eval_env_radius,
            pbrs_alpha=pbrs_alpha,
        ))])
        eval_env = VecNormalize(eval_env_inner, training=False,
                                norm_reward=False, clip_obs=10.0)
    else:
        eval_env = Monitor(PlatformCreatureEnv(
            vision=True, v9=v9, target_radius_override=eval_env_radius,
            pbrs_alpha=pbrs_alpha,
        ))

    if stage1_path is None:
        stage1_path = str(RESULTS_DIR / "stage1_v8_checkpoint")

    obs_dim = env.observation_space.shape[0]
    print(f"Obs dim: {obs_dim} (proprio={PROPRIO_DIM_V8}, pixels={obs_dim - PROPRIO_DIM_V8})")

    if followon_init_from is not None:
        # Continue training from an existing followon checkpoint (phase-2
        # pattern). Loads the complete model — proprio + vision actor weights,
        # critic, replay-buffer-less optimizer state — then binds the new env.
        # learning_starts=0 so the already-trained policy is used immediately
        # rather than being overwritten by random actions.
        print(f"FOLLOWON-INIT-FROM: loading full checkpoint {followon_init_from}")
        model = ConsistencySAC.load(
            followon_init_from, env=env, device=_best_device(),
        )
        model.set_env(env)
        model.learning_starts = 0
        # Re-assert scheduler-style config (CLI values override whatever the
        # saved model had). lambda_consistency is a ConsistencySAC attr.
        model.lambda_consistency = lambda_consistency
        print(f"Entropy coefficient: {ent_coef} "
              f"{'(carried from loaded model)' if ent_coef == 'auto' else '(PINNED — no decay)'}")
    else:
        model = ConsistencySAC(
            "MlpPolicy",
            env,
            lambda_consistency=lambda_consistency,
            proprio_dim=PROPRIO_DIM_V8,
            learning_rate=1e-4,
            buffer_size=500_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=N_ENVS_FOLLOWON,
            learning_starts=5000,
            ent_coef=ent_coef,
            verbose=1,
            seed=seed,
            device=_best_device(),
        )
        print(f"Entropy coefficient: {ent_coef} "
              f"{'(auto-tuned)' if ent_coef == 'auto' else '(PINNED — no decay)'}")

        # Transfer stage 1 weights (unless from_scratch — then vision starts with
        # a fresh random policy and has to learn the task together with proprio).
        if not from_scratch:
            stage1_model = SAC.load(stage1_path)
            from alien_baby.agents.train_staged import _transfer_proprio_weights
            _transfer_proprio_weights(stage1_model, model, proprio_dim=PROPRIO_DIM_V8)
        else:
            print("FROM-SCRATCH: no stage-1 weight transfer; vision and proprio learn jointly.")

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
    component_cb = RewardComponentCallback()
    extra_cbs = []
    if radius_anneal is not None:
        extra_cbs.append(RadiusAnnealCallback(
            radius_start=radius_anneal["radius_start"],
            radius_end=radius_anneal["radius_end"],
            anneal_begin_step=radius_anneal["anneal_begin_step"],
            anneal_end_step=radius_anneal["anneal_end_step"],
        ))
    if vec_normalize:
        extra_cbs.append(VecNormalizeSaveCallback(
            vec_env=env,
            save_freq=max(checkpoint_interval // N_ENVS_FOLLOWON, 1),
            save_dir=str(RESULTS_DIR / f"followon_v8{suffix}_checkpoints"),
            name_prefix=f"followon_v8{suffix}",
        ))
    # SB3 callbacks count `env.step()` calls, not env-steps. With N_ENVS parallel
    # envs each call = N_ENVS env-steps, so divide intended env-step frequencies
    # by N_ENVS_FOLLOWON so the callbacks fire at the intended env-step cadence.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RESULTS_DIR / f"followon_v8{suffix}_best"),
        log_path=str(RESULTS_DIR / f"followon_v8{suffix}_logs"),
        eval_freq=max(10000 // N_ENVS_FOLLOWON, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(checkpoint_interval // N_ENVS_FOLLOWON, 1),
        save_path=str(RESULTS_DIR / f"followon_v8{suffix}_checkpoints"),
        name_prefix=f"followon_v8{suffix}",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(
            [metrics_cb, component_cb, *extra_cbs, eval_cb, ckpt_cb]
        ),
    )

    if freeze_proprio and proprio_cols_snapshot is not None:
        drift = (first_layer_weight.data[:, :PROPRIO_DIM_V8] - proprio_cols_snapshot).abs().max().item()
        print(f"Proprio-column drift: {drift:.2e}")

    final_path = str(RESULTS_DIR / f"followon_v8{suffix}_checkpoint")
    model.save(final_path)
    print(f"\nFollow-on v8 ({run_tag}) saved to {final_path}")

    if vec_normalize:
        vecnorm_final = RESULTS_DIR / f"followon_v8{suffix}_vec_normalize.pkl"
        env.save(str(vecnorm_final))
        print(f"VecNormalize stats saved to {vecnorm_final}")

    # Final 20-episode eval. Under --vec-normalize, the model expects
    # normalized obs; wrap in DummyVecEnv → VecNormalize, sync stats from
    # the trained env, and use the vec API for the loop.
    successes = 0
    falls = 0
    if vec_normalize:
        final_eval_inner = DummyVecEnv([lambda: PlatformCreatureEnv(
            vision=True, v9=v9, target_radius_override=target_radius_override,
            pbrs_alpha=pbrs_alpha,
        )])
        final_eval_env = VecNormalize(final_eval_inner, training=False,
                                      norm_reward=False, clip_obs=10.0)
        sync_envs_normalization(env, final_eval_env)
        for i in range(20):
            final_eval_env.seed(i + 1000)
            obs = final_eval_env.reset()
            done = False
            info = None
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = final_eval_env.step(action)
                done = bool(dones[0])
                info = infos[0]
            if info.get("touched"):
                successes += 1
            if info.get("fell"):
                falls += 1
    else:
        final_eval_env = PlatformCreatureEnv(
            vision=True, v9=v9, target_radius_override=target_radius_override,
            pbrs_alpha=pbrs_alpha,
        )
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
    _play_done_sound()
    return model, metrics_cb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "followon", "both"], default="both")
    parser.add_argument("--stage1-steps", type=int, default=500_000)
    parser.add_argument("--followon-steps", type=int, default=250_000,
                        help="Default 250K per CLAUDE.md workflow: render video, then extend if promising.")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--lambda-consistency", type=float, default=0.1)
    parser.add_argument("--freeze-proprio", action="store_true",
                        help="Freeze proprio weights in follow-on (default: unfrozen)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum-stage", type=int, default=1, choices=[0, 1],
                        help="Env curriculum: 0 = locomotion-only scaffold (big floor, "
                             "no fall penalty, target close and in front), 1 = full v8 "
                             "survival env. Only applied to Stage 1 training.")
    parser.add_argument("--stage1-checkpoint", default=None,
                        help="Path to stage-1 checkpoint that follow-on should load. "
                             "Defaults to results/stage1_v8_checkpoint.")
    parser.add_argument("--stage1-init-from", default=None,
                        help="Warm-start stage-1 training from this SAC checkpoint "
                             "(e.g. the c0 scaffold checkpoint). Skips learning_starts.")
    parser.add_argument("--v9", action="store_true",
                        help="Use v9 env: moving target (free-joint ball, initial "
                             "velocity, rolling friction decay, ball-lost termination).")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Follow-on only: skip the stage-1 weight transfer and "
                             "train the full policy (proprio + vision) from scratch.")
    parser.add_argument("--run-tag", default=None,
                        help="Custom run tag for follow-on output dirs (best_model, "
                             "checkpoints, logs). Default: auto-derived from "
                             "v9/scratch/frozen flags. Use this to avoid clobbering "
                             "prior runs with matching auto-tags.")
    parser.add_argument("--ent-coef", default="auto",
                        help="SAC entropy coefficient. 'auto' = SB3 auto-tuned "
                             "(default, shrinks as policy converges). Pass a float "
                             "like '0.1' to pin it — prevents lock-in to asymmetric "
                             "attractors by forcing continued exploration.")
    parser.add_argument("--stage1-radius", default=None,
                        help="Override stage-1 spawn radius. Accepts a single "
                             "float '0.5' (pinned) or a range 'lo,hi' like "
                             "'0.4,0.7' (uniform sampling). Combined with --v9, "
                             "this lets stage 1 train under the v9 XML and "
                             "v9 front-hemisphere spawn geometry.")
    parser.add_argument("--warmstart-radius", type=float, default=None,
                        help="Override spawn radius (in meters) for follow-on "
                             "training. Forces _target_radius_lo = _target_radius_hi "
                             "= this value. Use to shrink the search volume when "
                             "SAC can't bootstrap from the default radius under a "
                             "sparse/last-mile reward. E.g. 0.25 for a warm-start "
                             "probe before the default 0.5 (v9).")
    parser.add_argument("--followon-radius", default=None,
                        help="Override follow-on spawn radius as a uniform "
                             "range. Accepts 'x' (pinned) or 'lo,hi' (e.g. "
                             "'0.30,0.70'). Mutually exclusive with "
                             "--warmstart-radius and --anneal-* flags.")
    parser.add_argument("--pbrs-alpha", type=float, default=0.0,
                        help="Potential-based reward shaping coefficient "
                             "(Ng, Harada & Russell 1999). Φ(s) = -α·body_ball_dist. "
                             "0.0 disables. 0.3 is a reasonable starting value — "
                             "provides a dense approach-gradient without changing "
                             "the optimal policy. Recommended when sparse-reward "
                             "bootstrap stalls under pinned entropy.")
    parser.add_argument("--followon-init-from", default=None,
                        help="Continue training from a full followon checkpoint "
                             "(ConsistencySAC.load). Unlike --stage1-checkpoint "
                             "(which only copies proprio columns), this loads the "
                             "complete model including vision weights. Use for "
                             "multi-phase training: e.g. phase 1 at r=0.25, then "
                             "phase 2 at r=0.50 with --followon-init-from=<phase1>.")
    parser.add_argument("--anneal-radius-start", type=float, default=None,
                        help="Radius anneal: starting spawn radius (m). "
                             "If any --anneal-* flag is set, all four must be.")
    parser.add_argument("--anneal-radius-end", type=float, default=None,
                        help="Radius anneal: ending spawn radius (m).")
    parser.add_argument("--anneal-begin-step", type=int, default=None,
                        help="Radius anneal: env-step count at which linear "
                             "interpolation from start to end begins. Before "
                             "this, radius is held at start.")
    parser.add_argument("--anneal-end-step", type=int, default=None,
                        help="Radius anneal: env-step count at which radius "
                             "reaches end. After this, radius is held at end.")
    parser.add_argument("--mirror-augmentation", action="store_true",
                        help="Wrap the training env with MirrorWrapper "
                             "(per-episode L/R coin flip). Applies to BOTH "
                             "stage-1 and follow-on training envs (eval envs "
                             "stay unwrapped so best_model.zip reflects "
                             "raw-task performance). Breaks the lateralization "
                             "attractor by forcing the policy to generalize "
                             "across the creature's bilateral symmetry.")
    parser.add_argument("--vec-normalize", action="store_true",
                        help="Follow-on only: wrap the SubprocVecEnv with "
                             "VecNormalize (running mean/std on obs and "
                             "reward). Eliminates the proprio (~±10) vs "
                             "pixel ([0,1]) scale mismatch that suppresses "
                             "pixel-column gradients in the MlpPolicy first "
                             "layer. Eval env is VecNormalize with "
                             "training=False; stats synced from training env "
                             "before each eval. vec_normalize.pkl saved "
                             "alongside model checkpoints.")
    parser.add_argument("--stage1-anneal-max-radius-to", type=float, default=None,
                        help="Stage-1 only: target upper bound for a radius "
                             "anneal. Lower bound stays pinned at the initial "
                             "--stage1-radius min; upper bound grows linearly "
                             "from its initial value to this target over "
                             "--stage1-anneal-steps relative steps. Used to "
                             "bridge contact-reflex (fixed close spawns) → "
                             "locomotion (mixed-range spawns).")
    parser.add_argument("--stage1-anneal-steps", type=int, default=None,
                        help="Stage-1 radius anneal duration in env-steps, "
                             "counted RELATIVE to the start of this run (with "
                             "--stage1-init-from, 0 = first step after warm "
                             "start, not checkpoint's num_timesteps).")
    parser.add_argument("--closure-bonus", type=float, default=0.0,
                        help="Per-step reward `scale * max(0, Δd)` where Δd "
                             "is the body→ball distance reduction this step. "
                             "Markovian, non-telescoping, clipped at 0 so "
                             "only approach pays. Direct replacement for the "
                             "old --locomotion-bonus (body-forward velocity), "
                             "which required the policy to discover that "
                             "forward velocity correlates with approach. "
                             "Start around 0.1; at Δd=1cm/step and scale=0.1, "
                             "bonus is 0.001 per approach step — comparable "
                             "to PBRS α=0.1 but directly tied to progress.")
    args = parser.parse_args()
    # Parse ent_coef: accept 'auto' or a float string
    try:
        ent_coef = float(args.ent_coef)
    except ValueError:
        ent_coef = args.ent_coef  # leave as 'auto' etc.

    if args.stage in ("stage1", "both"):
        # Parse --stage1-radius: accepts "0.5" or "0.4,0.7"
        stage1_radius_override = None
        if args.stage1_radius is not None:
            parts = [p.strip() for p in args.stage1_radius.split(",")]
            if len(parts) == 1:
                r = float(parts[0])
                stage1_radius_override = (r, r)
            elif len(parts) == 2:
                stage1_radius_override = (float(parts[0]), float(parts[1]))
            else:
                parser.error("--stage1-radius must be 'x' or 'lo,hi'")
        if (args.stage1_anneal_max_radius_to is not None) ^ \
           (args.stage1_anneal_steps is not None):
            parser.error(
                "--stage1-anneal-max-radius-to and --stage1-anneal-steps "
                "must be set together."
            )
        train_stage1_v8(
            total_timesteps=args.stage1_steps,
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
            curriculum_stage=args.curriculum_stage,
            init_from=args.stage1_init_from,
            ent_coef=ent_coef,
            v9=args.v9,
            target_radius_override=stage1_radius_override,
            run_tag=args.run_tag,
            pbrs_alpha=args.pbrs_alpha,
            mirror_augmentation=args.mirror_augmentation,
            stage1_anneal_max_radius=args.stage1_anneal_max_radius_to,
            stage1_anneal_steps=args.stage1_anneal_steps,
            closure_bonus=args.closure_bonus,
        )
    if args.stage in ("followon", "both"):
        # --warmstart-radius (pinned float) and --followon-radius (range)
        # are mutually exclusive static overrides.
        if args.warmstart_radius is not None and args.followon_radius is not None:
            parser.error(
                "--warmstart-radius and --followon-radius are mutually "
                "exclusive (both set target_radius_override statically)."
            )
        radius_override = None
        if args.warmstart_radius is not None:
            radius_override = (args.warmstart_radius, args.warmstart_radius)
        elif args.followon_radius is not None:
            parts = [p.strip() for p in args.followon_radius.split(",")]
            if len(parts) == 1:
                r = float(parts[0])
                radius_override = (r, r)
            elif len(parts) == 2:
                radius_override = (float(parts[0]), float(parts[1]))
            else:
                parser.error("--followon-radius must be 'x' or 'lo,hi'")
        anneal_args = [
            args.anneal_radius_start, args.anneal_radius_end,
            args.anneal_begin_step, args.anneal_end_step,
        ]
        n_set = sum(a is not None for a in anneal_args)
        if n_set not in (0, 4):
            parser.error(
                "--anneal-* flags are all-or-nothing; set all four or none."
            )
        radius_anneal = None
        if n_set == 4:
            if radius_override is not None:
                parser.error(
                    "--warmstart-radius / --followon-radius and --anneal-* "
                    "flags are mutually exclusive (one overrides statically; "
                    "the other sets up a schedule)."
                )
            radius_anneal = {
                "radius_start": args.anneal_radius_start,
                "radius_end": args.anneal_radius_end,
                "anneal_begin_step": args.anneal_begin_step,
                "anneal_end_step": args.anneal_end_step,
            }
        train_followon_v8(
            stage1_path=args.stage1_checkpoint,
            total_timesteps=args.followon_steps,
            lambda_consistency=args.lambda_consistency,
            freeze_proprio=args.freeze_proprio,
            seed=args.seed,
            checkpoint_interval=args.checkpoint_interval,
            v9=args.v9,
            from_scratch=args.from_scratch,
            run_tag=args.run_tag,
            ent_coef=ent_coef,
            target_radius_override=radius_override,
            radius_anneal=radius_anneal,
            followon_init_from=args.followon_init_from,
            pbrs_alpha=args.pbrs_alpha,
            mirror_augmentation=args.mirror_augmentation,
            vec_normalize=args.vec_normalize,
        )
