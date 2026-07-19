"""
train_crawler.py — SAC training for the MIMo-based infant crawler.

Curriculum: --strength-scale controls muscle strength (0.0–1.0).
  Start weak (0.2–0.3) so the body settles naturally from standing before
  the policy learns to exploit full torque. Increase in later runs.

Usage:
    python -m alien_baby.crawler.train_crawler
    python -m alien_baby.crawler.train_crawler --strength-scale 0.3 --steps 500000
    python -m alien_baby.crawler.train_crawler --strength-scale 1.0 --init-from results/...
"""

import argparse
import faulthandler
import os
import pathlib
import datetime
import numpy as np

# Python 3.13 + MuJoCo + PyTorch on macOS has a finalization race that crashes
# in _datetime.delta_new during gc_collect_main. faulthandler gives a real
# Python traceback for any other crash; the os._exit(0) at end of __main__
# skips Python's broken finalization entirely once training completes.
faulthandler.enable()

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor


class _SafeSaveEvalCallback(EvalCallback):
    """EvalCallback that retries best-model saves on transient filesystem
    timeouts (macOS Spotlight / iCloud can briefly lock files during sync,
    producing Errno 60 inside zipfile.close). Without this wrapper the entire
    training process dies on a recoverable IO error."""
    def _on_step(self) -> bool:
        import time as _time
        for attempt in range(4):
            try:
                return super()._on_step()
            except (TimeoutError, OSError) as e:
                wait = 2 ** attempt
                print(f"[SafeSave] EvalCallback save failed "
                      f"(attempt {attempt+1}/4): {e!r} — retrying in {wait}s",
                      flush=True)
                _time.sleep(wait)
        print("[SafeSave] EvalCallback save failed 4x — continuing training "
              "without saving this best model", flush=True)
        return True


class EntropyAnnealCallback(BaseCallback):
    """Anneal SAC's fixed ent_coef from start to end over [anneal_start, anneal_end].

    Works only when ent_coef was passed as a float (not 'auto'/'auto_X'). Updates
    SAC's `ent_coef_tensor` attribute on every _on_step. The model's train()
    method reads this tensor when computing the actor's entropy bonus.
    """
    def __init__(self, start_ent, end_ent, anneal_start, anneal_end, verbose=1):
        super().__init__(verbose)
        self.start_ent = float(start_ent)
        self.end_ent = float(end_ent)
        self.anneal_start = int(anneal_start)
        self.anneal_end = int(anneal_end)
        self._last_logged_ent = None

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t < self.anneal_start:
            ent = self.start_ent
        elif t < self.anneal_end:
            span = max(1, self.anneal_end - self.anneal_start)
            ent = self.start_ent + (t - self.anneal_start) / span * (self.end_ent - self.start_ent)
        else:
            ent = self.end_ent
        import torch as th
        if hasattr(self.model, "ent_coef_tensor"):
            self.model.ent_coef_tensor = th.tensor(float(ent), device=self.model.device)
        if (self.verbose >= 1 and
                (self._last_logged_ent is None or abs(ent - self._last_logged_ent) > 0.01)):
            print(f"[EntAnneal] t={t} ent_coef={ent:.3f}", flush=True)
            self._last_logged_ent = ent
        return True


class CurriculumCallback(BaseCallback):
    """Ramp ball x-offset over training so AB must reach further from the cart
    path as training progresses. Asymmetric: ball1 to +x, ball2 to -x.

    Schedule (in env-timesteps, matching SB3's num_timesteps):
      [0, warmup_steps)            offset = 0           (warmup)
      [warmup_steps, ramp_end)     offset linear 0 -> final_offset
      [ramp_end, infinity)         offset = final_offset
    """

    def __init__(self, train_env, eval_env, warmup_steps, ramp_end_steps,
                 final_offset, ball_y=0.35, verbose=1):
        super().__init__(verbose)
        self._train_env = train_env
        self._eval_env  = eval_env
        self.warmup_steps  = int(warmup_steps)
        self.ramp_end_steps = int(ramp_end_steps)
        self.final_offset  = float(final_offset)
        self.ball_y        = float(ball_y)
        self._last_offset  = None

    def _on_training_start(self) -> None:
        self._update_envs(0.0)

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t < self.warmup_steps:
            offset = 0.0
        elif t < self.ramp_end_steps:
            span = max(1, self.ramp_end_steps - self.warmup_steps)
            offset = ((t - self.warmup_steps) / span) * self.final_offset
        else:
            offset = self.final_offset
        if self._last_offset is None or abs(offset - self._last_offset) > 0.005:
            self._update_envs(offset)
            self._last_offset = offset
        return True

    def _update_envs(self, offset: float) -> None:
        b1 = (offset, self.ball_y)
        b2 = (-offset, -self.ball_y)
        self._train_env.env_method("set_ball_positions", b1, b2)
        self._eval_env.env_method("set_ball_positions", b1, b2)
        if self.verbose >= 1:
            print(f"[Curriculum] t={self.num_timesteps} offset={offset:+.4f} "
                  f"ball1={b1} ball2={b2}", flush=True)

class LivenessGateCallback(BaseCallback):
    """The "realm of possibility" gate (see CLAUDE.md).

    A run must PROVE the creature moved. We accumulate the env's `body_motion`
    signal (mean joint speed + body/cart translation) and, at `gate_step`,
    check its running mean. If the creature has been effectively motionless
    (mean < `min_motion`), the run is VOID: we print "Nothing happened", write
    a marker file, and stop training. No conclusions are drawn from a corpse —
    a frozen run measures our broken setup, not the creature's ability.

    The gate always PRINTS the observed mean motion, pass or fail, so the
    threshold can be calibrated from real numbers rather than guessed.
    """

    def __init__(self, gate_step, min_motion, out_dir, verbose=1):
        super().__init__(verbose)
        self.gate_step = int(gate_step)
        self.min_motion = float(min_motion)
        self.out_dir = out_dir
        self._sum = 0.0
        self._cnt = 0
        self._fired = False

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "body_motion" in info:
                self._sum += float(info["body_motion"])
                self._cnt += 1
        if not self._fired and self.num_timesteps >= self.gate_step:
            self._fired = True
            mean_motion = self._sum / max(1, self._cnt)
            bar = "=" * 70
            if mean_motion < self.min_motion:
                msg = (f"VOID at {self.num_timesteps} steps: mean body_motion="
                       f"{mean_motion:.5f} < threshold {self.min_motion}. "
                       f"Nothing happened — the creature did not move. This run "
                       f"is void; fix the setup, do not analyze the numbers.")
                print(f"\n{bar}\n[LivenessGate] {msg}\n{bar}", flush=True)
                try:
                    (pathlib.Path(self.out_dir) /
                     "RUN_VOID_NO_MOVEMENT.txt").write_text(msg + "\n")
                except Exception:
                    pass
                return False  # stop training immediately
            print(f"[LivenessGate] PASS at {self.num_timesteps}: mean body_motion="
                  f"{mean_motion:.5f} >= threshold {self.min_motion} — the "
                  f"creature is alive; continuing.", flush=True)
        return True


from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES
from alien_baby.crawler.her_wrapper import HERCrawlerWrapper
from alien_baby.crawler.crawler_cnn_extractor import StereoCrawlerCNN, PIXEL_LATENT_DIM
from alien_baby.crawler.mimo_crawler_env import PROPRIO_DIM
from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv, CAM_H, CAM_W
from alien_baby.agents.micoa_architecture import (
    MICOAExtractor, MICOASAC, MICOAConfirmationCallback, LATENT_DIM,
    EgoTargetReplayBuffer,
)

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"


def make_env(rank, seed, strength_scale, spawn_cone_deg, max_steps, n_substeps,
             vision=False, approach_reward_scale=2.0, velocity_bonus_scale=0.05,
             fixed_ball_positions=None, random_start_orientation=False,
             memory_obs=False, stereo=True, her=False,
             cart_mode="none", cart_speed=0.15,
             hunger_mode="flat", hunger_base=0.05, hunger_rate=0.20, hunger_scale=500.0,
             hip_actuation=True, ball_radius=None, random_ball_box=None,
             ball_timeout_steps=None, ball_speed=0.0,
             random_ball_radius=None, random_ball_shape=None,
             target_obs=False, hand_success=False, pin_targets=False,
             near_contact_bonus_scale=0.0, near_contact_range=0.12,
             actuate_hands=False, contact_reward=None,
             action_mode="torque", spawn_radius=None, spawn_radius_hole=None,
             frame_stack=1, frame_stride=1,
             spawn_disc_lo=0.30, spawn_disc_hi=0.55, step_cost=-0.05,
             xml_path=None, crawl_pose=None, terminate_tilt_deg=None, tip_penalty=0.0,
             tilt_cost=0.0, decoy_ball=False, prism_offset_deg=0.0, gaze_spawn=False):
    def _init():
        if cart_mode != "none":
            # Phase G: cart substrate. HER not used; plain MimoCrawlerCartEnv.
            env = MimoCrawlerCartEnv(
                vision=vision,
                strength_scale=strength_scale,
                max_steps=max_steps,
                n_substeps=n_substeps,
                approach_reward_scale=approach_reward_scale,
                velocity_bonus_scale=velocity_bonus_scale,
                fixed_ball_positions=fixed_ball_positions,
                memory_obs=memory_obs,
                stereo=stereo,
                cart_speed=cart_speed,
                cart_mode=cart_mode,
                hunger_base=hunger_base,
                hunger_rate=hunger_rate,
                hunger_scale=hunger_scale,
                hip_actuation=hip_actuation,
                ball_radius=ball_radius,
                random_ball_box=random_ball_box,
                ball_timeout_steps=ball_timeout_steps,
                ball_speed=ball_speed,
                random_ball_radius=random_ball_radius,
                random_ball_shape=random_ball_shape,
                target_obs=target_obs,
                hand_success=hand_success,
                pin_targets=pin_targets,
                near_contact_bonus_scale=near_contact_bonus_scale,
                near_contact_range=near_contact_range,
                actuate_hands=actuate_hands,
                contact_reward=contact_reward,
                spawn_disc_lo=spawn_disc_lo,
                spawn_disc_hi=spawn_disc_hi,
            )
        else:
            env_kwargs = dict(
                vision=vision,
                strength_scale=strength_scale,
                spawn_cone_deg=spawn_cone_deg,
                max_steps=max_steps,
                n_substeps=n_substeps,
                approach_reward_scale=approach_reward_scale,
                velocity_bonus_scale=velocity_bonus_scale,
                fixed_ball_positions=fixed_ball_positions,
                random_start_orientation=random_start_orientation,
                memory_obs=memory_obs,
                stereo=stereo,
                action_mode=action_mode,
                spawn_radius=spawn_radius,
                spawn_radius_hole=spawn_radius_hole,
                frame_stack=frame_stack,
                frame_stride=frame_stride,
                step_cost=step_cost,
                xml_path=xml_path,
                crawl_pose=crawl_pose,
                terminate_tilt_deg=terminate_tilt_deg,
                tip_penalty=tip_penalty,
                tilt_cost=tilt_cost,
                target_obs=target_obs,
                decoy_ball=decoy_ball,
                prism_offset_deg=prism_offset_deg,
                gaze_spawn=gaze_spawn,
            )
            if her:
                # HERCrawlerWrapper instantiates MimoCrawlerEnv internally and adds
                # the goal-conditioned Dict observation.
                env = HERCrawlerWrapper(**env_kwargs)
            else:
                env = MimoCrawlerEnv(**env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    tag = args.run_tag or (
        f"crawler_s{args.strength_scale:.1f}_"
        + datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    )
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = RESULTS_DIR / (tag + "_best")
    best_dir.mkdir(parents=True, exist_ok=True)

    cart_mode = getattr(args, "cart_mode", "none")
    print(f"\n=== Crawler training: {tag} ===")
    print(f"  strength_scale={args.strength_scale}  steps={args.steps}")
    print(f"  spawn_cone={args.spawn_cone_deg}°  max_steps={args.max_steps}")
    print(f"  n_envs={args.n_envs}  seed={args.seed}")
    if cart_mode != "none":
        print(f"  cart_mode={cart_mode}  cart_speed={args.cart_speed}")
        print(f"  hunger_mode={args.hunger_mode}  hunger_base={args.hunger_base}"
              f"  hunger_rate={args.hunger_rate}  hunger_scale={args.hunger_scale}")
        print(f"  hip_actuation={args.hip_actuation}")
        print(f"  ball_speed={getattr(args, 'ball_speed', 0.0)}")
    print()

    # Shared kwargs for cart mode
    cart_kwargs = dict(
        cart_mode=cart_mode,
        cart_speed=getattr(args, "cart_speed", 0.15),
        hunger_mode=getattr(args, "hunger_mode", "flat"),
        hunger_base=getattr(args, "hunger_base", 0.05),
        hunger_rate=getattr(args, "hunger_rate", 0.20),
        hunger_scale=getattr(args, "hunger_scale", 500.0),
        hip_actuation=(getattr(args, "hip_actuation", "on") != "off"),
        ball_radius=getattr(args, "ball_radius", None),
        random_ball_box=getattr(args, "random_ball_box", None),
        ball_timeout_steps=getattr(args, "ball_timeout_steps", None),
        ball_speed=getattr(args, "ball_speed", 0.0),
        random_ball_radius=getattr(args, "random_ball_radius", None),
        random_ball_shape=getattr(args, "random_ball_shape", None),
        target_obs=getattr(args, "target_obs", False),
        hand_success=getattr(args, "hand_success", False),
        pin_targets=getattr(args, "pin_targets", False),
        near_contact_bonus_scale=getattr(args, "near_contact_bonus_scale", 0.0),
        near_contact_range=getattr(args, "near_contact_range", 0.12),
        actuate_hands=getattr(args, "actuate_hands", False),
        contact_reward=getattr(args, "contact_reward", None),
        spawn_disc_lo=getattr(args, "spawn_disc_lo", 0.30),
        spawn_disc_hi=getattr(args, "spawn_disc_hi", 0.55),
    )

    # Vision=True: MuJoCo Metal renderer fails in forked subprocesses on macOS.
    # Use DummyVecEnv (single process) for vision; SubprocVecEnv for no-vision.
    # --force-dummy-vec-env: also use DummyVecEnv without vision (controls).
    use_dummy = args.vision or getattr(args, "force_dummy_vec_env", False)
    VecEnvCls = DummyVecEnv if use_dummy else SubprocVecEnv

    # Phase XVI R49 new flags (default values preserve backward compat)
    _action_mode  = getattr(args, "action_mode",  "torque")
    _spawn_radius = getattr(args, "spawn_radius",  None)

    # Minimal-change crawl track (2026-07-02): widened body + crawl-ready default pose
    # + tip-termination. All None/0 unless the flags are set, so defaults are unchanged.
    _xml_path     = getattr(args, "xml_path", None)
    _crawl_pose   = CRAWL_POSES.get(getattr(args, "crawl_pose", None) or "", None)
    _term_tilt    = getattr(args, "terminate_tilt_deg", None)
    _tip_penalty  = getattr(args, "tip_penalty", 0.0)
    _crawl_kwargs = dict(xml_path=_xml_path, crawl_pose=_crawl_pose,
                         terminate_tilt_deg=_term_tilt, tip_penalty=_tip_penalty)

    # Training envs
    train_env = VecEnvCls([
        make_env(i, args.seed, args.strength_scale, args.spawn_cone_deg,
                 args.max_steps, args.n_substeps, vision=args.vision,
                 approach_reward_scale=args.approach_reward_scale,
                 velocity_bonus_scale=args.velocity_bonus_scale,
                 fixed_ball_positions=args.fixed_ball_positions,
                 random_start_orientation=args.random_start_orientation,
                 memory_obs=args.memory_obs,
                 stereo=not args.mono,
                 her=args.her,
                 action_mode=_action_mode,
                 spawn_radius=_spawn_radius,
                 step_cost=getattr(args, "step_cost", -0.05),
                 **_crawl_kwargs,
                 **cart_kwargs)
        for i in range(args.n_envs)
    ])
    # VecNormalize with Dict obs (HER) requires norm_obs_keys instead of norm_obs.
    # We normalize the "observation" key only; goals are left in world coords.
    if args.her:
        train_env = VecNormalize(train_env, norm_obs_keys=["observation"],
                                 norm_reward=True, clip_obs=10.0)
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval env (single, deterministic)
    eval_env = DummyVecEnv([
        make_env(0, args.seed + 1000, args.strength_scale, args.spawn_cone_deg,
                 args.max_steps, args.n_substeps, vision=args.vision,
                 approach_reward_scale=args.approach_reward_scale,
                 velocity_bonus_scale=args.velocity_bonus_scale,
                 fixed_ball_positions=args.fixed_ball_positions,
                 random_start_orientation=args.random_start_orientation,
                 memory_obs=args.memory_obs,
                 stereo=not args.mono,
                 her=args.her,
                 action_mode=_action_mode,
                 spawn_radius=_spawn_radius,
                 step_cost=getattr(args, "step_cost", -0.05),
                 **_crawl_kwargs,
                 **cart_kwargs)
    ])
    if args.her:
        eval_env = VecNormalize(eval_env, norm_obs_keys=["observation"],
                                norm_reward=False, clip_obs=10.0, training=False)
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)

    if args.init_from:
        print(f"  Loading checkpoint: {args.init_from}")
        model = SAC.load(
            args.init_from,
            env=train_env,
            device="mps" if args.mps else "cpu",
        )
        # Load matching VecNormalize stats if present
        vn_path = pathlib.Path(args.init_from).parent / "vec_normalize.pkl"
        if vn_path.exists():
            train_env = VecNormalize.load(str(vn_path), train_env)
            eval_env  = VecNormalize.load(str(vn_path), eval_env)
            eval_env.training = False
            print(f"  Loaded VecNormalize stats: {vn_path}")
        # Load replay buffer if a sibling .pkl exists. Without buffer
        # preservation, every warm-start so far collapsed in <10K steps
        # because the policy lost its exploration history. With buffer
        # loaded, SAC continues with the gradient support of past episodes.
        # Look in both the _best dir and the matching run dir.
        rb_candidates = [
            pathlib.Path(args.init_from).parent / "replay_buffer.pkl",
            (pathlib.Path(args.init_from).parent.parent /
             pathlib.Path(args.init_from).parent.name.replace("_best", "") /
             "replay_buffer.pkl"),
        ]
        for rb_path in rb_candidates:
            if rb_path.exists():
                model.load_replay_buffer(str(rb_path))
                print(f"  Loaded replay buffer: {rb_path}  "
                      f"(size={model.replay_buffer.size()})")
                break
        else:
            print("  WARNING: no replay_buffer.pkl found beside init_from. "
                  "Policy will warm-start with EMPTY buffer (likely to drift).")
    else:
        if args.vision and args.micoa:
            # Phase I: MICOA extractor. Each modality encodes to a Gaussian
            # over a shared latent Z; Product of Experts fuses them; the
            # downstream MLP sees [z_combined | mu_p | mu_v] (3 * LATENT_DIM).
            # share_features_extractor=True: one MICOAExtractor instance
            # serves both actor and critic. Required for MICOA's design:
            # the corner-forming pressure must be applied to a single set
            # of encoder weights, not two unrelated copies.
            effective_proprio_dim = (PROPRIO_DIM + (2 if args.memory_obs else 0)
                                     + (6 if getattr(args, "target_obs", False) else 0))
            in_channels = 3 if args.mono else 6
            policy_kwargs = dict(
                features_extractor_class=MICOAExtractor,
                features_extractor_kwargs=dict(
                    proprio_dim=effective_proprio_dim,
                    cam_h=CAM_H, cam_w=CAM_W,
                    in_channels=in_channels,
                    latent_dim=LATENT_DIM,
                ),
                share_features_extractor=True,
                net_arch=[256, 256],
            )
        elif args.vision:
            policy_kwargs = dict(
                features_extractor_class=StereoCrawlerCNN,
                features_extractor_kwargs=dict(
                    proprio_dim=PROPRIO_DIM,
                    pixel_latent_dim=PIXEL_LATENT_DIM,
                ),
                net_arch=[256, 256],
            )
        else:
            policy_kwargs = dict(net_arch=[256, 256])

        # Phase W: DroQ critic regularization.
        # When --droq is set, inject dropout_rate into policy_kwargs so
        # DroQSACPolicy can pass it through to DroQCritic. Actor unchanged.
        use_droq = getattr(args, "droq", False)
        if use_droq:
            from alien_baby.crawler.droq_policy import DroQSACPolicy
            policy_kwargs["dropout_rate"] = getattr(args, "dropout_rate", 0.01)
            print(f"  DroQ: dropout_rate={policy_kwargs['dropout_rate']}  "
                  f"utd={getattr(args, 'utd', 4)}")

        sac_kwargs = dict(
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef=args.ent_coef,
            target_entropy=args.target_entropy,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device="mps" if args.mps else "cpu",
        )

        # Phase W: higher update-to-data ratio (gradient_steps per env step).
        if use_droq:
            sac_kwargs["gradient_steps"] = getattr(args, "utd", 4)

        if args.her:
            # HER: dict obs → MultiInputPolicy; relabel failed transitions
            # using the "future" strategy from Andrychowicz et al. 2017.
            sac_kwargs["replay_buffer_class"]  = HerReplayBuffer
            sac_kwargs["replay_buffer_kwargs"] = dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            )
            model = SAC("MultiInputPolicy", train_env, **sac_kwargs)
        elif args.micoa:
            # Phase XVI: when aux_ball_decode is on, swap in EgoTargetReplayBuffer
            # so ego_xy targets from the env's info dict are stored alongside
            # each transition and can be sampled in MICOASAC.train().
            aux_decode = getattr(args, "aux_ball_decode", False)
            if aux_decode:
                sac_kwargs["replay_buffer_class"] = EgoTargetReplayBuffer
            model = MICOASAC("MlpPolicy", train_env,
                             micoa_beta=args.micoa_beta,
                             micoa_pred_beta=args.micoa_pred_beta,
                             micoa_pred_horizons=args.micoa_pred_horizons,
                             aux_ball_decode=aux_decode,
                             aux_ball_decode_coef=getattr(args, "aux_ball_decode_coef", 1.0),
                             aux_ball_decode_target=getattr(args, "aux_ball_decode_target", "ego_xy"),
                             **sac_kwargs)
        elif use_droq:
            model = SAC(DroQSACPolicy, train_env, **sac_kwargs)
        else:
            model = SAC("MlpPolicy", train_env, **sac_kwargs)

    # RND intrinsic reward (curiosity): wrap the TRAIN env only — the eval env
    # stays pure-task so eval still measures the real objective. set_env() so
    # both fresh and warm-started models pick up the wrapped env.
    if getattr(args, "rnd", False):
        from alien_baby.crawler.rnd_wrapper import RNDRewardWrapper
        train_env = RNDRewardWrapper(train_env, coef=args.rnd_coef, verbose=1)
        model.set_env(train_env)
        # Wrap eval with a pass-through RND (augment=False) so its wrapper stack
        # matches train — EvalCallback syncs VecNormalize stats by walking both
        # stacks in lockstep and asserts equal depth. Eval reward stays pure.
        eval_env = RNDRewardWrapper(eval_env, augment=False, verbose=0)
        print(f"  RND: ON  coef={args.rnd_coef} "
              f"(intrinsic curiosity reward added to the train env)")

    callback_list = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_interval // args.n_envs, 1),
            save_path=str(out_dir),
            name_prefix="crawler_ckpt",
            save_vecnormalize=True,
            verbose=1,
        ),
        _SafeSaveEvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(out_dir),
            eval_freq=max(args.checkpoint_interval // args.n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
        ),
    ]
    # "Realm of possibility" liveness gate (default ON). Voids any run where
    # the creature never moved, before any number from it can mean anything.
    if not getattr(args, "no_liveness_gate", False):
        callback_list.append(LivenessGateCallback(
            gate_step=args.liveness_gate_step,
            min_motion=args.liveness_min_motion,
            out_dir=str(out_dir),
            verbose=1,
        ))
        print(f"  liveness gate: ON  gate_step={args.liveness_gate_step}  "
              f"min_motion={args.liveness_min_motion}")
    if getattr(args, "micoa", False):
        callback_list.append(MICOAConfirmationCallback(log_freq=500))
        if args.micoa_pred_horizons:
            horizon_str = ", ".join(f"k={k}:β={b}" for k, b in args.micoa_pred_horizons)
        else:
            horizon_str = f"k=1:β={args.micoa_pred_beta}"
        print(f"  MICOA: beta_sym={args.micoa_beta}  "
              f"pred_horizons=[{horizon_str}]  "
              f"latent_dim={LATENT_DIM}  features_dim={LATENT_DIM*3}")
    if getattr(args, "ent_anneal_end_val", None) is not None:
        if args.ent_coef == "auto" or (isinstance(args.ent_coef, str) and args.ent_coef.startswith("auto")):
            raise ValueError("--ent-anneal-* requires a fixed --ent-coef, not 'auto'.")
        callback_list.append(EntropyAnnealCallback(
            start_ent=float(args.ent_coef),
            end_ent=args.ent_anneal_end_val,
            anneal_start=args.ent_anneal_start_step,
            anneal_end=args.ent_anneal_end_step,
            verbose=1,
        ))
        print(f"  entropy anneal: {args.ent_coef} -> {args.ent_anneal_end_val} "
              f"over [{args.ent_anneal_start_step}, {args.ent_anneal_end_step}]")
    if getattr(args, "curriculum", False):
        if cart_mode == "none":
            raise ValueError("--curriculum requires --cart-mode constant_velocity_bouncer "
                             "(it calls set_ball_positions, only defined on MimoCrawlerCartEnv).")
        callback_list.append(CurriculumCallback(
            train_env=train_env,
            eval_env=eval_env,
            warmup_steps=args.curriculum_warmup,
            ramp_end_steps=args.curriculum_ramp_end,
            final_offset=args.curriculum_final_offset,
            ball_y=args.curriculum_ball_y,
            verbose=1,
        ))
        print(f"  curriculum: warmup={args.curriculum_warmup}  "
              f"ramp_end={args.curriculum_ramp_end}  "
              f"final_offset={args.curriculum_final_offset}  "
              f"ball_y={args.curriculum_ball_y}")
    callbacks = CallbackList(callback_list)

    # progress_bar=False: tqdm.rich.__del__ segfaults on Python 3.13 + macOS
    # via Live.refresh → datetime.timedelta during interpreter teardown.
    # SB3's stdout logger still prints rollout/train stats every interval.
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        reset_num_timesteps=(args.init_from is None),
        progress_bar=False,
    )

    # Save final model + VecNormalize + replay buffer
    final_path = out_dir / "final_model"
    model.save(str(final_path))
    train_env.save(str(out_dir / "vec_normalize.pkl"))
    # Save replay buffer so a follow-on warm-start (--init-from) can resume
    # SAC training without losing the policy's exploration history. This
    # closes the bug that made R7/R11/R21 collapse despite starting from
    # a perfect policy. NOTE: replay buffers can be large (~hundreds of MB
    # for buffer_size=500K); skip saving in vision mode where pixel obs
    # bloat the buffer beyond useful size.
    if not args.vision:
        rb_path = out_dir / "replay_buffer.pkl"
        try:
            model.save_replay_buffer(str(rb_path))
            print(f"Replay buffer saved: {rb_path}  "
                  f"(size={model.replay_buffer.size()})")
        except Exception as e:
            print(f"Replay buffer save failed: {e}")
    print(f"\nFinal model: {final_path}.zip")
    print(f"Best model:  {best_dir}/best_model.zip")

    train_env.close()
    eval_env.close()

    # Done-sound chime so the user knows to come back to the session
    # (macOS only; silent on other platforms). See CLAUDE.md training-run rules.
    import subprocess, sys
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
        except Exception:
            pass

    return str(final_path) + ".zip"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-substeps", type=int, default=4,
                        help="Physics steps per policy step. 4 × 5ms = 20ms per action "
                             "(50Hz control). Increases effective episode length without "
                             "adding more policy decisions.")
    parser.add_argument("--strength-scale", type=float, default=0.7,
                        help="Muscle strength curriculum (0.1=newborn, 1.0=18-month). "
                             "Start at 0.3-0.4, increase in later runs.")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--spawn-cone-deg", type=float, default=180,
                        help="Half-angle of ball spawn cone (180=full forward hemisphere).")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Episode step limit (~30s at 20Hz).")
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--init-from", default=None,
                        help="Path to checkpoint .zip to continue from.")
    parser.add_argument("--run-tag", default=None,
                        help="Output directory name. Auto-generated if omitted.")
    parser.add_argument("--mps", action="store_true",
                        help="Use Apple MPS (Metal) GPU acceleration.")
    parser.add_argument("--vision", action="store_true",
                        help="Enable stereo vision (left_eye + right_eye cameras). "
                             "Uses StereoCrawlerCNN feature extractor.")
    parser.add_argument("--micoa", action="store_true",
                        help="Phase I: use MICOA architecture (Product of Experts + KL "
                             "agreement loss) instead of StereoCrawlerCNN concatenation. "
                             "Requires --vision. Each modality encodes to a Gaussian over "
                             "a shared latent space; PoE detects when they agree. "
                             "Incompatible with --her and --mono.")
    parser.add_argument("--micoa-beta", type=float, default=0.1,
                        help="Phase I: weight on the encoder-agreement KL loss. "
                             "Start 0.1; raise to 0.3 then 1.0 if kl_agreement flat. "
                             "0.0 disables the KL step (PoE still active).")
    parser.add_argument("--micoa-pred-beta", type=float, default=0.0,
                        help="Phase II: weight on the single-horizon (t+1) temporal "
                             "predictive KL loss. 0.0 = off. Ignored when "
                             "--micoa-pred-horizons is set.")
    parser.add_argument("--micoa-pred-horizons", default=None,
                        help="Phase III: multi-horizon temporal predictive loss. "
                             "Format: 'k1:beta1,k2:beta2,...' e.g. "
                             "'1:0.03,5:0.05,25:0.1,50:0.15'. Each horizon "
                             "contributes KL(v(t) || p(t+k).detach()) with its own "
                             "weight. Overrides --micoa-pred-beta.")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="SAC learning rate. DrQ-v2 recommends 1e-4 for pixel obs "
                             "(not 3e-4 which is for state obs). Higher causes actor to "
                             "over-exploit critic Q-estimates → collapse.")
    parser.add_argument("--buffer-size", type=int, default=500_000,
                        help="SAC replay buffer size. 500K caused cliff collapse when buffer "
                             "filled at ~125K steps; 100K rotates continuously.")
    parser.add_argument("--learning-starts", type=int, default=10_000,
                        help="Steps before SAC begins training. Larger = more diverse "
                             "initial data before first update.")
    parser.add_argument("--velocity-bonus-scale", type=float, default=0.05,
                        help="Reward per unit hip horizontal speed. 0.0 = no bribery; "
                             "creature must find its own reason to move.")
    parser.add_argument("--approach-reward-scale", type=float, default=2.0,
                        help="Scale for ball-approach reward (0.0 = off). "
                             "High values cause critic instability with pixel obs.")
    parser.add_argument("--ent-coef", default="auto",
                        help="SAC entropy coefficient. 'auto' = adaptive. "
                             "Fixed float (e.g. 0.2) holds exploration open.")
    parser.add_argument("--target-entropy", default="auto",
                        help="SAC target entropy for auto ent_coef. "
                             "'auto' = -dim(action). Float overrides (e.g. -10.0).")
    parser.add_argument("--fixed-ball-positions", default=None,
                        help="Phase C: comma-separated x,y pairs separated by semicolons. "
                             "Example: '0.7,0.0' for one fixed ball, "
                             "'0.7,0.0;0.0,0.7' for two. Overrides random spawn.")
    parser.add_argument("--random-start-orientation", action="store_true",
                        help="Phase C: rotate prone quaternion around world Z by a random "
                             "angle per reset. Creature spawns facing a random direction.")
    parser.add_argument("--memory-obs", action="store_true",
                        help="Phase C: append 2 binary flags (touched_ball1, touched_ball2) "
                             "to the proprio observation. Lets a stateless policy condition "
                             "on its own past contacts within an episode.")
    parser.add_argument("--target-obs", action="store_true",
                        help="Rung 0 (reach-to): append the hand->target error vector for "
                             "both hands (6 dims) to the observation, giving AB a SENSE of "
                             "where the target is. The TEST arm of the rung-0 pair.")
    parser.add_argument("--hand-success", action="store_true",
                        help="Rung 0: contact reward AND termination fire only on a HAND "
                             "touch (not incidental body/leg contact). Scores a real reach.")
    parser.add_argument("--pin-targets", action="store_true",
                        help="Rung 0: pin the target(s) in place each step so a bump can't "
                             "roll them away (a stable landmark).")
    parser.add_argument("--near-contact-bonus-scale", type=float, default=0.0,
                        help="Rung 0 endgame: dense per-step bonus = scale * "
                             "(near_contact_range - nearest_hand_dist) when in range, "
                             "to give the last few cm a gradient into contact.")
    parser.add_argument("--near-contact-range", type=float, default=0.12,
                        help="Distance (m) at which the near-contact bonus starts ramping.")
    parser.add_argument("--actuate-hands", action="store_true",
                        help="Rung 0 embodiment fix: load the body with actuated wrist + "
                             "finger joints (action 25->33, proprio 69->85) so AB can orient "
                             "its hand and grasp. Default off = original passive-hand body.")
    parser.add_argument("--contact-reward", type=float, default=None,
                        help="Override the sparse contact reward (default 200). Smaller "
                             "(e.g. 10) keeps the SAC value range sane to avoid the collapse.")
    parser.add_argument("--mono", action="store_true",
                        help="Use a single forward camera (left_eye) instead of stereo. "
                             "Halves the pixel observation dimension. The XML still shows "
                             "two visible eyes — only the camera input is mono.")
    parser.add_argument("--her", action="store_true",
                        help="Use Hindsight Experience Replay. Wraps the env to expose "
                             "goal-conditioned observations and switches SAC to use "
                             "HerReplayBuffer + MultiInputPolicy. Requires "
                             "--fixed-ball-positions (HER needs stable goals).")
    # Phase G: cart substrate flags
    parser.add_argument("--cart-mode", default="none",
                        choices=["none", "constant_velocity_bouncer", "steer"],
                        help="Phase G: 'constant_velocity_bouncer' mounts AB on a cart that "
                             "traverses the platform autonomously. 'none' = standard env. "
                             "Phase XVII: 'steer' = AB commands cart velocity via 2 appended "
                             "action dims; single-ball disc spawn. Default 'none'.")
    parser.add_argument("--cart-speed", type=float, default=0.15,
                        help="Phase G: cart speed in m/s (default 0.15).")
    parser.add_argument("--hunger-mode", default="flat",
                        choices=["flat", "quadratic"],
                        help="Phase G: hunger cost schedule. 'flat' = constant -BASE_COST/step. "
                             "'quadratic' = -(BASE + RATE*(steps_since_contact/SCALE)^2).")
    parser.add_argument("--hunger-base", type=float, default=0.05,
                        help="Phase G: base hunger cost per step (default 0.05).")
    parser.add_argument("--hunger-rate", type=float, default=0.20,
                        help="Phase G: quadratic hunger rate coefficient (default 0.20).")
    parser.add_argument("--hunger-scale", type=float, default=500.0,
                        help="Phase G: quadratic hunger scale (denominator, default 500).")
    parser.add_argument("--hip-actuation", default="on",
                        choices=["on", "off"],
                        help="Phase G: 'off' zeros trunk (0-4) and leg (15-24) actuator "
                             "commands so the policy cannot locomote — only arms and head.")
    parser.add_argument("--ball-radius", type=float, default=None,
                        help="Phase G: override ball sphere radius (XML default ~0.053). "
                             "Larger balls increase contact cross-section.")
    # Phase G curriculum: ramp ball x-offset to force learned arm extension
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable ball-offset curriculum (cart-substrate only). "
                             "Ramps ball x-offset from 0 to --curriculum-final-offset "
                             "over training; asymmetric (+x for ball1, -x for ball2).")
    parser.add_argument("--curriculum-warmup", type=int, default=1000,
                        help="Curriculum: env-timesteps held at offset=0 (default 1000).")
    parser.add_argument("--curriculum-ramp-end", type=int, default=5000,
                        help="Curriculum: env-timestep at which offset reaches "
                             "--curriculum-final-offset (default 5000).")
    parser.add_argument("--curriculum-final-offset", type=float, default=0.15,
                        help="Curriculum: final ball x-offset in meters (default 0.15).")
    parser.add_argument("--curriculum-ball-y", type=float, default=0.35,
                        help="Curriculum: ball y-coordinate (cart-path coordinate, default 0.35).")
    parser.add_argument("--ent-anneal-end-val", type=float, default=None,
                        help="Anneal ent_coef from its starting value to this final value.")
    parser.add_argument("--ent-anneal-start-step", type=int, default=15000,
                        help="Env-timestep at which entropy annealing starts.")
    parser.add_argument("--ent-anneal-end-step", type=int, default=30000,
                        help="Env-timestep at which entropy reaches end-val.")
    parser.add_argument("--ball-timeout-steps", type=int, default=None,
                        help="If set, each ball disappears (moves offscreen) after this "
                             "many env-steps if not touched. Forces AB to act within a "
                             "deadline. Used to test whether vision is genuinely needed.")
    parser.add_argument("--random-ball-box", default=None,
                        help="Make ball positions random per episode within a box "
                             "around the fixed_ball_positions anchors. Format: "
                             "'dx,dy' (half-ranges in meters). Example: '0.10,0.05'.")
    # Phase H: moving-balls flag
    parser.add_argument("--ball-speed", type=float, default=0.0,
                        help="Phase H: constant-velocity ball speed in m/s (default 0.0 = "
                             "stationary, bit-identical to all Phase G runs). Non-zero: each "
                             "ball gets a per-episode random heading and integrates "
                             "pos += vel * dt each env step, bouncing off platform edges. "
                             "Ball velocity is NOT exposed in proprio.")
    # Phase X: multi-size ball training (default OFF — existing runs bit-identical)
    parser.add_argument("--random-ball-radius", default=None,
                        help="Phase X: comma-separated list of ball radii to sample from "
                             "per episode (e.g. '0.040,0.053,0.075'). One radius is sampled "
                             "uniformly at reset and applied to both ball geoms. "
                             "When None (default), behavior is bit-identical to today.")
    # Phase X2: multi-shape ball training (default OFF — existing runs bit-identical)
    parser.add_argument("--random-ball-shape", default=None,
                        help="Phase X2: comma-separated list of MuJoCo primitive shape names "
                             "to sample from per episode. Valid: sphere,box,cylinder,ellipsoid,"
                             "capsule. One shape is sampled uniformly at reset and applied to "
                             "both ball geoms (geom_type + geom_size updated). "
                             "When None (default), behavior is bit-identical to today.")
    # Phase W: DroQ critic regularization flags (all default OFF — existing runs bit-identical)
    parser.add_argument("--droq", action="store_true",
                        help="Phase W: enable DroQ critic regularization. Adds LayerNorm + "
                             "Dropout after each hidden layer in the Q-networks (actor untouched). "
                             "Combines with --utd for a higher update-to-data ratio. "
                             "Default OFF — all existing runs are bit-identical without this flag.")
    parser.add_argument("--dropout-rate", type=float, default=0.01,
                        help="Phase W: Dropout rate for DroQ critic (default 0.01). "
                             "Ignored unless --droq is set.")
    parser.add_argument("--utd", type=int, default=4,
                        help="Phase W: update-to-data ratio — gradient steps per env step "
                             "(maps to SAC gradient_steps). Default 4. "
                             "Ignored unless --droq is set.")
    parser.add_argument("--force-dummy-vec-env", action="store_true",
                        help="Force DummyVecEnv (single-process) even without --vision. "
                             "Used for controlled comparisons against vision runs (which must "
                             "use DummyVecEnv on macOS due to Metal renderer + fork issues).")
    # Phase XVI: auxiliary ball-position decode loss flags (all default OFF)
    parser.add_argument("--aux-ball-decode", action="store_true", default=False,
                        help="Phase XVI: add a supervised MSE loss that forces the vision "
                             "encoder's latent (mu_v, 64-dim) to predict the ball's "
                             "egocentric position [x_ego, y_ego] = [ball_x-cart_x, "
                             "ball_y-cart_y]. Requires --micoa and --cart-mode. "
                             "Default OFF — all existing runs are bit-identical.")
    parser.add_argument("--aux-ball-decode-coef", type=float, default=1.0,
                        help="Phase XVI: coefficient on the aux ball-decode MSE loss "
                             "(default 1.0). Ignored unless --aux-ball-decode is set.")
    parser.add_argument("--aux-ball-decode-target", default="ego_xy",
                        help="Phase XVI: supervision target for the aux loss. "
                             "'ego_xy' = [ball_x-cart_x, ball_y-cart_y] in world coords "
                             "(default, matches probe_vision_latent.py definition). "
                             "Ignored unless --aux-ball-decode is set.")
    # Phase XVI R49: position-offset actuation (all default to torque for backward compat)
    parser.add_argument("--action-mode", default="torque",
                        choices=["torque", "position_offset"],
                        help="Phase XVI R49: 'torque' = original ctrl=action*strength (default, "
                             "all prior runs bit-identical). 'position_offset' = action∈[-1,1] "
                             "maps linearly onto each actuator's ctrlrange in mimo_crawler_pos.xml "
                             "(limbs are PD position controllers; head stays torque).")
    parser.add_argument("--spawn-radius", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Phase XVI R49: ball spawn radius range [min, max] in metres. "
                             "Overrides the env default of (0.5, 1.2). Use '0.18 0.30' for "
                             "a very close smoke test. Requires --cart-mode none (free-body env).")
    # Phase XVII: steer mode disc spawn radius flags
    parser.add_argument("--spawn-disc-lo", type=float, default=0.30,
                        help="Phase XVII steer: minimum ball spawn radius from platform center "
                             "in metres (default 0.30). Only used with --cart-mode steer.")
    parser.add_argument("--spawn-disc-hi", type=float, default=0.55,
                        help="Phase XVII steer: maximum ball spawn radius from platform center "
                             "in metres (default 0.55). Only used with --cart-mode steer.")
    # "Make it move": curiosity drive + stop punishing existence + liveness gate
    parser.add_argument("--rnd", action="store_true",
                        help="Add Random Network Distillation intrinsic reward (curiosity) to "
                             "the train env. Pays a novelty bonus that decays as states become "
                             "familiar — directly counters the freeze attractor. Flat Box obs "
                             "only (no --vision/--her). Default OFF (existing runs bit-identical).")
    parser.add_argument("--rnd-coef", type=float, default=1.0,
                        help="Weight on the normalized RND intrinsic reward (default 1.0). "
                             "Ignored unless --rnd is set.")
    parser.add_argument("--step-cost", type=float, default=-0.05,
                        help="Per-step penalty in the free-body env ('cost of existing'). "
                             "Default -0.05 (all prior runs bit-identical). Set 0.0 to stop "
                             "punishing existence so 'do nothing' is no longer the smart move. "
                             "Cart mode uses --hunger-base/--hunger-rate instead.")
    parser.add_argument("--liveness-gate-step", type=int, default=10_000,
                        help="'Realm of possibility' gate: env-timestep at which to check that "
                             "the creature actually moved (default 10000 = first checkpoint).")
    parser.add_argument("--liveness-min-motion", type=float, default=0.05,
                        help="Gate threshold on mean body_motion (joint speed + body/cart "
                             "translation). Below this = motionless = run VOID. Default 0.05 is "
                             "a calibration starting point; the gate prints the measured value.")
    parser.add_argument("--no-liveness-gate", action="store_true",
                        help="Disable the liveness gate (not recommended). The gate exists so a "
                             "run that never moved is voided, not analyzed.")
    # Minimal-change crawl track (2026-07-02)
    parser.add_argument("--xml-path", dest="xml_path", default=None,
                        help="Override body XML (e.g. alien_baby/crawler/mimo_crawler_pos_wide.xml "
                             "for the widened-clamp body).")
    parser.add_argument("--crawl-pose", dest="crawl_pose", default=None,
                        choices=list(CRAWL_POSES.keys()),
                        help="Re-centre the position-offset ranges on a crawl-ready default pose "
                             "(offsets around a locomotion-adjacent pose, per 'A Walk in the Park'). "
                             "'arms_fwd' = commando/belly-crawl pose. Requires --action-mode "
                             "position_offset and the widened body via --xml-path.")
    parser.add_argument("--terminate-tilt-deg", dest="terminate_tilt_deg", type=float, default=None,
                        help="End the episode (with --tip-penalty) if the body's dorsal axis tilts "
                             "past this many degrees from world-up. Kills the tip-over/roll-and-slide "
                             "exploit (DeepMimic-style early termination). Off by default.")
    parser.add_argument("--tip-penalty", dest="tip_penalty", type=float, default=0.0,
                        help="Reward added when the tip-termination fires (use a negative value).")
    args = parser.parse_args()
    # Convert numeric strings to float
    if args.ent_coef != "auto":
        args.ent_coef = float(args.ent_coef)
    if args.target_entropy != "auto":
        args.target_entropy = float(args.target_entropy)
    # RND preconditions: flat Box obs only (the no-vision crawler).
    if args.rnd and (args.vision or args.her):
        raise SystemExit("--rnd supports flat Box observations only; not "
                         "compatible with --vision or --her.")
    # MICOA preconditions: vision required; HER + MICOA out of scope for R36.
    if args.micoa:
        if not args.vision:
            raise SystemExit("--micoa requires --vision (vision encoder needs pixel obs).")
        if args.her:
            raise SystemExit("--micoa + --her not supported (HER uses Dict obs; "
                             "MICOAExtractor expects flat Box obs).")
    # Parse --micoa-pred-horizons "k1:β1,k2:β2,..." into a list of tuples
    if getattr(args, "micoa_pred_horizons", None):
        pairs = []
        for chunk in args.micoa_pred_horizons.split(","):
            k_s, b_s = chunk.split(":")
            pairs.append((int(k_s), float(b_s)))
        args.micoa_pred_horizons = pairs
    else:
        args.micoa_pred_horizons = None
    # Parse fixed-ball-positions string into list of (x, y) tuples
    if args.fixed_ball_positions:
        balls = []
        for chunk in args.fixed_ball_positions.split(";"):
            x, y = chunk.split(",")
            balls.append((float(x), float(y)))
        args.fixed_ball_positions = balls
    # Parse random-ball-box "dx,dy"
    if args.random_ball_box:
        dx, dy = args.random_ball_box.split(",")
        args.random_ball_box = (float(dx), float(dy))
    # Parse random-ball-radius "r1,r2,r3" into list of floats
    if args.random_ball_radius:
        args.random_ball_radius = [float(r) for r in args.random_ball_radius.split(",")]
    # Parse random-ball-shape "sphere,box,cylinder" into list of strings
    if args.random_ball_shape:
        args.random_ball_shape = [s.strip() for s in args.random_ball_shape.split(",")]
    # Curriculum: when enabled and no explicit fixed_ball_positions given, seed
    # the envs at the warmup positions (offset=0). The CurriculumCallback will
    # then push updated positions on every step.
    if args.curriculum and not args.fixed_ball_positions:
        args.fixed_ball_positions = [
            (0.0, args.curriculum_ball_y),
            (0.0, -args.curriculum_ball_y),
        ]
    # Always exit via os._exit so we skip Python's broken finalization on
    # macOS + Python 3.13 + MuJoCo + PyTorch, whether train() succeeded or
    # raised. Without this, any uncaught exception triggers the same
    # _datetime.delta_new segfault during interpreter teardown and we lose
    # the real traceback to a misleading crash report.
    exit_code = 0
    try:
        train(args)
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    os._exit(exit_code)
