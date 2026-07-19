"""
train_head_search.py — HEAD-SEARCH phase: the first task where vision is behaviorally
NECESSARY. Ball spawns on a WIDE cone (±68°, beyond the forward-crawl reach envelope, so
the diagnostic shows the bearing is worth ~+20 pts) and OUTSIDE the static ±22° camera view,
so the policy must actively SEARCH — turn its head (new head_yaw actuator = head_tilt_side,
which yaws when prone) and/or its body — to bring the ball into view, read its direction from
pixels, and crawl to it.

This is the setup the ±22° confound could not provide: a winnable task (ball bringable into
view by a ±60° head-yaw, precheck-confirmed) where forward-crawl FAILS, so vision can finally
convert its R²=0.84 directional read into a real contact gain ("reinforced" testable at last).

Vision encoder warm-started from Stage A (bearing_cnn.pt) so vision isn't learned from scratch;
gait + search + steering ARE learned by reward (the hard part). Entropy drives search
exploration. macOS: vision => DummyVecEnv. Honest expectation: this is the hard follow-on; a
first run is exploratory.

Usage:
  PYTHONPATH=<repo> python -u -m alien_baby.crawler.train_head_search \
    --steps 2000000 --run-tag head_search_v1 --warmstart-cnn alien_baby/results/stage_a_v1/bearing_cnn.pt
"""
import argparse
import pathlib
import subprocess

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from stable_baselines3.common.callbacks import BaseCallback

from alien_baby.crawler.mimo_crawler_env import CRAWL_POSES
from alien_baby.crawler.train_crawler import make_env, LivenessGateCallback, _SafeSaveEvalCallback
from alien_baby.crawler.crawler_cnn_extractor import StereoCrawlerCNN

RESULTS = pathlib.Path(__file__).parent.parent / "results"
XML = "alien_baby/crawler/mimo_crawler_pos_wide_hs.xml"   # has the head_yaw actuator

# Search-shaping curriculum: (fraction of total steps, spawn_cone_deg). Start narrow so the
# policy masters gait+approach where forward-crawl works, then widen so the ball moves off the
# static view and SEARCH becomes necessary — build the skills in sequence, not all at once.
CONE_SCHEDULE = [(0.0, 44.0), (0.15, 68.0), (0.35, 90.0), (0.60, 136.0)]


class CurriculumConeCallback(BaseCallback):
    """Widen spawn_cone_deg on train + eval envs as training progresses."""

    def __init__(self, schedule, total_steps, eval_env, verbose=0):
        super().__init__(verbose)
        self.schedule = schedule
        self.total = total_steps
        self.eval_env = eval_env
        self._current = None

    def _target_cone(self):
        frac = self.num_timesteps / max(1, self.total)
        cone = self.schedule[0][1]
        for f, c in self.schedule:
            if frac >= f:
                cone = c
        return cone

    def _on_step(self):
        cone = self._target_cone()
        if cone != self._current:
            self._current = cone
            self.training_env.set_attr("spawn_cone_deg", cone)
            self.eval_env.set_attr("spawn_cone_deg", cone)
            print(f"[curriculum] step {self.num_timesteps}: spawn cone -> +/-{cone/2:.0f}")
        return True


class MismatchAuxCallback(BaseCallback):
    """Seen-vs-contacted visual recalibration signal (David 2026-07-11).

    Trains the encoder's bearing head (theta_vis) to agree with the TRUE ego-bearing the
    body confirms by reaching the ball — via a SEPARATE optimizer over the vision encoder
    only. PPO's optimizer is rebuilt to train the policy/value heads only, so reward can
    never reshape the eye to enable a non-visual shortcut. Honesty: with --gate-contact,
    the eye learns ONLY from episodes where the body actually touched the red ball (the
    location was genuinely revealed), not from an omniscient oracle.

    Under a +offset lens the pixels show the ball at bearing+offset while this target stays
    at the true bearing, so minimizing the mismatch forces the encoder to subtract the
    offset: that is recalibration.
    """

    def __init__(self, aux_opt, coef, gate_contact, n_envs, device, epochs=4,
                 max_samples=4096, verbose=0):
        super().__init__(verbose)
        self.aux_opt = aux_opt
        self.coef = coef
        self.gate_contact = gate_contact
        self.n_envs = n_envs
        self.device = device
        self.epochs = epochs
        self.max_samples = max_samples
        self._ep_obs = [[] for _ in range(n_envs)]
        self._ep_bear = [[] for _ in range(n_envs)]
        self._X, self._Y = [], []

    def _on_step(self):
        infos = self.locals["infos"]
        new_obs = self.locals["new_obs"]
        dones = self.locals["dones"]
        for i in range(self.n_envs):
            info = infos[i]
            obs_i = info["terminal_observation"] if (dones[i] and "terminal_observation" in info) else new_obs[i]
            bear = info.get("ball1_bearing", None)
            if bear is not None:
                self._ep_obs[i].append(np.asarray(obs_i, dtype=np.float32))
                self._ep_bear[i].append(float(bear))
            if dones[i]:
                keep = (not self.gate_contact) or bool(info.get("touched_ball1", False))
                if keep and self._ep_obs[i]:
                    self._X.extend(self._ep_obs[i]); self._Y.extend(self._ep_bear[i])
                self._ep_obs[i] = []; self._ep_bear[i] = []
        return True

    def _on_rollout_end(self):
        if not self._X:
            self.model.logger.record("mismatch/n_samples", 0)
            return
        fe = self.model.policy.features_extractor
        X = np.asarray(self._X, dtype=np.float32); Y = np.asarray(self._Y, dtype=np.float32)
        if X.shape[0] > self.max_samples:
            sel = np.random.choice(X.shape[0], self.max_samples, replace=False)
            X, Y = X[sel], Y[sel]
        Xt = torch.as_tensor(X, device=self.device)
        th = torch.as_tensor(Y, device=self.device)
        target = torch.stack([torch.sin(th), torch.cos(th)], dim=1)
        B, bs, last = Xt.shape[0], 512, float("nan")
        for _ in range(self.epochs):
            perm = torch.randperm(B, device=self.device)
            for s in range(0, B, bs):
                idx = perm[s:s + bs]
                self.aux_opt.zero_grad()
                pred = fe.bearing_pred(Xt[idx])
                pred = pred / pred.norm(dim=1, keepdim=True).clamp_min(1e-6)
                loss = self.coef * ((pred - target[idx]) ** 2).sum(1).mean()
                loss.backward()
                self.aux_opt.step()
                last = float(loss.detach())
        self.model.logger.record("mismatch/aux_loss", last)
        self.model.logger.record("mismatch/n_samples", B)
        self._X, self._Y = [], []
        return


class MismatchPPO(PPO):
    """PPO whose gradient update trains the policy/value HEADS only — the vision encoder
    is frozen for the duration of PPO.train(), so reward can never reshape the eye. The eye
    is driven solely by the seen-vs-contacted mismatch aux loss (MismatchAuxCallback). This
    keeps the default optimizer structure intact, so saved checkpoints load normally with
    plain PPO.load (unlike replacing policy.optimizer, which breaks reconstruction)."""

    def train(self):
        fe = self.policy.features_extractor
        for p_ in fe.parameters():
            p_.requires_grad_(False)
        try:
            super().train()          # updates heads only; encoder grads are None -> Adam skips
        finally:
            for p_ in fe.parameters():
                p_.requires_grad_(True)


def _chime():
    try:
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], timeout=5)
    except Exception:
        pass


def build_envs(args):
    ck = dict(
        vision=True, stereo=True, target_obs=False, approach_reward_scale=10.0,
        velocity_bonus_scale=0.0, action_mode="position_offset",
        spawn_radius=tuple(args.spawn_radius),
        spawn_radius_hole=tuple(args.spawn_radius_hole) if args.spawn_radius_hole else None,
        step_cost=0.0, xml_path=args.xml,
        prism_offset_deg=args.prism_offset,
        crawl_pose=CRAWL_POSES["arms_fwd"], terminate_tilt_deg=50.0, tip_penalty=-5.0,
        tilt_cost=args.tilt_cost,
        decoy_ball=args.decoy, gaze_spawn=args.gaze_spawn,
    )
    train_env = DummyVecEnv([
        make_env(i, args.seed, 0.7, args.spawn_cone_deg, args.max_steps, 4, **ck)
        for i in range(args.n_envs)
    ])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_obs=10.0)
    eval_env = DummyVecEnv([
        make_env(0, args.seed + 1000, 0.7, args.spawn_cone_deg, args.max_steps, 4, **ck)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_obs=10.0,
                            training=False)
    return train_env, eval_env


def warmstart_encoder(model, path):
    """Load Stage A bearing CNN conv-trunk weights into the policy's StereoCrawlerCNN."""
    sd = torch.load(path, map_location="cpu")
    conv = {k[len("cnn."):]: v for k, v in sd.items() if k.startswith("cnn.")}
    fe = model.policy.features_extractor
    missing = fe.cnn.load_state_dict(conv, strict=False)
    print(f"[head_search] warm-started encoder conv from {path} ({missing})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--spawn-cone-deg", type=float, default=136.0)   # +/-68, vision-necessary
    p.add_argument("--spawn-radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--spawn-radius-hole", type=float, nargs=2, default=None,
                   help="held-out (lo hi) distance gap the ball never spawns in "
                        "(train the ends, eval the middle — generalization test)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.01)   # >0 to drive search exploration
    p.add_argument("--run-tag", default="head_search_v1")
    p.add_argument("--device", default="mps")
    p.add_argument("--warmstart-cnn", default=None)
    p.add_argument("--curriculum", action="store_true",
                   help="search-shaping: widen the spawn cone over training (CONE_SCHEDULE)")
    p.add_argument("--tilt-cost", type=float, default=0.0,
                   help="graded posture penalty as body tilts toward the tip limit (0=off)")
    p.add_argument("--decoy", action="store_true",
                   help="two-ball discrimination: blue decoy spawns each episode; touching it "
                        "ends the episode with a penalty (removes the touch-search escape)")
    p.add_argument("--init-model", default=None,
                   help="continue training from this checkpoint .zip (extension run; "
                        "use a fixed cone, not --curriculum)")
    p.add_argument("--xml", default=XML,
                   help="env XML (prism adaptation uses mimo_crawler_pos_wide_prism.xml)")
    p.add_argument("--prism-offset", type=float, default=0.0,
                   help="train under a fixed prism displacement (whole-field with --decoy)")
    p.add_argument("--gaze-spawn", action="store_true",
                   help="corrected prism: place the ball so the VISIBLE target (ghost/real) is in "
                        "the gaze cone (see PRISM_GAZE_RELATIVE_PROPOSAL.md). Default off = old spawn.")
    p.add_argument("--gain-field", action="store_true",
                   help="GAIN-FIELD bearing head: FiLM-modulate the pixel latent by head-pose "
                        "(proprio) so theta_vis maps retinal->body frame. Panel-recommended for the "
                        "corrected prism (Pouget&Sejnowski; Salinas&Abbott; Taylor 8.12). Default off.")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="with --init-model: freeze the vision CNN; locates adaptation "
                        "in the policy heads vs the encoder")
    p.add_argument("--mismatch-coef", type=float, default=0.0,
                   help="seen-vs-contacted visual recalibration aux loss weight (0=off). "
                        "Trains the eye's bearing head via a SEPARATE optimizer; PPO trains "
                        "only the policy heads. See MismatchAuxCallback.")
    p.add_argument("--aux-lr", type=float, default=3e-4, help="lr for the mismatch aux optimizer")
    p.add_argument("--no-gate-contact", action="store_true",
                   help="disable contact-gating of the mismatch loss (train the eye on ALL "
                        "steps, incl. episodes that never reached the ball = oracle-ish). "
                        "Default OFF: honest, contact-confirmed episodes only.")
    args = p.parse_args()

    tag = args.run_tag
    # With the curriculum, envs START at the first (narrow) cone; the callback widens them.
    if args.curriculum:
        args.spawn_cone_deg = CONE_SCHEDULE[0][1]
    print(f"\n=== HEAD-SEARCH training: {tag} ===")
    print(f"  steps={args.steps} n_envs={args.n_envs} start_cone=+/-{args.spawn_cone_deg/2:.0f} "
          f"curriculum={args.curriculum} ent={args.ent_coef} device={args.device}")

    train_env, eval_env = build_envs(args)

    def _fresh_ppo():
        cls = MismatchPPO if args.mismatch_coef > 0 else PPO
        return cls(
            "MlpPolicy", train_env,
            learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
            n_epochs=args.n_epochs, gamma=0.99, gae_lambda=0.95, ent_coef=args.ent_coef,
            clip_range=0.2, verbose=1, seed=args.seed, device=args.device,
            policy_kwargs=dict(
                features_extractor_class=StereoCrawlerCNN,
                features_extractor_kwargs=dict(gain_field=args.gain_field),
                net_arch=[256, 256],
            ),
        )

    if args.mismatch_coef > 0 and args.init_model:
        # The new bearing_head means the policy has params the old checkpoint lacks, so a
        # strict PPO.load would fail: build fresh and load matching weights non-strict
        # (bearing_head stays freshly initialized; everything else = the init model).
        from stable_baselines3.common.save_util import load_from_zip_file
        model = _fresh_ppo()
        _, params, _ = load_from_zip_file(args.init_model, device=args.device)
        res = model.policy.load_state_dict(params["policy"], strict=False)
        print(f"[mismatch] init from {args.init_model} non-strict: "
              f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    elif args.init_model:
        # Extension run: continue training an existing checkpoint. Use with a FIXED
        # cone (no --curriculum): the curriculum callback keys off num_timesteps,
        # which restarts at 0 here and would re-narrow the cone.
        model = PPO.load(args.init_model, env=train_env, device=args.device)
        model.ent_coef = args.ent_coef
        print(f"[head_search] continuing from {args.init_model}")
        if args.freeze_encoder:
            n = 0
            for p_ in model.policy.features_extractor.parameters():
                p_.requires_grad = False; n += 1
            print(f"[head_search] froze features_extractor ({n} tensors) — "
                  f"adaptation must happen in the policy/value heads")
    else:
        model = _fresh_ppo()
    if args.warmstart_cnn and not args.init_model:
        warmstart_encoder(model, args.warmstart_cnn)

    # Mismatch aux: PPO trains the policy/value HEADS only; a separate optimizer trains the
    # vision encoder (incl. bearing head) via the seen-vs-contacted loss. Reward can then
    # never reshape the eye toward the non-visual shortcut — the whole point of the design.
    aux_cb = None
    if args.mismatch_coef > 0:
        fe = model.policy.features_extractor
        aux_opt = torch.optim.Adam(list(fe.parameters()), lr=args.aux_lr)
        aux_cb = MismatchAuxCallback(aux_opt, args.mismatch_coef,
                                     gate_contact=not args.no_gate_contact,
                                     n_envs=args.n_envs, device=args.device)
        print(f"[mismatch] ON coef={args.mismatch_coef} aux_lr={args.aux_lr} "
              f"gate_contact={not args.no_gate_contact}; encoder frozen in PPO.train (heads only), "
              f"eye driven by aux over {len(list(fe.parameters()))} encoder tensors")

    best_dir = str(RESULTS / f"{tag}_best")
    ckpt_dir = str(RESULTS / tag)
    callbacks = [
        CheckpointCallback(save_freq=max(1, 50_000 // args.n_envs), save_path=ckpt_dir,
                           name_prefix="ckpt", save_vecnormalize=True),
        _SafeSaveEvalCallback(eval_env, best_model_save_path=best_dir, log_path=ckpt_dir,
                              eval_freq=max(1, 25_000 // args.n_envs), n_eval_episodes=10,
                              deterministic=True, render=False),
        LivenessGateCallback(gate_step=10_000, min_motion=0.05, out_dir=ckpt_dir),
    ]
    if args.curriculum:
        callbacks.append(CurriculumConeCallback(CONE_SCHEDULE, args.steps, eval_env))
    if aux_cb is not None:
        callbacks.append(aux_cb)
    model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks), progress_bar=False)
    model.save(str(RESULTS / f"{tag}_final"))
    train_env.save(str(RESULTS / tag / "vec_normalize.pkl"))
    print(f"=== head-search done: {tag} ===")
    _chime()


if __name__ == "__main__":
    main()
