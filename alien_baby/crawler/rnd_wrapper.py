"""
rnd_wrapper.py — Random Network Distillation intrinsic reward (Burda et al. 2018).

Wraps a VecEnv and adds a curiosity bonus to each step's reward equal to the
prediction error of a learned "predictor" net against a fixed random "target"
net, evaluated on the observation. Novel states (high error) pay a bonus;
familiar states (low error) don't.

Why this exists: it directly attacks the freeze attractor (see GLOSSARY.md).
A motionless body sees the same observation forever, the predictor learns it,
the intrinsic reward decays to ~0 — so stillness stops paying and moving does.
This is the "turn up a knob to make it move" lever: curiosity, not task reward.

Scope: flat Box observations only (the no-vision crawler). Dict/HER obs raise a
clear error — RND over pixels/dicts is a separate, heavier design.

Off-policy caveat: SAC stores the intrinsic reward in its replay buffer at
collection time; as the predictor learns, the *stored* bonus for old states
goes stale (stays high). That is acceptable here — the bonus is strongest early,
which is exactly when we need to break the freeze.
"""

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.running_mean_std import RunningMeanStd


class _MLP(nn.Module):
    """Small fixed-width MLP. Same shape for target (frozen) and predictor."""
    def __init__(self, in_dim, hidden=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RNDRewardWrapper(VecEnvWrapper):
    """Augment env reward with a normalized RND novelty bonus.

    reward_out = reward_in + coef * (intrinsic / running_std(intrinsic))

    Observations fed to the RND nets are normalized by a running mean/std and
    clipped to +/-5 (Burda et al.), independent of any VecNormalize on the
    policy's observations.
    """

    def __init__(self, venv, coef=1.0, lr=1e-4, out_dim=64,
                 device="cpu", verbose=1):
        super().__init__(venv)
        obs_space = venv.observation_space
        if not hasattr(obs_space, "shape") or len(obs_space.shape) != 1:
            raise ValueError(
                "RNDRewardWrapper supports flat Box observations only; got "
                f"{obs_space}. (Dict/HER or image obs are out of scope.)")
        self.in_dim = int(obs_space.shape[0])
        self.coef = float(coef)
        self.device = th.device(device)

        self.target = _MLP(self.in_dim, out_dim=out_dim).to(self.device)
        self.predictor = _MLP(self.in_dim, out_dim=out_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = th.optim.Adam(self.predictor.parameters(), lr=lr)

        self.obs_rms = RunningMeanStd(shape=(self.in_dim,))
        self.rew_rms = RunningMeanStd(shape=())
        self.verbose = verbose
        self._n = 0
        self._intrinsic_ema = None

    # VecEnvWrapper requires reset() and step_wait().
    def reset(self):
        return self.venv.reset()

    def _normalize_obs(self, obs):
        self.obs_rms.update(obs)
        norm = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(norm, -5.0, 5.0).astype(np.float32)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        norm_obs = self._normalize_obs(np.asarray(obs, dtype=np.float32))
        t = th.as_tensor(norm_obs, device=self.device)

        # Per-sample prediction error = intrinsic reward (before normalization).
        with th.no_grad():
            tgt = self.target(t)
        pred = self.predictor(t)
        err = ((pred - tgt) ** 2).mean(dim=1)            # [n_envs]
        intrinsic = err.detach().cpu().numpy()

        # Normalize intrinsic by its running std (keeps the bonus scale stable).
        self.rew_rms.update(intrinsic)
        norm_intrinsic = intrinsic / np.sqrt(self.rew_rms.var + 1e-8)

        # Train the predictor toward the target on this batch (one step).
        loss = err.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        rews = np.asarray(rews, dtype=np.float32) + self.coef * norm_intrinsic.astype(np.float32)

        # Telemetry: smoothed mean intrinsic, and per-step value in info.
        self._n += 1
        m = float(norm_intrinsic.mean())
        self._intrinsic_ema = (m if self._intrinsic_ema is None
                               else 0.99 * self._intrinsic_ema + 0.01 * m)
        if self.verbose and self._n % 2000 == 0:
            print(f"[RND] step={self._n} intrinsic(norm)~{self._intrinsic_ema:.4f} "
                  f"coef={self.coef}", flush=True)
        for info in infos:
            info["rnd_intrinsic"] = m

        return obs, rews, dones, infos
