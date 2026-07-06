"""
residual_vision_policy.py — Stage C: additive zero-init residual vision head on a
FROZEN blind crawler (the "No More Blind Spots" architecture, Duan et al. 2025,
arXiv:2508.11929; residual RL, Johannink 2019; zero-init from ControlNet, Zhang 2023).

  action_mean = base(proprio)   [FROZEN blind-base actor, no grad]
              + residual(pixels) [trainable CNN, final layer ZERO-INIT]

At init the residual outputs 0, so the policy == the blind base and the crawl gait is
preserved BY CONSTRUCTION (drift = 0). PPO reward then grows the residual, which can
only ADD steering from vision — it cannot destroy the frozen proprioceptive gait.
This is the literal "reinforced, not destroyed" architecture.

Obs layout (vision=True, stereo=True, target_obs=False):
  [0:69]  proprio     [69:]  stereo pixels (2*32*32*3), already /255

The blind base was trained under VecNormalize(norm_obs), so its actor expects
normalized proprio; we bake that normalization (mean/std buffers) into the policy and
run the Stage C env WITHOUT obs normalization (norm_obs=False) so the pixel block is
never corrupted.
"""
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution

from alien_baby.crawler.mimo_crawler_env import PROPRIO_DIM, CAM_H, CAM_W


def _conv_trunk():
    return nn.Sequential(
        nn.Conv2d(6, 32, 3, stride=2), nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        nn.Flatten(),
    )


def _pixels_to_img(pixels_flat, cam_h, cam_w):
    half = cam_h * cam_w * 3
    l = pixels_flat[:, :half].view(-1, cam_h, cam_w, 3).permute(0, 3, 1, 2)
    r = pixels_flat[:, half:].view(-1, cam_h, cam_w, 3).permute(0, 3, 1, 2)
    return torch.cat([l, r], dim=1)


class ResidualVisionPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 base_model_path=None, base_mean=None, base_std=None, base_clip=10.0,
                 proprio_dim=PROPRIO_DIM, cam_h=CAM_H, cam_w=CAM_W,
                 log_std_init=-1.0, **kwargs):
        self._base_model_path = base_model_path
        self._base_mean = np.asarray(base_mean, np.float32)
        self._base_std = np.asarray(base_std, np.float32)
        self._base_clip = float(base_clip)
        self._proprio_dim = proprio_dim
        self._cam_h, self._cam_w = cam_h, cam_w
        self._log_std_init = log_std_init
        # Let the parent build its defaults (unused) then _build overrides ours.
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        action_dim = int(np.prod(self.action_space.shape))
        self.action_dist = DiagGaussianDistribution(action_dim)

        # Frozen blind base (its actor maps normalized proprio -> action mean).
        base = PPO.load(self._base_model_path, device="cpu").policy
        for p in base.parameters():
            p.requires_grad_(False)
        base.eval()
        self.base = base

        # Proprio normalization (from the blind base's VecNormalize), as buffers.
        self.register_buffer("pmean", torch.as_tensor(self._base_mean))
        self.register_buffer("pstd", torch.as_tensor(self._base_std))

        # Trainable residual actor head on pixels -> action_dim, ZERO-INIT final layer.
        with torch.no_grad():
            n = _conv_trunk()(torch.zeros(1, 6, self._cam_h, self._cam_w)).shape[1]
        self.res_cnn = _conv_trunk()
        self.res_head = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, action_dim))
        nn.init.zeros_(self.res_head[-1].weight)
        nn.init.zeros_(self.res_head[-1].bias)

        # Trainable critic: pixels (sees the ball) + normalized proprio -> value.
        self.val_cnn = _conv_trunk()
        self.val_head = nn.Sequential(
            nn.Linear(n + self._proprio_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

        self.log_std = nn.Parameter(torch.ones(action_dim) * self._log_std_init)

        trainable = (list(self.res_cnn.parameters()) + list(self.res_head.parameters())
                     + list(self.val_cnn.parameters()) + list(self.val_head.parameters())
                     + [self.log_std])
        self.optimizer = self.optimizer_class(trainable, lr=lr_schedule(1), **self.optimizer_kwargs)

    # -- helpers -------------------------------------------------------------
    def _split(self, obs):
        return obs[:, :self._proprio_dim], obs[:, self._proprio_dim:]

    def _norm_proprio(self, proprio):
        return torch.clamp((proprio - self.pmean) / self.pstd, -self._base_clip, self._base_clip)

    def _mean_actions(self, obs):
        proprio, pixels = self._split(obs)
        pn = self._norm_proprio(proprio)
        with torch.no_grad():
            base_mean = self.base.get_distribution(pn).distribution.mean
        img = _pixels_to_img(pixels, self._cam_h, self._cam_w)
        residual = self.res_head(self.res_cnn(img))
        return base_mean + residual

    def _value(self, obs):
        proprio, pixels = self._split(obs)
        pn = self._norm_proprio(proprio)
        img = _pixels_to_img(pixels, self._cam_h, self._cam_w)
        feat = torch.cat([self.val_cnn(img), pn], dim=1)
        return self.val_head(feat).flatten()

    # -- SB3 ActorCriticPolicy interface ------------------------------------
    def forward(self, obs, deterministic=False):
        mean = self._mean_actions(obs)
        dist = self.action_dist.proba_distribution(mean, self.log_std)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, self._value(obs), log_prob

    def evaluate_actions(self, obs, actions):
        mean = self._mean_actions(obs)
        dist = self.action_dist.proba_distribution(mean, self.log_std)
        return self._value(obs), dist.log_prob(actions), dist.entropy()

    def predict_values(self, obs):
        return self._value(obs)

    def _predict(self, obs, deterministic=False):
        mean = self._mean_actions(obs)
        dist = self.action_dist.proba_distribution(mean, self.log_std)
        return dist.get_actions(deterministic=deterministic)

    def get_residual_norm(self, obs):
        """Diagnostic: mean |residual| — how much vision is adding on top of the base."""
        proprio, pixels = self._split(obs)
        img = _pixels_to_img(pixels, self._cam_h, self._cam_w)
        with torch.no_grad():
            r = self.res_head(self.res_cnn(img))
        return float(r.abs().mean().item())
