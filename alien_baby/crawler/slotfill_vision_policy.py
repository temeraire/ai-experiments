"""
slotfill_vision_policy.py — Stage B: reward-grown vision that fills the FROZEN teacher's
ball-bearing slot from pixels.

  bearing_hat = g(pixels)                          [trainable CNN]
  action_mean = teacher( normalize([proprio, bearing_hat]) )   [teacher FROZEN, but in
                                                                 the gradient path -> g]

The teacher (crawl_ppo_2M, proprio+bearing -> action) is frozen, so its motor mapping —
the crawl gait AND the steering it already knows — is preserved. Vision's only job is to
supply the one signal proprio lacks (the bearing), grown by PPO reward alone (no bearing
supervision, unlike Stage A). Because the teacher params are frozen but NOT detached,
gradients flow through the teacher into g.

Env obs (vision=True, target_obs=False, same as Stage C): [0:69] proprio, [69:] pixels.
Teacher VecNormalize stats are 72-dim (proprio69 + bearing3); we normalize the assembled
[proprio, bearing_hat] with them.
"""
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution

from alien_baby.crawler.mimo_crawler_env import PROPRIO_DIM, CAM_H, CAM_W
from alien_baby.crawler.residual_vision_policy import _conv_trunk, _pixels_to_img


class SlotFillVisionPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 teacher_model_path=None, teacher_mean=None, teacher_std=None,
                 teacher_clip=10.0, warmstart_cnn=None,
                 proprio_dim=PROPRIO_DIM, cam_h=CAM_H, cam_w=CAM_W,
                 log_std_init=-1.0, **kwargs):
        self._teacher_model_path = teacher_model_path
        self._warmstart_cnn = warmstart_cnn
        self._tmean = np.asarray(teacher_mean, np.float32)   # 72-dim
        self._tstd = np.asarray(teacher_std, np.float32)     # 72-dim
        self._tclip = float(teacher_clip)
        self._proprio_dim = proprio_dim
        self._cam_h, self._cam_w = cam_h, cam_w
        self._log_std_init = log_std_init
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        action_dim = int(np.prod(self.action_space.shape))
        self.action_dist = DiagGaussianDistribution(action_dim)

        teacher = PPO.load(self._teacher_model_path, device="cpu").policy
        for p in teacher.parameters():
            p.requires_grad_(False)   # frozen params, but grads still flow THROUGH to g
        self.teacher = teacher

        self.register_buffer("tmean", torch.as_tensor(self._tmean))
        self.register_buffer("tstd", torch.as_tensor(self._tstd))

        with torch.no_grad():
            n = _conv_trunk()(torch.zeros(1, 6, self._cam_h, self._cam_w)).shape[1]
        # Bearing head: pixels -> 3-vec, ZERO-INIT so start == teacher-with-zero-bearing.
        self.g_cnn = _conv_trunk()
        self.g_head = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.g_head[-1].weight)
        nn.init.zeros_(self.g_head[-1].bias)
        # Warm-start the vision head from Stage A's distilled bearing CNN (R2~0.84):
        # vision then already reads direction at init, so the frozen teacher gets a good
        # bearing immediately and reward only fine-tunes (distill-then-RL).
        if self._warmstart_cnn:
            sd = torch.load(self._warmstart_cnn, map_location="cpu")
            g_cnn_sd = {k[len("cnn."):]: v for k, v in sd.items() if k.startswith("cnn.")}
            g_head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
            self.g_cnn.load_state_dict(g_cnn_sd)
            self.g_head.load_state_dict(g_head_sd)
            print(f"[SlotFill] warm-started vision head from {self._warmstart_cnn}")

        # Fresh critic: pixels (sees the ball) + proprio -> value.
        self.val_cnn = _conv_trunk()
        self.val_head = nn.Sequential(
            nn.Linear(n + self._proprio_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

        self.log_std = nn.Parameter(torch.ones(action_dim) * self._log_std_init)

        trainable = (list(self.g_cnn.parameters()) + list(self.g_head.parameters())
                     + list(self.val_cnn.parameters()) + list(self.val_head.parameters())
                     + [self.log_std])
        self.optimizer = self.optimizer_class(trainable, lr=lr_schedule(1), **self.optimizer_kwargs)

    def _split(self, obs):
        return obs[:, :self._proprio_dim], obs[:, self._proprio_dim:]

    def _bearing(self, pixels):
        img = _pixels_to_img(pixels, self._cam_h, self._cam_w)
        return self.g_head(self.g_cnn(img))

    def _mean_actions(self, obs):
        proprio, pixels = self._split(obs)
        bearing = self._bearing(pixels)
        raw72 = torch.cat([proprio, bearing], dim=1)
        norm72 = torch.clamp((raw72 - self.tmean) / self.tstd, -self._tclip, self._tclip)
        # teacher frozen but differentiable -> gradient reaches g through it
        return self.teacher.get_distribution(norm72).distribution.mean

    def _value(self, obs):
        proprio, pixels = self._split(obs)
        img = _pixels_to_img(pixels, self._cam_h, self._cam_w)
        feat = torch.cat([self.val_cnn(img), proprio], dim=1)
        return self.val_head(feat).flatten()

    def forward(self, obs, deterministic=False):
        mean = self._mean_actions(obs)
        dist = self.action_dist.proba_distribution(mean, self.log_std)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, self._value(obs), dist.log_prob(actions)

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
