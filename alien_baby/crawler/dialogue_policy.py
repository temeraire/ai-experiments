"""
dialogue_policy.py — Two-stream SAC Actor for MimoCrawlerEnv.

The actor has TWO independent decision streams that are equal peers:
  proprio stream:  proprio (69) → MLP → (μ_p, log_σ_p)
  vision  stream:  pixels (2·32·32·3) → StereoCNN → MLP → (μ_v, log_σ_v)

A learned scalar gate w ∈ [0, 1] (computed from concatenated stream features)
combines them:
  μ        = w · μ_p     + (1−w) · μ_v
  log_σ    = w · log_σ_p + (1−w) · log_σ_v   (weighted avg — approximation)

The standard SAC tanh-squashed Gaussian then samples actions and computes
log-probs from (μ, log_σ). Vision can be ablated by zeroing the pixel observation
or by forcing w=1; the per-stream means are also exposed for inspection.

Consistency loss:
  L_consistency = λ · ||μ_p − μ_v||²
This is added to the actor loss in the trainer (see DialogueSAC). It pressures
the two channels toward agreement: where they already agree, the loss is zero
and the gate is free to use whichever pathway has lower variance; where they
disagree, the gradient pushes both toward each other.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy, LOG_STD_MAX, LOG_STD_MIN
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from alien_baby.crawler.mimo_crawler_env import PROPRIO_DIM, CAM_H, CAM_W


STEREO_CHANNELS = 6
PIXEL_LATENT_DIM = 64


class _IdentityExtractor(BaseFeaturesExtractor):
    """Pass the raw observation through; DialogueActor does its own splitting."""

    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=int(observation_space.shape[0]))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class _StereoCNN(nn.Module):
    """DrQ-v2 4-conv encoder on stacked stereo pair → pixel_latent_dim."""

    def __init__(self, cam_h: int, cam_w: int, latent_dim: int):
        super().__init__()
        self.cam_h, self.cam_w = cam_h, cam_w
        self.cnn = nn.Sequential(
            nn.Conv2d(STEREO_CHANNELS, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, STEREO_CHANNELS, cam_h, cam_w)).shape[1]
        self.proj = nn.Sequential(
            nn.Linear(cnn_out, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def forward(self, pixels_flat: torch.Tensor) -> torch.Tensor:
        half = self.cam_h * self.cam_w * 3
        left  = pixels_flat[:, :half].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
        right = pixels_flat[:, half:].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
        stereo = torch.cat([left, right], dim=1)
        return self.proj(self.cnn(stereo))


class DialogueActor(Actor):
    """Two-stream actor whose final action is a learned-weighted combination."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        proprio_dim: int = PROPRIO_DIM,
        pixel_latent_dim: int = PIXEL_LATENT_DIM,
        cam_h: int = CAM_H,
        cam_w: int = CAM_W,
    ):
        # Bypass parent constructor's network building (we build our own)
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )
        self.use_sde = False
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init

        self.proprio_dim = proprio_dim
        self.pixel_latent_dim = pixel_latent_dim

        action_dim = get_action_dim(self.action_space)
        last_dim = net_arch[-1] if net_arch else max(proprio_dim, pixel_latent_dim)

        # Proprio stream
        proprio_layers = create_mlp(proprio_dim, -1, net_arch, activation_fn)
        self.proprio_trunk = nn.Sequential(*proprio_layers)
        self.proprio_mu      = nn.Linear(last_dim, action_dim)
        self.proprio_log_std = nn.Linear(last_dim, action_dim)

        # Vision stream
        self.vision_cnn = _StereoCNN(cam_h, cam_w, pixel_latent_dim)
        vision_layers = create_mlp(pixel_latent_dim, -1, net_arch, activation_fn)
        self.vision_trunk = nn.Sequential(*vision_layers)
        self.vision_mu      = nn.Linear(last_dim, action_dim)
        self.vision_log_std = nn.Linear(last_dim, action_dim)

        # Gate: takes both trunk outputs, outputs scalar w ∈ [0,1]
        self.gate = nn.Sequential(
            nn.Linear(last_dim * 2, 64), activation_fn(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

        # Standard SAC squashed-Gaussian distribution
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

        # Buffers for last-forward inspection (filled in get_action_dist_params)
        self._last_mu_p = None
        self._last_mu_v = None
        self._last_gate = None

    # ------------------------------------------------------------------
    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return obs[:, : self.proprio_dim], obs[:, self.proprio_dim :]

    # ------------------------------------------------------------------
    def get_action_dist_params(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        proprio, pixels = self._split_obs(obs)

        p_feat = self.proprio_trunk(proprio)
        v_feat = self.vision_trunk(self.vision_cnn(pixels))

        mu_p     = self.proprio_mu(p_feat)
        log_std_p = self.proprio_log_std(p_feat).clamp(LOG_STD_MIN, LOG_STD_MAX)

        mu_v      = self.vision_mu(v_feat)
        log_std_v = self.vision_log_std(v_feat).clamp(LOG_STD_MIN, LOG_STD_MAX)

        w = self.gate(torch.cat([p_feat, v_feat], dim=-1))   # [B, 1]

        mu      = w * mu_p     + (1 - w) * mu_v
        log_std = w * log_std_p + (1 - w) * log_std_v

        # Cache for consistency loss / diagnostics
        self._last_mu_p = mu_p
        self._last_mu_v = mu_v
        self._last_gate = w

        return mu, log_std, {}

    # ------------------------------------------------------------------
    def consistency_loss(self) -> torch.Tensor:
        """MSE between proprio and vision action proposals from last forward."""
        if self._last_mu_p is None or self._last_mu_v is None:
            return torch.tensor(0.0)
        return ((self._last_mu_p - self._last_mu_v) ** 2).mean()

    # ------------------------------------------------------------------
    def diagnostics(self) -> Dict[str, float]:
        """Scalar summaries of the last forward pass — for logging."""
        out: Dict[str, float] = {}
        if self._last_gate is not None:
            out["dialogue/gate_mean"] = self._last_gate.mean().item()
            out["dialogue/gate_std"]  = self._last_gate.std().item()
        if self._last_mu_p is not None and self._last_mu_v is not None:
            disagreement = (self._last_mu_p - self._last_mu_v).abs().mean().item()
            out["dialogue/disagreement_mean"] = disagreement
            out["dialogue/mu_proprio_mean"]   = self._last_mu_p.abs().mean().item()
            out["dialogue/mu_vision_mean"]    = self._last_mu_v.abs().mean().item()
        return out


class DialogueSACPolicy(SACPolicy):
    """SACPolicy that uses DialogueActor instead of the standard one."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = _IdentityExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        # SB3 SACPolicy expects net_arch dict {pi, qf}; default to [256, 256]
        if net_arch is None:
            net_arch = [256, 256]
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DialogueActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        # Use only the standard actor kwargs; ignore SB3 SDE-specific ones
        return DialogueActor(
            observation_space=actor_kwargs["observation_space"],
            action_space=actor_kwargs["action_space"],
            net_arch=actor_kwargs["net_arch"],
            features_extractor=actor_kwargs["features_extractor"],
            features_dim=actor_kwargs["features_dim"],
            activation_fn=actor_kwargs["activation_fn"],
            normalize_images=actor_kwargs.get("normalize_images", True),
        ).to(self.device)
