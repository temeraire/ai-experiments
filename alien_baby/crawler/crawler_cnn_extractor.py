"""
crawler_cnn_extractor.py — CNN feature extractor for MimoCrawlerEnv.

Auto-detects mono vs stereo from the observation shape so the same class
works for both --mono and --stereo training runs.

Observation layout (vision=True):
  [0:69]   proprio  (root pos/quat/vel, 25 jpos, 25 jvel, vestibular)
  [69:]    pixels   either:
                      - single eye   (mono):    H*W*3
                      - left+right (stereo):  2*H*W*3
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from alien_baby.crawler.mimo_crawler_env import PROPRIO_DIM, CAM_H, CAM_W

PIXEL_LATENT_DIM = 64


class StereoCrawlerCNN(BaseFeaturesExtractor):
    """DrQ-v2 4-conv encoder on the pixel observation (mono or stereo) + raw proprio."""

    def __init__(self, observation_space: spaces.Box,
                 proprio_dim: int = PROPRIO_DIM,
                 pixel_latent_dim: int = PIXEL_LATENT_DIM,
                 gain_field: bool = False):
        self.proprio_dim   = proprio_dim
        self.pixel_latent_dim = pixel_latent_dim
        self.gain_field = bool(gain_field)
        self.cam_h = CAM_H
        self.cam_w = CAM_W

        # Decide mono (3-channel) vs stereo (6-channel) and whether memory flags
        # are present, based on obs shape relative to proprio_dim.
        pixel_dim = observation_space.shape[0] - proprio_dim
        if pixel_dim == 2 + CAM_H * CAM_W * 3:
            self.in_channels = 3
            self.memory_dim = 2
        elif pixel_dim == 2 + CAM_H * CAM_W * 3 * 2:
            self.in_channels = 6
            self.memory_dim = 2
        elif pixel_dim == CAM_H * CAM_W * 3:
            self.in_channels = 3
            self.memory_dim = 0
        elif pixel_dim == CAM_H * CAM_W * 3 * 2:
            self.in_channels = 6
            self.memory_dim = 0
        else:
            raise ValueError(
                f"Cannot infer mono/stereo from obs dim {observation_space.shape[0]} "
                f"with proprio_dim={proprio_dim}; pixel_dim={pixel_dim}"
            )
        self.proprio_total = proprio_dim + self.memory_dim
        feature_dim = self.proprio_total + pixel_latent_dim
        super().__init__(observation_space, features_dim=feature_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.cam_h, self.cam_w)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.proj = nn.Sequential(
            nn.Linear(cnn_out_dim, pixel_latent_dim),
            nn.LayerNorm(pixel_latent_dim),
            nn.Tanh(),
        )

        # Bearing readout: theta_vis = "where the eye says the ball is", from the pixel
        # latent, as a (sin, cos) unit vector. Trained ONLY by the seen-vs-contacted
        # mismatch aux loss (see train_head_search), never by PPO reward — so it is the
        # eye's own estimate, and driving it to agree with the contact-confirmed true
        # bearing is what re-aligns vision under a prism. Unused unless --mismatch-coef>0;
        # inert for existing runs (extra params, no effect on forward()).
        self.bearing_head = nn.Linear(pixel_latent_dim, 2)

        # GAIN-FIELD bridge (opt-in): the prism displacement lives in the RETINAL frame but the
        # recalibration TARGET is body/effector-frame (2026-07-12 result; Taylor 8.12/9.20; Tsay/Ivry
        # PReMo; VICES). The brain resolves this NOT by choosing a frame but by carrying head/eye
        # posture as a MULTIPLICATIVE gain on the retinal code, then reading out the body frame
        # (Pouget & Sejnowski 1997; Salinas & Abbott 2001). Here: FiLM-modulate the pixel latent by a
        # function of proprio (which contains head/neck pose), so theta_vis = f(retinal, head-pose).
        # This is the neck-proprioception term the ~90deg torso-vs-gaze misalignment demands. Applied
        # ONLY in bearing_pred (the eye's estimate); forward() and the policy input are untouched.
        # Default off => the plain readout is byte-identical for existing runs. See
        # PRISM_GAZE_RELATIVE_PROPOSAL.md and the CLAUDE.md GAZE-frame section.
        if self.gain_field:
            self.gain_mlp = nn.Sequential(
                nn.Linear(self.proprio_total, 64), nn.ReLU(),
                nn.Linear(64, 2 * pixel_latent_dim),   # -> (gamma, beta) FiLM params
            )
            nn.init.zeros_(self.gain_mlp[-1].weight)   # start at identity: gamma=0, beta=0
            nn.init.zeros_(self.gain_mlp[-1].bias)

    def _pixel_latent(self, observations: torch.Tensor) -> torch.Tensor:
        pixels_flat = observations[:, self.proprio_total :]
        if self.in_channels == 3:
            img = pixels_flat.view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
        else:
            half = self.cam_h * self.cam_w * 3
            left  = pixels_flat[:, :half].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
            right = pixels_flat[:, half:].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
            img = torch.cat([left, right], dim=1)
        return self.proj(self.cnn(img))

    def bearing_pred(self, observations: torch.Tensor) -> torch.Tensor:
        """(N, 2) raw (sin, cos) of the eye's estimated ball bearing. Normalize before use.
        With gain_field, the pixel latent is FiLM-modulated by head/eye posture (from proprio)
        BEFORE the readout — a retinal x head-pose gain field that maps the retinal bearing into
        the body/effector frame the target lives in."""
        lat = self._pixel_latent(observations)
        if self.gain_field:
            proprio = observations[:, : self.proprio_total]
            gamma, beta = self.gain_mlp(proprio).chunk(2, dim=1)
            lat = (1.0 + gamma) * lat + beta            # FiLM; init gamma=beta=0 => identity
        return self.bearing_head(lat)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        proprio = observations[:, : self.proprio_total]
        pixel_latent = self._pixel_latent(observations)
        return torch.cat([proprio, pixel_latent], dim=1)
