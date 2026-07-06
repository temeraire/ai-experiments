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
                 pixel_latent_dim: int = PIXEL_LATENT_DIM):
        self.proprio_dim   = proprio_dim
        self.pixel_latent_dim = pixel_latent_dim
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

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Proprio + memory pass through unchanged
        proprio = observations[:, : self.proprio_total]
        pixels_flat = observations[:, self.proprio_total :]

        if self.in_channels == 3:
            # Mono: single (H, W, 3) image
            img = pixels_flat.view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
        else:
            # Stereo: left + right stacked as 6 channels
            half = self.cam_h * self.cam_w * 3
            left  = pixels_flat[:, :half].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
            right = pixels_flat[:, half:].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
            img = torch.cat([left, right], dim=1)

        pixel_latent = self.proj(self.cnn(img))
        return torch.cat([proprio, pixel_latent], dim=1)
