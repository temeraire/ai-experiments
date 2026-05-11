"""
crawler_cnn_extractor.py — Stereo CNN feature extractor for MimoCrawlerEnv.

Observation layout (vision=True):
  [0:69]   proprio  (root pos/quat/vel, 25 jpos, 25 jvel, vestibular)
  [69:]    pixels   left_eye + right_eye, each (H, W, 3) flattened
                    stored in HWC order, total = 2 * H * W * 3

Architecture:
  - Stereo pair stacked as a 6-channel image (3ch left | 3ch right)
  - 4-conv DrQ-v2 encoder (stride-2, then 3× stride-1)
  - Linear projection → pixel_latent_dim
  - Output: concat(proprio, pixel_latent)
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from alien_baby.crawler.mimo_crawler_env import PROPRIO_DIM, CAM_H, CAM_W

PIXEL_LATENT_DIM = 64
STEREO_CHANNELS  = 6   # 3ch left + 3ch right


class StereoCrawlerCNN(BaseFeaturesExtractor):
    """DrQ-v2 4-conv encoder on stacked stereo pair + raw proprio passthrough."""

    def __init__(self, observation_space: spaces.Box,
                 proprio_dim: int = PROPRIO_DIM,
                 pixel_latent_dim: int = PIXEL_LATENT_DIM):
        feature_dim = proprio_dim + pixel_latent_dim
        super().__init__(observation_space, features_dim=feature_dim)
        self.proprio_dim   = proprio_dim
        self.pixel_latent_dim = pixel_latent_dim
        self.cam_h = CAM_H
        self.cam_w = CAM_W

        expected = proprio_dim + CAM_H * CAM_W * 3 * 2
        assert observation_space.shape[0] == expected, (
            f"StereoCrawlerCNN expects obs dim {expected}; "
            f"got {observation_space.shape[0]}. Use vision=True env."
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(STEREO_CHANNELS, 32, kernel_size=3, stride=2),
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
            dummy = torch.zeros(1, STEREO_CHANNELS, self.cam_h, self.cam_w)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.proj = nn.Sequential(
            nn.Linear(cnn_out_dim, pixel_latent_dim),
            nn.LayerNorm(pixel_latent_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        proprio = observations[:, : self.proprio_dim]
        pixels_flat = observations[:, self.proprio_dim :]

        half = self.cam_h * self.cam_w * 3
        left  = pixels_flat[:, :half].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
        right = pixels_flat[:, half:].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
        stereo = torch.cat([left, right], dim=1)   # (B, 6, H, W)

        pixel_latent = self.proj(self.cnn(stereo))
        return torch.cat([proprio, pixel_latent], dim=1)
