"""
DrQ-v2 style CNN feature extractor for the v8 hybrid proprio+pixel observation.

Splits a flat (29 + 32*32*3) = 3101-dim obs into:
  - proprio (first 29 dims) — passed through unchanged
  - pixels  (remaining 3072 dims, reshaped to 3x32x32) — fed through 4 conv layers
    (Yarats 2021 / DrQ-v2 default kernel sizes), flattened, projected to 50-dim latent

Output is concat([proprio, pixel_latent]) of shape (29 + 50,) = 79.

The downstream MLP (SB3's default 256-256) then operates on this compact
representation. This is the architecture GAP_ANALYSIS Rank 1 prescribes:
"freeze the proprio columns of the first MLP layer; let the CNN and the
pixel-column slice of the first MLP layer train. Consistency loss applies
identically since proprio_dim=29 is preserved at the output."
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

PROPRIO_DIM_V8 = 29
CAM_HEIGHT = 32
CAM_WIDTH = 32
CAM_CHANNELS = 3
PIXEL_LATENT_DIM = 50


class HybridProprioCNN(BaseFeaturesExtractor):
    """4-conv encoder (DrQ-v2 default) on the pixel slice + raw proprio passthrough.

    The combined feature dim is `PROPRIO_DIM_V8 + PIXEL_LATENT_DIM` so that the
    first 29 entries of the policy's first MLP layer continue to correspond to
    proprio scalars — preserving the consistency-loss / frozen-proprio
    semantics from train_v5.py.
    """

    def __init__(self, observation_space: spaces.Box,
                 proprio_dim: int = PROPRIO_DIM_V8,
                 pixel_latent_dim: int = PIXEL_LATENT_DIM):
        feature_dim = proprio_dim + pixel_latent_dim
        super().__init__(observation_space, features_dim=feature_dim)
        self.proprio_dim = int(proprio_dim)
        self.pixel_latent_dim = int(pixel_latent_dim)
        self.cam_h = CAM_HEIGHT
        self.cam_w = CAM_WIDTH
        self.cam_c = CAM_CHANNELS

        flat_pixels = self.cam_h * self.cam_w * self.cam_c
        expected = self.proprio_dim + flat_pixels
        assert observation_space.shape[0] == expected, (
            f"HybridProprioCNN expects obs of shape ({expected},); got "
            f"{observation_space.shape}. Did you build it for a vision env?"
        )

        # DrQ-v2 default: 4 conv layers, 32 channels, 3x3 kernels, stride 2-1-1-1.
        # Applied to 32x32 input: 32 -> 15 -> 13 -> 11 -> 9 (output 32 * 9 * 9 = 2592).
        self.cnn = nn.Sequential(
            nn.Conv2d(self.cam_c, 32, kernel_size=3, stride=2),
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
            dummy = torch.zeros(1, self.cam_c, self.cam_h, self.cam_w)
            cnn_out_dim = self.cnn(dummy).shape[1]
        self.proj = nn.Sequential(
            nn.Linear(cnn_out_dim, self.pixel_latent_dim),
            nn.LayerNorm(self.pixel_latent_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        proprio = observations[:, : self.proprio_dim]
        pixels_flat = observations[:, self.proprio_dim :]
        # Reshape to (B, C, H, W). The env stores pixels as the result of
        # `.flatten()` on an (H, W, C) array, so undo that ordering.
        pixels = pixels_flat.view(-1, self.cam_h, self.cam_w, self.cam_c)
        pixels = pixels.permute(0, 3, 1, 2).contiguous()
        pixel_feat = self.cnn(pixels)
        pixel_latent = self.proj(pixel_feat)
        return torch.cat([proprio, pixel_latent], dim=1)
