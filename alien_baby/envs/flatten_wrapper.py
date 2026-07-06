"""
Wrapper that flattens Dict observations (proprio + vision) into a single vector.

This forces interpenetration: vision features are concatenated directly into
the same input vector as proprioception, rather than processed by a separate
encoder. The MLP must learn to integrate both modalities through shared pathways.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Downsample vision to keep observation vector manageable
# 64x64x3 = 12288 is too large — downsample to 16x16x3 = 768
VISION_DOWNSCALE = 4  # 64/4 = 16


class FlattenVisionWrapper(gym.ObservationWrapper):
    """Flatten Dict(proprio, vision) into a single Box observation."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), \
            "FlattenVisionWrapper requires Dict observation space"

        proprio_dim = env.observation_space["proprio"].shape[0]
        vis_shape = env.observation_space["vision"].shape
        # Downsampled vision size
        h = vis_shape[0] // VISION_DOWNSCALE
        w = vis_shape[1] // VISION_DOWNSCALE
        self._vision_flat_dim = h * w * vis_shape[2]
        total_dim = proprio_dim + self._vision_flat_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )
        self._proprio_dim = proprio_dim

    def observation(self, obs):
        proprio = obs["proprio"]
        vision = obs["vision"]

        # Downsample vision
        h, w = vision.shape[0] // VISION_DOWNSCALE, vision.shape[1] // VISION_DOWNSCALE
        vision_small = vision[::VISION_DOWNSCALE, ::VISION_DOWNSCALE, :]

        # Normalize pixels to [0, 1] and flatten
        vision_flat = vision_small.flatten().astype(np.float32) / 255.0

        return np.concatenate([proprio, vision_flat]).astype(np.float32)
