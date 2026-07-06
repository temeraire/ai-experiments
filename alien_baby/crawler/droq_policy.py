"""
droq_policy.py — DroQ critic regularization for SAC (Phase W).

DroQ (Drozdov et al. 2021 / Nauman et al. 2024) stabilizes SAC critics via:
  - LayerNorm after each hidden layer in the critic networks
  - Dropout (rate=0.01 default) after each hidden layer in the critic networks
  - Higher update-to-data (UTD) ratio (e.g. 4 gradient steps per env step)

Only the CRITIC is regularized. The actor is left exactly as stock SACPolicy.
The UTD ratio is passed as `gradient_steps` in sac_kwargs (handled in train_crawler.py).

Usage in train_crawler.py (when --droq is active):
    from alien_baby.crawler.droq_policy import DroQSACPolicy
    policy_kwargs = dict(net_arch=[256, 256], dropout_rate=0.01)
    model = SAC(DroQSACPolicy, train_env, **sac_kwargs)
"""

from __future__ import annotations

import torch as th
import torch.nn as nn
from typing import Any

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import create_mlp, BaseFeaturesExtractor
from gymnasium import spaces


class DroQCritic(ContinuousCritic):
    """ContinuousCritic with LayerNorm + Dropout injected after each hidden linear layer.

    Identical to the parent in every other respect (forward, q1_forward, etc. are inherited).
    The regularization lives only in the q_networks; the features extractor is untouched.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        dropout_rate: float = 0.01,
    ):
        # Call nn.Module grandparent (BaseModel) init via ContinuousCritic's parent chain,
        # but we need to skip ContinuousCritic.__init__ because it calls create_mlp without
        # post_linear_modules. We replicate its logic here.
        # BaseModel.__init__ → BasePolicy.__init__
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        from stable_baselines3.common.preprocessing import get_action_dim
        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []

        # post_linear_modules: after each hidden Linear, apply LayerNorm then Dropout.
        # create_mlp calls each item as module(feature_dim) — so the callable receives
        # the layer width as its first argument.
        # - nn.LayerNorm(feature_dim) is correct (it normalizes over that dimension).
        # - nn.Dropout(p) does NOT take a feature dim; we wrap it so the dim is ignored.
        class _DropoutIgnoreDim(nn.Module):
            """Dropout that ignores the feature-dim arg create_mlp passes."""
            def __init__(self, _dim: int, p: float = dropout_rate):
                super().__init__()
                self._drop = nn.Dropout(p=p)
            def forward(self, x: th.Tensor) -> th.Tensor:
                return self._drop(x)

        post_modules = [nn.LayerNorm, _DropoutIgnoreDim]

        for idx in range(n_critics):
            q_net_list = create_mlp(
                features_dim + action_dim,
                1,
                net_arch,
                activation_fn,
                post_linear_modules=post_modules,
            )
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        self.dropout_rate = dropout_rate


class DroQSACPolicy(SACPolicy):
    """SACPolicy that uses DroQCritic (LayerNorm + Dropout on Q-networks only).

    Accepts an extra policy_kwarg: ``dropout_rate`` (float, default 0.01).
    The actor is built identically to stock SACPolicy — only the critic changes.
    """

    def __init__(self, *args, dropout_rate: float = 0.01, **kwargs):
        self._droq_dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> DroQCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        # Inject dropout_rate; DroQCritic accepts it, ContinuousCritic does not.
        critic_kwargs = dict(critic_kwargs, dropout_rate=self._droq_dropout_rate)
        return DroQCritic(**critic_kwargs).to(self.device)
