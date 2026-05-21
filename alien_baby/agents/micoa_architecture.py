"""
micoa_architecture.py — Architecture sketch for Multiple Inputs Confirming One Another.

CORE IDEA
---------
Current approach (broken): concat(proprio, pixels) → single MLP → action
SAC finds the easier gradient (proprio) and the pixel channel gets no training signal.

MICOA approach: each modality encodes INDEPENDENTLY to a probability distribution
over a shared latent space Z. A Product of Experts detects when they agree.

When both channels point at the same region of Z → the combined distribution
is tight (low σ). That tightness IS the confirmation signal. The policy acts
on a z sampled from this distribution — so it is more confident when both channels
confirm each other, and appropriately uncertain when they conflict.

This is architecturally different from concatenation. Concatenation is two rods
welded end to end. This is a corner: two constraints meeting, each enforcing
something the other can't.

INDEPENDENCE REQUIREMENT
------------------------
For the confirmation to mean anything, the two encoders must arrive at their
distributions WITHOUT SEEING EACH OTHER'S INPUT. No shared weights, no shared
gradient path between encoder_p and encoder_v. They are trained jointly
(same optimizer step) but each has its own information channel.

The independence is enforced structurally: proprio never touches encoder_v,
pixels never touch encoder_p. The only place they meet is the PoE combination,
which is parameter-free — it's pure math, not learned.

THE BOX ANALOGY
---------------
A box made of tubes is rigid because corners constrain three dimensions
simultaneously. Removing one tube collapses the structure.

Here: each "tube" is one encoder asserting where Z should be.
The "corner" is the region of Z both encoders agree on.
When both agree, Z is tightly constrained — the corner is formed.
When one channel is missing (ablation), Z becomes uncertain — the box sags.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

# ── Shared latent space dimensionality ──────────────────────────────────────
LATENT_DIM = 64  # Z lives here; both encoders project into this space


# ── Independent encoders ─────────────────────────────────────────────────────

class ProprioEncoder(nn.Module):
    """
    Maps proprio observation to a Gaussian (μ_p, σ_p) in latent space Z.

    Outputs a DISTRIBUTION, not a point. The width σ_p represents how
    certain proprio is about where in Z the current state belongs.
    High σ_p = proprio is ambiguous. Low σ_p = proprio is confident.
    """
    def __init__(self, proprio_dim: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),         nn.ReLU(),
        )
        self.mu_head        = nn.Linear(128, latent_dim)
        self.log_sigma_head = nn.Linear(128, latent_dim)

    def forward(self, proprio: torch.Tensor):
        h = self.net(proprio)
        mu        = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-4, 2)  # σ in [0.018, 7.4]
        return mu, log_sigma.exp()


class VisionEncoder(nn.Module):
    """
    Maps stereo pixel observation to a Gaussian (μ_v, σ_v) in latent space Z.

    Same latent space as ProprioEncoder — that's the contract. Both encoders
    are trying to locate the SAME underlying world state in Z, from different
    sensory evidence.

    High σ_v = vision is ambiguous (blurry, occluded, early training).
    Low σ_v = vision is confident about where in Z the state belongs.
    """
    def __init__(self, cam_h: int, cam_w: int, in_channels: int = 6,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.cam_h, self.cam_w, self.in_channels = cam_h, cam_w, in_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),           nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),           nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),           nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, in_channels, cam_h, cam_w)).shape[1]

        self.proj           = nn.Sequential(nn.Linear(cnn_out_dim, 128), nn.ReLU())
        self.mu_head        = nn.Linear(128, latent_dim)
        self.log_sigma_head = nn.Linear(128, latent_dim)

    def forward(self, pixels_flat: torch.Tensor):
        half = self.cam_h * self.cam_w * 3
        left  = pixels_flat[:, :half ].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
        right = pixels_flat[:, half: ].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
        img = torch.cat([left, right], dim=1)

        h         = self.proj(self.cnn(img))
        mu        = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-4, 2)
        return mu, log_sigma.exp()


# ── Confirmation detector ─────────────────────────────────────────────────────

def product_of_experts(mu_p, sigma_p, mu_v, sigma_v):
    """
    Combine two independent Gaussians into one.

    This is the MICOA confirmation detector. It is parameter-free —
    it's mathematics, not a learned layer.

    When both channels agree (mu_p ≈ mu_v):
        precisions add → sigma_combined shrinks → the CORNER IS FORMED.
        The combined distribution is tighter than either alone.

    When they disagree (mu_p far from mu_v):
        sigma_combined stays large → honest uncertainty.
        No false confidence is created.

    This is how Bayesian inference works: two independent measurements
    of the same thing multiply their evidence, not add it.
    """
    prec_p = 1.0 / sigma_p.pow(2)   # precision = certainty
    prec_v = 1.0 / sigma_v.pow(2)

    prec_combined = prec_p + prec_v
    mu_combined   = (mu_p * prec_p + mu_v * prec_v) / prec_combined
    sigma_combined = (1.0 / prec_combined).sqrt()

    return mu_combined, sigma_combined


def kl_gaussian(mu1, sigma1, mu2, sigma2):
    """
    KL( N(mu1, sigma1) || N(mu2, sigma2) ) — per latent dimension, summed.

    KL is asymmetric: it measures how surprised N(mu2,sigma2) is by samples
    from N(mu1,sigma1). We use the symmetric version (see below) so neither
    encoder is treated as ground truth.
    """
    return (
        torch.log(sigma2 / sigma1)
        + (sigma1.pow(2) + (mu1 - mu2).pow(2)) / (2.0 * sigma2.pow(2))
        - 0.5
    ).sum(dim=-1)  # sum over latent dims, keep batch dim


def encoder_agreement_loss(mu_p, sigma_p, mu_v, sigma_v):
    """
    Symmetric KL between the two encoder distributions.

    Neither encoder is ground truth — we penalise them equally for disagreeing.
    This closes the escape hatch: vision cannot simply output high sigma_v
    (wide uncertainty) and let proprio carry the signal, because that wide
    distribution is also penalised for being far from proprio's tight one.

    Typical values:
      ~0.0  → near-perfect agreement (corner is tight)
      ~1-5  → moderate disagreement (corner forming but loose)
      >>10  → encoders pointing at different regions of Z (no corner yet)

    beta controls the trade-off:
      too high → encoders collapse to the same distribution regardless of input
                 (forced agreement, not earned agreement — defeats independence)
      too low  → no pressure, same as before
      start at beta=0.1, watch the curve, increase only if kl_agreement flat
    """
    kl_pv = kl_gaussian(mu_p, sigma_p, mu_v, sigma_v)
    kl_vp = kl_gaussian(mu_v, sigma_v, mu_p, sigma_p)
    return ((kl_pv + kl_vp) * 0.5).mean()  # scalar


# ── SB3-compatible extractor ─────────────────────────────────────────────────

class MICOAExtractor(BaseFeaturesExtractor):
    """
    Drop-in SB3 feature extractor implementing MICOA.

    The SAC policy (actor + critic) acts on a feature vector of shape:
        [z_combined (64) | mu_p (64) | mu_v (64)] = 192 dims

    Exposing mu_p and mu_v separately (in addition to z_combined) lets the
    downstream MLP learn to detect agreement: when mu_p ≈ mu_v ≈ z_combined,
    all three columns are similar → the network can learn "I am confident."

    sigma_combined and kl_agreement are NOT passed to the policy — they are
    stored here and logged by MICOAConfirmationCallback. They are the primary
    diagnostics for whether the corner is actually forming.
    """
    def __init__(self, observation_space: spaces.Box,
                 proprio_dim: int, cam_h: int, cam_w: int,
                 in_channels: int = 6, latent_dim: int = LATENT_DIM):
        super().__init__(observation_space, features_dim=latent_dim * 3)
        self.proprio_dim     = proprio_dim
        self.latent_dim      = latent_dim
        self.proprio_encoder = ProprioEncoder(proprio_dim, latent_dim)
        self.vision_encoder  = VisionEncoder(cam_h, cam_w, in_channels, latent_dim)

        # Stored each forward() pass; read by MICOASAC.train() and the callback
        self.last_sigma_combined = None
        self.last_kl_agreement   = None
        self.last_mu_p           = None
        self.last_sigma_p        = None
        self.last_mu_v           = None
        self.last_sigma_v        = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio     = obs[:, :self.proprio_dim]
        pixels_flat = obs[:, self.proprio_dim:]

        # Independent encoding — no shared computation between these two calls
        mu_p, sigma_p = self.proprio_encoder(proprio)
        mu_v, sigma_v = self.vision_encoder(pixels_flat)

        # Confirmation detection: parameter-free PoE
        mu_z, sigma_z = product_of_experts(mu_p, sigma_p, mu_v, sigma_v)

        # Store distribution params so MICOASAC.train() can compute KL loss.
        # Kept in the graph (not detached) so the KL backward pass can reach
        # both encoder weight sets.
        self.last_mu_p    = mu_p;    self.last_sigma_p = sigma_p
        self.last_mu_v    = mu_v;    self.last_sigma_v = sigma_v
        # Detached scalars for logging only
        self.last_sigma_combined = sigma_z.detach().mean().item()
        self.last_kl_agreement   = encoder_agreement_loss(
            mu_p.detach(), sigma_p.detach(),
            mu_v.detach(), sigma_v.detach(),
        ).item()

        # Sample during training, use mean during eval
        if self.training:
            z = mu_z + sigma_z * torch.randn_like(mu_z)
        else:
            z = mu_z

        # Policy receives: combined z + both individual encodings
        return torch.cat([z, mu_p, mu_v], dim=1)


# ── Confirmation diagnostic callback ─────────────────────────────────────────

class MICOAConfirmationCallback(BaseCallback):
    """
    Logs sigma_combined every N steps so we can watch confirmation develop.

    What to look for:
      - Early training: sigma_combined is large (channels disagree or are uncertain)
      - As vision learns: sigma_combined should DECREASE at ball-contact steps
      - If sigma_combined never decreases: vision is not encoding the same
        world state as proprio — the corner is not being formed

    This is the primary diagnostic for whether MICOA is actually happening.
    If sigma_combined at contact moments is not smaller than at non-contact
    moments, vision and proprio are not confirming each other — they are
    parallel but not convergent.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            extractor = _get_micoa_extractor(self.model.policy)
            if extractor is None:
                return True
            sigma = extractor.last_sigma_combined
            kl    = extractor.last_kl_agreement
            if sigma is not None:
                self.logger.record("micoa/sigma_combined", sigma)
            if kl is not None:
                self.logger.record("micoa/kl_agreement", kl)
            # Also print to stdout so the values land in the training log
            # without requiring tensorboard_log to be configured. This is the
            # primary diagnostic for whether the corner is forming — must be
            # visible in plain text for the overnight heartbeat loop.
            if sigma is not None or kl is not None:
                t = self.num_timesteps
                print(f"[MICOA] t={t}  sigma_combined={sigma}  "
                      f"kl_agreement={kl}", flush=True)
        return True


def _get_micoa_extractor(policy):
    """
    Return the MICOAExtractor used by the policy, or None.

    SAC stores features_extractor differently depending on share_features_extractor:
      - share=True:  policy.features_extractor is the shared MICOAExtractor.
      - share=False: policy.features_extractor is None; actor and critic each
                     have their OWN extractor at policy.actor.features_extractor
                     and policy.critic.features_extractor. In that case we
                     return the actor's — that is the one called during
                     model.predict() / inference.

    We accept either layout so MICOA still functions if SB3 changes its default
    or if share_features_extractor is overridden.
    """
    ext = getattr(policy, "features_extractor", None)
    if isinstance(ext, MICOAExtractor):
        return ext
    actor = getattr(policy, "actor", None)
    if actor is not None:
        ext = getattr(actor, "features_extractor", None)
        if isinstance(ext, MICOAExtractor):
            return ext
    return None


# ── SAC subclass that adds the KL agreement loss ──────────────────────────────

from stable_baselines3 import SAC as _SAC
import numpy as np

class MICOASAC(_SAC):
    """
    SAC + MICOA agreement loss.

    After each normal SAC gradient step, computes the symmetric KL between
    the two encoder distributions and adds beta * KL to the actor loss.
    This runs a second backward pass through the actor optimizer, which also
    covers the feature extractor (encoder) weights.

    Why the actor optimizer and not a separate one:
    The encoders feed both the actor and the critic. Tying the KL step to the
    actor optimizer is the minimal change — the critic already pulls the
    encoders toward useful task representations; the actor KL step adds the
    agreement pressure on top.

    beta=0.1 is the starting point. Diagnostic guidance:
      kl_agreement drops fast → beta may be too high (forced agreement)
      kl_agreement never moves → beta too low, increase to 0.3 then 1.0
      sigma_combined decreases while kl_agreement is moderate → corner forming
    """
    def __init__(self, *args, micoa_beta: float = 0.1,
                 micoa_pred_beta: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.micoa_beta      = micoa_beta
        self.micoa_pred_beta = micoa_pred_beta

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps, batch_size)

        # If both MICOA losses are disabled, nothing to do — the policy still
        # uses the MICOAExtractor for inference but no agreement pressure is
        # applied to the encoder weights.
        if self.micoa_beta == 0.0 and self.micoa_pred_beta == 0.0:
            return

        extractor = _get_micoa_extractor(self.policy)
        if extractor is None:
            return

        # Sample one batch and do two forward passes — at t and at t+1.
        # The same batch carries both observations, so we get a consistent
        # (o_t, o_{t+1}) pair for the predictive loss.
        replay_data = self.replay_buffer.sample(
            batch_size, env=self._vec_normalize_env
        )
        obs_t      = replay_data.observations
        obs_t_next = replay_data.next_observations

        was_training = extractor.training
        extractor.train()
        try:
            with torch.set_grad_enabled(True):
                # Forward at t — populates last_mu_p/sigma_p/mu_v/sigma_v with t-distribution
                _ = extractor(obs_t)
                mu_p_t, sigma_p_t = extractor.last_mu_p, extractor.last_sigma_p
                mu_v_t, sigma_v_t = extractor.last_mu_v, extractor.last_sigma_v

                losses = []
                logged = {}

                # --- Same-time symmetric KL (Phase I) -----------------------
                if self.micoa_beta > 0.0:
                    kl_sym = encoder_agreement_loss(
                        mu_p_t, sigma_p_t, mu_v_t, sigma_v_t,
                    )
                    losses.append(self.micoa_beta * kl_sym)
                    logged["kl_agreement"] = float(kl_sym.item())

                # --- Predictive KL (Phase II) -------------------------------
                # vision(t) should look like proprio(t+1). Detach the proprio
                # side so this loss only updates vision encoder weights —
                # vision learns to predict proprio's future, proprio is the
                # ground truth and is unaffected by this loss.
                if self.micoa_pred_beta > 0.0:
                    # Second forward overwrites extractor's last_* attributes;
                    # our locals (mu_v_t, mu_p_t, ...) keep pointing at the
                    # t-tensors with their grad attached, so the symmetric KL
                    # above already captured what it needs.
                    _ = extractor(obs_t_next)
                    mu_p_next    = extractor.last_mu_p.detach()
                    sigma_p_next = extractor.last_sigma_p.detach()
                    kl_pred = kl_gaussian(
                        mu_v_t, sigma_v_t, mu_p_next, sigma_p_next,
                    ).mean()
                    losses.append(self.micoa_pred_beta * kl_pred)
                    logged["kl_predictive"] = float(kl_pred.item())
        finally:
            if not was_training:
                extractor.eval()

        if not losses:
            return

        # --- One backward / one optimizer step ---------------------------
        # With share_features_extractor=True, the extractor is shared between
        # actor and critic. SB3 SAC excludes the shared extractor params from
        # the actor optimizer (to avoid double-stepping with the critic), so
        # we maintain our own optimizer over the extractor params.
        ext_params = list(extractor.parameters())
        if not hasattr(self, "_micoa_opt"):
            try:
                lr = self.policy.actor.optimizer.param_groups[0]["lr"]
            except Exception:
                lr = 1e-4
            self._micoa_opt = torch.optim.Adam(ext_params, lr=lr)

        total_loss = sum(losses)
        self._micoa_opt.zero_grad()
        total_loss.backward()
        self._micoa_opt.step()

        # Logging — both TB record and stdout mirror
        if "kl_agreement" in logged:
            self.logger.record("micoa/kl_agreement_loss",
                               self.micoa_beta * logged["kl_agreement"])
        if "kl_predictive" in logged:
            self.logger.record("micoa/kl_predictive_loss",
                               self.micoa_pred_beta * logged["kl_predictive"])
        parts = []
        if "kl_agreement" in logged:
            parts.append(f"kl_sym={logged['kl_agreement']:.4f}")
        if "kl_predictive" in logged:
            parts.append(f"kl_pred={logged['kl_predictive']:.4f}")
        print(f"[MICOA] total_loss={total_loss.item():.4f}  "
              f"(beta_sym={self.micoa_beta}  beta_pred={self.micoa_pred_beta}  "
              + "  ".join(parts) + ")", flush=True)


# ── Temporal prediction (Phase II — not yet implemented) ─────────────────────
#
# Once PoE + KL training is stable and sigma_combined shows the corner forming,
# the next step is to make vision ANTICIPATORY, not just confirmatory:
#
#   At step t:   vision encodes pixels → mu_v(t)
#   At step t+1: proprio detects contact → mu_p(t+1)
#   Loss: KL( vision_distribution(t) || proprio_distribution(t+1) )
#
# "Vision's encoding NOW should already be close to what proprio will feel NEXT."
# This requires storing (mu_v_t, sigma_v_t) in the replay buffer as extra fields.
# Implement only after Phase I (PoE + KL) shows sigma_combined actually moving.


# ── Wiring into train_crawler.py ─────────────────────────────────────────────
#
#   from alien_baby.agents.micoa_architecture import (
#       MICOAExtractor, MICOASAC, MICOAConfirmationCallback, LATENT_DIM
#   )
#
#   policy_kwargs = dict(
#       features_extractor_class=MICOAExtractor,
#       features_extractor_kwargs=dict(
#           proprio_dim=PROPRIO_DIM,
#           cam_h=CAM_H, cam_w=CAM_W,
#           in_channels=6,
#           latent_dim=LATENT_DIM,
#       ),
#       net_arch=[256, 256],
#   )
#
#   model = MICOASAC("MlpPolicy", train_env, micoa_beta=0.1, **sac_kwargs)
#
#   callback_list.append(MICOAConfirmationCallback(log_freq=500))
#
# Key diagnostics in TensorBoard:
#   micoa/sigma_combined    — shrinks as corner forms (good)
#   micoa/kl_agreement      — moderate and slowly decreasing (good)
#                             drops instantly → beta too high
#                             never moves     → beta too low
#   micoa/kl_agreement_loss — scaled by beta; confirms the loss is registering
