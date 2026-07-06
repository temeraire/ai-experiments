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
        log_sigma = self.log_sigma_head(h).clamp(-2, 2)  # σ in [0.135, 7.4] (was -4,2 → 0.018; R38 hit precision blow-up)
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
        log_sigma = self.log_sigma_head(h).clamp(-2, 2)
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


# ── Replay buffer with ego-target sidecar (Phase XVI) ────────────────────────

from stable_baselines3 import SAC as _SAC
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np


class EgoTargetReplayBuffer(ReplayBuffer):
    """ReplayBuffer that stores a 2-dim ego_xy sidecar alongside each transition.

    The env puts ``ego_xy`` = [x_ego, y_ego] = [ball_x - cart_x, ball_y - cart_y]
    into the ``info`` dict at every step. This buffer reads it from ``infos`` in
    ``add()`` and writes it into a dedicated ``ego_targets`` array.

    ``sample_with_ego(batch_size)`` returns the standard SB3 ReplaySamples tuple
    PLUS the corresponding ``ego_targets`` slice (numpy, shape [B, 2]).

    When ``info["ego_xy"]`` is absent (e.g. non-cart envs or legacy callers),
    the slot is filled with NaN so the aux loss can detect and skip stale entries.

    All existing ReplayBuffer behaviour is preserved — this subclass adds the
    sidecar array only; the standard ``sample()`` path is untouched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sidecar: same time dimension as observations, 2 floats per env per slot.
        self.ego_targets = np.full(
            (self.buffer_size, self.n_envs, 2), np.nan, dtype=np.float32
        )

    def add(self, obs, next_obs, action, reward, done, infos):
        # Write ego_xy from infos before calling super() so pos hasn't advanced yet
        for env_idx, info in enumerate(infos):
            xy = info.get("ego_xy", None)
            if xy is not None:
                self.ego_targets[self.pos, env_idx, 0] = float(xy[0])
                self.ego_targets[self.pos, env_idx, 1] = float(xy[1])
            else:
                self.ego_targets[self.pos, env_idx, :] = np.nan
        super().add(obs, next_obs, action, reward, done, infos)

    def sample_with_ego(self, batch_size: int, env=None):
        """Return (replay_data, ego_targets_np) where ego_targets_np is [B, 2].

        Uses the same random index draw as the parent ``sample()``, applied
        manually here to guarantee the indices are consistent between the
        standard replay data and the ego sidecar.
        """
        if not self.optimize_memory_usage:
            upper = self.buffer_size if self.full else self.pos
            idxs = np.random.randint(0, upper, size=batch_size)
        else:
            # memory-optimised layout: last slot reserved for next_obs
            upper = (self.buffer_size if self.full else self.pos) - 1
            idxs = np.random.randint(0, upper, size=batch_size)
        env_idxs = np.random.randint(0, max(1, self.n_envs), size=batch_size)

        data = self._get_samples(idxs, env=env)
        # Gather sidecar slice: shape [B, 2]
        ego_np = self.ego_targets[idxs, env_idxs, :]
        # Filter out NaN rows (missing ego_xy from env) so the loss is not
        # contaminated by placeholder values.
        valid_mask = ~np.isnan(ego_np[:, 0])
        if not valid_mask.all():
            # Keep only valid rows to avoid poisoning the MSE
            # Note: this shortens the effective batch but is safe.
            valid_idxs = np.where(valid_mask)[0]
            if len(valid_idxs) == 0:
                return data, ego_np  # all NaN — caller should skip
            # Rebuild data with valid rows only (rebuild obs tensor slice)
            import torch as _th
            def _slice(t):
                if isinstance(t, _th.Tensor):
                    return t[valid_idxs]
                return t
            from stable_baselines3.common.type_aliases import ReplayBufferSamples
            data = ReplayBufferSamples(
                observations=_slice(data.observations),
                actions=_slice(data.actions),
                next_observations=_slice(data.next_observations),
                dones=_slice(data.dones),
                rewards=_slice(data.rewards),
            )
            ego_np = ego_np[valid_idxs]
        return data, ego_np


# ── SAC subclass that adds the KL agreement loss ──────────────────────────────

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

    Phase XVI / R49: aux_ball_decode
      When aux_ball_decode=True, a small linear head (BallDecoderHead, 64→2)
      is attached to the vision encoder's latent (mu_v, 64-dim). An MSE loss
      between head(mu_v) and [x_ego, y_ego] — the ball's egocentric position —
      is added to the existing _micoa_opt step (no new optimizer). This gives
      the vision encoder a direct supervised signal to encode ball direction,
      which the plain RL gradient failed to provide.

      x_ego = ball_x - cart_x (lateral / direction signal)
      y_ego = ball_y - cart_y (forward / distance signal)

      These are identical to the probe_vision_latent.py definition so training
      target == eval target (no metric drift).

      The ego_xy targets are stored in the replay buffer as a sidecar array by
      EgoTargetReplayBuffer. The env must put ego_xy into the info dict at each
      step. Default OFF (aux_ball_decode=False) so all existing runs are
      bit-identical.
    """
    def __init__(self, *args, micoa_beta: float = 0.1,
                 micoa_pred_beta: float = 0.0,
                 micoa_pred_horizons=None,
                 aux_ball_decode: bool = False,
                 aux_ball_decode_coef: float = 1.0,
                 aux_ball_decode_target: str = "ego_xy",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.micoa_beta = micoa_beta
        # pred_horizons: list of (k, beta) tuples. Backwards-compat: if not
        # given, fall back to single-horizon t+1 with weight micoa_pred_beta.
        if micoa_pred_horizons is None:
            self.pred_horizons = (
                [(1, micoa_pred_beta)] if micoa_pred_beta > 0.0 else []
            )
        else:
            self.pred_horizons = [(int(k), float(b)) for k, b in micoa_pred_horizons]

        # Phase XVI: auxiliary ball-position decode loss
        self.aux_ball_decode = bool(aux_ball_decode)
        self.aux_ball_decode_coef = float(aux_ball_decode_coef)
        self.aux_ball_decode_target = str(aux_ball_decode_target)

        # BallDecoderHead: linear head maps mu_v (64-dim) → [x_ego, y_ego] (2-dim).
        # Created only when the feature is enabled so all existing runs are
        # bit-identical. Parameters are added to _micoa_opt lazily in train().
        if self.aux_ball_decode:
            self.ball_decoder_head = nn.Linear(LATENT_DIM, 2)
        else:
            self.ball_decoder_head = None

    def _sample_kstep(self, k: int, batch_size: int):
        """Return (obs_t, obs_t_plus_k) tensors from the replay buffer.

        For k=1 we use SB3's standard sample() (which provides obs and
        next_obs and handles VecNormalize correctly). For k>1 we index
        directly into the buffer's observations array. Episode-boundary
        crossings (where t+k lands in a different episode) are tolerated
        as noise — they're a small fraction and average out.
        """
        buf = self.replay_buffer
        if k == 1:
            data = buf.sample(batch_size, env=self._vec_normalize_env)
            return data.observations, data.next_observations

        # k > 1 — custom indexing
        valid = (buf.buffer_size if buf.full else buf.pos) - k
        if valid <= 0:
            # Buffer not full enough; fall back to k=1 for this batch
            data = buf.sample(batch_size, env=self._vec_normalize_env)
            return data.observations, data.next_observations

        idx     = np.random.randint(0, valid, size=batch_size)
        env_idx = np.random.randint(0, buf.n_envs, size=batch_size)
        obs_t_np   = buf._normalize_obs(buf.observations[idx,     env_idx, :],
                                        env=self._vec_normalize_env)
        obs_tk_np  = buf._normalize_obs(buf.observations[idx + k, env_idx, :],
                                        env=self._vec_normalize_env)
        return (
            torch.as_tensor(obs_t_np,  device=self.device, dtype=torch.float32),
            torch.as_tensor(obs_tk_np, device=self.device, dtype=torch.float32),
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps, batch_size)

        # If all MICOA losses are disabled, nothing to do — the policy still
        # uses the MICOAExtractor for inference but no agreement pressure is
        # applied to the encoder weights.
        if self.micoa_beta == 0.0 and not self.pred_horizons and not self.aux_ball_decode:
            return

        extractor = _get_micoa_extractor(self.policy)
        if extractor is None:
            return

        was_training = extractor.training
        extractor.train()
        losses = []
        logged = {}

        try:
            with torch.set_grad_enabled(True):
                # --- Same-time symmetric KL (Phase I) ----------------------
                # Uses standard (o_t, o_{t+1}) sample — only needs o_t here.
                if self.micoa_beta > 0.0:
                    obs_t, _ = self._sample_kstep(1, batch_size)
                    _ = extractor(obs_t)
                    mu_p_t, sigma_p_t = extractor.last_mu_p, extractor.last_sigma_p
                    mu_v_t, sigma_v_t = extractor.last_mu_v, extractor.last_sigma_v
                    kl_sym = encoder_agreement_loss(
                        mu_p_t, sigma_p_t, mu_v_t, sigma_v_t,
                    )
                    losses.append(self.micoa_beta * kl_sym)
                    logged["kl_sym"] = float(kl_sym.item())

                # --- Multi-horizon predictive KL (Phase II/III) ------------
                # For each horizon k, sample fresh (o_t, o_{t+k}) pairs, do
                # two forward passes through the extractor, compute
                # KL(v(t) || p(t+k).detach()) and sum into the total loss.
                # Each horizon's gradient flows only through the vision
                # encoder (proprio side detached).
                for (k, beta_k) in self.pred_horizons:
                    if beta_k <= 0.0:
                        continue
                    obs_t, obs_tk = self._sample_kstep(k, batch_size)

                    # Forward at t — capture vision side; the local refs
                    # to mu_v_t survive the next forward pass.
                    _ = extractor(obs_t)
                    mu_v_t_k    = extractor.last_mu_v
                    sigma_v_t_k = extractor.last_sigma_v

                    # Forward at t+k — capture proprio side (detached)
                    _ = extractor(obs_tk)
                    mu_p_next    = extractor.last_mu_p.detach()
                    sigma_p_next = extractor.last_sigma_p.detach()

                    kl_pred_k = kl_gaussian(
                        mu_v_t_k, sigma_v_t_k, mu_p_next, sigma_p_next,
                    ).mean()
                    losses.append(beta_k * kl_pred_k)
                    logged[f"kl_pred_k{k}"] = float(kl_pred_k.item())

                # --- Phase XVI: auxiliary ball-position decode loss ----------
                # Supervision: head(mu_v) → [x_ego, y_ego].
                # Requires EgoTargetReplayBuffer to have stored ego_xy in the
                # replay buffer. Skipped silently if the buffer doesn't support
                # it (backward-compat: existing SAC/MICOASAC runs unaffected).
                if self.aux_ball_decode and self.ball_decoder_head is not None:
                    # Ensure the decoder head lives on the same device as the
                    # policy (MPS / CUDA / CPU). Do this once lazily because
                    # self.device is not available until after super().__init__.
                    if next(self.ball_decoder_head.parameters()).device != self.device:
                        self.ball_decoder_head = self.ball_decoder_head.to(self.device)
                    buf = self.replay_buffer
                    if hasattr(buf, "sample_with_ego"):
                        data_aux, ego_targets = buf.sample_with_ego(
                            batch_size, env=self._vec_normalize_env
                        )
                        obs_aux = data_aux.observations
                        # Forward through extractor to get mu_v
                        _ = extractor(obs_aux)
                        mu_v_aux = extractor.last_mu_v  # shape [B, 64]
                        # Decode to 2D ego-position
                        pred_ego = self.ball_decoder_head(mu_v_aux)  # [B, 2]
                        target_ego = torch.as_tensor(
                            ego_targets, device=self.device, dtype=torch.float32
                        )  # [B, 2]
                        l_aux = F.mse_loss(pred_ego, target_ego)
                        losses.append(self.aux_ball_decode_coef * l_aux)
                        logged["aux_ball_decode"] = float(l_aux.item())
        finally:
            if not was_training:
                extractor.eval()

        if not losses:
            return

        # --- One backward / one optimizer step ---------------------------
        # With share_features_extractor=True, the extractor is shared between
        # actor and critic. SB3 SAC excludes the shared extractor params from
        # the actor optimizer (to avoid double-stepping with the critic), so
        # we maintain our own optimizer over the extractor params. All
        # horizons' gradients accumulate in one .backward() call.
        # Phase XVI: also include ball_decoder_head params when aux decode
        # is enabled — the head and the vision encoder train together.
        ext_params = list(extractor.parameters())
        if self.ball_decoder_head is not None:
            ext_params = ext_params + list(self.ball_decoder_head.parameters())
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

        # Logging
        if "kl_sym" in logged:
            self.logger.record("micoa/kl_agreement_loss",
                               self.micoa_beta * logged["kl_sym"])
        for key, val in logged.items():
            if key.startswith("kl_pred_k"):
                self.logger.record(f"micoa/{key}", val)
        if "aux_ball_decode" in logged:
            self.logger.record("micoa/aux_ball_decode", logged["aux_ball_decode"])
        parts = [f"{k}={v:.4f}" for k, v in logged.items()]
        print(f"[MICOA] total_loss={total_loss.item():.4f}  "
              + "  ".join(parts), flush=True)


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
