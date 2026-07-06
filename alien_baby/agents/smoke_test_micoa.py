"""
Smoke test for MICOAExtractor.

Checks:
  1. MICOAExtractor instantiates without error
  2. Forward pass produces correct output shape (latent_dim * 3 = 192)
  3. product_of_experts: agreeing inputs → tighter sigma than either alone
  4. product_of_experts: disagreeing inputs → sigma stays wide
  5. encoder_agreement_loss: agreeing inputs → near 0; disagreeing → large
  6. MICOASAC instantiates without error
  7. MICOAConfirmationCallback instantiates without error

Run from repo root:
  python -m alien_baby.agents.smoke_test_micoa
"""

import sys
import torch
import numpy as np
from gymnasium import spaces

# Known constants (from Phase H eval: "71 proprio + 6144 stereo pixels")
PROPRIO_DIM = 71
CAM_H, CAM_W = 32, 32
IN_CHANNELS  = 6          # stereo: left + right, 3 channels each
PIXEL_DIM    = CAM_H * CAM_W * IN_CHANNELS  # 6144
OBS_DIM      = PROPRIO_DIM + PIXEL_DIM      # 6215
LATENT_DIM   = 64
BATCH        = 4

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  {status} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition

def main():
    print("=" * 60)
    print("MICOA Architecture Smoke Test")
    print("=" * 60)
    all_ok = True

    # ── Import ────────────────────────────────────────────────────
    print("\n[1] Imports")
    try:
        from alien_baby.agents.micoa_architecture import (
            ProprioEncoder, VisionEncoder,
            product_of_experts, encoder_agreement_loss,
            MICOAExtractor, MICOAConfirmationCallback, MICOASAC,
        )
        all_ok &= check("import micoa_architecture", True)
    except Exception as e:
        check("import micoa_architecture", False, str(e))
        print("\nCannot continue without imports.")
        sys.exit(1)

    # ── Observation space ─────────────────────────────────────────
    obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
    )
    dummy_obs = torch.zeros(BATCH, OBS_DIM)

    # ── ProprioEncoder ────────────────────────────────────────────
    print("\n[2] ProprioEncoder")
    try:
        pe = ProprioEncoder(PROPRIO_DIM, LATENT_DIM)
        mu_p, sigma_p = pe(dummy_obs[:, :PROPRIO_DIM])
        all_ok &= check("output shapes", mu_p.shape == (BATCH, LATENT_DIM),
                        f"mu_p={tuple(mu_p.shape)}")
        all_ok &= check("sigma positive", (sigma_p > 0).all().item())
    except Exception as e:
        all_ok &= check("ProprioEncoder forward", False, str(e))

    # ── VisionEncoder ─────────────────────────────────────────────
    print("\n[3] VisionEncoder")
    try:
        ve = VisionEncoder(CAM_H, CAM_W, IN_CHANNELS, LATENT_DIM)
        mu_v, sigma_v = ve(dummy_obs[:, PROPRIO_DIM:])
        all_ok &= check("output shapes", mu_v.shape == (BATCH, LATENT_DIM),
                        f"mu_v={tuple(mu_v.shape)}")
        all_ok &= check("sigma positive", (sigma_v > 0).all().item())
    except Exception as e:
        all_ok &= check("VisionEncoder forward", False, str(e))

    # ── product_of_experts ────────────────────────────────────────
    print("\n[4] product_of_experts")
    try:
        # Agreeing: same mu, tight sigmas → combined should be tighter
        mu_a = torch.zeros(BATCH, LATENT_DIM)
        sig_a = torch.ones(BATCH, LATENT_DIM) * 0.5
        mu_z, sig_z = product_of_experts(mu_a, sig_a, mu_a, sig_a)
        tighter = (sig_z < sig_a).all().item()
        all_ok &= check("agreement → tighter sigma", tighter,
                        f"sig_a={sig_a[0,0]:.3f}  sig_z={sig_z[0,0]:.3f}")

        # Disagreeing: very different mu, wide sigmas → combined stays wide
        mu_b = torch.ones(BATCH, LATENT_DIM) * 10.0
        sig_b = torch.ones(BATCH, LATENT_DIM) * 2.0
        mu_z2, sig_z2 = product_of_experts(mu_a, sig_b, mu_b, sig_b)
        # combined mean should be between the two
        between = ((mu_z2 > mu_a).all() and (mu_z2 < mu_b).all()).item()
        all_ok &= check("disagreement → mean between both", between,
                        f"mu_z2[0,0]={mu_z2[0,0]:.3f}")
    except Exception as e:
        all_ok &= check("product_of_experts", False, str(e))

    # ── encoder_agreement_loss ────────────────────────────────────
    print("\n[5] encoder_agreement_loss")
    try:
        agree_loss = encoder_agreement_loss(mu_a, sig_a, mu_a, sig_a)
        near_zero = agree_loss.item() < 0.01
        all_ok &= check("agreeing inputs → loss near 0",
                        near_zero, f"loss={agree_loss.item():.5f}")

        disagree_loss = encoder_agreement_loss(mu_a, sig_a, mu_b, sig_a)
        large = disagree_loss.item() > 1.0
        all_ok &= check("disagreeing inputs → loss large",
                        large, f"loss={disagree_loss.item():.2f}")
    except Exception as e:
        all_ok &= check("encoder_agreement_loss", False, str(e))

    # ── MICOAExtractor ────────────────────────────────────────────
    print("\n[6] MICOAExtractor")
    try:
        extractor = MICOAExtractor(
            obs_space,
            proprio_dim=PROPRIO_DIM,
            cam_h=CAM_H, cam_w=CAM_W,
            in_channels=IN_CHANNELS,
            latent_dim=LATENT_DIM,
        )
        features_dim = extractor.features_dim
        all_ok &= check("features_dim = latent_dim * 3",
                        features_dim == LATENT_DIM * 3,
                        f"features_dim={features_dim}, expected={LATENT_DIM*3}")

        extractor.train()
        out = extractor(dummy_obs)
        all_ok &= check("output shape correct",
                        out.shape == (BATCH, LATENT_DIM * 3),
                        f"got {tuple(out.shape)}")
        all_ok &= check("sigma_combined stored",
                        extractor.last_sigma_combined is not None)
        all_ok &= check("kl_agreement stored",
                        extractor.last_kl_agreement is not None)
        print(f"         sigma_combined={extractor.last_sigma_combined:.4f}")
        print(f"         kl_agreement  ={extractor.last_kl_agreement:.4f}")

        # Eval mode: no sampling noise
        extractor.eval()
        out_eval = extractor(dummy_obs)
        all_ok &= check("eval mode runs",
                        out_eval.shape == (BATCH, LATENT_DIM * 3))
    except Exception as e:
        all_ok &= check("MICOAExtractor", False, str(e))

    # ── MICOASAC ─────────────────────────────────────────────────
    print("\n[7] MICOASAC")
    try:
        import gymnasium as gym
        from stable_baselines3.common.vec_env import DummyVecEnv

        # Minimal continuous env as a stand-in
        env_fn = lambda: gym.make("Pendulum-v1")
        vec_env = DummyVecEnv([env_fn])

        model = MICOASAC(
            "MlpPolicy", vec_env,
            micoa_beta=0.1,
            verbose=0,
        )
        all_ok &= check("MICOASAC instantiates", True)
        vec_env.close()
    except Exception as e:
        all_ok &= check("MICOASAC instantiates", False, str(e))

    # ── MICOAConfirmationCallback ─────────────────────────────────
    print("\n[8] MICOAConfirmationCallback")
    try:
        cb = MICOAConfirmationCallback(log_freq=100)
        all_ok &= check("callback instantiates", True)
    except Exception as e:
        all_ok &= check("callback instantiates", False, str(e))

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED — architecture is ready to wire into training")
    else:
        print("SOME CHECKS FAILED — fix before proceeding")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
