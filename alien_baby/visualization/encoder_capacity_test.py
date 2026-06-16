#!/usr/bin/env python3
"""Encoder capacity test — can the vision CNN encode reachable ball direction AT ALL?

The vision-latent probe showed mu_v decodes ball lateral direction only weakly
(R² ~0.08-0.14) under the trained MICOA encoder. Two explanations:

  (LOSS-GEOMETRY limit) The 32x32 CNN *can* represent ball direction, but MICOA's
    predictive-KL target (proprio(t+1)) never rewarded it for doing so under static
    balls — so it didn't. Fixable by changing the loss/task.

  (CAPACITY limit) The 32x32 stereo CNN physically cannot resolve ball direction
    well enough. No loss change would help; need higher resolution.

This decides between them with a direct supervised objective:
  1. Collect (pixels, egocentric ball position) pairs from R43 rollouts.
  2. FROZEN encoder: train ONLY a fresh linear head mu_v -> (x_ego, y_ego).
     (Reproduces the probe; lower bound = what MICOA training actually produced.)
  3. UNFROZEN encoder: train the CNN end-to-end on the SAME supervised target.
     (Upper bound = what the architecture CAN represent if the loss asks for it.)

Reading:
  - frozen low AND unfrozen low  -> CAPACITY limit (raise resolution).
  - frozen low BUT unfrozen high -> LOSS-GEOMETRY limit (the encoder can do it; the
    MICOA objective just never asked). Fix the loss/task, not the camera.

Restricted to the reachable band (|x_ego| <= reach_x) per the "one thing at a
time" principle — we only ask whether vision can encode REACHABLE direction.

Usage:
    python -m alien_baby.visualization.encoder_capacity_test \
        --run-tag phase_v_R43_micoa_vision_static_randbox
"""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import pathlib
import copy
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(1)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv
from alien_baby.agents.micoa_architecture import MICOASAC, _get_micoa_extractor

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
MAX_STEPS = 2000


def build(run_tag, box, anchor_y=0.35):
    def _env_fn():
        return MimoCrawlerCartEnv(
            vision=True, strength_scale=1.0, max_steps=MAX_STEPS, cart_speed=0.15,
            ball_speed=0.0, hip_actuation=False, memory_obs=True,
            hunger_base=0.05, hunger_rate=0.20, hunger_scale=500.0,
            fixed_ball_positions=[(0.0, anchor_y), (0.0, -anchor_y)],
            random_ball_box=box,
        )
    raw = DummyVecEnv([_env_fn])
    vn = RESULTS_DIR / run_tag / "vec_normalize.pkl"
    vec = VecNormalize.load(str(vn), raw)
    vec.training = False
    vec.norm_reward = False
    ckpt = RESULTS_DIR / run_tag / "final_model.zip"
    if not ckpt.exists():
        ckpt = RESULTS_DIR / f"{run_tag}_best" / "best_model.zip"
    model = MICOASAC.load(str(ckpt), env=vec, device="cpu")
    return vec, model


def collect(run_tag, n_episodes, steps_per_ep, box, seed, reach_x):
    """Collect raw normalized PIXEL vectors + egocentric ball targets."""
    vec, model = build(run_tag, box)
    raw = vec.venv.envs[0]
    extractor = _get_micoa_extractor(model.policy)
    pdim = extractor.proprio_dim
    PIX, Y = [], []
    for ep in range(n_episodes):
        obs, _ = raw.reset(seed=seed + ep)
        obs = vec.normalize_obs(obs.reshape(1, -1))
        for t in range(steps_per_ep):
            a, _ = model.predict(obs, deterministic=True)
            cart_x = float(raw.data.qpos[raw._cart_x_qadr])
            cart_y = float(raw.data.qpos[raw._cart_y_qadr])
            x_ego = float(raw.data.qpos[raw._tgt1_qadr]) - cart_x
            y_ego = float(raw.data.qpos[raw._tgt1_qadr + 1]) - cart_y
            if reach_x is None or abs(x_ego) <= reach_x:
                PIX.append(obs[0, pdim:].copy())   # normalized pixel vector
                Y.append([x_ego, y_ego])
            obs_raw, _, term, trunc, _ = raw.step(a[0])
            obs = vec.normalize_obs(obs_raw.reshape(1, -1))
            if term or trunc:
                break
    vec.close()
    return np.array(PIX, dtype=np.float32), np.array(Y, dtype=np.float32), extractor


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0) + 1e-9
    return (1.0 - ss_res / ss_tot)  # per-target


def train_head(encoder, head, X, Y, *, train_encoder, epochs=60, bs=256, lr=1e-3):
    params = list(head.parameters()) + (list(encoder.parameters()) if train_encoder else [])
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.MSELoss()
    n = len(X)
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(Y)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], Yt[idx]
            if train_encoder:
                mu, _ = encoder(xb)
            else:
                with torch.no_grad():
                    mu, _ = encoder(xb)
            opt.zero_grad()
            loss = lossf(head(mu), yb)
            loss.backward()
            opt.step()
    return head


def evaluate(encoder, head, X, Y):
    with torch.no_grad():
        mu, _ = encoder(torch.as_tensor(X))
        pred = head(mu).numpy()
    return r2(pred, Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="phase_v_R43_micoa_vision_static_randbox")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--steps-per-ep", type=int, default=40)
    ap.add_argument("--box", default="0.25,0.12")
    ap.add_argument("--reachable-x", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=50_000)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    box = tuple(float(v) for v in args.box.split(","))

    print(f"\n{'='*70}\nENCODER CAPACITY TEST: {args.run_tag}\n{'='*70}")
    print(f"Collecting (pixels, ball_ego) pairs, reach_x={args.reachable_x}...")
    X, Y, extractor = collect(args.run_tag, args.episodes, args.steps_per_ep,
                              box, args.seed, args.reachable_x)
    print(f"  {len(X)} samples; x_ego range [{Y[:,0].min():+.3f},{Y[:,0].max():+.3f}]")

    # split
    n = len(X); ntr = int(0.8 * n)
    idx = np.random.RandomState(0).permutation(n)
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
    latent_dim = extractor.latent_dim

    def fresh_head():
        return nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 2))

    # --- FROZEN encoder (what MICOA training produced) ---
    enc_frozen = extractor.vision_encoder
    head1 = fresh_head()
    train_head(enc_frozen, head1, Xtr, Ytr, train_encoder=False, epochs=args.epochs)
    r2_frozen = evaluate(enc_frozen, head1, Xte, Yte)

    # --- UNFROZEN encoder (deep-copied so we don't mutate the loaded model) ---
    enc_unfrozen = copy.deepcopy(extractor.vision_encoder)
    head2 = fresh_head()
    train_head(enc_unfrozen, head2, Xtr, Ytr, train_encoder=True, epochs=args.epochs)
    r2_unfrozen = evaluate(enc_unfrozen, head2, Xte, Yte)

    print(f"\n{'target':>28} | {'FROZEN R²':>10} | {'UNFROZEN R²':>11}")
    print("-" * 56)
    for name, j in [("ball x_ego (LATERAL/dir)", 0), ("ball y_ego (forward/dist)", 1)]:
        print(f"{name:>28} | {r2_frozen[j]:>10.3f} | {r2_unfrozen[j]:>11.3f}")

    fx, ux = float(r2_frozen[0]), float(r2_unfrozen[0])
    print("\n=== Verdict (capacity vs loss-geometry) ===")
    print(f"lateral-direction R²: frozen={fx:.3f}  unfrozen={ux:.3f}")
    if ux >= 0.5 and ux - fx >= 0.2:
        print("=> LOSS-GEOMETRY LIMIT: the 32x32 CNN CAN encode reachable ball")
        print("   direction when a supervised objective asks for it. MICOA's")
        print("   predictive-KL (proprio(t+1)) simply never asked under static balls.")
        print("   FIX = the loss/task (add a direction-relevant target), not the camera.")
    elif ux < 0.5:
        print("=> CAPACITY LIMIT: even end-to-end supervised, the 32x32 CNN cannot")
        print("   resolve reachable ball direction well. FIX = higher camera resolution.")
    else:
        print("=> AMBIGUOUS: unfrozen improves but stays moderate; partial capacity.")
    print("Done.")


if __name__ == "__main__":
    main()
