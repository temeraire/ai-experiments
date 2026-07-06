#!/usr/bin/env python3
"""Linear-decode ball position from the MICOA vision latent (mu_v).

Phase V left a sharp question: vision-ablation L2 is 0.6-1.0 (vision strongly
influences actions) yet vision adds nothing to task outcomes, and ablation is
HIGHEST where the ball is dead ahead (direction irrelevant). Two hypotheses:

  (A) REPRESENTATION failure — mu_v never encodes ball direction. Then a linear
      probe trained on mu_v cannot predict ball lateral position above chance,
      despite high ablation. High ablation = vision changes actions, but on
      something other than ball location (e.g. ball presence, self-motion).

  (B) POLICY failure — mu_v DOES encode ball position (probe R^2 is high) but the
      actor's policy gradient never learned to act on it.

This script decides between them. It decodes the ball's EGOCENTRIC position
(ball_x - cart_x lateral; ball_y - cart_y forward) from:
  - mu_v : the vision expert's latent (the thing under test)
  - mu_p : the proprio expert's latent (CONTROL — proprio has no target signal in
           obs, so mu_p should NOT decode ball position; if it can't but mu_v can,
           that is clean evidence for (B); if neither can, that is (A))

Cross-validated Ridge regression, R^2 reported per target and per latent.

Usage:
    python -m alien_baby.visualization.probe_vision_latent \
        --run-tag phase_v_R43_micoa_vision_static_randbox
"""
import argparse
import os
# Force single-threaded BLAS BEFORE numpy/sklearn import — macOS Accelerate +
# the MuJoCo/torch process pool can deadlock sklearn's cross_val_score otherwise.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import pathlib
import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv, PROPRIO_DIM
from alien_baby.agents.micoa_architecture import MICOASAC, _get_micoa_extractor

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
MAX_STEPS = 2000


def build(run_tag, random_ball_box, anchor_y=0.35):
    def _env_fn():
        return MimoCrawlerCartEnv(
            vision=True, strength_scale=1.0, max_steps=MAX_STEPS, cart_speed=0.15,
            ball_speed=0.0, hip_actuation=False, memory_obs=True,
            hunger_base=0.05, hunger_rate=0.20, hunger_scale=500.0,
            fixed_ball_positions=[(0.0, anchor_y), (0.0, -anchor_y)],
            random_ball_box=random_ball_box,
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


def collect(run_tag, n_episodes, steps_per_ep, box, seed, reach_x=None):
    vec, model = build(run_tag, box)
    raw = vec.venv.envs[0]
    extractor = _get_micoa_extractor(model.policy)
    assert extractor is not None, "no MICOA extractor found on policy"
    device = model.policy.device

    def latents(obs_norm):
        """Run the two encoders directly on normalized obs -> (mu_v, mu_p).
        Bypasses predict()'s stored-attribute path (which doesn't reliably
        populate last_mu_*) by calling the encoders ourselves."""
        with torch.no_grad():
            t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
            proprio = t[:, :extractor.proprio_dim]
            pixels = t[:, extractor.proprio_dim:]
            mu_v, _ = extractor.vision_encoder(pixels)
            mu_p, _ = extractor.proprio_encoder(proprio)
        return mu_v.cpu().numpy().reshape(-1), mu_p.cpu().numpy().reshape(-1)

    MU_V, MU_P, Y = [], [], []
    for ep in range(n_episodes):
        obs, _ = raw.reset(seed=seed + ep)
        obs = vec.normalize_obs(obs.reshape(1, -1))
        for t in range(steps_per_ep):
            action, _ = model.predict(obs, deterministic=True)
            mu_v, mu_p = latents(obs)
            # egocentric ball1 position (ball1 is the forward/front ball)
            cart_x = float(raw.data.qpos[raw._cart_x_qadr])
            cart_y = float(raw.data.qpos[raw._cart_y_qadr])
            b1x = float(raw.data.qpos[raw._tgt1_qadr])
            b1y = float(raw.data.qpos[raw._tgt1_qadr + 1])
            x_ego = b1x - cart_x      # lateral offset (the DIRECTION signal)
            y_ego = b1y - cart_y      # forward distance
            # Reachability filter: only keep samples where the ball is in the
            # band AB can actually touch (|x_ego| <= reach_x). Asking vision to
            # encode positions AB can never act on inflates the apparent failure
            # — we only care whether vision encodes REACHABLE direction.
            if reach_x is not None and abs(x_ego) > reach_x:
                obs_raw, _, term, trunc, _ = raw.step(action[0])
                obs = vec.normalize_obs(obs_raw.reshape(1, -1))
                if term or trunc:
                    break
                continue
            MU_V.append(mu_v); MU_P.append(mu_p); Y.append([x_ego, y_ego])
            obs_raw, _, term, trunc, _ = raw.step(action[0])
            obs = vec.normalize_obs(obs_raw.reshape(1, -1))
            if term or trunc:
                break
    vec.close()
    return np.array(MU_V), np.array(MU_P), np.array(Y)


def decode_r2(X, y, alpha=1.0, folds=5):
    # Explicit KFold + manual loop (single-threaded) instead of cross_val_score,
    # which can hang under macOS Accelerate threading.
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    scores = []
    for tr, te in kf.split(X):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], pipe.predict(X[te])))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--steps-per-ep", type=int, default=40)
    ap.add_argument("--box", default="0.25,0.10",
                    help="random ball jitter half-ranges dx,dy (wide -> diverse ball positions)")
    ap.add_argument("--seed", type=int, default=30_000)
    ap.add_argument("--reachable-x", type=float, default=None,
                    help="If set, keep only samples with |ball x_ego| <= this "
                         "(the band AB can actually touch, ~0.20). Tests whether "
                         "vision encodes REACHABLE direction, not unreachable.")
    args = ap.parse_args()
    box = tuple(float(v) for v in args.box.split(","))

    print(f"\n{'='*68}\nVISION-LATENT PROBE: {args.run_tag}\n{'='*68}")
    print(f"Collecting samples (episodes={args.episodes}, steps/ep={args.steps_per_ep}, box={box})...")
    MU_V, MU_P, Y = collect(args.run_tag, args.episodes, args.steps_per_ep, box,
                            args.seed, reach_x=args.reachable_x)
    if args.reachable_x is not None:
        print(f"  reachability filter: |x_ego| <= {args.reachable_x}")
    print(f"  collected {len(Y)} (latent, ball-position) samples")
    print(f"  ball x_ego range [{Y[:,0].min():+.3f}, {Y[:,0].max():+.3f}]  "
          f"y_ego range [{Y[:,1].min():+.3f}, {Y[:,1].max():+.3f}]")

    targets = [("ball x_ego (LATERAL / direction)", 0),
               ("ball y_ego (forward / distance)", 1)]
    print(f"\n{'target':>34} | {'mu_v R^2 (vision)':>18} | {'mu_p R^2 (proprio ctl)':>22}")
    print("-" * 82)
    rows = {}
    for name, j in targets:
        v_m, v_s = decode_r2(MU_V, Y[:, j])
        p_m, p_s = decode_r2(MU_P, Y[:, j])
        rows[name] = (v_m, p_m)
        print(f"{name:>34} | {v_m:>10.3f} ±{v_s:.3f} | {p_m:>13.3f} ±{p_s:.3f}")

    # ---- verdict ----
    vx = rows["ball x_ego (LATERAL / direction)"][0]
    print("\n=== Verdict ===")
    print("Phase V: vision-ablation 0.6-1.0 (vision strongly affects actions) but")
    print("vision adds nothing to outcomes. This probe asks WHY.")
    if vx < 0.15:
        print(f"mu_v decodes ball LATERAL position at R^2={vx:.3f} (~chance).")
        print("=> REPRESENTATION FAILURE (A): vision never encoded ball direction.")
        print("   High ablation is vision acting on something OTHER than ball location.")
        print("   Fix target = the visual encoder / MICOA loss, not the policy.")
    else:
        print(f"mu_v decodes ball LATERAL position at R^2={vx:.3f} (above chance).")
        print("=> POLICY FAILURE (B): direction IS encoded but the actor doesn't use it.")
        print("   Fix target = the RL update / actor, not the encoder.")
    print("Done.")


if __name__ == "__main__":
    main()
