#!/usr/bin/env python3
"""Categorical vs metric distance generalization (desensitization model).

Motivated by the systematic-desensitization observation (2026-05-30 discussion):
anxiety is NOT linear in height — past some point the organism stops discriminating
("30th vs 50th floor both = 'high'"), so a few exemplars generalize to all. The
analog here: does AB's policy treat ball distance METRICALLY (behavior changes
smoothly and distinguishably at every distance) or CATEGORICALLY (beyond some
point, distances collapse into one "far" response)?

Test: for a sweep of forward ball distances, collect the policy's deterministic
START action vector across many seeds at each distance. Then for ADJACENT distance
pairs compute a discriminability d' :

    d'(A,B) = ||mean_action(A) - mean_action(B)|| / pooled_std

- METRIC generalization: d' roughly constant across the whole range (every
  distance step is equally distinguishable).
- CATEGORICAL generalization: d' is high among near distances (where reaching
  geometry differs sharply) but COLLAPSES toward 0 among far distances — the
  policy runs the same "far → paddle" program and can't tell them apart.

Also reports mean time-to-contact per distance (the metric view) so the two
readings can be compared side by side.

Works on plain-SAC proprio runs (R44) and MICOA runs (R43). Uses the action
output as the behavioral signal — model-agnostic, interpretable.

Usage:
    python -m alien_baby.visualization.categorical_distance_test \
        --run-tag phase_v_R44_proprio_static_randbox
"""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import pathlib
import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from alien_baby.crawler.mimo_crawler_cart_env import MimoCrawlerCartEnv
from alien_baby.agents.micoa_architecture import MICOASAC

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
MAX_STEPS = 2000
START_ACTIONS = 5   # average the first N deterministic actions as the "response"


def build(run_tag, dist):
    vision = ("vision" in run_tag) or ("micoa" in run_tag)
    micoa = "micoa" in run_tag

    def _env_fn():
        return MimoCrawlerCartEnv(
            vision=vision, strength_scale=1.0, max_steps=MAX_STEPS, cart_speed=0.15,
            ball_speed=0.0, hip_actuation=False, memory_obs=True,
            hunger_base=0.05, hunger_rate=0.20, hunger_scale=500.0,
            fixed_ball_positions=[(0.0, dist), (0.0, -dist)],
        )
    raw = DummyVecEnv([_env_fn])
    vn = RESULTS_DIR / run_tag / "vec_normalize.pkl"
    vec = VecNormalize.load(str(vn), raw)
    vec.training = False
    vec.norm_reward = False
    ckpt = RESULTS_DIR / run_tag / "final_model.zip"
    if not ckpt.exists():
        ckpt = RESULTS_DIR / f"{run_tag}_best" / "best_model.zip"
    loader = MICOASAC if micoa else SAC
    model = loader.load(str(ckpt), env=vec, device="cpu")
    return vec, model


def collect_responses(run_tag, dist, n_seeds, seed0):
    """Return (start_actions[n_seeds, act_dim], contact_steps[list])."""
    vec, model = build(run_tag, dist)
    raw = vec.venv.envs[0]
    starts, contacts = [], []
    for s in range(n_seeds):
        obs, _ = raw.reset(seed=seed0 + s)
        obs = vec.normalize_obs(obs.reshape(1, -1))
        acts = []
        contact = None
        for t in range(MAX_STEPS):
            a, _ = model.predict(obs, deterministic=True)
            if t < START_ACTIONS:
                acts.append(a.reshape(-1))
            obs_raw, _, term, trunc, info = raw.step(a[0])
            obs = vec.normalize_obs(obs_raw.reshape(1, -1))
            if (info.get("touched_ball1") or info.get("touched_ball2")) and contact is None and t > 0:
                contact = t
            if term or trunc:
                break
        starts.append(np.mean(acts, axis=0))
        if contact is not None:
            contacts.append(contact)
    vec.close()
    return np.array(starts), contacts


def dprime(A, B):
    """Discriminability between two sets of action vectors."""
    mean_diff = np.linalg.norm(A.mean(0) - B.mean(0))
    pooled = np.sqrt(0.5 * (A.var(0).sum() + B.var(0).sum())) + 1e-9
    return float(mean_diff / pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--distances", default="0.30,0.40,0.50,0.60,0.70,0.80")
    ap.add_argument("--n-seeds", type=int, default=24)
    ap.add_argument("--seed", type=int, default=40_000)
    args = ap.parse_args()
    dists = [float(x) for x in args.distances.split(",")]

    print(f"\n{'='*70}\nCATEGORICAL-DISTANCE TEST: {args.run_tag}\n{'='*70}")
    print(f"distances={dists}  n_seeds={args.n_seeds}  start_actions_avg={START_ACTIONS}")

    responses, contact_means = {}, {}
    for d in dists:
        starts, contacts = collect_responses(args.run_tag, d, args.n_seeds, args.seed)
        responses[d] = starts
        contact_means[d] = (np.mean(contacts) if contacts else float("nan"),
                            len(contacts))

    print(f"\n{'distance':>9} | {'mean_contact_step':>17} | {'n_touched':>9}")
    print("-" * 42)
    for d in dists:
        cm, n = contact_means[d]
        print(f"{d:>9.2f} | {cm:>17.1f} | {n:>9}")

    print(f"\nAdjacent-pair discriminability d' (action response):")
    print(f"{'pair':>14} | {'d-prime':>8}")
    print("-" * 26)
    dprimes = []
    for i in range(len(dists) - 1):
        a, b = dists[i], dists[i + 1]
        dp = dprime(responses[a], responses[b])
        dprimes.append((a, b, dp))
        print(f"{a:.2f}->{b:.2f}".rjust(14) + f" | {dp:>8.3f}")

    # ---- verdict ----
    print("\n=== Verdict (metric vs categorical) ===")
    near_half = dprimes[:len(dprimes) // 2]
    far_half = dprimes[len(dprimes) // 2:]
    near_mean = np.mean([d for *_, d in near_half]) if near_half else float("nan")
    far_mean = np.mean([d for *_, d in far_half]) if far_half else float("nan")
    print(f"near-range mean d' = {near_mean:.3f}   far-range mean d' = {far_mean:.3f}")
    if np.isfinite(near_mean) and np.isfinite(far_mean) and near_mean > 1e-6:
        ratio = far_mean / near_mean
        print(f"far/near d' ratio = {ratio:.2f}")
        if ratio < 0.5:
            print("=> CATEGORICAL: discriminability COLLAPSES at far distances.")
            print("   The policy stops telling far distances apart — it runs one")
            print("   'far' response. This is the desensitization / floor-collapse")
            print("   pattern: past a point, distances become one category.")
        else:
            print("=> METRIC: discriminability roughly preserved across the range.")
            print("   The policy distinguishes every distance step ~equally — a")
            print("   smooth metric map, not a saturating categorical one.")
    print("Done.")


if __name__ == "__main__":
    main()
