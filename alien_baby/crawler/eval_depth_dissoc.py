"""eval_depth_dissoc.py — does AB's memory-equipped vision encode DISTANCE, and via motion parallax?

Decodes true distance-to-ball from the vision pixel-latent in two conditions:
  MOVING  = policy acts normally -> the strided frames carry self-motion -> parallax available
  FROZEN  = actions held at neutral (hold pose) -> no self-motion -> no parallax (disparity + size only)
Reports latent-R2 (vision) vs proprio-R2 (baseline) in each. MOVING latent-R2 > FROZEN latent-R2 =
AB uses motion parallax for distance. The familiar-size cue is present in BOTH, so it cancels in the
moving-vs-frozen difference.
"""
import argparse, numpy as np, torch
from stable_baselines3 import PPO
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, PROPRIO_DIM
from alien_baby.crawler.train_head_search import CRAWL_POSES


def cv_r2(X, y, folds=5, alpha=1.0):
    X = np.asarray(X); y = np.asarray(y)
    Xa = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    n = len(y); idx = np.arange(n); rng = np.random.default_rng(0); rng.shuffle(idx)
    scores = []
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        A = Xa[tr]; reg = alpha * np.eye(A.shape[1]); reg[-1, -1] = 0.0
        w = np.linalg.solve(A.T @ A + reg, A.T @ y[tr])
        pred = Xa[te] @ w
        ss_res = np.sum((y[te] - pred) ** 2); ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
        scores.append(1 - ss_res / (ss_tot + 1e-9))
    return float(np.mean(scores))


def _lat(fe, o):
    with torch.no_grad():
        return fe._pixel_latent(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))[0].numpy()


def collect(model, env, episodes):
    """Moving rollouts. For each step: full 2-frame latent (HAS motion) vs the SAME view with
    motion removed (past frame = current frame, so no parallax). Same churn + same size cue in
    both, so the only difference is the inter-frame motion = parallax."""
    fe = model.policy.features_extractor
    one = PROPRIO_DIM + (env.observation_space.shape[0] - PROPRIO_DIM) // env.frame_stack  # end of current frame
    lat_full, lat_nomo, prop, dist = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=2000 + ep)
        for _ in range(env.max_steps):
            nomo = obs.copy()
            nomo[one:] = np.tile(obs[PROPRIO_DIM:one], env.frame_stack - 1)  # past := current (no motion)
            lat_full.append(_lat(fe, obs)); lat_nomo.append(_lat(fe, nomo))
            prop.append(obs[:PROPRIO_DIM].copy()); dist.append(float(env._ball_dist()))
            obs, _, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
            if term or trunc:
                break
    return np.array(lat_full), np.array(lat_nomo), np.array(prop), np.array(dist)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--frame-stack", type=int, default=2)
    p.add_argument("--frame-stride", type=int, default=6)
    args = p.parse_args()
    env = MimoCrawlerEnv(
        vision=True, stereo=True, frame_stack=args.frame_stack, frame_stride=args.frame_stride,
        gaze_spawn=True, spawn_cone_deg=136, spawn_radius=(0.35, 0.85), decoy_ball=True,
        action_mode="position_offset", crawl_pose=CRAWL_POSES["arms_fwd"], terminate_tilt_deg=50.0,
        xml_path="alien_baby/crawler/mimo_crawler_pos_wide_prism.xml", max_steps=400)
    model = PPO.load(args.model, device="cpu")
    lat_full, lat_nomo, prop, dist = collect(model, env, args.episodes)
    print(f"=== distance decode (moving rollouts, n={len(dist)}): {args.model.split('/')[-2]} ===")
    print(f"  full 2-frame latent (HAS motion):        R2={cv_r2(lat_full, dist):+.3f}")
    print(f"  same view, motion removed (no parallax): R2={cv_r2(lat_nomo, dist):+.3f}")
    print(f"  proprio baseline:                        R2={cv_r2(prop, dist):+.3f}")
    print("  [full > no-parallax => the inter-frame motion (parallax) adds distance info]")


if __name__ == "__main__":
    main()
