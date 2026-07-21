"""preflight_bearing.py — the two cheap checks that must PASS before the spatial-bearing run.

Two questions, both answerable in minutes, both able to silently invalidate a 300K training run:

  (A) PROBE — do the frozen transplanted eyes carry the ball's DIRECTION at all?
      Collect reset frames with the ball at a known gaze-frame bearing, push them through the
      FROZEN conv trunk + proj, ridge-regress latent -> sin(bearing) (and cos), report R^2 on a
      held-out split. If the trunk is direction-blind, freezing it makes the whole experiment
      unanswerable: a null would mean "this trunk lacks direction", not "vision can't steer".
      (Prior: a comparable crawler encoder measured lateral decode R^2 = 0.010 -- chance.)

  (B) FLOOR — how often does a creature that NEVER TURNS still win?
      Success is d < reach centre-to-centre, so driving straight ahead succeeds whenever the
      ball's lateral offset is under `reach`: |bearing| < asin(reach/r). Run a hardcoded
      zero-turn command and MEASURE the success rate, stratified by |bearing|. That measured
      number -- not an estimate -- is the bar the trained sighted policy has to clear.

Both are HYPOTHESIS-GATE checks (CLAUDE.md): measure first, claim after.

  python -m alien_baby.crawler.preflight_bearing --frames 2000 --floor-episodes 60
"""
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from alien_baby.crawler.train_vision_steer import make_bearing_env, PROP


def build_frozen_extractor(encoder_zip, env, device="cpu"):
    """Instantiate the EXACT feature extractor the training run would use, with the crawler's
    conv/proj transplanted in -- so the probe measures the real trunk, not a lookalike."""
    from stable_baselines3.common.save_util import load_from_zip_file
    from alien_baby.crawler.crawler_cnn_extractor import StereoCrawlerCNN
    model = PPO("MlpPolicy", env, device=device,
                policy_kwargs=dict(features_extractor_class=StereoCrawlerCNN,
                                   features_extractor_kwargs=dict(proprio_dim=PROP),
                                   net_arch=[128, 128]))
    _, params, _ = load_from_zip_file(encoder_zip, device=device)
    px = {k: v for k, v in params["policy"].items()
          if any(s in k for s in ["cnn.", "proj.", "bearing_head."])}
    model.policy.load_state_dict(px, strict=False)
    print(f"[probe] transplanted {len(px)} pixel-pathway tensors from {encoder_zip}")
    return model.policy.features_extractor.eval()


def probe(encoder_zip, env, n_frames, seed0, device="cpu"):
    fe = build_frozen_extractor(encoder_zip, env, device)
    lat, bear = [], []
    for i in range(n_frames):
        env.rng = np.random.default_rng(seed0 + i)
        obs, _ = env.reset()
        with torch.no_grad():
            z = fe._pixel_latent(torch.as_tensor(obs, dtype=torch.float32)[None])
        lat.append(z.numpy()[0]); bear.append(env.gaze_bearing())
    X, b = np.array(lat), np.array(bear)
    # regress the latent onto (sin, cos) of the bearing -- angle-safe, and sin is the left/right
    # signed component we actually care about for steering.
    Y = np.stack([np.sin(b), np.cos(b)], 1)
    n_tr = int(0.8 * len(X))
    Xtr, Xte, Ytr, Yte = X[:n_tr], X[n_tr:], Y[:n_tr], Y[n_tr:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.c_[Xtr, np.ones(len(Xtr))]; Xte = np.c_[Xte, np.ones(len(Xte))]
    W = np.linalg.solve(Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)  # ridge
    P = Xte @ W
    r2 = 1.0 - ((Yte - P) ** 2).sum(0) / (((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12)
    return dict(r2_sin=float(r2[0]), r2_cos=float(r2[1]), n=len(X),
                bearing_std=float(b.std()), bearing_range=(float(b.min()), float(b.max())))


def zero_turn_floor(env, episodes, seed0):
    """A creature that only ever drives straight: cmd = [max fwd, no turn]. Its success rate is
    the motor-habit baseline the vision policy must beat."""
    rows = []
    for e in range(episodes):
        env.rng = np.random.default_rng(seed0 + 5000 + e)
        obs, _ = env.reset()
        b0 = abs(env.gaze_bearing())
        done = won = False
        while not done:
            obs, _, term, trunc, info = env.step(np.array([1.0, 0.0], np.float32))  # fwd max, turn 0
            won = bool(info["red"]); done = term or trunc
            if won:
                break
        rows.append((b0, won))
    rows = np.array(rows)
    out = {"overall": 100.0 * rows[:, 1].mean(), "n": episodes}
    for lo, hi in [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]:
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        out[f"|bearing| {lo}-{hi}"] = (f"{100.0 * rows[m, 1].mean():.0f}% (n={int(m.sum())})"
                                      if m.sum() else "n=0")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="alien_baby/results/mildhead_vis_s0_best/best_model.zip")
    p.add_argument("--frames", type=int, default=2000)
    p.add_argument("--floor-episodes", type=int, default=60)
    p.add_argument("--cone", type=float, default=0.9)
    p.add_argument("--reach", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-floor", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    args = p.parse_args()
    env = make_bearing_env(seed=args.seed, cone=args.cone, reach=args.reach)
    print(f"=== PREFLIGHT spatial-bearing  cone=+-{args.cone} rad  reach={args.reach} m ===")
    # geometry note: straight-ahead wins whenever |bearing| < asin(reach/r)
    print(f"  predicted no-steer win cone: {np.arcsin(min(1, args.reach / 2.5)):.2f} rad @2.5m "
          f"/ {np.arcsin(min(1, args.reach / 4.0)):.2f} rad @4.0m  (measured below)")

    if not args.skip_probe:
        r = probe(args.encoder, env, args.frames, args.seed)
        print(f"\n(A) FROZEN-TRUNK BEARING PROBE  n={r['n']}")
        print(f"    bearing spread: std {r['bearing_std']:.2f} rad, range {r['bearing_range'][0]:+.2f}..{r['bearing_range'][1]:+.2f}")
        print(f"    R^2 sin(bearing) [the left/right signal] : {r['r2_sin']:+.3f}")
        print(f"    R^2 cos(bearing)                         : {r['r2_cos']:+.3f}")
        v = ("CARRIES DIRECTION -> proceed frozen" if r["r2_sin"] >= 0.30 else
             "WEAK -> proceed with caution" if r["r2_sin"] >= 0.10 else
             "DIRECTION-BLIND -> do NOT freeze; go to aux-decode / distil-then-RL")
        print(f"    --> {v}")

    if not args.skip_floor:
        f = zero_turn_floor(env, args.floor_episodes, args.seed)
        print(f"\n(B) ZERO-TURN MOTOR-HABIT FLOOR  n={f['n']}")
        print(f"    overall success WITHOUT ever turning: {f['overall']:.0f}%")
        for k in [k for k in f if k.startswith("|bearing|")]:
            print(f"      {k}: {f[k]}")
        print(f"    --> {'floor OK, task needs steering' if f['overall'] <= 30 else 'TOO EASY -- widen cone / tighten reach'}")


if __name__ == "__main__":
    main()
