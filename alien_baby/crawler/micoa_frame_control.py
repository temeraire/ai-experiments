"""ISOLATING CONTROL: run the SAME gaze-frame bearing probe on the MICOA-lineage (Phase V / R43)
vision encoder that we ran on the transplanted crawler encoder (mildhead, R2=0.996).

H1 (representation): MICOA never encoded direction  -> R2 at the random-init floor.
H2 (measurement frame): MICOA did encode it, Phase V probed the wrong frame -> R2 high.

Every encoder sees the IDENTICAL set of frames (collected once), so the comparison is
apples-to-apples. Tensor loading is VERIFIED (name + shape) -- an encoder that does not
genuinely load is reported NOT TESTABLE, never as a low score.
"""
import argparse, sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file

from alien_baby.crawler.train_vision_steer import make_bearing_env, PROP, CAM
from alien_baby.crawler.crawler_cnn_extractor import StereoCrawlerCNN
from alien_baby.agents.micoa_architecture import VisionEncoder


# ---------------------------------------------------------------- loaders
def load_mildhead(zip_path, env, device="cpu"):
    """Crawler-lineage StereoCrawlerCNN. Returns (fn, n_loaded, n_expected, note)."""
    model = PPO("MlpPolicy", env, device=device,
                policy_kwargs=dict(features_extractor_class=StereoCrawlerCNN,
                                   features_extractor_kwargs=dict(proprio_dim=PROP),
                                   net_arch=[128, 128]))
    _, params, _ = load_from_zip_file(zip_path, device=device)
    px = {k: v for k, v in params["policy"].items()
          if any(s in k for s in ["cnn.", "proj.", "bearing_head."])}
    tgt = model.policy.state_dict()
    ok = {k: v for k, v in px.items() if k in tgt and tuple(tgt[k].shape) == tuple(v.shape)}
    bad = [k for k in px if k not in ok]
    model.policy.load_state_dict(ok, strict=False)
    fe = model.policy.features_extractor.eval()
    fn = lambda obs: fe._pixel_latent(obs).detach().numpy()
    return fn, len(ok), len(px), ("all matched" if not bad else f"MISMATCH: {bad[:3]}")


def load_micoa(zip_path, device="cpu", which="mu"):
    """MICOA-lineage VisionEncoder lifted out of the SAC actor. Strict-loads 14 tensors."""
    enc = VisionEncoder(cam_h=CAM, cam_w=CAM, in_channels=6)
    if zip_path is None:                       # random-init floor
        enc.eval()
        return _micoa_fn(enc, which), 0, 0, "RANDOM INIT (floor)"
    _, params, _ = load_from_zip_file(zip_path, device=device)
    pre = "actor.features_extractor.vision_encoder."
    sd = {k[len(pre):]: v for k, v in params["policy"].items() if k.startswith(pre)}
    tgt = enc.state_dict()
    ok = {k: v for k, v in sd.items() if k in tgt and tuple(tgt[k].shape) == tuple(v.shape)}
    bad = [f"{k}{tuple(sd[k].shape)}vs{tuple(tgt[k].shape) if k in tgt else 'MISSING'}"
           for k in sd if k not in ok]
    missing = [k for k in tgt if k not in ok]
    enc.load_state_dict(ok, strict=False)
    enc.eval()
    note = "all matched" if not bad and not missing else f"BAD={bad[:3]} MISSING={missing[:3]}"
    return _micoa_fn(enc, which), len(ok), len(tgt), note


def _micoa_fn(enc, which):
    def fn(obs):
        px = obs[:, PROP:]
        with torch.no_grad():
            if which == "mu":
                return enc(px)[0].numpy()
            return enc.proj(enc.cnn(_stack(enc, px))).numpy()
    return fn


def _stack(enc, px):
    half = CAM * CAM * 3
    l = px[:, :half].view(-1, CAM, CAM, 3).permute(0, 3, 1, 2)
    r = px[:, half:].view(-1, CAM, CAM, 3).permute(0, 3, 1, 2)
    return torch.cat([l, r], 1)


def load_random_crawler(env, device="cpu"):
    model = PPO("MlpPolicy", env, device=device,
                policy_kwargs=dict(features_extractor_class=StereoCrawlerCNN,
                                   features_extractor_kwargs=dict(proprio_dim=PROP),
                                   net_arch=[128, 128]))
    fe = model.policy.features_extractor.eval()
    return (lambda obs: fe._pixel_latent(obs).detach().numpy()), 0, 0, "RANDOM INIT (floor)"


# ---------------------------------------------------------------- probe
def ridge_r2(X, b, seed=0):
    Y = np.stack([np.sin(b), np.cos(b)], 1)
    n_tr = int(0.8 * len(X))
    Xtr, Xte, Ytr, Yte = X[:n_tr], X[n_tr:], Y[:n_tr], Y[n_tr:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.c_[Xtr, np.ones(len(Xtr))]; Xte = np.c_[Xte, np.ones(len(Xte))]
    W = np.linalg.solve(Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    P = Xte @ W
    r2 = 1.0 - ((Yte - P) ** 2).sum(0) / (((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12)
    return float(r2[0]), float(r2[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=1200)
    p.add_argument("--cone", type=float, default=0.9)
    p.add_argument("--reach", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    env = make_bearing_env(seed=a.seed, cone=a.cone, reach=a.reach)
    print(f"collecting {a.frames} reset frames (cone=+-{a.cone}, reach={a.reach}) ...", flush=True)
    OBS, B = [], []
    for i in range(a.frames):
        env.rng = np.random.default_rng(a.seed + i)
        obs, _ = env.reset()
        OBS.append(obs); B.append(env.gaze_bearing())
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{a.frames}", flush=True)
    OBS = torch.as_tensor(np.array(OBS), dtype=torch.float32)
    B = np.array(B)
    print(f"bearing std {B.std():.3f} rad, range {B.min():+.3f}..{B.max():+.3f}\n")

    MILD = "alien_baby/results/mildhead_vis_s0_best/best_model.zip"
    R43 = "alien_baby/results/phase_v_R43_micoa_vision_static_randbox_best/best_model.zip"
    R49 = "alien_baby/results/phase_xvi_R49_micoa_vision_auxdecode_best/best_model.zip"

    specs = [
        ("mildhead_vis_s0 (crawler, TRANSPLANTED)", lambda: load_mildhead(MILD, env)),
        ("R43 MICOA vision mu_v (Phase V)",          lambda: load_micoa(R43, which="mu")),
        ("R43 MICOA vision proj-h (128d)",           lambda: load_micoa(R43, which="proj")),
        ("R49 MICOA vision mu_v (aux-decode)",       lambda: load_micoa(R49, which="mu")),
        ("FLOOR random-init MICOA mu_v",             lambda: load_micoa(None, which="mu")),
        ("FLOOR random-init MICOA proj-h",           lambda: load_micoa(None, which="proj")),
        ("FLOOR random-init crawler trunk",          lambda: load_random_crawler(env)),
    ]

    rows = []
    for name, mk in specs:
        try:
            fn, n_ok, n_exp, note = mk()
        except Exception as e:
            rows.append((name, "LOAD FAIL", "-", "-", repr(e)[:80])); continue
        testable = (n_exp == 0) or (n_ok == n_exp)
        Z = np.concatenate([fn(OBS[i:i+256]) for i in range(0, len(OBS), 256)], 0)
        rs, rc = ridge_r2(Z, B)
        # shuffled-label control: same estimator, destroyed relationship -> must be ~0
        ss, sc = ridge_r2(Z, np.random.default_rng(0).permutation(B))
        note = note + f" [shuffled-label ctrl {ss:+.3f}/{sc:+.3f}]"
        rows.append((name, f"{n_ok}/{n_exp}" if n_exp else "n/a (random)",
                     f"{rs:+.3f}", f"{rc:+.3f}",
                     note if testable else "NOT TESTABLE: " + note))
        print(f"{name:42s} loaded={rows[-1][1]:>8s}  R2sin={rs:+.3f}  R2cos={rc:+.3f}  {rows[-1][4]}",
              flush=True)

    print("\n| encoder | tensors loaded | R2 sin | R2 cos | note |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    main()
