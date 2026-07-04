"""
stage_a_distill.py — Stage A of the vision-as-reinforcement plan.

Distil the ball-bearing (the privileged `target_obs` 3-vector) from the head-camera
PIXELS into a CNN, then feed the CNN's prediction into the FROZEN crawl_ppo_2M motor
policy in place of the privileged signal. The motor policy is never touched, so the
proprioceptive crawl gait is preserved BY DEFINITION ("reinforced, not destroyed").
If the CNN reconstructs the bearing, we recover the teacher's contact rate for free;
if it is noisy we degrade gracefully.

Recipe (Learning by Cheating, Chen et al. 2019 + DAgger, per literature-scout 2026-07-04):
  - teacher = crawl_ppo_2M_best driving with the TRUE bearing (covers good states)
  - collect (pixels -> true ego-bearing); train CNN g by regression
  - DAgger rounds: roll out with g's PREDICTED bearing, relabel with truth, retrain
  - eval by injecting g(pixels) into obs slot [69:72] through the saved VecNormalize

Success (todo.md contract): decode-R2 > 0.30 AND student contact rate recovers toward the
teacher ceiling, with the motor substrate unchanged (it is frozen, so this is guaranteed).

Usage:
  PYTHONPATH=<repo> python -m alien_baby.crawler.stage_a_distill --run-tag stage_a_v1
"""
import argparse
import pathlib
import pickle
import subprocess

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES, CAM_H, CAM_W, PROPRIO_DIM

RESULTS = pathlib.Path(__file__).parent.parent / "results"
XML = "alien_baby/crawler/mimo_crawler_pos_wide.xml"
TARGET_LO, TARGET_HI = PROPRIO_DIM, PROPRIO_DIM + 3   # slot [69:72]


def _device(pref=None):
    if pref:
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _chime():
    try:
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], timeout=5)
    except Exception:
        pass


class BearingCNN(nn.Module):
    """DrQ-v2 4-conv encoder on the stereo pair -> 3-vector ego bearing (raw units)."""

    def __init__(self, cam_h=CAM_H, cam_w=CAM_W):
        super().__init__()
        self.cam_h, self.cam_w = cam_h, cam_w
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, 6, cam_h, cam_w)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 3),
        )

    def _to_img(self, pixels_flat):
        half = self.cam_h * self.cam_w * 3
        l = pixels_flat[:, :half].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
        r = pixels_flat[:, half:].view(-1, self.cam_h, self.cam_w, 3).permute(0, 3, 1, 2)
        return torch.cat([l, r], dim=1)

    def forward(self, pixels_flat):
        return self.head(self.cnn(self._to_img(pixels_flat)))


def make_env(cone_deg, radius, max_steps, seed):
    env = MimoCrawlerEnv(
        vision=True, stereo=True, target_obs=True, crawl_pose=CRAWL_POSES["arms_fwd"],
        action_mode="position_offset", xml_path=XML, spawn_cone_deg=cone_deg,
        spawn_radius=tuple(radius), random_start_orientation=False, max_steps=max_steps,
    )
    env.reset(seed=seed)
    return env


def load_teacher(device):
    # Teacher is an MLP policy: CPU predict is faster per-step than MPS dispatch, and it
    # is called once per env step (the collection/eval bottleneck). Keep it on CPU.
    teacher = PPO.load(str(RESULTS / "crawl_ppo_2M_best" / "best_model.zip"), device="cpu")
    with open(RESULTS / "crawl_ppo_2M" / "vec_normalize.pkl", "rb") as f:
        vn = pickle.load(f)
    mean, var = vn.obs_rms.mean.astype(np.float32), vn.obs_rms.var.astype(np.float32)
    clip, eps = float(vn.clip_obs), float(vn.epsilon)

    def normalize72(o72):
        return np.clip((o72 - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)

    return teacher, normalize72


def teacher_action(teacher, normalize72, proprio69, bearing3):
    o72 = np.concatenate([proprio69, bearing3]).astype(np.float32)
    act, _ = teacher.predict(normalize72(o72), deterministic=True)
    return act


def split_obs(obs):
    return obs[:TARGET_LO], obs[TARGET_LO:TARGET_HI], obs[TARGET_HI:]


def collect(env, teacher, normalize72, g, device, n_eps, bearing_source, max_steps):
    """Roll out; record (pixels, true_bearing). bearing_source drives the policy:
    'true' = teacher-in-the-loop (BC data); 'student' = g-in-the-loop (DAgger data)."""
    P, B = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        for _ in range(max_steps):
            proprio, true_b, pix = split_obs(obs)
            P.append(pix.copy()); B.append(true_b.copy())
            if bearing_source == "student":
                with torch.no_grad():
                    pb = g(torch.as_tensor(pix[None], device=device)).cpu().numpy()[0]
                drive_b = pb
            else:
                drive_b = true_b
            act = teacher_action(teacher, normalize72, proprio, drive_b)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
    return np.array(P, np.float32), np.array(B, np.float32)


def train_cnn(g, device, P, B, epochs, bs=256, lr=3e-4):
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    Pt = torch.as_tensor(P, device=device); Bt = torch.as_tensor(B, device=device)
    n = len(P); idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        tot = 0.0
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            pred = g(Pt[j])
            loss = ((pred - Bt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j)
    return tot / n


def r2_per_axis(g, device, P, B):
    with torch.no_grad():
        pred = g(torch.as_tensor(P, device=device)).cpu().numpy()
    out = []
    for k in range(3):
        ss_res = ((B[:, k] - pred[:, k]) ** 2).sum()
        ss_tot = ((B[:, k] - B[:, k].mean()) ** 2).sum() + 1e-8
        out.append(1.0 - ss_res / ss_tot)
    return out  # [x(lateral), y(forward), z]


def eval_policy(env, teacher, normalize72, g, device, n_eps, max_steps,
                bearing_source, ablate_pixels=False):
    """bearing_source: 'true' (ceiling), 'student' (Stage A), 'zero' (blind floor)."""
    contacts = 0
    disps, dtoward = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        proprio0, _, _ = split_obs(obs)
        root0 = proprio0[:2].copy()
        d0 = env._ball_dist()
        touched = False
        for _ in range(max_steps):
            proprio, true_b, pix = split_obs(obs)
            if bearing_source == "true":
                drive_b = true_b
            elif bearing_source == "zero":
                drive_b = np.zeros(3, np.float32)
            else:
                px = np.zeros_like(pix) if ablate_pixels else pix
                with torch.no_grad():
                    drive_b = g(torch.as_tensor(px[None], device=device)).cpu().numpy()[0]
            act = teacher_action(teacher, normalize72, proprio, drive_b)
            obs, _, term, trunc, info = env.step(act)
            if info.get("touched_ball1"):
                touched = True
            if term or trunc:
                break
        proprio, _, _ = split_obs(obs)
        disps.append(float(np.linalg.norm(proprio[:2] - root0)))
        dtoward.append(d0 - env._ball_dist())
        contacts += int(touched)
    return dict(contact_rate=contacts / n_eps, n=n_eps,
                mean_disp=float(np.mean(disps)), mean_toward=float(np.mean(dtoward)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", default="stage_a_v1")
    p.add_argument("--cone-deg", type=float, default=44.0)   # +/-22
    p.add_argument("--radius", type=float, nargs=2, default=[0.70, 0.80])
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--bc-eps", type=int, default=120)
    p.add_argument("--dagger-rounds", type=int, default=2)
    p.add_argument("--dagger-eps", type=int, default=60)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--eval-eps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu avoids MPS contention with a concurrent GPU run")
    args = p.parse_args()

    device = _device(args.device)
    tag = args.run_tag
    print(f"\n=== Stage A distillation: {tag} (device={device}) ===")
    print(f"  cone=+/-{args.cone_deg/2:.0f} radius={args.radius} "
          f"bc_eps={args.bc_eps} dagger={args.dagger_rounds}x{args.dagger_eps}")

    teacher, normalize72 = load_teacher(device)
    env = make_env(args.cone_deg, args.radius, args.max_steps, args.seed)
    eval_env = make_env(args.cone_deg, args.radius, args.max_steps, args.seed + 777)
    g = BearingCNN().to(device)

    # Benchmarks first (frozen policy; measured on the +/-22 cone).
    ceil = eval_policy(eval_env, teacher, normalize72, g, device, args.eval_eps,
                       args.max_steps, "true")
    floor = eval_policy(eval_env, teacher, normalize72, g, device, args.eval_eps,
                        args.max_steps, "zero")
    print(f"\n[BENCHMARK] teacher TRUE-bearing (ceiling): {ceil['contact_rate']*100:.1f}% "
          f"contacts, mean_toward={ceil['mean_toward']:.3f}")
    print(f"[BENCHMARK] teacher ZERO-bearing (blind floor): {floor['contact_rate']*100:.1f}% "
          f"contacts, mean_toward={floor['mean_toward']:.3f}")

    # Round 0: behaviour cloning on teacher-driven data.
    print("\n[collect] BC round (teacher-driven) ...")
    P, B = collect(env, teacher, normalize72, g, device, args.bc_eps, "true", args.max_steps)
    print(f"  collected {len(P)} samples")
    loss = train_cnn(g, device, P, B, args.epochs)
    print(f"  train MSE={loss:.4f}  R2(x_lat,y_fwd,z)={[f'{r:.3f}' for r in r2_per_axis(g, device, P, B)]}")

    # DAgger rounds: student-in-the-loop, relabel with truth.
    for rd in range(args.dagger_rounds):
        print(f"\n[collect] DAgger round {rd+1} (student-driven) ...")
        Pn, Bn = collect(env, teacher, normalize72, g, device, args.dagger_eps,
                         "student", args.max_steps)
        P = np.concatenate([P, Pn]); B = np.concatenate([B, Bn])
        loss = train_cnn(g, device, P, B, args.epochs)
        r2 = r2_per_axis(g, device, P, B)
        print(f"  agg {len(P)} samples  MSE={loss:.4f}  R2(x_lat,y_fwd,z)={[f'{r:.3f}' for r in r2]}")

    # Held-out decode R2 (fresh rollout, student-driven).
    Ph, Bh = collect(env, teacher, normalize72, g, device, 20, "student", args.max_steps)
    r2_ho = r2_per_axis(g, device, Ph, Bh)

    # Final behavioural eval: student bearing, plus vision-ablation.
    stud = eval_policy(eval_env, teacher, normalize72, g, device, args.eval_eps,
                       args.max_steps, "student")
    abl = eval_policy(eval_env, teacher, normalize72, g, device, args.eval_eps,
                      args.max_steps, "student", ablate_pixels=True)

    print("\n==================== STAGE A RESULT ====================")
    print(f"  decode R2 held-out (x_lat, y_fwd, z): {[f'{r:.3f}' for r in r2_ho]}")
    print(f"  LATERAL R2 (the bar, >0.30 to pass): {r2_ho[0]:.3f}")
    print(f"  contacts: TRUE {ceil['contact_rate']*100:.1f}%  |  STUDENT "
          f"{stud['contact_rate']*100:.1f}%  |  BLIND {floor['contact_rate']*100:.1f}%  "
          f"|  STUDENT+pixels-zeroed {abl['contact_rate']*100:.1f}%")
    print(f"  mean_toward: TRUE {ceil['mean_toward']:.3f} STUDENT {stud['mean_toward']:.3f} "
          f"BLIND {floor['mean_toward']:.3f} ABLATED {abl['mean_toward']:.3f}")
    recover = ((stud['contact_rate'] - floor['contact_rate']) /
               max(1e-6, ceil['contact_rate'] - floor['contact_rate']))
    print(f"  recovery of ceiling over floor: {recover*100:.0f}%")
    passed = r2_ho[0] > 0.30 and stud['contact_rate'] > floor['contact_rate'] + 0.10
    print(f"  STAGE A PASS (lat R2>0.30 AND student>floor+10pts): {passed}")
    print("=======================================================")

    outdir = RESULTS / tag; outdir.mkdir(parents=True, exist_ok=True)
    torch.save(g.state_dict(), outdir / "bearing_cnn.pt")
    np.savez(outdir / "result.npz", r2_ho=r2_ho, ceil=ceil['contact_rate'],
             student=stud['contact_rate'], floor=floor['contact_rate'],
             ablated=abl['contact_rate'])
    print(f"  saved CNN + result to {outdir}")
    env.close(); eval_env.close()
    _chime()


if __name__ == "__main__":
    main()
