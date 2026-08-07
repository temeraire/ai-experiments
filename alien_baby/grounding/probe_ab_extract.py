"""
probe_ab_extract.py — AB side of the concept-anchoring probe (GROUNDING_LLMS.md §7).

Extract AB's GROUNDED visual latents over a grid of scene states, paired with the
ground-truth sensorimotor invariants for each state. This is the fixed, model-agnostic
half of the probe; the LLM/embedding ladder is aligned against it separately.

For N env resets (the ball spawns across the ±68 cone at varied radius) we record:
  - latent128 : the Stage-A BearingCNN penultimate (head Linear->ReLU, 128-d) = the
                grounded visual representation the policy actually uses
  - convfeat  : the flattened conv output (higher-dim alternative latent)
  - bearing3  : the privileged ego-bearing 3-vector (the ground-truth invariant)
  - theta_deg : signed bearing angle atan2(x, y) in degrees
  - dist      : ego distance (norm of the planar bearing components)

Saved to results/grounding/ab_latents.npz for the alignment step.

Usage:
  PYTHONPATH=<repo> python -u -m alien_baby.grounding.probe_ab_extract --n 400
"""
import argparse
import pathlib

import numpy as np
import torch

from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES, PROPRIO_DIM
from alien_baby.crawler.stage_a_distill import BearingCNN

RESULTS = pathlib.Path(__file__).parent.parent / "results"
XML = "alien_baby/crawler/mimo_crawler_pos_wide.xml"
LO, HI = PROPRIO_DIM, PROPRIO_DIM + 3   # bearing3 slot in obs


def load_encoder(ckpt, device):
    g = BearingCNN().to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    g.load_state_dict(sd)
    g.eval()
    return g


@torch.no_grad()
def latents(g, pix, device):
    """penultimate 128-d and flattened conv feature for a flat stereo pixel vector."""
    x = torch.as_tensor(pix[None], device=device, dtype=torch.float32)
    conv = g.cnn(g._to_img(x))            # [1, n]
    h = g.head[1](g.head[0](conv))        # Linear -> ReLU -> [1, 128]
    return h.cpu().numpy()[0], conv.cpu().numpy()[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(RESULTS / "stage_a_v1" / "bearing_cnn.pt"))
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--cone-deg", type=float, default=136.0)
    p.add_argument("--radius", type=float, nargs=2, default=[0.55, 0.95])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    g = load_encoder(args.ckpt, args.device)
    env = MimoCrawlerEnv(
        vision=True, stereo=True, target_obs=True, crawl_pose=CRAWL_POSES["arms_fwd"],
        action_mode="position_offset", xml_path=XML, spawn_cone_deg=args.cone_deg,
        spawn_radius=tuple(args.radius), random_start_orientation=False, max_steps=1000,
    )
    env.reset(seed=args.seed)

    L128, CONV, B3, TH, DIST = [], [], [], [], []
    for _ in range(args.n):
        obs, _ = env.reset()
        b3 = obs[LO:HI].astype(np.float32)
        pix = obs[HI:].astype(np.float32)
        h, conv = latents(g, pix, args.device)
        L128.append(h); CONV.append(conv); B3.append(b3)
        TH.append(float(np.rad2deg(np.arctan2(b3[0], b3[1]))))
        DIST.append(float(np.linalg.norm(b3[:2])))

    out = RESULTS / "grounding"
    out.mkdir(exist_ok=True)
    f = out / "ab_latents.npz"
    np.savez(f, latent128=np.array(L128), convfeat=np.array(CONV), bearing3=np.array(B3),
             theta_deg=np.array(TH), dist=np.array(DIST))
    th = np.array(TH)
    print(f"wrote {f}  N={args.n}  latent128 dim={np.array(L128).shape[1]}  "
          f"conv dim={np.array(CONV).shape[1]}")
    print(f"  bearing spread: theta {th.min():.0f}..{th.max():.0f} deg, "
          f"dist {min(DIST):.2f}..{max(DIST):.2f}")


if __name__ == "__main__":
    main()
