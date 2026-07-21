"""train_prism_regional.py — does prism realignment stay LOCAL to the region acted in?

Taylor (Ch.9 Exp II): wearing the lens, the narrow strip of ground he actually WALKED ON came back
into correct alignment while the rest of the visual field stayed wrong -- "the distortion has to be
ironed out bit by bit." A conv trunk shares weights across the whole image and might realign
globally instead. Our architecture forces neither answer, so the result is information either way.

REUSE, not re-derivation: MismatchPPO and MismatchAuxCallback already exist in train_head_search.py
and are the proven mildhead recipe. MismatchPPO freezes the encoder inside PPO.train(), so REWARD CAN
NEVER RESHAPE THE EYE; a separate Adam drives the encoder from the seen-vs-contacted mismatch alone.
That separation is what makes the arms attributable -- one signal per encoder, not two.

ARMS
  --arm act    : contact-gated aux. Only ACTED-band episodes (where the body actually touched)
                 contribute. This is the main run and Taylor's condition.
  --arm clamp  : aux target supplied NON-CONTINGENTLY (ungated: every episode contributes,
                 touched or not) -- the Morehead et al. 2017 error-clamp analogue, the one design
                 that breaks the circularity because "the error exists but movement did not produce
                 it" is true by construction. If this realigns, Taylor's strong form fails in AB.
  --arm none   : no aux at all. Plumbing control: the encoder cannot change, so bias must stay at
                 the full offset in BOTH bands. If it moves, something else is training the eye.

The encoder starts from the LEARNED mildhead weights and is TRAINABLE (this is recalibration of an
already-competent eye, not acquisition from scratch -- so no freezing, unlike the vbear runs).

  python -m alien_baby.crawler.train_prism_regional --arm act --steps 150000 --run-tag prism_act_s0
"""
import argparse, os, subprocess
import numpy as np, torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.save_util import load_from_zip_file
from alien_baby.crawler.crawler_cnn_extractor import StereoCrawlerCNN
from alien_baby.crawler.prism_bearing_env import PrismBearingEnv, PROP_CUE
from alien_baby.crawler.train_head_search import MismatchPPO, MismatchAuxCallback

MILDHEAD = "alien_baby/results/mildhead_vis_s0_best/best_model.zip"


def _chime():
    try: subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], timeout=5)
    except Exception: pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["act", "clamp", "none"], required=True)
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--prism-deg", type=float, default=15.0,
                   help="NOMINAL camera-frame offset. Effective shift in the READOUT frame is "
                        "smaller (eye tilted ~35deg down): 15 -> +12.6deg measured. Calibrated in "
                        "gate P2; always report the MEASURED value, never the nominal.")
    p.add_argument("--band-hi", type=float, default=0.6,
                   help="0.6 keeps the displaced ghost in view 100%% of resets (winnability, gate P2)")
    p.add_argument("--acted-band", type=int, default=1)
    p.add_argument("--aux-lr", type=float, default=1e-4)
    p.add_argument("--mismatch-coef", type=float, default=1.0)
    p.add_argument("--init-model", default=None,
                   help="Resume from a prior checkpoint instead of the raw mildhead transplant. This "
                        "is how the LENS-OFF PRE-ADAPTATION phase feeds the prism phase: run once "
                        "with --prism-deg 0 to bring the bearing head into the GAZE frame, gate it "
                        "with check_prism_baseline.py, then run the prism phase with --init-model "
                        "pointing at that checkpoint. Without this the first run is spent converting "
                        "the crawler's TORSO-frame readout (~90 deg away) and the 12.6 deg prism "
                        "shift is swamped -- the failure that voided the 2026-07-20 three-arm run.")
    args = p.parse_args()
    os.makedirs("alien_baby/results", exist_ok=True)

    def mk(i):
        return lambda: Monitor(PrismBearingEnv(
            prism_offset_deg=args.prism_deg, acted_band=args.acted_band,
            band_hi=args.band_hi, seed=args.seed + i))

    venv = DummyVecEnv([mk(i) for i in range(args.n_envs)])
    model = MismatchPPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=8, gamma=0.99,
                        gae_lambda=0.95, ent_coef=0.005, learning_rate=3e-4, clip_range=0.2,
                        policy_kwargs=dict(features_extractor_class=StereoCrawlerCNN,
                                           features_extractor_kwargs=dict(proprio_dim=PROP_CUE),
                                           net_arch=[128, 128]),
                        verbose=1, seed=args.seed, device=args.device)

    # Transplant the LEARNED mildhead eye and leave it TRAINABLE. This experiment recalibrates an
    # already-competent eye; freezing it (as the vbear runs do) would make realignment impossible
    # by construction, which is the tautology we are specifically avoiding.
    src = args.init_model or MILDHEAD
    _, params, _ = load_from_zip_file(src, device=args.device)
    if args.init_model:
        # Resuming a prior phase: take the WHOLE policy, so the gaze-frame calibration established
        # in the lens-off phase carries into the prism phase.
        px = dict(params["policy"])
    else:
        px = {k: v for k, v in params["policy"].items()
              if any(s in k for s in ["cnn.", "proj.", "bearing_head."])}
    tgt = model.policy.state_dict()
    ok = {k: v for k, v in px.items() if k in tgt and tuple(tgt[k].shape) == tuple(v.shape)}
    model.policy.load_state_dict(ok, strict=False)
    print(f"[init] loaded {len(ok)}/{len(px)} tensors from {src} (TRAINABLE, not frozen)")
    if len(ok) == 0:
        raise SystemExit("REFUSING TO RUN: zero encoder tensors loaded -- would be probing a random "
                         "net and reporting it as a result (the trap that bit us on the MICOA probe).")

    cbs = [CheckpointCallback(save_freq=25000, save_path=f"alien_baby/results/{args.run_tag}_ckpt",
                              name_prefix=args.run_tag)]
    if args.arm != "none":
        fe = model.policy.features_extractor
        aux_opt = torch.optim.Adam(list(fe.parameters()), lr=args.aux_lr)
        cbs.append(MismatchAuxCallback(aux_opt, args.mismatch_coef,
                                       gate_contact=(args.arm == "act"),
                                       n_envs=args.n_envs, device=args.device))
        print(f"[aux] arm={args.arm} gate_contact={args.arm == 'act'} coef={args.mismatch_coef} "
              f"lr={args.aux_lr}; PPO trains heads only, eye driven by mismatch alone")
    else:
        print("[aux] arm=none -- NO aux. Encoder cannot change; bias must stay at the full offset.")

    print(f"=== PRISM REGIONAL {args.run_tag} arm={args.arm} steps={args.steps} "
          f"nominal={args.prism_deg}deg band_hi={args.band_hi} ===")
    model.learn(total_timesteps=args.steps, callback=cbs, progress_bar=False)
    model.save(f"alien_baby/results/{args.run_tag}_final")
    _chime()
    print(f"=== done: {args.run_tag} ===")


if __name__ == "__main__":
    main()
