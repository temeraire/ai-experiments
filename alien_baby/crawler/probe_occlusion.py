"""probe_occlusion.py — is the steering REACTIVE, or is a held estimate of the target driving it?

Recommended by the theory-monitor as the single most informative follow-up to the 2026-07-20
spatial-bearing result. Mid-approach, we take vision away (noise pixels) for a block of steps and
ask what the creature does with the turn command it was already executing.

  - If steering PERSISTS toward the last-seen bearing, something is holding an estimate of where
    the ball was -- a memory, and the beginnings of a predictive model.
  - If steering COLLAPSES to blind-level noise the instant vision goes, the behaviour is pure
    closed-loop visual servoing: "turn toward wherever the blob is on my retina RIGHT NOW", with no
    representation of where the ball is going to be.

IMPORTANT -- the answer is architecturally constrained, and saying so up front is the point. The
driver is an SB3 ActorCriticPolicy: Conv2d/Linear/LayerNorm/ReLU/Tanh, NO LSTM or GRU, no recurrent
state. It maps the CURRENT frame to an action with zero memory. So it CANNOT hold an estimate across
steps, and collapse is the only outcome it can produce. This probe is therefore a CONFIRMATION that
the architecture behaves as its wiring says, not a live test between two open hypotheses -- and it
exists so the writeup claims reactive servoing rather than anything predictive. Persistence here
would mean something is wrong with our understanding of the setup, not that AB has memory.

  python -m alien_baby.crawler.probe_occlusion --model alien_baby/results/vbear_s0_best/best_model.zip
"""
import argparse
import numpy as np
from stable_baselines3 import PPO
from alien_baby.crawler.train_vision_steer import make_bearing_env, PROP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--occlude-at", type=float, default=2.2, help="occlude once dist < this (m)")
    p.add_argument("--occlude-steps", type=int, default=6)
    p.add_argument("--seed", type=int, default=7000)
    args = p.parse_args()
    m = PPO.load(args.model, device="cpu")
    env = make_bearing_env(seed=args.seed, cone=0.9, reach=1.0)
    rng = np.random.default_rng(99)

    pre, occ, bear = [], [], []      # turn just before occlusion, turns during, true bearing then
    for e in range(args.episodes):
        env.rng = np.random.default_rng(args.seed + e)
        obs, _ = env.reset()
        done, fired, last_turn, n_occ = False, False, None, 0
        while not done:
            occluding = fired and n_occ < args.occlude_steps
            o = obs.copy()
            if occluding:
                o[PROP:] = rng.uniform(0.0, 1.0, o.shape[0] - PROP)
            a, _ = m.predict(o, deterministic=True)
            turn = 0.6 * float(a[1])
            if occluding:
                occ.append((bear[-1], turn)); n_occ += 1
            elif not fired and env._dist() < args.occlude_at:
                fired = True; bear.append(env.gaze_bearing()); pre.append(turn)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
    if not occ:
        print("no episodes reached the occlusion trigger"); return

    b_pre = np.array(bear); t_pre = np.array(pre)
    b_occ = np.array([x for x, _ in occ]); t_occ = np.array([t for _, t in occ])

    def r2(x, y):
        if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-3:
            return float("nan")
        A = np.c_[x, np.ones(len(x))]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(1 - ((y - A @ c) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))

    print(f"=== OCCLUSION PROBE  n={args.episodes} eps, occlude at d<{args.occlude_at} m "
          f"for {args.occlude_steps} steps ===")
    print(f"  policy is FEEDFORWARD (no recurrent state) -- collapse is the architecturally forced outcome")
    print(f"  LAST SIGHTED step : R2(true bearing -> turn) {r2(b_pre, t_pre):+.3f}   turn-std {t_pre.std():.3f}  n={len(t_pre)}")
    print(f"  DURING OCCLUSION  : R2(last-seen bearing -> turn) {r2(b_occ, t_occ):+.3f}   turn-std {t_occ.std():.3f}  n={len(t_occ)}")
    print(f"  --> {'PERSISTS (unexpected -- investigate)' if r2(b_occ, t_occ) > 0.20 else 'COLLAPSES == reactive visual servoing, no held estimate (as the wiring predicts)'}")


if __name__ == "__main__":
    main()
