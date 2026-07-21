"""probe_proprio_ablation.py — the MISSING ARM of the spatial-bearing result.

We ablated VISION (noise pixels) and steering collapsed. We never ablated PROPRIOCEPTION.
Taylor (Ch.4 s4.6, "he perceives his environment but not his own position in it") predicts vision
ALONE can never determine a steering action -- the action is a JOINT function of vision and
proprioception. That claim was recorded as "reconciled but untested". This is the test.

FOUR ARMS on MATCHED episode seeds, same policy, same episodes:
  1. INTACT           obs untouched
  2. VISION-ABLATED   obs[PROP:] <- uniform(0,1) noise (the fair blind used in eval_vsteer_bearing)
  3. PROPRIO-ABLATED  obs[:PROP] <- IN-DISTRIBUTION Gaussian noise, pixels intact
  4. BOTH             both of the above

WHY THE PROPRIO NOISE IS NOT UNIFORM(0,1). The 9 proprio numbers are torso up-vector(3) +
linear velocity in the body frame(3) + angular velocity(3) -- they live on wildly different
scales (up-vector ~[0,0,1]; velocities small and signed). Uniform(0,1) would be far outside the
range the net has ever seen, so a collapse would test OUT-OF-DISTRIBUTION SHOCK, not information
removal -- we would learn nothing about whether the DIRECTION signal needs proprio. So we first
MEASURE the per-column mean/std over intact rollouts and sample N(mean_i, std_i) per column. That
destroys the moment-to-moment information (the values no longer track the body) while keeping the
marginal distribution the net expects.

Reported per arm: contact %, R2(true gaze bearing -> first commanded turn), slope, turn-std.
r2_and_slope and its DEGENERATE_TURN_STD guard are IMPORTED from eval_vsteer_bearing -- when the
turn is near-constant the guard returns nan and nan is what gets printed. A nan is not a low R2;
it means there is no steering behaviour to measure.

  python -m alien_baby.crawler.probe_proprio_ablation \
      --model alien_baby/results/vbear_s0_best/best_model.zip --episodes 60
"""
import argparse
import numpy as np
from stable_baselines3 import PPO
from alien_baby.crawler.train_vision_steer import make_bearing_env, PROP
from alien_baby.crawler.eval_vsteer_bearing import r2_and_slope, DEGENERATE_TURN_STD


def measure_proprio_stats(model, env, n, seed0):
    """Per-column mean/std of the 9 proprio numbers over INTACT rollouts of this same policy."""
    buf = []
    for e in range(n):
        env.rng = np.random.default_rng(seed0 + e)
        obs, _ = env.reset()
        done = False
        while not done:
            buf.append(obs[:PROP].copy())
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
    A = np.array(buf, float)
    return A.mean(0), A.std(0), len(A)


def ablate(o, rng, kill_vision, kill_proprio, mu, sd):
    o = o.copy()
    if kill_vision:
        o[PROP:] = rng.uniform(0.0, 1.0, o.shape[0] - PROP)
    if kill_proprio:
        o[:PROP] = rng.normal(mu, np.maximum(sd, 1e-6)).astype(o.dtype)
    return o


def run(model, env, n, seed0, kill_vision, kill_proprio, mu, sd, noise_seed=1234):
    rng = np.random.default_rng(noise_seed)
    rows = []
    for e in range(n):
        env.rng = np.random.default_rng(seed0 + e)
        obs, _ = env.reset()
        b0 = env.gaze_bearing()
        in_view = env.red_in_view()
        turns, won, done = [], False, False
        while not done:
            o = ablate(obs, rng, kill_vision, kill_proprio, mu, sd)
            a, _ = model.predict(o, deterministic=True)
            turns.append(0.6 * float(a[1]))
            obs, _, term, trunc, info = env.step(a)
            won = bool(info["red"]); done = term or trunc
            if won:
                break
        rows.append((b0, turns[0], float(np.mean(turns[:5])), won, in_view))
    return np.array(rows, float)


def report(tag, rows):
    b, t1, t5, won = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    r2a, sla = r2_and_slope(b, t1)
    r2b, slb = r2_and_slope(b, t5)
    print(f"  {tag:16s} contact {100 * won.mean():5.1f}%   "
          f"R2(bearing->first turn) {r2a:+.3f} slope {sla:+.2f}   "
          f"R2(->mean turn5) {r2b:+.3f} slope {slb:+.2f}   turn-std {t1.std():.4f}")
    if np.isnan(r2a):
        print(f"                   !! turn is DEGENERATE (std {t1.std():.5f} < {DEGENERATE_TURN_STD}) "
              f"-- R2 UNDEFINED, not low. No steering behaviour to measure.")
    return r2a, sla, t1.std(), won.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--stats-episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--cone", type=float, default=0.9)
    p.add_argument("--reach", type=float, default=1.0)
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = make_bearing_env(seed=args.seed, cone=args.cone, reach=args.reach)

    mu, sd, nsteps = measure_proprio_stats(model, env, args.stats_episodes, args.seed)
    names = ["up_x", "up_y", "up_z", "vlin_x", "vlin_y", "vlin_z", "wang_x", "wang_y", "wang_z"]
    print(f"=== PROPRIO NOISE CONSTRUCTION (measured over {args.stats_episodes} intact episodes, "
          f"{nsteps} steps) ===")
    for i, nm in enumerate(names):
        print(f"  {nm:7s} mean {mu[i]:+8.4f}   std {sd[i]:8.4f}")
    print("  proprio ablation = per-column N(mean, std) resample -> marginal distribution preserved,")
    print("  moment-to-moment body information destroyed.\n")

    print(f"=== FOUR-ARM ABLATION  n={args.episodes} matched seeds ({args.seed}..{args.seed + args.episodes - 1}), "
          f"cone +-{args.cone}, reach {args.reach} ===")
    arms = [("INTACT", False, False), ("VISION-ABLATED", True, False),
            ("PROPRIO-ABLATED", False, True), ("BOTH-ABLATED", True, True)]
    out = {}
    first = None
    for tag, kv, kp in arms:
        rows = run(model, env, args.episodes, args.seed, kv, kp, mu, sd)
        if first is None:
            first = rows
            print(f"  ball in view at reset: {100 * rows[:, 4].mean():.0f}%  (winnability)")
        out[tag] = report(tag, rows)

    r2_i, r2_p = out["INTACT"][0], out["PROPRIO-ABLATED"][0]
    print("\n  VERDICT INPUTS: "
          f"intact R2 {r2_i:+.3f} | vision-ablated R2 {out['VISION-ABLATED'][0]:+.3f} | "
          f"proprio-ablated R2 {r2_p:+.3f} | both {out['BOTH-ABLATED'][0]:+.3f}")
    if np.isnan(r2_p):
        print("  --> PROPRIO ablation left a DEGENERATE turn: R2 undefined. Report as collapse of "
              "steering BEHAVIOUR, not as a low correlation.")
    elif r2_p >= 0.30 and r2_p >= 0.6 * r2_i:
        print("  --> PROPRIO IS NOT NEEDED FOR DIRECTION: bearing still drives the turn without it.")
    else:
        print("  --> PROPRIO IS LOAD-BEARING for the steering action (joint determination).")


if __name__ == "__main__":
    main()
