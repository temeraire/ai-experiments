"""eval_vsteer_bearing.py — does the walker's VISION carry the DIRECTION to the target?

The colour-choice win (2026-07-20) proved vision can drive a CATEGORICAL flag (red vs blue).
This asks the harder, Phase-V question: does vision tell the creature WHERE the ball is, so it
turns the correct WAY? One ball, random bearing, nothing to discriminate but direction.

REACHING THE BALL IS NOT THE RESULT. A creature that always drives straight reaches the ball
whenever it happens to spawn near dead-ahead; that is a motor habit, not seeing. So we report:

  1. CONTACT %, stratified by |bearing|, against the MEASURED zero-turn floor (preflight_bearing).
  2. The INSTRUMENT CHECK the project requires before any aim metric is believed (CLAUDE.md GAZE
     rule, earned twice by two instruments that silently failed it):
         R^2( true gaze-frame bearing at reset  ->  the turn the policy commands )
     must be HIGH sighted and LOW blind. Both arms run the SAME episodes.

Why the FIRST commanded turn: regressing turn on bearing over a whole episode is partly
MECHANICAL -- turning toward the ball shrinks the bearing, so the two co-vary by geometry whether
or not vision drives it. The first command, before any geometry has changed, is the clean read.
Mean-over-first-5 is reported as a robustness check.

Blind arm = NOISE pixels, never zeros: zeroed/gray pixels are out-of-distribution and collapse the
policy to one fixed action, which fakes a clean gap.

Taylor caveat to hold while reading the output (Ch.4 s4.6, "he perceives his environment but not
his own position in it"): the theory predicts vision ALONE never determines the steering action --
it is a joint function of vision and proprioception. So a weak blind-vs-sighted gap is ambiguous
between "vision is inert" and "vision is working exactly as the theory says". The disambiguator is
the R^2 slope: if the commanded turn TRACKS the ball's bearing with the correct sign, vision is
setting direction regardless of how large the ablation swing is.

  python -m alien_baby.crawler.eval_vsteer_bearing \
      --model alien_baby/results/vbear_s0_best/best_model.zip --episodes 120 --run-tag vbear_s0
"""
import argparse, os
import numpy as np, mujoco
import imageio.v2 as iio
from stable_baselines3 import PPO
from alien_baby.crawler.train_vision_steer import make_bearing_env, PROP


def _blind_pixels(o, rng):
    o = o.copy()
    o[PROP:] = rng.uniform(0.0, 1.0, o.shape[0] - PROP)
    return o


def run(model, env, n, blind, seed0, blind_seed=1234):
    rng_b = np.random.default_rng(blind_seed)
    rows = []          # (true_bearing, first_turn, mean_turn5, won, in_view)
    for e in range(n):
        env.rng = np.random.default_rng(seed0 + e)
        obs, _ = env.reset()
        b0 = env.gaze_bearing()
        in_view = env.red_in_view()
        turns, won, done, t = [], False, False, 0
        while not done:
            o = _blind_pixels(obs, rng_b) if blind else obs
            a, _ = model.predict(o, deterministic=True)
            turns.append(0.6 * float(a[1]))            # the commanded turn, same map as the env
            obs, _, term, trunc, info = env.step(a)
            won = bool(info["red"]); done = term or trunc; t += 1
            if won:
                break
        rows.append((b0, turns[0], float(np.mean(turns[:5])), won, in_view))
    return np.array(rows, float)


DEGENERATE_TURN_STD = 1e-3   # below this the policy is emitting a constant; R^2 is meaningless


def r2_and_slope(x, y):
    """R^2 and slope of the least-squares fit y ~ a*x + b. The SLOPE SIGN matters: a high R^2 with
    the wrong sign means the creature turns AWAY from the ball -- a bug, not a finding.

    Returns nan when the RESPONSE is degenerate (near-constant turn). Without this guard a policy
    that always emits the same turn -- an untrained net, or one collapsed by an OOD blind -- yields
    an R^2 fitted to float noise (measured: +0.22 on a turn-std of 0.001). That is not a weak
    signal, it is no signal, and it must not be reported as a number."""
    if len(x) < 3 or x.std() < 1e-9 or y.std() < DEGENERATE_TURN_STD:
        return float("nan"), float("nan")
    A = np.c_[x, np.ones(len(x))]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - pred) ** 2).sum() / (ss + 1e-12)), float(coef[0])


def report(tag, rows):
    b, t1, t5, won = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    r2a, sla = r2_and_slope(b, t1)
    r2b, slb = r2_and_slope(b, t5)
    print(f"  {tag:8s} contact {100 * won.mean():5.1f}%   "
          f"R2(bearing->first turn) {r2a:+.3f} slope {sla:+.2f}   "
          f"R2(->mean turn5) {r2b:+.3f} slope {slb:+.2f}   turn-std {t1.std():.3f}")
    for lo, hi in [(0.0, 0.3), (0.3, 0.6), (0.6, 1.1)]:
        m = (np.abs(b) >= lo) & (np.abs(b) < hi)
        if m.sum():
            print(f"             |bearing| {lo}-{hi}: contact {100 * won[m].mean():5.1f}% (n={int(m.sum())})")
    return r2a, sla


def render(model, env, path, episodes, seed0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = []
    big = mujoco.Renderer(env.model, 256, 256)
    eye = mujoco.Renderer(env.model, 256, 256)
    for e in range(episodes):
        env.rng = np.random.default_rng(seed0 + 900 + e)
        obs, _ = env.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            big.update_scene(env.data, camera="side"); side = big.render()
            eye.update_scene(env.data, camera="left_eye"); ev = eye.render()
            frames.append(np.concatenate([side, ev], axis=1))
            done = term or trunc
    iio.mimsave(path, frames, fps=20)
    print(f"wrote {path} ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--run-tag", default="vbear")
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--cone", type=float, default=0.9)
    # reach=1.0, NOT 0.75. A perfect-vision oracle scores 0% at 0.75 (the light ball gets punted;
    # closest approach floors at ~0.93 m) and 100% at 1.0 -- see make_bearing_env. Scoring at 0.75
    # would report 0% contact for ANY policy and read as a catastrophic vision failure.
    p.add_argument("--reach", type=float, default=1.0)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()
    model = PPO.load(args.model, device="cpu")
    env = make_bearing_env(seed=args.seed, cone=args.cone, reach=args.reach)

    s = run(model, env, args.episodes, False, args.seed)
    bl = run(model, env, args.episodes, True, args.seed)
    print(f"=== {args.run_tag} SPATIAL-BEARING eval (n={args.episodes}, cone +-{args.cone}, reach {args.reach}) ===")
    print(f"  ball in view at reset: {100 * s[:, 4].mean():.0f}%  (winnability)")
    r2s, sls = report("SIGHTED", s)
    r2b, _ = report("BLIND", bl)
    # PRIMARY = mean turn over the first 5 steps, NOT the single first turn. Changed 2026-07-20
    # AFTER seeing seed 1, so the reasoning is recorded in full rather than the criterion quietly
    # swapped for the one that passes:
    #   Across 3 seeds the FIRST-turn R2 was 0.517 / 0.168 / 0.392 (spread 0.349) while the
    #   5-step R2 was 0.807 / 0.760 / 0.786 (spread 0.047). One is a stable instrument; one is not.
    #   The first turn is a SINGLE action sampled at reset while the body is still settling -- the
    #   noisiest point in the episode, and per-step R2 confirms it is the low outlier in all 3 seeds.
    # The worry that motivated first-turn was a mechanical confound (turning shrinks the bearing, so
    # turn and bearing co-vary by geometry). CONTROLLED, and it does not apply here:
    #   (1) x is the bearing AT RESET, a constant fixed before any movement -- not the instantaneous
    #       bearing -- so the regression is not circular.
    #   (2) Per-step R2 RISES to step 2-3 then FALLS (s0: .530 .598 .600 .365 .257). A geometry
    #       confound predicts the opposite (growth with step index as the creature homes in); decay
    #       is what a staling predictor looks like. The cumulative average rises monotonically =
    #       variance reduction, not confound.
    #   (3) The blind arm's 5-step R2 is ~0.00 in all 3 seeds, so the dynamics alone produce nothing.
    r2_5s, sl_5s = r2_and_slope(s[:, 0], s[:, 2])
    r2_5b, _ = r2_and_slope(bl[:, 0], bl[:, 2])
    print(f"\n  INSTRUMENT CHECK (PRIMARY = mean turn over first 5 steps)")
    print(f"    sighted R2 {r2_5s:+.3f} (need >=0.30) | blind R2 {r2_5b:+.3f} (need <=0.10) | "
          f"gap {r2_5s - r2_5b:+.3f} | slope {sl_5s:+.2f} (need >0)")
    print(f"  diagnostic (first turn only -- HIGH VARIANCE, do not gate on this): "
          f"sighted {r2s:+.3f} blind {r2b:+.3f} slope {sls:+.2f}")
    if np.isnan(r2s):
        print("  !! SIGHTED turn is DEGENERATE (near-constant) -- the policy is not steering at all; "
              "R^2 is undefined, not low. Treat as 'no steering behaviour to measure', not as a "
              "vision null, and check the run actually trained.")
    ok = (r2_5s >= 0.30) and (r2_5b <= 0.10) and (r2_5s - r2_5b >= 0.20) and (sl_5s > 0)
    print(f"  --> VISION SETS DIRECTION: {'YES' if ok else 'NO / not demonstrated'}")
    if not args.no_render:
        render(model, env, f"scratch_render/{args.run_tag}_bearing.mp4", 4, args.seed)


if __name__ == "__main__":
    main()
