"""Diagnostic: is the ball-bearing behaviorally load-bearing on +/-22 vs a lateral band?
Fast proprio eval of the frozen teacher, TRUE vs ZERO bearing. If true>>zero on the lateral
band but true~=zero on the full cone, the +/-22 forward-crawl confound is real and the fix is a
lateral-band spawn (where vision's R2=0.84 bearing can finally convert to contacts)."""
import numpy as np
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv, CRAWL_POSES
from alien_baby.crawler.stage_a_distill import load_teacher

XML = "alien_baby/crawler/mimo_crawler_pos_wide.xml"


def run(min_deg, source, n=40, seed=123):
    teacher, norm = load_teacher("cpu")
    env = MimoCrawlerEnv(vision=False, target_obs=True, crawl_pose=CRAWL_POSES["arms_fwd"],
                         action_mode="position_offset", xml_path=XML, spawn_cone_deg=44.0,
                         spawn_cone_min_deg=min_deg, spawn_radius=(0.70, 0.80),
                         random_start_orientation=False, max_steps=1000, step_cost=0.0,
                         approach_reward_scale=10.0, velocity_bonus_scale=0.0,
                         terminate_tilt_deg=50.0, tip_penalty=-5.0)
    env.reset(seed=seed)
    contacts = 0
    for _ in range(n):
        obs, _ = env.reset()
        touched = False
        for _ in range(1000):
            proprio, true_b = obs[:69], obs[69:72]
            b = true_b if source == "true" else np.zeros(3, np.float32)
            act, _ = teacher.predict(norm(np.concatenate([proprio, b])), deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if info.get("touched_ball1"):
                touched = True
            if term or trunc:
                break
        contacts += int(touched)
    env.close()
    return contacts / n


if __name__ == "__main__":
    for label, mn in [("FULL cone +/-22", 0.0), ("LATERAL band +/-15-22", 30.0)]:
        t = run(mn, "true"); z = run(mn, "zero")
        print(f"{label:24s}: true-bearing {t*100:.1f}%  zero-bearing {z*100:.1f}%  "
              f"=> bearing worth {(t-z)*100:+.1f} pts")
