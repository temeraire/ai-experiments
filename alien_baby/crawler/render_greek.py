"""render_greek.py — first look at AB inside the Greek room. Panels: overhead | ringside | AB's eye.
Target spawns behind a far column (occlusion). Driven by the depth-capable policy so AB moves."""
import numpy as np, mujoco, imageio.v2 as imageio
from PIL import Image
from stable_baselines3 import PPO
from alien_baby.crawler.mimo_crawler_env import MimoCrawlerEnv
from alien_baby.crawler.train_head_search import CRAWL_POSES

env = MimoCrawlerEnv(
    vision=True, stereo=True, frame_stack=2, frame_stride=6,
    xml_path="alien_baby/crawler/mimo_crawler_greek.xml",
    fixed_ball_positions=[(0.35, 0.90)], spawn_cone_deg=136,
    action_mode="position_offset", crawl_pose=CRAWL_POSES["arms_fwd"],
    terminate_tilt_deg=50.0, max_steps=250)
model = PPO.load("alien_baby/results/parallax_mem_s0_best/best_model.zip", device="cpu")

def cam(name, res):
    r = mujoco.Renderer(env.model, res, res); r.update_scene(env.data, camera=name)
    out = r.render(); r.close(); return out
def up(img, s):
    return np.asarray(Image.fromarray(img).resize((s, s), Image.NEAREST))

writer = imageio.get_writer("scratch_render/greek_room.mp4", fps=30, macro_block_size=None)
for ep in range(3):
    obs, _ = env.reset(seed=10 + ep)
    for t in range(250):
        frame = np.concatenate([cam("overhead", 460), cam("ringside", 460), up(cam("left_eye", 200), 460)], axis=1)
        writer.append_data(frame)
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            for _ in range(10): writer.append_data(frame)
            break
writer.close()
print("wrote scratch_render/greek_room.mp4")
