"""render_quad_preflight.py — camera-visibility preflight for the eyes-on-quadruped walker.
Stands the Ant at its init pose and checks whether a floor ball is in the eye-view across a
range of distances (no gait needed). Panels rows = ball at 0.6/0.9/1.3/1.7 m; cols = side | eye."""
import numpy as np, mujoco
from PIL import Image
m = mujoco.MjModel.from_xml_path("alien_baby/crawler/quad_walker.xml")
d = mujoco.MjData(m)
init = [0,0,0.55, 1,0,0,0, 0,1.0,0,-1.0,0,-1.0,0,1.0]  # ant standing pose
def render(camera, res):
    r = mujoco.Renderer(m, res, res); r.update_scene(d, camera=camera); out = r.render(); r.close(); return out
def up(img, s): return np.asarray(Image.fromarray(img).resize((s, s), Image.NEAREST))
rows = []
for bx in (0.6, 0.9, 1.3, 1.7):
    d.qpos[:15] = init; d.qpos[15:22] = [bx, 0.0, 0.10, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    eye = render("left_eye", 32)
    R, G, B = eye[..., 0].astype(int), eye[..., 1].astype(int), eye[..., 2].astype(int)
    print(f"ball {bx:.1f} m ahead: red px in 32x32 eye = {int(np.sum((R>150)&(G<90)&(B<90)))}")
    rows.append(np.concatenate([render("side", 300), up(render("left_eye", 120), 300)], axis=1))
Image.fromarray(np.concatenate(rows, axis=0)).save("scratch_render/quad_preflight.png")
print("wrote scratch_render/quad_preflight.png")
