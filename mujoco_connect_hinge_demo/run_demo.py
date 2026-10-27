import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "connect_hinge_demo.xml"

# Torque around the fixed hinge axis (+X), in N*m.
TORQUE_X = 0.02

paused = True
single_step = False


def key_callback(keycode):
    global paused, single_step
    if keycode == 32:  # Space
        paused = not paused
        print("Running" if not paused else "Paused")
    elif keycode in (ord("N"), ord("n")):
        single_step = True


model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_link")
if body_id < 0:
    raise RuntimeError("Could not find body 'test_link'")

print("Demo: two connect constraints between a free body and the world.")
print("Expected motion: rotation about world X; the orange marker should swing.")
print("Space = run/pause, N = one simulation step.")

last_print = 0.0
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        t0 = time.time()

        if not paused or single_step:
            single_step = False

            # xfrc_applied layout: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
            data.xfrc_applied[body_id, :3] = 0.0
            data.xfrc_applied[body_id, 3:6] = [TORQUE_X, 0.0, 0.0]

            mujoco.mj_step(model, data)

            now = time.time()
            if now - last_print > 0.25:
                vel6 = np.zeros(6)
                mujoco.mj_objectVelocity(
                    model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel6, 0
                )
                # mj_objectVelocity returns rotational velocity first, then linear velocity.
                omega_world = vel6[:3]
                print(
                    f"t={data.time:7.3f} s | "
                    f"omega_world={omega_world} rad/s | "
                    f"omega_x={omega_world[0]: .6f} rad/s"
                )
                last_print = now

        else:
            # Clear the force while paused so no residual force is left pending.
            data.xfrc_applied[body_id, :] = 0.0

        viewer.sync()
        if not paused:
            remaining = model.opt.timestep - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)
