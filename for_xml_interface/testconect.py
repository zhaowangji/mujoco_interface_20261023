import time
import numpy as np
import mujoco
import mujoco.viewer

# =========================
# 参数设置
# =========================
XML_PATH = "connect_test.xml"

# 两个锚点的世界坐标
P1 = np.array([-0.065449, 0.14343, -0.032])
P2 = np.array([-0.09143, 0.12843, -0.032])

# 力矩大小，单位 N·m
TORQUE_MAG = 0.01

# 初始暂停
is_paused = True
step_trigger = False

# =========================
# 加载模型
# =========================
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "link_paw1"
)

if body_id < 0:
    raise ValueError("模型中找不到 link_paw1")

# 沿两锚点连线施加力矩
axis_world = P2 - P1
axis_world /= np.linalg.norm(axis_world)

print("旋转轴方向（P1 -> P2）：", axis_world)
print("施加力矩（世界坐标系）：", TORQUE_MAG * axis_world)

# 力矩沿世界坐标系表示，施加在刚体质心处
def apply_torque():
    data.xfrc_applied[body_id, :3] = 0.0
    data.xfrc_applied[body_id, 3:6] = TORQUE_MAG * axis_world


def key_callback(keycode):
    global is_paused, step_trigger

    if keycode == 32:  # Space
        is_paused = not is_paused
        print("运行中..." if not is_paused else "已暂停")

    elif keycode in (ord("N"), ord("n")):
        step_trigger = True


# =========================
# 仿真
# =========================
with mujoco.viewer.launch_passive(
    model, data, key_callback=key_callback
) as viewer:

    last_print_time = 0.0

    while viewer.is_running():
        start = time.time()

        if not is_paused or step_trigger:
            step_trigger = False

            # 每步都施加力矩
            apply_torque()
            mujoco.mj_step(model, data)

            # 每 0.2 秒输出一次状态
            now = time.time()
            if now - last_print_time > 0.2:
                # cvel 的前 3 项为世界坐标系下的角速度
                mujoco.mj_objectVelocity(
                    model, data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                    np.zeros(6),
                    0
                )

                vel = np.zeros(6)
                mujoco.mj_objectVelocity(
                    model, data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                    vel,
                    0
                )

                omega = vel[:3]
                omega_axis = np.dot(omega, axis_world)

                print(
                    f"t={data.time:.3f}s, "
                    f"角速度={omega}, "
                    f"沿轴角速度={omega_axis:.6f} rad/s"
                )

                last_print_time = now

            viewer.sync()

        else:
            viewer.sync()
            time.sleep(0.01)

        if not is_paused:
            elapsed = time.time() - start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)