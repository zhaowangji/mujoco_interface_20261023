import time
import numpy as np
import mujoco
import mujoco.viewer


XML_PATH = "new_interface.xml"

# 力矩大小，单位 N·m
TORQUE_MAG = -5e-3

# 仿真时长，单位 s
SIM_DURATION = 1.5

# 初始旋转轴在世界坐标系中的方向
axis = np.array([
    -0.86602572,
    -0.49999945,
    0.0
], dtype=float)

axis /= np.linalg.norm(axis)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

# 查询刚体 ID
body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "link_paw1"
)

# 查询两个活动 Site 和两个固定 Site
site_a_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "paw1_axis_site_a"
)

site_b_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "paw1_axis_site_b"
)

world_a_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "world_paw1_axis_site_a"
)

world_b_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "world_paw1_axis_site_b"
)

if min(body_id, site_a_id, site_b_id,
       world_a_id, world_b_id) < 0:
    raise RuntimeError("XML 中缺少指定的 body 或 site。")

# 保存初始状态
initial_pos = data.xpos[body_id].copy()
initial_R = data.xmat[body_id].reshape(3, 3).copy()

print("=" * 65)
print("旋转副测试：沿旋转轴施加恒定力矩")
print("=" * 65)

print("旋转轴方向:", axis)
print("施加力矩:", TORQUE_MAG * axis)
print("初始位置:", initial_pos)
print("初始旋转矩阵:\n", initial_R)


def measure():
    """测量约束误差、刚体位移和绕目标轴的转角。"""

    # 当前刚体上两个 Site 的世界坐标
    p_a = data.site_xpos[site_a_id].copy()
    p_b = data.site_xpos[site_b_id].copy()

    # 世界坐标系中两个固定 Site 的坐标
    fixed_a = data.site_xpos[world_a_id].copy()
    fixed_b = data.site_xpos[world_b_id].copy()

    # 约束位置误差
    err_a = np.linalg.norm(p_a - fixed_a)
    err_b = np.linalg.norm(p_b - fixed_b)

    # 刚体位置
    pos = data.xpos[body_id].copy()
    displacement = np.linalg.norm(pos - initial_pos)

    # 当前刚体旋转矩阵
    R = data.xmat[body_id].reshape(3, 3).copy()

    # 相对于初始姿态的旋转矩阵
    R_rel = R @ initial_R.T

    # 从相对旋转矩阵提取绕目标轴的有符号转角
    sin_angle = 0.5 * np.dot(
        axis,
        np.array([
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1]
        ])
    )

    cos_angle = 0.5 * (np.trace(R_rel) - 1.0)

    angle = np.arctan2(sin_angle, cos_angle)

    return err_a, err_b, displacement, angle, pos


paused = True
step_once = False


def key_callback(keycode):
    global paused, step_once

    if keycode == 32:
        paused = not paused
        print("已暂停" if paused else "继续运行")

    elif keycode in (ord("N"), ord("n")):
        step_once = True


print("\n开始仿真。")
print("空格：暂停/继续；N：单步运行。")

last_print_time = -1.0

model.geom_contype[0] = 0
model.geom_conaffinity[0] = 0

model.opt.gravity[:] = [0, 0, 0]

model.geom_rgba[0,3]=0.0

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    while viewer.is_running():

        if paused:
            if step_once:
                # 单步时也施加当前时刻的力矩
                ramp = min(data.time / 0.05, 1.0)
                data.xfrc_applied[body_id, 3:6] = (
                    TORQUE_MAG * ramp * axis
                )

                mujoco.mj_step(model, data)
                step_once = False
            else:
                time.sleep(0.01)

        else:
            # 前 0.05 s 逐渐增加力矩，避免突然加载
            ramp = min(data.time / 0.05, 1.0)

            # xfrc_applied 的后 3 个分量是世界坐标系中的力矩
            data.xfrc_applied[body_id, 3:6] = (
                TORQUE_MAG * ramp * axis
            )

            mujoco.mj_step(model, data)

        viewer.sync()

        if data.time - last_print_time >= 0.1:
            err_a, err_b, displacement, angle, pos = measure()

            print(
                f"t={data.time:6.3f}s | "
                f"angle={np.degrees(angle):9.4f} deg | "
                f"errA={err_a * 1000:9.6f} mm | "
                f"errB={err_b * 1000:9.6f} mm | "
                f"位移={displacement * 1000:9.6f} mm | "
                f"pos={np.round(pos, 6)}"
            )

            last_print_time = data.time

        # if data.time >= SIM_DURATION:
        #     print("\n测试结束。")
        #     break