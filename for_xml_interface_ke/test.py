import numpy as np
import mujoco

# =========================================================
# 1. 加载模型
# =========================================================
model = mujoco.MjModel.from_xml_path("interface_ke.xml")
data = mujoco.MjData(model)

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "link_ke1"
)

if body_id < 0:
    raise RuntimeError("找不到 link_ke1")

# 显式计算初始运动学状态
mujoco.mj_forward(model, data)

# =========================================================
# 2. 定义两个锚点的世界坐标
# =========================================================
P1 = np.array([-0.0654496, 0.1434300, -0.032])
P2 = np.array([-0.0914304, 0.1284300, -0.032])

# =========================================================
# 3. 计算世界坐标 -> link_ke1 局部坐标
# =========================================================
R0 = data.xmat[body_id].reshape(3, 3).copy()
p0 = data.xpos[body_id].copy()

local_P1 = R0.T @ (P1 - p0)
local_P2 = R0.T @ (P2 - p0)

print("=" * 60)
print("初始状态")
print("=" * 60)
print("刚体世界位置:", p0)
print("刚体世界旋转矩阵:\n", R0)
print("局部锚点 P1:", local_P1)
print("局部锚点 P2:", local_P2)
print("局部锚点间距:", np.linalg.norm(local_P1 - local_P2))

# 锚点相对原点应约为 15 mm
if np.linalg.norm(local_P1) < 1e-5:
    raise RuntimeError(
        "P1 局部坐标接近原点，请检查模型初始位姿和锚点坐标。"
    )

if np.linalg.norm(local_P2) < 1e-5:
    raise RuntimeError(
        "P2 局部坐标接近原点，请检查模型初始位姿和锚点坐标。"
    )

# =========================================================
# 4. 旋转轴与施加力矩
# =========================================================
axis = P2 - P1
axis /= np.linalg.norm(axis)

TORQUE_MAG = 0.01
torque = TORQUE_MAG * axis

print("旋转轴方向:", axis)
print("施加力矩:", torque)

# =========================================================
# 5. 仿真并监测锚点误差
# =========================================================
for i in range(5000):

    # 每步施加力矩
    data.xfrc_applied[body_id, :3] = 0.0
    data.xfrc_applied[body_id, 3:6] = torque

    mujoco.mj_step(model, data)

    if i % 100 == 0:

        # 当前刚体位姿
        R = data.xmat[body_id].reshape(3, 3)
        p = data.xpos[body_id]

        # 当前两个锚点的世界坐标
        current_P1 = p + R @ local_P1
        current_P2 = p + R @ local_P2

        # 与固定目标点的距离
        err1 = np.linalg.norm(current_P1 - P1)
        err2 = np.linalg.norm(current_P2 - P2)

        print(
            f"t={data.time:.3f}s | "
            f"err1={err1 * 1000:.6f} mm | "
            f"err2={err2 * 1000:.6f} mm | "
            f"pos={p}"
        )