from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
import time
import mujoco
import mujoco.viewer

def key_callback(keycode):
    global step_trigger, is_paused
    if keycode == 32:
        is_paused = not is_paused
        print("运行中..." if not is_paused else "已暂停")
    elif keycode in (ord('N'), ord('n')):
        step_trigger = True

def print_init(model,data):
    print("=" * 60)
    print("模型关节初始状态")
    print("=" * 60)

    print(f"关节总数：{model.njnt}")
    print(f"广义位置变量总数：{model.nq}")

    for i in range(model.njnt):
        # 读取关节名称
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, i
        )
        if name is None:
            name = f"joint_{i}"

        # 读取关节类型
        joint_type = model.jnt_type[i]
        
        # 读取该关节在 qpos 中的起始索引
        qpos_adr = model.jnt_qposadr[i]

        # 根据关节类型读取位置
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            # 转动副：弧度
            value = data.qpos[qpos_adr]
            print(f"{i},{name}: 转动副，初始角度 = {value:.6f} rad "
                  f"= {value * 180 / 3.141592653589793:.3f}°")

        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            # 移动副：米
            value = data.qpos[qpos_adr]
            print(f"{i},{name}: 移动副，初始位移 = {value:.6f} m")

        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            # 球关节：四元数
            quat = data.qpos[qpos_adr:qpos_adr + 4]
            print(f"{i},{name}: 球关节，初始四元数 = {quat}")

        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            # 自由关节：3个位置坐标 + 4个四元数分量
            qpos = data.qpos[qpos_adr:qpos_adr + 7]
            print(f"{i},{name}: 自由关节，初始位置和姿态 = {qpos}")
    print("=" * 60)
    for i in range(model.ngeom):
            body_id = model.geom_bodyid[i]
    
            body_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            print(i, body_name)
            # if body_name != "link_gan3":
            #     model.geom_rgba[i, 3] = 0.0
            #     print(i,body_name)
            # else:
            #     print(i,body_name)

    print("=" * 60)

def init_model(model, data):


    # 实验1：关闭所有几何体之间的接触碰撞
    # model.geom_contype[:] = 0
    # model.geom_conaffinity[:] = 0

    # 实验2：关闭重力
    model.opt.gravity[:] = [0, 0, 0]

    # 实验3：减小时间步长
    model.opt.timestep = 0.001

    return model, data

def main():

    global step_trigger, is_paused
    step_trigger = False
    is_paused = True

    model_path = Path(__file__).resolve().parent / "interface_ke_20261025.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    model, data = init_model(model, data)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print_init(model,data)

    print("\n[控制说明]：点击视窗后，按 [空格] 切换运行/暂停，按 [N] 单步运行。\n")

    # model.geom_rgba[0, 3] = 0.0
    # data.ctrl[0]=-0.013;    
    ctrl_speed = -0.1

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if not is_paused or step_trigger:
                step_trigger = False

                data.ctrl[0:3] += ctrl_speed * model.opt.timestep
                mujoco.mj_step(model, data)
                viewer.sync()

            else:
                # 暂停时仍然刷新视窗，保证界面正常响应
                viewer.sync()
                time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
