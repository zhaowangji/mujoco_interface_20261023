from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import mujoco
import mujoco.viewer


root = Path(__file__).resolve().parent
# urdf_path = root / "interface_20261023/interface/urdf/interface.urdf"
# mesh_dir = root / "interface_20261023/interface/meshes"

urdf_path = root / "interface_ke_20261025/urdf/interface_ke.urdf"
mesh_dir = root / "interface_ke_20261025/meshes"

# urdf_path = root / "interface_20261023/interface/urdf/interface.urdf"
# mesh_dir = root / "interface_20261023/interface/meshes"

with TemporaryDirectory() as temp_dir:
    model_dir = Path(temp_dir)

    urdf_text = urdf_path.read_text(encoding="utf-8")
    urdf_text = urdf_text.replace(
        # "package://interface_20261023/interface/meshes/", ""
        "package://interface_ke_20261025/meshes/", ""
        # "package://interface_20261023/interface/meshes/", ""
    )

    temp_urdf = model_dir / urdf_path.name
    temp_urdf.write_text(urdf_text, encoding="utf-8")

    for mesh_path in mesh_dir.glob("*.STL"):
        copy2(mesh_path, model_dir / mesh_path.name)
    

    model = mujoco.MjModel.from_xml_path(str(temp_urdf))
    data = mujoco.MjData(model)

    # 实验1：关闭所有几何体之间的接触碰撞
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0

    # 实验2：关闭重力
    model.opt.gravity[:] = [0, 0, 0]

    # 实验3：减小时间步长
    model.opt.timestep = 0.001

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    

    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     while viewer.is_running():
    #         mujoco.mj_step(model, data)
    #         viewer.sync()


    mujoco.mj_saveLastXML("interface_ke_20261025.xml", model)



    # hide and show the component

    # for i in range(model.ngeom):
        #     body_id = model.geom_bodyid[i]
    
        #     body_name = mujoco.mj_id2name(
        #         model,
        #         mujoco.mjtObj.mjOBJ_BODY,
        #         body_id,
        #     )
    
        #     if body_name != "link_gan3":
        #         model.geom_rgba[i, 3] = 0.0
        #         print(i,body_name)
        #     else:
        #         print(i,body_name)
    
        # model.geom_rgba[0, 3] = 0.0
        # model.geom_rgba[1, 3] = 0.0
        # model.geom_rgba[2, 3] = 0.0
        # model.geom_rgba[3, 3] = 0.0
        # model.geom_rgba[4, 3] = 0.0
        # model.geom_rgba[5, 3] = 0.0
        # model.geom_rgba[6, 3] = 0.0
        # model.geom_rgba[7, 3] = 0.0