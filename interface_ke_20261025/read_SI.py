import numpy as np
from pathlib import Path

def get_obj_size(filename):
    vertices = []

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                values = line.split()
                vertices.append([
                    float(values[1]),
                    float(values[2]),
                    float(values[3])
                ])

    vertices = np.asarray(vertices)

    if len(vertices) == 0:
        print(f"{filename}: 没有读取到顶点")
        return

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    size = maximum - minimum

    print(f"文件：{filename}")
    print(f"最小坐标：{minimum}")
    print(f"最大坐标：{maximum}")
    print(f"包围盒尺寸：{size}")
    print(f"最大边长：{size.max():.6f}")
    print("-" * 50)

model_path = Path(__file__).resolve().parent / "meshes/paw1/paw1.obj"
get_obj_size(model_path)