import re
from pathlib import Path

def fix_class(file_path):
    p = Path(file_path)
    text = p.read_text(encoding="utf-8")
    new_text = re.sub(r'class="collision"', 'class="my_collision"', text)
    if new_text != text:
        p.write_text(new_text, encoding="utf-8")
        print(f"✅ 已修改: {p}")
    else:
        print(f"ℹ️  无变化: {p}")

# 单个文件
fix_class("nx_keti.txt")

# 或者批量修改当前目录下所有 xml
# for f in Path(".").glob("*.xml"):
#     fix_class(f)