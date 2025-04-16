"""数据清理模块"""
import shutil
import os
from launcher import get_project_root


project_dir = get_project_root()
targets = [
    "medical_resnet_checkpoints(pad)",
    "nested_unet_checkpoints",
    "best_mlp_checkpoints",
    "medical_resnet_checkpoints(crop)",
    "nested_unet.om",
    "nested_unet.onnx",
    "medical_resnet.om",
    "medical_resnet.onnx",
    "trt_cache"
]

flag = False  # 记录是否删除了任何内容

for target in targets:
    path = os.path.join(project_dir, target)

    # 对于目录采用递归删除，对于文件采用直接删除
    if os.path.exists(path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"✅ 目录 {path} 及其内容已成功删除。", flush=True)
            else:
                os.remove(path)
                print(f"✅ 文件 {path} 已成功删除。", flush=True)
            flag = True  # 只要有一个删除成功就标记为 True
        except Exception as e:
            print(f"❌ 删除 {path} 时发生错误: {str(e)}", flush=True)

if not flag:
    print("⚠️ 未删除任何内容", flush=True)
