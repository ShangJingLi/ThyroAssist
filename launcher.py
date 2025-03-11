# launcher.py
import argparse
import subprocess
import sys
import os


def get_project_root():
    """获取项目根目录（存放 launcher.py 的目录）"""
    return os.path.dirname(os.path.abspath(__file__))


def get_demo_path(demo_name):
    """ 根据 demo 名称返回对应脚本路径 """
    demo_mapping = {
        "pathology": "resnet_pathological_image_classification.py",
        "ultrasound": "nested_unet_ultrasound_image_infer.py",
        "single_cell": "single_cell_infer.py"
    }
    script_name = demo_mapping.get(demo_name)
    if not script_name:
        raise ValueError(f"未知的 demo 名称: {demo_name}")
    # 直接使用项目根目录的 demo 路径
    return os.path.join(os.path.dirname(__file__), "demo", script_name)


def main():
    parser = argparse.ArgumentParser(description="ThyroAssist 命令行工具")
    parser.add_argument(
        "--pathology", action="store_true", help="运行病理分析 demo"
    )
    parser.add_argument(
        "--ultrasound", action="store_true", help="运行超声分析 demo"
    )
    parser.add_argument(
        "--single_cell", action="store_true", help="运行单细胞分析 demo"
    )
    args = parser.parse_args()

    # 确定要运行的 demo
    if args.pathology:
        demo = "pathology"
    elif args.ultrasound:
        demo = "ultrasound"
    elif args.single_cell:
        demo = "single_cell"
    else:
        print("请指定要运行的 demo (--pathology/--ultrasound/--single_cell)")
        sys.exit(1)

    # 获取脚本路径并执行
    script_path = get_demo_path(demo)
    if not os.path.exists(script_path):
        print(f"错误：未找到 {script_path}")
        sys.exit(1)
    subprocess.run([sys.executable, script_path])

if __name__ == "__main__":
    main()