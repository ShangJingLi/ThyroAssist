"""项目启动脚本"""
import argparse
import subprocess
import sys
import os
import pkg_resources


def get_project_root():
    """获取项目根目录（存放 launcher.py 的目录）"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "thyassist")


def get_demo_path(demo_name):
    """ 根据 demo 名称返回对应脚本路径 """
    demo_mapping = {
        "pathology": "resnet_pathological_image_classification.py",
        "ultrasound": "nested_unet_ultrasound_image_infer.py",
        "single_cell": "single_cell_infer.py",
        "download_use_case": "download_use_case.py",
        "clear":"clear_files.py"
    }
    script_name = demo_mapping.get(demo_name)
    if not script_name:
        raise ValueError(f"未知的 demo 名称: {demo_name}")

    # 关键修改：使用 pkg_resources 获取文件路径
    return pkg_resources.resource_filename(__name__, os.path.join("thyassist/demo", script_name))


def main():
    parser = argparse.ArgumentParser(description="ThyroAssist 命令行工具")
    subparsers = parser.add_subparsers(dest="command", required=True, help="子命令")

    # 病理分析子命令
    parser_pathology = subparsers.add_parser("pathology", help="运行病理分析 demo")
    parser_pathology.set_defaults(func=lambda: run_demo("pathology"))

    # 超声分析子命令
    parser_ultrasound = subparsers.add_parser("ultrasound", help="运行超声分析 demo")
    parser_ultrasound.set_defaults(func=lambda: run_demo("ultrasound"))

    # 单细胞分析子命令
    parser_single_cell = subparsers.add_parser("single_cell", help="运行单细胞分析 demo")
    parser_single_cell.set_defaults(func=lambda: run_demo("single_cell"))

    # 用例下载子命令
    parser_download_use_case = subparsers.add_parser("download_use_case", help="下载演示用例")
    parser_download_use_case.set_defaults(func=lambda: run_demo("download_use_case"))

    # 模型文件清理子命令
    parser_download_use_case = subparsers.add_parser("clear", help="模型文件清理")
    parser_download_use_case.set_defaults(func=lambda: run_demo("clear"))

    args = parser.parse_args()
    args.func()


def run_demo(demo_name):
    """ 执行对应的 demo 脚本 """
    script_path = get_demo_path(demo_name)
    if not os.path.exists(script_path):
        print(f"错误：未找到 {script_path}")
        sys.exit(1)
    subprocess.run([sys.executable, script_path])


if __name__ == "__main__":
    main()