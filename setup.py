from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="thyroassist",
    version="0.1.0",
    packages=find_packages(),  # 自动发现 thyassist 包
    package_data={
        "thyassist": [
            "demo/*.py",  # 包含 demo 目录下的所有 .py 文件
            "*.yaml", "*.json", "*.txt"
        ]
    },
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "thyassist = launcher:main"  # 直接使用 launcher.py 作为入口
        ],
    },
    py_modules=["launcher"],  # 关键！指定 launcher.py 作为模块
)
