# setup.py
from setuptools import setup

def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="thyroassist",
    version="0.1.0",
    py_modules=["launcher"],  # 直接包含 launcher.py 作为模块
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "thyassist = launcher:main"  # 关键：注册全局命令
        ],
    },
)