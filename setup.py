# setup.py
from setuptools import setup, find_packages
import os

def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="thyroassist",
    version="0.1.0",
    packages=find_packages(),  # 自动查找所有包
    package_data={
        "thyassist": ["*.yaml", "*.json", "*.txt"],  # 确保 yaml/json/txt 等文件被打包
    },
    include_package_data=True,  # 关键：启用包含数据文件
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "thyassist = launcher:main"
        ],
    },
)