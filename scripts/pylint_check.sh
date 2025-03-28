#!/bin/bash
# pylint检查脚本，用于进行代码风格检查
# Windows系统下无法直接运行本脚本，请依次输入如下命令进行代码风格检查：
# pylint --rcfile=.pylint.conf thyassist
# pylint --rcfile=.pylint.conf train_and_eval
NUM_CORES=$(nproc)
pylint --jobs=$NUM_CORES --rcfile=.pylint.conf thyassist
pylint --jobs=$NUM_CORES --rcfile=.pylint.conf train_and_eval