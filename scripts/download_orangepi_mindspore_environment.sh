# 运行此脚本之前，请做好如下准备工作：
# 1.已经手动从user模式进入root用户模式，否则将会因为需要输入密码导致脚本终止
# 2.对应的toolkit包和kernel包已经正确地安装在  /home/HwHiAiUser/Downloads  目录下
# 3.已经配好了swap内存，否则mindspore的run_check()可能会挂掉
sudo bash -c '

    set -e  # 如果任何命令失败，则立即退出脚本



    cd /usr/local/Ascend/ascend-toolkit || {

        echo "Error: Failed to change directory to /usr/local/Ascend/ascend-toolkit. No files will be deleted." >&2

        exit 1

    }

    rm -rf *  # 这将在cd成功后执行，如果cd失败则不会执行到这里,防止文件被误删

    cd /home/HwHiAiUser/Downloads || {

        echo "Error: Failed to change directory to /home/HwHiAiUser/Downloads." >&2
        exit 1

    }
'

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install openi
cd /home/HwHiAiUser/Downloads || exit



export TOOLKIT_NAME=$(python -c "import os;import fnmatch;prefix_toolkit='Ascend-cann-toolkit'; extension='.run';found_toolkit = any(f.startswith(prefix_toolkit) and f.endswith(extension) for f in os.listdir('.')); toolkit_path = next((f for f in os.listdir(os.getcwd()) if fnmatch.fnmatch(f, 'Ascend-cann-toolkit*')), None); print(toolkit_path)")
export KERNELS_NAME=$(python -c "import os;import fnmatch;prefix_kernels='Ascend-cann-kernels';extension='.run';found_kernels = any(f.startswith(prefix_kernels) and f.endswith(extension) for f in os.listdir('.'));kernels_path = next((f for f in os.listdir(os.getcwd()) if fnmatch.fnmatch(f, 'Ascend-cann-kernels*')), None);print( kernels_path);")



sudo -E bash -c "\"chmod +x ./${TOOLKIT_NAME} && chmod +x ./${KERNELS_NAME} && ./${TOOLKIT_NAME} --install -y && ./${KERNELS_NAME} --install -y\"" || exit

# 安装mindspore部分
python -m pip install -U pip

pip install sympy
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl

export MS_VERSION=2.4.0
pip install \
    https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/aarch64/mindspore-${MS_VERSION/-/}-cp39-cp39-linux_aarch64.whl \
    --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install jinja2 absl-py


{
  echo "export GLOG_v=2";
  echo "LOCAL_ASCEND=/usr/local/Ascend";
  echo "source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh" ;
}  >> ~/.bashrc

python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"

