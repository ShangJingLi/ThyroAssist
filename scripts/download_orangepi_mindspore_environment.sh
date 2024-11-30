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

# 定义要查找的包模式
kernel_pattern="Ascend-cann-kernels-*.run"
toolkit_pattern="Ascend-cann-toolkit_*.run"

# 当前目录
current_directory=$(pwd)

# 查找当前目录中的kernel包
kernel_files=$(find "$current_directory" -maxdepth 1 -type f -name "$kernel_pattern")

# 查找当前目录中的toolkit包
toolkit_files=$(find "$current_directory" -maxdepth 1 -type f -name "$toolkit_pattern")

# 检查toolkit包是否存在
if [ -z "$toolkit_files" ]; then
    echo "No toolkit package found. Starting download..."
    openi model download enter/nodule_segmentation cann-toolkit --save_path .

    # 检查下载是否成功（假设下载后的文件名与模式匹配）
    downloaded_toolkit_files=$(find "$current_directory" -maxdepth 1 -type f -name "$toolkit_pattern")
    if [ -z "$downloaded_toolkit_files" ]; then
        echo "Failed to download toolkit package."
        exit 1
    else
        echo "Toolkit package downloaded successfully: $downloaded_toolkit_files"
    fi
fi

# 检查kernel包是否存在
if [ -z "$kernel_files" ]; then
    echo "No kernel package found. Starting download..."
    openi model download enter/nodule_segmentation cann-kernels --save_path .

    # 检查下载是否成功（假设下载后的文件名与模式匹配）
    downloaded_kernel_files=$(find "$current_directory" -maxdepth 1 -type f -name "$kernel_pattern")
    if [ -z "$downloaded_kernel_files" ]; then
        echo "Failed to download kernel package."
        exit 1
    else
        echo "Kernel package downloaded successfully: $downloaded_kernel_files"
    fi
fi
pip uninstall openi -y

sudo bash -c '

    set -e  # 如果任何命令失败，则立即退出脚本

    CANN_DIR="."

    # 查找toolkit和kernels的安装包
    TOOLKIT_FILE=$(ls "$CANN_DIR"/Ascend-cann-toolkit_*.run 2>/dev/null | sort -V | tail -n 1)
    KERNELS_FILE=$(ls "$CANN_DIR"/Ascend-cann-kernels_*.run 2>/dev/null | sort -V | tail -n 1)

    # 检查是否找到了安装包
    if [ -z "$TOOLKIT_FILE" ]; then
      echo "Error: No CANN toolkit package found in $CANN_DIR"
      exit 1
    fi

    if [ -z "$KERNELS_FILE" ]; then
      echo "Error: No CANN kernels package found in $CANN_DIR"
      exit 1
    fi

    # 赋予安装包执行权限
    chmod +x "$TOOLKIT_FILE"
    chmod +x "$KERNELS_FILE"

    # 安装toolkit
    echo "Installing CANN toolkit..."
    sudo "$TOOLKIT_FILE" --install -y
    if [ $? -ne 0 ]; then
      echo "Error: Failed to install CANN toolkit"
      exit 1
    fi

    # 安装kernels
    echo "Installing CANN kernels..."
    sudo "$KERNELS_FILE" --install -y
    if [ $? -ne 0 ]; then
      echo "Error: Failed to install CANN kernels"
      exit 1
    fi

    echo "CANN toolkit and kernels have been installed successfully."
'

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

