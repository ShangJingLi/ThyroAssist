# 运行此脚本之前，请做好如下准备工作：
# 1.已经手动从user模式进入root用户模式，否则将会因为需要输入密码导致脚本终止
# 2.对应的toolkit包和kernel包已经正确地安装在  /home/HwHiAiUser/Downloads  目录下
# 3.已经配好了swap内存，否则mindspore的run_check()可能会挂掉
cd /usr/local/Ascend/ascend-toolkit
rm -rf *

cd /home/HwHiAiUser/Downloads

chmod +x ./Ascend-cann-toolkit_8.0.RC3.alpha003_linux-aarch64.run
chmod +x ./Ascend-cann-kernels-310b_8.0.RC3.alpha003_linux-aarch64.run
./Ascend-cann-toolkit_8.0.RC3.alpha003_linux-aarch64.run --install
./Ascend-cann-kernels-310b_8.0.RC3.alpha003_linux-aarch64.run --install

# 退出root模型，在user模式下
exit
conda create --name mindspore python=3.9 -y
conda activate mindspore

# 使用 vim 打开 .bashrc 并添加自启动mindspore环境命令
content_to_add='conda activate mindspore'
vim -c "normal G" -c "o${content_to_add}" -c "wq" ~/.bashrc

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

# 使用多次 echo 命令将多行内容追加到 .bashrc 中
echo "export GLOG_v=2" >> ~/.bashrc
echo "LOCAL_ASCEND=/usr/local/Ascend" >> ~/.bashrc
echo "source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh" >> ~/.bashrc

python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"

