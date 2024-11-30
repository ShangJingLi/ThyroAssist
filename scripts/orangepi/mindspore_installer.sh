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