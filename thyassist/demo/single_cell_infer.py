import os
import subprocess
import sys
import signal
import gradio as gr
import numpy as np
import pandas as pd
import mindspore
from mindspore import Tensor
from mindspore import context
from thyassist.machine_learning.dataloader import download_and_unzip_best_mlp_checkpoints, download_and_unzip_mlp_datasets
from thyassist.machine_learning.networks import CellSortMlp
from launcher import get_project_root

download_dir = get_project_root()
"""单甲状腺上皮细胞特征检测器,
   癌变细胞标签为[1, 0]，正常细胞标签为[0, 1],
   数据尺度要求：0.25 微米/像素"""


my_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="gray",
    text_size="md",
    spacing_size="md",
    radius_size="md"
)


MEAN = np.array([5.12456848e+01, 8.54135088e+01, 1.42719344e+01, 7.80080000e+01,
                  5.5712e+01, 1.402496e+02, 2.63336016e+01, 8.94158400e-01,
                  9.14244480e+00, 8.27600000e+01, 1.19590080e+03, 1.06849920e+03,
                  8.45765616e+01, 7.1711632, 1.26164320, 8.11416000e-01,
                  9.728832e-01]).astype(np.float32)

STD = np.array([1.90583571e+01, 2.80879155e+01, 4.29302940e+00, 2.99940421e+01,
                2.38584958e+01, 3.19020703e+01, 5.34224124e+00, 4.70980517e-02,
                1.90564516e+00, 2.84534595e+01, 3.98852822e+02, 3.59792823e+02,
                4.88220625e+01, 1.54805123e+00, 2.15220092e-01, 1.13677582e-01,
                9.75784313e-03]).astype(np.float32)

USE_ORANGE_PI = False
if os.name == 'nt':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
else:
    try:
        if subprocess.run(['whoami'], capture_output=True, text=True, check=True).stdout.strip() == 'HwHiAiUser':
            context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', jit_config={"jit_level": "O2"})
            USE_ORANGE_PI = True
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    except:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

def on_terminate(signum, frame):
    if USE_ORANGE_PI:
        os.system('sudo npu-smi set -t pwm-duty-ratio -d 30')
        sys.exit(0)

if not os.path.exists(os.path.join(download_dir, 'best_mlp_checkpoints')):
    download_and_unzip_best_mlp_checkpoints()
else:
    pass

if not os.path.exists(os.path.join(download_dir, 'mlp_datasets')):
    download_and_unzip_mlp_datasets()
else:
    pass

means = np.load(os.path.join(download_dir, "mlp_datasets", "mean.npy"))
stds = np.load(os.path.join(download_dir, "mlp_datasets", "std.npy"))

net = CellSortMlp()
params = mindspore.load_checkpoint(os.path.join(download_dir, "best_mlp_checkpoints", "best_mlp_model_checkpoints.ckpt"))
mindspore.load_param_into_net(net, params)

if USE_ORANGE_PI:
    os.system('sudo npu-smi set -t pwm-duty-ratio -d 100')
    net(Tensor(np.zeros(shape=(1, 17)).astype(np.float32)))

signal.signal(signal.SIGINT, on_terminate)
signal.signal(signal.SIGTERM, on_terminate)

# 定义输入组件
dataframe_input = gr.File()

# 定义输出组件
output = gr.Textbox(label='预测结果')

def infer_single_cell(input_features):
    input_features = pd.read_csv(input_features.name)

    features = input_features.iloc[:, 1:].values.astype(np.float32)
    features = (features - MEAN) / STD

    counts = 0

    input_tensor = Tensor(features)
    for i in range(input_tensor.shape[0]):
        result = net(input_tensor[i])
        if result[0] > result[1]:
            counts += 1

    if counts == 0:
        text = "经模型检测，未发现带有癌细胞特征的样本"
    else:
        text = f"经模型检测，共{counts}个样本具有疑似癌细胞的特征"

    return text

# 创建 Gradio 接口
iface = gr.Interface(
    fn=infer_single_cell,
    inputs=dataframe_input,
    outputs=output,
    title="甲状腺上皮细胞特征检测器",
    description="上传特征文件，检测细胞是否属于癌细胞"
)

iface.launch(inbrowser=True)
